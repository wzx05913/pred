from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import BEARING_CONFIG, FAULT_LABELS, PipelineConfig, all_bearing_ids, compute_fault_freqs, resolve_device
from .data_loader import count_windows, iter_windows
from .feature_engineering import FeatureBundle, build_feature_bundle
from .fault_diagnosis import OnlineFaultDiagnoser, build_support_samples, metadata_label_for_bearing
from .health_index import HealthIndexBuilder
from .rul import RULPredictor, build_rul_support_windows


def _write_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    extras = sorted({k for r in rows for k in r.keys()} - set(keys))
    keys.extend(extras)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("xjtu_sy_phm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _collect_bearing_features(bearing_id: str, cfg: PipelineConfig, max_windows: int = 0) -> Tuple[List[FeatureBundle], List[float]]:
    prev_base = None
    bundles: List[FeatureBundle] = []
    rms_like: List[float] = []
    for record in iter_windows(bearing_id, cfg):
        if max_windows and record.window_id > max_windows:
            break
        bundle = build_feature_bundle(record, cfg, prev_base)
        prev_base = bundle.base_features
        bundles.append(bundle)
        rms_like.append(float(bundle.base_features.get("rms_combined", 0.0)))
    return bundles, rms_like


def _build_support_context(support_bearings: List[str], cfg: PipelineConfig, logger: logging.Logger) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, float]]]:
    cls_rows: List[Tuple[np.ndarray, int]] = []
    rul_rows: List[Tuple[np.ndarray, float]] = []
    for bearing_id in support_bearings:
        bundles, hi_seed = _collect_bearing_features(bearing_id, cfg)
        if not bundles:
            continue
        label = metadata_label_for_bearing(bearing_id)
        cls_rows.extend([(b.feature_vector, label) for b in bundles])
        hi_seq = np.asarray(hi_seed, dtype=float)
        hi_seq = hi_seq / max(np.max(hi_seq), 1e-8)
        hi_seq = np.clip(1.0 - 0.75 * hi_seq, cfg.tau_fail, 1.0)
        rul_rows.extend(build_rul_support_windows(hi_seq.tolist(), cfg))
        logger.info("加载 support bearing=%s, windows=%d, label=%d", bearing_id, len(bundles), label)
    return cls_rows, rul_rows


def run_bearing(bearing_id: str, cfg: PipelineConfig, args: argparse.Namespace, logger: logging.Logger, stamp: str, support_cls: List[Tuple[np.ndarray, int]], support_rul: List[Tuple[np.ndarray, float]]) -> None:
    total_windows = count_windows(bearing_id, cfg)
    mode = "few_shot" if args.support_bearings else "zero_shot"
    diagnoser = OnlineFaultDiagnoser(cfg, mode=mode)
    if support_cls:
        diagnoser.set_support_samples(build_support_samples(support_cls, cfg))
    hi_builder = HealthIndexBuilder(cfg)
    rul_predictor = RULPredictor(cfg, mode=mode)
    if support_rul:
        rul_predictor.set_support_windows(support_rul)
    prev_base = None
    feature_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    hi_rows: List[Dict[str, object]] = []
    rul_rows: List[Dict[str, object]] = []
    for record in iter_windows(bearing_id, cfg):
        if args.max_windows and record.window_id > args.max_windows:
            break
        bundle = build_feature_bundle(record, cfg, prev_base)
        prev_base = bundle.base_features
        diag = diagnoser.update(bundle.feature_vector, bundle.feature_names, record.window_id, total_windows, bundle.base_features, bundle.spectrum_features, record.fault)
        hi = hi_builder.update(bundle.base_features, diag.shap_top, total_windows=total_windows)
        rul = rul_predictor.update(float(hi["HI"]))
        base_meta = {
            "bearing_id": bearing_id,
            "window_id": record.window_id,
            "csv_index": record.csv_index,
            "source_file": record.source_file,
            "fault": record.fault,
            "fault_freqs": json.dumps(compute_fault_freqs(record.speed_hz), ensure_ascii=False),
        }
        feature_rows.append({**base_meta, **bundle.full_features})
        diag_rows.append({
            **base_meta,
            "mode": diagnoser.mode,
            "y_true": metadata_label_for_bearing(bearing_id),
            "y_pred": diag.label,
            "label_name": FAULT_LABELS[diag.label],
            "weak_label": diag.weak_label,
            "weak_reason": diag.weak_reason,
            "weak_fault_combo": diag.weak_fault_combo,
            "failure_flag": diag.failure_flag,
            **{f"prob_{i}": float(diag.probs[i]) for i in range(6)},
            "shap_top": json.dumps(diag.shap_top, ensure_ascii=False),
            "accuracy": diag.metrics["accuracy"],
            "macro_f1": diag.metrics["macro_f1"],
            "weighted_f1": diag.metrics["weighted_f1"],
            "brier": diag.metrics["brier"],
            "confusion_matrix": json.dumps(diag.metrics["confusion_matrix"], ensure_ascii=False),
            "calibration": json.dumps(diag.calibration, ensure_ascii=False),
        })
        hi_rows.append({**base_meta, "HI": hi["HI"], "D": hi["D"], "level": hi["level"], "weights": json.dumps(hi["weights"], ensure_ascii=False), "damages": json.dumps(hi["damages"], ensure_ascii=False)})
        rul_rows.append({**base_meta, "rul": rul["rul"], "ci_low": rul["ci_low"], "ci_high": rul["ci_high"], "method": rul["method"], "hi_monotonic": rul["hi_monotonic"], "future_hi": json.dumps(rul["future_hi"], ensure_ascii=False)})
        logger.info("%s #%d mode=%s pred=%s weak=%s HI=%.4f RUL=%s CI=[%s,%s]", bearing_id, record.window_id, diagnoser.mode, diag.label, diag.weak_label, hi["HI"], rul["rul"], rul["ci_low"], rul["ci_high"])
    timing = hi_builder.timing_score(len(hi_rows))
    for row in hi_rows:
        row.update(timing)
    if not feature_rows:
        logger.warning("%s 没有产生窗口，请检查数据路径。", bearing_id)
        return
    _write_rows_csv(feature_rows, cfg.output_dir / "features" / f"{stamp}_{bearing_id}_features.csv")
    _write_rows_csv(diag_rows, cfg.output_dir / "diagnosis" / f"{stamp}_{bearing_id}_diagnosis.csv")
    _write_rows_csv(hi_rows, cfg.output_dir / "hi" / f"{stamp}_{bearing_id}_hi.csv")
    _write_rows_csv(rul_rows, cfg.output_dir / "rul" / f"{stamp}_{bearing_id}_rul.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XJTU-SY 轴承端到端在线 PHM 流水线")
    p.add_argument("--bearings", nargs="+", default=["Bearing1_1"], choices=all_bearing_ids())
    p.add_argument("--support-bearings", nargs="*", default=[])
    p.add_argument("--p", type=float, default=1.0, help="窗口占单 CSV 的比例，(0,1]")
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--lambda-hi", type=float, default=3.0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--data-root", default="XJTU-SY_Bearing_Datasets/Data")
    p.add_argument("--output-dir", default="output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(data_root=Path(args.data_root), output_dir=Path(args.output_dir), p=args.p, wavelet=args.wavelet, hi_lambda=args.lambda_hi, device=resolve_device(args.device))
    cfg.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = _setup_logger(cfg.output_dir / "logs" / f"{stamp}_run.log")
    logger.info("启动 XJTU-SY PHM 在线流水线: bearings=%s support=%s p=%s wavelet=%s device=%s lambda_hi=%s", args.bearings, args.support_bearings, cfg.p, cfg.wavelet, cfg.device, cfg.hi_lambda)
    logger.info("故障特征频率示例: %s", {b: compute_fault_freqs(float(BEARING_CONFIG[b]["speed_hz"])) for b in args.bearings})
    support_cls: List[Tuple[np.ndarray, int]] = []
    support_rul: List[Tuple[np.ndarray, float]] = []
    if args.support_bearings:
        support_cls, support_rul = _build_support_context(args.support_bearings, cfg, logger)
    for b in args.bearings:
        run_bearing(b, cfg, args, logger, stamp, support_cls, support_rul)
    logger.info("全部完成，结果已保存到 %s", cfg.output_dir)


if __name__ == "__main__":
    main()
