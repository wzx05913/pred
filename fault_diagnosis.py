from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import BEARING_CONFIG, FAULT_TO_CLASS, PipelineConfig, resolve_device


def _soft_one_hot(label: int, confidence: float = 0.92) -> np.ndarray:
    p = np.full(6, (1.0 - confidence) / 5.0, dtype=float)
    p[int(label)] = confidence
    return p


def true_fault_label_from_metadata(fault: str) -> int:
    return FAULT_TO_CLASS.get(fault, 5)


def add_lag_prob_features(x: np.ndarray, probs: Sequence[np.ndarray], lag_order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rows = len(x)
    if rows == 0:
        return x
    pri = list(probs)
    default = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    lag_blocks = []
    for lag in range(1, lag_order + 1):
        arr = []
        for i in range(rows):
            idx = i - lag
            arr.append(pri[idx] if 0 <= idx < len(pri) else default)
        lag_blocks.append(np.vstack(arr))
    return np.hstack([x] + lag_blocks)


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    return p / max(s, 1e-12)


@dataclass
class DiagnosisResult:
    probs: np.ndarray
    label: int
    weak_label: int
    weak_reason: str
    weak_fault_combo: str
    failure_flag: bool
    shap_top: List[Tuple[str, float]]
    metrics: Dict[str, object]
    calibration: List[Tuple[float, float, float]]


@dataclass
class SupportSample:
    features: np.ndarray
    label: int
    prior: np.ndarray


@dataclass
class OnlineFaultDiagnoser:
    cfg: PipelineConfig
    mode: str = "zero_shot"
    feature_names: Optional[List[str]] = None
    x_hist: List[np.ndarray] = field(default_factory=list)
    y_weak: List[int] = field(default_factory=list)
    y_true: List[int] = field(default_factory=list)
    p_hist: List[np.ndarray] = field(default_factory=list)
    y_pred: List[int] = field(default_factory=list)
    healthy_rms: List[float] = field(default_factory=list)
    support_samples: List[SupportSample] = field(default_factory=list)
    _clf: object = None

    def __post_init__(self) -> None:
        try:
            from tabpfn import TabPFNClassifier  # type: ignore
            self._clf = TabPFNClassifier(model_path=str(self.cfg.tabpfn_classifier_model_path), device=resolve_device(self.cfg.device))
        except Exception:
            self._clf = None

    def set_support_samples(self, samples: List[SupportSample]) -> None:
        self.support_samples = list(samples)
        if self.support_samples:
            self.mode = "few_shot"

    def _physical_weak_label(
        self,
        window_id: int,
        total_windows: int,
        base_features: Dict[str, float],
        spectrum_features: Dict[str, Dict[str, float]],
    ) -> Tuple[int, str, str, bool]:
        ref_windows = self.cfg.healthy_reference_windows(total_windows)
        rms_mean = float(np.mean([base_features.get("horizontal_rms", 0.0), base_features.get("vertical_rms", 0.0)]))
        peak_max = float(max(base_features.get("horizontal_peak", 0.0), base_features.get("vertical_peak", 0.0)))
        if window_id <= ref_windows:
            self.healthy_rms.append(rms_mean)
            return 0, "healthy_reference", "Healthy", False
        if self.healthy_rms:
            ah = float(np.percentile(self.healthy_rms, 95))
        else:
            ah = max(rms_mean, 1e-8)
        damage_detected = rms_mean > self.cfg.weak_label_rms_multiplier * ah
        failure_flag = peak_max > self.cfg.failure_multiplier * ah
        if not damage_detected and not failure_flag:
            return 0, "below_damage_threshold", "Healthy", failure_flag
        names = {"BPFI": (1, "Inner"), "BPFO": (2, "Outer"), "FTF": (3, "Cage"), "BSF": (4, "Ball")}
        scores: Dict[str, float] = {}
        baseline = max(base_features.get("horizontal_rms", 0.0), base_features.get("vertical_rms", 0.0), 1e-8)
        for comp, (_, tag) in names.items():
            comp_scores = []
            for ch in ("horizontal", "vertical"):
                band_amp = spectrum_features.get(ch, {}).get(comp, 0.0)
                comp_scores.append(float(band_amp / baseline))
            scores[tag] = max(comp_scores) if comp_scores else 0.0
        hits = [tag for tag, val in scores.items() if val >= self.cfg.amplitude_ratio_threshold]
        if len(hits) >= 2:
            return 5, "physical_multi_fault", "+".join(sorted(hits)), failure_flag
        if len(hits) == 1:
            mapping = {"Inner": 1, "Outer": 2, "Cage": 3, "Ball": 4}
            return mapping[hits[0]], f"physical_{hits[0].lower()}", hits[0], failure_flag
        top = max(scores.items(), key=lambda kv: kv[1])[0] if scores else "Outer"
        mapping = {"Inner": 1, "Outer": 2, "Cage": 3, "Ball": 4}
        return mapping[top], "physical_top_band", top, failure_flag

    def _prepare_context(self, current_x: np.ndarray, weak_label: int, prior: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        X_parts: List[np.ndarray] = []
        y_parts: List[int] = []
        pri_parts: List[np.ndarray] = []
        if self.support_samples:
            for s in self.support_samples:
                X_parts.append(np.asarray(s.features, dtype=float))
                y_parts.append(int(s.label))
                pri_parts.append(np.asarray(s.prior, dtype=float))
        if self.x_hist:
            X_parts.extend(self.x_hist)
            y_parts.extend(self.y_weak)
            pri_parts.extend(self.p_hist if self.p_hist else [_soft_one_hot(v) for v in self.y_weak])
        X_parts.append(np.asarray(current_x, dtype=float))
        y_parts.append(int(weak_label))
        pri_parts.append(np.asarray(prior, dtype=float))
        return np.vstack(X_parts), np.asarray(y_parts, dtype=int), pri_parts

    def _expand_classes(self, raw: np.ndarray) -> np.ndarray:
        out = np.zeros(6, dtype=float)
        classes = getattr(self._clf, "classes_", np.arange(len(raw))) if self._clf is not None else np.arange(len(raw))
        for c, val in zip(classes, raw):
            out[int(c)] = float(val)
        return _normalize_prob(out)

    def _predict_with_tabpfn(self, X: np.ndarray, y: np.ndarray, priors: List[np.ndarray], weak_label: int) -> np.ndarray:
        if self._clf is None or len(np.unique(y)) < 2:
            return _soft_one_hot(weak_label)
        X2 = add_lag_prob_features(X, priors, self.cfg.probability_lag_order)
        self._clf.fit(X2, y)
        p1 = self._expand_classes(np.asarray(self._clf.predict_proba(X2[-1:]))[0])
        weak_prior = _soft_one_hot(weak_label, confidence=0.80)
        if weak_label != 0 and (p1[weak_label] + self.cfg.correction_margin) >= np.max(p1):
            p1 = _normalize_prob(np.maximum(p1, weak_prior))
        pri2 = list(priors[:-1]) + [p1]
        X3 = add_lag_prob_features(X, pri2, self.cfg.probability_lag_order)
        self._clf.fit(X3, y)
        return self._expand_classes(np.asarray(self._clf.predict_proba(X3[-1:]))[0])

    def _kernel_shap_exact(self, x: np.ndarray, X_context: np.ndarray, y_context: np.ndarray, priors: List[np.ndarray], p: np.ndarray) -> List[Tuple[str, float]]:
        names = self.feature_names or [f"f{i}" for i in range(len(x))]
        if self._clf is None or len(X_context) < 3:
            raise RuntimeError("insufficient context for SHAP")
        import shap  # type: ignore
        bg = X_context[max(0, len(X_context) - self.cfg.shap_background): -1]
        if len(bg) == 0:
            bg = X_context[:-1]
        target = int(np.argmax(p))

        def f(z: np.ndarray) -> np.ndarray:
            out = []
            for row in np.asarray(z, dtype=float):
                X_new = np.vstack([X_context[:-1], row])
                X_new2 = add_lag_prob_features(X_new, priors, self.cfg.probability_lag_order)
                self._clf.fit(X_new2, y_context)
                pp = self._expand_classes(np.asarray(self._clf.predict_proba(X_new2[-1:]))[0])
                out.append(pp[target])
            return np.asarray(out)

        explainer = shap.KernelExplainer(f, bg)
        vals = np.asarray(explainer.shap_values(x.reshape(1, -1), nsamples=self.cfg.shap_nsamples))[0]
        idx = np.argsort(np.abs(vals))[::-1][: self.cfg.shap_top_k]
        return [(names[i], float(vals[i])) for i in idx]

    def _shap_top(self, x: np.ndarray, X_context: np.ndarray, y_context: np.ndarray, priors: List[np.ndarray], p: np.ndarray) -> List[Tuple[str, float]]:
        names = self.feature_names or [f"f{i}" for i in range(len(x))]
        try:
            return self._kernel_shap_exact(x, X_context, y_context, priors, p)
        except Exception:
            hist = np.vstack(self.x_hist) if self.x_hist else np.zeros((1, len(x)))
            vals = (x - np.nanmean(hist, axis=0)) / (np.nanstd(hist, axis=0) + 1e-8) * float(np.max(p))
            idx = np.argsort(np.abs(vals))[::-1][: self.cfg.shap_top_k]
            return [(names[i], float(vals[i])) for i in idx]

    def _metrics(self) -> Dict[str, object]:
        y = np.asarray(self.y_true, dtype=int)
        yp = np.asarray(self.y_pred, dtype=int)
        if len(y) == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "brier": 0.0, "confusion_matrix": [[0] * 6 for _ in range(6)]}
        cm = np.zeros((6, 6), dtype=int)
        for yt, yp_ in zip(y, yp):
            cm[int(yt), int(yp_)] += 1
        acc = float(np.mean(y == yp))
        f1s, weights = [], []
        for c in range(6):
            tp = cm[c, c]
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            f1s.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
            weights.append(float(np.mean(y == c)))
        brier = float(np.mean([np.mean((p - np.eye(6)[yy]) ** 2) for p, yy in zip(self.p_hist, y)]))
        return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "weighted_f1": float(np.sum(np.asarray(weights) * np.asarray(f1s))), "brier": brier, "confusion_matrix": cm.tolist()}

    def _calibration(self) -> List[Tuple[float, float, float]]:
        if not self.p_hist or not self.y_true:
            return []
        conf = np.max(np.vstack(self.p_hist), axis=1)
        correct = (np.asarray(self.y_pred) == np.asarray(self.y_true))
        bins = np.linspace(0.0, 1.0, self.cfg.calibration_bins + 1)
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            avg_conf = float(np.mean(conf[mask])) if np.any(mask) else float("nan")
            avg_acc = float(np.mean(correct[mask])) if np.any(mask) else float("nan")
            out.append((float((lo + hi) / 2), avg_conf, avg_acc))
        return out

    def update(
        self,
        feature_vector: np.ndarray,
        feature_names: List[str],
        window_id: int,
        total_windows: int,
        base_features: Dict[str, float],
        spectrum_features: Dict[str, Dict[str, float]],
        fault: str,
    ) -> DiagnosisResult:
        self.feature_names = feature_names
        weak, reason, combo, failure_flag = self._physical_weak_label(window_id, total_windows, base_features, spectrum_features)
        y_true = true_fault_label_from_metadata(fault)
        prior = _soft_one_hot(weak)
        X_context, y_context, priors = self._prepare_context(feature_vector, weak, prior)
        p = self._predict_with_tabpfn(X_context, y_context, priors, weak)
        label = int(np.argmax(p))
        shap_top = self._shap_top(feature_vector, X_context, y_context, priors[:-1] + [p], p)
        self.x_hist.append(np.asarray(feature_vector, dtype=float))
        self.y_weak.append(int(weak))
        self.y_true.append(int(y_true))
        self.p_hist.append(p)
        self.y_pred.append(label)
        metrics = self._metrics()
        calibration = self._calibration()
        return DiagnosisResult(p, label, weak, reason, combo, failure_flag, shap_top, metrics, calibration)


def build_support_samples(feature_rows: List[Tuple[np.ndarray, int]], cfg: PipelineConfig) -> List[SupportSample]:
    if not feature_rows:
        return []
    start = int(max(0, len(feature_rows) * (1.0 - cfg.support_degradation_portion)))
    selected = feature_rows[start:: max(cfg.support_stride, 1)]
    out = []
    for x, label in selected:
        out.append(SupportSample(features=np.asarray(x, dtype=float), label=int(label), prior=_soft_one_hot(int(label))))
    return out


def metadata_label_for_bearing(bearing_id: str) -> int:
    return true_fault_label_from_metadata(str(BEARING_CONFIG[bearing_id]["fault"]))
