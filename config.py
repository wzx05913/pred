from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BEARING_GEOMETRY = {"n_b": 8, "d_mm": 7.92, "D_mm": 34.55, "alpha_deg": 0.0}
SAMPLE_RATE_HZ = 25600.0
SAMPLING_INTERVAL_MIN = 1.0
SAMPLE_DURATION_SEC = 1.28
CSV_ROWS = 32768
CHANNEL_NAMES = ("horizontal", "vertical")

BEARING_CONFIG: Dict[str, Dict[str, object]] = {
    "Bearing1_1": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Outer race"},
    "Bearing1_2": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Outer race"},
    "Bearing1_3": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Outer race"},
    "Bearing1_4": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Cage"},
    "Bearing1_5": {"condition": 1, "speed_hz": 35.0, "load_kn": 12, "fault": "Mixed"},
    "Bearing2_1": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Inner race"},
    "Bearing2_2": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Outer race"},
    "Bearing2_3": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Cage"},
    "Bearing2_4": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Outer race"},
    "Bearing2_5": {"condition": 2, "speed_hz": 37.5, "load_kn": 11, "fault": "Outer race"},
    "Bearing3_1": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Outer race"},
    "Bearing3_2": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Mixed"},
    "Bearing3_3": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Inner race"},
    "Bearing3_4": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Inner race"},
    "Bearing3_5": {"condition": 3, "speed_hz": 40.0, "load_kn": 10, "fault": "Outer race"},
}
CONDITION_DIRS = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}
FAULT_LABELS = {
    0: "非常健康",
    1: "疑似内圈问题",
    2: "疑似外圈问题",
    3: "疑似保持架问题",
    4: "疑似滚动体问题",
    5: "疑似混合问题",
}
FAULT_TO_CLASS = {
    "Healthy": 0,
    "Inner race": 1,
    "Outer race": 2,
    "Cage": 3,
    "Ball": 4,
    "Rolling element": 4,
    "Mixed": 5,
}


@dataclass
class PipelineConfig:
    data_root: Path = Path("XJTU-SY_Bearing_Datasets/Data")
    output_dir: Path = Path("output")
    sampling_rate_hz: float = SAMPLE_RATE_HZ
    sampling_interval_min: float = SAMPLING_INTERVAL_MIN
    sample_duration_sec: float = SAMPLE_DURATION_SEC
    csv_rows: int = CSV_ROWS
    p: float = 1.0
    wavelet: str = "db4"
    wavelet_level: int = 4
    healthy_quantile: float = 0.95
    failure_quantile: float = 0.05
    hi_lambda: float = 3.0
    hi_epsilon: float = 1e-8
    hi_ema_alpha: float = 0.30
    rul_ema_alpha: float = 0.15
    hi_thresholds: Tuple[float, float, float] = (0.75, 0.50, 0.25)
    tau_fail: float = 0.25
    rul_max_steps: int = 200
    rul_recent_points: int = 8
    rul_context_window: int = 8
    rul_ci_z: float = 1.96
    healthy_reference_ratio: float = 0.10
    healthy_reference_windows_min: int = 3
    healthy_reference_windows_max: int = 20
    weak_label_rms_multiplier: float = 3.0
    failure_multiplier: float = 10.0
    fault_band_width_hz: float = 5.0
    harmonic_orders: Tuple[int, ...] = (1, 2, 3)
    amplitude_ratio_threshold: float = 3.0
    correction_margin: float = 0.08
    probability_lag_order: int = 3
    support_degradation_portion: float = 0.30
    support_stride: int = 3
    shap_top_k: int = 8
    shap_background: int = 20
    shap_nsamples: int = 50
    calibration_bins: int = 5
    tabpfn_regressor_model_path: Path = Path("model/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt")
    tabpfn_classifier_model_path: Path = Path("model/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt")
    device: Optional[str] = None
    random_seed: int = 42
    degradation_directions: Dict[str, str] = field(default_factory=lambda: {
        "horizontal_rms": "positive", "vertical_rms": "positive", "rms_combined": "positive",
        "horizontal_kurt": "positive", "vertical_kurt": "positive", "kurt_max": "positive",
        "horizontal_crest": "positive", "vertical_crest": "positive", "horizontal_impulse": "positive", "vertical_impulse": "positive",
        "horizontal_peak": "positive", "vertical_peak": "positive", "energyratio_hv": "positive", "rho_hv": "reverse",
        "horizontal_amp_BPFO": "positive", "vertical_amp_BPFO": "positive", "horizontal_amp_BPFI": "positive", "vertical_amp_BPFI": "positive",
        "horizontal_amp_BSF": "positive", "vertical_amp_BSF": "positive", "horizontal_amp_FTF": "positive", "vertical_amp_FTF": "positive",
    })

    def window_size(self) -> int:
        if not (0 < self.p <= 1):
            raise ValueError("p must be in (0, 1]")
        return max(1, int(math.floor(self.csv_rows * self.p)))

    def ensure_dirs(self) -> None:
        for sub in ("features", "diagnosis", "hi", "rul", "figures", "logs"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def healthy_reference_windows(self, total_windows: Optional[int] = None) -> int:
        if total_windows is None or total_windows <= 0:
            return self.healthy_reference_windows_min
        return max(
            self.healthy_reference_windows_min,
            min(self.healthy_reference_windows_max, int(math.ceil(total_windows * self.healthy_reference_ratio))),
        )


def compute_fault_freqs(speed_hz: float) -> dict:
    nb = BEARING_GEOMETRY["n_b"]
    d = BEARING_GEOMETRY["d_mm"]
    D = BEARING_GEOMETRY["D_mm"]
    alpha = math.radians(BEARING_GEOMETRY["alpha_deg"])
    lam = (d / D) * math.cos(alpha)
    return {
        "BPFO": nb / 2 * speed_hz * (1 - lam),
        "BPFI": nb / 2 * speed_hz * (1 + lam),
        "BSF": D / (2 * d) * speed_hz * (1 - lam**2),
        "FTF": 0.5 * speed_hz * (1 - lam),
    }


def resolve_device(device: Optional[str] = None) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def all_bearing_ids() -> List[str]:
    return sorted(BEARING_CONFIG)
