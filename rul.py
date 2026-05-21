from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig, resolve_device


@dataclass
class RULPredictor:
    cfg: PipelineConfig
    mode: str = "zero_shot"
    hi_raw: List[float] = field(default_factory=list)
    hi_mono: List[float] = field(default_factory=list)
    support_windows: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    _reg: object = None

    def __post_init__(self) -> None:
        try:
            from tabpfn import TabPFNRegressor  # type: ignore
            self._reg = TabPFNRegressor(model_path=str(self.cfg.tabpfn_regressor_model_path), device=resolve_device(self.cfg.device))
        except Exception:
            self._reg = None

    def set_support_windows(self, support_windows: List[Tuple[np.ndarray, float]]) -> None:
        self.support_windows = list(support_windows)
        if self.support_windows:
            self.mode = "few_shot"

    def _monotonic_hi(self, hi: float) -> float:
        self.hi_raw.append(float(hi))
        prev = self.hi_mono[-1] if self.hi_mono else float(hi)
        alpha = self.cfg.rul_ema_alpha
        smooth = alpha * float(hi) + (1 - alpha) * prev
        mono = min(prev, smooth) if self.hi_mono else smooth
        self.hi_mono.append(float(mono))
        return float(mono)

    def _fallback_predict(self, y: np.ndarray, mono: float, sigma_lin: float):
        t = np.arange(len(y), dtype=float)
        beta_fit, alpha_fit = np.polyfit(t, y, 1)
        pred_lin = alpha_fit + beta_fit * np.arange(len(y), len(y) + self.cfg.rul_max_steps)
        cross = np.where(pred_lin <= self.cfg.tau_fail)[0]
        if len(cross):
            resid = y - (alpha_fit + beta_fit * t)
            sigma = float(np.std(resid)) if len(resid) > 1 else sigma_lin
            return int(cross[0] + 1), "linear_cross", sigma, beta_fit, pred_lin
        if beta_fit < -1e-6:
            resid = y - (alpha_fit + beta_fit * t)
            sigma = float(np.std(resid)) if len(resid) > 1 else sigma_lin
            rul = int(np.ceil((self.cfg.tau_fail - mono) / beta_fit))
            return rul, "linear_extrapolate", sigma, beta_fit, pred_lin
        log_y = np.log(np.maximum(y, 1e-6))
        log_beta, log_alpha = np.polyfit(t, log_y, 1)
        decay = max(1e-4, -log_beta)
        pred = mono * np.exp(-decay * np.arange(1, self.cfg.rul_max_steps + 1))
        rul = int(np.ceil(np.log(self.cfg.tau_fail / max(mono, 1e-6)) / -decay))
        y_fit = np.exp(log_alpha + log_beta * t)
        sigma = float(np.std(y - y_fit)) if len(y) > 1 else sigma_lin
        beta_val = -decay * mono
        return rul, "exponential_decay", sigma, beta_val, pred

    def _few_shot_predict(self) -> Optional[Tuple[int, str, float, float, np.ndarray]]:
        L = self.cfg.rul_context_window
        if self._reg is None or len(self.hi_mono) < L or not self.support_windows:
            return None
        try:
            Xs = np.vstack([x for x, _ in self.support_windows])
            ys = np.asarray([y for _, y in self.support_windows], dtype=float)
            x_now = np.asarray(self.hi_mono[-L:], dtype=float).reshape(1, -1)
            self._reg.fit(Xs, ys)
            pred = float(np.asarray(self._reg.predict(x_now)).reshape(-1)[0])
            train_pred = np.asarray(self._reg.predict(Xs)).reshape(-1)
            sigma = float(np.std(ys - train_pred)) if len(ys) > 1 else 0.0
            future = np.array([max(self.hi_mono[-1] - (i + 1) * ((self.hi_mono[-1] - self.cfg.tau_fail) / max(pred, 1.0)), self.cfg.tau_fail) for i in range(self.cfg.rul_max_steps)], dtype=float)
            beta = float((future[min(len(future) - 1, 1)] - self.hi_mono[-1]))
            return int(max(pred, 0)), "tabpfn_few_shot", sigma, beta, future
        except Exception:
            return None

    def update(self, hi: float) -> Dict[str, object]:
        mono = self._monotonic_hi(hi)
        y = np.array(self.hi_mono[-self.cfg.rul_recent_points:], dtype=float)
        if mono <= self.cfg.tau_fail:
            return {"rul": 1, "ci_low": 1, "ci_high": 1, "method": "already_failed", "future_hi": [mono], "hi_monotonic": mono}
        if self.mode == "few_shot":
            res = self._few_shot_predict()
            if res is not None:
                rul, method, sigma, beta_val, pred = res
                rul = int(np.clip(rul, 0, self.cfg.rul_max_steps))
                rul = max(rul, 1)
                half = self.cfg.rul_ci_z * sigma / max(abs(float(beta_val)), 1e-6)
                return {"rul": rul, "ci_low": int(np.clip(np.floor(rul - half), 1, self.cfg.rul_max_steps)), "ci_high": int(np.clip(np.ceil(rul + half), 1, self.cfg.rul_max_steps)), "method": method, "future_hi": [float(v) for v in pred[: self.cfg.rul_max_steps]], "hi_monotonic": mono}
        if len(y) < 2:
            rul = self.cfg.rul_max_steps
            sigma = 0.0
            beta_val = -1.0
            pred = np.repeat(mono, self.cfg.rul_max_steps)
            method = "insufficient_history"
        else:
            t = np.arange(len(y), dtype=float)
            beta_fit, alpha_fit = np.polyfit(t, y, 1)
            resid_lin = y - (alpha_fit + beta_fit * t)
            sigma = float(np.std(resid_lin)) if len(resid_lin) > 1 else 0.0
            pred = alpha_fit + beta_fit * np.arange(len(y), len(y) + self.cfg.rul_max_steps)
            cross = np.where(pred <= self.cfg.tau_fail)[0]
            if len(cross):
                rul = int(cross[0] + 1)
                method = "linear_cross"
                beta_val = beta_fit
            else:
                rul, method, sigma, beta_val, pred = self._fallback_predict(y, mono, sigma)
        rul = int(np.clip(rul, 1, self.cfg.rul_max_steps))
        half = self.cfg.rul_ci_z * sigma / max(abs(float(beta_val)), 1e-6)
        return {"rul": rul, "ci_low": int(np.clip(np.floor(rul - half), 1, self.cfg.rul_max_steps)), "ci_high": int(np.clip(np.ceil(rul + half), 1, self.cfg.rul_max_steps)), "method": method, "future_hi": [float(v) for v in pred[: self.cfg.rul_max_steps]], "hi_monotonic": mono}


def build_rul_support_windows(hi_sequence: Sequence[float], cfg: PipelineConfig) -> List[Tuple[np.ndarray, float]]:
    L = cfg.rul_context_window
    mono = []
    prev = None
    for hi in hi_sequence:
        smooth = hi if prev is None else cfg.rul_ema_alpha * hi + (1 - cfg.rul_ema_alpha) * prev
        prev = smooth if prev is None else min(prev, smooth)
        mono.append(prev)
    mono = np.asarray(mono, dtype=float)
    total = len(mono)
    out: List[Tuple[np.ndarray, float]] = []
    for t in range(L - 1, total, max(cfg.support_stride, 1)):
        x = mono[t - L + 1: t + 1]
        y = float(total - (t + 1))
        out.append((x.astype(float), y))
    return out
