from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig


@dataclass
class HealthIndexBuilder:
    cfg: PipelineConfig
    history: List[Dict[str, float]] = field(default_factory=list)
    damage_history: List[float] = field(default_factory=list)
    hi_history: List[float] = field(default_factory=list)

    def _selected_features(self, row: Dict[str, float]) -> List[str]:
        return [k for k in self.cfg.degradation_directions if k in row]

    def update(self, row: Dict[str, float], shap_top: List[Tuple[str, float]], total_windows: Optional[int] = None) -> Dict[str, object]:
        self.history.append(row)
        feats = self._selected_features(row)
        ref_n = self.cfg.healthy_reference_windows(total_windows if total_windows else len(self.history))
        healthy = self.history[: max(1, min(len(self.history), ref_n))]
        recent = self.history[-max(1, min(len(self.history), ref_n)):]
        shap_abs = {k: 0.0 for k in feats}
        for name, val in shap_top:
            base = name[6:] if name.startswith("delta_") else name
            if base in shap_abs:
                shap_abs[base] += abs(float(val))
        ssum = sum(shap_abs.values())
        weights = {k: (shap_abs[k] / ssum if ssum > 0 else 1.0 / max(len(feats), 1)) for k in feats}
        damages = {}
        for k in feats:
            hvals = np.array([r[k] for r in healthy if k in r], dtype=float)
            rvals = np.array([r[k] for r in recent if k in r], dtype=float)
            qh = float(np.quantile(hvals, self.cfg.healthy_quantile)) if len(hvals) else row[k]
            qf = float(np.quantile(rvals, self.cfg.failure_quantile)) if len(rvals) else row[k]
            direction = self.cfg.degradation_directions.get(k, "positive")
            if direction == "reverse":
                d = (qh - row[k]) / (qh - qf + self.cfg.hi_epsilon)
            else:
                if qf <= qh:
                    qf = max(float(np.max(rvals)) if len(rvals) else row[k], qh + self.cfg.hi_epsilon)
                d = (row[k] - qh) / (qf - qh + self.cfg.hi_epsilon)
            damages[k] = float(np.clip(d, 0.0, 1.0))
        D = float(sum(weights[k] * damages[k] for k in feats)) if feats else 0.0
        if self.damage_history:
            D = float(self.cfg.hi_ema_alpha * D + (1 - self.cfg.hi_ema_alpha) * self.damage_history[-1])
        self.damage_history.append(D)
        hi = float(np.exp(-self.cfg.hi_lambda * D))
        self.hi_history.append(hi)
        t0, t1, t2 = self.cfg.hi_thresholds
        level = 0 if hi >= t0 else 1 if hi >= t1 else 2 if hi >= t2 else 3
        return {"HI": hi, "D": D, "level": level, "weights": weights, "damages": damages}

    def timing_score(self, total_windows: int) -> Dict[str, float]:
        if not self.hi_history:
            return {"alert_window": float("nan"), "fail_window": float("nan"), "delta_t": float("nan"), "score_timing": 0.0}
        alert_idx = next((i + 1 for i, hi in enumerate(self.hi_history) if hi < self.cfg.tau_fail), total_windows)
        fail_idx = total_windows
        delta_t = max(fail_idx - alert_idx, 0)
        ideal = 0.1 * total_windows
        sigma = max(0.05 * total_windows, 1e-8)
        score = float(np.exp(-0.5 * ((delta_t - ideal) / sigma) ** 2))
        return {"alert_window": float(alert_idx), "fail_window": float(fail_idx), "delta_t": float(delta_t), "score_timing": score}
