from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CHANNEL_NAMES, PipelineConfig, compute_fault_freqs
from .data_loader import WindowRecord

EPS = 1e-12


@dataclass
class FeatureBundle:
    full_features: Dict[str, float]
    base_features: Dict[str, float]
    spectrum_features: Dict[str, Dict[str, float]]
    feature_names: List[str]
    feature_vector: np.ndarray


def _entropy(values: np.ndarray) -> float:
    v = np.maximum(np.asarray(values, dtype=float), 0.0)
    s = float(v.sum())
    if s <= EPS:
        return 0.0
    p = v / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + EPS)))


def _time_features(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    centered = x - mean
    var = float(np.mean(centered**2))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    mav = float(np.mean(np.abs(x)))
    return {
        "rms": rms,
        "peak": peak,
        "mav": mav,
        "kurt": float(np.mean(centered**4) / (var**2 + EPS) - 3.0),
        "crest": peak / (rms + EPS),
        "impulse": peak / (mav + EPS),
        "skew": float(np.mean(centered**3) / ((var + EPS) ** 1.5)),
    }


def _freq_features(x: np.ndarray, fs: float, speed_hz: float, band: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    n = len(x)
    if n < 4:
        amp_feats = {f"amp_{k}": 0.0 for k in ("BPFO", "BPFI", "BSF", "FTF")}
        return {"spectral_centroid": 0.0, "spectral_entropy": 0.0, **amp_feats}, amp_feats
    w = np.hanning(n)
    amp = (2.0 / n) * np.abs(np.fft.rfft((x - np.mean(x)) * w))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = float(amp.sum() + EPS)
    out = {
        "spectral_centroid": float(np.sum(freqs * amp) / total),
        "spectral_entropy": _entropy(amp),
    }
    band_amps: Dict[str, float] = {}
    for name, fc in compute_fault_freqs(speed_hz).items():
        mask = (freqs >= fc - band) & (freqs <= fc + band)
        band_amps[name] = float(np.max(amp[mask])) if np.any(mask) else 0.0
        out[f"amp_{name}"] = band_amps[name]
    return out, band_amps


def _wavelet_features(x: np.ndarray, wavelet: str, level: int) -> Dict[str, float]:
    try:
        import pywt  # type: ignore
        coeffs = [np.asarray(c, dtype=float) for c in pywt.wavedec(x, wavelet, level=level, mode="periodization")]
    except Exception:
        coeffs = []
        approx = np.asarray(x, dtype=float)
        for _ in range(level):
            if len(approx) < 2:
                coeffs.append(np.zeros(1))
                continue
            if len(approx) % 2:
                approx = approx[:-1]
            a = (approx[0::2] + approx[1::2]) / np.sqrt(2)
            d = (approx[0::2] - approx[1::2]) / np.sqrt(2)
            coeffs.append(d)
            approx = a
        coeffs.insert(0, approx)
    energies = np.array([float(np.sum(c * c)) for c in coeffs], dtype=float)
    total = float(energies.sum() + EPS)
    out = {"wavelet_entropy": _entropy(energies)}
    for i in range(level + 1):
        out[f"wavelet_energy_l{i}"] = float(energies[i] / total) if i < len(energies) else 0.0
    return out


def extract_base_features(record: WindowRecord, cfg: PipelineConfig) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    x = np.asarray(record.history, dtype=float)
    feats: Dict[str, float] = {}
    spectra: Dict[str, Dict[str, float]] = {}
    channel_time = {}
    for ch, cname in enumerate(CHANNEL_NAMES):
        tf = _time_features(x[:, ch])
        channel_time[cname] = tf
        ff, band_amps = _freq_features(x[:, ch], cfg.sampling_rate_hz, record.speed_hz, cfg.fault_band_width_hz)
        wf = _wavelet_features(x[:, ch], cfg.wavelet, cfg.wavelet_level)
        spectra[cname] = band_amps
        for group in (tf, ff, wf):
            for k, v in group.items():
                feats[f"{cname}_{k}"] = v
    h, v = x[:, 0], x[:, 1]
    hc, vc = h - h.mean(), v - v.mean()
    feats["rms_combined"] = float(np.sqrt(0.5 * (channel_time["horizontal"]["rms"]**2 + channel_time["vertical"]["rms"]**2)))
    feats["kurt_max"] = float(max(channel_time["horizontal"]["kurt"], channel_time["vertical"]["kurt"]))
    feats["rho_hv"] = float(np.sum(hc * vc) / (np.sqrt(np.sum(hc**2) * np.sum(vc**2)) + EPS))
    feats["energyratio_hv"] = float(channel_time["horizontal"]["rms"]**2 / (channel_time["vertical"]["rms"]**2 + EPS))
    return feats, spectra


def build_feature_bundle(record: WindowRecord, cfg: PipelineConfig, prev_base: Optional[Dict[str, float]] = None) -> FeatureBundle:
    base, spectra = extract_base_features(record, cfg)
    full = dict(base)
    for k, v in base.items():
        full[f"delta_{k}"] = 0.0 if prev_base is None else float(v - prev_base.get(k, 0.0))
    full.update(
        {
            "run_index": record.window_id,
            "speed_hz": record.speed_hz,
            "load_kn": record.load_kn,
            "condition": record.condition,
        }
    )
    exclude = {"run_index", "speed_hz", "load_kn", "condition"}
    feature_names = sorted([k for k, v in full.items() if k not in exclude and np.isscalar(v)])
    feature_vector = np.array([float(full[k]) for k in feature_names], dtype=float)
    return FeatureBundle(full, base, spectra, feature_names, feature_vector)
