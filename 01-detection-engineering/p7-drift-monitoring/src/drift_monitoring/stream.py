"""Synthetic tabular stream with an injected concept/data drift.

Models a fraud-/intrusion-style detector's input traffic. The first part of the
stream is "normal" (matches training), then drift is injected to simulate, e.g.,
an attacker shifting feature distributions to evade a model, or the upstream data
pipeline silently changing. No external data needed -> works fully offline.

Features (all numeric, the kind a tabular detector ingests):
    f0  request_rate        gaussian, shifts UP under drift   (volumetric change)
    f1  payload_size        lognormal, scale grows under drift (evasion / fuzzing)
    f2  inter_arrival       gaussian, variance grows under drift
    f3  entropy             gaussian, mean shifts (encrypted/obfuscated payloads)
    f4  session_len         gaussian, *stable* (control feature, should NOT alarm)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FEATURE_NAMES = [
    "request_rate",
    "payload_size",
    "inter_arrival",
    "entropy",
    "session_len",
]


@dataclass
class StreamConfig:
    n_windows: int = 24          # e.g. 24 hourly monitoring windows
    window_size: int = 500       # samples per window
    drift_start: int = 12        # window index where drift begins
    drift_ramp: int = 6          # windows over which drift ramps to full strength
    seed: int = 42
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))


def _normal_window(rng: np.random.Generator, n: int) -> np.ndarray:
    """One window drawn from the training-time ('normal') distribution."""
    f0 = rng.normal(100.0, 15.0, n)              # request_rate
    f1 = rng.lognormal(mean=6.0, sigma=0.4, size=n)  # payload_size
    f2 = rng.normal(0.5, 0.1, n)                 # inter_arrival
    f3 = rng.normal(4.0, 0.5, n)                 # entropy
    f4 = rng.normal(30.0, 5.0, n)                # session_len (control)
    return np.column_stack([f0, f1, f2, f3, f4])


def _apply_drift(window: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Distort a normal window. `strength` in [0, 1] scales the distortion.

    f4 (session_len) is intentionally left untouched so the monitor must
    *avoid* false-alarming on it.
    """
    w = window.copy()
    w[:, 0] += 40.0 * strength                                    # mean shift up
    w[:, 1] *= 1.0 + 1.5 * strength                              # scale blow-up
    w[:, 2] += rng.normal(0.0, 0.25 * strength, w.shape[0])     # variance growth
    w[:, 3] -= 1.5 * strength                                    # mean shift down
    return w


def generate_stream(config: StreamConfig | None = None):
    """Return (reference, windows, strengths).

    reference : (window_size, n_features) clean baseline ('training' snapshot)
    windows   : list of n_windows arrays, each (window_size, n_features)
    strengths : per-window drift strength in [0, 1] (0 before drift_start)
    """
    cfg = config or StreamConfig()
    rng = np.random.default_rng(cfg.seed)

    reference = _normal_window(rng, cfg.window_size)

    windows: list[np.ndarray] = []
    strengths: list[float] = []
    for i in range(cfg.n_windows):
        base = _normal_window(rng, cfg.window_size)
        if i < cfg.drift_start:
            strength = 0.0
        else:
            ramp = (i - cfg.drift_start + 1) / cfg.drift_ramp
            strength = float(min(1.0, ramp))
        windows.append(_apply_drift(base, rng, strength) if strength > 0 else base)
        strengths.append(strength)

    return reference, windows, strengths
