"""Feature engineering for OHLCV manipulation detection.

All features are causal (computed from past + current bar only) so a flag at bar
``t`` could in principle be raised in real time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "log_ret",
    "roll_vol",
    "vol_ratio",
    "price_z",
    "ret_z",
    "abs_log_ret",
    "hl_range",
]


def build_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Engineer causal features from an OHLCV frame.

    Returns a frame aligned to ``df`` (same index) with the columns in
    :data:`FEATURES`. The first ``window`` rows are warm-up and are filled with
    neutral values (0) rather than dropped, so indices stay aligned to events.
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    log_ret = np.log(close).diff()
    out["log_ret"] = log_ret
    out["abs_log_ret"] = log_ret.abs()

    # Rolling realised volatility of returns.
    out["roll_vol"] = log_ret.rolling(window, min_periods=2).std()

    # Volume vs its *trailing* moving average (the spoofing/pump signal).
    # shift(1) excludes the current bar so a one-bar spike is not diluted by
    # itself -- this is also the only causal (no look-ahead) choice.
    vol_ma = volume.shift(1).rolling(window, min_periods=2).mean()
    out["vol_ratio"] = volume / vol_ma

    # Price z-score vs a trailing window (detects the pump ramp/crash).
    price_ma = close.rolling(window, min_periods=2).mean()
    price_sd = close.rolling(window, min_periods=2).std()
    out["price_z"] = (close - price_ma) / price_sd.replace(0.0, np.nan)

    # Return z-score (detects abnormally large single-bar moves).
    ret_ma = log_ret.rolling(window, min_periods=2).mean()
    ret_sd = log_ret.rolling(window, min_periods=2).std()
    out["ret_z"] = (log_ret - ret_ma) / ret_sd.replace(0.0, np.nan)

    # Intraday range relative to close.
    out["hl_range"] = (df["high"] - df["low"]) / close

    return out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
