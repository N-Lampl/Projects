"""Anomaly detectors for OHLCV manipulation.

Two unsupervised detectors that need no labels at train time:

  * ``rolling_zscore_score`` -- a transparent composite of the per-bar feature
    z-scores (price + return + volume). Fast, interpretable, no model.
  * ``IsolationForest`` -- scikit-learn's tree-based outlier detector over the
    engineered feature matrix.

Both return a per-bar anomaly score where higher = more anomalous, so they share
the same scoring / thresholding code downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def rolling_zscore_score(feats: pd.DataFrame) -> np.ndarray:
    """Composite z-score anomaly signal (higher = more anomalous).

    Combines the magnitude of the volume ratio (in log space), the absolute
    price z-score and the absolute return z-score. This is the classical
    "rolling statistics" baseline.
    """
    vol_signal = np.log1p(np.clip(feats["vol_ratio"].to_numpy(), 0, None))
    price_signal = np.abs(feats["price_z"].to_numpy())
    ret_signal = np.abs(feats["ret_z"].to_numpy())

    # Standardise each component so no single feature dominates, then sum.
    def _z(x: np.ndarray) -> np.ndarray:
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)

    return _z(vol_signal) + _z(price_signal) + _z(ret_signal)


def isolation_forest_score(
    feats: pd.DataFrame, seed: int = 42, contamination: float = 0.05
) -> np.ndarray:
    """IsolationForest anomaly score (higher = more anomalous).

    ``score_samples`` returns higher = more normal, so we negate it to keep the
    "higher = more anomalous" convention shared with the z-score detector.
    """
    x = StandardScaler().fit_transform(feats.to_numpy())
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x)
    return -model.score_samples(x)


def threshold_at_budget(scores: np.ndarray, budget: float) -> float:
    """Return the score threshold that flags the top ``budget`` fraction of bars.

    ``budget`` is the per-bar false-positive / alert budget (e.g. 0.02 flags the
    most anomalous 2% of bars).
    """
    budget = float(np.clip(budget, 1e-6, 1.0))
    return float(np.quantile(scores, 1.0 - budget))
