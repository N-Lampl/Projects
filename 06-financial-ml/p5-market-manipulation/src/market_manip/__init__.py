"""Market-manipulation detection on synthetic OHLCV time series.

Public API:
    set_seed, get_device        -- reproducibility helpers
    generate, Series, Event     -- seeded synthetic OHLCV + injected events
    build_features, FEATURES    -- causal feature engineering
    rolling_zscore_score,
    isolation_forest_score,
    threshold_at_budget         -- unsupervised detectors + budget thresholding
    ranking_metrics,
    operating_point,
    event_metrics               -- PR-AUC / ROC-AUC / event recall + lead-time
"""

from .data import Event, Series, generate
from .detect import isolation_forest_score, rolling_zscore_score, threshold_at_budget
from .evaluate import event_metrics, operating_point, ranking_metrics
from .features import FEATURES, build_features
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "generate",
    "Series",
    "Event",
    "build_features",
    "FEATURES",
    "rolling_zscore_score",
    "isolation_forest_score",
    "threshold_at_budget",
    "ranking_metrics",
    "operating_point",
    "event_metrics",
]
