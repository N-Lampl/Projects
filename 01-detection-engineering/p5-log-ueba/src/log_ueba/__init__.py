"""Auth-log anomaly detection / UEBA on synthetic enterprise auth events.

Public API:
    set_seed, get_device          -- reproducibility helpers (get_device needs torch)
    generate_auth_events, GenConfig -- offline synthetic auth-event generator
    build_features, FEATURE_NAMES -- per-user streaming UEBA features
    isolation_forest_scores       -- default sklearn detector (higher = more anomalous)
    autoencoder_scores            -- optional torch autoencoder detector
    precision_at_k, recall_at_k, time_to_detect, summary_metrics -- SOC metrics
"""

from .detect import autoencoder_scores, isolation_forest_scores
from .features import FEATURE_NAMES, build_features
from .generate import GenConfig, generate_auth_events
from .metrics import (
    precision_at_k,
    recall_at_k,
    summary_metrics,
    time_to_detect,
)
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "generate_auth_events",
    "GenConfig",
    "build_features",
    "FEATURE_NAMES",
    "isolation_forest_scores",
    "autoencoder_scores",
    "precision_at_k",
    "recall_at_k",
    "time_to_detect",
    "summary_metrics",
]
