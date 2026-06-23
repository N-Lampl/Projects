"""Unsupervised transaction anomaly detection.

Label-free detection over a seeded synthetic transaction stream with injected
anomalies (amount spikes, off-hours activity, velocity bursts). The injected
labels are used ONLY to evaluate the detectors -- never at fit time.

Public API:
    set_seed, get_device            -- reproducibility helpers
    make_transactions, feature_matrix, FEATURES  -- synthetic stream + features
    IForestDetector                 -- default detector (sklearn IsolationForest)
    AutoencoderDetector             -- optional torch AE (sklearn fallback)
    evaluate_scores                 -- PR-AUC / ROC-AUC / P@k / recall@budget
"""

from .data import FEATURES, feature_matrix, make_transactions
from .detectors import AutoencoderDetector, IForestDetector, torch_available
from .evaluate import (
    confusion_at_threshold,
    evaluate_scores,
    precision_recall_at_k,
    recall_at_fpr_budget,
)
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "make_transactions",
    "feature_matrix",
    "FEATURES",
    "IForestDetector",
    "AutoencoderDetector",
    "torch_available",
    "evaluate_scores",
    "precision_recall_at_k",
    "recall_at_fpr_budget",
    "confusion_at_threshold",
]
