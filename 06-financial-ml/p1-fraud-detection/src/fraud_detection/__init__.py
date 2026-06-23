"""Supervised credit-card fraud detection on a highly imbalanced table.

Public API:
    set_seed, get_device          -- reproducibility helpers (CPU-only)
    make_transactions, split_xy   -- seeded synthetic transaction generator
    build_models, predict_scores  -- logreg + RF (+ optional xgboost) classifiers
    pr_auc, roc_auc, ...          -- fraud-appropriate metrics (PR-AUC primary)
"""

from .data import FEATURES, LABEL, make_transactions, split_xy
from .metrics import (
    best_f1_threshold,
    confusion_at,
    pr_auc,
    precision_at_k,
    recall_precision_from_confusion,
    roc_auc,
    threshold_at_fpr,
)
from .models import build_models, predict_scores
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "make_transactions",
    "split_xy",
    "FEATURES",
    "LABEL",
    "build_models",
    "predict_scores",
    "pr_auc",
    "roc_auc",
    "precision_at_k",
    "threshold_at_fpr",
    "best_f1_threshold",
    "confusion_at",
    "recall_precision_from_confusion",
]
