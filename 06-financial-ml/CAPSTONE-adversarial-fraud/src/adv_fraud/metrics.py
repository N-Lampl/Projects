"""Financial detection metrics (not raw accuracy)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision among the k highest-scored transactions (analyst review queue)."""
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top]))


def threshold_for_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_budget: float) -> float:
    """Lowest score threshold that keeps the false-positive rate <= budget."""
    legit_scores = scores[y_true == 0]
    if len(legit_scores) == 0:
        return 1.0
    # threshold = (1 - budget) quantile of legit scores
    return float(np.quantile(legit_scores, 1.0 - fpr_budget))


def detection_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    fpr_budget: float = 0.05,
) -> dict:
    """PR-AUC (primary), ROC-AUC, p@k, recall at a fixed FP budget, confusion."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    thr = threshold_for_fpr(y_true, scores, fpr_budget)
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "fpr_budget": float(fpr_budget),
        "operating_threshold": float(thr),
        "recall_at_fpr_budget": float(recall),
        "precision_at_threshold": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "precision_at_k": {
            "p@50": precision_at_k(y_true, scores, 50),
            "p@100": precision_at_k(y_true, scores, 100),
            "p@200": precision_at_k(y_true, scores, 200),
        },
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }
