"""Fraud-appropriate evaluation: PR-AUC (primary), ROC-AUC, precision@k,
recall at a fixed false-positive budget, and a confusion matrix at a chosen
threshold. Accuracy is deliberately NOT used -- on a 1% base rate it is
meaningless (predicting "never fraud" scores 99%).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve (== average precision)."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


def precision_at_k(y_true, scores, k: int) -> float:
    """Precision among the k highest-scored transactions (analyst review budget)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = min(k, len(scores))
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].sum() / k)


def threshold_at_fpr(y_true, scores, target_fpr: float) -> tuple[float, float]:
    """Pick the score threshold whose false-positive rate is closest to (but not
    above) `target_fpr`. Return (threshold, recall_at_that_threshold).
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = np.where(fpr <= target_fpr)[0]
    idx = ok[-1] if len(ok) else int(np.argmin(np.abs(fpr - target_fpr)))
    return float(thr[idx]), float(tpr[idx])


def best_f1_threshold(y_true, scores) -> float:
    """Threshold maximizing F1 on the PR curve -- a balanced operating point."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # thr has len = len(prec) - 1; align by dropping the last prec/rec point
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


def confusion_at(y_true, scores, threshold: float) -> dict[str, int]:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def recall_precision_from_confusion(c: dict[str, int]) -> tuple[float, float]:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return recall, precision
