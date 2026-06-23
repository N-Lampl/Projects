"""Detection metrics for label-free anomaly scoring.

The labels are used here ONLY to grade the unsupervised scores -- never to fit.
We report the metrics that matter for a heavily imbalanced fraud/anomaly queue:
PR-AUC (primary), ROC-AUC, precision@k / recall@k, and recall at a fixed
false-positive budget.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Top-k by score: fraction that are true anomalies (P@k) and caught (R@k)."""
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    hits = int(y_true[order].sum())
    total_pos = int(y_true.sum())
    precision = hits / k if k else 0.0
    recall = hits / total_pos if total_pos else 0.0
    return precision, recall


def recall_at_fpr_budget(
    y_true: np.ndarray, scores: np.ndarray, fpr_budget: float
) -> tuple[float, float, float]:
    """Pick the threshold giving ~fpr_budget false-positive rate; report recall.

    Returns (threshold, recall, achieved_fpr).
    """
    neg = scores[y_true == 0]
    if len(neg) == 0:
        return float("inf"), 0.0, 0.0
    thr = float(np.quantile(neg, 1.0 - fpr_budget))
    pred = scores >= thr
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    recall = tp / pos if pos else 0.0
    achieved = fp / n_neg if n_neg else 0.0
    return thr, recall, achieved


def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, int]:
    pred = scores >= thr
    return {
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
    }


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, contamination: float) -> dict:
    """Full metric bundle for a single detector's scores."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())

    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))

    ks = [50, 100, 200]
    p_at_k = {}
    r_at_k = {}
    for k in ks:
        p, r = precision_recall_at_k(y_true, scores, k)
        p_at_k[f"p@{k}"] = round(p, 4)
        r_at_k[f"r@{k}"] = round(r, 4)

    fpr_budget = 0.01
    thr, rec_at_budget, achieved = recall_at_fpr_budget(y_true, scores, fpr_budget)
    confusion = confusion_at_threshold(y_true, scores, thr)

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "n_anomalies": n_pos,
        "base_rate": round(n_pos / len(y_true), 5),
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "fpr_budget": fpr_budget,
        "operating_threshold": thr,
        "recall_at_fpr_budget": round(rec_at_budget, 4),
        "achieved_fpr": round(achieved, 5),
        "confusion": confusion,
    }
