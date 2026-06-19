"""SOC-relevant evaluation metrics for the tabular IDS.

A SOC cares about more than accuracy: alert precision (how many flagged flows
are real), recall (how many attacks we catch), precision@k (the top-k alerts an
analyst actually reviews), the confusion matrix, and ROC-AUC (threshold-free
ranking quality). :func:`evaluate` returns all of them as a plain dict so it
serialises straight into ``results/metrics.json``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .pipeline import predict_proba


def precision_at_k(y_true, y_score, k: int) -> float:
    """Fraction of the top-k highest-scored flows that are real attacks.

    Models the analyst workflow: you can only triage the top-k alerts, so the
    quality of the *ranking head* matters as much as a global threshold.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(k, len(y_score))
    if k == 0:
        return 0.0
    top = np.argsort(y_score)[::-1][:k]
    return float(y_true[top].mean())


def evaluate(pipeline, dataset, *, threshold: float = 0.5, ks: tuple[int, ...] = (50, 100, 200)) -> dict:
    """Compute SOC metrics on the held-out TEST split.

    Returns a JSON-serialisable dict: precision/recall/F1, precision@k for a few
    k, ROC-AUC, and the confusion matrix (as nested lists + a labelled dict).
    """
    y_true = dataset.y_test
    y_score = predict_proba(pipeline, dataset.X_test)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n_test = int(len(y_true))
    p_at_k = {f"p@{k}": precision_at_k(y_true, y_score, k) for k in ks if k <= n_test}

    return {
        "n_test": n_test,
        "n_attack_test": int(np.asarray(y_true).sum()),
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision_at_k": p_at_k,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labelled": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
