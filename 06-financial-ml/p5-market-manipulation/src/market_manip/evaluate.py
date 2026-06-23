"""Scoring: bar-level ranking metrics + event-level detection & lead-time.

Financial-anomaly framing (not accuracy):

  * **PR-AUC** (primary) and **ROC-AUC** over the per-bar scores vs the per-bar
    ground-truth labels -- honest under heavy class imbalance.
  * **precision@k** at the most-anomalous k bars.
  * **recall at a fixed alert budget** (per-bar false-positive budget) plus the
    confusion matrix at that operating threshold.
  * **event-level** detection: an injected manipulation counts as *caught* if
    any bar inside its window is flagged. We report event recall and the
    **median lead-time-to-flag** = (event peak bar) - (first flagged bar in the
    window). Positive lead-time means we flagged before the worst bar.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from .data import Event


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of the top-``k`` scored bars that are truly manipulated."""
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(labels[idx].mean())


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """PR-AUC (primary), ROC-AUC and precision@k for several k."""
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision_at_k": {
            f"p@{k}": precision_at_k(scores, labels, k) for k in (25, 50, 100)
        },
    }


def operating_point(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Bar-level confusion + rates at a chosen score threshold."""
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    n_pos = tp + fn
    n_neg = tn + fp
    return {
        "threshold": float(threshold),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "bar_recall": float(tp / n_pos) if n_pos else 0.0,
        "bar_precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "false_positive_rate": float(fp / n_neg) if n_neg else 0.0,
    }


def event_metrics(
    scores: np.ndarray, events: list[Event], threshold: float
) -> dict:
    """Event-level recall + lead-time at a chosen threshold.

    An event is detected if any bar in [start, end] scores >= threshold.
    Lead-time = peak - (first flagged bar in window); >0 means early warning.
    """
    flagged = scores >= threshold
    n_detected = 0
    lead_times: list[int] = []
    per_kind: dict[str, list[int]] = {}

    for ev in events:
        window = np.where(flagged[ev.start : ev.end + 1])[0]
        caught = window.size > 0
        per_kind.setdefault(ev.kind, [0, 0])
        per_kind[ev.kind][1] += 1
        if caught:
            n_detected += 1
            per_kind[ev.kind][0] += 1
            first_flag = ev.start + int(window[0])
            lead_times.append(ev.peak - first_flag)

    n_events = len(events)
    out = {
        "n_events": n_events,
        "n_detected": n_detected,
        "event_recall": float(n_detected / n_events) if n_events else 0.0,
        "median_lead_time_bars": float(np.median(lead_times)) if lead_times else None,
        "mean_lead_time_bars": float(np.mean(lead_times)) if lead_times else None,
        "event_recall_by_kind": {
            kind: {
                "detected": c[0],
                "total": c[1],
                "recall": float(c[0] / c[1]) if c[1] else 0.0,
            }
            for kind, c in sorted(per_kind.items())
        },
    }
    return out


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision/recall arrays for plotting (drops the trailing sentinel point)."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return precision, recall
