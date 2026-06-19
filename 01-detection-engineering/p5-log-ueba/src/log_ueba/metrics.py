"""Detection-quality metrics that matter for a SOC -- NOT just ROC-AUC.

An analyst triages a *ranked queue*. So the questions are:
  * Of the top-k alerts I look at, how many are real?            -> precision@k
  * Across the whole anomaly, when do I get my FIRST true hit?  -> time-to-detect
  * What fraction of the malicious events did I ever surface?   -> recall@k

ROC-AUC is reported too, but as a secondary number: it averages over operating points
no analyst actually uses (an analyst never reviews the bottom 90% of the queue).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scoring events that are truly anomalous."""
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(labels[idx].sum() / k)


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of all true anomalies captured within the top-k alerts."""
    total = labels.sum()
    if total == 0:
        return 0.0
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(labels[idx].sum() / total)


def time_to_detect(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    entities: np.ndarray | None = None,
    k: int = 100,
) -> dict:
    """Rank events by score; measure how quickly true anomalies surface.

    `rank` / `alerts_before` describe the queue position of the FIRST true hit.

    `ttd_seconds` is the detection latency of a single attack: the gap between the
    attack's first malicious event and the first malicious event that lands inside the
    analyst's top-k queue. If `entities` (e.g. the user per event) is given, we average
    this latency across each distinct compromised entity -- the realistic SOC question
    "once an account is popped, how long until something it does is flagged?".
    """
    order = np.argsort(scores)[::-1]
    ranked_labels = labels[order]
    hits = np.where(ranked_labels == 1)[0]
    if len(hits) == 0:
        return {"rank": -1, "alerts_before": -1, "ttd_seconds": -1, "ttd_per_entity": {}}
    rank = int(hits[0]) + 1
    alerts_before = int(rank - 1)

    topk = set(order[:k].tolist())
    per_entity: dict[str, int] = {}
    if entities is not None:
        for ent in np.unique(entities[labels == 1]):
            mask = (labels == 1) & (entities == ent)
            ent_ts = timestamps[mask]
            ent_idx = np.where(mask)[0]
            detected_ts = [t for t, idx in zip(ent_ts, ent_idx) if idx in topk]
            if detected_ts:
                per_entity[str(ent)] = int(min(detected_ts) - ent_ts.min())
            else:
                per_entity[str(ent)] = -1  # never caught in top-k
        caught = [v for v in per_entity.values() if v >= 0]
        ttd = int(np.mean(caught)) if caught else -1
    else:
        anom_ts = timestamps[labels == 1]
        first_detected_ts = timestamps[order[hits[0]]]
        ttd = int(first_detected_ts - anom_ts.min())

    return {
        "rank": rank,
        "alerts_before": alerts_before,
        "ttd_seconds": ttd,
        "ttd_per_entity": per_entity,
    }


def summary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    entities: np.ndarray | None = None,
    ks: tuple[int, ...] = (10, 25, 50, 100),
) -> dict:
    """Bundle the SOC-relevant metrics for one detector."""
    out = {
        "precision_at_k": {str(k): precision_at_k(scores, labels, k) for k in ks},
        "recall_at_k": {str(k): recall_at_k(scores, labels, k) for k in ks},
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "time_to_detect": time_to_detect(scores, labels, timestamps, entities),
    }
    return out
