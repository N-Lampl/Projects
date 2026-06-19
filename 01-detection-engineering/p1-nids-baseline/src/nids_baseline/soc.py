"""SOC-facing reporting layer on top of the shared ``ids_pipeline`` library.

The shared library gives us a leak-free preprocess+RandomForest pipeline and
threshold-free metrics (ROC-AUC, precision@k). This module adds the bits a SOC
actually argues about:

- **Alert-budget thresholding.** An analyst team can only triage so many alerts
  a day. :func:`threshold_for_alert_rate` picks the score cutoff that flags a
  target *fraction* of traffic, instead of the arbitrary 0.5.
- **Operating-point summary.** :func:`soc_report` reports detection rate
  (recall), alert precision, the daily false-positive load, and the implied
  analyst workload at a chosen budget — the numbers that decide whether a
  detector ships.

All functions are pure NumPy/scikit-learn so the default path stays offline.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def threshold_for_alert_rate(y_score, alert_rate: float) -> float:
    """Score cutoff that flags ~``alert_rate`` fraction of flows as alerts.

    Models a fixed *alert budget*: the SOC reviews the top ``alert_rate`` of
    scored traffic. Returns the probability threshold at that quantile.
    """
    y_score = np.asarray(y_score, dtype=float)
    alert_rate = float(np.clip(alert_rate, 1e-6, 1.0))
    # the (1 - alert_rate) quantile is the cutoff above which we alert
    return float(np.quantile(y_score, 1.0 - alert_rate))


def soc_report(
    y_true,
    y_score,
    *,
    threshold: float = 0.5,
    flows_per_day: int = 1_000_000,
) -> dict:
    """Operating-point report at a fixed score ``threshold``.

    Parameters
    ----------
    flows_per_day:
        Used only to extrapolate the held-out false-positive *rate* into a
        human-readable "false alerts per day" figure — the number that tells a
        SOC lead whether the detector is triagable.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    n = int(len(y_true))
    n_attack = int(y_true.sum())
    n_benign = n - n_attack

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_alerts = tp + fp
    alert_rate = n_alerts / n if n else 0.0
    fp_rate = fp / n_benign if n_benign else 0.0  # per benign flow
    detection_rate = recall_score(y_true, y_pred, zero_division=0)
    alert_precision = precision_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "n_test": n,
        "n_attack": n_attack,
        "n_benign": n_benign,
        "detection_rate": float(detection_rate),     # recall: attacks caught
        "alert_precision": float(alert_precision),    # PPV: alerts that are real
        "miss_rate": float(fn / n_attack) if n_attack else 0.0,
        "false_positive_rate": float(fp_rate),        # per benign flow
        "alert_rate": float(alert_rate),              # fraction of all flows flagged
        "alerts_per_day_est": int(round(alert_rate * flows_per_day)),
        "false_alerts_per_day_est": int(round(fp_rate * flows_per_day * (n_benign / n if n else 0))),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "roc_auc": float(roc_auc_score(y_true, y_score)) if n_attack and n_benign else float("nan"),
    }


def sweep_operating_points(y_true, y_score, alert_rates) -> list[dict]:
    """SOC report at each alert budget in ``alert_rates`` (e.g. (0.01, 0.05, 0.1)).

    Shows the precision/recall trade-off a SOC navigates as it widens or
    tightens its daily alert budget.
    """
    rows = []
    for ar in alert_rates:
        thr = threshold_for_alert_rate(y_score, ar)
        rep = soc_report(y_true, y_score, threshold=thr)
        rep["target_alert_rate"] = float(ar)
        rows.append(rep)
    return rows
