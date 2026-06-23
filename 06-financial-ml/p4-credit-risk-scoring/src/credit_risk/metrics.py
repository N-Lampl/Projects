"""Credit-risk evaluation: discrimination, calibration and fairness.

Discrimination
    roc_auc, ks_statistic (max gap between cumulative good/bad distributions),
    gini = 2*AUC - 1.
Calibration
    brier_score, plus a reliability curve (bin centers + observed default rate).
Fairness (at a fixed approval threshold)
    approval-rate gap and TPR gap between the two protected groups.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between defaulters' and non-defaulters'
    score CDFs. Equals max(TPR - FPR) over thresholds.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def discrimination(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "roc_auc": auc,
        "gini": 2.0 * auc - 1.0,
        "ks_statistic": ks_statistic(y_true, y_score),
    }


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Equal-width reliability curve.

    Returns mean predicted probability and observed default rate per non-empty
    bin, plus the bin weights (fraction of samples). Used for the calibration
    plot and an Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    mean_pred, obs_rate, weight = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        obs_rate.append(float(y_true[mask].mean()))
        weight.append(float(mask.mean()))
    return {"mean_predicted": mean_pred, "observed_rate": obs_rate, "weight": weight}


def expected_calibration_error(curve: dict[str, list[float]]) -> float:
    """Weighted mean |predicted - observed| over reliability bins."""
    mp = np.asarray(curve["mean_predicted"])
    obs = np.asarray(curve["observed_rate"])
    w = np.asarray(curve["weight"])
    if len(mp) == 0:
        return float("nan")
    return float(np.sum(w * np.abs(mp - obs)))


def calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(curve),
        "curve": curve,
    }


def confusion_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, int]:
    """Confusion counts where the *positive* class is default (score >= thr)."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def fairness_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> dict:
    """Approval-rate gap and TPR gap between the two protected groups.

    "Approval" = predicted *not* to default (score < threshold). TPR here is the
    true-positive rate on the *default* class (sensitivity of the risk flag),
    reported per group; the gap is |group_A - group_B|.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups = np.asarray(groups)
    approve = y_score < threshold  # approved for credit
    flag = ~approve  # flagged high-risk (default-positive prediction)

    per_group = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        approval_rate = float(approve[m].mean()) if m.any() else float("nan")
        pos = m & (y_true == 1)
        tpr = float(flag[pos].mean()) if pos.any() else float("nan")
        per_group[str(g)] = {
            "n": int(m.sum()),
            "approval_rate": approval_rate,
            "tpr": tpr,
            "default_rate": float(y_true[m].mean()) if m.any() else float("nan"),
        }

    keys = sorted(per_group)
    ar = [per_group[k]["approval_rate"] for k in keys]
    tpr = [per_group[k]["tpr"] for k in keys]
    return {
        "groups": keys,
        "per_group": per_group,
        "approval_rate_gap": float(abs(ar[0] - ar[1])) if len(ar) == 2 else float("nan"),
        "tpr_gap": float(abs(tpr[0] - tpr[1])) if len(tpr) == 2 else float("nan"),
    }


def threshold_for_default_rate(y_score: np.ndarray, target_decline_rate: float) -> float:
    """Pick the score threshold that declines ~``target_decline_rate`` of apps.

    A simple, defensible operating point: decline the riskiest X% of applicants.
    """
    return float(np.quantile(y_score, 1.0 - target_decline_rate))
