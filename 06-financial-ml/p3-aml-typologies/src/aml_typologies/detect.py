"""AML account-level detectors and financial-grade scoring.

Two detectors, both scikit-learn-only (CPU, no extras):

* ``score_isolation_forest`` -- unsupervised anomaly score. Realistic for AML where
  labels are scarce; ranks accounts by how structurally unusual they are.
* ``score_rules_rf`` -- a rules+RandomForest hybrid. Hand-built typology rules give a
  prior, and a class-weighted RandomForest (honest handling of heavy imbalance)
  learns from the engineered features. Their scores are blended.

Scoring uses AML-appropriate metrics, not accuracy: PR-AUC (primary, because the
positive class is rare), ROC-AUC, precision@k, and recall at a fixed false-positive
budget (an analyst can only review so many alerts a day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .features import FEATURE_NAMES

SEED = 42


def score_isolation_forest(features: pd.DataFrame, *, seed: int = SEED) -> np.ndarray:
    """Unsupervised anomaly score in [0,1]; higher = more suspicious."""
    clf = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    clf.fit(features.values)
    raw = -clf.score_samples(features.values)  # higher = more anomalous
    return _minmax(raw)


def _rule_score(features: pd.DataFrame) -> np.ndarray:
    """Cheap, explainable typology prior in [0,1] from the engineered features."""
    f = features
    structuring = (f["sub_threshold_deposits"] >= 6).astype(float) + f["sub_threshold_ratio"]
    layering = f["rapid_passthrough"].astype(float) + f["in_cycle"].astype(float)
    score = structuring + layering + (f["fan_in"] >= 10).astype(float)
    return _minmax(score.to_numpy(dtype=float))


def score_rules_rf(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    seed: int = SEED,
    blend: float = 0.5,
) -> np.ndarray:
    """Blend of an explainable rule prior and a class-weighted RandomForest probability.

    The RF probability is produced with **out-of-fold** cross-validation
    (``cross_val_predict``): every account is scored by a forest that never saw it in
    training. This avoids the in-sample over-optimism of fitting and scoring the same
    rows, so the reported PR-AUC reflects genuine feature separability. The rule prior
    keeps it explainable and is blended in.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",  # honest handling of the rare positive class
        random_state=seed,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(
        rf, features.values, labels.values, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    rules = _rule_score(features)
    return blend * _minmax(proba) + (1 - blend) * rules


def feature_importances(features: pd.DataFrame, labels: pd.Series, *, seed: int = SEED):
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(features.values, labels.values)
    return pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    return float(labels[order].mean()) if k > 0 else 0.0


def recall_at_fpr_budget(scores: np.ndarray, labels: np.ndarray, fpr_budget: float):
    """Threshold so the false-positive rate <= budget; report recall there.

    Returns (threshold, recall, achieved_fpr, precision, confusion-dict).
    """
    neg = labels == 0
    n_neg = int(neg.sum())
    max_fp = int(np.floor(fpr_budget * n_neg))
    # candidate thresholds = each negative's score; pick the one whose FP count
    # stays within budget while admitting as many alerts as possible
    neg_scores = np.sort(scores[neg])[::-1]
    if max_fp <= 0:
        thresh = neg_scores[0] + 1e-9 if len(neg_scores) else 1.0
    elif max_fp >= len(neg_scores):
        thresh = scores.min()
    else:
        thresh = neg_scores[max_fp - 1]
    pred = (scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    achieved_fpr = fp / n_neg if n_neg else 0.0
    return (
        float(thresh),
        float(recall),
        float(achieved_fpr),
        float(precision),
        {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    )


def evaluate(scores: np.ndarray, labels: np.ndarray, *, fpr_budget: float = 0.01) -> dict:
    """Full AML-style scorecard for one detector's account scores."""
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    pr_auc = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    thresh, recall, achieved_fpr, prec, conf = recall_at_fpr_budget(scores, labels, fpr_budget)
    ks = _ks_statistic(scores, labels)
    return {
        "n_accounts": int(len(labels)),
        "n_suspicious": n_pos,
        "prevalence": n_pos / len(labels),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fpr_budget": fpr_budget,
        "operating_threshold": thresh,
        "recall_at_budget": recall,
        "precision_at_budget": prec,
        "achieved_fpr": achieved_fpr,
        "ks_statistic": ks,
        "precision_at_k": {
            f"p@{k}": precision_at_k(scores, labels, k) for k in (50, 100, 200)
        },
        "confusion_at_budget": conf,
    }


def _ks_statistic(scores: np.ndarray, labels: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between positive/negative score distributions."""
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    grid = np.sort(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def pr_curve(scores: np.ndarray, labels: np.ndarray):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return precision, recall
