"""The default detector: a scikit-learn pipeline over lexical features.

A StandardScaler + LogisticRegression keeps the model interpretable (you can read
off which lexical signals drive the score) and trains in well under a second on CPU.
RandomForest is available as an alternative via `build_classifier(kind="rf")`.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier(kind: str = "logreg", seed: int = 42) -> Pipeline:
    """Return an unfitted sklearn pipeline. kind in {'logreg', 'rf'}."""
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=seed, n_jobs=1
        )
        return Pipeline([("clf", clf)])  # trees don't need scaling
    raise ValueError(f"unknown classifier kind: {kind!r}")


def evaluate(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute the standard binary-detection metrics + ROC points."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def top_feature_weights(pipe: Pipeline, feature_names: list[str], k: int = 8) -> list[tuple[str, float]]:
    """For a logreg pipeline, the most influential lexical features (signed)."""
    clf = pipe.named_steps.get("clf")
    if not hasattr(clf, "coef_"):
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return []
        order = np.argsort(np.abs(importances))[::-1][:k]
        return [(feature_names[i], float(importances[i])) for i in order]
    coef = clf.coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1][:k]
    return [(feature_names[i], float(coef[i])) for i in order]
