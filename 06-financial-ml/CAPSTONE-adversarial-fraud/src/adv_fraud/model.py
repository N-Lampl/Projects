"""Fraud classifier: a scikit-learn pipeline (scaler + logistic regression).

Logistic regression is deliberately chosen as the *default* because it gives the
hand-rolled evasion search a clean, smooth decision surface (the attack uses the
model's predicted probability as its objective and exploits the local gradient
of that surface - see ``attack.py``). ``class_weight="balanced"`` handles the
~4% fraud imbalance honestly instead of resampling.

A GradientBoosting variant is available for the "harder, non-linear target"
sensitivity check; it is pure scikit-learn so the default path needs no extra
deps.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str = "logreg", seed: int = 42) -> Pipeline:
    """Return an unfitted scaler+classifier pipeline."""
    if kind == "logreg":
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    elif kind == "gboost":
        from sklearn.ensemble import GradientBoostingClassifier

        clf = GradientBoostingClassifier(random_state=seed)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown model kind: {kind!r}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def fraud_proba(model: Pipeline, X):
    """Predicted P(fraud) for each row (column 1 of predict_proba)."""
    return model.predict_proba(X)[:, 1]
