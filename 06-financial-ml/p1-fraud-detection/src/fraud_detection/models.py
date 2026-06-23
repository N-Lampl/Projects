"""Fraud classifiers. LogisticRegression + RandomForest are the always-available
defaults (class_weight='balanced' to handle the ~1% imbalance honestly).

xgboost is OPTIONAL: imported lazily; if missing we silently skip it so the
module still imports and runs with only scikit-learn installed.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import SEED


def build_models() -> dict[str, object]:
    """Return name -> unfitted estimator. xgboost added only if importable."""
    models: dict[str, object] = {
        "logreg": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
    }

    xgb_model = _maybe_xgboost()
    if xgb_model is not None:
        models["xgboost"] = xgb_model
    return models


def _maybe_xgboost():
    """Return a configured XGBClassifier, or None if xgboost is not installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=SEED,
        n_jobs=-1,
    )


def predict_scores(model, x) -> "object":
    """Uniform fraud-probability accessor across estimators."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    # fallback for margin-only estimators
    return model.decision_function(x)
