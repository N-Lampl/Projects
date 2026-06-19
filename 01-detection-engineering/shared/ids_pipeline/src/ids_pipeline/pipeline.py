"""Leak-free preprocessing + classifier pipeline for the tabular IDS.

The key security/ML-hygiene point: the scaler and one-hot encoder are fit on
TRAIN data only. We wrap everything in a single ``sklearn.Pipeline`` so that
``fit`` touches train data exclusively and ``predict`` reuses the train-fit
statistics -- no information leaks from test into preprocessing.

Default classifier: sklearn ``RandomForestClassifier`` (installed, no extra
deps). ``xgboost`` is an OPTIONAL upgrade imported lazily so the module still
imports without it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import Dataset


def _make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """ColumnTransformer that scales numerics and one-hot-encodes categoricals.

    Fit (on TRAIN only) learns the means/variances and category vocabulary;
    ``handle_unknown="ignore"`` keeps test-time-unseen categories from crashing.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _make_classifier(
    kind: Literal["rf", "xgb"] = "rf", seed: int = 42
):
    """Return the estimator. ``rf`` is the default; ``xgb`` is optional."""
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:  # pragma: no cover - optional path
            raise ImportError(
                "xgboost is not installed. Install it (pip install xgboost) "
                "or use kind='rf' for the default offline path."
            ) from e
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
    # default: RandomForest, class_weight balanced for the SOC class imbalance
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def build_pipeline(
    dataset: Dataset,
    classifier: Literal["rf", "xgb"] = "rf",
    seed: int = 42,
) -> Pipeline:
    """Assemble a leak-free preprocessing + classifier ``sklearn.Pipeline``.

    The pipeline is NOT yet fit -- call :func:`train` (or ``pipeline.fit``).
    """
    pre = _make_preprocessor(dataset.numeric_features, dataset.categorical_features)
    clf = _make_classifier(classifier, seed=seed)
    return Pipeline([("preprocess", pre), ("classifier", clf)])


def train(pipeline: Pipeline, dataset: Dataset) -> Pipeline:
    """Fit the pipeline on TRAIN data only and return it (fit in place)."""
    pipeline.fit(dataset.X_train, dataset.y_train)
    return pipeline


def predict_proba(pipeline: Pipeline, X) -> np.ndarray:
    """Positive-class (attack) probabilities, robust to estimators lacking proba."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    # fallback: decision_function squashed to [0,1]
    scores = pipeline.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))
