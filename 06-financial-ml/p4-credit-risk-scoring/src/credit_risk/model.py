"""Two credit-risk classifiers + probability calibration.

Credit models are used to *price* risk, so a well-calibrated probability of
default (PD) matters more than raw discrimination. We therefore wrap each base
estimator in scikit-learn's CalibratedClassifierCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES


def _base_estimator(kind: str):
    """Return an uncalibrated pipeline for the requested model kind.

    Imbalance is handled honestly: logistic regression uses
    ``class_weight="balanced"``; gradient boosting (no class_weight) is left to
    the threshold-tuning + calibration steps. Both are stated in the README.
    """
    pre = ColumnTransformer(
        [("num", StandardScaler(), FEATURES)], remainder="drop"
    )
    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        )
    elif kind == "gbm":
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
    else:
        raise ValueError(f"unknown model kind: {kind!r} (expected 'logreg' or 'gbm')")
    return Pipeline([("pre", pre), ("clf", clf)])


def build_model(kind: str = "gbm", calibrate: bool = True, method: str = "isotonic"):
    """Build a (optionally calibrated) credit-risk classifier.

    ``method`` is "isotonic" (default, non-parametric) or "sigmoid" (Platt).
    Calibration uses internal cross-validation so it never peeks at test data.
    """
    base = _base_estimator(kind)
    if not calibrate:
        return base
    return CalibratedClassifierCV(base, method=method, cv=5)


def fit_predict_proba(
    model, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[object, np.ndarray]:
    """Fit on train_df and return (fitted_model, test default-probabilities)."""
    model.fit(train_df[FEATURES], train_df["default"])
    proba = model.predict_proba(test_df[FEATURES])[:, 1]
    return model, proba
