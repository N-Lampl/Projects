"""Credit default-risk scoring: discrimination + calibration + a fairness check.

Public API:
    set_seed, get_device          -- reproducibility helpers
    make_credit_data              -- seeded synthetic borrower table w/ default label
    train_test_split_df           -- deterministic stratified split
    build_model, fit_predict_proba-- (calibrated) LogReg / GradientBoosting
    discrimination                -- ROC-AUC, KS, Gini
    calibration                   -- Brier score + reliability curve + ECE
    confusion_at_threshold        -- confusion counts at an operating point
    fairness_at_threshold         -- approval-rate gap + TPR gap across groups
    threshold_for_default_rate    -- decline-the-riskiest-X% operating point
"""

from .data import FEATURES, make_credit_data, train_test_split_df
from .metrics import (
    calibration,
    confusion_at_threshold,
    discrimination,
    fairness_at_threshold,
    ks_statistic,
    reliability_curve,
    threshold_for_default_rate,
)
from .model import build_model, fit_predict_proba
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "FEATURES",
    "make_credit_data",
    "train_test_split_df",
    "build_model",
    "fit_predict_proba",
    "discrimination",
    "ks_statistic",
    "calibration",
    "reliability_curve",
    "confusion_at_threshold",
    "fairness_at_threshold",
    "threshold_for_default_rate",
]
