"""Fast smoke tests (run in CI). The one slow test that trains both calibrated
models end-to-end is marked @slow and excluded via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk import (
    confusion_at_threshold,
    discrimination,
    fairness_at_threshold,
    ks_statistic,
    make_credit_data,
    reliability_curve,
    set_seed,
    threshold_for_default_rate,
    train_test_split_df,
)
from credit_risk.metrics import calibration


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_make_credit_data_is_deterministic_and_shaped():
    a = make_credit_data(n=500, seed=7)
    b = make_credit_data(n=500, seed=7)
    assert a.equals(b)  # same seed -> same table
    assert {"default", "group"}.issubset(a.columns)
    assert set(a["default"].unique()).issubset({0, 1})
    assert set(a["group"].unique()).issubset({"A", "B"})
    # signal injected -> not all one class, plausible imbalance
    rate = a["default"].mean()
    assert 0.02 < rate < 0.5


def test_split_is_stratified_and_disjoint():
    df = make_credit_data(n=1000, seed=1)
    train, test = train_test_split_df(df, test_size=0.3, seed=1)
    assert len(train) + len(test) == len(df)
    # stratification keeps default rates close between splits
    assert abs(train["default"].mean() - test["default"].mean()) < 0.05


def test_ks_and_gini_bounds_on_known_scores():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    d = discrimination(y, perfect)
    assert d["roc_auc"] == 1.0
    assert d["gini"] == pytest.approx(1.0)
    assert ks_statistic(y, perfect) == pytest.approx(1.0)
    assert 0.0 <= ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) <= 1.0


def test_calibration_perfect_probs_have_low_brier():
    set_seed(0)
    y = np.random.binomial(1, 0.3, size=2000)
    perfect = np.full_like(y, 0.3, dtype=float)
    cal = calibration(y, perfect, n_bins=10)
    assert cal["brier_score"] < 0.25  # 0.3*0.7 = 0.21 is the floor here
    curve = reliability_curve(y, perfect)
    assert len(curve["mean_predicted"]) == len(curve["observed_rate"]) >= 1


def test_confusion_counts_sum_to_n():
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
    c = confusion_at_threshold(y, s, 0.5)
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(y)
    assert c["tp"] == 2 and c["fp"] == 0  # 0.9 and 0.6 are the two flagged, both defaults


def test_fairness_reports_two_group_gaps():
    set_seed(0)
    df = make_credit_data(n=2000, seed=2)
    s = np.random.rand(len(df))
    thr = threshold_for_default_rate(s, 0.2)
    fair = fairness_at_threshold(df["default"].to_numpy(), s, df["group"].to_numpy(), thr)
    assert set(fair["groups"]) == {"A", "B"}
    assert 0.0 <= fair["approval_rate_gap"] <= 1.0
    assert 0.0 <= fair["tpr_gap"] <= 1.0


def test_threshold_declines_target_fraction():
    set_seed(0)
    s = np.random.rand(10000)
    thr = threshold_for_default_rate(s, 0.2)
    assert np.mean(s >= thr) == pytest.approx(0.2, abs=0.02)


@pytest.mark.slow
def test_end_to_end_both_models_train_and_calibrate():
    """Train + calibrate LogReg and GBM; both should discriminate well above
    chance and stay reasonably calibrated on the synthetic data.
    """
    from credit_risk import build_model, fit_predict_proba

    set_seed(42)
    df = make_credit_data(n=4000, seed=42)
    train, test = train_test_split_df(df, test_size=0.3, seed=42)
    y = test["default"].to_numpy()
    for kind in ("logreg", "gbm"):
        model = build_model(kind=kind, calibrate=True, method="isotonic")
        model, proba = fit_predict_proba(model, train, test)
        d = discrimination(y, proba)
        c = calibration(y, proba)
        assert d["roc_auc"] > 0.65  # real, recoverable signal
        assert 0.0 <= proba.min() and proba.max() <= 1.0
        assert c["brier_score"] < 0.25
