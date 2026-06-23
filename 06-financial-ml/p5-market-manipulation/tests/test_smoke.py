"""Fast smoke tests (run in CI). The one slow end-to-end test is @slow."""

from __future__ import annotations

import numpy as np
import pytest

from market_manip import (
    FEATURES,
    build_features,
    event_metrics,
    generate,
    isolation_forest_score,
    rolling_zscore_score,
    set_seed,
    threshold_at_budget,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_generate_is_reproducible_and_labelled():
    s1 = generate(n=400, seed=7)
    s2 = generate(n=400, seed=7)
    assert np.allclose(s1.df["close"].to_numpy(), s2.df["close"].to_numpy())
    assert len(s1.events) > 0
    # labels mark exactly the union of event windows, and are a strict subset.
    y = s1.label
    assert y.sum() > 0
    assert y.sum() < len(y)


def test_ohlc_is_internally_consistent():
    df = generate(n=300, seed=1).df
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()


def test_features_aligned_and_finite():
    s = generate(n=300, seed=2)
    feats = build_features(s.df, window=20)
    assert list(feats.columns) == FEATURES
    assert len(feats) == len(s.df)
    assert np.isfinite(feats.to_numpy()).all()


def test_detectors_higher_on_events():
    """Both detectors should score manipulated bars above normal bars on average."""
    s = generate(n=800, seed=3)
    feats = build_features(s.df)
    y = s.label.astype(bool)
    for scores in (isolation_forest_score(feats), rolling_zscore_score(feats)):
        assert scores[y].mean() > scores[~y].mean()


def test_threshold_at_budget_respects_budget():
    s = generate(n=600, seed=4)
    scores = isolation_forest_score(build_features(s.df))
    thr = threshold_at_budget(scores, 0.05)
    flagged_frac = (scores >= thr).mean()
    assert flagged_frac <= 0.05 + 0.02  # small slack for ties at the quantile


def test_event_metrics_shape():
    s = generate(n=600, seed=5)
    scores = isolation_forest_score(build_features(s.df))
    thr = threshold_at_budget(scores, 0.05)
    evm = event_metrics(scores, s.events, thr)
    assert evm["n_events"] == len(s.events)
    assert 0 <= evm["n_detected"] <= evm["n_events"]
    assert 0.0 <= evm["event_recall"] <= 1.0


@pytest.mark.slow
def test_end_to_end_pipeline_detects_manipulation():
    """Full pipeline: IsolationForest beats prevalence and catches most events."""
    set_seed(42)
    s = generate(n=1500, seed=42)
    feats = build_features(s.df)
    scores = isolation_forest_score(feats, seed=42)
    y = s.label
    from market_manip import ranking_metrics

    rank = ranking_metrics(scores, y)
    assert rank["pr_auc"] > y.mean() * 2  # well above the chance baseline
    assert rank["roc_auc"] > 0.7

    thr = threshold_at_budget(scores, 0.05)
    evm = event_metrics(scores, s.events, thr)
    assert evm["event_recall"] >= 0.5
    # every multi-bar pump-and-dump should be caught at this budget.
    assert evm["event_recall_by_kind"]["pump_dump"]["recall"] == 1.0
