"""Fast smoke tests (run in CI). The one slow end-to-end test is marked @slow and
excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from log_ueba import (
    FEATURE_NAMES,
    GenConfig,
    build_features,
    generate_auth_events,
    precision_at_k,
    recall_at_k,
    set_seed,
    time_to_detect,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_generator_is_reproducible_and_labeled():
    df1 = generate_auth_events(GenConfig(seed=7))
    df2 = generate_auth_events(GenConfig(seed=7))
    assert df1.equals(df2)  # deterministic
    assert df1["is_anomaly"].sum() > 0  # anomalies were injected
    assert set(df1["is_anomaly"].unique()) <= {0, 1}


def test_generator_timestamps_sorted():
    df = generate_auth_events(GenConfig(seed=1))
    ts = df["timestamp"].to_numpy()
    assert np.all(ts[:-1] <= ts[1:])


def test_features_shape_and_columns():
    df = generate_auth_events(GenConfig(n_users=10, n_days=3, seed=3))
    feats = build_features(df)
    assert len(feats) == len(df)
    for c in FEATURE_NAMES:
        assert c in feats.columns
    assert np.isfinite(feats[FEATURE_NAMES].to_numpy()).all()


def test_features_are_causal_first_event_has_no_history():
    """The very first event for any user must look 'novel' (baseline empty)."""
    df = generate_auth_events(GenConfig(n_users=5, n_days=2, seed=5))
    feats = build_features(df)
    # first chronological event overall: nothing seen yet -> novel src & dst, zero card
    first = feats.iloc[0]
    assert first["novel_dst"] == 1.0
    assert first["novel_src"] == 1.0
    assert first["user_dst_card"] == 0.0


def test_precision_and_recall_at_k_perfect_ranking():
    # scores rank the two anomalies first -> precision@2 == 1, recall@2 == 1
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.05])
    assert precision_at_k(scores, labels, 2) == 1.0
    assert recall_at_k(scores, labels, 2) == 1.0
    assert precision_at_k(scores, labels, 4) == 0.5


def test_time_to_detect_first_hit_rank():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.9, 0.1, 0.95])  # ranks: idx3, idx1, idx0, idx2
    ttd = time_to_detect(scores, labels, np.array([0, 100, 200, 300]))
    assert ttd["rank"] == 1  # top-ranked event (idx3) is a true anomaly
    assert ttd["alerts_before"] == 0


def test_isolation_forest_imports_and_scores():
    from log_ueba import isolation_forest_scores

    df = generate_auth_events(GenConfig(n_users=12, n_days=4, seed=2))
    feats = build_features(df)
    scores = isolation_forest_scores(feats)
    assert scores.shape == (len(feats),)
    assert np.isfinite(scores).all()


@pytest.mark.slow
def test_end_to_end_detector_finds_lateral_movement():
    """Full pipeline: anomalies should rank far above chance and be detected early."""
    from log_ueba import isolation_forest_scores, summary_metrics

    set_seed(42)
    df = generate_auth_events(GenConfig(seed=42))
    feats = build_features(df)
    labels = feats["is_anomaly"].to_numpy()
    ts = feats["timestamp"].to_numpy()
    scores = isolation_forest_scores(feats)
    m = summary_metrics(scores, labels, ts)

    base_rate = labels.mean()
    # precision@25 must clear the base anomaly rate by a wide margin
    assert m["precision_at_k"]["25"] > max(0.3, 5 * base_rate)
    assert m["roc_auc"] > 0.8
    # first true hit should appear near the very top of the queue
    assert m["time_to_detect"]["rank"] <= 5
