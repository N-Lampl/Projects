"""Fast smoke tests (run in CI). The one slow end-to-end test is marked @slow
and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from txn_anomaly import (
    FEATURES,
    IForestDetector,
    evaluate_scores,
    feature_matrix,
    make_transactions,
    precision_recall_at_k,
    set_seed,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_synthetic_stream_is_reproducible_and_labeled():
    df1 = make_transactions(n=2000, seed=7)
    df2 = make_transactions(n=2000, seed=7)
    # same seed -> identical engineered features
    assert np.allclose(feature_matrix(df1), feature_matrix(df2))
    # anomalies were actually injected, but stay a small minority (label-free regime)
    rate = df1["is_anomaly"].mean()
    assert 0 < rate < 0.05
    assert set(df1["anomaly_type"].unique()) >= {"normal"}


def test_feature_matrix_shape_and_finiteness():
    df = make_transactions(n=1500, seed=1)
    X = feature_matrix(df)
    assert X.shape == (len(df), len(FEATURES))
    assert np.isfinite(X).all()


def test_precision_at_k_perfect_ranking():
    # scores perfectly rank the 3 positives first -> P@3 == 1.0, R@3 == 1.0
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    scores = np.array([9.0, 8.0, 7.0, 1.0, 0.5, 0.2, 0.1])
    p, r = precision_recall_at_k(y, scores, 3)
    assert p == 1.0 and r == 1.0


def test_iforest_scores_higher_on_injected_anomalies():
    """Core invariant: the unsupervised detector ranks injected anomalies above
    normal traffic on average, without ever seeing the labels."""
    set_seed(42)
    df = make_transactions(n=4000, seed=42)
    X = feature_matrix(df)
    y = df["is_anomaly"].to_numpy().astype(int)
    det = IForestDetector(seed=42).fit(X)
    scores = det.score(X)
    assert scores[y == 1].mean() > scores[y == 0].mean()


@pytest.mark.slow
def test_end_to_end_detection_quality():
    """Full pipeline: detector should beat the random-ranking PR baseline by a wide
    margin and clear a useful PR-AUC bar."""
    set_seed(42)
    df = make_transactions(n=12000, seed=42)
    X = feature_matrix(df)
    y = df["is_anomaly"].to_numpy().astype(int)
    det = IForestDetector(seed=42).fit(X)
    scores = det.score(X)
    m = evaluate_scores(y, scores, contamination=0.012)
    base_rate = y.mean()
    assert m["pr_auc"] > 5 * base_rate  # far above random ranking
    assert m["roc_auc"] > 0.85
