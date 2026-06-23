"""Fast smoke tests (run in CI). The one slow test that trains every model
end-to-end is marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection import (
    build_models,
    confusion_at,
    make_transactions,
    pr_auc,
    precision_at_k,
    set_seed,
    split_xy,
    threshold_at_fpr,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_generator_is_deterministic_and_imbalanced():
    df1 = make_transactions(n=5000, fraud_rate=0.01, seed=7)
    df2 = make_transactions(n=5000, fraud_rate=0.01, seed=7)
    assert df1.equals(df2)  # same seed -> identical table
    rate = df1["is_fraud"].mean()
    # imbalanced: a small minority, but not zero (so models can learn)
    assert 0.001 < rate < 0.05
    assert df1["is_fraud"].sum() > 0


def test_split_xy_shapes_and_onehot():
    df = make_transactions(n=2000, seed=1)
    x, y = split_xy(df)
    assert len(x) == len(y) == len(df)
    # 6 numeric features (amount, hour, account_age_d, velocity_1h/24h, amount_to_avg)
    # + 8 one-hot merchant categories = 14 columns
    assert x.shape[1] == 14
    assert set(y.unique()) <= {0, 1}


def test_precision_at_k_bounds():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.05, 0.95])
    p = precision_at_k(y, scores, k=4)
    assert 0.0 <= p <= 1.0
    # the 4 highest scores (0.95,0.9,0.8,0.7) are all fraud -> p@4 == 1.0
    assert p == 1.0


def test_threshold_at_fpr_respects_budget():
    set_seed(0)
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.random.rand(90) * 0.5, 0.5 + np.random.rand(10) * 0.5])
    thr, recall = threshold_at_fpr(y, scores, target_fpr=0.05)
    fp = int(((scores >= thr) & (y == 0)).sum())
    # false positives stay within ~5% of the 90 negatives
    assert fp <= max(1, int(0.05 * 90) + 1)
    assert 0.0 <= recall <= 1.0


def test_confusion_counts_sum_to_n():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    c = confusion_at(y, scores, threshold=0.5)
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(y)


def test_build_models_has_defaults():
    models = build_models()
    # logreg + rf are always present regardless of optional deps
    assert "logreg" in models
    assert "rf" in models


@pytest.mark.slow
def test_end_to_end_models_beat_chance():
    """Train every available model; PR-AUC must clear the base rate by a margin."""
    set_seed(42)
    df = make_transactions(n=12_000, fraud_rate=0.01, seed=42)
    x, y = split_xy(df)
    from sklearn.model_selection import train_test_split

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    base_rate = y_te.mean()
    for name, model in build_models().items():
        model.fit(x_tr, y_tr)
        from fraud_detection import predict_scores

        s = predict_scores(model, x_te)
        ap = pr_auc(y_te, s)
        # a real injected signal: PR-AUC should be well above chance (base rate)
        assert ap > base_rate * 3, f"{name} PR-AUC {ap:.3f} not above chance {base_rate:.3f}"
