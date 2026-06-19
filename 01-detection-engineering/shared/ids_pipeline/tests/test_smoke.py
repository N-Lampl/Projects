"""Fast smoke / invariant tests (run in CI). One @slow end-to-end test that
trains the full pipeline is excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from ids_pipeline import (
    build_pipeline,
    evaluate,
    load_data,
    make_synthetic_flows,
    precision_at_k,
    set_seed,
)
from ids_pipeline.data import LABEL_COL


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_synthetic_generator_is_deterministic_and_balanced_as_requested():
    df1 = make_synthetic_flows(n_samples=2000, attack_fraction=0.3, seed=7)
    df2 = make_synthetic_flows(n_samples=2000, attack_fraction=0.3, seed=7)
    assert df1.equals(df2)
    assert len(df1) == 2000
    # attack_fraction respected (exact, since we round then split deterministically)
    assert df1[LABEL_COL].sum() == 600


def test_load_data_split_is_stratified_and_disjoint():
    ds = load_data(synthetic=True, n_samples=4000, test_size=0.25)
    assert len(ds.X_train) == 3000
    assert len(ds.X_test) == 1000
    # both classes present in both splits (stratified)
    assert set(ds.y_train.unique()) == {0, 1}
    assert set(ds.y_test.unique()) == {0, 1}
    assert ds.n_features == len(ds.numeric_features) + len(ds.categorical_features)


def test_preprocessor_is_fit_on_train_only_no_leak():
    """The scaler must learn statistics from TRAIN only. We verify by checking
    the StandardScaler's learned mean equals the TRAIN mean, not the full-data
    mean (which would indicate leakage)."""
    ds = load_data(synthetic=True, n_samples=4000, seed=1)
    pipe = build_pipeline(ds)
    pipe.fit(ds.X_train, ds.y_train)
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"]
    train_mean = ds.X_train[ds.numeric_features].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_mean, rtol=1e-6)
    # and it must differ from the mean over the full dataset (train+test)
    full_mean = (
        np.concatenate([ds.X_train[ds.numeric_features].to_numpy(),
                        ds.X_test[ds.numeric_features].to_numpy()])
        .mean(axis=0)
    )
    assert not np.allclose(scaler.mean_, full_mean, rtol=1e-6)


def test_precision_at_k_perfect_ranking():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    assert precision_at_k(y_true, y_score, k=3) == 1.0
    assert precision_at_k(y_true, y_score, k=5) == pytest.approx(0.6)
    assert precision_at_k(y_true, y_score, k=0) == 0.0


def test_evaluate_returns_expected_keys_and_ranges():
    set_seed(42)
    ds = load_data(synthetic=True, n_samples=3000)
    pipe = build_pipeline(ds)
    pipe.fit(ds.X_train, ds.y_train)
    m = evaluate(pipe, ds, ks=(50,))
    for key in ("precision", "recall", "f1", "roc_auc", "confusion_matrix", "precision_at_k"):
        assert key in m
    for key in ("precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= m[key] <= 1.0
    assert np.array(m["confusion_matrix"]).shape == (2, 2)


@pytest.mark.slow
def test_full_pipeline_learns_the_attack_signal():
    """Train the full default pipeline and confirm it clears a sensible SOC bar."""
    set_seed(42)
    ds = load_data(synthetic=True, n_samples=12000)
    pipe = build_pipeline(ds)
    pipe.fit(ds.X_train, ds.y_train)
    m = evaluate(pipe, ds)
    assert m["roc_auc"] > 0.9
    assert m["f1"] > 0.75
    assert m["recall"] > 0.7
