"""Fast smoke tests (run in CI). One @slow end-to-end test is excluded via
`-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from dga_detection import (
    DGAClassifier,
    EntropyBaseline,
    extract_stats,
    make_dataset,
    set_seed,
    shannon_entropy,
    train_test_split_df,
)
from dga_detection.features import FeatureExtractor, second_level


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_make_dataset_is_balanced_and_deterministic():
    df1 = make_dataset(n_per_class=200, seed=7)
    df2 = make_dataset(n_per_class=200, seed=7)
    assert df1.equals(df2)  # reproducible
    assert (df1["label"] == 0).sum() == 200
    assert (df1["label"] == 1).sum() == 200
    # families present
    assert {"benign", "random", "hexish", "dict"} <= set(df1["family"])


def test_second_level_strips_tld():
    assert second_level("cloudshop.com") == "cloudshop"
    assert second_level("ABC.io") == "abc"


def test_shannon_entropy_bounds():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0  # zero entropy
    # uniform 2 symbols -> 1 bit/char
    assert abs(shannon_entropy("abab") - 1.0) < 1e-9


def test_dga_labels_have_higher_mean_entropy():
    df = make_dataset(n_per_class=500, seed=1)
    ent = extract_stats(df["domain"].tolist())[:, 1]
    y = df["label"].to_numpy()
    assert ent[y == 1].mean() > ent[y == 0].mean()


def test_feature_extractor_shapes_match():
    df = make_dataset(n_per_class=100, seed=2)
    fe = FeatureExtractor()
    xtr = fe.fit_transform(df["domain"].tolist())
    xte = fe.transform(df["domain"].head(10).tolist())
    assert xtr.shape[0] == len(df)
    assert xte.shape[1] == xtr.shape[1]  # same feature space


def test_extract_stats_in_valid_ranges():
    stats = extract_stats(["cloudshop.com", "x7k9q2zhf3mn.net"])
    # ratios (cols 2..5) live in [0, 1]
    assert stats[:, 2:6].min() >= 0.0
    assert stats[:, 2:6].max() <= 1.0


def test_entropy_baseline_threshold_between_classes():
    df = make_dataset(n_per_class=300, seed=3)
    y = df["label"].to_numpy()
    base = EntropyBaseline().fit(df["domain"].tolist(), y)
    ent = extract_stats(df["domain"].tolist())[:, 1]
    assert ent[y == 0].mean() <= base.threshold <= ent[y == 1].mean()


@pytest.mark.slow
def test_classifier_beats_baseline_end_to_end():
    """Train the real detector and confirm it (a) detects DGAs well and
    (b) outperforms the entropy baseline on ROC-AUC."""
    from sklearn.metrics import roc_auc_score

    set_seed(42)
    df = make_dataset(n_per_class=2000)
    tr, te = train_test_split_df(df)
    clf = DGAClassifier().fit(tr["domain"].tolist(), tr["label"].to_numpy())
    base = EntropyBaseline().fit(tr["domain"].tolist(), tr["label"].to_numpy())

    y_te = te["label"].to_numpy()
    auc_clf = roc_auc_score(y_te, clf.predict_proba(te["domain"].tolist()))
    auc_base = roc_auc_score(y_te, base.score(te["domain"].tolist()))

    assert auc_clf > 0.9
    assert auc_clf > auc_base
