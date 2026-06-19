"""Fast smoke tests (run in CI). The one slow test that trains end-to-end is
marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np

from phishing_url import (
    FEATURE_NAMES,
    build_classifier,
    evaluate,
    extract_features,
    extract_one,
    make_synthetic,
    set_seed,
    train_test_split_df,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_synthetic_is_deterministic_and_labeled():
    df1 = make_synthetic(n=200, seed=7)
    df2 = make_synthetic(n=200, seed=7)
    assert df1.equals(df2)  # same seed -> identical data
    assert set(df1["label"].unique()) <= {0, 1}
    assert df1["label"].sum() > 0 and df1["label"].sum() < len(df1)  # both classes present


def test_feature_extraction_shape_and_names():
    urls = ["https://www.google.com/search", "http://192.168.0.1@evil.tk/login"]
    X = extract_features(urls)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert X.dtype == np.float64


def test_features_flag_known_phishing_signals():
    """The @-trick + IP host + suspicious TLD URL should trip the relevant flags."""
    f = extract_one("http://paypal.com@203.0.113.9.tk/secure/verify?token=abcd")
    assert f["has_at"] == 1.0
    assert f["n_suspicious_words"] >= 2  # 'secure' + 'verify'
    benign = extract_one("https://www.wikipedia.org/help")
    assert benign["has_at"] == 0.0
    assert benign["is_https"] == 1.0


def test_split_is_disjoint_and_complete():
    df = make_synthetic(n=400, seed=1)
    tr, te = train_test_split_df(df, test_frac=0.25, seed=1)
    assert len(tr) + len(te) == len(df)
    assert set(tr["url"]).isdisjoint(set(te["url"]))


def test_classifier_builds_and_predicts_probabilities():
    set_seed(0)
    df = make_synthetic(n=300, seed=0)
    X = extract_features(df["url"])
    y = df["label"].to_numpy()
    pipe = build_classifier("logreg")
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(df),)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


import pytest  # noqa: E402


@pytest.mark.slow
def test_detector_separates_classes_end_to_end():
    """Train on synthetic URLs and confirm the detector is well above chance."""
    set_seed(42)
    df = make_synthetic(n=3000, seed=42)
    tr, te = train_test_split_df(df, test_frac=0.25, seed=42)
    pipe = build_classifier("logreg")
    pipe.fit(extract_features(tr["url"]), tr["label"].to_numpy())
    res = evaluate(pipe, extract_features(te["url"]), te["label"].to_numpy())
    assert res["roc_auc"] > 0.9  # the lexical signal is strong on synthetic data
    assert res["recall"] > 0.7
