"""Fast smoke tests (run in CI). They use the offline STUB backend + the seeded
synthetic review generator, so nothing hits the network. The one slow test that
loads a real HuggingFace model on a tiny sample is marked @slow and excluded
from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from car_reviews import (
    ASPECT_LEXICON,
    Calibrator,
    StubBackend,
    add_parsed_columns,
    aspect_sentiment,
    attach_scores,
    brand_sentiment,
    fit_topics,
    make_coverage,
    make_reviews,
    make_splits,
    parse_vehicle_title,
    predictions_from_star_probs,
    set_seed,
    tfidf_distinctive_terms,
    validate_against_rating,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b)


def test_parse_multiword_make():
    p = parse_vehicle_title("2015 Land Rover Range Rover Sport 4dr SUV AWD")
    assert p.make == "Land Rover"
    assert p.year == 2015
    assert p.model_key.startswith("Land Rover")


def test_parse_hyphen_make():
    p = parse_vehicle_title("2012 Mercedes-Benz C-Class C300 Sedan")
    assert p.make == "Mercedes-Benz"
    assert p.model.startswith("C-Class")


def test_parse_alias_folds_to_canonical():
    assert parse_vehicle_title("2016 Chevy Malibu LT Sedan").make == "Chevrolet"
    assert parse_vehicle_title("2014 VW Jetta SE").make == "Volkswagen"


def test_parse_missing_year_and_model_key():
    p = parse_vehicle_title("Toyota Previa Minivan LE 3dr")
    assert p.year is None
    assert p.make == "Toyota"
    assert p.model == "Previa"  # stops at the body-style token "Minivan"
    assert p.model_key == "Toyota Previa"


def test_parse_unknown_make_is_bucketed():
    p = parse_vehicle_title("2010 Batmobile Tumbler")
    assert p.make == "UNKNOWN"


def test_synthetic_reproducible_and_valid():
    df1 = make_reviews(n=200, seed=7)
    df2 = make_reviews(n=200, seed=7)
    assert df1.equals(df2)
    assert set(df1["Rating"].unique()) <= {1, 2, 3, 4, 5}
    assert (df1["Review"].str.len() > 0).all()


def test_stub_backend_polarity():
    b = StubBackend()
    pos = b.predict(["I love this car, it is excellent and very reliable."])[0]
    neg = b.predict(["I regret this, it is terrible and totally unreliable."])[0]
    assert pos.score > neg.score
    assert 1.0 <= neg.star <= pos.star <= 5.0


def test_make_coverage_full_on_synthetic():
    df = add_parsed_columns(make_reviews(n=100, seed=4))
    assert make_coverage(df) == 1.0


def test_pipeline_recovers_planted_brand_order():
    df = add_parsed_columns(make_reviews(n=500, seed=42))
    preds = StubBackend().predict(df["Review"].tolist())
    scored = attach_scores(df, preds)
    means = brand_sentiment(scored, min_reviews=1).set_index("make")["mean_sentiment"]
    # planted quality: Lexus/Toyota high, Suzuki/Fiat low
    assert means["Lexus"] > means["Suzuki"]
    assert means["Toyota"] > means["Fiat"]


def test_aspect_extraction_has_signal():
    df = add_parsed_columns(make_reviews(n=200, seed=1))
    res = aspect_sentiment(df, StubBackend(), max_reviews=200)
    assert res["by_brand"].shape[0] == len(ASPECT_LEXICON)
    assert any(res["overall"][a] is not None for a in ASPECT_LEXICON)


def test_validate_against_rating_shapes():
    df = add_parsed_columns(make_reviews(n=150, seed=2))
    preds = StubBackend().predict(df["Review"].tolist())
    val = validate_against_rating(preds, df["Rating"], scale="1-5")
    assert {"exact_accuracy", "within_1_accuracy", "mae", "confusion"} <= set(val)
    assert len(val["confusion"]) == 5 and len(val["confusion"][0]) == 5
    assert 0.0 <= val["within_1_accuracy"] <= 1.0
    # stub sentiment is built to track the planted rating -> most within +/-1
    assert val["within_1_accuracy"] > 0.5


def test_keywords_and_topics_shapes():
    df = add_parsed_columns(make_reviews(n=300, seed=3))
    scored = attach_scores(df, StubBackend().predict(df["Review"].tolist()))
    kw = tfidf_distinctive_terms(scored, min_reviews=5)
    assert isinstance(kw, dict) and len(kw) >= 1
    topics = fit_topics(scored, n_topics=4, seed=3)
    assert topics["n_topics"] >= 2
    assert len(topics["topics"]) == topics["n_topics"]
    assert all("label" in t and "top_terms" in t for t in topics["topics"])


def test_make_splits_disjoint_and_deterministic():
    df = add_parsed_columns(make_reviews(n=600, seed=5))
    tr, val, te = make_splits(df, test_size=100, val_size=60, seed=5)
    assert len(tr) + len(val) + len(te) == len(df)
    assert len(te) == 100 and len(val) == 60
    _tr2, _v2, te2 = make_splits(df, test_size=100, val_size=60, seed=5)
    assert te.reset_index(drop=True).equals(te2.reset_index(drop=True))  # deterministic


def test_calibrator_fixes_systematic_bias():
    # a biased "model": probability mass centered one star BELOW the truth.
    rng = np.random.default_rng(0)
    n = 2400
    y = rng.integers(1, 6, size=n)
    probs = np.zeros((n, 5))
    for i, yi in enumerate(y):
        center = max(1, yi - 1)
        w = np.exp(-np.abs(np.arange(1, 6) - center))
        probs[i] = w / w.sum()
    tr, te = slice(0, 1200), slice(1200, n)
    base = validate_against_rating(predictions_from_star_probs(probs[te]), y[te], "1-5")
    cal = Calibrator(seed=0).fit(probs[tr], y[tr])
    calib = validate_against_rating(cal.predict(probs[te]), y[te], "1-5")
    assert calib["mae"] < base["mae"]  # calibration removes the systematic shift
    assert calib["exact_accuracy"] >= base["exact_accuracy"]


@pytest.mark.slow
def test_finetune_smoke(tmp_path):
    """Fine-tune DistilBERT 1 epoch on a tiny sample; it must save, load, and score."""
    pytest.importorskip("transformers")
    from car_reviews import finetune_distilbert, load_finetuned

    df = add_parsed_columns(make_reviews(n=64, seed=1))
    out = finetune_distilbert(
        df.iloc[:52],
        val_df=None,
        out_dir=tmp_path / "ft",
        epochs=1,
        max_length=64,
        batch_size=8,
        seed=1,
        verbose=False,
    )
    ft = load_finetuned(out, max_length=64, batch_size=8, verbose=False)
    probs, labels = ft.predict_proba(df["Review"].tolist()[:8])
    assert probs.shape == (8, 5)
    assert labels == [1, 2, 3, 4, 5]
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)


@pytest.mark.slow
def test_end_to_end_hf_small():
    """Load a REAL small HF model on a tiny synthetic sample (downloads weights)."""
    pytest.importorskip("transformers")
    from scipy.stats import spearmanr

    from car_reviews import get_sentiment_backend

    df = add_parsed_columns(make_reviews(n=48, seed=42))
    backend = get_sentiment_backend("hf", model="sst2", batch_size=8, verbose=False)
    preds = backend.predict(df["Review"].tolist())
    assert len(preds) == len(df)
    # predicted sentiment should positively track the ground-truth star rating
    rho = spearmanr([p.score for p in preds], df["Rating"]).correlation
    assert rho > 0.2
