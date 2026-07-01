"""Aspect-based sentiment: which *parts* of the car do reviewers like?

Lightweight and offline-capable: for each aspect we pull the review sentences
that mention it (keyword lexicon, word-boundary), then score just those
sentences with the SAME sentiment backend. No dedicated ABSA model to download;
the attribution stays transparent (you can see the sentences that drove it).
"""

from __future__ import annotations

import re

import pandas as pd

from .parsing import UNKNOWN
from .sentiment import SentimentBackend

ASPECT_LEXICON: dict[str, list[str]] = {
    "performance": [
        "performance",
        "acceleration",
        "horsepower",
        "hp",
        "engine",
        "power",
        "fast",
        "torque",
        "handling",
        "speed",
        "quick",
        "sluggish",
    ],
    "comfort": [
        "comfort",
        "comfortable",
        "seats",
        "seat",
        "ride",
        "legroom",
        "quiet",
        "noisy",
        "interior",
        "spacious",
        "cramped",
        "suspension",
    ],
    "reliability": [
        "reliable",
        "reliability",
        "breakdown",
        "repair",
        "dependable",
        "problem",
        "issue",
        "recall",
        "warranty",
        "broke",
        "failure",
        "maintenance",
    ],
    "price": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "value",
        "worth",
        "affordable",
        "overpriced",
        "deal",
        "money",
        "msrp",
        "resale",
    ],
    "fuel_economy": [
        "mpg",
        "fuel",
        "gas",
        "economy",
        "mileage",
        "efficient",
        "hybrid",
        "consumption",
        "gas mileage",
        "tank",
    ],
    "safety": [
        "safety",
        "safe",
        "airbag",
        "brakes",
        "braking",
        "crash",
        "abs",
        "stability",
        "traction",
        "visibility",
    ],
}

_ASPECT_RE = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for aspect, kws in ASPECT_LEXICON.items()
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split review text into non-empty sentences (regex; no nltk dependency)."""
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if s.strip()]


def aspect_sentiment(
    df: pd.DataFrame,
    backend: SentimentBackend,
    text_col: str = "Review",
    make_col: str = "make",
    max_reviews: int = 5000,
    min_mentions: int = 5,
    seed: int = 42,
) -> dict:
    """Score aspect-mentioning sentences and aggregate by aspect and by brand.

    Returns ``{"overall": {aspect: {...}}, "by_brand": DataFrame}`` where the
    ``by_brand`` frame is aspects (rows) × makes (cols) of mean sentiment.
    """
    work = (
        df.sample(n=min(len(df), max_reviews), random_state=seed) if len(df) > max_reviews else df
    )

    makes: list[str] = []
    aspects: list[str] = []
    sentences: list[str] = []
    for make, text in zip(work[make_col], work[text_col], strict=False):
        for sent in split_sentences(text):
            for aspect, rx in _ASPECT_RE.items():
                if rx.search(sent):
                    makes.append(make)
                    aspects.append(aspect)
                    sentences.append(sent)

    if not sentences:
        empty = pd.DataFrame(index=list(ASPECT_LEXICON))
        return {"overall": {a: None for a in ASPECT_LEXICON}, "by_brand": empty}

    preds = backend.predict(sentences)
    long = pd.DataFrame({"make": makes, "aspect": aspects, "score": [p.score for p in preds]})

    overall: dict[str, dict | None] = {}
    for aspect in ASPECT_LEXICON:
        sub = long[long["aspect"] == aspect]
        if len(sub) < min_mentions:
            overall[aspect] = None
            continue
        by_make = (
            sub[sub["make"] != UNKNOWN]
            .groupby("make", observed=True)["score"]
            .agg(mean="mean", n="count")
        )
        by_make = by_make[by_make["n"] >= min_mentions]
        top = None
        if len(by_make):
            best = by_make["mean"].idxmax()
            top = {"make": str(best), "mean": round(float(by_make.loc[best, "mean"]), 4)}
        overall[aspect] = {
            "mean_sentiment": round(float(sub["score"].mean()), 4),
            "n_mentions": int(len(sub)),
            "top_brand": top,
        }

    by_brand = (
        long[long["make"] != UNKNOWN]
        .pivot_table(index="aspect", columns="make", values="score", aggfunc="mean")
        .reindex(list(ASPECT_LEXICON))
    )
    return {"overall": overall, "by_brand": by_brand}
