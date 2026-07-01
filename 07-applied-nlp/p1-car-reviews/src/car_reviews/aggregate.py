"""Aggregate per-review sentiment into brand- and model-level rankings.

Two guards keep the rankings honest on 600+ models where most have a handful of
reviews: a ``min_reviews`` gate for the ranked tables, and **empirical-Bayes
shrinkage** toward the global mean so a single 5-star review can't top the chart:

    shrunk = (n * group_mean + m * global_mean) / (n + m)
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .parsing import UNKNOWN
from .sentiment import Prediction


def attach_scores(df: pd.DataFrame, preds: Sequence[Prediction]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``score`` and ``pred_star`` columns."""
    out = df.copy()
    out["score"] = [p.score for p in preds]
    out["pred_star"] = [p.star for p in preds]
    return out


def cap_per_group(df: pd.DataFrame, by: str, cap: int | None, seed: int = 42) -> pd.DataFrame:
    """Randomly down-sample each ``by`` group to at most ``cap`` rows (seeded)."""
    if not cap:
        return df
    parts = [
        g.sample(n=min(len(g), cap), random_state=seed) for _, g in df.groupby(by, observed=True)
    ]
    return pd.concat(parts).reset_index(drop=True)


def _aggregate(
    df: pd.DataFrame, by: str, score_col: str, m: float, min_reviews: int
) -> pd.DataFrame:
    global_mean = float(df[score_col].mean())
    grp = df[df[by] != UNKNOWN].groupby(by, observed=True)[score_col]
    table = grp.agg(mean_sentiment="mean", n="count").reset_index()
    table["mean_sentiment_shrunk"] = (table["n"] * table["mean_sentiment"] + m * global_mean) / (
        table["n"] + m
    )
    table = table[table["n"] >= min_reviews]
    table = table.sort_values("mean_sentiment_shrunk", ascending=False).reset_index(drop=True)
    table["mean_sentiment"] = table["mean_sentiment"].round(4)
    table["mean_sentiment_shrunk"] = table["mean_sentiment_shrunk"].round(4)
    return table


def brand_sentiment(
    df: pd.DataFrame, score_col: str = "score", m: float = 20, min_reviews: int = 1
) -> pd.DataFrame:
    """Rank makes by shrunk mean sentiment. Columns: make, mean_sentiment, n, shrunk."""
    return _aggregate(df, "make", score_col, m, min_reviews)


def model_sentiment(
    df: pd.DataFrame, score_col: str = "score", m: float = 20, min_reviews: int = 30
) -> pd.DataFrame:
    """Rank make-namespaced models by shrunk mean sentiment (``n >= min_reviews``)."""
    return _aggregate(df, "model_key", score_col, m, min_reviews)


def rank_top_bottom(table: pd.DataFrame, k: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top-k, bottom-k) rows of an already-sorted ranking table."""
    top = table.head(k).reset_index(drop=True)
    bottom = table.tail(k).sort_values("mean_sentiment_shrunk").reset_index(drop=True)
    return top, bottom


def to_records(table: pd.DataFrame, key: str, k: int | None = None) -> list[dict]:
    """Serialize a ranking table to compact JSON records for ``metrics.json``."""
    rows = table.head(k) if k else table
    return [
        {key: r[key], "mean_sentiment": float(r["mean_sentiment"]), "n": int(r["n"])}
        for _, r in rows.iterrows()
    ]
