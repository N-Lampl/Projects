"""Distinctive keywords per model via TF-IDF (offline, scikit-learn).

For each model with enough reviews, we surface the terms whose mean TF-IDF weight
in *that model's* reviews most exceeds their weight across the whole corpus — i.e.
what reviewers say about this model that they don't say about cars in general.
An optional KeyBERT path (``method="keybert"``) is available behind an extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .parsing import UNKNOWN


def tfidf_distinctive_terms(
    df: pd.DataFrame,
    text_col: str = "Review",
    group_col: str = "model_key",
    top_models: int = 30,
    top_k: int = 10,
    min_reviews: int = 30,
) -> dict[str, list[str]]:
    """Return ``{model_key: [distinctive terms]}`` for the most-reviewed models."""
    counts = df[df[group_col] != UNKNOWN][group_col].value_counts()
    keep = [g for g, c in counts.items() if c >= min_reviews][:top_models]
    if not keep:
        return {}

    min_df = max(1, min(5, len(df) // 20))
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=min_df,
        max_features=20000,
        sublinear_tf=True,
    )
    try:
        matrix = vec.fit_transform(df[text_col].astype(str))
    except ValueError:  # empty vocabulary on a tiny corpus
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = vec.fit_transform(df[text_col].astype(str))
    terms = np.asarray(vec.get_feature_names_out())
    global_mean = np.asarray(matrix.mean(axis=0)).ravel()

    group_index = df[group_col].to_numpy()
    out: dict[str, list[str]] = {}
    for g in keep:
        rows = np.where(group_index == g)[0]
        group_mean = np.asarray(matrix[rows].mean(axis=0)).ravel()
        distinct = group_mean - global_mean
        top_idx = np.argsort(distinct)[::-1][:top_k]
        out[str(g)] = [str(terms[i]) for i in top_idx if distinct[i] > 0]
    return out
