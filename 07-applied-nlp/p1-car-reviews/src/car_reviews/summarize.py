"""Per-model review summaries.

Default is **extractive** (TF-IDF sentence ranking with light redundancy control) -
offline, instant, deterministic. An optional **abstractive** path uses a small
CPU-runnable HF summarizer (``sshleifer/distilbart-cnn-12-6``); it is slow on CPU,
so it only ever runs over the top-N most-reviewed models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .aspects import split_sentences
from .parsing import UNKNOWN


def extractive_summary(texts: Sequence[str], max_sentences: int = 4) -> str:
    """Rank sentences by TF-IDF salience and return the top few (low-redundancy)."""
    sentences: list[str] = []
    for t in texts:
        sentences.extend(split_sentences(t))
    sentences = [s for s in sentences if len(s.split()) >= 4]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vec = TfidfVectorizer(stop_words="english", min_df=1)
    matrix = vec.fit_transform(sentences)
    norms = np.asarray(np.sqrt(matrix.multiply(matrix).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    salience = np.asarray(matrix.sum(axis=1)).ravel() / norms  # length-normalised

    order = np.argsort(salience)[::-1]
    picked: list[int] = []
    for idx in order:
        row = matrix[idx]
        redundant = False
        for j in picked:
            sim = float(row.multiply(matrix[j]).sum()) / (norms[idx] * norms[j])
            if sim > 0.6:
                redundant = True
                break
        if not redundant:
            picked.append(idx)
        if len(picked) >= max_sentences:
            break
    return " ".join(sentences[i] for i in sorted(picked))


def abstractive_summary(
    texts: Sequence[str], max_input_chars: int = 3000, max_output_tokens: int = 130
) -> str:
    """Optional HF abstractive summary (distilbart). Downloads weights on first use."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - optional heavy path
        raise SystemExit(
            "transformers not installed. Abstractive summaries need it; "
            "use --summaries extractive for the offline path."
        ) from exc
    if not hasattr(abstractive_summary, "_pipe"):
        abstractive_summary._pipe = pipeline(
            "summarization", model="sshleifer/distilbart-cnn-12-6", device=-1
        )
    joined = " ".join(str(t) for t in texts)[:max_input_chars]
    result = abstractive_summary._pipe(
        joined, max_length=max_output_tokens, min_length=30, truncation=True, do_sample=False
    )
    return result[0]["summary_text"].strip()


def summarize_top_models(
    df: pd.DataFrame,
    method: str = "extractive",
    text_col: str = "Review",
    group_col: str = "model_key",
    top_n: int = 10,
    min_reviews: int = 30,
    max_reviews_per_model: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, str]:
    """Summarize the ``top_n`` most-reviewed models. Returns ``{model_key: summary}``."""
    if method == "none":
        return {}
    counts = df[df[group_col] != UNKNOWN][group_col].value_counts()
    keep = [g for g, c in counts.items() if c >= min_reviews][:top_n]
    summaries: dict[str, str] = {}
    for g in keep:
        pool = df[df[group_col] == g][text_col]
        if len(pool) > max_reviews_per_model:
            pool = pool.sample(n=max_reviews_per_model, random_state=seed)
        texts = pool.astype(str).tolist()
        if verbose and method == "abstractive":
            print(f"[summarize] {g} ({len(texts)} reviews)...", flush=True)
        summaries[str(g)] = (
            abstractive_summary(texts) if method == "abstractive" else extractive_summary(texts)
        )
    return summaries
