"""Topic modeling over the review corpus (scikit-learn, offline, deterministic).

Default is NMF over a TF-IDF matrix (parts-based factors read cleanly on review
text); LDA over counts is available with ``method="lda"``. Both are seeded. Each
topic gets its top terms, a 2-word heuristic label, a prevalence (share of
reviews for which it is the dominant topic), and the model it loads highest on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .parsing import UNKNOWN


def fit_topics(
    df: pd.DataFrame,
    text_col: str = "Review",
    group_col: str = "model_key",
    n_topics: int = 8,
    method: str = "nmf",
    max_features: int = 5000,
    max_docs: int = 10000,
    top_terms: int = 8,
    seed: int = 42,
) -> dict:
    """Fit a topic model and return a JSON-friendly summary dict."""
    work = df.sample(n=min(len(df), max_docs), random_state=seed) if len(df) > max_docs else df
    texts = work[text_col].astype(str)

    min_df = max(1, min(5, len(work) // 20))
    if method == "lda":
        vec = CountVectorizer(stop_words="english", min_df=min_df, max_features=max_features)
    else:
        vec = TfidfVectorizer(stop_words="english", min_df=min_df, max_features=max_features)
    try:
        matrix = vec.fit_transform(texts)
    except ValueError:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        matrix = vec.fit_transform(texts)
    terms = np.asarray(vec.get_feature_names_out())

    k = max(2, min(n_topics, matrix.shape[0] - 1, matrix.shape[1]))
    if method == "lda":
        model = LatentDirichletAllocation(
            n_components=k, learning_method="batch", max_iter=20, random_state=seed
        )
    else:
        model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=seed)
    doc_topic = model.fit_transform(matrix)
    components = model.components_

    dominant = doc_topic.argmax(axis=1)
    group_index = work[group_col].to_numpy()
    topics: list[dict] = []
    for t in range(k):
        top_idx = np.argsort(components[t])[::-1][:top_terms]
        words = [str(terms[i]) for i in top_idx]
        prevalence = float(np.mean(dominant == t))

        loadings = pd.DataFrame({"model_key": group_index, "load": doc_topic[:, t]})
        loadings = loadings[loadings["model_key"] != UNKNOWN]
        by_model = loadings.groupby("model_key", observed=True)["load"].agg(["mean", "count"])
        by_model = by_model[by_model["count"] >= 3]
        top_model = str(by_model["mean"].idxmax()) if len(by_model) else None

        topics.append(
            {
                "id": t,
                "label": " / ".join(words[:2]),
                "top_terms": words,
                "prevalence": round(prevalence, 4),
                "top_model": top_model,
            }
        )

    topics.sort(key=lambda d: d["prevalence"], reverse=True)
    return {"method": method, "n_topics": k, "topics": topics}
