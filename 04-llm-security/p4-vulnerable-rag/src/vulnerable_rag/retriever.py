"""Retrievers for the RAG lab.

Default: a TF-IDF + cosine retriever built on scikit-learn (always installed).
Optional: a sentence-transformers dense retriever, imported lazily so the module
still imports when that library is absent.

The retriever deliberately exposes its internals (`scores`, `rank`) so the
attack projects (p5/p6) can inspect *what* was retrieved and *why*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Document


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    rank: int


class TfidfRetriever:
    """Bag-of-words TF-IDF retriever with cosine similarity ranking."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        corpus_text = [f"{d.title}. {d.text}" for d in documents]
        self._matrix = self._vectorizer.fit_transform(corpus_text)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        """Return the top-k documents by cosine similarity to the query."""
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).ravel()
        order = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(self.documents[i], float(scores[i]), rank)
            for rank, i in enumerate(order)
        ]


class DenseRetriever:
    """Optional dense retriever using sentence-transformers.

    Imported lazily; raises a clear error if the optional dependency is missing
    so the default TF-IDF path is never blocked.
    """

    def __init__(self, documents: list[Document], model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "DenseRetriever needs `sentence-transformers` (optional). "
                "Install it or use TfidfRetriever (the default)."
            ) from exc
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self._model = SentenceTransformer(model_name)
        corpus_text = [f"{d.title}. {d.text}" for d in documents]
        self._embeddings = self._model.encode(corpus_text, normalize_embeddings=True)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:  # pragma: no cover
        q_emb = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q_emb.T).ravel()
        order = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(self.documents[i], float(scores[i]), rank)
            for rank, i in enumerate(order)
        ]
