"""Character-level features for DGA detection.

We extract two complementary signals from the second-level label (the part before
the TLD):

  1. **Hand-crafted statistics** -- length, Shannon entropy, vowel/digit ratios,
     and the fraction of characters that are consonants. DGA labels tend to be
     long and high-entropy with few vowels.
  2. **Character n-grams** -- a TF-IDF over 2- and 3-grams. This is what catches
     the *dictionary* DGA family, whose entropy is benign-looking but whose
     character transitions are unusual.

The default classifier uses both, concatenated. ``extract_stats`` alone is enough
for the simple entropy-threshold baseline.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

try:  # scipy ships with scikit-learn; guard anyway for a clean import
    from scipy.sparse import csr_matrix, hstack
except ImportError:  # pragma: no cover
    csr_matrix = None
    hstack = None

from sklearn.feature_extraction.text import TfidfVectorizer

_VOWELS = set("aeiou")
_STAT_NAMES = ["length", "entropy", "vowel_ratio", "digit_ratio", "consonant_ratio", "unique_ratio"]


def second_level(domain: str) -> str:
    """Return the registrable label, stripped of the TLD (best-effort)."""
    parts = domain.lower().split(".")
    return parts[0] if parts else domain.lower()


def shannon_entropy(s: str) -> float:
    """Shannon entropy (bits/char) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_stats(domains: list[str]) -> np.ndarray:
    """Hand-crafted per-domain statistics. Shape (n, len(_STAT_NAMES))."""
    rows = []
    for d in domains:
        s = second_level(d)
        n = max(len(s), 1)
        vowels = sum(ch in _VOWELS for ch in s)
        digits = sum(ch.isdigit() for ch in s)
        letters = sum(ch.isalpha() for ch in s)
        consonants = letters - vowels
        rows.append(
            [
                len(s),
                shannon_entropy(s),
                vowels / n,
                digits / n,
                consonants / n,
                len(set(s)) / n,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def stats_frame(domains: list[str]) -> pd.DataFrame:
    """Same as ``extract_stats`` but as a labelled DataFrame (for plots)."""
    return pd.DataFrame(extract_stats(domains), columns=_STAT_NAMES)


class FeatureExtractor:
    """Fit a char-n-gram TF-IDF and standardize the hand-crafted stats.

    ``fit_transform`` / ``transform`` return a sparse matrix combining both blocks
    so it can be fed straight into a scikit-learn LogisticRegression.
    """

    def __init__(self, ngram_range: tuple[int, int] = (2, 3), max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
        )
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _labels(self, domains: list[str]) -> list[str]:
        return [second_level(d) for d in domains]

    def _scale_stats(self, stats: np.ndarray) -> np.ndarray:
        return (stats - self._mean) / self._std

    def fit_transform(self, domains: list[str]):
        labels = self._labels(domains)
        ngrams = self.vectorizer.fit_transform(labels)
        stats = extract_stats(domains)
        self._mean = stats.mean(axis=0)
        self._std = stats.std(axis=0) + 1e-9
        scaled = self._scale_stats(stats)
        return hstack([ngrams, csr_matrix(scaled)]).tocsr()

    def transform(self, domains: list[str]):
        if self._mean is None:
            raise RuntimeError("FeatureExtractor must be fit before transform()")
        labels = self._labels(domains)
        ngrams = self.vectorizer.transform(labels)
        scaled = self._scale_stats(extract_stats(domains))
        return hstack([ngrams, csr_matrix(scaled)]).tocsr()
