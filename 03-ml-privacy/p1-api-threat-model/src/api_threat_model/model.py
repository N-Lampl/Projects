"""The served model.

The DEFAULT path uses a tiny scikit-learn logistic-regression classifier trained on
SYNTHETIC tabular data, so the serving app has something real to score without any
download. The model is deliberately small: this project is about the *security
controls around* a model-serving endpoint, not the model itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .utils import set_seed

N_FEATURES = 8


@dataclass
class ServedModel:
    """A trained classifier plus the metadata the API needs to validate inputs."""

    clf: LogisticRegression
    n_features: int
    classes: list[int]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(x)


def make_synthetic_dataset(
    n: int = 2000, n_features: int = N_FEATURES, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic two-class blob dataset in roughly [-4, 4] per feature."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n_features)
    x = rng.normal(scale=1.5, size=(n, n_features))
    logits = x @ w + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(int)
    return x.astype(np.float64), y


def train_model(n_features: int = N_FEATURES, seed: int = 42) -> ServedModel:
    """Train the served logistic-regression model on synthetic data."""
    x, y = make_synthetic_dataset(n_features=n_features, seed=seed)
    clf = LogisticRegression(max_iter=500)
    clf.fit(x, y)
    return ServedModel(clf=clf, n_features=n_features, classes=[int(c) for c in clf.classes_])
