"""Unsupervised anomaly detectors.

Default: scikit-learn IsolationForest (always available).
Optional: a tiny PyTorch autoencoder scored by reconstruction error, imported
lazily inside a try/except so this module imports and runs with no torch.

Both expose ``fit(X)`` and ``score(X) -> higher == more anomalous``.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .utils import SEED


class IForestDetector:
    """IsolationForest wrapped so higher score == more anomalous (label-free)."""

    def __init__(self, contamination: float = 0.012, n_estimators: int = 300, seed: int = SEED):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> IForestDetector:
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        # score_samples: higher == more normal -> negate so higher == more anomalous
        return -self.model.score_samples(Xs)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


class AutoencoderDetector:
    """Tiny dense autoencoder; anomaly score = reconstruction MSE.

    Falls back to ``IForestDetector`` if torch is not installed, so callers can
    always construct and use it. Set ``self.backend`` to inspect which ran.
    """

    def __init__(self, contamination: float = 0.012, epochs: int = 40, seed: int = SEED):
        self.seed = seed
        self.epochs = epochs
        self.contamination = contamination
        self.backend = "torch" if torch_available() else "iforest-fallback"
        self.scaler = StandardScaler()
        self._fallback: IForestDetector | None = None
        self._net = None

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        if self.backend != "torch":
            self._fallback = IForestDetector(self.contamination, seed=self.seed).fit(X)
            return self

        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        d = Xs.shape[1]
        self._net = nn.Sequential(
            nn.Linear(d, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.ReLU(),
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, d),
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        xt = torch.from_numpy(Xs)
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self._net(xt)
            loss = loss_fn(out, xt)
            loss.backward()
            opt.step()
        self._net.eval()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.backend != "torch":
            assert self._fallback is not None
            return self._fallback.score(X)

        import torch

        Xs = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            out = self._net(torch.from_numpy(Xs)).numpy()
        return ((out - Xs) ** 2).mean(axis=1)
