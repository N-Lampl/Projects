"""Synthetic data for the hierarchical model — offline and fully known.

The default path draws ``J`` groups whose true means come from a global
``N(mu, tau)``, then noisy observations per group. Because the true group means
are stored on the dataset, every posterior can be *scored* against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HierDataset:
    """A hierarchical dataset with known generating parameters."""

    y: np.ndarray  # (J, n) noisy observations per group
    group_true_means: np.ndarray  # (J,) the latent true group means
    sigma: np.ndarray  # (J,) observation sd per group (known)
    mu: float  # global mean of the group-mean prior
    tau: float  # global sd of the group-mean prior
    source: str

    @property
    def n_groups(self) -> int:
        return int(self.y.shape[0])

    @property
    def group_ybar(self) -> np.ndarray:
        """Per-group sample mean (the no-pooling MLE)."""
        return self.y.mean(axis=1)


def make_hierarchical(
    n_groups: int = 16,
    n_per_group: int = 4,
    mu: float = 5.0,
    tau: float = 3.0,
    sigma: float = 7.0,
    seed: int = 0,
) -> HierDataset:
    """Draw ``n_groups`` groups from a global ``N(mu, tau)`` with noisy readings.

    ``sigma`` is the (known) observation sd shared across groups. Small
    ``n_per_group`` makes per-group MLEs noisy, so partial pooling has room to
    help — that is the whole point of the shrinkage demo.
    """
    rng = np.random.default_rng(seed)
    theta = mu + tau * rng.standard_normal(n_groups)  # true group means
    y = theta[:, None] + sigma * rng.standard_normal((n_groups, n_per_group))
    sigmas = np.full(n_groups, float(sigma))
    return HierDataset(
        y=y,
        group_true_means=theta,
        sigma=sigmas,
        mu=float(mu),
        tau=float(tau),
        source=f"synthetic hierarchical (J={n_groups}, n={n_per_group})",
    )
