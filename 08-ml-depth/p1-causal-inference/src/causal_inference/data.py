"""Data sources: a synthetic SCM by default, real IHDP benchmark as an option.

The default path is fully offline (:func:`causal_inference.scm.make_scm`) so tests
and CI never touch the network. :func:`load_ihdp` pulls the standard semi-synthetic
**IHDP** benchmark (Hill 2011, Shalit et al. simulations) — it has a known ATE from
the simulated potential outcomes — and is exercised only by the ``@slow`` test.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

IHDP_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"


@dataclass
class CausalDataset:
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    true_ate: float
    source: str


def load_ihdp(url: str = IHDP_URL, realization: int = 0) -> CausalDataset:
    """Download IHDP and return realization ``i`` with its simulated true ATE.

    Raises on any network/parse failure — callers fall back to the synthetic SCM.
    """
    from urllib.request import urlopen

    with urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted benchmark host)
        raw = resp.read()
    npz = np.load(io.BytesIO(raw))
    i = realization
    X = npz["x"][:, :, i]
    T = npz["t"][:, i].astype(float)
    Y = npz["yf"][:, i]
    true_ate = float((npz["mu1"][:, i] - npz["mu0"][:, i]).mean())
    return CausalDataset(X=X, T=T, Y=Y, true_ate=true_ate, source=f"IHDP realization {i}")
