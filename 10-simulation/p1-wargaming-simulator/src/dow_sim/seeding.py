"""Reproducibility helpers (CPU-only, numpy-based).

The load-bearing rule for this project: **every stochastic call threads an explicit
``numpy.random.Generator``**. No module ever touches global ``random``/``np.random`` state
during a battle, so thousands of Monte Carlo runs stay independent yet fully reproducible —
and parallelise cleanly across processes.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python + NumPy global state (used for top-level determinism, not per-battle)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent, reproducible generator for a single battle."""
    return np.random.default_rng(seed)


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
