"""Reproducibility helpers shared across the project (CPU-only, numpy-based)."""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy so every run is deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
