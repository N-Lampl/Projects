"""Reproducibility helpers shared across the project (CPU-only, no torch)."""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy so `make detect` is deterministic.

    torch is seeded too *iff* it happens to be installed, but this project
    never requires it (classical scikit-learn only).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional: only if torch is present
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
