"""Reproducibility helpers shared across the project."""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy so `make run` is deterministic.

    torch is seeded too if it happens to be installed, but this project is
    pure scikit-learn and never requires it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional and unused here; seed it only if present
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
