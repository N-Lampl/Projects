"""Reproducibility helpers shared across the IDS pipeline library."""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if available) PyTorch so runs are deterministic.

    torch is imported lazily: the IDS pipeline default path only needs
    scikit-learn, so we don't want to hard-require torch just to set a seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional - keep determinism consistent if torch is installed
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
