"""Reproducibility helpers shared across the project."""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy (and torch if present) so every run is deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional; the default path never needs it
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
