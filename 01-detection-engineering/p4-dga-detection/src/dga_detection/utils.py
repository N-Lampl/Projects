"""Reproducibility helpers shared across the project."""

from __future__ import annotations

import os
import random

import numpy as np

try:  # torch is only needed for the optional char-LSTM path
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is in the default deps
    _HAS_TORCH = False

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy (and torch if present) so runs are deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    if _HAS_TORCH:
        return torch.device("cpu")
    return "cpu"
