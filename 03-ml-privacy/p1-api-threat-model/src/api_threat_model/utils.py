"""Reproducibility helpers shared across the project."""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if available) PyTorch so runs are deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for the privacy-controls path
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_device():
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    try:
        import torch

        return torch.device("cpu")
    except Exception:
        return "cpu"
