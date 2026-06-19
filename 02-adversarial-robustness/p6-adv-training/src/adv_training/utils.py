"""Reproducibility helpers shared across the project."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch so `make run` is deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return torch.device("cpu")
