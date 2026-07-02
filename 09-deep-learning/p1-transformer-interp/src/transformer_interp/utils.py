"""Reproducibility helpers shared across the project (CPU-only, torch-based)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and torch so every run is deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
