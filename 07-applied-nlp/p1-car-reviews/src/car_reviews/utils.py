"""Reproducibility + CPU helpers shared across the project."""

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
    try:  # torch is a hard dep for the HF backend, but keep the guarded form
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"


def configure_torch_threads() -> int:
    """Let torch use every core - the single biggest CPU inference lever.

    Returns the thread count set (0 if torch is unavailable).
    """
    try:
        import torch

        n = os.cpu_count() or 1
        torch.set_num_threads(n)
        return n
    except Exception:
        return 0
