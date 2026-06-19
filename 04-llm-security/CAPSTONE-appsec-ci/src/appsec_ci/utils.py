"""Reproducibility helpers shared across the CI pipeline.

Mirrors the convention used everywhere else in the portfolio: ``set_seed(42)``
seeds Python/NumPy (and torch if available) and ``get_device()`` returns CPU.

``torch`` is imported lazily so this module (and the whole CI harness) imports
with ONLY numpy + matplotlib present -- the offline default needs nothing else.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for this offline-first harness
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - torch absent is fine on the default path
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop).

    Returns a plain string so callers do not need torch installed.
    """
    return "cpu"
