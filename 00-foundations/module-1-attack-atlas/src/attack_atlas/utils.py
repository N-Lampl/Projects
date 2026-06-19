"""Reproducibility helpers shared across the project.

This project is stdlib-only (no numpy/torch), so set_seed seeds Python's
`random` and PYTHONHASHSEED. The numpy/torch branches are optional and only
fire if those libraries happen to be installed, keeping the helper consistent
with the rest of the portfolio's `utils.set_seed(42)` convention.
"""

from __future__ import annotations

import os
import random

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python's RNG (+ numpy/torch if present) so builds are deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:  # optional, not a hard dependency of this stdlib-only project
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
