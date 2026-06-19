"""Reproducibility helpers shared across the network labs.

This module deliberately has no heavy deps so the package always imports.
torch/numpy are imported lazily inside `set_seed` so a pure-stdlib run still works.
"""

from __future__ import annotations

import os
import random


class _Device:
    """Tiny stand-in so callers can `get_device()` without importing torch.

    The network labs are CPU-only and don't use torch for computation; this keeps
    the seed-project API surface (`set_seed`, `get_device`) without a torch dependency.
    """

    type = "cpu"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "device(type='cpu')"


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python's RNG (and numpy/torch if available) so traces are deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:  # optional, only if numpy is installed
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:  # optional, only if torch is installed
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_device() -> _Device:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return _Device()
