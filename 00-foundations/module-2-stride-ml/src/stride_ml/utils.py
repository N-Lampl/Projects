"""Reproducibility helpers shared across the project (mirrors the repo convention)."""

from __future__ import annotations

import os
import random

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python's RNG + PYTHONHASHSEED so report ordering is deterministic.

    numpy/torch are seeded too when available, but this project is stdlib-only by
    default, so we import them lazily and never hard-depend on them.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:  # optional: only if the heavy ML libs happen to be installed
        import numpy as np  # noqa: WPS433

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is optional here
        pass
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch is optional here
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"
