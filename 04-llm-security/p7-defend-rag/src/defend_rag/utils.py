"""Reproducibility + path helpers shared across the defense project."""

from __future__ import annotations

import os
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch so training + eval are deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Locate + import the authorized target (../p4-vulnerable-rag) lazily by path.
# --------------------------------------------------------------------------- #
_P4_SRC = Path(__file__).resolve().parents[3] / "p4-vulnerable-rag" / "src"


def _ensure_target_on_path() -> None:
    if _P4_SRC.is_dir() and str(_P4_SRC) not in sys.path:
        sys.path.insert(0, str(_P4_SRC))


@lru_cache(maxsize=1)
def load_target():
    """Return the p4 `vulnerable_rag` module, or raise a clear error.

    p7 defends the deliberately-vulnerable RAG built in p4. We import it lazily
    by path so this package still imports if p4 is absent.
    """
    _ensure_target_on_path()
    try:
        import vulnerable_rag  # type: ignore
    except ImportError as exc:  # pragma: no cover - only if p4 is missing
        raise ImportError(
            "Could not import the target p4-vulnerable-rag. Expected it at "
            f"{_P4_SRC}. Build p4 first; this project defends it."
        ) from exc
    return vulnerable_rag
