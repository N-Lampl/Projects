"""Reproducibility helpers + locating the shared ids_pipeline library by path."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for deterministic runs.

    torch is imported lazily: this project's default path only needs
    scikit-learn, so we never hard-require torch just to set a seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional — keep determinism consistent if torch is present
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_device() -> str:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return "cpu"


def shared_pipeline_src() -> Path:
    """Absolute path to ``../shared/ids_pipeline/src`` (the reusable IDS library).

    This project deliberately does NOT vendor the pipeline — it imports the
    shared library by path so ``p1-nids-baseline`` and the adversarial-IDS
    capstone share exactly one leak-free preprocessing + model implementation.
    """
    here = Path(__file__).resolve()
    # src/nids_baseline/utils.py -> project root is parents[2]
    project_root = here.parents[2]
    return (project_root / ".." / "shared" / "ids_pipeline" / "src").resolve()


def ensure_ids_pipeline_on_path() -> Path:
    """Insert the shared ids_pipeline ``src/`` onto ``sys.path`` and return it."""
    src = shared_pipeline_src()
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)
    return src
