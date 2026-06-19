"""Reproducibility + config helpers shared across the attack project."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch so the attack sweep is deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """CPU-only by design (the whole portfolio targets a no-GPU laptop)."""
    return torch.device("cpu")


def load_dotenv() -> None:
    """Minimal .env loader (no python-dotenv dependency). Project-local only."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_int_env(name: str, default: int) -> int:
    """Read a capped integer knob from the environment (after loading .env)."""
    load_dotenv()
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
