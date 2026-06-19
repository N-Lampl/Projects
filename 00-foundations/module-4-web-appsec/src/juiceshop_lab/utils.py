"""Reproducibility + small shared helpers for the web-appsec lab.

The portfolio convention is a `set_seed(42)` and a CPU-only `get_device()`. This module is
mostly HTTP/exploit code, but we keep the same surface so tooling/tests stay uniform across
the repo, and so the offline self-check is deterministic.
"""

from __future__ import annotations

import os
import random

SEED = 42

# Default base URL of the LOCAL Juice Shop container. Override with JUICE_SHOP_URL.
DEFAULT_TARGET = os.environ.get("JUICE_SHOP_URL", "http://localhost:3000")


def set_seed(seed: int = SEED) -> None:
    """Seed Python (and NumPy/torch if present) so any sampling is deterministic.

    torch/numpy are optional here — the lab's default path is pure-stdlib HTTP, so we import
    them lazily and ignore absence.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:  # numpy is optional for this networking project
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy simply not installed
        pass
    try:  # torch is optional for this networking project
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - torch simply not installed
        pass


def get_device() -> str:
    """CPU-only by design (whole portfolio targets a no-GPU laptop). Returns 'cpu'."""
    return "cpu"


def require_local_target(url: str) -> None:
    """Authorization guardrail: refuse to point exploit traffic at non-local hosts.

    Every script in this lab calls this before sending a single packet. The lab is for the
    self-hosted Juice Shop container ONLY (see ../../ETHICS.md). Set
    JUICE_SHOP_ALLOW_REMOTE=1 *only* if you own/are authorized to test the remote target.
    """
    if os.environ.get("JUICE_SHOP_ALLOW_REMOTE") == "1":
        return
    allowed = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    if host not in allowed:
        raise SystemExit(
            f"refusing to attack non-local host {host!r}. This lab targets the local Juice "
            "Shop container only (see ../../ETHICS.md). Export JUICE_SHOP_ALLOW_REMOTE=1 "
            "only if you are authorized to test this host."
        )
