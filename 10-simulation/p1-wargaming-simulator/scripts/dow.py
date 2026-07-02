#!/usr/bin/env python3
"""Thin wrapper so the CLI runs from a source checkout without installing (used by the Makefile)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dow_sim.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
