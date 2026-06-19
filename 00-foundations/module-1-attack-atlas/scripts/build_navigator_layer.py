#!/usr/bin/env python3
"""Regenerate navigator/portfolio_layer.json — a valid MITRE ATT&CK Navigator
layer highlighting the ATT&CK techniques this portfolio exercises (bridged from
ATLAS). Stdlib only. Run via `make navigator`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attack_atlas import build_navigator_layer, set_seed  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
LAYER = PROJECT / "navigator" / "portfolio_layer.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=LAYER)
    args = ap.parse_args()

    set_seed()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    layer = build_navigator_layer()
    args.out.write_text(json.dumps(layer, indent=2) + "\n")
    print(f"wrote {args.out.relative_to(PROJECT)} ({len(layer['techniques'])} techniques)")


if __name__ == "__main__":
    main()
