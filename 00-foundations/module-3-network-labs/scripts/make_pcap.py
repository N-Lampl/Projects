#!/usr/bin/env python3
"""Generate the synthetic .pcap trace. Run via `make pcap` (also called by `make run`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netlabs import generate_pcap, set_seed  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "data" / "synthetic.pcap"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output pcap path")
    ap.add_argument("--engine", choices=["python", "scapy"], default="python",
                    help="python = pure-stdlib (offline default); scapy = optional enhanced")
    args = ap.parse_args()

    set_seed()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = generate_pcap(str(args.out), engine=args.engine)
    print(f"wrote {n} packets -> {args.out.relative_to(PROJECT)} (engine={args.engine})")


if __name__ == "__main__":
    main()
