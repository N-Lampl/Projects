#!/usr/bin/env python3
"""Detonate the benign pickle PoC INSIDE the sandbox container only.

This is the one place that actually calls pickle.load() on untrusted bytes.
It must NEVER be run on the host directly — that is what run_in_docker.sh and the
container hardening (--network none --read-only --cap-drop ALL) exist to enforce.

It prints whether the benign payload executed (i.e. the marker file appeared),
demonstrating that pickle.load == arbitrary code execution.
"""

from __future__ import annotations

import pickle  # noqa: S403 - deliberate: this is the demonstration of the risk
import sys
from pathlib import Path


def main() -> int:
    poc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/poc/malicious_model.pkl")
    marker = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/PWNED_DEMO")  # noqa: S108

    print(f"[sandbox] loading {poc_path} via pickle.load() ...")
    if marker.exists():
        marker.unlink()

    with open(poc_path, "rb") as f:
        obj = pickle.load(f)  # noqa: S301 - intentional detonation in sandbox

    print(f"[sandbox] pickle.load returned: {obj!r}")
    if marker.exists():
        print(f"[sandbox] EXPLOIT FIRED: marker file {marker} was written during load:")
        print("    " + marker.read_text().strip())
        print("[sandbox] => pickle.load executed attacker-controlled code. This is why")
        print("[sandbox]    untrusted .pkl/.pt model files are a supply-chain risk.")
        return 0
    print("[sandbox] marker not written - payload did not fire.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
