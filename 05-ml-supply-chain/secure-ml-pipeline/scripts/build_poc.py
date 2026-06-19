#!/usr/bin/env python3
"""Build the BENIGN pickle-deserialization PoC at runtime, then statically scan it.

This NEVER unpickles the file. It writes a benign-but-malicious-shaped pickle to a
git-ignored / temp path and runs the offline opcode scanner so you can SEE that the
detector flags REDUCE/GLOBAL before anything is loaded. Actually executing the
payload only happens inside Docker via scripts/run_in_docker.sh.

    python scripts/build_poc.py                 # default: data/poc/malicious_model.pkl
    python scripts/build_poc.py --out /tmp/x.pkl --marker /tmp/PWNED_DEMO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secure_ml_pipeline import scan_pickle_file, write_poc  # noqa: E402
from secure_ml_pipeline.poc import DEFAULT_MARKER  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "data" / "poc" / "malicious_model.pkl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the PoC pickle")
    ap.add_argument("--marker", default=DEFAULT_MARKER, help="benign marker file the payload writes")
    args = ap.parse_args()

    print("=" * 70)
    print("THREAT DEMO  ·  building a BENIGN pickle-deserialization PoC")
    print("=" * 70)
    print("  The payload only writes a marker file. No shell, no network, no")
    print("  destructive ops. This artifact is git-ignored and never committed.\n")

    out = write_poc(args.out, marker_path=args.marker)
    print(f"wrote PoC pickle    -> {out}")
    print(f"benign payload will -> write {args.marker} (only when LOADED, e.g. in Docker)\n")

    print("running the offline opcode scanner (read-only, does NOT unpickle)...")
    res = scan_pickle_file(out)
    print(f"  verdict: {res.verdict}")
    print(f"  globals seen: {res.globals_seen or '(none resolved statically)'}")
    for f in res.findings:
        print(f"    [{f.opcode:<12}] @byte {f.pos:<4} {f.reason}")

    print("\nNEXT: detonate it SAFELY in a sandbox:")
    print("    bash scripts/run_in_docker.sh", out)
    print("Then run the secure pipeline that ships safetensors instead:")
    print("    python scripts/secure_pipeline.py")


if __name__ == "__main__":
    main()
