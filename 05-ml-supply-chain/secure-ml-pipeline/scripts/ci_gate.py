#!/usr/bin/env python3
"""The CI gate the GitHub Actions workflow calls. Fails closed (non-zero exit).

Two modes:
  --scan-pickles DIR --fail-on-malicious
      walk DIR for *.pkl/*.pt/*.pickle/*.bin, opcode-scan each, exit 1 if any is
      flagged malicious. (safetensors files are skipped: not pickles.)
  --verify ARTIFACT --signature SIG
      verify the artifact against its signature; exit 1 if tampered/unsigned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secure_ml_pipeline import local_verify, scan_pickle_file  # noqa: E402

PICKLE_EXTS = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".ckpt"}


def _scan_pickles(root: Path, fail_on_malicious: bool) -> int:
    bad = 0
    scanned = 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in PICKLE_EXTS or not p.is_file():
            continue
        scanned += 1
        res = scan_pickle_file(p)
        status = res.verdict
        print(f"[scan] {p}: {status} ({len(res.findings)} findings)")
        if res.malicious:
            bad += 1
            for f in res.findings:
                print(f"        - {f.opcode} @ {f.pos}: {f.reason}")
    print(f"[scan] scanned {scanned} pickle-like file(s); {bad} malicious")
    if fail_on_malicious and bad:
        print("GATE FAILED: malicious pickle artifact(s) present.", file=sys.stderr)
        return 1
    return 0


def _verify(artifact: Path, signature: Path) -> int:
    ok = local_verify(artifact, signature)
    print(f"[verify] {artifact}: {'PASS' if ok else 'FAIL (blocked)'}")
    if not ok:
        print("GATE FAILED: artifact signature missing/invalid (fail closed).", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-pickles", type=Path, metavar="DIR")
    ap.add_argument("--fail-on-malicious", action="store_true")
    ap.add_argument("--verify", type=Path, metavar="ARTIFACT")
    ap.add_argument("--signature", type=Path, metavar="SIG")
    args = ap.parse_args()

    rc = 0
    if args.scan_pickles is not None:
        rc |= _scan_pickles(args.scan_pickles, args.fail_on_malicious)
    if args.verify is not None:
        if args.signature is None:
            ap.error("--verify requires --signature")
        rc |= _verify(args.verify, args.signature)
    if args.scan_pickles is None and args.verify is None:
        ap.error("nothing to do: pass --scan-pickles or --verify")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
