#!/usr/bin/env python3
"""promptfoo `exec` provider bridge: stdin prompt -> local mock RAG -> stdout.

promptfoo's `exec:` provider passes the rendered prompt as the last CLI arg (or
on stdin for some versions); we accept both. It calls the SAME offline target
the python harness uses (p4 RAG, else the built-in mock), so the optional
promptfoo path attacks the exact same app with NO real LLM/network.

This file is only invoked when you run promptfoo (node/npx). The default
`make redteam` path does not need it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project's src importable regardless of promptfoo's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptfoo_redteam import load_target  # noqa: E402


def _read_prompt() -> str:
    if len(sys.argv) > 1 and sys.argv[-1].strip():
        return sys.argv[-1]
    data = sys.stdin.read()
    return data.strip()


def main() -> None:
    prompt = _read_prompt()
    target, _ = load_target()
    sys.stdout.write(target.query(prompt))


if __name__ == "__main__":
    main()
