"""Run probes against a target and emit a garak-compatible ``report.jsonl``.

garak writes one JSON object per *attempt* (probe x prompt x generation). We emit
the same essential fields so the SAME parser (``report.py``) consumes both a real
garak report and our offline report:

    {"entry_type": "attempt", "probe": "...", "prompt": "...",
     "outputs": ["..."], "detector_results": {"<detector>": [0.0|1.0, ...]}}

A score >= 0.5 in ``detector_results`` is a "hit" (attack succeeded), matching
garak's convention.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .probes import Probe
from .target import Target


def run_scan(
    target: Target,
    probes: Iterable[Probe],
    out_path: str | Path,
    generations: int = 1,
) -> Path:
    """Attack ``target`` with each probe prompt ``generations`` times.

    Writes a garak-style JSONL to ``out_path`` and returns the path. Each line is
    one attempt with the target's outputs and per-detector scores (0.0/1.0).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        # A run-header line, like garak's "init"/"start_run" record.
        fh.write(
            json.dumps(
                {
                    "entry_type": "start_run",
                    "garak_version": "offline-builtin-1.0",
                    "target": getattr(target, "name", "unknown"),
                    "generations": generations,
                }
            )
            + "\n"
        )
        for probe in probes:
            for prompt in probe.prompts:
                outputs = [target(prompt) for _ in range(generations)]
                scores = [1.0 if probe.detector(o) else 0.0 for o in outputs]
                attempt = {
                    "entry_type": "attempt",
                    "probe": probe.name,
                    "category": probe.category,
                    "goal": probe.goal,
                    "prompt": prompt,
                    "outputs": outputs,
                    "detector_results": {probe.name: scores},
                }
                fh.write(json.dumps(attempt) + "\n")
    return out_path


__all__ = ["run_scan"]
