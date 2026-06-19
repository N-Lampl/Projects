#!/usr/bin/env python3
"""Run BOTH privacy attacks (model inversion + attribute inference) and merge
their results into the single top-level results/metrics.json that the portfolio
dashboard auto-discovers. Run via `make run`.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
METRICS = PROJECT / "results" / "metrics.json"


def _run(script: str) -> None:
    print(f"\n=== {script} ===")
    # reset argv so the sub-scripts see no extra flags
    sys.argv = [script]
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")


def main() -> None:
    _run("run_inversion.py")
    _run("run_attribute.py")

    inv = json.loads((PROJECT / "results" / "metrics_inversion.json").read_text())
    attr = json.loads((PROJECT / "results" / "metrics_attribute.json").read_text())

    merged = {
        "project": "p4-inversion-attribute",
        "summary": (
            "Two ML-privacy attacks from scratch: (1) gradient-ascent model "
            "inversion reconstructs class-representative inputs from a trained "
            "image classifier; (2) MAP attribute inference recovers a sensitive "
            "tabular attribute from a released model."
        ),
        "inversion": {
            "target_test_accuracy": inv["target_test_accuracy"],
            "mean_recovery_confidence": inv["mean_recovery_confidence"],
            "top1_match_rate": inv["top1_match_rate"],
            "mean_own_class_correlation": inv["mean_own_class_correlation"],
        },
        "attribute_inference": {
            "max_attack_accuracy": attr["max_attack_accuracy"],
            "max_lift_over_baseline": attr["max_lift_over_baseline"],
            "target_model": attr["target_model"],
        },
        "figures": inv["figures"] + attr["figures"],
    }
    METRICS.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nwrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
