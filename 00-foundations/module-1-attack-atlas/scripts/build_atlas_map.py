#!/usr/bin/env python3
"""Build results/atlas_map.json (track/project -> ATLAS technique IDs),
results/metrics.json (counts), and the ASCII coverage figure. Run via
`make run`. Stdlib only — no torch, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attack_atlas import (  # noqa: E402
    build_atlas_map,
    build_metrics,
    render_coverage_chart,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIG_DIR = RESULTS / "figures"
ATLAS_MAP = RESULTS / "atlas_map.json"
METRICS = RESULTS / "metrics.json"
COVERAGE_FIG = FIG_DIR / "coverage_by_tactic.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--print", action="store_true", dest="do_print",
        help="also print the atlas map and metrics to stdout",
    )
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    atlas_map = build_atlas_map()
    metrics = build_metrics(atlas_map)
    chart = render_coverage_chart(metrics)

    ATLAS_MAP.write_text(json.dumps(atlas_map, indent=2) + "\n")
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    COVERAGE_FIG.write_text(chart)

    print(f"wrote {ATLAS_MAP.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")
    print(f"wrote {COVERAGE_FIG.relative_to(PROJECT)}")
    print()
    print(metrics["summary"])
    print()
    print(chart, end="")

    if args.do_print:
        print()
        print(json.dumps(atlas_map, indent=2))


if __name__ == "__main__":
    main()
