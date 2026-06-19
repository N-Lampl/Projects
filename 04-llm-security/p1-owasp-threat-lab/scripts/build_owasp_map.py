#!/usr/bin/env python3
"""Emit the OWASP-LLM-Top-10 -> repo-project map (owasp_map.json), a coverage
figure, and metrics.json. stdlib + matplotlib only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from owasp_threat_lab import coverage_summary, coverage_table  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIG = RESULTS / "figures"


def _coverage_figure(rows) -> Path:
    labels = [f"{r['id']} {r['name']}" for r in rows]
    counts = [len(r["projects"]) for r in rows]
    colors = ["#f39c12" if r["partial"] else ("#27ae60" if r["covered"] else "#c0392b") for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, counts, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("# repo projects addressing this risk")
    ax.set_title("OWASP LLM Top 10 (2025): repo coverage\n(green=covered, orange=partial, red=gap)")
    fig.tight_layout()
    out = FIG / "owasp_coverage.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = coverage_table()
    summary = coverage_summary()

    (RESULTS / "owasp_map.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig = _coverage_figure(rows)

    metrics = {
        "project": "p1-owasp-threat-lab",
        "summary": "OWASP LLM Top 10 (2025) mapped to the repo's LLM-security projects.",
        **summary,
        "figures": [str(fig.relative_to(PROJECT))],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"coverage: {summary['fully_covered']}/{summary['total_risks']} fully, "
          f"{summary['partial']} partial ({summary['coverage_pct']}%)")
    for r in rows:
        tag = "partial" if r["partial"] else ("ok" if r["covered"] else "GAP")
        print(f"  {r['id']} {r['name']:34} [{tag}] {len(r['projects'])} project(s)")
    print(f"\nwrote results/owasp_map.json, results/metrics.json, {fig.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
