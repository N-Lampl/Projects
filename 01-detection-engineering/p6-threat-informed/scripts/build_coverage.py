#!/usr/bin/env python3
"""Validate the Sigma rules, map them to ATT&CK, and produce an ATT&CK-coverage
heatmap + metrics.json. Run via ``make detect``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threat_informed import (  # noqa: E402
    build_coverage,
    coverage_grid,
    load_rules,
    set_seed,
    summarize,
    technique_name,
)

PROJECT = Path(__file__).resolve().parents[1]
RULES_DIR = PROJECT / "rules"
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_heatmap(tactics, techniques, matrix) -> Path:
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(1, arr.max()))

    ax.set_xticks(range(len(tactics)))
    ax.set_xticklabels(tactics, rotation=45, ha="right", fontsize=8)
    ylabels = [f"{t}  {technique_name(t)}" for t in techniques]
    ax.set_yticks(range(len(techniques)))
    ax.set_yticklabels(ylabels, fontsize=7)

    # annotate covered cells with rule counts
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c] > 0:
                ax.text(
                    c,
                    r,
                    int(arr[r, c]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_title("ATT&CK detection coverage (cell = #Sigma rules)", pad=12)
    ax.set_xlabel("Tactic")
    ax.set_ylabel("Technique")
    fig.colorbar(im, ax=ax, label="# rules", fraction=0.025, pad=0.02)
    fig.tight_layout()
    out = FIG_DIR / "attack_coverage_heatmap.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_tactic_bar(summary) -> Path:
    from threat_informed import TACTICS, tactics_for_technique

    covered_counts = {t: 0 for t in TACTICS}
    for tid in summary["techniques_covered"]:
        for t in tactics_for_technique(tid):
            covered_counts[t] += 1

    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = [covered_counts[t] for t in TACTICS]
    colors = ["#27ae60" if v > 0 else "#bdc3c7" for v in vals]
    ax.bar(range(len(TACTICS)), vals, color=colors)
    ax.set_xticks(range(len(TACTICS)))
    ax.set_xticklabels(TACTICS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("# techniques covered")
    ax.set_title("Detection coverage per ATT&CK tactic", pad=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tactic_coverage_bar.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules-dir", default=str(RULES_DIR), help="directory of Sigma .yml rules")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rules = load_rules(args.rules_dir)
    print(f"loaded {len(rules)} rule(s) from {Path(args.rules_dir).name}/")
    for r in rules:
        status = "OK " if r.valid else "BAD"
        techs = ",".join(r.techniques) or "-"
        print(f"  [{status}] {r.path.name:<42} {techs}")
        for err in r.errors:
            print(f"        ! {err}")

    coverage = build_coverage(rules)
    tactics, techniques, matrix = coverage_grid(coverage)
    summary = summarize(rules, coverage)

    heatmap = _plot_heatmap(tactics, techniques, matrix)
    bar = _plot_tactic_bar(summary)

    metrics = {
        "project": "p6-threat-informed",
        "summary": (
            f"{summary['n_valid_rules']}/{summary['n_rules']} Sigma rules valid, covering "
            f"{summary['n_techniques_covered']}/{summary['n_techniques_total']} ATT&CK techniques "
            f"({summary['technique_coverage_pct']}%) across "
            f"{summary['n_tactics_covered']}/{summary['n_tactics_total']} tactics"
        ),
        "seed": 42,
        **summary,
        "figures": [
            str(heatmap.relative_to(PROJECT)),
            str(bar.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\n{metrics['summary']}")
    print(f"wrote {heatmap.relative_to(PROJECT)}")
    print(f"wrote {bar.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")

    if summary["n_invalid_rules"]:
        print(f"\nWARNING: {summary['n_invalid_rules']} invalid rule(s) - see above.")


if __name__ == "__main__":
    main()
