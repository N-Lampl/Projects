#!/usr/bin/env python3
"""Run the OFFLINE pure-python red-team harness against the p4 RAG target.

Produces:
  results/figures/owasp_success_by_category.png  -- attack-success per OWASP cat
  results/figures/probe_outcomes.png             -- per-probe pass/fail strip
  results/metrics.json                           -- dashboard-discoverable shape

Run via `make redteam`. No Node, no network, no API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from promptfoo_redteam import (  # noqa: E402
    load_target,
    run_probes,
    set_seed,
    summarize,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_by_category(summary: dict) -> Path:
    by = summary["by_owasp"]
    cats = sorted(by)
    rates = [by[c]["success_rate"] * 100 for c in cats]
    labels = [c.split(":")[0] if ":" in c else c for c in cats]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, rates, color="#c0392b")
    ax.set_ylabel("attack success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("OWASP-LLM red team vs vulnerable RAG: attack success by category", pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    for b, c in zip(bars, cats):
        ax.annotate(
            f"{by[c]['succeeded']}/{by[c]['total']}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    out = FIG_DIR / "owasp_success_by_category.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_probe_outcomes(runs) -> Path:
    ids = [r.probe.id for r in runs]
    succeeded = [r.grade.attack_succeeded for r in runs]
    is_benign = [r.probe.plugin == "benign" for r in runs]

    colors = []
    for s, b in zip(succeeded, is_benign):
        if b:
            colors.append("#27ae60" if not s else "#e67e22")  # green good / orange FP
        else:
            colors.append("#c0392b" if s else "#7f8c8d")  # red = attack landed

    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = range(len(ids))
    ax.barh(list(y), [1] * len(ids), color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Per-probe outcome (red = attack succeeded / vuln hit)", pad=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#c0392b"),
        plt.Rectangle((0, 0), 1, 1, color="#7f8c8d"),
        plt.Rectangle((0, 0), 1, 1, color="#27ae60"),
    ]
    ax.legend(handles, ["attack landed", "defended", "benign ok"], loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "probe_outcomes.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline OWASP-LLM red-team harness.")
    ap.add_argument("--json-out", type=Path, default=METRICS, help="metrics.json path")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    target, source = load_target()
    print(f"target: {source}")

    runs = run_probes(target)
    summary = summarize(runs)

    print(
        f"ran {summary['n_attack_probes']} attack probes -> "
        f"{summary['attacks_succeeded']} landed "
        f"({summary['attack_success_rate'] * 100:.0f}% success); "
        f"benign false-positives: {summary['benign_false_positives']}"
    )
    for r in runs:
        flag = "HIT " if r.grade.attack_succeeded else "ok  "
        if r.probe.plugin == "benign" and not r.grade.attack_succeeded:
            flag = "ctrl"
        print(f"  [{flag}] {r.probe.id:<16} {r.probe.plugin:<24} {r.grade.reasons}")

    cat_fig = _plot_by_category(summary)
    probe_fig = _plot_probe_outcomes(runs)

    metrics = {
        "project": "p3-promptfoo-redteam",
        "summary": (
            "OWASP-LLM red team (offline pure-python harness, promptfoo-style) vs the "
            f"p4 vulnerable RAG: {summary['attacks_succeeded']}/{summary['n_attack_probes']} "
            f"attacks landed ({summary['attack_success_rate'] * 100:.0f}%), "
            f"{summary['benign_false_positives']} benign false-positives."
        ),
        "target": source,
        "harness": "pure-python (offline default); promptfoo owasp:llm preset = optional path",
        "n_probes": summary["n_probes"],
        "n_attack_probes": summary["n_attack_probes"],
        "attacks_succeeded": summary["attacks_succeeded"],
        "attack_success_rate": summary["attack_success_rate"],
        "benign_false_positives": summary["benign_false_positives"],
        "by_owasp": summary["by_owasp"],
        "probes": [
            {
                "id": r.probe.id,
                "owasp": r.probe.owasp,
                "plugin": r.probe.plugin,
                "attack_succeeded": r.grade.attack_succeeded,
                "reasons": r.grade.reasons,
            }
            for r in runs
        ],
        "figures": [
            str(cat_fig.relative_to(PROJECT)),
            str(probe_fig.relative_to(PROJECT)),
        ],
    }
    args.json_out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {cat_fig.relative_to(PROJECT)}")
    print(f"wrote {probe_fig.relative_to(PROJECT)}")
    print(f"wrote {args.json_out.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
