#!/usr/bin/env python3
"""Run the offline garak-style scan, parse the report, write figures + metrics.json.

Default (offline): built-in probes vs the p4 VulnerableRAG mock. Produces
``results/report.jsonl`` (garak-compatible), ``results/metrics.json`` and two
charts. Pass ``--report PATH`` to instead PARSE an existing (e.g. real garak)
report.jsonl with the same code path. Run via ``make scan``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garak_scan import (  # noqa: E402
    get_probes,
    get_target,
    run_scan,
    set_seed,
    summarize,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
REPORT = PROJECT / "results" / "report.jsonl"
ALLOWLIST = PROJECT / "configs" / "probe_allowlist.yaml"


def _load_allowlist(path: Path) -> list[str]:
    """Read probe names from the YAML allowlist (no PyYAML dependency).

    Format is a simple ``probes:`` list of ``- name`` entries; lines after a
    ``#`` are comments. We parse the minimal subset we emit ourselves.
    """
    if not path.exists():
        return []
    names: list[str] = []
    in_list = False
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.strip() == "probes:":
            in_list = True
            continue
        if in_list and line.lstrip().startswith("- "):
            names.append(line.lstrip()[2:].strip().strip('"').strip("'"))
        elif in_list and not line.startswith(" "):
            in_list = False
    return names


def _bar_asr(summary: dict, out: Path) -> None:
    """The money plot: per-probe attack-success rate with bootstrap-CI error bars."""
    probes = summary["by_probe"]
    names = list(probes.keys())
    asr = [probes[n]["asr"] for n in names]
    lo = [probes[n]["asr"] - probes[n]["ci_low"] for n in names]
    hi = [probes[n]["ci_high"] - probes[n]["asr"] for n in names]
    yerr = [lo, hi]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if a >= 0.5 else "#e67e22" if a > 0 else "#27ae60" for a in asr]
    bars = ax.bar(range(len(names)), asr, color=colors, yerr=yerr, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("attack-success rate (ASR)")
    ax.set_ylim(0, 1.05)
    ax.set_title("garak-style scan: per-probe ASR (95% bootstrap CI)")
    for b, a in zip(bars, asr):
        ax.text(b.get_x() + b.get_width() / 2, min(a + 0.03, 1.0), f"{a:.0%}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _bar_category(summary: dict, out: Path) -> None:
    """ASR aggregated by garak probe category."""
    cats = summary["by_category"]
    names = list(cats.keys())
    asr = [cats[n]["asr"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, asr, color="#2c3e50")
    ax.set_ylabel("attack-success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("ASR by probe category")
    for i, a in enumerate(asr):
        ax.text(i, min(a + 0.03, 1.0), f"{a:.0%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=str, default=None,
                    help="parse an EXISTING report.jsonl (e.g. real garak) instead of scanning")
    ap.add_argument("--target", default="rag", choices=["rag", "standalone"],
                    help="offline target (default: p4 RAG, falling back to standalone mock)")
    ap.add_argument("--generations", type=int, default=1,
                    help="repeats per prompt (garak --generations); mock is deterministic")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for CIs")
    args = ap.parse_args()

    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    target_name = "external"

    if args.report:
        report_path = Path(args.report)
        if not report_path.exists():
            raise SystemExit(f"report not found: {report_path}")
        print(f"[parse] using existing report: {report_path}")
    else:
        allowlist = _load_allowlist(ALLOWLIST)
        probes = get_probes(allowlist)
        target = get_target(args.target)
        target_name = target.name
        print(f"[scan] target={target.name} probes={len(probes)} "
              f"generations={args.generations}")
        report_path = run_scan(target, probes, REPORT, generations=args.generations)
        print(f"[scan] wrote garak-style report: {report_path}")

    summary = summarize(report_path, n_boot=args.n_boot)

    fig1 = FIG_DIR / "asr_by_probe.png"
    fig2 = FIG_DIR / "asr_by_category.png"
    _bar_asr(summary, fig1)
    _bar_category(summary, fig2)

    metrics = {
        "project": "p2-garak-scan",
        "summary": (
            f"Offline garak-style red-team scan: {summary['n_probes']} probes / "
            f"{summary['total_attempts']} attempts vs the p4 RAG mock; overall ASR "
            f"{summary['overall_asr']:.0%} "
            f"(95% CI {summary['overall_ci_low']:.0%}-{summary['overall_ci_high']:.0%}). "
            f"Same parser ingests a real garak report.jsonl."
        ),
        "mode": "parse" if args.report else "scan",
        "target": target_name,
        "hit_threshold": 0.5,
        "n_probes": summary["n_probes"],
        "total_attempts": summary["total_attempts"],
        "total_hits": summary["total_hits"],
        "overall_asr": summary["overall_asr"],
        "overall_ci_low": summary["overall_ci_low"],
        "overall_ci_high": summary["overall_ci_high"],
        "by_probe": summary["by_probe"],
        "by_category": summary["by_category"],
        "figures": [
            "results/figures/asr_by_probe.png",
            "results/figures/asr_by_category.png",
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[done] overall ASR={summary['overall_asr']:.0%} "
          f"[{summary['overall_ci_low']:.0%}, {summary['overall_ci_high']:.0%}]")
    print(f"[done] metrics -> {METRICS}")
    print(f"[done] figures -> {fig1.name}, {fig2.name}")


if __name__ == "__main__":
    main()
