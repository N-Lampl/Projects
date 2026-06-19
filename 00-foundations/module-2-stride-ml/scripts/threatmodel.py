#!/usr/bin/env python3
"""Generate the STRIDE threat model of the ML inference service.

Default (offline) path: stdlib only — writes docs/threat-model.md, a per-category
bar chart, and results/metrics.json. The optional `--pytm` flag uses pytm if it is
installed (and gracefully falls back to the stdlib path otherwise).

Run via `make detect`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stride_ml import (  # noqa: E402
    STRIDE,
    analyze,
    build_ml_inference_service,
    render_markdown,
    set_seed,
    summarize,
    try_pytm_model,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
DOC = PROJECT / "docs" / "threat-model.md"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_counts(counts: dict[str, int]) -> Path | None:
    """Bar chart of threats per STRIDE category. Returns None if matplotlib is absent."""
    try:
        import matplotlib  # noqa: WPS433

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        print("matplotlib not installed - skipping figure (markdown + metrics still written)")
        return None

    cats = list(counts)
    vals = [counts[c] for c in cats]
    short = [c.split()[0] if c != "Information Disclosure" else "Info.Disc." for c in cats]
    short = ["Elev.Priv." if c == "Elevation" else c for c in short]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(short, vals, color="#2c6e91")
    ax.set_ylabel("number of threats")
    ax.set_title("STRIDE threat counts — ML Inference Service", pad=12)
    for i, v in enumerate(vals):
        ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) + 2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "stride_counts.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pytm", action="store_true",
                    help="use pytm if installed (falls back to stdlib renderer otherwise)")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    system = build_ml_inference_service()
    threats = analyze(system)
    counts = summarize(threats)

    generator = "stride_ml (stdlib)"
    pytm_used = False
    if args.pytm:
        tm = try_pytm_model(system)
        if tm is not None:
            pytm_used = True
            generator = "stride_ml + pytm"
            print("pytm available: built equivalent pytm TM object.")
        else:
            print("pytm not installed: using stdlib renderer (default path).")

    DOC.write_text(render_markdown(system, threats, generator))
    fig = _plot_counts(counts)

    metrics = {
        "project": "module-2-stride-ml",
        "summary": (
            f"STRIDE threat model of an ML inference service: {len(threats)} threats "
            f"across {len(STRIDE)} categories ({sum(1 for t in threats if t.severity == 'High')} High)."
        ),
        "seed": 42,
        "system": system.name,
        "n_elements": len(system.elements),
        "n_data_flows": len(system.flows),
        "n_trust_boundaries": len(system.boundaries),
        "total_threats": len(threats),
        "threats_by_category": counts,
        "threats_by_severity": {
            sev: sum(1 for t in threats if t.severity == sev)
            for sev in ("High", "Medium", "Low")
        },
        "pytm_used": pytm_used,
        "doc": str(DOC.relative_to(PROJECT)),
        "figures": [str(fig.relative_to(PROJECT))] if fig else [],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\n{system.name}: {len(threats)} threats")
    for cat in STRIDE:
        print(f"  {cat:<24} {counts[cat]}")
    print(f"\nwrote {DOC.relative_to(PROJECT)}")
    if fig:
        print(f"wrote {fig.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
