#!/usr/bin/env python3
"""Run the OWASP Top 10 probes and write results/findings.json + metrics.json + a figure.

DEFAULT path is OFFLINE (deterministic MockJuiceShop) so it runs with zero infrastructure and
proves every exploit's logic. Add --live to hit a real local Juice Shop container instead:

    make lab                 # offline mock (no docker)
    make lab ARGS=--live     # against http://localhost:3000 (docker compose up first)

Authorization: --live refuses non-local hosts unless JUICE_SHOP_ALLOW_REMOTE=1. See ETHICS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from juiceshop_lab import make_client, run_all, set_seed  # noqa: E402
from juiceshop_lab.utils import DEFAULT_TARGET, require_local_target  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIG_DIR = RESULTS / "figures"
FINDINGS = RESULTS / "findings.json"
METRICS = RESULTS / "metrics.json"


def _plot(findings: list[dict], mode: str) -> Path:
    sev_order = ["critical", "high", "medium", "low"]
    colors = {"critical": "#922b21", "high": "#c0392b", "medium": "#e67e22", "low": "#f1c40f"}
    counts = Counter(f["severity"] for f in findings if f["success"])
    vals = [counts.get(s, 0) for s in sev_order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sev_order, vals, color=[colors[s] for s in sev_order])
    ax.set_ylabel("confirmed findings")
    ax.set_title(f"OWASP Top 10 lab vs Juice Shop ({mode}): findings by severity", pad=12)
    ax.set_ylim(0, max(vals + [1]) + 1)
    for i, v in enumerate(vals):
        ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 6), ha="center")
    fig.tight_layout()
    out = FIG_DIR / "findings_by_severity.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="hit a real local Juice Shop container")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="base URL when --live (default localhost:3000)")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.live:
        require_local_target(args.target)  # authorization guardrail
        mode = "live"
        print(f"[live] targeting {args.target} (authorized local container only)")
    else:
        mode = "offline-mock"
        print("[offline] using deterministic MockJuiceShop (no docker needed)")

    client = make_client(args.target, offline=not args.live)
    findings = run_all(client)

    for f in findings:
        flag = "VULNERABLE" if f["success"] else "not-confirmed"
        print(f"  [{f['severity']:>8}] {f['owasp']:<45} -> {flag}")

    categories = sorted({f["owasp"] for f in findings})
    confirmed = [f for f in findings if f["success"]]

    findings_doc = {
        "project": "module-4-web-appsec",
        "target": "OWASP Juice Shop (local container)",
        "mode": mode,
        "owasp_categories_covered": categories,
        "findings": findings,
    }
    FINDINGS.write_text(json.dumps(findings_doc, indent=2) + "\n")

    fig = _plot(findings, mode)

    metrics = {
        "project": "module-4-web-appsec",
        "summary": (
            f"{len(confirmed)}/{len(findings)} OWASP Top 10 probes confirmed vulnerable against "
            f"Juice Shop ({mode}); {len(categories)} categories covered."
        ),
        "mode": mode,
        "categories_covered": categories,
        "n_categories": len(categories),
        "n_probes": len(findings),
        "n_confirmed": len(confirmed),
        "severity_breakdown": dict(Counter(f["severity"] for f in confirmed)),
        "findings_by_id": {f["id"]: f["success"] for f in findings},
        "figures": [str(fig.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\nwrote {FINDINGS.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")
    print(f"wrote {fig.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
