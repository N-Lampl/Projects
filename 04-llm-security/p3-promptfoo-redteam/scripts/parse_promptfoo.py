#!/usr/bin/env python3
"""Parse a real `promptfoo redteam eval -o results.json` export into our metrics
shape + chart, so the OPTIONAL node/npx path produces the SAME dashboard
artifacts as the offline harness.

Usage (after `npx promptfoo@latest redteam run -o promptfoo/results.json` in the
promptfoo/ dir; needs node + npx):

    python scripts/parse_promptfoo.py promptfoo/results.json

promptfoo's export schema (v0.x) puts each test under results.results[*] with a
`success` boolean, `gradingResult`, and `vars`/`metadata` carrying the plugin /
pluginId. We map plugin -> OWASP category, then reuse run_redteam's plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"

# promptfoo redteam plugin -> OWASP LLM Top-10 category label.
PLUGIN_TO_OWASP = {
    "harmful:prompt-injection": "LLM01:2025 Prompt Injection",
    "indirect-prompt-injection": "LLM01:2025 Prompt Injection",
    "jailbreak": "LLM01:2025 Prompt Injection",
    "jailbreak:composite": "LLM01:2025 Prompt Injection",
    "prompt-injection": "LLM01:2025 Prompt Injection",
    "system-prompt-override": "LLM07:2025 System Prompt Leakage",
    "rbac": "LLM07:2025 System Prompt Leakage",
    "pii:direct": "LLM02:2025 Sensitive Information Disclosure",
    "harmful:privacy": "LLM02:2025 Sensitive Information Disclosure",
    "excessive-agency": "LLM06:2025 Excessive Agency",
}


def _extract_plugin(test: dict) -> str:
    meta = test.get("metadata") or {}
    for key in ("pluginId", "plugin"):
        if key in meta:
            return str(meta[key])
    vars_ = test.get("vars") or {}
    return str(vars_.get("plugin", "unknown"))


def parse(results_path: Path) -> dict:
    raw = json.loads(results_path.read_text())
    tests = raw.get("results", {}).get("results") or raw.get("results") or []

    by_owasp: dict[str, dict[str, int]] = {}
    probes = []
    n_attacks = 0
    n_success = 0
    for t in tests:
        plugin = _extract_plugin(t)
        owasp = PLUGIN_TO_OWASP.get(plugin, "uncategorized")
        # In promptfoo redteam, success=False means the attack was NOT blocked
        # (the assertion that the model should refuse FAILED) => attack landed.
        passed_assertion = bool(t.get("success", True))
        attack_succeeded = not passed_assertion
        n_attacks += 1
        n_success += int(attack_succeeded)
        bucket = by_owasp.setdefault(owasp, {"total": 0, "succeeded": 0})
        bucket["total"] += 1
        bucket["succeeded"] += int(attack_succeeded)
        probes.append(
            {
                "id": t.get("id") or t.get("testIdx") or plugin,
                "owasp": owasp,
                "plugin": plugin,
                "attack_succeeded": attack_succeeded,
                "reasons": [(t.get("gradingResult") or {}).get("reason", "")],
            }
        )

    by_owasp_out = {
        k: {**v, "success_rate": round(v["succeeded"] / v["total"], 4) if v["total"] else 0.0}
        for k, v in by_owasp.items()
    }
    return {
        "n_attack_probes": n_attacks,
        "attacks_succeeded": n_success,
        "attack_success_rate": round(n_success / n_attacks, 4) if n_attacks else 0.0,
        "by_owasp": by_owasp_out,
        "probes": probes,
    }


def _plot(by_owasp: dict) -> Path:
    cats = sorted(by_owasp)
    rates = [by_owasp[c]["success_rate"] * 100 for c in cats]
    labels = [c.split(":")[0] if ":" in c else c for c in cats]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, rates, color="#8e44ad")
    ax.set_ylabel("attack success rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("promptfoo redteam (owasp:llm): attack success by category", pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "promptfoo_success_by_category.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse promptfoo redteam export -> metrics.json.")
    ap.add_argument("results", type=Path, help="promptfoo results.json export")
    ap.add_argument("--json-out", type=Path, default=METRICS)
    args = ap.parse_args()

    if not args.results.exists():
        sys.exit(
            f"error: {args.results} not found. Run the promptfoo eval first "
            "(needs node/npx): cd promptfoo && npx promptfoo@latest redteam run "
            "-o results.json"
        )

    summary = parse(args.results)
    fig = _plot(summary["by_owasp"])
    metrics = {
        "project": "p3-promptfoo-redteam",
        "summary": (
            f"promptfoo owasp:llm redteam: {summary['attacks_succeeded']}/"
            f"{summary['n_attack_probes']} attacks landed "
            f"({summary['attack_success_rate'] * 100:.0f}%)."
        ),
        "target": "promptfoo provider (see promptfooconfig.yaml)",
        "harness": "promptfoo redteam (owasp:llm preset) -- node/npx path",
        **summary,
        "figures": [str(fig.relative_to(PROJECT))],
    }
    args.json_out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {args.json_out.relative_to(PROJECT)} and {fig.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
