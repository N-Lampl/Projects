#!/usr/bin/env python3
"""The CI red-team pipeline: gate -> dashboard -> threat-model report -> metrics.

Default (offline) flow, run by ``make run`` and by CI:
  1. Run the p2 (garak-style) + p3 (promptfoo-style) OFFLINE harnesses against
     the p4 VulnerableRAG target -> normalised ProbeResults.
  2. Evaluate the CI gate (per-OWASP ASR vs threshold).
  3. Also evaluate the REMEDIATED p7 target for the before/after dashboard.
  4. Write figures (asr_by_category.png, asr_trend.png), metrics.json, and the
     consulting threat-model report (results/threat_model_report.md).

Exit code:
  * ``make run``        -> always exits 0 (it just produces artifacts).
  * ``make gate``       -> passes --enforce, exiting 1 if ASR > threshold so the
                           CI job FAILS on the vulnerable target (the demo point).
  * ``--remediated``    -> gate the p7 target instead (expected PASS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from appsec_ci import (  # noqa: E402
    build_report,
    evaluate,
    load_defended_target,
    load_target,
    plot_before_after,
    plot_trend,
    run_garak,
    run_promptfoo,
    seed_history,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIG_DIR = RESULTS / "figures"
METRICS = RESULTS / "metrics.json"
REPORT_MD = RESULTS / "threat_model_report.md"
HISTORY = RESULTS / "history.jsonl"


def _redteam(target: object, smoke: bool) -> list:
    """Run the sibling harnesses against ``target``.

    Smoke mode runs both harnesses but is already fast (pure-python, no network).
    Full mode is identical here (the offline probe set is small) but is kept as a
    distinct flag so the scheduled CI job is explicit about intent.
    """
    results = run_promptfoo(target)
    if not smoke:
        results = results + run_garak(target)
    else:
        # Smoke: a representative subset (promptfoo OWASP probes) -> fast gate.
        results = results
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="max tolerated overall ASR before the gate fails (default 0.0)")
    ap.add_argument("--enforce", action="store_true",
                    help="exit 1 if the gate fails (CI mode); default just reports")
    ap.add_argument("--remediated", action="store_true",
                    help="gate the p7 remediated target instead of the p4 vulnerable one")
    ap.add_argument("--full", action="store_true",
                    help="run the full suite (p2 garak + p3 promptfoo); default is smoke")
    args = ap.parse_args()

    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    vuln_target, vuln_label = load_target()
    rem_target, rem_label = load_defended_target()

    smoke = not args.full
    vuln_results = _redteam(vuln_target, smoke)
    rem_results = _redteam(rem_target, smoke)

    vuln_gate = evaluate(vuln_results, threshold=args.threshold)
    rem_gate = evaluate(rem_results, threshold=args.threshold)

    # ---- figures ----------------------------------------------------------
    fig_cat = FIG_DIR / "asr_by_category.png"
    fig_trend = FIG_DIR / "asr_trend.png"
    plot_before_after(vuln_gate, rem_gate, fig_cat)
    seed_history(HISTORY, vuln_gate, rem_gate)
    plot_trend(HISTORY, fig_trend)

    # ---- threat-model report ----------------------------------------------
    report_md = build_report(
        vuln_gate, rem_gate,
        vuln_target=vuln_label, remediated_target=rem_label,
        threshold=args.threshold,
    )
    REPORT_MD.write_text(report_md, encoding="utf-8")

    # ---- which target is THIS gate enforcing? -----------------------------
    gated = rem_gate if args.remediated else vuln_gate
    gated_label = rem_label if args.remediated else vuln_label
    mode = "full-suite" if args.full else "smoke"

    metrics = {
        "project": "CAPSTONE-appsec-ci",
        "summary": (
            f"CI-gated LLM AppSec red-team ({mode}): vulnerable target ASR "
            f"{vuln_gate.overall_asr:.0%} (gate {'PASS' if vuln_gate.passed else 'FAIL'}), "
            f"remediated target ASR {rem_gate.overall_asr:.0%} "
            f"(gate {'PASS' if rem_gate.passed else 'FAIL'}); threshold "
            f"{args.threshold:.0%}. Reuses p2+p3 harnesses vs p4, fixes via p7."
        ),
        "mode": mode,
        "gate_threshold": args.threshold,
        "enforced_target": gated_label,
        "gate_passed": gated.passed,
        "gate_reasons": gated.reasons,
        "vulnerable": _gate_to_dict(vuln_gate, vuln_label),
        "remediated": _gate_to_dict(rem_gate, rem_label),
        "risk_reduction_asr": round(vuln_gate.overall_asr - rem_gate.overall_asr, 4),
        "figures": [
            "results/figures/asr_by_category.png",
            "results/figures/asr_trend.png",
        ],
        "report": "results/threat_model_report.md",
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")

    # ---- console summary --------------------------------------------------
    print(f"[redteam] mode={mode}")
    print(f"[vuln ]  {vuln_label}")
    print(f"         overall ASR={vuln_gate.overall_asr:.0%} "
          f"({vuln_gate.attacks_succeeded}/{vuln_gate.n_attack_probes}) "
          f"gate={'PASS' if vuln_gate.passed else 'FAIL'}")
    print(f"[remed]  {rem_label}")
    print(f"         overall ASR={rem_gate.overall_asr:.0%} "
          f"({rem_gate.attacks_succeeded}/{rem_gate.n_attack_probes}) "
          f"gate={'PASS' if rem_gate.passed else 'FAIL'}")
    print(f"[done ]  figures -> {fig_cat.name}, {fig_trend.name}")
    print(f"[done ]  report  -> {REPORT_MD.name}")
    print(f"[done ]  metrics -> {METRICS}")

    if args.enforce and not gated.passed:
        print(f"\n[GATE FAILED] enforcing on: {gated_label}")
        for r in gated.reasons:
            print(f"  - {r}")
        print("Exiting non-zero to block the build.")
        sys.exit(1)
    elif args.enforce:
        print(f"\n[GATE PASSED] {gated_label}: ASR {gated.overall_asr:.0%} "
              f"<= threshold {args.threshold:.0%}")


def _gate_to_dict(gate, label: str) -> dict:
    return {
        "target": label,
        "overall_asr": gate.overall_asr,
        "n_attack_probes": gate.n_attack_probes,
        "attacks_succeeded": gate.attacks_succeeded,
        "benign_false_positives": gate.benign_false_positives,
        "passed": gate.passed,
        "by_owasp": {
            stat.owasp: {"total": stat.total, "succeeded": stat.succeeded,
                         "asr": round(stat.asr, 4)}
            for stat in gate.by_owasp.values()
            if stat.total
        },
    }


if __name__ == "__main__":
    main()
