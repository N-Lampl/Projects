"""Render the 'IDS Robustness Report Card' (markdown + a comparison figure)."""

from __future__ import annotations

from pathlib import Path


def _grade(asr_after: float) -> str:
    """Letter grade from post-hardening attack-success-rate (lower is better)."""
    if asr_after < 0.10:
        return "A"
    if asr_after < 0.25:
        return "B"
    if asr_after < 0.45:
        return "C"
    if asr_after < 0.65:
        return "D"
    return "F"


def render_report_card(metrics: dict) -> str:
    """Build the markdown report card from the metrics dict."""
    s = metrics["summary"]
    clean = metrics["clean"]
    before = metrics["attack_before"]
    after = metrics["attack_after"]
    grade = _grade(s["asr_after"])
    drop = s["asr_before"] - s["asr_after"]

    lines = [
        "# IDS Robustness Report Card",
        "",
        f"**Target:** shared `ids_pipeline` RandomForest on {metrics['source']} flows  ",
        f"**Attack:** constrained {metrics['attack_engine'].upper()} "
        f"(mutable-feature evasion, epsilon={metrics['epsilon']}, steps={metrics['steps']})  ",
        f"**Defense:** {metrics['defense']}",
        "",
        f"## Overall grade: **{grade}**",
        "",
        "| Metric | Before hardening | After hardening |",
        "|---|---|---|",
        f"| Clean accuracy | {clean['accuracy_before']:.3f} | {clean['accuracy_after']:.3f} |",
        f"| Clean ROC-AUC | {clean['roc_auc_before']:.3f} | {clean['roc_auc_after']:.3f} |",
        f"| Clean recall (attacks caught) | {clean['recall_before']:.3f} "
        f"| {clean['recall_after']:.3f} |",
        f"| **Attack success rate** | **{s['asr_before']:.1%}** | **{s['asr_after']:.1%}** |",
        f"| Detected attacks evaded | {before['n_evaded']}/{before['n_detected_before']} "
        f"| {after['n_evaded']}/{after['n_detected_before']} |",
        "",
        f"**Attack-success-rate dropped {drop:.1%}** "
        f"(from {s['asr_before']:.1%} to {s['asr_after']:.1%}) after hardening, "
        f"while clean accuracy moved {clean['accuracy_after'] - clean['accuracy_before']:+.3f}.",
        "",
        "## Constraint compliance (adversarial flows)",
        "",
        f"- Immutable features preserved: {before['immutable_preserved_rate']:.1%}",
        f"- Per-feature validity (consistent flows): {before['consistency_rate']:.1%}",
        f"- Feasible (would survive a real network sanity check): "
        f"{before['fraction_feasible']:.1%}",
        "",
        "Only mutable features (duration, src/dst bytes, connection counts) were "
        "perturbed, in attacker-feasible directions only; error/rate aggregates "
        "and the protocol/service/flag identity were held fixed.",
        "",
        "## How to read this",
        "",
        "A high *clean* accuracy says nothing about robustness. The pre-hardening "
        "row shows how many attacks the deployed model caught but a feasible, "
        "constraint-respecting perturbation then slipped past. The after row shows "
        "the same attack re-run against the hardened model. The gap between the two "
        "is the value of the defense.",
        "",
    ]
    return "\n".join(lines)


def write_report_card(metrics: dict, out_path: Path) -> Path:
    out_path.write_text(render_report_card(metrics))
    return out_path
