"""Turn parsed Sigma rules into an ATT&CK coverage matrix + summary stats."""

from __future__ import annotations

from collections import defaultdict

from .attack import TACTICS, TECHNIQUES, tactics_for_technique, technique_name
from .loader import SigmaRule


def build_coverage(rules: list[SigmaRule]) -> dict[str, int]:
    """Return ``technique_id -> number of valid rules`` covering that technique."""
    counts: dict[str, int] = defaultdict(int)
    for rule in rules:
        if not rule.valid:
            continue
        for tid in rule.techniques:
            counts[tid] += 1
    return dict(counts)


def coverage_grid(coverage: dict[str, int]) -> tuple[list[str], list[str], list[list[int]]]:
    """Build a (tactics, techniques, matrix) grid for the heatmap.

    Rows are techniques (sorted by id), columns are tactics. Cell value is the
    number of rules covering that technique, placed in every tactic the technique
    belongs to.
    """
    techniques = sorted(TECHNIQUES)
    tactic_index = {t: i for i, t in enumerate(TACTICS)}
    matrix = [[0 for _ in TACTICS] for _ in techniques]
    for r, tid in enumerate(techniques):
        n = coverage.get(tid, 0)
        if not n:
            continue
        for tactic in tactics_for_technique(tid):
            matrix[r][tactic_index[tactic]] = n
    return TACTICS, techniques, matrix


def summarize(rules: list[SigmaRule], coverage: dict[str, int]) -> dict:
    """Compute headline coverage metrics."""
    valid = [r for r in rules if r.valid]
    invalid = [r for r in rules if not r.valid]
    covered = set(coverage)
    total_techniques = len(TECHNIQUES)
    covered_tactics = set()
    for tid in covered:
        covered_tactics.update(tactics_for_technique(tid))

    return {
        "n_rules": len(rules),
        "n_valid_rules": len(valid),
        "n_invalid_rules": len(invalid),
        "techniques_covered": sorted(covered),
        "n_techniques_covered": len(covered),
        "n_techniques_total": total_techniques,
        "technique_coverage_pct": round(100.0 * len(covered) / total_techniques, 1),
        "tactics_covered": sorted(covered_tactics),
        "n_tactics_covered": len(covered_tactics),
        "n_tactics_total": len(TACTICS),
        "tactic_coverage_pct": round(100.0 * len(covered_tactics) / len(TACTICS), 1),
        "coverage_by_technique": {
            tid: {"name": technique_name(tid), "n_rules": n}
            for tid, n in sorted(coverage.items())
        },
    }
