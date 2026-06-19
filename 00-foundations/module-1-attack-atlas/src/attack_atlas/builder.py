"""Build the ATLAS map and the derived counts/metrics from the catalog + the
portfolio map in `atlas.py`. Pure stdlib, pure functions — easy to unit test.
"""

from __future__ import annotations

from collections import Counter

from .atlas import (
    ATLAS_TACTICS,
    ATLAS_TECHNIQUES,
    PORTFOLIO_MAP,
    all_referenced_ids,
    techniques_for_entry,
)

PROJECT_NAME = "module-1-attack-atlas"


def build_atlas_map() -> dict:
    """Build the full track/project -> ATLAS technique mapping document."""
    tactics_by_id = {t["id"]: t["name"] for t in ATLAS_TACTICS}

    entries = []
    for entry in PORTFOLIO_MAP:
        techniques = techniques_for_entry(entry)
        entries.append(
            {
                "track": entry["track"],
                "project": entry["project"],
                "title": entry["title"],
                "note": entry["note"],
                "techniques": [
                    {
                        "id": t["id"],
                        "name": t["name"],
                        "tactic": t["tactic"],
                        "tactic_name": tactics_by_id.get(t["tactic"], t["tactic_name"]),
                        "attack_refs": t["attack_refs"],
                    }
                    for t in techniques
                ],
            }
        )

    return {
        "project": PROJECT_NAME,
        "schema": "attack-atlas-map/v1",
        "description": (
            "Maps each ml-security-portfolio track/project to the MITRE ATLAS "
            "techniques it demonstrates, with the closest MITRE ATT&CK references."
        ),
        "tactics": ATLAS_TACTICS,
        "entries": entries,
    }


def build_metrics(atlas_map: dict) -> dict:
    """Derive counts/coverage metrics from a built atlas map."""
    referenced = all_referenced_ids()
    technique_usage: Counter[str] = Counter()
    tactic_usage: Counter[str] = Counter()
    attack_refs: set[str] = set()

    for entry in atlas_map["entries"]:
        for t in entry["techniques"]:
            technique_usage[t["id"]] += 1
            tactic_usage[t["tactic_name"]] += 1
            attack_refs.update(t["attack_refs"])

    n_entries = len(atlas_map["entries"])
    n_projects = sum(1 for e in atlas_map["entries"] if e["project"])

    summary = (
        f"Mapped {n_entries} portfolio entries to {len(referenced)} MITRE ATLAS "
        f"techniques across {len(tactic_usage)} tactics, bridged to "
        f"{len(attack_refs)} MITRE ATT&CK techniques."
    )

    return {
        "project": PROJECT_NAME,
        "summary": summary,
        "n_entries": n_entries,
        "n_projects": n_projects,
        "n_tracks": len({e["track"] for e in atlas_map["entries"]}),
        "n_atlas_techniques_referenced": len(referenced),
        "n_atlas_techniques_in_catalog": len(ATLAS_TECHNIQUES),
        "n_atlas_tactics_covered": len(tactic_usage),
        "n_attack_techniques_bridged": len(attack_refs),
        "technique_usage": dict(sorted(technique_usage.items())),
        "tactic_coverage": dict(sorted(tactic_usage.items())),
        "attack_refs": sorted(attack_refs),
        "figures": ["results/figures/coverage_by_tactic.txt"],
    }


def render_coverage_chart(metrics: dict) -> str:
    """A tiny stdlib-only ASCII bar chart of ATLAS-tactic coverage.

    matplotlib is intentionally NOT a dependency (stdlib-only project), so the
    committed 'figure' is a text bar chart. It still gives a visual artifact.
    """
    coverage = metrics["tactic_coverage"]
    if not coverage:
        return "(no coverage)\n"
    width = max(len(k) for k in coverage)
    peak = max(coverage.values())
    lines = ["ATLAS tactic coverage across the portfolio (count of mappings)", ""]
    for tactic, count in sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0])):
        bar = "#" * int(round(count / peak * 30)) or "#"
        lines.append(f"{tactic.ljust(width)} | {bar} {count}")
    return "\n".join(lines) + "\n"
