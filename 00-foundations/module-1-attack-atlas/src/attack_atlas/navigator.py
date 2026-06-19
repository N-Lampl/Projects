"""Emit a valid MITRE ATT&CK Navigator layer JSON highlighting the ATT&CK
techniques this portfolio exercises (bridged from ATLAS via `atlas.py`).

The Navigator layer format (layer v4.5, ATT&CK Enterprise) is documented at:
  https://github.com/mitre-attack/attack-navigator/blob/master/layers/LAYERFORMAT.md

Load the emitted file with File > Open Existing Layer at
  https://mitre-attack.github.io/attack-navigator/
"""

from __future__ import annotations

from collections import Counter

from .atlas import ATLAS_TECHNIQUES, PORTFOLIO_MAP

# Sub-technique refs (e.g. T1195.001) are kept as-is; Navigator accepts them.


def _attack_usage() -> Counter[str]:
    """Count how many portfolio entries map (via ATLAS) to each ATT&CK technique."""
    usage: Counter[str] = Counter()
    for entry in PORTFOLIO_MAP:
        seen: set[str] = set()
        for tid in entry["atlas_techniques"]:
            for ref in ATLAS_TECHNIQUES.get(tid, {}).get("attack_refs", []):
                seen.add(ref)
        for ref in seen:
            usage[ref] += 1
    return usage


def build_navigator_layer() -> dict:
    """Build a Navigator layer dict ready to json.dump."""
    usage = _attack_usage()
    peak = max(usage.values()) if usage else 1

    techniques = []
    for tid, count in sorted(usage.items()):
        # which ATLAS techniques drove this ATT&CK ref (for the tooltip)
        atlas_drivers = sorted(
            aid
            for aid, rec in ATLAS_TECHNIQUES.items()
            if tid in rec.get("attack_refs", [])
        )
        techniques.append(
            {
                "techniqueID": tid,
                "score": count,
                "color": "",
                "comment": "ATLAS: " + ", ".join(atlas_drivers),
                "enabled": True,
                "metadata": [{"name": "atlas_techniques", "value": ", ".join(atlas_drivers)}],
                "showSubtechniques": "." in tid,
            }
        )

    return {
        "name": "ML-Security Portfolio — ATLAS x ATT&CK",
        "versions": {"attack": "15", "navigator": "5.1.0", "layer": "4.5"},
        "domain": "enterprise-attack",
        "description": (
            "ATT&CK techniques exercised by the ml-security-portfolio, bridged from "
            "MITRE ATLAS. Score = number of portfolio entries that touch the technique."
        ),
        "filters": {"platforms": ["Linux", "Windows", "macOS", "Containers"]},
        "sorting": 3,
        "layout": {"layout": "side", "showName": True, "showID": True},
        "hideDisabled": False,
        "techniques": techniques,
        "gradient": {
            "colors": ["#ffe0e0", "#ff6666", "#b30000"],
            "minValue": 0,
            "maxValue": peak,
        },
        "legendItems": [],
        "showTacticRowBackground": True,
        "tacticRowBackground": "#dddddd",
        "selectTechniquesAcrossTactics": True,
        "selectSubtechniquesWithParent": False,
    }
