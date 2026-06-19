"""A small, offline MITRE ATT&CK (Enterprise) reference.

We embed a minimal technique/tactic catalog so the project runs fully offline with
no network calls. The catalog covers the tactics and techniques referenced by the
Sigma rules in ``rules/`` plus enough neighbours to make the coverage heatmap
meaningful. IDs and names follow the public ATT&CK matrix
(https://attack.mitre.org/). Extend ``TECHNIQUES`` to grow coverage.
"""

from __future__ import annotations

# Ordered list of ATT&CK tactics (the heatmap columns).
TACTICS: list[str] = [
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

# technique_id -> {"name": ..., "tactic": ...}
# A compact slice of Enterprise ATT&CK (techniques + sub-techniques) used as the
# coverage grid. This is deliberately small so it is reviewable and offline.
TECHNIQUES: dict[str, dict[str, str]] = {
    # Execution
    "T1059.001": {"name": "PowerShell", "tactic": "Execution"},
    "T1059.004": {"name": "Unix Shell", "tactic": "Execution"},
    "T1047": {"name": "Windows Management Instrumentation", "tactic": "Execution"},
    "T1053.005": {"name": "Scheduled Task", "tactic": "Execution"},
    # Persistence (Scheduled Task and Run keys live here in ATT&CK too)
    "T1547.001": {
        "name": "Registry Run Keys / Startup Folder",
        "tactic": "Persistence",
    },
    "T1136.001": {"name": "Local Account", "tactic": "Persistence"},
    # Privilege Escalation
    "T1548.002": {"name": "Bypass User Account Control", "tactic": "Privilege Escalation"},
    # Defense Evasion
    "T1027": {"name": "Obfuscated Files or Information", "tactic": "Defense Evasion"},
    "T1070.001": {"name": "Clear Windows Event Logs", "tactic": "Defense Evasion"},
    # Credential Access
    "T1003.001": {"name": "LSASS Memory", "tactic": "Credential Access"},
    "T1110": {"name": "Brute Force", "tactic": "Credential Access"},
    # Discovery
    "T1057": {"name": "Process Discovery", "tactic": "Discovery"},
    "T1018": {"name": "Remote System Discovery", "tactic": "Discovery"},
    # Lateral Movement
    "T1021.002": {"name": "SMB/Windows Admin Shares", "tactic": "Lateral Movement"},
    # Command and Control
    "T1071.001": {"name": "Web Protocols", "tactic": "Command and Control"},
    # Exfiltration
    "T1041": {"name": "Exfiltration Over C2 Channel", "tactic": "Exfiltration"},
    # Impact
    "T1486": {"name": "Data Encrypted for Impact", "tactic": "Impact"},
}

# Some techniques map to more than one tactic in the real matrix. We keep a
# secondary mapping so a single rule can light up multiple cells.
EXTRA_TACTICS: dict[str, list[str]] = {
    "T1047": ["Lateral Movement"],
    "T1053.005": ["Persistence", "Privilege Escalation"],
}


def normalize_technique_id(tid: str) -> str:
    """Uppercase + strip an ATT&CK technique id (e.g. ``t1003.001`` -> ``T1003.001``)."""
    return tid.strip().upper()


def is_known_technique(tid: str) -> bool:
    return normalize_technique_id(tid) in TECHNIQUES


def technique_name(tid: str) -> str:
    return TECHNIQUES.get(normalize_technique_id(tid), {}).get("name", "UNKNOWN")


def tactics_for_technique(tid: str) -> list[str]:
    """All tactics a technique belongs to (primary + any extras)."""
    tid = normalize_technique_id(tid)
    entry = TECHNIQUES.get(tid)
    if not entry:
        return []
    tactics = [entry["tactic"]]
    tactics.extend(EXTRA_TACTICS.get(tid, []))
    return tactics
