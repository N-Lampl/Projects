"""The data: MITRE ATLAS technique catalog (subset) + a mapping of each repo
track/project to the ATLAS techniques it demonstrates, plus the ATT&CK
techniques those overlap with.

ATLAS (Adversarial Threat Landscape for AI Systems) is MITRE's ATT&CK-style
matrix of real-world attacks against ML systems. IDs look like `AML.T0043`.
The catalog below is a hand-curated subset covering the techniques exercised by
this portfolio; it is intentionally small and self-contained (stdlib only, no
network calls) so the build is fully offline and reproducible.

References:
  - MITRE ATLAS matrix: https://atlas.mitre.org/matrices/ATLAS
  - MITRE ATT&CK Enterprise: https://attack.mitre.org/
"""

from __future__ import annotations

# ATLAS tactics (the column headers of the matrix), in matrix order.
ATLAS_TACTICS: list[dict[str, str]] = [
    {"id": "AML.TA0002", "name": "Reconnaissance"},
    {"id": "AML.TA0003", "name": "Resource Development"},
    {"id": "AML.TA0004", "name": "Initial Access"},
    {"id": "AML.TA0000", "name": "ML Model Access"},
    {"id": "AML.TA0005", "name": "Execution"},
    {"id": "AML.TA0006", "name": "Persistence"},
    {"id": "AML.TA0012", "name": "Privilege Escalation"},
    {"id": "AML.TA0007", "name": "Defense Evasion"},
    {"id": "AML.TA0008", "name": "Discovery"},
    {"id": "AML.TA0009", "name": "Collection"},
    {"id": "AML.TA0001", "name": "ML Attack Staging"},
    {"id": "AML.TA0010", "name": "Exfiltration"},
    {"id": "AML.TA0011", "name": "Impact"},
]

# Curated ATLAS technique catalog. `attack_refs` lists the closest MITRE ATT&CK
# Enterprise technique(s) so the artifact bridges classic threat modeling and ML
# threat modeling — the whole point of this primer.
ATLAS_TECHNIQUES: dict[str, dict] = {
    "AML.T0043": {
        "name": "Craft Adversarial Data",
        "tactic": "AML.TA0001",
        "tactic_name": "ML Attack Staging",
        "summary": (
            "Perturb inputs so a model misclassifies them (evasion) — e.g. FGSM/PGD "
            "on images or gradient-guided edits to tabular IDS features."
        ),
        "attack_refs": ["T1565"],  # Data Manipulation
    },
    "AML.T0015": {
        "name": "Evade ML Model",
        "tactic": "AML.TA0007",
        "tactic_name": "Defense Evasion",
        "summary": (
            "Use crafted adversarial inputs at inference time to bypass an ML-based "
            "control (classifier, detector, IDS)."
        ),
        "attack_refs": ["T1027"],  # Obfuscated Files or Information
    },
    "AML.T0024": {
        "name": "Exfiltration via ML Inference API",
        "tactic": "AML.TA0010",
        "tactic_name": "Exfiltration",
        "summary": (
            "Abuse query access to leak data: model extraction, membership-inference, "
            "or attribute-inference against an inference endpoint."
        ),
        "attack_refs": ["T1041"],  # Exfiltration Over C2 Channel
    },
    "AML.T0024.000": {
        "name": "Infer Training Data Membership",
        "tactic": "AML.TA0010",
        "tactic_name": "Exfiltration",
        "summary": (
            "Membership-inference: decide whether a given record was in the training "
            "set from the model's confidence/loss signal."
        ),
        "attack_refs": ["T1041"],
    },
    "AML.T0048": {
        "name": "External Harms",
        "tactic": "AML.TA0011",
        "tactic_name": "Impact",
        "summary": (
            "Privacy/financial/reputational harm caused by a successful ML attack "
            "(e.g. leaked PII via inversion)."
        ),
        "attack_refs": ["T1531"],  # Account Access Removal (impact family)
    },
    "AML.T0010": {
        "name": "ML Supply Chain Compromise",
        "tactic": "AML.TA0004",
        "tactic_name": "Initial Access",
        "summary": (
            "Tamper with a model artifact, dependency, or dataset before it reaches "
            "the victim — poisoned weights, malicious pickles, typosquatted packages."
        ),
        "attack_refs": ["T1195"],  # Supply Chain Compromise
    },
    "AML.T0010.001": {
        "name": "ML Supply Chain Compromise: ML Software",
        "tactic": "AML.TA0004",
        "tactic_name": "Initial Access",
        "summary": (
            "Compromise via the ML toolchain/dependencies (e.g. unsafe pickle/"
            "deserialization, tampered package)."
        ),
        "attack_refs": ["T1195.001"],  # Compromise Software Dependencies
    },
    "AML.T0018": {
        "name": "Manipulate ML Model (Backdoor / Poison)",
        "tactic": "AML.TA0006",
        "tactic_name": "Persistence",
        "summary": (
            "Embed a trigger-activated backdoor or otherwise poison the model so it "
            "behaves maliciously on attacker-chosen inputs."
        ),
        "attack_refs": ["T1505"],  # Server Software Component (implant)
    },
    "AML.T0051": {
        "name": "LLM Prompt Injection",
        "tactic": "AML.TA0005",
        "tactic_name": "Execution",
        "summary": (
            "Inject instructions (direct or via retrieved/3rd-party content) to "
            "override an LLM's intended behavior or tool use."
        ),
        "attack_refs": ["T1059"],  # Command and Scripting Interpreter
    },
    "AML.T0054": {
        "name": "LLM Jailbreak",
        "tactic": "AML.TA0007",
        "tactic_name": "Defense Evasion",
        "summary": (
            "Bypass an LLM's safety/guardrail alignment to elicit restricted "
            "behavior or content."
        ),
        "attack_refs": ["T1562"],  # Impair Defenses
    },
    "AML.T0057": {
        "name": "LLM Data Leakage",
        "tactic": "AML.TA0010",
        "tactic_name": "Exfiltration",
        "summary": (
            "Coax an LLM/RAG system into revealing sensitive context, system prompts, "
            "or training data."
        ),
        "attack_refs": ["T1041"],
    },
    "AML.T0040": {
        "name": "ML Model Inference API Access",
        "tactic": "AML.TA0000",
        "tactic_name": "ML Model Access",
        "summary": (
            "Legitimate or stolen query access to a deployed model — the precondition "
            "for evasion, extraction, and inference attacks."
        ),
        "attack_refs": ["T1133"],  # External Remote Services
    },
}


# Map each repo TRACK/PROJECT to the ATLAS techniques it demonstrates.
# `track` is the on-disk folder; `project` is "" for a whole-track entry.
PORTFOLIO_MAP: list[dict] = [
    {
        "track": "00-foundations",
        "project": "module-1-attack-atlas",
        "title": "Attack Atlas — ATT&CK x ATLAS primer",
        "atlas_techniques": [
            "AML.T0043",
            "AML.T0024",
            "AML.T0010",
            "AML.T0051",
        ],
        "note": "This artifact: the threat-model map for the whole portfolio.",
    },
    {
        "track": "01-detection-engineering",
        "project": "",
        "title": "Detection engineering / adversarial IDS (capstone)",
        "atlas_techniques": ["AML.T0040", "AML.T0043", "AML.T0015"],
        "note": "Evading a tabular intrusion detector with gradient-guided edits.",
    },
    {
        "track": "02-adversarial-robustness",
        "project": "p1-fgsm-mnist",
        "title": "FGSM on MNIST (evasion from scratch)",
        "atlas_techniques": ["AML.T0040", "AML.T0043", "AML.T0015"],
        "note": "Classic L-inf evasion: craft adversarial data, evade the model.",
    },
    {
        "track": "03-ml-privacy",
        "project": "",
        "title": "ML privacy — membership inference / extraction",
        "atlas_techniques": ["AML.T0040", "AML.T0024", "AML.T0024.000", "AML.T0048"],
        "note": "Exfiltration of training-data signal via the inference API.",
    },
    {
        "track": "04-llm-security",
        "project": "p4-vulnerable-rag",
        "title": "Vulnerable RAG — prompt injection / data leakage",
        "atlas_techniques": ["AML.T0051", "AML.T0054", "AML.T0057"],
        "note": "Indirect prompt injection via retrieved content; context leakage.",
    },
    {
        "track": "05-ml-supply-chain",
        "project": "",
        "title": "ML supply chain — poisoned artifacts / unsafe deserialization",
        "atlas_techniques": ["AML.T0010", "AML.T0010.001", "AML.T0018"],
        "note": "Tampered model artifacts and dependencies before deployment.",
    },
]


def techniques_for_entry(entry: dict) -> list[dict]:
    """Resolve an entry's technique IDs to full catalog records (skips unknown IDs)."""
    out = []
    for tid in entry["atlas_techniques"]:
        if tid in ATLAS_TECHNIQUES:
            rec = {"id": tid, **ATLAS_TECHNIQUES[tid]}
            out.append(rec)
    return out


def all_referenced_ids() -> set[str]:
    """Every ATLAS technique ID referenced by the portfolio map."""
    ids: set[str] = set()
    for entry in PORTFOLIO_MAP:
        ids.update(entry["atlas_techniques"])
    return ids
