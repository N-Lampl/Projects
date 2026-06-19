"""OWASP Top 10 for LLM Applications (2025) and how this repo's projects map to it.

Single source of truth for the LLM-security track. Other projects link back here
instead of restating the framework. Reference: genai.owasp.org/llm-top-10.
"""

from __future__ import annotations

# (id, short name) -- the official 2025 list.
OWASP_LLM_2025: list[tuple[str, str]] = [
    ("LLM01", "Prompt Injection"),
    ("LLM02", "Sensitive Information Disclosure"),
    ("LLM03", "Supply Chain"),
    ("LLM04", "Data and Model Poisoning"),
    ("LLM05", "Improper Output Handling"),
    ("LLM06", "Excessive Agency"),
    ("LLM07", "System Prompt Leakage"),
    ("LLM08", "Vector and Embedding Weaknesses"),
    ("LLM09", "Misinformation"),
    ("LLM10", "Unbounded Consumption"),
]

# Which repo projects exercise (attack ▸) or mitigate (defend ▸) each risk.
COVERAGE: dict[str, list[str]] = {
    "LLM01": [
        "04-llm-security/p4-vulnerable-rag",
        "04-llm-security/p5-attack-rag-pyrit",
        "04-llm-security/p2-garak-scan",
        "04-llm-security/p3-promptfoo-redteam",
        "04-llm-security/p7-defend-rag",
    ],
    "LLM02": [
        "04-llm-security/p5-attack-rag-pyrit",
        "03-ml-privacy/p5-llm-privacy",
        "04-llm-security/p7-defend-rag",
    ],
    "LLM03": ["05-ml-supply-chain/secure-ml-pipeline"],
    "LLM04": ["01-detection-engineering/CAPSTONE-adversarial-ids (poisoning: stretch repo)"],
    "LLM05": ["04-llm-security/p4-vulnerable-rag", "04-llm-security/p7-defend-rag"],
    "LLM06": ["04-llm-security/p6-agent-tool-abuse"],
    "LLM07": ["04-llm-security/p4-vulnerable-rag", "04-llm-security/p5-attack-rag-pyrit"],
    "LLM08": ["04-llm-security/p4-vulnerable-rag", "04-llm-security/p7-defend-rag"],
    "LLM09": ["04-llm-security/p7-defend-rag", "01-detection-engineering/p7-drift-monitoring"],
    "LLM10": [
        "04-llm-security/CAPSTONE-appsec-ci",
        "03-ml-privacy/p1-api-threat-model",
    ],
}

# Risks with only a partial/stretch mapping (be honest about coverage).
PARTIAL = {"LLM04"}


def coverage_table() -> list[dict]:
    """Structured rows: id, name, projects, covered, partial."""
    rows = []
    for rid, name in OWASP_LLM_2025:
        projects = COVERAGE.get(rid, [])
        rows.append(
            {
                "id": rid,
                "name": name,
                "projects": projects,
                "covered": len(projects) > 0,
                "partial": rid in PARTIAL,
            }
        )
    return rows


def coverage_summary() -> dict:
    rows = coverage_table()
    full = [r for r in rows if r["covered"] and not r["partial"]]
    partial = [r for r in rows if r["partial"]]
    return {
        "total_risks": len(rows),
        "fully_covered": len(full),
        "partial": len(partial),
        "uncovered": len([r for r in rows if not r["covered"]]),
        "coverage_pct": round(100 * len(full) / len(rows), 1),
    }
