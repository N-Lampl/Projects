"""OWASP LLM Top 10 (2025) mapping for the LLM-security track.

Public API:
    OWASP_LLM_2025, COVERAGE   -- the framework + repo mapping
    coverage_table, coverage_summary
"""

from .mapping import (
    COVERAGE,
    OWASP_LLM_2025,
    coverage_summary,
    coverage_table,
)

__all__ = [
    "OWASP_LLM_2025",
    "COVERAGE",
    "coverage_table",
    "coverage_summary",
]
