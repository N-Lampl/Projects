"""Fast smoke tests for the OWASP LLM Top 10 mapping."""

from __future__ import annotations

from owasp_threat_lab import OWASP_LLM_2025, coverage_summary, coverage_table


def test_ten_risks_listed():
    assert len(OWASP_LLM_2025) == 10
    ids = [rid for rid, _ in OWASP_LLM_2025]
    assert ids[0] == "LLM01" and ids[-1] == "LLM10"


def test_every_row_has_id_and_name():
    for r in coverage_table():
        assert r["id"].startswith("LLM")
        assert r["name"]
        assert isinstance(r["projects"], list)


def test_summary_is_consistent():
    s = coverage_summary()
    assert s["total_risks"] == 10
    assert s["fully_covered"] + s["partial"] + s["uncovered"] == 10
    assert 0 <= s["coverage_pct"] <= 100


def test_most_risks_have_some_coverage():
    rows = coverage_table()
    covered = [r for r in rows if r["covered"]]
    assert len(covered) >= 8  # honest: at least 8/10 touched by some project
