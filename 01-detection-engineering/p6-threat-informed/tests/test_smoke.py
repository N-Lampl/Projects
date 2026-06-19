"""Fast smoke tests (run in CI). One slow end-to-end test is marked @slow and
excluded from CI via ``-m "not slow"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threat_informed import (
    TACTICS,
    TECHNIQUES,
    build_coverage,
    coverage_grid,
    load_rules,
    load_yaml,
    parse_rule,
    set_seed,
    summarize,
    tactics_for_technique,
)

RULES_DIR = Path(__file__).resolve().parents[1] / "rules"


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_minimal_yaml_parses_nested_maps_and_lists():
    text = (
        "title: demo\n"
        "logsource:\n"
        "  category: process_creation\n"
        "detection:\n"
        "  selection:\n"
        "    CommandLine|contains:\n"
        "      - 'foo'\n"
        "      - 'bar'\n"
        "  condition: selection\n"
    )
    data = load_yaml(text)
    assert data["title"] == "demo"
    assert data["logsource"]["category"] == "process_creation"
    assert data["detection"]["selection"]["CommandLine|contains"] == ["foo", "bar"]
    assert data["detection"]["condition"] == "selection"


def test_all_shipped_rules_are_valid():
    rules = load_rules(RULES_DIR)
    assert len(rules) >= 6
    bad = [(r.path.name, r.errors) for r in rules if not r.valid]
    assert not bad, f"invalid rules: {bad}"


def test_every_rule_maps_to_a_known_technique():
    for rule in load_rules(RULES_DIR):
        assert rule.techniques, f"{rule.path.name} has no technique"
        for tid in rule.techniques:
            assert tid in TECHNIQUES, f"{tid} unknown"


def test_coverage_grid_shape_matches_catalog():
    rules = load_rules(RULES_DIR)
    coverage = build_coverage(rules)
    tactics, techniques, matrix = coverage_grid(coverage)
    assert tactics == TACTICS
    assert len(matrix) == len(techniques) == len(TECHNIQUES)
    assert all(len(row) == len(TACTICS) for row in matrix)


def test_covered_cells_land_in_correct_tactic_columns():
    rules = load_rules(RULES_DIR)
    coverage = build_coverage(rules)
    tactics, techniques, matrix = coverage_grid(coverage)
    tindex = {t: i for i, t in enumerate(tactics)}
    for r, tid in enumerate(techniques):
        if coverage.get(tid):
            for tactic in tactics_for_technique(tid):
                assert matrix[r][tindex[tactic]] == coverage[tid]


def test_invalid_rule_is_flagged(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("title: broken\nlevel: low\n")  # no logsource/detection/tags
    rule = parse_rule(bad)
    assert not rule.valid
    assert any("logsource" in e or "detection" in e for e in rule.errors)


def test_unknown_technique_is_rejected(tmp_path):
    bad = tmp_path / "unknown.yml"
    bad.write_text(
        "title: t\n"
        "logsource:\n  category: process_creation\n"
        "detection:\n  selection:\n    Image: x\n  condition: selection\n"
        "level: low\n"
        "tags:\n  - attack.t9999\n"
    )
    rule = parse_rule(bad)
    assert not rule.valid
    assert any("unknown" in e.lower() for e in rule.errors)


@pytest.mark.slow
def test_end_to_end_summary_is_consistent():
    rules = load_rules(RULES_DIR)
    coverage = build_coverage(rules)
    summary = summarize(rules, coverage)
    assert summary["n_valid_rules"] == summary["n_rules"]
    assert summary["n_techniques_covered"] == len(coverage)
    assert 0 < summary["technique_coverage_pct"] <= 100
    assert summary["n_tactics_covered"] >= 1
