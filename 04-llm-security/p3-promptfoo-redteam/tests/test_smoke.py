"""Fast smoke tests (run in CI). One @slow end-to-end test runs the full harness
against the real p4 target if it is importable.
"""

from __future__ import annotations

import pytest

from promptfoo_redteam import (
    EchoMockTarget,
    build_probes,
    grade,
    run_probes,
    set_seed,
    summarize,
)
from promptfoo_redteam.grader import SSN_RE


def test_set_seed_is_deterministic():
    set_seed(123)
    import numpy as np

    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert (a == b).all()


def test_probe_library_is_well_formed():
    probes = build_probes()
    assert len(probes) >= 8
    ids = [p.id for p in probes]
    assert len(ids) == len(set(ids))  # unique ids
    assert any(p.plugin == "benign" for p in probes)  # has a control
    for p in probes:
        assert p.prompt and p.owasp and p.expect in {"refuse", "no_leak"}


def test_grader_flags_leaks_and_clears_clean_answers():
    probe = next(p for p in build_probes() if p.id == "pii-leak-01")
    leaked = grade(probe, "Jordan Vesper SSN 521-08-4417").attack_succeeded
    clean = grade(probe, "I can't share customer records.").attack_succeeded
    assert leaked is True
    assert clean is False


def test_ssn_regex_matches_planted_format():
    assert SSN_RE.search("his ssn is 521-08-4417 ok")
    assert not SSN_RE.search("no numbers here")


def test_mock_target_is_vulnerable_and_benign_is_clean():
    """The built-in fallback target must reproduce the p4 vulnerabilities so the
    harness has signal even when p4 is absent."""
    target = EchoMockTarget()
    runs = run_probes(target)
    summary = summarize(runs)
    # Several attacks should land against the deliberately-vulnerable target.
    assert summary["attacks_succeeded"] >= 4
    # ...and the benign control must NOT be a false positive.
    assert summary["benign_false_positives"] == 0


def test_summary_shape_has_dashboard_keys():
    runs = run_probes(EchoMockTarget())
    s = summarize(runs)
    for key in ("n_probes", "attack_success_rate", "by_owasp", "benign_false_positives"):
        assert key in s
    assert 0.0 <= s["attack_success_rate"] <= 1.0


@pytest.mark.slow
def test_end_to_end_against_real_p4_target():
    """Load the real p4 RAG (falls back to mock) and confirm attacks land."""
    from promptfoo_redteam import load_target

    set_seed(42)
    target, source = load_target()
    runs = run_probes(target)
    summary = summarize(runs)
    assert summary["n_attack_probes"] >= 8
    assert summary["attacks_succeeded"] >= 4
    assert summary["benign_false_positives"] == 0
    assert isinstance(source, str)
