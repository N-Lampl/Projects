"""Fast smoke tests for the CI red-team gate (run in CI via -m 'not slow').

These are the invariants the *gate itself* depends on -- if any break, the gate
could pass a vulnerable app or fail a clean one. The one slow test runs the whole
pipeline end-to-end against the real sibling targets.
"""

from __future__ import annotations

import pytest

from appsec_ci import (
    OWASP_CATEGORIES,
    ProbeResult,
    evaluate,
    load_defended_target,
    load_target,
    run_promptfoo,
    set_seed,
)


def test_set_seed_is_deterministic():
    import numpy as np

    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert (a == b).all()


def _mk(owasp: str, ok: bool, control: bool = False) -> ProbeResult:
    return ProbeResult(
        harness="t", probe_id="p", owasp=owasp, succeeded=ok, is_control=control
    )


def test_gate_fails_when_any_attack_lands_at_zero_threshold():
    label = OWASP_CATEGORIES["LLM01"]
    res = [_mk(label, True), _mk(label, False)]
    g = evaluate(res, threshold=0.0)
    assert not g.passed
    assert g.overall_asr == 0.5
    assert g.reasons  # explains why


def test_gate_passes_when_all_attacks_blocked():
    label = OWASP_CATEGORIES["LLM02"]
    res = [_mk(label, False), _mk(label, False)]
    g = evaluate(res, threshold=0.0)
    assert g.passed
    assert g.overall_asr == 0.0


def test_gate_respects_threshold():
    label = OWASP_CATEGORIES["LLM01"]
    res = [_mk(label, True)] + [_mk(label, False)] * 9  # ASR 10%
    assert not evaluate(res, threshold=0.0).passed
    assert evaluate(res, threshold=0.1).passed  # 10% <= 10%


def test_controls_excluded_from_asr_but_fp_fails_gate():
    label = OWASP_CATEGORIES["LLM01"]
    # 1 attack blocked + 1 benign control that wrongly leaked
    res = [_mk(label, False), _mk("control", True, control=True)]
    g = evaluate(res, threshold=0.0)
    assert g.overall_asr == 0.0  # control not counted in ASR
    assert g.benign_false_positives == 1
    assert not g.passed  # but a false positive still fails the gate


def test_aggregate_buckets_by_owasp():
    res = [
        _mk(OWASP_CATEGORIES["LLM01"], True),
        _mk(OWASP_CATEGORIES["LLM01"], False),
        _mk(OWASP_CATEGORIES["LLM02"], True),
    ]
    g = evaluate(res, threshold=0.0)
    assert g.by_owasp[OWASP_CATEGORIES["LLM01"]].asr == 0.5
    assert g.by_owasp[OWASP_CATEGORIES["LLM02"]].asr == 1.0


def test_targets_load_and_query():
    """Both target loaders return something with a working query()."""
    vuln, vlabel = load_target()
    rem, rlabel = load_defended_target()
    assert isinstance(vlabel, str) and vlabel
    assert isinstance(rlabel, str) and rlabel
    assert isinstance(vuln.query("hello"), str)
    assert isinstance(rem.query("hello"), str)


def test_vulnerable_target_is_actually_vulnerable():
    """Sanity: the lab target must leak, else the gate demo is meaningless."""
    vuln, _ = load_target()
    g = evaluate(run_promptfoo(vuln), threshold=0.0)
    assert g.overall_asr > 0.5  # the deliberately-vulnerable app leaks


def test_remediated_target_blocks_the_smoke_suite():
    """The remediated target must drive ASR down (gate should pass)."""
    rem, _ = load_defended_target()
    g = evaluate(run_promptfoo(rem), threshold=0.0)
    assert g.overall_asr == 0.0
    assert g.passed


@pytest.mark.slow
def test_full_pipeline_end_to_end(tmp_path, monkeypatch):
    """Run the actual pipeline script and assert it produces all artifacts."""
    import json
    import runpy
    import sys
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    script = project / "scripts" / "run_pipeline.py"
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--full"])
    runpy.run_path(str(script), run_name="__main__")

    metrics = json.loads((project / "results" / "metrics.json").read_text())
    assert metrics["project"] == "CAPSTONE-appsec-ci"
    assert metrics["vulnerable"]["overall_asr"] > 0.5
    assert metrics["remediated"]["overall_asr"] == 0.0
    for fig in metrics["figures"]:
        assert (project / fig).exists()
    assert (project / metrics["report"]).exists()
