"""Fast smoke tests (run in CI). One slow end-to-end test (marked @slow) runs the
full script and checks the artifacts it writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stride_ml import (
    STRIDE,
    analyze,
    build_ml_inference_service,
    mermaid_dfd,
    render_markdown,
    set_seed,
    summarize,
)

PROJECT = Path(__file__).resolve().parents[1]


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_model_has_flows_and_boundaries():
    system = build_ml_inference_service()
    assert len(system.elements) >= 5
    assert len(system.flows) >= 5
    assert len(system.boundaries) >= 1


def test_every_stride_category_has_at_least_one_threat():
    """A useful invariant: the analysis should exercise all six STRIDE categories."""
    threats = analyze(build_ml_inference_service())
    counts = summarize(threats)
    assert set(counts) == set(STRIDE)
    for cat in STRIDE:
        assert counts[cat] >= 1, f"no threats found for {cat}"


def test_summary_counts_match_total():
    threats = analyze(build_ml_inference_service())
    counts = summarize(threats)
    assert sum(counts.values()) == len(threats)


def test_analysis_is_deterministic():
    a = [t.id for t in analyze(build_ml_inference_service())]
    b = [t.id for t in analyze(build_ml_inference_service())]
    assert a == b


def test_threat_ids_are_unique():
    threats = analyze(build_ml_inference_service())
    ids = [t.id for t in threats]
    assert len(ids) == len(set(ids))


def test_severities_are_valid():
    for t in analyze(build_ml_inference_service()):
        assert t.severity in {"Low", "Medium", "High"}


def test_mermaid_dfd_is_wellformed():
    md = mermaid_dfd(build_ml_inference_service())
    assert md.startswith("flowchart")
    assert "-->" in md  # has at least one edge
    assert "subgraph" in md  # has trust-boundary grouping


def test_render_markdown_contains_sections():
    system = build_ml_inference_service()
    threats = analyze(system)
    md = render_markdown(system, threats, "test")
    assert "# STRIDE Threat Model" in md
    assert "```mermaid" in md
    assert "## STRIDE summary" in md
    for cat in STRIDE:
        assert cat in md


@pytest.mark.slow
def test_script_end_to_end_writes_artifacts(tmp_path):
    """Run threatmodel.py and confirm it writes the doc + metrics.json with valid shape."""
    out = subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "threatmodel.py")],
        capture_output=True, text=True, cwd=str(PROJECT),
    )
    assert out.returncode == 0, out.stderr
    doc = PROJECT / "docs" / "threat-model.md"
    metrics = PROJECT / "results" / "metrics.json"
    assert doc.exists()
    assert metrics.exists()
    data = json.loads(metrics.read_text())
    assert data["project"] == "module-2-stride-ml"
    assert "summary" in data
    assert data["total_threats"] == sum(data["threats_by_category"].values())
    assert set(data["threats_by_category"]) == set(STRIDE)
