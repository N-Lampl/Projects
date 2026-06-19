"""Fast smoke tests (run in CI) validating the SY0-701 domain map is well-formed.
The one slow test runs the coverage script end-to-end and is excluded via -m "not slow".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cert_coverage import compute_coverage, is_covered, load_syllabus

PROJECT = Path(__file__).resolve().parents[1]


def test_syllabus_loads_and_has_five_domains():
    data = load_syllabus()
    assert data["exam"].startswith("CompTIA Security+ SY0-701")
    assert len(data["domains"]) == 5


def test_exam_weights_sum_to_one():
    """Official SY0-701 domain weights must total 100%."""
    data = load_syllabus()
    total = sum(float(d["weight"]) for d in data["domains"])
    assert abs(total - 1.0) < 1e-9


def test_every_domain_maps_to_an_artifact():
    """The whole point: no exam domain is left unmapped."""
    data = load_syllabus()
    assert all(is_covered(d) for d in data["domains"])


def test_domain_ids_are_unique():
    data = load_syllabus()
    ids = [d["id"] for d in data["domains"]]
    assert len(ids) == len(set(ids))


def test_compute_coverage_full():
    data = load_syllabus()
    cov = compute_coverage(data)
    assert cov["n_domains"] == 5
    assert cov["domain_coverage"] == 1.0
    assert abs(cov["weighted_coverage"] - 1.0) < 1e-9


def test_is_covered_handles_empty():
    assert is_covered({"repo_modules": []}) is False
    assert is_covered({"repo_modules": ["  "]}) is False
    assert is_covered({"repo_modules": ["module-x"]}) is True


@pytest.mark.slow
def test_coverage_script_writes_artifacts(tmp_path, monkeypatch):
    """Run the script end-to-end and confirm it emits metrics.json + a figure."""
    import cert_coverage as cov_mod

    fig_dir = tmp_path / "figures"
    metrics = tmp_path / "metrics.json"
    monkeypatch.setattr(cov_mod, "FIG_DIR", fig_dir)
    monkeypatch.setattr(cov_mod, "METRICS", metrics)
    monkeypatch.setattr("sys.argv", ["cert_coverage.py"])

    cov_mod.main()

    assert (fig_dir / "domain_coverage.png").exists()
    data = json.loads(metrics.read_text())
    assert data["project"] == "certpath"
    assert data["domain_coverage"] == 1.0
    assert "summary" in data and data["figures"]
