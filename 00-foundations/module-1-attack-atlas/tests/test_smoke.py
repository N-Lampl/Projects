"""Fast invariant tests (run in CI). One @slow end-to-end test runs the actual
build script and validates the emitted artifacts.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from attack_atlas import (
    ATLAS_TECHNIQUES,
    PORTFOLIO_MAP,
    build_atlas_map,
    build_metrics,
    build_navigator_layer,
    render_coverage_chart,
    set_seed,
)
from attack_atlas.atlas import all_referenced_ids

PROJECT = Path(__file__).resolve().parents[1]

ATLAS_ID_RE = re.compile(r"^AML\.T\d{4}(\.\d{3})?$")
ATTACK_ID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_every_mapped_technique_is_in_catalog():
    """No portfolio entry may reference an ATLAS ID we haven't catalogued."""
    for tid in all_referenced_ids():
        assert tid in ATLAS_TECHNIQUES, f"{tid} referenced but not in catalog"


def test_atlas_ids_are_well_formed():
    for tid in ATLAS_TECHNIQUES:
        assert ATLAS_ID_RE.match(tid), f"bad ATLAS id: {tid}"


def test_attack_refs_are_well_formed():
    for tid, rec in ATLAS_TECHNIQUES.items():
        assert rec["attack_refs"], f"{tid} has no ATT&CK ref"
        for ref in rec["attack_refs"]:
            assert ATTACK_ID_RE.match(ref), f"bad ATT&CK id {ref} on {tid}"


def test_required_spec_techniques_present():
    """Spec calls out these specific IDs — they must exist and be mapped."""
    referenced = all_referenced_ids()
    for tid in ("AML.T0043", "AML.T0024", "AML.T0010"):
        assert tid in ATLAS_TECHNIQUES
        assert tid in referenced


def test_every_entry_maps_to_at_least_one_technique():
    amap = build_atlas_map()
    for entry in amap["entries"]:
        assert entry["techniques"], f"{entry['track']} maps to nothing"


def test_build_atlas_map_shape():
    amap = build_atlas_map()
    assert amap["project"] == "module-1-attack-atlas"
    assert len(amap["entries"]) == len(PORTFOLIO_MAP)
    assert amap["tactics"]
    # round-trips through json
    json.loads(json.dumps(amap))


def test_metrics_shape_and_required_keys():
    metrics = build_metrics(build_atlas_map())
    for key in ("project", "summary", "figures"):
        assert key in metrics, f"metrics missing required key {key}"
    assert metrics["n_entries"] == len(PORTFOLIO_MAP)
    assert metrics["n_atlas_techniques_referenced"] == len(all_referenced_ids())
    assert metrics["n_attack_techniques_bridged"] > 0
    assert isinstance(metrics["figures"], list) and metrics["figures"]


def test_navigator_layer_is_valid():
    layer = build_navigator_layer()
    # required-ish layer fields
    assert layer["domain"] == "enterprise-attack"
    assert layer["versions"]["layer"] == "4.5"
    assert layer["techniques"], "no techniques in layer"
    seen = set()
    for t in layer["techniques"]:
        assert ATTACK_ID_RE.match(t["techniqueID"]), t["techniqueID"]
        assert t["score"] >= 1
        assert t["techniqueID"] not in seen, "duplicate technique in layer"
        seen.add(t["techniqueID"])
    # serializable
    json.loads(json.dumps(layer))


def test_coverage_chart_is_nonempty_text():
    chart = render_coverage_chart(build_metrics(build_atlas_map()))
    assert "ATLAS tactic coverage" in chart
    assert "#" in chart


@pytest.mark.slow
def test_build_script_emits_valid_artifacts(tmp_path):
    """End-to-end: run both scripts, then validate the committed-style outputs."""
    env = {"PYTHONHASHSEED": "42"}
    import os

    env = {**os.environ, **env}
    for script in ("build_atlas_map.py", "build_navigator_layer.py"):
        r = subprocess.run(
            [sys.executable, str(PROJECT / "scripts" / script)],
            capture_output=True, text=True, env=env,
        )
        assert r.returncode == 0, r.stderr

    amap = json.loads((PROJECT / "results" / "atlas_map.json").read_text())
    metrics = json.loads((PROJECT / "results" / "metrics.json").read_text())
    layer = json.loads((PROJECT / "navigator" / "portfolio_layer.json").read_text())

    assert amap["entries"] and metrics["summary"] and layer["techniques"]
    # every figure listed in metrics actually exists on disk
    for fig in metrics["figures"]:
        assert (PROJECT / fig).exists(), f"missing figure {fig}"
