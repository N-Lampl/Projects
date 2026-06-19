"""Fast smoke tests (run in CI). The one slow end-to-end test that runs the whole
pipeline + writes figures is marked @slow and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import torch

from refusal_interp import (
    ablate_direction,
    build_toy_model,
    cosine_similarity,
    extract_refusal_direction,
    generate_activations,
    make_ablation_hook,
    refusal_rate,
    set_seed,
)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_extracted_direction_is_unit_vector():
    set_seed(0)
    model = build_toy_model()
    hf, hl = generate_activations(model, 64, 64, seed=2)
    r = extract_refusal_direction(hf, hl)
    assert r.shape == (model.r_true.shape[0],)
    assert abs(r.norm().item() - 1.0) < 1e-5


def test_extraction_recovers_planted_direction():
    """Difference-in-means should recover the ground-truth refusal axis."""
    set_seed(0)
    model = build_toy_model()
    hf, hl = generate_activations(model, 256, 256, seed=3)
    r = extract_refusal_direction(hf, hl)
    assert abs(cosine_similarity(r, model.r_true)) > 0.9


def test_ablation_removes_the_component():
    """After ablation, the projection onto r̂ must be ~0 (orthogonal)."""
    set_seed(0)
    model = build_toy_model()
    hf, hl = generate_activations(model, 64, 64, seed=4)
    r = extract_refusal_direction(hf, hl)
    hf_ab = ablate_direction(hf, r)
    proj = (hf_ab @ r).abs().max().item()
    assert proj < 1e-4


def test_ablation_preserves_norm_orthogonal_part():
    """Ablation only removes the r̂ component; the orthogonal energy is unchanged."""
    set_seed(0)
    r = torch.zeros(8)
    r[0] = 1.0
    h = torch.randn(10, 8)
    h_ab = ablate_direction(h, r)
    # columns 1..7 untouched
    assert torch.allclose(h_ab[:, 1:], h[:, 1:], atol=1e-6)
    # column 0 zeroed
    assert h_ab[:, 0].abs().max().item() < 1e-6


def test_ablation_reduces_refusal_not_capability():
    """The core security finding: ablation cuts refusals, keeps capability."""
    set_seed(0)
    model = build_toy_model()
    hf, hl = generate_activations(model, 256, 256, seed=5)
    r = extract_refusal_direction(hf, hl)

    rr_before = refusal_rate(model.p_refuse(hf))
    rr_after = refusal_rate(model.p_refuse(ablate_direction(hf, r)))
    cap_before = float(model.p_capable(hl).mean())
    cap_after = float(model.p_capable(ablate_direction(hl, r)).mean())

    assert rr_before > 0.5  # model refused most harmful prompts
    assert rr_after < rr_before  # ablation reduced refusals
    assert cap_after > 0.8 * cap_before  # capability largely retained


def test_ablation_hook_matches_functional_ablation():
    """The forward-hook (real-model path) must match ablate_direction()."""
    r = torch.randn(16)
    hook = make_ablation_hook(r)
    h = torch.randn(4, 16)
    via_hook = hook(None, None, h)
    via_fn = ablate_direction(h, r)
    assert torch.allclose(via_hook, via_fn, atol=1e-6)


def test_ablation_hook_handles_tuple_output():
    r = torch.randn(16)
    hook = make_ablation_hook(r)
    h = torch.randn(2, 3, 16)
    out = hook(None, None, (h, "kv-cache"))
    assert isinstance(out, tuple) and out[1] == "kv-cache"


import pytest  # noqa: E402


@pytest.mark.slow
def test_full_pipeline_writes_artifacts(tmp_path, monkeypatch):
    """End-to-end: run the analysis script and confirm figures + metrics exist."""
    import json
    import runpy
    import sys
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    script = project / "scripts" / "run_analysis.py"
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--n-harmful", "64", "--n-harmless", "64"])
    runpy.run_path(str(script), run_name="__main__")

    metrics = json.loads((project / "results" / "metrics.json").read_text())
    assert metrics["refusal_rate_after"] < metrics["refusal_rate_before"]
    assert metrics["direction_recovery_cosine"] > 0.85
    for fig in metrics["figures"]:
        assert (project / fig).exists()
