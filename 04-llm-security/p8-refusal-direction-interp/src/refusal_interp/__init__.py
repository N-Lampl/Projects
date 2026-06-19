"""Refusal-direction interpretability: extract a model's refusal axis from
mean-difference of residual activations, ablate it via forward hooks, and measure
refusal rate vs capability retention.

Framed strictly as safety-robustness / interpretability research. The committed
artifact is ANALYSIS, never a redistributed modified model. See ../../ETHICS.md.

Public API:
    set_seed, get_device              -- reproducibility helpers
    build_toy_model, generate_activations, sample_prompts  -- offline simulation
    extract_refusal_direction         -- difference-in-means refusal axis
    ablate_direction, make_ablation_hook  -- the inference-time intervention
    refusal_rate, cosine_similarity   -- measurement
"""

from .direction import (
    ablate_direction,
    cosine_similarity,
    extract_refusal_direction,
    make_ablation_hook,
    refusal_rate,
)
from .synthetic import build_toy_model, generate_activations, sample_prompts
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "build_toy_model",
    "generate_activations",
    "sample_prompts",
    "extract_refusal_direction",
    "ablate_direction",
    "make_ablation_hook",
    "refusal_rate",
    "cosine_similarity",
]
