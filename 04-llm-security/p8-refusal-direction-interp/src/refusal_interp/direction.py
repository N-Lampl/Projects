"""The refusal-direction method: extract, ablate, measure.

These functions are model-agnostic: they operate on residual-activation tensors,
so the SAME code drives the synthetic offline path and (optionally) a real
transformer's hooked activations.

Method (Arditi et al. 2024, "Refusal in LLMs is mediated by a single direction"):

  1. Extract.  r = mean(h_harmful) - mean(h_harmless), then normalise to a unit
     vector r_hat. This difference-in-means isolates the axis that most separates
     "I will refuse" from "I will answer".

  2. Ablate.  For every activation h, remove its component along r_hat:
        h' = h - (h . r_hat) r_hat
     Applied at inference time via forward hooks on the residual stream, this is
     "abliteration": the model can no longer represent the refusal feature.

  3. Measure.  Refusal rate (P(refuse)) before vs after ablation, plus a
     capability-retention proxy, to show the ablation is *surgical*.
"""

from __future__ import annotations

import torch


def extract_refusal_direction(
    h_harmful: torch.Tensor, h_harmless: torch.Tensor
) -> torch.Tensor:
    """Difference-in-means refusal direction (unit vector), shape (D,)."""
    r = h_harmful.mean(dim=0) - h_harmless.mean(dim=0)
    return r / (r.norm() + 1e-8)


def ablate_direction(h: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    """Project activations orthogonal to r_hat (the inference-time intervention).

    h: (N, D), r_hat: (D,) unit vector. Returns h with the r_hat component removed.
    """
    r_hat = r_hat / (r_hat.norm() + 1e-8)
    proj = (h @ r_hat).unsqueeze(-1) * r_hat  # (N, D)
    return h - proj


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors (sign-invariant magnitude)."""
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a @ b).item())


def refusal_rate(p_refuse: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of prompts the model would refuse (P(refuse) > threshold)."""
    return float((p_refuse > threshold).float().mean().item())


def make_ablation_hook(r_hat: torch.Tensor):
    """Return a PyTorch forward-hook that ablates r_hat from a module's output.

    Usage with a real transformer (optional path):
        h = layer.register_forward_hook(make_ablation_hook(r_hat))
    The hook handles both bare-tensor and tuple module outputs.
    """
    r = r_hat / (r_hat.norm() + 1e-8)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs - (hs @ r).unsqueeze(-1) * r
            return (hs, *output[1:])
        return output - (output @ r).unsqueeze(-1) * r

    return hook
