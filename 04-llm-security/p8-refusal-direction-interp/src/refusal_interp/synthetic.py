"""Synthetic-activations simulation of an instruct LLM's residual stream.

This is the DEFAULT, fully-offline path. It fabricates last-token residual
activations for two prompt sets -- "harmful" (AdvBench-style) and "harmless"
(Alpaca-style) -- with a single *planted* refusal direction so the full
extract -> ablate -> measure -> plot methodology runs on CPU without
downloading any model weights.

Design of the toy model (so the experiment is honest, not circular):

  h = base_content(prompt)                # shared semantic content subspace
      + alpha(prompt) * r_true            # projection onto a planted "refusal" axis
      + noise

  - `r_true` is a fixed unit vector (the ground-truth refusal direction).
  - harmful prompts get a large positive `alpha` (the model "wants to refuse");
    harmless prompts get a small `alpha`.
  - A separate frozen "behaviour head" maps h -> P(refuse) and P(correct-answer).
    P(refuse) keys on the refusal axis; capability (P(correct)) keys on the
    content subspace, which is *orthogonal* to r_true. So ablating r_true should
    cut refusals while leaving capability largely intact -- exactly the empirical
    finding from the abliteration / refusal-direction literature.

The point of the planted direction is that mean-difference extraction should
*recover* it from data alone, which we verify (cosine similarity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

D_MODEL = 64  # toy residual width
CONTENT_DIM = 8  # rank of the shared "semantic content" subspace


@dataclass
class ToyModel:
    """A frozen synthetic instruct model with a known refusal axis."""

    r_true: torch.Tensor  # (D_MODEL,) unit vector: ground-truth refusal direction
    content_basis: torch.Tensor  # (CONTENT_DIM, D_MODEL) orthonormal content subspace
    w_refuse: torch.Tensor  # (D_MODEL,) read-out weight for P(refuse)
    b_refuse: float  # bias for P(refuse)
    w_capab: torch.Tensor  # (CONTENT_DIM,) read-out weight for capability on content coords
    b_capab: float

    def p_refuse(self, h: torch.Tensor) -> torch.Tensor:
        """P(model emits a refusal) given residual activations h: (N, D)."""
        return torch.sigmoid(h @ self.w_refuse + self.b_refuse)

    def p_capable(self, h: torch.Tensor) -> torch.Tensor:
        """Capability-retention proxy: P(model gives a correct/on-task answer).

        Reads ONLY the content subspace (orthogonal to the refusal axis), so a
        clean directional ablation of r_true cannot directly destroy capability.
        """
        content_coords = h @ self.content_basis.T  # (N, CONTENT_DIM)
        return torch.sigmoid(content_coords @ self.w_capab + self.b_capab)


def _orthonormal_rows(n_rows: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    """Return n_rows orthonormal vectors in R^dim (rows of the result)."""
    mat = torch.randn(dim, n_rows, generator=generator)
    q, _ = torch.linalg.qr(mat)  # (dim, n_rows), orthonormal columns
    return q.T.contiguous()


def build_toy_model(seed: int = 42) -> ToyModel:
    """Construct the frozen synthetic model with a planted refusal direction."""
    g = torch.Generator().manual_seed(seed)

    # Build an orthonormal set: row 0 = refusal axis, rows 1..CONTENT_DIM = content.
    basis = _orthonormal_rows(CONTENT_DIM + 1, D_MODEL, g)
    r_true = basis[0]
    content_basis = basis[1:]

    # Refusal read-out aligned with r_true (strong) + a whisper of content leakage.
    # The leakage is tiny so that removing the r_true axis collapses P(refuse) to
    # the bias regime (~0) -- mirroring the real finding that refusal is ~rank-1.
    w_refuse = 5.0 * r_true + 0.01 * torch.randn(D_MODEL, generator=g)

    # Capability read-out lives in content coordinates only (orthogonal to r_true).
    w_capab = 2.5 * torch.randn(CONTENT_DIM, generator=g)

    return ToyModel(
        r_true=r_true,
        content_basis=content_basis,
        w_refuse=w_refuse,
        b_refuse=-3.0,  # without the refusal signal, default is to answer
        w_capab=w_capab,
        b_capab=2.0,  # harmless prompts are usually answerable
    )


def generate_activations(
    model: ToyModel,
    n_harmful: int,
    n_harmless: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fabricate last-token residual activations for the two prompt sets.

    Returns (h_harmful, h_harmless), each (n, D_MODEL).
    """
    g = torch.Generator().manual_seed(seed)

    def _make(n: int, alpha_mean: float, alpha_std: float) -> torch.Tensor:
        # Shared content (capability-relevant) lives in the content subspace.
        content_coords = torch.randn(n, CONTENT_DIM, generator=g)
        content = content_coords @ model.content_basis  # (n, D)
        # Projection onto the refusal axis: harmful -> large, harmless -> small.
        alpha = alpha_mean + alpha_std * torch.randn(n, 1, generator=g)
        refusal_component = alpha * model.r_true
        noise = 0.15 * torch.randn(n, D_MODEL, generator=g)
        return content + refusal_component + noise

    h_harmful = _make(n_harmful, alpha_mean=1.8, alpha_std=0.3)
    h_harmless = _make(n_harmless, alpha_mean=-0.4, alpha_std=0.3)
    return h_harmful, h_harmless


def sample_prompts(n_harmful: int = 64, n_harmless: int = 64) -> tuple[list[str], list[str]]:
    """Synthetic AdvBench-/Alpaca-style prompt labels (templates only, no payloads).

    These are deliberately benign placeholders -- the offline path never needs
    real harmful text. They exist so the README/figures read like the real
    pipeline (which would feed actual prompt sets to a real model).
    """
    rng = np.random.default_rng(42)
    harmful_templates = [
        "[harmful-prompt-{:03d}] (AdvBench-style placeholder, no payload)",
        "[unsafe-request-{:03d}] (refused-by-design placeholder)",
    ]
    harmless_templates = [
        "[alpaca-instruction-{:03d}] summarize the following paragraph",
        "[alpaca-instruction-{:03d}] write a haiku about the ocean",
        "[alpaca-instruction-{:03d}] explain photosynthesis simply",
    ]
    harmful = [harmful_templates[rng.integers(len(harmful_templates))].format(i)
               for i in range(n_harmful)]
    harmless = [harmless_templates[rng.integers(len(harmless_templates))].format(i)
                for i in range(n_harmless)]
    return harmful, harmless
