"""Train the tiny transformer on the induction task until a head emerges.

Config, steps and seed are chosen so that an induction head *reliably* forms on
CPU in a few seconds. Training is deterministic given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import ModelConfig, TinyTransformer
from .task import make_induction_batch
from .utils import set_seed


@dataclass
class TrainResult:
    """Outcome of a training run."""

    model: TinyTransformer
    final_loss: float
    val_loss: float
    losses: list[float]


def masked_next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over next-token predictions, restricted to ``mask``."""
    b, t, v = logits.shape
    flat_logits = logits.reshape(b * t, v)
    flat_targets = targets.reshape(b * t)
    flat_mask = mask.reshape(b * t)
    loss = nn.functional.cross_entropy(
        flat_logits[flat_mask], flat_targets[flat_mask], reduction="mean"
    )
    return loss


def train_induction(
    cfg: ModelConfig | None = None,
    steps: int = 400,
    batch_size: int = 64,
    half: int = 16,
    lr: float = 3e-3,
    seed: int = 42,
) -> TrainResult:
    """Train on freshly drawn induction batches; score the induction half only.

    We only ask the model to predict the *repeated* half — that is where the
    induction rule holds — so the loss pressure produces an induction head.
    """
    set_seed(seed)
    if cfg is None:
        cfg = ModelConfig(vocab_size=64, n_ctx=2 * half, d_model=64, n_heads=4, n_layers=2)

    model = TinyTransformer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    model.train()
    for step in range(steps):
        batch = make_induction_batch(
            batch_size=batch_size, half=half, vocab_size=cfg.vocab_size, seed=1000 + step
        )
        logits = model(batch.tokens)
        loss = masked_next_token_loss(logits, batch.targets, batch.repeat_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        val = make_induction_batch(
            batch_size=256, half=half, vocab_size=cfg.vocab_size, seed=99_999
        )
        val_logits = model(val.tokens)
        val_loss = float(masked_next_token_loss(val_logits, val.targets, val.repeat_mask).item())

    return TrainResult(model=model, final_loss=losses[-1], val_loss=val_loss, losses=losses)
