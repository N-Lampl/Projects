"""A tiny char-level language model (single-layer GRU) — small enough for CPU."""

from __future__ import annotations

import torch
from torch import nn

from .corpus import VOCAB


class CharLM(nn.Module):
    """Embedding -> GRU -> linear. Predicts the next character index."""

    def __init__(self, vocab_size: int = len(VOCAB), embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T) long indices -> logits (B, T, vocab), hidden state."""
        emb = self.embed(x)
        out, hidden = self.gru(emb, hidden)
        logits = self.head(out)
        return logits, hidden
