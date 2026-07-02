"""A tiny decoder-only transformer in pure PyTorch, built for interpretability.

The model is deliberately small (``d_model=64, n_heads=4, n_layers=2``) and its
``forward`` optionally returns a **cache** exposing exactly what mechanistic
interpretability needs:

* per-head attention weights after softmax, for every layer, and
* the residual-stream activation after the embedding and after each layer.

``forward`` also accepts a ``resid_patch`` argument so interp code can *overwrite*
the residual stream at a chosen ``(layer, position)`` — the write side of the
read/patch pair used by activation patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Hyper-parameters for the tiny transformer."""

    vocab_size: int = 64
    n_ctx: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 128


@dataclass
class RunCache:
    """Activations captured during a forward pass (read side of the hooks)."""

    # attn[layer]: (B, n_heads, T, T) post-softmax attention weights.
    attn: list[torch.Tensor] = field(default_factory=list)
    # resid[k]: (B, T, d_model). resid[0] is post-embedding; resid[i] is the
    # residual stream after layer i-1. len == n_layers + 1.
    resid: list[torch.Tensor] = field(default_factory=list)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention that can emit its attention weights."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(z: torch.Tensor) -> torch.Tensor:
            return z.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)  # (B, H, T, d_head)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)  # (B, H, T, T)
        mask = self.causal_mask[:t, :t]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        out = attn @ v  # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.d_head)
        return self.proj(out), attn


class MLP(nn.Module):
    """Standard two-layer feed-forward block with GELU."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block: attention + MLP with residual connections."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn


class TinyTransformer(nn.Module):
    """Decoder-only transformer with interpretability hooks."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        return_cache: bool = False,
        resid_patch: tuple[int, int, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, RunCache]:
        """Run the model.

        ``resid_patch`` is ``(layer, position, value)``: after the residual
        stream at ``resid[layer]`` is formed, overwrite it at ``[:, position]``
        with ``value`` (shape ``(B, d_model)``) before continuing. ``layer==0``
        patches the post-embedding stream; ``layer==i`` patches the output of
        block ``i-1``.
        """
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]

        cache = RunCache()

        def maybe_patch(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if resid_patch is not None and resid_patch[0] == layer_idx:
                _, position, value = resid_patch
                x = x.clone()
                x[:, position, :] = value
            return x

        x = maybe_patch(x, 0)
        if return_cache:
            cache.resid.append(x)

        for i, block in enumerate(self.blocks):
            x, attn = block(x)
            x = maybe_patch(x, i + 1)
            if return_cache:
                cache.attn.append(attn)
                cache.resid.append(x)

        logits = self.unembed(self.ln_f(x))
        if return_cache:
            return logits, cache
        return logits

    def resid_to_logits(self, resid: torch.Tensor) -> torch.Tensor:
        """Project a residual-stream activation through the unembedding.

        This is the ``logit_lens`` primitive: apply the final LayerNorm and the
        unembedding to any intermediate residual stream. Shape in ``(..., d_model)``
        -> out ``(..., vocab_size)``.
        """
        return self.unembed(self.ln_f(resid))
