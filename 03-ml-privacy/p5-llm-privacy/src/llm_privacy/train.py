"""Train / save / load the tiny char-LM on the canary-laced corpus."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .corpus import VOCAB, encode
from .model import CharLM
from .utils import get_device


def make_batches(
    text: str, seq_len: int = 64, batch_size: int = 64, seed: int = 42
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Chop the encoded corpus into (input, target) next-char chunks."""
    data = torch.tensor(encode(text), dtype=torch.long)
    n_chunks = (len(data) - 1) // seq_len
    data = data[: n_chunks * seq_len + 1]
    xs = torch.stack([data[i * seq_len : i * seq_len + seq_len] for i in range(n_chunks)])
    ys = torch.stack([data[i * seq_len + 1 : i * seq_len + seq_len + 1] for i in range(n_chunks)])

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_chunks, generator=g)
    xs, ys = xs[perm], ys[perm]

    batches = []
    for i in range(0, n_chunks, batch_size):
        batches.append((xs[i : i + batch_size], ys[i : i + batch_size]))
    return batches


def train(
    model: CharLM,
    text: str,
    epochs: int = 8,
    lr: float = 3e-3,
    seq_len: int = 64,
    batch_size: int = 64,
    device: torch.device | None = None,
    log_every: int = 1,
) -> list[float]:
    """Standard next-char cross-entropy training. Returns per-epoch loss."""
    device = device or get_device()
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    for ep in range(epochs):
        batches = make_batches(text, seq_len, batch_size, seed=ep)
        total, count = 0.0, 0
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, len(VOCAB)), yb.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        avg = total / max(count, 1)
        epoch_losses.append(avg)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1}/{epochs}  loss={avg:.4f}")
    model.eval()
    return epoch_losses


def save_model(model: CharLM, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, device: torch.device | None = None) -> CharLM:
    device = device or get_device()
    model = CharLM()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
