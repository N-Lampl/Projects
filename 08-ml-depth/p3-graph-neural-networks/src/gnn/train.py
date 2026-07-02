"""Deterministic Adam training loop for the GCN / MLP on CPU (a few seconds)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .graph import GraphDataset, normalized_adjacency
from .model import GCN, MLP
from .utils import set_seed


@dataclass
class TrainResult:
    """What a single train run returns."""

    test_acc: float
    val_acc: float
    embeddings: torch.Tensor  # (N, hidden) node embeddings for t-SNE
    logits: torch.Tensor  # (N, n_classes) final predictions


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of masked nodes classified correctly."""
    pred = logits[mask].argmax(dim=1)
    return float((pred == labels[mask]).float().mean().item())


def train_model(
    model: nn.Module,
    ds: GraphDataset,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 42,
) -> TrainResult:
    """Train ``model`` with Adam and return test accuracy + hidden embeddings."""
    set_seed(seed)
    a_hat = normalized_adjacency(ds.adj)
    x, y = ds.features, ds.labels
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, a_hat)
        loss = loss_fn(logits[ds.train_mask], y[ds.train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x, a_hat)
    return TrainResult(
        test_acc=accuracy(logits, y, ds.test_mask),
        val_acc=accuracy(logits, y, ds.val_mask),
        embeddings=model.embeddings(x, a_hat),
        logits=logits,
    )


def train_gcn(
    ds: GraphDataset, *, hidden: int = 16, epochs: int = 200, seed: int = 42
) -> TrainResult:
    """Convenience: build + train a 2-layer GCN on ``ds``."""
    set_seed(seed)
    model = GCN(ds.n_features, hidden, ds.n_classes)
    return train_model(model, ds, epochs=epochs, seed=seed)


def train_mlp(
    ds: GraphDataset, *, hidden: int = 16, epochs: int = 200, seed: int = 42
) -> TrainResult:
    """Convenience: build + train the graph-blind MLP baseline on ``ds``."""
    set_seed(seed)
    model = MLP(ds.n_features, hidden, ds.n_classes)
    return train_model(model, ds, epochs=epochs, seed=seed)
