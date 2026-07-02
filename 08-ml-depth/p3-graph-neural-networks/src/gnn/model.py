"""Two-layer GCN and a graph-blind MLP baseline — pure PyTorch, no torch-geometric.

The GCN and the MLP have the *identical* architecture (two linear layers, one ReLU,
same widths); the ONLY difference is whether the normalized adjacency ``Â`` is
applied between layers. So any accuracy gap is attributable to message passing over
the graph structure, not to model capacity.
"""

from __future__ import annotations

import torch
from torch import nn


class GCN(nn.Module):
    """H1 = relu(Â X W0);  out = Â H1 W1  (Kipf & Welling, 2017)."""

    def __init__(self, in_features: int, hidden: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden)
        self.lin1 = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self._hidden: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = a_hat @ self.lin0(x)  # aggregate neighbours, then transform
        h = torch.relu(h)
        self._hidden = h  # cache pre-dropout embeddings for t-SNE
        h = self.dropout(h)
        return a_hat @ self.lin1(h)

    def embeddings(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """Hidden-layer node embeddings (used for the t-SNE plot)."""
        self.eval()
        with torch.no_grad():
            h = torch.relu(a_hat @ self.lin0(x))
        return h


class MLP(nn.Module):
    """Same shape as the GCN but WITHOUT Â — a graph-blind baseline.

    ``forward`` accepts the same ``(x, a_hat)`` signature as :class:`GCN` so the
    training loop is shared; ``a_hat`` is simply ignored.
    """

    def __init__(self, in_features: int, hidden: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden)
        self.lin1 = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self._hidden: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.relu(self.lin0(x))
        self._hidden = h
        h = self.dropout(h)
        return self.lin1(h)

    def embeddings(self, x: torch.Tensor, a_hat: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h = torch.relu(self.lin0(x))
        return h
