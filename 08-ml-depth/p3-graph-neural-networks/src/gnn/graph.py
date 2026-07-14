"""Synthetic graph data - a stochastic block model (SBM) with planted communities.

The default path is fully offline. An SBM draws ``n_classes`` communities; a pair of
nodes in the *same* community is connected with probability ``p_in`` and a pair in
*different* communities with ``p_out < p_in``. Each node also gets a feature vector
that is *weakly* correlated with its community (so features alone don't solve the
task) - that is the whole point: message passing over the edges is what closes the
gap. Because the community labels are stored on the dataset, every prediction is
scored against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphDataset:
    """A node-classification graph with known community labels and node masks."""

    adj: torch.Tensor  # (N, N) dense 0/1 adjacency (symmetric, no self-loops)
    features: torch.Tensor  # (N, F) node features (community-correlated but noisy)
    labels: torch.Tensor  # (N,) integer community labels
    train_mask: torch.Tensor  # (N,) bool
    val_mask: torch.Tensor  # (N,) bool
    test_mask: torch.Tensor  # (N,) bool
    source: str

    @property
    def n_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_classes(self) -> int:
        return int(self.labels.max().item()) + 1

    @property
    def n_edges(self) -> int:
        """Number of undirected edges (upper triangle of the adjacency)."""
        return int(torch.triu(self.adj, diagonal=1).sum().item())


def make_sbm(
    n_nodes: int = 300,
    n_classes: int = 3,
    n_features: int = 20,
    p_in: float = 0.08,
    p_out: float = 0.02,
    feature_snr: float = 0.45,
    train_per_class: int = 20,
    val_per_class: int = 15,
    seed: int = 0,
) -> GraphDataset:
    """Draw an SBM with planted communities and community-correlated node features.

    ``p_in`` / ``p_out`` control how block-structured the graph is; the larger the
    gap, the more the edges reveal. ``feature_snr`` scales a per-class mean signal
    added on top of unit Gaussian noise - small enough that a graph-blind model
    struggles, so message passing has room to help.
    """
    rng = np.random.default_rng(seed)

    # Balanced community assignment, then shuffled so masks aren't block-ordered.
    labels = np.repeat(np.arange(n_classes), int(np.ceil(n_nodes / n_classes)))[:n_nodes]
    labels = rng.permutation(labels)

    # Symmetric adjacency: same-community edges at p_in, cross-community at p_out.
    same = labels[:, None] == labels[None, :]
    probs = np.where(same, p_in, p_out)
    upper = np.triu(rng.random((n_nodes, n_nodes)) < probs, k=1)
    adj_np = (upper | upper.T).astype(np.float32)

    # Features: a per-class mean signal (weak) buried in unit Gaussian noise.
    class_means = rng.standard_normal((n_classes, n_features)).astype(np.float32)
    noise = rng.standard_normal((n_nodes, n_features)).astype(np.float32)
    features_np = feature_snr * class_means[labels] + noise

    # Deterministic, class-balanced train/val/test masks.
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        idx = rng.permutation(idx)
        train_mask[idx[:train_per_class]] = True
        val_mask[idx[train_per_class : train_per_class + val_per_class]] = True
    test_mask = ~(train_mask | val_mask)

    return GraphDataset(
        adj=torch.from_numpy(adj_np),
        features=torch.from_numpy(features_np),
        labels=torch.from_numpy(labels.astype(np.int64)),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        source=(
            f"synthetic SBM (N={n_nodes}, C={n_classes}, "
            f"p_in={p_in}, p_out={p_out}, snr={feature_snr})"
        ),
    )


def normalized_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops: Â = D^{-1/2}(A+I)D^{-1/2}.

    This is the GCN propagation operator (Kipf & Welling, 2017). It is symmetric
    and its self-loops let a node keep its own signal during message passing.
    """
    n = adj.shape[0]
    a_hat = adj + torch.eye(n, dtype=adj.dtype, device=adj.device)
    deg = a_hat.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a_hat @ d_mat
