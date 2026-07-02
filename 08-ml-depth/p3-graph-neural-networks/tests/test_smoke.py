"""Fast, offline, deterministic tests for the GNN project.

They build a seeded stochastic block model with planted communities and train a
from-scratch 2-layer GCN and a graph-blind MLP of identical shape. Because the
community labels are known, the tests assert real behaviour with no network: the
GCN beats the MLP and clears an absolute accuracy bar, the normalized adjacency is
symmetric, forward-pass shapes are right, and seeding is reproducible. The one real
Cora download is marked ``@slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnn import (
    GCN,
    MLP,
    make_sbm,
    normalized_adjacency,
    set_seed,
    train_gcn,
    train_mlp,
)
from gnn.data import load_cora


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)
    set_seed(123)
    c = np.random.randn(5)
    set_seed(123)
    d = np.random.randn(5)
    assert np.array_equal(c, d)


def test_normalized_adjacency_is_symmetric():
    ds = make_sbm(n_nodes=120, seed=1)
    a_hat = normalized_adjacency(ds.adj)
    assert a_hat.shape == (ds.n_nodes, ds.n_nodes)
    assert torch.allclose(a_hat, a_hat.t(), atol=1e-6)
    # Self-loops mean every diagonal entry is strictly positive.
    assert torch.all(torch.diagonal(a_hat) > 0)


def test_gcn_forward_shape():
    ds = make_sbm(n_nodes=100, n_classes=3, seed=2)
    a_hat = normalized_adjacency(ds.adj)
    model = GCN(ds.n_features, 16, ds.n_classes)
    out = model(ds.features, a_hat)
    assert out.shape == (ds.n_nodes, ds.n_classes)


def test_mlp_ignores_graph():
    ds = make_sbm(n_nodes=100, seed=3)
    model = MLP(ds.n_features, 16, ds.n_classes)
    model.eval()  # disable dropout so only the adjacency could change the output
    a_hat = normalized_adjacency(ds.adj)
    with torch.no_grad():
        out_with = model(ds.features, a_hat)
        out_without = model(ds.features, None)
    # The MLP must produce identical output regardless of the adjacency.
    assert torch.allclose(out_with, out_without)


def test_sbm_masks_are_disjoint_and_cover_all_nodes():
    ds = make_sbm(n_nodes=150, seed=4)
    total = ds.train_mask.int() + ds.val_mask.int() + ds.test_mask.int()
    assert torch.all(total == 1)  # every node in exactly one split
    assert ds.train_mask.sum() > 0 and ds.test_mask.sum() > 0


def test_training_is_reproducible():
    ds = make_sbm(n_nodes=150, seed=5)
    r1 = train_gcn(ds, epochs=80, seed=7)
    r2 = train_gcn(ds, epochs=80, seed=7)
    assert r1.test_acc == r2.test_acc


def test_gcn_beats_mlp_and_clears_bar():
    ds = make_sbm(n_nodes=300, n_classes=3, seed=42)
    gcn = train_gcn(ds, epochs=200, seed=42)
    mlp = train_mlp(ds, epochs=200, seed=42)
    assert gcn.test_acc > mlp.test_acc
    assert gcn.test_acc > 0.7


def test_embeddings_shape():
    ds = make_sbm(n_nodes=120, seed=6)
    gcn = train_gcn(ds, hidden=16, epochs=50, seed=6)
    assert gcn.embeddings.shape == (ds.n_nodes, 16)


@pytest.mark.slow
def test_gcn_beats_mlp_on_real_cora():
    """On the real Cora citation graph the GCN should beat the graph-blind MLP."""
    try:
        ds = load_cora(seed=42)
    except Exception as exc:  # any network / parse failure
        pytest.skip(f"Cora unavailable: {type(exc).__name__}: {exc}")

    gcn = train_gcn(ds, hidden=16, epochs=200, seed=42)
    mlp = train_mlp(ds, hidden=16, epochs=200, seed=42)
    assert gcn.test_acc > mlp.test_acc
    assert gcn.test_acc > 0.7
