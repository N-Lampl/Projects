"""Graph neural networks: a from-scratch 2-layer GCN vs a graph-blind MLP (CPU)."""

from __future__ import annotations

from .data import load_cora
from .graph import GraphDataset, make_sbm, normalized_adjacency
from .model import GCN, MLP
from .train import TrainResult, accuracy, train_gcn, train_mlp, train_model
from .utils import get_device, set_seed

__all__ = [
    "GCN",
    "MLP",
    "GraphDataset",
    "TrainResult",
    "accuracy",
    "get_device",
    "load_cora",
    "make_sbm",
    "normalized_adjacency",
    "set_seed",
    "train_gcn",
    "train_mlp",
    "train_model",
]
