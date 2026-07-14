# Data

This project runs on a **synthetic stochastic block model (SBM)** by default, so
tests and CI need no network. Nothing here is committed (`data/` is git-ignored
except this README).

## Default: synthetic SBM (offline, known communities)

[`../src/gnn/graph.py`](../src/gnn/graph.py) draws an SBM with planted communities:
nodes in the same community connect with probability `p_in`, nodes in different
communities with a much smaller `p_out`. Each node also gets a feature vector that
is **only weakly correlated** with its community (a small per-class signal buried in
Gaussian noise). Because the true community labels are stored on the dataset, every
prediction is scored against ground truth. The weak features are deliberate: a
graph-blind model struggles, so message passing over the edges has something real to
fix - that is the whole point of the GCN-vs-MLP comparison.

## Optional: real Cora citation graph

The `@slow` test fits the *same* models on the standard **Cora** citation network
(2708 papers, 7 classes, bag-of-words features), downloaded from a stable mirror by
[`load_cora`](../src/gnn/data.py). It raises on any network/parse failure and the
test skips; the default path needs only numpy/torch/scikit-learn.

> Authorized use only: synthetic data used for education. See
> [../../../ETHICS.md](../../../ETHICS.md).
