# p3 · Graph neural networks — a 2-layer GCN from scratch on CPU

> **Synthetic-by-default, known communities.** Committed results come from a
> stochastic block model whose planted community labels are known, so predictions
> are *scored*, not asserted. `make run` regenerates them; `make test` runs offline
> in pure PyTorch (no torch-geometric). A `@slow` test repeats the win on the real
> Cora citation graph.

Some data is *relational*: papers cite papers, users follow users, transactions
touch shared accounts. A model that looks at each node in isolation throws away the
edges. A **graph convolutional network (GCN)** instead lets each node's prediction
depend on its neighbours by averaging their features through the graph — "message
passing". This project implements a **2-layer GCN from scratch in pure PyTorch**
(no torch-geometric, which is painful CPU-only) and pits it against a **graph-blind
MLP of identical shape**. The only difference between the two is whether the
normalized adjacency `Â` is applied between layers, so any accuracy gap is
attributable to the graph structure — nothing else. CPU-only, fully deterministic.

**Authorized use only.** Synthetic data used for education. See
[../../ETHICS.md](../../ETHICS.md).

## The idea

The GCN ([`model.py`](src/gnn/model.py)) is the classic Kipf & Welling layer:

```
Â  = D^{-1/2}(A + I)D^{-1/2}       # symmetric-normalized adjacency w/ self-loops
H1  = relu(Â X W0)                 # aggregate neighbours, transform, non-linearity
out = Â H1 W1                      # a second hop of message passing -> class logits
```

Applying `Â` mixes each node's representation with its neighbours' before every
linear map, so after two layers a node "sees" its 2-hop neighbourhood. The MLP
baseline is `relu(X W0)` then `X W1` with the **same** widths — identical capacity,
just no `Â`.

The data ([`graph.py`](src/gnn/graph.py)) is a **stochastic block model** with
planted communities: same-community nodes connect with probability `p_in`,
cross-community nodes with a much smaller `p_out`. Crucially, the node features are
only **weakly** correlated with the community (a small per-class signal buried in
Gaussian noise), so features alone can't solve the task — the community structure
lives mostly in the *edges*. That is exactly the regime where message passing pays.
Because the community labels are known, every prediction is scored against ground
truth on held-out test nodes.

## Run it

```bash
make run     # train GCN + graph-blind MLP on an SBM -> figures + metrics.json
make test    # fast offline smoke tests (-m 'not slow'); pure PyTorch, no network
make run ARGS='--nodes 400 --epochs 300'
```

Outputs land in [results/](results/):
- `figures/accuracy_gcn_vs_mlp.png` — the **money plot**: GCN vs graph-blind MLP
  test accuracy side by side.
- `figures/embedding_tsne.png` — a t-SNE of the GCN's hidden embeddings, colored by
  true community; message passing pulls each community into its own cluster.
- `metrics.json` — both accuracies, graph size, and the summary.

## What the result shows

On a synthetic SBM (300 nodes, ~1.8k edges, 3 planted communities) where the node
features are deliberately weak:

| model | uses edges? | test accuracy |
|---|---|---|
| MLP (graph-blind) | no | 0.72 |
| **GCN (message passing)** | **yes** | **0.95** |

The GCN lifts test accuracy by **23 points** over an MLP of identical shape — the
*only* difference is whether it propagates over `Â`. The graph-blind MLP is capped
at 0.72 because the features alone are barely informative; the GCN recovers the
missing signal from the community structure in the edges. The t-SNE plot makes it
visual: the GCN's hidden embeddings separate cleanly into one blob per community,
even though the raw features overlap heavily. Everything is deterministic under a
fixed seed, and a `@slow` test reproduces the same qualitative win (GCN > MLP,
> 0.7) on the real **Cora** citation graph.

## Interview story (3 sentences)

> I implemented a 2-layer graph convolutional network from scratch in pure PyTorch —
> just `Â X W` with the symmetric-normalized adjacency, no torch-geometric — and
> benchmarked it against a graph-blind MLP of identical shape so the *only* variable
> is whether the model uses the edges. On a stochastic block model with weak node
> features the GCN hit 0.95 test accuracy versus 0.72 for the MLP, a 23-point lift
> that comes entirely from message passing, and a t-SNE of the hidden embeddings
> shows the communities separating cleanly. It shows I understand what a GCN
> actually computes and *why* propagating over the graph helps — not just how to
> call a library layer — and a `@slow` test confirms the same result on real Cora.

## Layout

```
src/gnn/       utils · graph (SBM generator · normalized adjacency Â)
               model (2-layer GCN + graph-blind MLP baseline)
               train (deterministic Adam loop -> test acc + embeddings)
               data (optional real Cora download) · plots
scripts/       run_analysis.py  -> results/figures + metrics.json
tests/         test_smoke.py  (offline pure-PyTorch SBM; @slow real-Cora check)
results/       figures/*.png + metrics.json  (committed)
data/ models/  git-ignored (synthetic graph; models trained in memory)
```

## References

- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional
  Networks* — the GCN layer `Â = D^{-1/2}(A+I)D^{-1/2}` this implements.
- Holland, Laskey & Leinhardt (1983) — the stochastic block model.
- Sen et al. (2008) — the Cora citation-network benchmark used by the `@slow` test.
