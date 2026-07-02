#!/usr/bin/env python3
"""Run the GNN experiment -> results/metrics.json + figures.

Builds a synthetic stochastic block model with planted communities, trains a
from-scratch 2-layer GCN and a graph-blind MLP baseline of identical shape, and
shows message passing over the edges beats the baseline on held-out nodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn import (  # noqa: E402
    make_sbm,
    set_seed,
    train_gcn,
    train_mlp,
)
from gnn.plots import plot_accuracy, plot_tsne  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=300)
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ds = make_sbm(n_nodes=args.nodes, n_classes=args.classes, seed=args.seed)
    gcn = train_gcn(ds, epochs=args.epochs, seed=args.seed)
    mlp = train_mlp(ds, epochs=args.epochs, seed=args.seed)

    fig1 = plot_accuracy(gcn.test_acc, mlp.test_acc, FIGURES / "accuracy_gcn_vs_mlp.png")
    fig2 = plot_tsne(gcn.embeddings, ds.labels, FIGURES / "embedding_tsne.png", seed=args.seed)

    lift = gcn.test_acc - mlp.test_acc
    summary_str = (
        f"On a synthetic SBM ({ds.n_nodes} nodes, {ds.n_edges} edges, "
        f"{ds.n_classes} planted communities) where node features alone are only "
        f"weakly informative, a 2-layer GCN reaches {gcn.test_acc:.2f} test accuracy "
        f"versus {mlp.test_acc:.2f} for a graph-blind MLP of identical shape — a "
        f"{lift * 100:.0f}-point lift from message passing over the graph structure."
    )

    metrics = {
        "project": "p3-graph-neural-networks",
        "summary": summary_str,
        "data_source": ds.source,
        "seed": args.seed,
        "gcn_test_acc": gcn.test_acc,
        "mlp_test_acc": mlp.test_acc,
        "n_nodes": ds.n_nodes,
        "n_edges": ds.n_edges,
        "n_classes": ds.n_classes,
        "figures": [f"results/figures/{p.name}" for p in (fig1, fig2)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary_str)
    print(f"[ok] wrote {RESULTS / 'metrics.json'} + 2 figures")


if __name__ == "__main__":
    main()
