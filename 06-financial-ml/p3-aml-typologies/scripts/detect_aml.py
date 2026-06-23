#!/usr/bin/env python3
"""Generate the synthetic AML graph, build features, run both detectors, and write
figures + metrics.json. Run via `make detect`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml_typologies import (  # noqa: E402
    HAVE_NETWORKX,
    build_features,
    evaluate,
    feature_importances,
    generate_aml_graph,
    pr_curve,
    score_isolation_forest,
    score_rules_rf,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_pr(scores_by_name: dict, labels, ap_by_name: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"isolation_forest": "#7f8c8d", "rules_rf": "#c0392b"}
    for name, scores in scores_by_name.items():
        precision, recall = pr_curve(scores, labels)
        ax.plot(
            recall,
            precision,
            "-",
            color=colors.get(name, None),
            linewidth=2,
            label=f"{name} (PR-AUC={ap_by_name[name]:.3f})",
        )
    ax.axhline(labels.mean(), color="black", ls=":", lw=1, label=f"prevalence={labels.mean():.3f}")
    ax.set_xlabel("recall (fraction of laundering accounts caught)")
    ax.set_ylabel("precision (fraction of alerts that are real)")
    ax.set_title("AML detectors: precision-recall (rare-class metric)", pad=10)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "pr_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_importances(importances) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp = importances.iloc[::-1]
    ax.barh(imp.index, imp.values, color="#2980b9")
    ax.set_xlabel("RandomForest feature importance")
    ax.set_title("What flags a laundering account", pad=10)
    fig.tight_layout()
    out = FIG_DIR / "feature_importances.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_ring(graph) -> Path:
    """Visualize one detected layering ring; networkx layout if present else heatmap."""
    out = FIG_DIR / "laundering_ring.png"
    if HAVE_NETWORKX and graph.rings:
        import networkx as nx

        ring = graph.rings[0]
        tx = graph.transactions
        sub = tx[tx["src"].isin(ring) & tx["dst"].isin(ring)]
        g = nx.DiGraph()
        g.add_nodes_from(ring)
        for s, d, a in zip(sub["src"], sub["dst"], sub["amount"], strict=True):
            g.add_edge(int(s), int(d), amount=a)
        pos = nx.circular_layout(g)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        nx.draw_networkx_nodes(g, pos, node_color="#c0392b", node_size=1100, ax=ax)
        nx.draw_networkx_labels(g, pos, font_color="white", font_size=9, ax=ax)
        nx.draw_networkx_edges(
            g, pos, ax=ax, arrowstyle="-|>", arrowsize=18, width=2, connectionstyle="arc3,rad=0.12"
        )
        elabels = {(u, v): f"${d['amount']:,.0f}" for u, v, d in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=elabels, font_size=7, ax=ax)
        ax.set_title("A detected LAYERING ring (funds cycle through intermediaries)", pad=10)
        ax.axis("off")
    else:
        # fallback: adjacency heatmap of the busiest accounts
        tx = graph.transactions
        top = tx["src"].value_counts().head(25).index.tolist()
        order = {a: i for i, a in enumerate(top)}
        mat = np.zeros((len(top), len(top)))
        sub = tx[tx["src"].isin(top) & tx["dst"].isin(top)]
        for s, d in zip(sub["src"], sub["dst"], strict=True):
            mat[order[s], order[d]] += 1
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(mat, cmap="magma")
        fig.colorbar(im, ax=ax, label="# transfers")
        ax.set_title("Adjacency heatmap of busiest accounts (networkx absent)", pad=10)
        ax.set_xlabel("destination")
        ax.set_ylabel("source")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=1500)
    ap.add_argument(
        "--fpr-budget", type=float, default=0.01, help="false-positive budget for recall"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"generating synthetic AML graph (networkx={'on' if HAVE_NETWORKX else 'off'})...")
    graph = generate_aml_graph(n_accounts=args.accounts, seed=args.seed)
    print(
        f"  {graph.n_accounts} accounts, {len(graph.transactions)} transfers, "
        f"{int(graph.labels.sum())} suspicious accounts, {len(graph.rings)} layering rings"
    )

    feats = build_features(graph)
    labels = graph.labels.to_numpy()

    print("scoring detectors...")
    s_if = score_isolation_forest(feats, seed=args.seed)
    s_rf = score_rules_rf(feats, graph.labels, seed=args.seed)

    eval_if = evaluate(s_if, labels, fpr_budget=args.fpr_budget)
    eval_rf = evaluate(s_rf, labels, fpr_budget=args.fpr_budget)
    for name, ev in (("isolation_forest", eval_if), ("rules_rf", eval_rf)):
        print(
            f"  {name:16s} PR-AUC={ev['pr_auc']:.3f} ROC-AUC={ev['roc_auc']:.3f} "
            f"recall@{args.fpr_budget:.0%}FPR={ev['recall_at_budget']:.3f} "
            f"p@100={ev['precision_at_k']['p@100']:.3f}"
        )

    importances = feature_importances(feats, graph.labels, seed=args.seed)

    fig_pr = _plot_pr(
        {"isolation_forest": s_if, "rules_rf": s_rf},
        labels,
        {"isolation_forest": eval_if["pr_auc"], "rules_rf": eval_rf["pr_auc"]},
    )
    fig_imp = _plot_importances(importances)
    fig_ring = _plot_ring(graph)

    summary = (
        f"Rules+RF detects AML typologies on a synthetic transaction graph: "
        f"PR-AUC={eval_rf['pr_auc']:.3f}, ROC-AUC={eval_rf['roc_auc']:.3f}; at a "
        f"{args.fpr_budget:.0%} false-positive budget it catches "
        f"{eval_rf['recall_at_budget']:.0%} of laundering accounts at "
        f"{eval_rf['precision_at_budget']:.0%} alert precision. Unsupervised "
        f"IsolationForest baseline PR-AUC={eval_if['pr_auc']:.3f}."
    )

    metrics = {
        "project": "p3-aml-typologies",
        "summary": summary,
        "source": "synthetic",
        "networkx_available": HAVE_NETWORKX,
        "seed": args.seed,
        "n_accounts": graph.n_accounts,
        "n_transfers": int(len(graph.transactions)),
        "n_suspicious": int(graph.labels.sum()),
        "n_layering_rings": len(graph.rings),
        "prevalence": eval_rf["prevalence"],
        "fpr_budget": args.fpr_budget,
        "pr_auc": eval_rf["pr_auc"],
        "roc_auc": eval_rf["roc_auc"],
        "ks_statistic": eval_rf["ks_statistic"],
        "operating_threshold": eval_rf["operating_threshold"],
        "recall_at_budget": eval_rf["recall_at_budget"],
        "precision_at_budget": eval_rf["precision_at_budget"],
        "achieved_fpr": eval_rf["achieved_fpr"],
        "precision_at_k": eval_rf["precision_at_k"],
        "confusion_at_budget": eval_rf["confusion_at_budget"],
        "detectors": {"rules_rf": eval_rf, "isolation_forest": eval_if},
        "top_features": importances.head(5).round(4).to_dict(),
        "figures": [
            str(fig_pr.relative_to(PROJECT)),
            str(fig_imp.relative_to(PROJECT)),
            str(fig_ring.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {fig_pr.relative_to(PROJECT)}")
    print(f"wrote {fig_imp.relative_to(PROJECT)}")
    print(f"wrote {fig_ring.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
