#!/usr/bin/env python3
"""Run the causal-inference experiment -> results/metrics.json + figures.

Generates a confounded SCM with a known ATE, estimates it four ways, measures
confidence-interval coverage over many seeds, and checks covariate balance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_inference import (  # noqa: E402
    balance_table,
    coverage_study,
    point_estimates,
    set_seed,
)
from causal_inference.plots import plot_ate_by_method, plot_covariate_balance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="samples per dataset")
    ap.add_argument("--p", type=int, default=5, help="number of confounders")
    ap.add_argument("--tau", type=float, default=2.0, help="true treatment effect (ATE)")
    ap.add_argument("--confounding", type=float, default=1.5, help="confounding strength")
    ap.add_argument("--n-sims", type=int, default=200, help="datasets for CI coverage")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    kw = dict(n=args.n, p=args.p, tau=args.tau, confounding=args.confounding)

    point = point_estimates(seed=args.seed, **kw)
    coverage = coverage_study(n_sims=args.n_sims, base_seed=args.seed, **kw)
    balance = balance_table(seed=args.seed, **kw)

    fig1 = plot_ate_by_method(point, point["aipw_se"], FIGURES / "ate_by_method.png")
    fig2 = plot_covariate_balance(balance, FIGURES / "covariate_balance.png")

    est = point["estimates"]
    summary = (
        f"True ATE {point['true_ate']:.2f}. Naive (confounded) = {est['naive']:.2f} "
        f"(bias {point['bias']['naive']:+.2f}); doubly-robust AIPW = {est['aipw']:.2f} "
        f"(bias {point['bias']['aipw']:+.2f}). AIPW 95% CI covers the truth "
        f"{coverage['aipw_coverage'] * 100:.0f}% of the time vs "
        f"{coverage['naive_coverage'] * 100:.0f}% for naive. IPW weighting cuts mean |SMD| "
        f"from {balance['mean_abs_smd_before']:.2f} to {balance['mean_abs_smd_after']:.2f}."
    )

    metrics = {
        "project": "p1-causal-inference",
        "summary": summary,
        "data_source": "synthetic SCM (known ATE)",
        "seed": args.seed,
        "n": args.n,
        "p": args.p,
        "true_ate": point["true_ate"],
        "estimates": est,
        "bias": point["bias"],
        "aipw_se": point["aipw_se"],
        "aipw_ci": point["aipw_ci"],
        "coverage": coverage,
        "balance": balance,
        "figures": [f"results/figures/{p.name}" for p in (fig1, fig2)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary)
    print(f"[ok] wrote {RESULTS / 'metrics.json'} + {fig1.name}, {fig2.name}")


if __name__ == "__main__":
    main()
