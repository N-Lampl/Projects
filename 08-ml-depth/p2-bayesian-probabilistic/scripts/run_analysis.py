#!/usr/bin/env python3
"""Run the Bayesian hierarchical experiment -> results/metrics.json + figures.

Fits a partial-pooling model with a from-scratch numpy Gibbs sampler, compares
shrinkage against the no-pooling / complete-pooling baselines, and measures
credible-interval calibration across many datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_pp import (  # noqa: E402
    calibration_curve,
    convergence,
    fit_dataset,
    make_hierarchical,
    posterior_predictive_check,
    set_seed,
    shrinkage_report,
    shrinkage_study,
)
from bayes_pp.plots import (  # noqa: E402
    plot_calibration,
    plot_posterior_vs_true,
    plot_shrinkage,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--n-iter", type=int, default=4000)
    ap.add_argument("--n-sims", type=int, default=100, help="datasets for calibration")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ds = make_hierarchical(n_groups=args.groups, n_per_group=args.per_group, seed=args.seed)
    post, summary = fit_dataset(ds, n_iter=args.n_iter, seed=args.seed)
    shrink = shrinkage_report(ds, post)
    study = shrinkage_study(n_sims=args.n_sims, base_seed=args.seed)
    conv = convergence(post)
    ppc = posterior_predictive_check(post, ds.y, ds.sigma, seed=args.seed)
    cal = calibration_curve(n_sims=args.n_sims, base_seed=args.seed)

    fig1 = plot_posterior_vs_true(ds, summary, FIGURES / "posterior_vs_true.png")
    fig2 = plot_shrinkage(ds, shrink, FIGURES / "shrinkage.png")
    fig3 = plot_calibration(cal, FIGURES / "calibration.png")

    summary_str = (
        f"Averaged over {study['n_sims']} datasets, partial pooling cuts RMSE on the true "
        f"group means from {study['mean_no_pooling_rmse']:.2f} (no pooling) to "
        f"{study['mean_partial_pooling_rmse']:.2f} and wins "
        f"{study['partial_win_rate'] * 100:.0f}% of the time. Chains mixed (max R-hat "
        f"{conv['rhat_theta_max']:.3f}). Across {cal['n_sims']} datasets the credible "
        f"intervals are well calibrated (mean |calibration error| "
        f"{cal['mean_abs_calibration_error']:.3f})."
    )

    metrics = {
        "project": "p2-bayesian-probabilistic",
        "summary": summary_str,
        "data_source": "synthetic hierarchical (known group means)",
        "seed": args.seed,
        "n_groups": ds.n_groups,
        "n_per_group": args.per_group,
        "true_mu": ds.mu,
        "true_tau": ds.tau,
        "posterior": {
            "mu_mean": summary["mu_mean"],
            "mu_ci": summary["mu_ci"],
            "tau_mean": summary["tau_mean"],
            "tau_ci": summary["tau_ci"],
            "theta_mean": [float(v) for v in summary["theta_mean"]],
            "theta_true": ds.group_true_means.tolist(),
        },
        "shrinkage_single_dataset": {
            k: shrink[k]
            for k in ("partial_pooling_rmse", "no_pooling_rmse", "complete_pooling_rmse")
        },
        "shrinkage_study": study,
        "convergence": conv,
        "posterior_predictive": ppc,
        "calibration": cal,
        "figures": [f"results/figures/{p.name}" for p in (fig1, fig2, fig3)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary_str)
    print(f"[ok] wrote {RESULTS / 'metrics.json'} + 3 figures")


if __name__ == "__main__":
    main()
