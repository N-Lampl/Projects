# 08 · ML Depth

The security tracks (00-06) are the pivot; tracks 07-09 are the other half of the
story — the **ML/data-science depth** the security work is built on. This track
collects three classical-but-deep methods that every strong ML practitioner should
be able to implement and defend, not just call: **causal inference**, **Bayesian
hierarchical modeling**, and **graph neural networks**.

Same engineering bar as everywhere else: each project is self-contained, has a
reproducible `make run`, commits figures + `metrics.json`, ships an offline
synthetic fallback, and runs **CPU-only**. No forced security framing — these are
honest ML depth pieces, each scored against a **known ground truth**.

Authorized use only — see [../ETHICS.md](../ETHICS.md). Datasets and model weights
are not committed; the default path is synthetic and offline.

## Projects

| Project | What it does | Ground truth |
|---|---|---|
| `p1-causal-inference/` | Estimate an average treatment effect four ways — naive, regression adjustment, IPW, doubly-robust AIPW — on a confounded SCM; measure bias **and** confidence-interval coverage | known ATE (synthetic SCM; real IHDP `@slow`) |
| `p2-bayesian-probabilistic/` | Hierarchical partial-pooling model via a **from-scratch numpy Gibbs sampler**; shrinkage vs baselines, R-hat convergence, and credible-interval **calibration** | known group means (synthetic; PyMC cross-check `@slow`) |
| `p3-graph-neural-networks/` | A **pure-PyTorch 2-layer GCN** for node classification vs a graph-blind MLP baseline, proving message passing over the graph helps | planted communities (synthetic SBM; real Cora `@slow`) |

## Notes

- **Known-answer by design.** Every project generates synthetic data whose true
  parameters are known, so estimates are *scored* (bias, coverage, RMSE,
  calibration, accuracy) rather than asserted. Real datasets (IHDP, Cora) and
  optional libraries (PyMC) live behind `@pytest.mark.slow` and never run in CI.
- **From scratch where it teaches.** The causal estimators, the Gibbs sampler, and
  the GCN layer are implemented directly (numpy / plain PyTorch) — no DoWhy, no
  PyMC-on-the-fast-path, no torch-geometric — because the point is understanding
  the mechanics.
- **CPU-only.** Small models and data subsets; the whole track runs on a no-GPU
  laptop.
