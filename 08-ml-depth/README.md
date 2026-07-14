# 08 · ML Depth

The security tracks are the pivot; tracks 07-09 are the other half of the
story - the **ML/data-science depth** the security work is built on. This track
carries a classical-but-deep method every strong ML practitioner should be able to
implement and defend, not just call: **graph neural networks**.

Same engineering bar as everywhere else: the project is self-contained, has a
reproducible `make run`, commits figures + `metrics.json`, ships an offline
synthetic fallback, and runs **CPU-only**. No forced security framing - an
honest ML depth piece, scored against a **known ground truth**.

Authorized use only - see [../ETHICS.md](../ETHICS.md). Datasets and model weights
are not committed; the default path is synthetic and offline.

## Project

| Project | What it does | Ground truth |
|---|---|---|
| `p3-graph-neural-networks/` | A **pure-PyTorch 2-layer GCN** for node classification vs a graph-blind MLP baseline, proving message passing over the graph helps | planted communities (synthetic SBM; real Cora `@slow`) |

## Notes

- **Known-answer by design.** The project generates synthetic data whose true
  parameters are known, so estimates are *scored* (RMSE, accuracy) rather than
  asserted. Real datasets (Cora) live behind `@pytest.mark.slow` and never run in CI.
- **From scratch where it teaches.** The GCN layer is implemented directly (plain
  PyTorch) - no torch-geometric - because the point is understanding the mechanics.
- **CPU-only.** Small models and data subsets; the whole track runs on a no-GPU
  laptop.
