# 09 · Deep Learning

The other half of the portfolio (alongside 07-08): **modern deep-learning depth**.
Three topics that sit at the center of how contemporary neural networks are
understood, aligned, and shipped — **transformer internals & mechanistic
interpretability**, **reinforcement learning / RLHF**, and **model compression &
efficient inference**.

Same engineering bar as the rest of the monorepo: each project is self-contained,
has a reproducible `make run`, commits figures + `metrics.json`, ships an offline
synthetic fallback, and runs **CPU-only** (tiny models, short training, few steps).
No forced security framing — honest DL depth pieces.

Authorized use only — see [../ETHICS.md](../ETHICS.md). Model weights are not
committed; the default path trains tiny models in memory with no downloads.

## Projects

| Project | What it does | Runs on |
|---|---|---|
| `p1-transformer-interp/` | Train a **tiny transformer from scratch** until an **induction head** forms, then reverse-engineer it: attention inspection, logit lens, activation patching | synthetic induction task (distilgpt2 `@slow`) |
| `p2-rl-rlhf/` | A policy-gradient agent on a **numpy gridworld** (no gym), plus **RLHF-lite**: learn a reward model from preferences, optimize a policy against it, score win-rate under the true reward | self-contained env + synthetic preferences (CartPole `@slow`) |
| `p3-model-compression/` | Prune, **quantize**, and **distill** a trained network; measure the accuracy vs size vs CPU-latency **Pareto frontier** | synthetic classification (MNIST `@slow`) |

## Notes

- **Tiny by design.** Everything trains on CPU in seconds-to-minutes — a 2-layer
  transformer on a toy induction task, a small policy net on a gridworld, a compact
  MLP for compression. The mechanisms (induction heads, preference-based reward
  learning, quantization trade-offs) are the point, not scale.
- **From scratch where it teaches.** The transformer and its interpretability
  hooks, the gridworld MDP and policy-gradient loop, and the pruning/quantization/
  distillation passes are implemented directly in PyTorch.
- **Offline-first.** Real assets (distilgpt2, gymnasium CartPole, MNIST) are used
  only by `@pytest.mark.slow` tests that skip cleanly when the optional library or
  network is unavailable; CI never touches them.
