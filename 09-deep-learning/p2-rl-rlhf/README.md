# p2 · Reinforcement learning + RLHF — policy gradients and a reward model from scratch on CPU

> **Synthetic-by-default, ground truth known.** Committed results come from (A) a
> numpy gridworld MDP where the agent's return is *scored* against a random policy,
> and (B) a bandit world with a **hidden true reward** so the RLHF reward model and
> policy are scored against ground truth. `make run` regenerates them; `make test`
> runs offline with no gymnasium. A `@slow` test cross-checks the policy gradient on
> gymnasium's `CartPole-v1`.

RLHF is the alignment recipe behind instruction-tuned LLMs, but the mechanics are
plain RL: you don't have the reward function, so you **learn one from human
preferences** and then optimise a policy against it. This project builds both
halves from scratch on CPU. Part A is a **REINFORCE policy-gradient agent** (a torch
MLP with a baseline) that learns to solve a self-contained numpy gridworld. Part B
is a **minimal RLHF pipeline**: sample preference pairs from a *hidden* true reward
via Bradley-Terry, fit a **reward model** on those preferences with the BT/logistic
loss, optimise a policy against the *learned* reward, and measure its **win-rate
against the base policy under the true reward**. Fully deterministic, no gym on the
fast path.

**Authorized use only.** Synthetic data used for education. See
[../../ETHICS.md](../../ETHICS.md).

## The idea

**Part A — policy gradient ([`agent.py`](src/rl_rlhf/agent.py)).** The gridworld
([`envs.py`](src/rl_rlhf/envs.py)) is an `NxN` MDP: each step costs a small penalty,
reaching the goal pays a terminal reward and ends the episode, so shorter paths
score higher. The policy is a small MLP over normalised state features, trained with
**REINFORCE + a moving-average baseline**:

```
maximise  E[ sum_t log pi(a_t | s_t) * (G_t - b) ]
```

where `G_t` is the discounted return and `b` is a baseline that cuts gradient
variance. Given a seed the whole thing is reproducible.

**Part B — RLHF-lite ([`rlhf.py`](src/rl_rlhf/rlhf.py)).** A contextual-bandit world
has a hidden reward `r*(context, action)` that we never observe. Humans only give
**preferences**, modelled with **Bradley-Terry**:

```
P(a preferred over b) = sigma( beta * (r*_a - r*_b) )
```

We fit a reward model `r_hat` by minimising the BT/logistic loss
`-log sigma(r_hat_win - r_hat_lose)` (a BCE on the reward difference), then optimise
a softmax policy against `r_hat`. The honest test is the one a real RLHF system can't
run: does the learned-reward policy actually win under the **true** reward?

## Run it

```bash
make run     # PG agent + RLHF pipeline -> figures + metrics.json
make test    # fast offline smoke tests (-m 'not slow'); no gymnasium needed
make run ARGS='--episodes 600 --n-pairs 6000 --seed 1'
```

Outputs land in [results/](results/):
- `figures/training_return.png` — the PG agent's per-episode return climbing past
  the random-policy baseline line.
- `figures/rlhf_winrate.png` — RLHF vs base win-rate under the true reward, plus
  average true reward (base / RLHF / optimal).
- `figures/reward_model_fit.png` — learned vs true reward scatter with held-out
  preference accuracy.
- `metrics.json` — returns, reward-model accuracy/correlation, win-rate.

## What the result shows

Seed 42, 5x5 gridworld, 4000 preference pairs:

| metric | value |
|---|---|
| PG agent return (greedy) | **0.60** |
| random policy return | −1.98 |
| reward-model held-out preference accuracy | **0.95** |
| reward-model correlation with true reward | 0.82 |
| RLHF win-rate vs base (under TRUE reward) | **0.83** |

The policy gradient turns a −1.98 random return into **+0.60** — it stops wandering
and walks the near-shortest path to the goal. On the RLHF side, the reward model
sees *only* noisy Bradley-Terry preference labels yet recovers the hidden reward's
ordering to **95% held-out accuracy** (0.82 rank correlation). Optimising a policy
against that *learned* reward and then judging it by the **true** reward it never saw,
it beats the uniform base policy on **83% of contexts** and reaches the optimal
average true reward — the whole RLHF premise (a good enough reward model is a usable
optimisation target) holds up on a problem where we can actually check it.

## Interview story (3 sentences)

> I built both halves of RLHF from scratch on CPU: a REINFORCE policy-gradient agent
> with a baseline that solves a numpy gridworld (return −1.98 random → +0.60), and a
> preference pipeline that learns a reward model from Bradley-Terry labels over a
> *hidden* true reward, then optimises a policy against the learned reward. The
> honest check is the one production RLHF can't run — judging the learned-reward
> policy by the true reward — and it wins on 83% of contexts with a reward model at
> 95% held-out preference accuracy. It shows I understand policy gradients, the
> Bradley-Terry preference loss, and *why* reward-model quality is the thing that
> makes or breaks RLHF — not just how to call a library.

## Layout

```
src/rl_rlhf/   utils (seed numpy+torch) · envs (numpy gridworld MDP)
               agent (REINFORCE + baseline policy-gradient MLP)
               rlhf (BT preferences · reward model · policy optimisation) · plots
scripts/       run_analysis.py  -> results/figures + metrics.json
tests/         test_smoke.py  (offline torch PG + RLHF; @slow gymnasium CartPole)
results/       figures/*.png + metrics.json  (committed)
data/ models/  git-ignored (synthetic env + preferences; models trained in memory)
```

## References

- Williams (1992) — REINFORCE, the likelihood-ratio policy gradient with a baseline.
- Sutton & Barto, *Reinforcement Learning: An Introduction* — MDPs, returns, and
  policy-gradient methods.
- Christiano et al. (2017), *Deep RL from Human Preferences* — learning a reward
  model from pairwise preferences and optimising a policy against it.
- Ouyang et al. (2022), *InstructGPT* — the RLHF recipe (reward model + PPO) applied
  to language models; Bradley-Terry (1952) is the preference model it uses.
