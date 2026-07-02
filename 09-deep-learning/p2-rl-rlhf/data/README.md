# Data

This project runs on **synthetic data** by default, so tests and CI need no
network. Nothing here is committed (`data/` is git-ignored except this README).

## Part A: synthetic gridworld MDP (offline)

[`../src/rl_rlhf/envs.py`](../src/rl_rlhf/envs.py) is a self-contained `NxN`
gridworld written in pure numpy — no gym. The agent starts in a corner and must
reach the goal; each step costs a small penalty and the goal pays a terminal
reward, so a shorter path scores higher. Because the dynamics are known and
deterministic, the policy-gradient agent's return is *scored* against a random
policy with no ambiguity.

## Part B: synthetic Bradley-Terry preferences (offline)

[`../src/rl_rlhf/rlhf.py`](../src/rl_rlhf/rlhf.py) builds a small contextual-bandit
world with a **hidden true reward** `r*(context, action)`. Preference pairs are
labelled by the **Bradley-Terry** model on that hidden reward, so labels are noisy
but informative — exactly the signal a real RLHF pipeline sees. The true reward is
stored on the world, so both the reward model's accuracy and the RLHF policy's
win-rate are checked against ground truth.

## Optional: gymnasium CartPole cross-check

The `@slow` test lazily imports **gymnasium**, trains a brief policy gradient on
`CartPole-v1`, and checks it beats a random policy. gymnasium is optional (`pip
install gymnasium`); the default path needs only numpy/torch/matplotlib.

> Authorized use only: synthetic data used for education. See
> [../../../ETHICS.md](../../../ETHICS.md).
