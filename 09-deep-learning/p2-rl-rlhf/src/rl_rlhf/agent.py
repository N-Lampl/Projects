"""A compact policy-gradient agent (REINFORCE with a moving-average baseline).

The policy is a small MLP over the gridworld's state features. Training is pure
torch on CPU and fully deterministic given a seed. ``train`` returns the
per-episode return curve so we can plot the agent beating a random policy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .envs import GridWorld
from .utils import set_seed


class PolicyMLP(nn.Module):
    """State-features -> action logits."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _rollout(
    env: GridWorld, policy: PolicyMLP, greedy: bool = False
) -> tuple[list[torch.Tensor], list[float]]:
    """Run one episode; return per-step log-probs and rewards."""
    state = env.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    done = False
    while not done:
        logits = policy(torch.from_numpy(state))
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if greedy else dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, done = env.step(int(action.item()))
        rewards.append(reward)
    return log_probs, rewards


def _discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    return torch.from_numpy(out)


def random_return(env: GridWorld, episodes: int = 200, seed: int = 0) -> float:
    """Average episode return of a uniform-random policy (the baseline line)."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(episodes):
        env.reset()
        done = False
        total = 0.0
        while not done:
            _, reward, done = env.step(int(rng.integers(env.n_actions)))
            total += reward
        totals.append(total)
    return float(np.mean(totals))


def train(
    env: GridWorld | None = None,
    episodes: int = 300,
    lr: float = 0.02,
    gamma: float = 0.99,
    hidden: int = 32,
    seed: int = 42,
) -> tuple[PolicyMLP, list[float]]:
    """REINFORCE with a baseline. Returns (trained policy, per-episode returns)."""
    set_seed(seed)
    if env is None:
        env = GridWorld()
    policy = PolicyMLP(env.state_dim, env.n_actions, hidden=hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    baseline = 0.0
    curve: list[float] = []
    for _ in range(episodes):
        log_probs, rewards = _rollout(env, policy)
        returns = _discounted_returns(rewards, gamma)
        ep_return = float(sum(rewards))
        curve.append(ep_return)

        baseline = 0.9 * baseline + 0.1 * float(returns[0])
        advantages = returns - baseline
        loss = -(torch.stack(log_probs) * advantages).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy, curve


def evaluate(env: GridWorld, policy: PolicyMLP, episodes: int = 100, greedy: bool = True) -> float:
    """Average return of a (by default greedy) policy over several episodes."""
    totals = []
    with torch.no_grad():
        for _ in range(episodes):
            _, rewards = _rollout(env, policy, greedy=greedy)
            totals.append(float(sum(rewards)))
    return float(np.mean(totals))
