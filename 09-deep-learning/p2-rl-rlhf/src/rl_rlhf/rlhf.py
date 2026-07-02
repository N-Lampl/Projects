"""RLHF-lite: learn a reward model from preferences, then optimise a policy on it.

The setup is a small contextual-bandit world. A *hidden* true reward
``r*(context, action)`` scores every (context, action) pair; humans never reveal
it directly. Instead we:

1. sample pairs of (context, action) and label which one a human prefers via the
   **Bradley-Terry** model on the hidden reward (higher true reward -> more likely
   preferred), so labels are noisy but informative;
2. fit a **reward model** (small MLP) to those preferences with the BT / logistic
   loss ``-log sigma(r_hat_win - r_hat_lose)``;
3. optimise a policy against the *learned* reward (softmax / REINFORCE-style
   contextual bandit);
4. evaluate the RLHF policy's **win-rate vs the base policy under the TRUE reward**.

Everything is numpy/torch, CPU-only, deterministic. Nothing here needs gym.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .utils import set_seed


@dataclass
class RLHFWorld:
    """A hidden linear-ish reward over one-hot contexts x one-hot actions."""

    n_contexts: int
    n_actions: int
    feat_dim: int
    context_feats: np.ndarray  # (n_contexts, feat_dim)
    true_w: np.ndarray  # (feat_dim, n_actions)

    def true_reward(self, context: int, action: int) -> float:
        return float(self.context_feats[context] @ self.true_w[:, action])

    def true_reward_matrix(self) -> np.ndarray:
        return self.context_feats @ self.true_w  # (n_contexts, n_actions)


def make_world(
    n_contexts: int = 12, n_actions: int = 5, feat_dim: int = 8, seed: int = 0
) -> RLHFWorld:
    """Build a bandit world with a hidden, non-trivial true reward function."""
    rng = np.random.default_rng(seed)
    context_feats = rng.standard_normal((n_contexts, feat_dim)).astype(np.float32)
    true_w = rng.standard_normal((feat_dim, n_actions)).astype(np.float32)
    return RLHFWorld(n_contexts, n_actions, feat_dim, context_feats, true_w)


def sample_preferences(
    world: RLHFWorld, n_pairs: int = 4000, beta: float = 4.0, seed: int = 0
) -> dict[str, np.ndarray]:
    """Sample (context, action_a, action_b, label) with Bradley-Terry labels.

    ``label == 1`` means action_a was preferred. Preference probability is
    ``sigma(beta * (r*_a - r*_b))`` — higher ``beta`` = less label noise.
    """
    rng = np.random.default_rng(seed)
    ctx = rng.integers(world.n_contexts, size=n_pairs)
    a = rng.integers(world.n_actions, size=n_pairs)
    b = rng.integers(world.n_actions, size=n_pairs)
    # Avoid degenerate identical pairs.
    same = a == b
    b[same] = (b[same] + 1) % world.n_actions

    rmat = world.true_reward_matrix()
    ra = rmat[ctx, a]
    rb = rmat[ctx, b]
    p_a = 1.0 / (1.0 + np.exp(-beta * (ra - rb)))
    label = (rng.random(n_pairs) < p_a).astype(np.int64)
    return {"ctx": ctx, "a": a, "b": b, "label": label}


class RewardMLP(nn.Module):
    """context features + one-hot action -> scalar reward estimate."""

    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(feat_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(actions, self.n_actions).float()
        return self.net(torch.cat([feats, onehot], dim=-1)).squeeze(-1)

    def reward_matrix(self, context_feats: np.ndarray) -> np.ndarray:
        """(n_contexts, n_actions) learned reward for every pair."""
        with torch.no_grad():
            feats = torch.from_numpy(context_feats)
            n_c = feats.shape[0]
            out = np.zeros((n_c, self.n_actions), dtype=np.float32)
            for act in range(self.n_actions):
                acts = torch.full((n_c,), act, dtype=torch.long)
                out[:, act] = self(feats, acts).numpy()
            return out


def train_reward_model(
    world: RLHFWorld,
    prefs: dict[str, np.ndarray],
    epochs: int = 200,
    lr: float = 0.01,
    hidden: int = 32,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[RewardMLP, float]:
    """Fit the reward model with BT/logistic loss. Returns (model, val accuracy)."""
    set_seed(seed)
    feats = torch.from_numpy(world.context_feats)
    ctx = torch.from_numpy(prefs["ctx"])
    a = torch.from_numpy(prefs["a"])
    b = torch.from_numpy(prefs["b"])
    label = torch.from_numpy(prefs["label"]).float()

    n = len(label)
    n_val = int(n * val_frac)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = RewardMLP(world.feat_dim, world.n_actions, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        r_a = model(feats[ctx[tr_idx]], a[tr_idx])
        r_b = model(feats[ctx[tr_idx]], b[tr_idx])
        # P(a preferred) = sigma(r_a - r_b); BT loss = BCE on the logit (r_a - r_b).
        loss = bce(r_a - r_b, label[tr_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        r_a = model(feats[ctx[val_idx]], a[val_idx])
        r_b = model(feats[ctx[val_idx]], b[val_idx])
        pred = (r_a > r_b).long()
        acc = float((pred == label[val_idx].long()).float().mean())
    return model, acc


def optimise_policy(
    reward_matrix: np.ndarray, steps: int = 400, lr: float = 0.1, seed: int = 0
) -> np.ndarray:
    """Optimise a per-context softmax policy against a (learned) reward matrix.

    Returns policy logits of shape (n_contexts, n_actions). Contextual-bandit
    policy gradient: maximise expected learned reward under the softmax policy.
    """
    set_seed(seed)
    r = torch.from_numpy(reward_matrix.astype(np.float32))
    logits = torch.zeros_like(r, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * r).sum(dim=-1).mean()
        loss = -expected
        opt.zero_grad()
        loss.backward()
        opt.step()
    return logits.detach().numpy()


def base_policy_logits(world: RLHFWorld) -> np.ndarray:
    """A uniform base policy (all-zero logits -> uniform softmax)."""
    return np.zeros((world.n_contexts, world.n_actions), dtype=np.float32)


def _greedy_actions(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=-1)


def win_rate_vs_base(world: RLHFWorld, rlhf_logits: np.ndarray, base_logits: np.ndarray) -> float:
    """Fraction of contexts where the RLHF policy's greedy action has >= true reward.

    Ties (same true reward, e.g. identical greedy action) count as wins for RLHF
    only when strictly better; equal picks are counted as 0.5 to be honest.
    """
    rmat = world.true_reward_matrix()
    rlhf_a = _greedy_actions(rlhf_logits)
    base_a = _greedy_actions(base_logits)
    idx = np.arange(world.n_contexts)
    r_rlhf = rmat[idx, rlhf_a]
    r_base = rmat[idx, base_a]
    wins = (r_rlhf > r_base).astype(np.float64)
    ties = np.isclose(r_rlhf, r_base)
    wins[ties] = 0.5
    return float(wins.mean())


def true_reward_of_policy(world: RLHFWorld, logits: np.ndarray) -> float:
    """Average true reward of a policy's greedy action across contexts."""
    rmat = world.true_reward_matrix()
    a = _greedy_actions(logits)
    return float(rmat[np.arange(world.n_contexts), a].mean())


def run_rlhf(
    n_contexts: int = 12,
    n_actions: int = 5,
    feat_dim: int = 8,
    n_pairs: int = 4000,
    beta: float = 4.0,
    reward_epochs: int = 200,
    policy_steps: int = 400,
    seed: int = 42,
) -> dict:
    """End-to-end RLHF-lite pipeline. Returns a metrics dict."""
    world = make_world(n_contexts, n_actions, feat_dim, seed=seed)
    prefs = sample_preferences(world, n_pairs=n_pairs, beta=beta, seed=seed)
    reward_model, val_acc = train_reward_model(world, prefs, epochs=reward_epochs, seed=seed)
    learned_r = reward_model.reward_matrix(world.context_feats)
    rlhf_logits = optimise_policy(learned_r, steps=policy_steps, seed=seed)
    base_logits = base_policy_logits(world)

    winrate = win_rate_vs_base(world, rlhf_logits, base_logits)
    true_r = world.true_reward_matrix()
    # Reward-model quality: correlation between learned and true rewards.
    corr = float(np.corrcoef(learned_r.ravel(), true_r.ravel())[0, 1])
    return {
        "world": world,
        "reward_model": reward_model,
        "learned_reward": learned_r,
        "true_reward": true_r,
        "rlhf_logits": rlhf_logits,
        "base_logits": base_logits,
        "reward_model_acc": val_acc,
        "reward_corr": corr,
        "rlhf_winrate": winrate,
        "rlhf_true_reward": true_reward_of_policy(world, rlhf_logits),
        "base_true_reward": true_reward_of_policy(world, base_logits),
        "optimal_true_reward": float(true_r.max(axis=-1).mean()),
    }
