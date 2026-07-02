"""Policy-gradient RL + a minimal RLHF pipeline (reward model from preferences)."""

from __future__ import annotations

from .agent import PolicyMLP, evaluate, random_return, train
from .envs import GridWorld
from .rlhf import (
    RewardMLP,
    RLHFWorld,
    base_policy_logits,
    make_world,
    optimise_policy,
    run_rlhf,
    sample_preferences,
    train_reward_model,
    true_reward_of_policy,
    win_rate_vs_base,
)
from .utils import get_device, set_seed

__all__ = [
    "GridWorld",
    "PolicyMLP",
    "RLHFWorld",
    "RewardMLP",
    "base_policy_logits",
    "evaluate",
    "get_device",
    "make_world",
    "optimise_policy",
    "random_return",
    "run_rlhf",
    "sample_preferences",
    "set_seed",
    "train",
    "train_reward_model",
    "true_reward_of_policy",
    "win_rate_vs_base",
]
