"""Fast, offline, deterministic tests for the RL + RLHF project.

They train real torch models on synthetic problems (a numpy gridworld and a
bandit world with a HIDDEN true reward), so every assertion checks genuine
learning with no network and no gymnasium: the policy-gradient agent beats a
random policy, the reward model learns the preference ordering from Bradley-Terry
labels, and the RLHF-optimised policy beats the base policy under the TRUE reward.
The one gymnasium CartPole cross-check is marked ``@slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_rlhf import (
    GridWorld,
    evaluate,
    make_world,
    optimise_policy,
    random_return,
    run_rlhf,
    sample_preferences,
    set_seed,
    train,
    train_reward_model,
    win_rate_vs_base,
)
from rl_rlhf.rlhf import base_policy_logits


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    ta = torch.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    tb = torch.randn(5)
    assert np.array_equal(a, b)
    assert torch.equal(ta, tb)


def test_env_reset_and_step_shapes():
    env = GridWorld(size=4)
    s = env.reset()
    assert s.shape == (env.state_dim,)
    assert s.dtype == np.float32
    s2, r, done = env.step(0)
    assert s2.shape == (env.state_dim,)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    with pytest.raises(ValueError):
        env.step(99)


def test_env_terminal_reward_at_goal():
    # A 2x2 grid: one step right then down reaches the goal.
    env = GridWorld(size=2, start=(0, 0), goal=(1, 1), step_penalty=0.05, goal_reward=1.0)
    env.reset()
    _, _, done = env.step(3)  # right -> (0,1), not goal
    assert not done
    _, r, done = env.step(1)  # down -> (1,1) = goal
    assert done
    assert r == pytest.approx(1.0 - 0.05)


def test_env_max_steps_terminates():
    env = GridWorld(size=5, goal=(4, 4), max_steps=3)
    env.reset()
    env.step(0)
    env.step(0)
    _, _, done = env.step(0)  # third step hits max_steps
    assert done


def test_agent_beats_random_policy():
    env = GridWorld(size=4, max_steps=40)
    policy, _curve = train(env, episodes=300, seed=7)
    agent_return = evaluate(env, policy, episodes=40, greedy=True)
    rand_return = random_return(GridWorld(size=4, max_steps=40), episodes=200, seed=0)
    assert agent_return > rand_return
    # And it actually solves it (positive return means it reaches the goal cheaply).
    assert agent_return > 0.0


def test_agent_training_is_reproducible():
    env = GridWorld(size=4, max_steps=40)
    _, c1 = train(env, episodes=120, seed=3)
    _, c2 = train(GridWorld(size=4, max_steps=40), episodes=120, seed=3)
    assert c1 == c2


def test_reward_model_learns_preferences():
    world = make_world(seed=7)
    prefs = sample_preferences(world, n_pairs=2000, seed=7)
    _model, acc = train_reward_model(world, prefs, epochs=150, seed=7)
    assert acc > 0.7


def test_rlhf_policy_beats_base_under_true_reward():
    world = make_world(seed=7)
    prefs = sample_preferences(world, n_pairs=2000, seed=7)
    model, _acc = train_reward_model(world, prefs, epochs=150, seed=7)
    learned = model.reward_matrix(world.context_feats)
    rlhf_logits = optimise_policy(learned, steps=300, seed=7)
    base_logits = base_policy_logits(world)
    winrate = win_rate_vs_base(world, rlhf_logits, base_logits)
    assert winrate > 0.5


def test_run_rlhf_end_to_end():
    m = run_rlhf(n_pairs=2000, reward_epochs=150, policy_steps=300, seed=7)
    assert m["reward_model_acc"] > 0.7
    assert m["rlhf_winrate"] > 0.5
    assert m["rlhf_true_reward"] > m["base_true_reward"]
    assert m["reward_corr"] > 0.5


@pytest.mark.slow
def test_cartpole_pg_improves_over_random():
    """PG training on gymnasium CartPole should beat a random policy."""
    try:
        import gymnasium as gym
    except Exception as exc:  # gymnasium not installed
        pytest.skip(f"gymnasium unavailable: {type(exc).__name__}")

    try:
        env = gym.make("CartPole-v1")
    except Exception as exc:
        pytest.skip(f"CartPole unavailable: {type(exc).__name__}")

    set_seed(0)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = torch.nn.Sequential(
        torch.nn.Linear(state_dim, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, n_actions),
    )
    opt = torch.optim.Adam(policy.parameters(), lr=0.02)

    def rollout(greedy: bool) -> float:
        obs, _ = env.reset(seed=0)
        log_probs, rewards = [], []
        done = False
        total = 0.0
        steps = 0
        while not done and steps < 500:
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax() if greedy else dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, term, trunc, _ = env.step(int(action.item()))
            done = term or trunc
            rewards.append(float(r))
            total += float(r)
            steps += 1
        return total, log_probs, rewards

    random_baseline = float(np.mean([rollout(greedy=False)[0] for _ in range(5)]))

    for _ in range(120):
        _total, log_probs, rewards = rollout(greedy=False)
        ret = np.cumsum(rewards[::-1])[::-1].astype(np.float32)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        loss = -(torch.stack(log_probs) * torch.from_numpy(ret.copy())).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    trained = float(np.mean([rollout(greedy=False)[0] for _ in range(5)]))
    env.close()
    assert trained > random_baseline
