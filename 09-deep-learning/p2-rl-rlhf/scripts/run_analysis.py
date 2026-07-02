#!/usr/bin/env python3
"""Run the RL + RLHF experiment -> results/metrics.json + figures.

Part A trains a REINFORCE policy-gradient agent on a self-contained numpy
gridworld and shows it beats a random policy. Part B runs a minimal RLHF
pipeline: learn a reward model from Bradley-Terry preferences over a hidden true
reward, optimise a policy against the *learned* reward, and measure its win-rate
against the base policy under the TRUE reward.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_rlhf import (  # noqa: E402
    GridWorld,
    evaluate,
    random_return,
    run_rlhf,
    set_seed,
    train,
)
from rl_rlhf.plots import (  # noqa: E402
    plot_reward_model_fit,
    plot_rlhf_winrate,
    plot_training_return,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=5, help="gridworld side length")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--n-pairs", type=int, default=4000, help="RLHF preference pairs")
    ap.add_argument("--reward-epochs", type=int, default=200)
    ap.add_argument("--policy-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # --- Part A: policy-gradient agent on the gridworld ---------------------
    env = GridWorld(size=args.size)
    policy, curve = train(env, episodes=args.episodes, seed=args.seed)
    final_return = evaluate(env, policy, episodes=100, greedy=True)
    rand_return = random_return(GridWorld(size=args.size), episodes=200, seed=args.seed)

    # --- Part B: RLHF-lite --------------------------------------------------
    rlhf = run_rlhf(
        n_pairs=args.n_pairs,
        reward_epochs=args.reward_epochs,
        policy_steps=args.policy_steps,
        seed=args.seed,
    )

    fig1 = plot_training_return(curve, rand_return, FIGURES / "training_return.png")
    fig2 = plot_rlhf_winrate(rlhf, FIGURES / "rlhf_winrate.png")
    fig3 = plot_reward_model_fit(rlhf, FIGURES / "reward_model_fit.png")

    summary_str = (
        f"REINFORCE with a baseline reaches an average return of {final_return:.2f} on a "
        f"{args.size}x{args.size} gridworld, far above the random policy's {rand_return:.2f}. "
        f"The RLHF reward model, trained only on Bradley-Terry preference labels, hits "
        f"{rlhf['reward_model_acc'] * 100:.0f}% held-out preference accuracy (corr "
        f"{rlhf['reward_corr']:.2f} with the hidden true reward); optimising a policy against "
        f"the learned reward beats the base policy under the TRUE reward on "
        f"{rlhf['rlhf_winrate'] * 100:.0f}% of contexts."
    )

    metrics = {
        "project": "p2-rl-rlhf",
        "summary": summary_str,
        "data_source": "synthetic gridworld MDP + synthetic Bradley-Terry preferences (offline)",
        "seed": args.seed,
        "final_return": final_return,
        "random_return": rand_return,
        "rlhf_winrate": rlhf["rlhf_winrate"],
        "reward_model_acc": rlhf["reward_model_acc"],
        "rl": {
            "grid_size": args.size,
            "episodes": args.episodes,
            "final_return": final_return,
            "random_return": rand_return,
            "return_improvement": final_return - rand_return,
        },
        "rlhf": {
            "n_pairs": args.n_pairs,
            "reward_model_acc": rlhf["reward_model_acc"],
            "reward_corr": rlhf["reward_corr"],
            "winrate_vs_base_true_reward": rlhf["rlhf_winrate"],
            "base_true_reward": rlhf["base_true_reward"],
            "rlhf_true_reward": rlhf["rlhf_true_reward"],
            "optimal_true_reward": rlhf["optimal_true_reward"],
        },
        "figures": [f"results/figures/{p.name}" for p in (fig1, fig2, fig3)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary_str)
    print(f"[ok] wrote {RESULTS / 'metrics.json'} + 3 figures")


if __name__ == "__main__":
    main()
