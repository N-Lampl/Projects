#!/usr/bin/env python3
"""Train the tiny transformer, run three interp analyses -> metrics + figures.

Trains a 2-layer decoder-only transformer from scratch on a synthetic induction
task until an induction head emerges, then runs: (1) an induction-head score over
all heads, (2) the logit lens over residual-stream depth, and (3) activation
patching of the clean residual into a corrupted run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformer_interp import (  # noqa: E402
    induction_head_score,
    logit_lens,
    make_induction_batch,
    patching_sweep,
    plot_activation_patching,
    plot_attention_induction,
    plot_logit_lens,
    set_seed,
    train_induction,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--half", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    result = train_induction(
        steps=args.steps, batch_size=args.batch_size, half=args.half, seed=args.seed
    )
    model = result.model

    # Held-out evaluation batch for all analyses.
    eval_batch = make_induction_batch(
        batch_size=128, half=args.half, vocab_size=model.cfg.vocab_size, seed=7
    )

    head = induction_head_score(model, eval_batch)
    lens = logit_lens(model, eval_batch)

    # Activation patching: corrupt = a different random batch (breaks the
    # induction cue); patch at the last repeated-half position.
    corrupt = make_induction_batch(
        batch_size=128, half=args.half, vocab_size=model.cfg.vocab_size, seed=13
    )
    patch_pos = model.cfg.n_ctx - 2  # last position with a defined next token
    sweep = patching_sweep(model, eval_batch, corrupt, position=patch_pos)

    # Figure 1: attention heatmap of the discovered induction head on one sample.
    import torch  # local import keeps top-level deps torch-free at import time

    with torch.no_grad():
        _, cache = model(eval_batch.tokens, return_cache=True)
    sample_attn = cache.attn[head["best_layer"]][0, head["best_head"]].numpy()
    sample_tokens = eval_batch.tokens[0].numpy()

    fig1 = plot_attention_induction(
        sample_attn,
        sample_tokens,
        head["best_layer"],
        head["best_head"],
        FIGURES / "attention_induction.png",
    )
    fig2 = plot_logit_lens(lens, FIGURES / "logit_lens.png")
    fig3 = plot_activation_patching(sweep, FIGURES / "activation_patching.png")

    last_layer_acc = lens["accuracy_by_layer"][-1]
    first_acc = lens["accuracy_by_layer"][0]
    best_patch = max(sweep["effect_by_layer"])

    summary_str = (
        f"A 2-layer transformer trained from scratch on the induction task reaches "
        f"val loss {result.val_loss:.3f}; an induction head emerged at layer "
        f"{head['best_layer']} head {head['best_head']} with attention mass "
        f"{head['max_score']:.2f} on the previous-occurrence+1 token. The logit lens "
        f"shows next-token accuracy rising from {first_acc:.2f} (embedding) to "
        f"{last_layer_acc:.2f} (final layer), and activation patching recovers up to "
        f"{best_patch * 100:.0f}% of the clean-vs-corrupt logit gap — localizing where "
        f"the induction computation lives."
    )

    metrics = {
        "project": "p1-transformer-interp",
        "summary": summary_str,
        "data_source": "synthetic induction task (repeated random subsequences)",
        "seed": args.seed,
        "val_loss": result.val_loss,
        "final_train_loss": result.final_loss,
        "induction_head_score": {
            "max_score": head["max_score"],
            "best_layer": head["best_layer"],
            "best_head": head["best_head"],
            "per_head": head["per_head"],
        },
        "logit_lens_by_layer": {
            "accuracy": lens["accuracy_by_layer"],
            "correct_logit": lens["correct_logit_by_layer"],
        },
        "patching_effect": {
            "position": sweep["position"],
            "effect_by_layer": sweep["effect_by_layer"],
        },
        "figures": [f"results/figures/{p.name}" for p in (fig1, fig2, fig3)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary_str)
    print(f"[ok] wrote {RESULTS / 'metrics.json'} + 3 figures")


if __name__ == "__main__":
    main()
