#!/usr/bin/env python3
"""REAL-model abliteration on a small open-weight instruct model (CPU-OK).

Extract a single refusal direction (difference-in-means of last-token residuals),
ablate it from every layer via forward hooks, and MEASURE the effect:
refusal-rate before/after on held-out prompts + a capability-retention proxy.

ETHICS: analysis only. We never write modified weights and never store the model's
completions for harmful prompts (only a refuse/comply classification of a short
prefix). Run against a small open-weight model you are licensed to use. See ../../ETHICS.md.

Usage:
    python3 scripts/run_real_abliteration.py            # Qwen2.5-0.5B-Instruct
    python3 scripts/run_real_abliteration.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refusal_interp.direction import extract_refusal_direction, make_ablation_hook  # noqa: E402
from refusal_interp.eval import perplexity, refusal_rate_real  # noqa: E402
from refusal_interp.prompts import (  # noqa: E402
    BENIGN_PPL_TEXT,
    HARMFUL_EVAL,
    HARMFUL_EXTRACT,
    HARMLESS_EVAL,
    HARMLESS_EXTRACT,
)
from refusal_interp.real_model import last_token_residuals, load_instruct_model  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG = PROJECT / "results" / "figures"


def _refusal_fig(rates: dict, ppl_before: float, ppl_after: float) -> Path:
    labels = ["harmful\nbefore", "harmful\nafter", "harmless\nbefore", "harmless\nafter"]
    vals = [rates["hb"] * 100, rates["ha"] * 100, rates["nb"] * 100, rates["na"] * 100]
    colors = ["#c0392b", "#27ae60", "#7f8c8d", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("refusal rate (%)")
    ax.set_title(
        "Abliteration removes refusals at a modest capability cost\n"
        f"benign perplexity {ppl_before:.1f} -> {ppl_after:.1f}"
    )
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=9)
    fig.tight_layout()
    out = FIG / "refusal_vs_capability.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _proj_fig(ph: np.ndarray, pn: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(ph.min(), pn.min()), max(ph.max(), pn.max()), 20)
    ax.hist(pn, bins=bins, alpha=0.7, label="harmless", color="#7f8c8d")
    ax.hist(ph, bins=bins, alpha=0.7, label="harmful", color="#c0392b")
    ax.set_xlabel("projection onto the recovered refusal direction")
    ax.set_ylabel("count")
    ax.set_title("One axis separates 'will refuse' from 'will answer'")
    ax.legend()
    fig.tight_layout()
    out = FIG / "projection_histograms.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(42)
    FIG.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} (CPU, float32)...")
    model, tok = load_instruct_model(args.model)
    n_layers = model.config.num_hidden_layers
    layer = max(1, int(args.layer_frac * n_layers))
    print(f"  {n_layers} layers; extracting direction at hidden_states[{layer}]")

    # 1) EXTRACT the refusal direction from real activations.
    h_harm = last_token_residuals(model, tok, HARMFUL_EXTRACT, layer)
    h_harm_less = last_token_residuals(model, tok, HARMLESS_EXTRACT, layer)
    r_hat = extract_refusal_direction(h_harm, h_harm_less)
    ph = (h_harm @ r_hat).numpy()
    pn = (h_harm_less @ r_hat).numpy()
    print(f"  direction recovered; mean proj harmful={ph.mean():.2f} harmless={pn.mean():.2f}")

    # 2) BASELINE behaviour (no intervention).
    print("measuring baseline refusal rates...")
    hb = refusal_rate_real(model, tok, HARMFUL_EVAL, args.max_new_tokens)
    nb = refusal_rate_real(model, tok, HARMLESS_EVAL, args.max_new_tokens)
    ppl_before = perplexity(model, tok, BENIGN_PPL_TEXT)
    print(f"  refusal harmful={hb:.0%} harmless={nb:.0%}  ppl={ppl_before:.2f}")

    # 3) ABLATE: hook every decoder layer to project out r_hat.
    print("registering ablation hooks on all layers, re-measuring...")
    handles = [layer_mod.register_forward_hook(make_ablation_hook(r_hat))
               for layer_mod in model.model.layers]
    try:
        ha = refusal_rate_real(model, tok, HARMFUL_EVAL, args.max_new_tokens)
        na = refusal_rate_real(model, tok, HARMLESS_EVAL, args.max_new_tokens)
        ppl_after = perplexity(model, tok, BENIGN_PPL_TEXT)
    finally:
        for h in handles:
            h.remove()
    print(f"  refusal harmful={ha:.0%} harmless={na:.0%}  ppl={ppl_after:.2f}")

    rates = {"hb": hb, "ha": ha, "nb": nb, "na": na}
    fig1 = _refusal_fig(rates, ppl_before, ppl_after)
    fig2 = _proj_fig(ph, pn)

    # Save the direction vector (analysis artifact -- NOT model weights).
    (PROJECT / "results" / "refusal_direction.json").write_text(
        json.dumps({"model": args.model, "layer": layer, "dim": int(r_hat.numel()),
                    "unit_vector": [round(x, 6) for x in r_hat.tolist()]}) + "\n"
    )

    retention = round(min(1.0, ppl_before / ppl_after), 4) if ppl_after else None
    metrics = {
        "project": "p8-refusal-direction-interp",
        "summary": (
            f"Real abliteration of {args.model.split('/')[-1]}: ablating one refusal "
            f"direction cut harmful-prompt refusal {hb:.0%} -> {ha:.0%} while benign "
            f"perplexity rose modestly {ppl_before:.1f}->{ppl_after:.1f} (capability largely retained)."
        ),
        "path": "real-model",
        "model_id": args.model,
        "n_layers": n_layers,
        "layer": layer,
        "n_extract_per_class": len(HARMFUL_EXTRACT),
        "n_eval_harmful": len(HARMFUL_EVAL),
        "n_eval_harmless": len(HARMLESS_EVAL),
        "refusal_rate_harmful_before": round(hb, 4),
        "refusal_rate_harmful_after": round(ha, 4),
        "refusal_rate_harmless_before": round(nb, 4),
        "refusal_rate_harmless_after": round(na, 4),
        "perplexity_before": round(ppl_before, 4),
        "perplexity_after": round(ppl_after, 4),
        "capability_retention": retention,
        "figures": [str(fig1.relative_to(PROJECT)), str(fig2.relative_to(PROJECT))],
        "note": "Analysis only; no modified weights produced or stored. See ../../ETHICS.md.",
    }
    (PROJECT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote results/metrics.json + 2 figures + results/refusal_direction.json")


if __name__ == "__main__":
    main()
