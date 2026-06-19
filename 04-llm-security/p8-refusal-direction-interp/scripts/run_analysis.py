#!/usr/bin/env python3
"""Run the full refusal-direction pipeline on the SYNTHETIC offline simulation:

    extract (difference-in-means) -> ablate (orthogonal projection)
    -> measure (refusal rate + capability retention) -> plot + metrics.json

No model weights are downloaded; everything runs on CPU in seconds. Run via
`make run`. The optional real-transformer path is documented in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refusal_interp import (  # noqa: E402
    ablate_direction,
    build_toy_model,
    cosine_similarity,
    extract_refusal_direction,
    generate_activations,
    refusal_rate,
    sample_prompts,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_projection_hist(
    proj_harmful: torch.Tensor,
    proj_harmless: torch.Tensor,
    proj_harmful_ab: torch.Tensor,
    proj_harmless_ab: torch.Tensor,
) -> Path:
    """Distribution of activation projections onto r_hat, before vs after ablation."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    bins = np.linspace(-3, 3, 30)
    ax0.hist(proj_harmful.numpy(), bins=bins, alpha=0.6, color="#c0392b", label="harmful")
    ax0.hist(proj_harmless.numpy(), bins=bins, alpha=0.6, color="#2980b9", label="harmless")
    ax0.set_title("Before ablation: refusal axis separates the sets")
    ax0.set_xlabel("projection onto refusal direction r̂")
    ax0.set_ylabel("count")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.hist(proj_harmful_ab.numpy(), bins=bins, alpha=0.6, color="#c0392b", label="harmful")
    ax1.hist(proj_harmless_ab.numpy(), bins=bins, alpha=0.6, color="#2980b9", label="harmless")
    ax1.set_title("After ablation: component collapses to ~0")
    ax1.set_xlabel("projection onto refusal direction r̂")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Residual activations projected onto the extracted refusal direction", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "projection_histograms.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_bars(metrics: dict) -> Path:
    """Refusal rate + capability before vs after ablation (the money plot)."""
    labels = ["Refusal rate\n(harmful prompts)", "Capability proxy\n(harmless prompts)"]
    before = [metrics["refusal_rate_before"], metrics["capability_before"]]
    after = [metrics["refusal_rate_after"], metrics["capability_after"]]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b0 = ax.bar(x - w / 2, [v * 100 for v in before], w, label="before ablation", color="#7f8c8d")
    b1 = ax.bar(x + w / 2, [v * 100 for v in after], w, label="after ablation", color="#27ae60")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Ablating the refusal direction is surgical:\nrefusals collapse, capability is retained", pad=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (b0, b1):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.0f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "refusal_vs_capability.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-harmful", type=int, default=128, help="harmful prompts to simulate")
    ap.add_argument("--n-harmless", type=int, default=128, help="harmless prompts to simulate")
    ap.add_argument("--threshold", type=float, default=0.5, help="P(refuse) decision threshold")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Build the frozen synthetic model (known planted refusal axis) + prompts.
    model = build_toy_model(seed=42)
    harmful_prompts, harmless_prompts = sample_prompts(args.n_harmful, args.n_harmless)
    print(f"simulating {args.n_harmful} harmful + {args.n_harmless} harmless prompts "
          f"(synthetic activations, no weights downloaded)")

    # Split into extraction set (to derive r) and a held-out eval set (to measure).
    h_harmful, h_harmless = generate_activations(model, args.n_harmful, args.n_harmless, seed=1)
    n_ex_hf, n_ex_hl = args.n_harmful // 2, args.n_harmless // 2
    hf_extract, hf_eval = h_harmful[:n_ex_hf], h_harmful[n_ex_hf:]
    hl_extract, hl_eval = h_harmless[:n_ex_hl], h_harmless[n_ex_hl:]

    # 1. EXTRACT the refusal direction from the extraction split only.
    r_hat = extract_refusal_direction(hf_extract, hl_extract)
    recovered_cos = abs(cosine_similarity(r_hat, model.r_true))
    print(f"extracted refusal direction; |cos(r_hat, r_true)| = {recovered_cos:.3f} "
          f"(1.0 = perfectly recovered the planted axis)")

    # 2/3. MEASURE on held-out eval set, before vs after ABLATION.
    rr_before = refusal_rate(model.p_refuse(hf_eval), args.threshold)
    cap_before = float(model.p_capable(hl_eval).mean().item())

    hf_eval_ab = ablate_direction(hf_eval, r_hat)
    hl_eval_ab = ablate_direction(hl_eval, r_hat)
    rr_after = refusal_rate(model.p_refuse(hf_eval_ab), args.threshold)
    cap_after = float(model.p_capable(hl_eval_ab).mean().item())

    print(f"refusal rate (harmful):  before={rr_before * 100:5.1f}%  ->  after={rr_after * 100:5.1f}%")
    print(f"capability  (harmless):  before={cap_before * 100:5.1f}%  ->  after={cap_after * 100:5.1f}%")

    # Projections for the histogram figure.
    proj_hf = hf_eval @ r_hat
    proj_hl = hl_eval @ r_hat
    proj_hf_ab = hf_eval_ab @ r_hat
    proj_hl_ab = hl_eval_ab @ r_hat

    metrics = {
        "project": "p8-refusal-direction-interp",
        "summary": (
            "Synthetic-activation simulation of the refusal-direction (abliteration) "
            "method: difference-in-means recovers a planted refusal axis "
            f"(|cos|={recovered_cos:.2f}); inference-time directional ablation drops the "
            f"refusal rate from {rr_before * 100:.0f}% to {rr_after * 100:.0f}% while "
            f"capability stays at {cap_after * 100:.0f}% (vs {cap_before * 100:.0f}%). "
            "Analysis only; no modified weights are produced."
        ),
        "method": "difference-in-means refusal direction + orthogonal-projection ablation",
        "path": "synthetic-offline",
        "seed": 42,
        "n_harmful": args.n_harmful,
        "n_harmless": args.n_harmless,
        "threshold": args.threshold,
        "direction_recovery_cosine": round(recovered_cos, 4),
        "refusal_rate_before": round(rr_before, 4),
        "refusal_rate_after": round(rr_after, 4),
        "capability_before": round(cap_before, 4),
        "capability_after": round(cap_after, 4),
        "capability_retention": round(cap_after / max(cap_before, 1e-8), 4),
    }

    hist = _plot_projection_hist(proj_hf, proj_hl, proj_hf_ab, proj_hl_ab)
    bars = _plot_bars(metrics)
    metrics["figures"] = [str(bars.relative_to(PROJECT)), str(hist.relative_to(PROJECT))]

    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {bars.relative_to(PROJECT)}")
    print(f"wrote {hist.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
