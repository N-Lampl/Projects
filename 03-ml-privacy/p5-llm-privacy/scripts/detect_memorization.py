#!/usr/bin/env python3
"""Train a tiny char-LM on a canary-laced corpus, then measure how much it
memorized via the canary-exposure metric. Writes figures + metrics.json.

Run via `make detect`. The default path is fully offline (synthetic data, torch).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_privacy import (  # noqa: E402
    CharLM,
    build_corpus,
    estimate_exposure,
    get_device,
    save_model,
    sequence_log_perplexity,
    set_seed,
    train,
)
from llm_privacy.corpus import Canary, make_canary  # noqa: E402
from llm_privacy.exposure import LOG2_R, RANDOMNESS_SPACE  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
WEIGHTS = PROJECT / "models" / "charlm.pt"


def _random_perp_sample(model, name: str, n: int, seed: int, device) -> np.ndarray:
    from llm_privacy.corpus import SECRET_ALPHABET, SECRET_LEN

    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        secret = "".join(rng.choice(list(SECRET_ALPHABET), size=SECRET_LEN))
        out[i] = sequence_log_perplexity(model, make_canary(name, secret), device)
    return out


def _plot_exposure_bar(results) -> Path:
    names = [r.name for r in results]
    expo = [r.exposure for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, expo, color="#c0392b")
    ax.axhline(LOG2_R, ls="--", color="#7f8c8d", label=f"max exposure = log2(|R|) = {LOG2_R:.1f}")
    ax.axhline(1.0, ls=":", color="#2c3e50", label="baseline (no memorization) ~ 1")
    ax.set_ylabel("exposure (bits)")
    ax.set_xlabel("canary (inserted secret)")
    ax.set_title("Canary exposure: how strongly the LM memorized each secret", pad=12)
    ax.legend(fontsize=8)
    for i, e in enumerate(expo):
        ax.annotate(f"{e:.0f}", (i, e), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "exposure_by_canary.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_perp_dist(model, canary: Canary, device) -> Path:
    """The Secret Sharer picture: real secret sits in the LEFT tail (very likely)
    vs. the bell of random alternatives.
    """
    randoms = _random_perp_sample(model, canary.name, 2000, seed=7, device=device)
    real = sequence_log_perplexity(model, canary.text, device)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(randoms, bins=40, color="#95a5a6", alpha=0.85, label="random secrets")
    ax.axvline(real, color="#c0392b", lw=2.5, label=f"REAL inserted secret\n(perp={real:.2f})")
    ax.set_xlabel("log-perplexity of completion (lower = model finds it more likely)")
    ax.set_ylabel("count")
    ax.set_title(f"'{canary.name}': real secret vs. {len(randoms)} random alternatives", pad=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "perplexity_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-background", type=int, default=4000)
    ap.add_argument("--n-canaries", type=int, default=4)
    ap.add_argument("--canary-repeats", type=int, default=16,
                    help="times each canary is inserted (more -> more memorization)")
    ap.add_argument("--samples", type=int, default=2000,
                    help="random secrets sampled for the exposure estimate")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"building corpus: {args.n_background} background lines + "
          f"{args.n_canaries} canaries x{args.canary_repeats}...")
    corpus, canaries = build_corpus(
        n_background=args.n_background,
        n_canaries=args.n_canaries,
        canary_repeats=args.canary_repeats,
    )
    print(f"corpus chars={len(corpus)}  |R|={RANDOMNESS_SPACE:,}  max exposure={LOG2_R:.1f} bits")

    print(f"training tiny char-LM for {args.epochs} epochs...")
    model = CharLM()
    losses = train(model, corpus, epochs=args.epochs, device=device)
    save_model(model, WEIGHTS)

    print("estimating exposure per canary...")
    results = [estimate_exposure(model, c, n_samples=args.samples, seed=11, device=device)
               for c in canaries]
    for r in results:
        print(f"  {r.name:<6} secret={r.secret}  exposure={r.exposure:6.1f} bits  "
              f"est.rank={r.estimated_rank:,.0f}/{RANDOMNESS_SPACE:,}")

    # Control: a NON-inserted canary should show ~baseline exposure (sanity check).
    control = Canary(name=canaries[0].name, secret="0000000000",
                     text=make_canary(canaries[0].name, "0000000000"))
    control_res = estimate_exposure(model, control, n_samples=args.samples, seed=99, device=device)
    print(f"  control (never-inserted secret) exposure={control_res.exposure:.1f} bits")

    bar = _plot_exposure_bar(results)
    # plot the most-exposed canary's perplexity distribution
    most = max(range(len(results)), key=lambda i: results[i].exposure)
    dist = _plot_perp_dist(model, canaries[most], device)

    mean_exposure = float(np.mean([r.exposure for r in results]))
    max_exposure = float(max(r.exposure for r in results))
    leaked = [r.name for r in results if r.exposure > LOG2_R * 0.5]

    metrics = {
        "project": "p5-llm-privacy",
        "summary": (
            f"Tiny char-LM trained on synthetic logs with {args.n_canaries} inserted "
            f"canaries (x{args.canary_repeats}). Mean canary exposure "
            f"{mean_exposure:.1f}/{LOG2_R:.1f} bits; {len(leaked)} secrets effectively "
            f"memorized (>50% of max exposure). Control (never-inserted) exposure "
            f"{control_res.exposure:.1f} bits."
        ),
        "method": "Secret Sharer canary exposure (perplexity-based membership test)",
        "seed": 42,
        "randomness_space": RANDOMNESS_SPACE,
        "max_exposure_bits": LOG2_R,
        "epochs": args.epochs,
        "n_background_lines": args.n_background,
        "canary_repeats": args.canary_repeats,
        "final_train_loss": losses[-1],
        "mean_exposure_bits": mean_exposure,
        "max_exposure_bits_observed": max_exposure,
        "control_exposure_bits": control_res.exposure,
        "leaked_canaries": leaked,
        "exposure_by_canary": {
            r.name: {
                "secret": r.secret,
                "exposure_bits": r.exposure,
                "real_perplexity": r.real_perplexity,
                "mean_random_perplexity": r.mean_random_perplexity,
                "estimated_rank": r.estimated_rank,
            }
            for r in results
        },
        "figures": [str(bar.relative_to(PROJECT)), str(dist.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {bar.relative_to(PROJECT)}")
    print(f"wrote {dist.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")
    assert not math.isnan(mean_exposure)


if __name__ == "__main__":
    main()
