#!/usr/bin/env python3
"""Overnight batch: the full flagship sweep over a finer epsilon grid + more
shadows, written to a timestamped sub-folder so runs don't clobber each other.

This is the "leave it running" entrypoint. It is just `run_tradeoff` with bigger,
slower presets and a richer epsilon grid {inf, 8, 3, 1, 0.5}. Still CPU-only --
expect tens of minutes, not hours, on a laptop. Run via `make overnight`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dp_defenses import (  # noqa: E402
    build_shared_world,
    evaluate_epsilon,
    make_synthetic_pool,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=str, nargs="+",
                    default=["inf", "8", "3", "1", "0.5"])
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-shadows", type=int, default=16)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    args = ap.parse_args()

    eps_vals = [math.inf if e.lower().startswith("inf") else float(e) for e in args.epsilons]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS / "batches" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed()
    t0 = time.time()
    print(f"[{stamp}] building shared world (n={args.n_samples}, shadows={args.n_shadows})")
    pool = make_synthetic_pool(n_samples=args.n_samples)
    world = build_shared_world(pool, n_shadows=args.n_shadows,
                               shadow_epochs=max(20, args.epochs - 10))

    rows = []
    for e in eps_vals:
        lbl = "inf" if math.isinf(e) else f"{e:g}"
        print(f"  -> epsilon={lbl} ...", flush=True)
        r = evaluate_epsilon(world, e, epochs=args.epochs,
                             max_grad_norm=args.max_grad_norm, delta=args.delta)
        rows.append({
            "epsilon": lbl,
            "accounted_epsilon": None if math.isinf(r.accounted_epsilon)
            else round(r.accounted_epsilon, 4),
            "noise_multiplier": round(r.noise_multiplier, 4),
            "train_acc": round(r.train_acc, 4),
            "test_acc": round(r.test_acc, 4),
            "train_test_gap": round(r.gen_gap, 4),
            "mia_auc": round(r.mia_auc, 4),
            "mia_tpr_at_1pct_fpr": round(r.mia_tpr_at_1pct, 4),
            "extraction_acc": round(r.steal_acc, 4),
            "extraction_fidelity": round(r.steal_fidelity, 4),
        })

    elapsed = round(time.time() - t0, 1)
    out = {
        "project": "CAPSTONE-dp-defenses",
        "kind": "overnight-batch",
        "timestamp_utc": stamp,
        "elapsed_seconds": elapsed,
        "config": vars(args),
        "rows": rows,
    }
    (out_dir / "batch.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[done in {elapsed}s] wrote {(out_dir / 'batch.json').relative_to(PROJECT)}")
    print("Tip: `make run ARGS=--full` regenerates the committed tradeoff figures.")


if __name__ == "__main__":
    main()
