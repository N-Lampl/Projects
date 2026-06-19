#!/usr/bin/env python3
"""Generate a drifting tabular stream, run the PSI+KS drift monitor over every
window, write the drift-over-time plot + a PSI heatmap + metrics.json.

Default path is numpy/sklearn/scipy-only -> fully offline. Run via `make detect`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_monitoring import (  # noqa: E402
    AlertThresholds,
    StreamConfig,
    first_alert_window,
    generate_stream,
    psi_matrix,
    run_monitor,
    set_seed,
)
from drift_monitoring.metrics import PSI_MAJOR, PSI_MODERATE  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _plot_drift_over_time(reports, feature_names, drift_start, thr) -> Path:
    """The money plot: PSI per feature across windows, with alert threshold."""
    mat = psi_matrix(reports, feature_names)  # (n_windows, n_features)
    windows = list(range(mat.shape[0]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#c0392b", "#e67e22", "#2980b9", "#8e44ad", "#27ae60"]
    for j, name in enumerate(feature_names):
        ax.plot(windows, mat[:, j], "o-", label=name, color=colors[j % len(colors)], lw=1.8, ms=4)

    ax.axhline(thr.psi, color="black", ls="--", lw=1.2, label=f"alert PSI={thr.psi}")
    ax.axhline(PSI_MODERATE, color="gray", ls=":", lw=1.0, label=f"moderate={PSI_MODERATE}")
    ax.axvspan(drift_start - 0.5, windows[-1] + 0.5, color="red", alpha=0.06, label="drift injected")

    fa = first_alert_window(reports)
    if fa is not None:
        ax.axvline(fa, color="red", ls="-", lw=1.0)
        ax.annotate("first alert", (fa, ax.get_ylim()[1] * 0.92),
                    color="red", fontsize=9, ha="center")

    ax.set_xlabel("monitoring window (e.g. hourly)")
    ax.set_ylabel("PSI (population stability index)")
    ax.set_title("Drift over time: PSI per feature vs. alert threshold", pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "drift_over_time.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_heatmap(reports, feature_names) -> Path:
    mat = psi_matrix(reports, feature_names).T  # (n_features, n_windows)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="inferno", vmin=0, vmax=max(PSI_MAJOR * 2, mat.max()))
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("monitoring window")
    ax.set_title("PSI heatmap (brighter = more drift)", pad=8)
    fig.colorbar(im, ax=ax, label="PSI", fraction=0.025, pad=0.01)
    fig.tight_layout()
    out = FIG_DIR / "psi_heatmap.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--window-size", type=int, default=500)
    ap.add_argument("--drift-start", type=int, default=12)
    ap.add_argument("--drift-ramp", type=int, default=6)
    ap.add_argument("--psi-threshold", type=float, default=PSI_MAJOR)
    ap.add_argument("--ks-threshold", type=float, default=0.15)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = StreamConfig(
        n_windows=args.n_windows,
        window_size=args.window_size,
        drift_start=args.drift_start,
        drift_ramp=args.drift_ramp,
    )
    reference, windows, strengths = generate_stream(cfg)
    thr = AlertThresholds(psi=args.psi_threshold, ks=args.ks_threshold)

    reports = run_monitor(reference, windows, cfg.feature_names, thr, n_bins=args.bins)

    fa = first_alert_window(reports)
    print(f"stream: {cfg.n_windows} windows x {cfg.window_size} samples; "
          f"drift injected at window {cfg.drift_start}")
    for r in reports:
        flag = "ALERT" if r["window_alert"] else "  ok "
        feats = ",".join(r["alerting_features"]) or "-"
        print(f"  w{r['window']:>2} [{flag}] drift_strength={strengths[r['window']]:.2f} "
              f"alerting={feats}")
    print(f"\nfirst alert at window: {fa}  (drift began at {cfg.drift_start})")

    # Detection latency = how many windows after true drift onset we first alert.
    latency = None if fa is None else fa - cfg.drift_start
    # False alarms = alerts strictly before drift onset.
    false_alarms = sum(1 for r in reports if r["window_alert"] and r["window"] < cfg.drift_start)

    curve = _plot_drift_over_time(reports, cfg.feature_names, cfg.drift_start, thr)
    heat = _plot_heatmap(reports, cfg.feature_names)

    metrics = {
        "project": "p7-drift-monitoring",
        "summary": (
            f"PSI+KS monitor over {cfg.n_windows} windows; drift injected at window "
            f"{cfg.drift_start}; first alert at window {fa} "
            f"(latency {latency} windows, {false_alarms} false alarms)."
        ),
        "seed": 42,
        "n_windows": cfg.n_windows,
        "window_size": cfg.window_size,
        "drift_start_window": cfg.drift_start,
        "thresholds": {"psi": thr.psi, "ks": thr.ks, "ks_pvalue": thr.ks_pvalue},
        "first_alert_window": fa,
        "detection_latency_windows": latency,
        "false_alarm_windows": false_alarms,
        "n_alert_windows": sum(1 for r in reports if r["window_alert"]),
        "final_window_psi": {
            f: reports[-1]["features"][f]["psi"] for f in cfg.feature_names
        },
        "per_window": [
            {
                "window": r["window"],
                "alert": r["window_alert"],
                "alerting_features": r["alerting_features"],
                "max_psi": max(r["features"][f]["psi"] for f in cfg.feature_names),
            }
            for r in reports
        ],
        "figures": [str(curve.relative_to(PROJECT)), str(heat.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {curve.relative_to(PROJECT)}")
    print(f"wrote {heat.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
