#!/usr/bin/env python3
"""Run the attack zoo (PGD / C&W-L2 / DeepFool) against the trained target,
build the comparison table + figures + metrics.json. Run via `make attack`.

Auto-trains the target if weights are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attack_zoo import (  # noqa: E402
    SmallCNN,
    cw_l2,
    deepfool,
    evaluate,
    get_device,
    get_loaders,
    pgd,
    run_attack,
    set_seed,
)
from attack_zoo.train import load_model, save_model, train  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
WEIGHTS = PROJECT / "models" / "smallcnn.pt"


def _get_model(device, source, num_classes, epochs):
    if WEIGHTS.exists():
        print(f"loading weights <- {WEIGHTS.relative_to(PROJECT)}")
        model, meta = load_model(WEIGHTS, device)
        return model, meta
    print("no weights found - training a fresh target...")
    train_loader, _, meta = get_loaders(source=source, num_classes=num_classes)
    model = SmallCNN(in_channels=meta["in_channels"], num_classes=meta["num_classes"])
    train(model, train_loader, epochs=epochs, device=device)
    meta["source"] = source
    save_model(model, WEIGHTS, meta=meta)
    return model.eval(), meta


def _bar_chart(rows: list[dict]) -> Path:
    names = [r["attack"] for r in rows]
    succ = [r["success_rate"] * 100 for r in rows]
    rt = [r["runtime_s"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#c0392b", "#2980b9", "#27ae60"]
    ax1.bar(names, succ, color=colors[: len(names)])
    ax1.set_ylabel("attack success rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Success rate by attack")
    for i, v in enumerate(succ):
        ax1.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=9)

    ax2.bar(names, rt, color=colors[: len(names)])
    ax2.set_ylabel("runtime (s)")
    ax2.set_title("Runtime by attack (lower = cheaper)")
    for i, v in enumerate(rt):
        ax2.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Attack zoo: PGD vs C&W-L2 vs DeepFool", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "attack_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _perturbation_chart(rows: list[dict]) -> Path:
    names = [r["attack"] for r in rows]
    l2 = [(r["mean_l2"] or 0.0) for r in rows]
    linf = [(r["mean_linf"] or 0.0) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, l2, color="#8e44ad")
    ax1.set_ylabel("mean L2 of perturbation")
    ax1.set_title("Perturbation size (L2) - smaller = stealthier")
    for i, v in enumerate(l2):
        ax1.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=9)
    ax2.bar(names, linf, color="#d35400")
    ax2.set_ylabel("mean L-inf of perturbation")
    ax2.set_title("Perturbation size (L-inf)")
    for i, v in enumerate(linf):
        ax2.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Cost of success: how much each attack has to perturb", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "perturbation_sizes.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _grid(model, meta, device, eps) -> Path:
    """Show clean vs each attack on a few correctly-classified samples."""
    _, test_loader, _ = get_loaders(
        source=meta.get("source", "synthetic"),
        batch_size=64,
        test_subset=64,
        num_classes=meta["num_classes"],
    )
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        keep = (model(x).argmax(1) == y).nonzero(as_tuple=True)[0][:4]
    x, y = x[keep], y[keep]

    advs = {
        "PGD": pgd(model, x, y, epsilon=eps, steps=20),
        "C&W-L2": cw_l2(model, x, y, steps=60),
        "DeepFool": deepfool(model, x, y, steps=30),
    }
    rows_imgs = [("clean", x)] + list(advs.items())
    n = x.shape[0]

    def show(ax, img):
        img = img.cpu()
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axes = plt.subplots(len(rows_imgs), n, figsize=(1.8 * n, 1.8 * len(rows_imgs)))
    for r, (label, batch) in enumerate(rows_imgs):
        with torch.no_grad():
            preds = model(batch).argmax(1)
        for col in range(n):
            ax = axes[r, col]
            show(ax, batch[col])
            wrong = preds[col] != y[col]
            color = "red" if (r > 0 and wrong) else "black"
            ax.set_title(f"pred {preds[col].item()}", fontsize=8, color=color)
        axes[r, 0].set_ylabel(label, fontsize=9)
    fig.suptitle("Clean vs PGD / C&W / DeepFool (red = misclassified)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "clean_vs_attacks.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "cifar10", "mnist"])
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=200, help="images for the benchmark")
    ap.add_argument("--epsilon", type=float, default=0.1, help="L-inf budget for PGD")
    ap.add_argument("--pgd-steps", type=int, default=20)
    ap.add_argument("--cw-steps", type=int, default=80)
    ap.add_argument("--deepfool-steps", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3, help="epochs if auto-training")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, meta = _get_model(device, args.source, args.num_classes, args.epochs)

    _, eval_loader, _ = get_loaders(
        source=meta.get("source", args.source),
        batch_size=64,
        test_subset=args.n_eval,
        num_classes=meta["num_classes"],
    )
    clean_acc = evaluate(model, eval_loader, device=device)
    print(f"clean accuracy on eval subset: {clean_acc:.4f}")

    specs = [
        ("PGD", pgd, dict(epsilon=args.epsilon, steps=args.pgd_steps)),
        ("C&W-L2", cw_l2, dict(steps=args.cw_steps)),
        ("DeepFool", deepfool, dict(steps=args.deepfool_steps)),
    ]
    rows = []
    for name, fn, kw in specs:
        print(f"running {name} ...")
        m = run_attack(model, eval_loader, fn, device=device, **kw)
        m["attack"] = name
        rows.append(m)
        l2 = f"{m['mean_l2']:.3f}" if m["mean_l2"] else "n/a"
        linf = f"{m['mean_linf']:.3f}" if m["mean_linf"] else "n/a"
        print(
            f"  success={m['success_rate'] * 100:5.1f}%  "
            f"meanL2={l2}  meanLinf={linf}  runtime={m['runtime_s']:.2f}s"
        )

    bar = _bar_chart(rows)
    pert = _perturbation_chart(rows)
    grid = _grid(model, meta, device, args.epsilon)

    metrics = {
        "project": "p2-attack-zoo",
        "summary": (
            "Hand-rolled PGD (L-inf), C&W-L2 and DeepFool evasion attacks compared on a "
            f"SmallCNN ({meta.get('source', args.source)} data, clean acc "
            f"{clean_acc * 100:.1f}%). Reports success rate, mean L2/L-inf and runtime."
        ),
        "source": meta.get("source", args.source),
        "seed": 42,
        "clean_accuracy": clean_acc,
        "n_eval_images": args.n_eval,
        "pgd_epsilon": args.epsilon,
        "attacks": {
            r["attack"]: {
                "success_rate": r["success_rate"],
                "mean_l2": r["mean_l2"],
                "mean_linf": r["mean_linf"],
                "runtime_s": r["runtime_s"],
                "n_correct": r["n_correct"],
                "n_success": r["n_success"],
            }
            for r in rows
        },
        "figures": [
            str(bar.relative_to(PROJECT)),
            str(pert.relative_to(PROJECT)),
            str(grid.relative_to(PROJECT)),
        ],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {bar.relative_to(PROJECT)}")
    print(f"wrote {pert.relative_to(PROJECT)}")
    print(f"wrote {grid.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
