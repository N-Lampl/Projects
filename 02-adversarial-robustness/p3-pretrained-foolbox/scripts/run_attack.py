#!/usr/bin/env python3
"""Attack a target model at inference time and write figures + metrics.json.

DEFAULT (offline): self-trains a SmallCNN on synthetic data, then runs FGSM &
PGD epsilon sweeps and a clean-vs-adversarial demo grid.

OPTIONAL (online): `--pretrained` attacks torchvision ResNet-18 on real CIFAR
images. If the weight/data download fails, it auto-falls back to the offline
SmallCNN and records that in metrics.json.

Run via `make attack` (offline) or `make attack ARGS=--pretrained`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pretrained_foolbox import (  # noqa: E402
    SmallCNN,
    fgsm_perturb,
    get_device,
    pgd_perturb,
    predict,
    set_seed,
    true_label_confidence,
)
from pretrained_foolbox.data import SYNTH_CLASSES, make_synthetic  # noqa: E402
from pretrained_foolbox.train import load_model, train_offline_target  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"
WEIGHTS = PROJECT / "models" / "smallcnn_synth.pt"

DEFAULT_EPS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]


def _build_offline_target(device: torch.device, epochs: int):
    if WEIGHTS.exists():
        print(f"loading offline target <- {WEIGHTS.relative_to(PROJECT)}")
        return load_model(WEIGHTS, device), SYNTH_CLASSES
    print("no weights found - training the offline SmallCNN on synthetic data...")
    model = train_offline_target(WEIGHTS, device, epochs=epochs)
    return model, SYNTH_CLASSES


def _build_pretrained_target(device: torch.device, n_images: int):
    """Try the online ResNet-18 + CIFAR path. Return (model, x, y, classes) or None."""
    try:
        from pretrained_foolbox.data import get_cifar_samples
        from pretrained_foolbox.model import load_pretrained_resnet18

        print("downloading ResNet-18 ImageNet weights + CIFAR samples (online)...")
        model = load_pretrained_resnet18().to(device).eval()
        x, _, _ = get_cifar_samples(n=n_images, root=str(PROJECT / "data"))
        # ResNet-18 is ImageNet (1000-class); upsample CIFAR to 224 and use the
        # model's OWN clean predictions as the labels we then try to flip.
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = x.to(device)
        with torch.no_grad():
            y = model(x).argmax(1)
        return model, x, y, [str(int(c)) for c in y]
    except Exception as e:  # offline / blocked download -> caller falls back
        print(f"  pretrained path unavailable ({type(e).__name__}: {e}); falling back offline.")
        return None


def _sweep(model, x, y, epsilons, attack_fn):
    """For each eps: accuracy (vs the reference labels y) + mean true-label conf."""
    accs, confs = {}, {}
    for eps in epsilons:
        x_adv = x if eps == 0 else attack_fn(model, x, y, eps)
        pred, _ = predict(model, x_adv)
        accs[eps] = float((pred == y).float().mean())
        confs[eps] = float(true_label_confidence(model, x_adv, y).mean())
    return accs, confs


def _plot_curves(fgsm_acc, pgd_acc, fgsm_conf, target_name) -> Path:
    eps = sorted(fgsm_acc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(eps, [fgsm_acc[e] * 100 for e in eps], "o-", color="#c0392b", lw=2, label="FGSM")
    ax1.plot(eps, [pgd_acc[e] * 100 for e in eps], "s--", color="#8e44ad", lw=2, label="PGD")
    ax1.set_xlabel("epsilon (L-inf budget)")
    ax1.set_ylabel("accuracy (%)")
    ax1.set_title(f"{target_name}: accuracy under attack")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(eps, [fgsm_conf[e] * 100 for e in eps], "o-", color="#2980b9", lw=2)
    ax2.set_xlabel("epsilon (L-inf budget)")
    ax2.set_ylabel("mean confidence in TRUE class (%)")
    ax2.set_title("Confidence collapse under FGSM")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Inference-time evasion: a tiny perturbation drops accuracy AND confidence", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "accuracy_confidence_vs_epsilon.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_grid(model, x, y, classes, eps_demo, n=6) -> Path:
    """Clean vs FGSM-adversarial images with predictions + confidence."""
    pred_c, conf_c = predict(model, x)
    keep = (pred_c == y).nonzero(as_tuple=True)[0][:n]
    if len(keep) < n:  # pad with first images if not enough correct ones
        keep = torch.arange(min(n, x.size(0)))
    xs, ys = x[keep], y[keep]

    x_adv = fgsm_perturb(model, xs, ys, eps_demo)
    pred_c, conf_c = predict(model, xs)
    pred_a, conf_a = predict(model, x_adv)

    def to_img(t):
        return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    m = len(keep)
    fig, axes = plt.subplots(2, m, figsize=(2.0 * m, 4.4))
    if m == 1:
        axes = axes.reshape(2, 1)
    for i in range(m):
        axes[0, i].imshow(to_img(xs[i]))
        axes[0, i].set_title(f"{classes[pred_c[i]]}\n{conf_c[i] * 100:.0f}%", fontsize=8)
        axes[1, i].imshow(to_img(x_adv[i]))
        ok = pred_a[i] == ys[i]
        color = "green" if ok else "red"
        axes[1, i].set_title(f"{classes[pred_a[i]]}\n{conf_a[i] * 100:.0f}%", fontsize=8, color=color)
        for ax in (axes[0, i], axes[1, i]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("clean", fontsize=10)
    axes[1, 0].set_ylabel(f"FGSM eps={eps_demo}", fontsize=10)
    fig.suptitle(f"Same images + L-inf FGSM (eps={eps_demo}) -> flipped labels, lower confidence", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "clean_vs_adversarial.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPS)
    ap.add_argument("--eps-demo", type=float, default=0.08, help="epsilon for the demo grid")
    ap.add_argument("--n-images", type=int, default=64, help="images for the sweep")
    ap.add_argument("--epochs", type=int, default=6, help="epochs for the offline target")
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--pretrained", action="store_true", help="attack torchvision ResNet-18 (online)")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    target_name = "SmallCNN (synthetic, offline)"
    used_pretrained = False

    if args.pretrained:
        built = _build_pretrained_target(device, args.n_images)
        if built is not None:
            model, x, y, classes = built
            target_name = "ResNet-18 (ImageNet, pretrained)"
            used_pretrained = True

    if not used_pretrained:
        model, classes = _build_offline_target(device, args.epochs)
        x_all, y_all = make_synthetic(n_per_class=args.n_images // 4 + 1, seed=7)
        x, y = x_all[: args.n_images].to(device), y_all[: args.n_images].to(device)
        # use the model's clean predictions on correctly-classified images as labels
        with torch.no_grad():
            pred = model(x).argmax(1)
        mask = pred == y
        x, y = x[mask], y[mask]

    print(f"target: {target_name}; attacking {x.size(0)} images")

    fgsm_acc, fgsm_conf = _sweep(model, x, y, args.epsilons, fgsm_perturb)
    pgd_acc, pgd_conf = _sweep(
        model, x, y, args.epsilons,
        lambda m, xx, yy, e: pgd_perturb(m, xx, yy, e, steps=args.pgd_steps),
    )

    for e in sorted(fgsm_acc):
        print(
            f"  eps={e:<5} FGSM acc={fgsm_acc[e] * 100:5.1f}% conf={fgsm_conf[e] * 100:5.1f}%"
            f"   PGD acc={pgd_acc[e] * 100:5.1f}%"
        )

    curve = _plot_curves(fgsm_acc, pgd_acc, fgsm_conf, target_name)
    grid = _plot_grid(model, x, y, classes, args.eps_demo)

    eps_keys = sorted(fgsm_acc)
    metrics = {
        "project": "p3-pretrained-foolbox",
        "summary": (
            f"Inference-time FGSM/PGD on {target_name}: clean accuracy "
            f"{fgsm_acc[eps_keys[0]] * 100:.0f}% -> FGSM {fgsm_acc[eps_keys[-1]] * 100:.0f}% "
            f"and PGD {pgd_acc[eps_keys[-1]] * 100:.0f}% at eps={eps_keys[-1]}; "
            f"mean true-class confidence drops "
            f"{fgsm_conf[eps_keys[0]] * 100:.0f}% -> {fgsm_conf[eps_keys[-1]] * 100:.0f}%."
        ),
        "target": target_name,
        "used_pretrained": used_pretrained,
        "seed": 42,
        "n_images": int(x.size(0)),
        "pgd_steps": args.pgd_steps,
        "clean_accuracy": fgsm_acc[eps_keys[0]],
        "fgsm_accuracy_by_epsilon": {str(e): fgsm_acc[e] for e in eps_keys},
        "pgd_accuracy_by_epsilon": {str(e): pgd_acc[e] for e in eps_keys},
        "fgsm_true_conf_by_epsilon": {str(e): fgsm_conf[e] for e in eps_keys},
        "eps_demo": args.eps_demo,
        "figures": [str(curve.relative_to(PROJECT)), str(grid.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {curve.relative_to(PROJECT)}")
    print(f"wrote {grid.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
