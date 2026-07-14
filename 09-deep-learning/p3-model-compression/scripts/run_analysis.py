#!/usr/bin/env python3
"""Run the compression study -> results/metrics.json + figures.

Trains a baseline teacher MLP, then produces three compressed variants -
magnitude pruning, post-training dynamic quantization, and a distilled student -
benchmarks each on accuracy / size / latency / sparsity, and draws the
accuracy-vs-size Pareto scatter plus the per-variant latency bar chart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compression import (  # noqa: E402
    Student,
    Teacher,
    benchmark,
    count_params,
    distill,
    dynamic_quantize,
    magnitude_prune,
    make_blobs,
    metrics_dict,
    set_seed,
    train_classifier,
)
from compression.plots import (  # noqa: E402
    plot_latency_vs_variant,
    plot_pareto_accuracy_vs_size,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--features", type=int, default=40)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--class-sep", type=float, default=1.1)
    ap.add_argument("--noise", type=float, default=1.6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--prune-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    data = make_blobs(
        n_samples=args.samples,
        n_features=args.features,
        n_classes=args.classes,
        class_sep=args.class_sep,
        noise=args.noise,
        seed=args.seed,
    )

    # Baseline teacher.
    teacher = Teacher(data.n_features, data.n_classes)
    train_classifier(teacher, data, epochs=args.epochs, seed=args.seed)

    # Three compressed variants.
    pruned = magnitude_prune(teacher, fraction=args.prune_frac)
    quantized = dynamic_quantize(teacher)
    student = Student(data.n_features, data.n_classes)
    distill(teacher, student, data, epochs=args.epochs + 10, seed=args.seed)

    models = {
        "baseline": teacher,
        "pruned": pruned,
        "quantized": quantized,
        "distilled": student,
    }
    variants = {name: metrics_dict(benchmark(m, data)) for name, m in models.items()}

    fig1 = plot_pareto_accuracy_vs_size(variants, FIGURES / "pareto_accuracy_vs_size.png")
    fig2 = plot_latency_vs_variant(variants, FIGURES / "latency_vs_variant.png")

    base = variants["baseline"]
    summary_str = (
        f"A {count_params(teacher):,}-param teacher hits "
        f"{base['accuracy'] * 100:.1f}% accuracy at {base['size_mb']:.2f} MB. "
        f"Magnitude pruning to {variants['pruned']['sparsity'] * 100:.0f}% sparsity keeps "
        f"{variants['pruned']['accuracy'] * 100:.1f}% accuracy; dynamic int8 quantization "
        f"shrinks the model to {variants['quantized']['size_mb']:.2f} MB "
        f"({base['size_mb'] / variants['quantized']['size_mb']:.1f}x smaller); and a "
        f"{count_params(student):,}-param distilled student "
        f"({count_params(teacher) / count_params(student):.0f}x fewer params) reaches "
        f"{variants['distilled']['accuracy'] * 100:.1f}% at "
        f"{variants['distilled']['size_mb']:.3f} MB and "
        f"{variants['distilled']['latency_ms']:.2f} ms/forward."
    )

    metrics = {
        "project": "p3-model-compression",
        "summary": summary_str,
        "data_source": data.source,
        "seed": args.seed,
        "teacher_params": count_params(teacher),
        "student_params": count_params(student),
        "prune_fraction": args.prune_frac,
        "variants": variants,
        "figures": [f"results/figures/{p.name}" for p in (fig1, fig2)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary_str)
    print(f"[ok] wrote {RESULTS / 'metrics.json'} + 2 figures")


if __name__ == "__main__":
    main()
