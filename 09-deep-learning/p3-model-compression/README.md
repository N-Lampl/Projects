# p3 · Model compression — pruning, quantization & distillation on CPU

> **Synthetic-by-default, honest trade-offs.** Committed results come from one
> teacher MLP compressed three ways on a deterministic synthetic classification
> problem, so the accuracy / size / latency numbers are *measured*, not asserted.
> `make run` regenerates them; `make test` runs offline with no torchvision. A
> `@slow` test trains the same net on real MNIST.

A trained network is usually far bigger and slower than it needs to be. This
project takes one baseline classifier and squeezes it three ways —
**magnitude pruning**, **post-training dynamic (int8) quantization**, and
**knowledge distillation** into a much smaller student — then benchmarks every
variant on the four axes that actually matter for shipping a model: **accuracy**,
**serialized size (MB)**, **CPU latency (ms/forward)** and **sparsity**. The point
is the *Pareto frontier*: each technique buys efficiency at a different cost, and
you can only see the trade-off by measuring all four together. CPU-only, fully
deterministic.

**Authorized use only.** Synthetic data used for education. See
[../../ETHICS.md](../../ETHICS.md).

## The idea

The baseline ([`models.py`](src/compression/models.py)) is a wide two-hidden-layer
teacher MLP. Three compressors ([`compress.py`](src/compression/compress.py)) each
attack a different kind of waste:

- **Magnitude pruning** — zero the smallest-|weight| fraction of every `Linear`
  layer (`torch.nn.utils.prune`, masks baked in with `prune.remove`). Most weights
  in an over-parameterised net barely matter, so 80% of them can go with little
  accuracy loss. This buys **sparsity and speed**; on a dense state_dict it does
  *not* shrink the file (the zeros are still stored) — an honest caveat that
  motivates sparse formats.
- **Dynamic quantization** — `quantize_dynamic` converts `Linear` weights to
  **int8** at inference time. This roughly **4x shrinks** the serialized model
  while keeping accuracy; on tiny CPU batches the quant/dequant dispatch can add
  latency, which is exactly the kind of trade-off the benchmark surfaces.
- **Knowledge distillation** — train a small student on the teacher's **soft
  targets** (KL on softened logits + a hard-label term). The student learns the
  teacher's "dark knowledge" and reaches near-teacher accuracy with **~50x fewer
  parameters**, the smallest and fastest variant of all.

Everything is benchmarked in [`benchmark.py`](src/compression/benchmark.py):
accuracy on a held-out split, size from the serialized `state_dict`, latency as
the median of many `time.perf_counter` forward passes, and sparsity as the zero
fraction of the weights.

## Run it

```bash
make run     # train teacher + prune/quantize/distill + benchmark -> figures + metrics.json
make test    # fast offline smoke tests (-m 'not slow'); no torchvision needed
make run ARGS='--classes 8 --noise 1.2 --prune-frac 0.9'
```

Outputs land in [results/](results/):
- `figures/pareto_accuracy_vs_size.png` — the **money plot**: test accuracy vs
  serialized size for baseline / pruned / quantized / distilled.
- `figures/latency_vs_variant.png` — median CPU latency per variant.
- `metrics.json` — per-variant `{accuracy, size_mb, latency_ms, sparsity}`.

## What the result shows

One teacher (≈79k params, 0.30 MB), compressed three ways on a 10-class synthetic
problem:

| variant | accuracy | size (MB) | latency (ms) | sparsity |
|---|---|---|---|---|
| baseline (teacher) | 96.4% | 0.303 | 0.34 | 0% |
| pruned (80%) | 91.5% | 0.303 | 0.26 | 80% |
| quantized (int8) | 96.5% | **0.081** | 1.97 | 0% |
| **distilled (student)** | **96.7%** | **0.008** | **0.06** | 0% |

Each technique lands on a different corner of the frontier. **Pruning** zeros 80%
of the weights and stays within ~5 points of the teacher — mostly a speed/sparsity
win (the dense file doesn't shrink). **Quantization** is the size lever: int8 cuts
the serialized model **~3.7x** with essentially no accuracy loss. **Distillation**
is the outright winner here — a ~48x-smaller student matches the teacher's accuracy
at **~40x less disk and the lowest latency**, because soft targets let a tiny net
imitate a big one. The headline: for this task you don't need the big model at
inference time at all — a distilled student ships the same accuracy for a fraction
of the cost, and quantization is the cheap drop-in when you must keep the original
architecture.

## Interview story (3 sentences)

> I took one baseline classifier and compressed it three ways — magnitude pruning,
> post-training int8 dynamic quantization, and knowledge distillation into a
> smaller student — then benchmarked accuracy, serialized size, CPU latency and
> sparsity to plot the actual Pareto frontier. Quantization shrank the model ~4x
> with no accuracy loss, pruning zeroed 80% of the weights for a speed win, and a
> ~50x-smaller distilled student matched the teacher's accuracy at the lowest size
> and latency of all. It shows I understand *why* each technique trades a different
> resource — and that "smaller model" is a multi-axis measurement, not a slogan.

## Layout

```
src/compression/  utils · data (synthetic blobs; optional MNIST via torchvision)
                  models (Teacher / Student MLPs) · train (short CPU loops)
                  compress (magnitude_prune · dynamic_quantize · distill)
                  benchmark (accuracy · size_mb · latency_ms · sparsity) · plots
scripts/          run_analysis.py  -> results/figures + metrics.json
tests/            test_smoke.py  (offline; @slow MNIST cross-check)
results/          figures/*.png + metrics.json  (committed)
data/ models/     git-ignored (synthetic data; models trained in memory)
```

## References

- Han, Mao & Dally (2016), *Deep Compression* — pruning + quantization + coding as
  a combined pipeline.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network* —
  soft targets and temperature, the basis of the distillation loss here.
- PyTorch docs — `torch.nn.utils.prune` and `torch.quantization.quantize_dynamic`.
