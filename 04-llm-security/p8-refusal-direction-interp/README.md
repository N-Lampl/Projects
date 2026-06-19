# p8 · Refusal direction (mechanistic interp of "abliteration")

Refusal in aligned LLMs is, empirically, mediated by **a single linear direction** in the residual
stream (Arditi et al., 2024). This project reproduces the *methodology* — extract that direction,
ablate it at inference time, and measure what happens to refusals vs capability — as **safety /
interpretability research**, entirely on CPU, with **no model weights downloaded** by default.

> ⚠️ **Authorized use only — analysis, not a jailbreak release.** The committed artifact is an
> *analysis* (a direction vector, metrics, figures). This project **never** writes out or
> redistributes a modified ("abliterated") model. The default path runs against a **synthetic**
> self-built model; the optional real-model path runs only against an open-weight model **you** load
> yourself. Understanding *how* a safety behaviour is encoded is what lets you make it more robust.
> See [../../ETHICS.md](../../ETHICS.md).

## The idea

Treat the last-token residual activation `h ∈ ℝ^d` as the model's "state of mind" before it answers.
Across many prompts, the difference between *will-refuse* and *will-answer* states concentrates on one
axis. Three steps:

**1. Extract** — difference-in-means refusal direction:
```
r = mean(h | harmful)  −  mean(h | harmless)        r̂ = r / ‖r‖
```

**2. Ablate** — remove that axis from every activation (orthogonal projection), applied live via a
forward hook on the residual stream ("abliteration"):
```
h' = h − (h · r̂) r̂
```

**3. Measure** — refusal rate before vs after, **plus a capability-retention proxy**, to show the
intervention is *surgical*: refusals collapse while general capability is largely preserved.

The whole method is ~15 lines ([src/refusal_interp/direction.py](src/refusal_interp/direction.py)):
```python
r = h_harmful.mean(0) - h_harmless.mean(0)        # extract
r_hat = r / r.norm()
h_ablated = h - (h @ r_hat).unsqueeze(-1) * r_hat # ablate (orthogonal projection)
```

### Why a synthetic model is a *fair* default

Downloading + running an instruct model to harvest activations is GPU-preferred and >50 MB, so the
default path simulates it. [src/refusal_interp/synthetic.py](src/refusal_interp/synthetic.py) builds a
frozen toy model whose residual stream contains a **planted** refusal axis `r_true` (orthogonal to a
separate "content/capability" subspace) plus noise. The experiment is honest, not circular:
- difference-in-means must **recover** `r_true` from data alone — we report `|cos(r̂, r_true)|`;
- a frozen behaviour head computes `P(refuse)` (keyed on the refusal axis) and `P(capable)` (keyed on
  the orthogonal content subspace), so we can show ablation is surgical rather than asserting it.

## Run it

```bash
# from this folder; uses uv if installed, else system python3
make run     # synthetic extract -> ablate -> measure -> figures + metrics.json (CPU, seconds)
make test    # fast smoke tests
make run ARGS=                 # (defaults to 128 harmful + 128 harmless prompts)
python3 scripts/run_analysis.py --n-harmful 256 --n-harmless 256
```

Outputs land in [results/](results/):
- `figures/refusal_vs_capability.png` — the **money plot**: refusal rate collapses, capability held.
- `figures/projection_histograms.png` — activations on the refusal axis, before/after ablation.
- `metrics.json` — recovery cosine + refusal/capability before & after (committed as evidence).

## Real-model path (optional, GPU-preferred)

[src/refusal_interp/real_model.py](src/refusal_interp/real_model.py) implements the same pipeline on a
real open-weight instruct model via `transformers` (imported lazily, so the package still imports
without it). Provide AdvBench-style + Alpaca-style prompt sets (see [data/README.md](data/README.md)),
collect last-token residuals per layer, then reuse `extract_refusal_direction` /
`make_ablation_hook` from the core module.

- **CPU path:** works with a tiny model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) on ~64 prompts/class;
  slow but functional.
- **Colab/GPU path:** `pip install -r requirements.txt` with the `real` extras uncommented, pick a
  1–3B instruct model, sweep the layer index, and choose the layer whose `r̂` best separates the sets.

It still only produces **analysis** — never a saved modified model.

## What the result shows

Difference-in-means recovers the planted refusal axis almost perfectly (`|cos| ≈ 1.0`). A single
rank-1 ablation drives the refusal rate from ~100% to ~0% on held-out harmful prompts while the
capability proxy on harmless prompts barely moves — the empirical signature behind real abliteration.
The security takeaway: **safety alignment that lives in one linear direction is brittle**, which is
exactly why refusal-direction analysis matters for building more robust guardrails.

## Interview story (3 sentences)

> I reproduced the "refusal is a single direction" result as interpretability research: extract the
> axis by difference-in-means over harmful vs harmless residual activations, ablate it with a forward
> hook, and measure that refusals collapse while a capability proxy is retained. I built it on a
> synthetic model with a *planted* direction so the whole extract→ablate→measure pipeline runs offline
> on CPU and I can verify the extractor actually recovers the ground-truth axis. The point is
> defensive — if a safety behaviour is one rank-1 feature, it's fragile, which tells you where to
> harden alignment.

## Layout

```
src/refusal_interp/  utils.py (seeds) · synthetic.py (toy model + planted axis)
                     direction.py (extract/ablate/measure + hook) · real_model.py (optional)
scripts/             run_analysis.py  (synthetic pipeline -> figures + metrics.json)
tests/               test_smoke.py    (fast invariants + one @slow end-to-end)
results/             figures/*.png + metrics.json  (committed)
data/ models/        git-ignored (synthetic by default; optional real model not committed)
```

## References

- Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. *Refusal in Language Models Is Mediated by
  a Single Direction.* NeurIPS 2024. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
- Zou et al. *Universal and Transferable Adversarial Attacks on Aligned Language Models* (AdvBench),
  2023. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043).
- Taori et al. *Stanford Alpaca: An Instruction-following LLaMA model*, 2023.
- nostalgebraist, *interpreting GPT: the logit lens*, 2020 (residual-stream readout intuition).
