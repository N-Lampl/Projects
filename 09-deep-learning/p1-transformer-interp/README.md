# p1 · Transformer internals & mechanistic interpretability - a tiny model from scratch on CPU

> **Synthetic-by-default, trained in-memory.** Committed results come from a
> 2-layer decoder-only transformer trained from scratch on a synthetic
> **induction task**, so the circuit it learns is *known* and can be probed, not
> asserted. `make run` regenerates everything; `make test` runs offline with no
> `transformers`. A `@slow` test cross-checks the story against `distilgpt2`.

Modern transformers aren't black boxes all the way down - specific
**circuits** implement specific behaviours. The most famous is the **induction
head**: the mechanism behind in-context learning, which on seeing token `A`
looks back to the previous `A` and copies whatever followed it (`[A][B] … [A] →
[B]`). This project trains a tiny transformer from scratch until an induction
head *emerges*, then runs three classic mechanistic-interpretability tools on it -
an **induction-head score**, the **logit lens**, and **activation patching** -
to read out and localize the circuit. Pure PyTorch, CPU-only, fully
deterministic.

**Authorized use only.** Synthetic data used for education. See
[../../ETHICS.md](../../ETHICS.md).

## The idea

The model ([`model.py`](src/transformer_interp/model.py)) is a small pre-norm
decoder-only transformer - token + positional embeddings, `n_layers=2` blocks of
causal multi-head attention + MLP, a final LayerNorm and an unembedding
(`d_model=64, n_heads=4, d_mlp=128`). Its `forward` optionally returns a **cache**
of per-head attention weights and the residual stream after each layer, and
accepts a `resid_patch` argument to *overwrite* an activation - the read/patch
pair every interpretability method needs.

The task ([`task.py`](src/transformer_interp/task.py)) draws a random sequence and
**duplicates** it, `[seq | seq]`. In the repeated half the only reliable way to
predict the next token is the induction rule, so training pressure
([`train.py`](src/transformer_interp/train.py)) forces a head to learn it. Then
three analyses ([`interp.py`](src/transformer_interp/interp.py)):

- **Induction-head score** - per head, the attention mass placed on the
  `previous-occurrence + 1` position; the max over heads is the headline.
- **Logit lens** - project each layer's residual stream through the unembedding
  and measure next-token accuracy; it should sharpen with depth.
- **Activation patching** - splice a *clean* residual activation into a
  *corrupted* run and measure how much of the correct-token logit is recovered,
  localizing where the computation lives.

## Run it

```bash
make run     # train + induction score + logit lens + activation patching -> figures + metrics.json
make test    # fast offline smoke tests (-m 'not slow'); no transformers needed
make run ARGS='--steps 600 --seed 7'
```

Outputs land in [results/](results/):
- `figures/attention_induction.png` - the **money plot**: the discovered head's
  attention, showing the off-diagonal induction stripe.
- `figures/logit_lens.png` - next-token accuracy + correct-token logit by depth.
- `figures/activation_patching.png` - fraction of the logit gap recovered per layer.
- `metrics.json` - val loss, induction score, logit-lens trend, patching effect.

## What the result shows

A 2-layer transformer trained from scratch (seed 42) on the induction task:

| probe | signal |
|---|---|
| next-token val loss | **0.002** |
| max induction-head score | **0.77** (layer 0, head 3) |
| logit-lens accuracy: embedding → final layer | **0.02 → 1.00** |
| activation patching (best layer) | **100% of the logit gap recovered** |

The model solves the task almost perfectly (val loss 0.002), and it does so with a
real **induction head** - one head puts **77%** of its attention on exactly the
previous-occurrence+1 token, versus ~2% for a random baseline. The **logit lens**
confirms the computation is genuinely *built up with depth*: reading predictions
straight off the embedding is chance (2% accuracy), but by the final residual
stream it is perfect (100%). And **activation patching** localizes the circuit -
splicing the clean final-layer residual into a corrupted run recovers **100%** of
the correct-token logit gap, while patching the raw embedding recovers essentially
none. Three independent tools, one consistent story about *where* and *how* the
induction circuit lives.

## Interview story (3 sentences)

> I trained a 2-layer transformer from scratch until an induction head emerged,
> then localized the circuit with three standard mechanistic-interpretability
> tools I implemented myself: an induction-head attention score, the logit lens,
> and activation patching over a clean/corrupt pair. The head put 77% of its
> attention on the induction source token, the logit lens showed accuracy rising
> from 2% at the embedding to 100% at the final layer, and patching recovered
> 100% of the correct-token logit gap at the right layer and almost none at the
> wrong one - then a `@slow` test confirms `distilgpt2` has the same induction-like
> head. It shows I understand attention, the residual stream, and causal
> interventions - not just how to call a model's `forward`.

## Layout

```
src/transformer_interp/  utils · task (synthetic induction sequences)
                         model (decoder-only transformer + attention/resid hooks)
                         train (from-scratch Adam training on the induction task)
                         interp (induction score · logit lens · activation patching) · plots
scripts/                 run_analysis.py  -> results/figures + metrics.json
tests/                   test_smoke.py  (offline torch training; @slow distilgpt2 cross-check)
results/                 figures/*.png + metrics.json  (committed)
data/ models/            git-ignored (synthetic task; model trained in memory)
```

## References

- Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits* - the
  attention/residual-stream decomposition this reads out.
- Olsson et al. (2022), *In-context Learning and Induction Heads* - defines the
  induction head and the score used here.
- nostalgebraist (2020), *interpreting GPT: the logit lens* - projecting
  intermediate residual streams through the unembedding.
- Meng et al. (2022), *Locating and Editing Factual Associations (ROME)* - causal
  tracing / activation patching to localize a computation.
