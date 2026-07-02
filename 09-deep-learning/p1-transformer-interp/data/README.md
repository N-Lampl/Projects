# Data

This project runs on a **synthetic induction task** by default, so tests and CI
need no network. Nothing here is committed (`data/` is git-ignored except this
README).

## Default: synthetic induction sequences (offline, deterministic)

[`../src/transformer_interp/task.py`](../src/transformer_interp/task.py) draws a
random sequence over a small vocab and then **duplicates** it: `[a b c ... | a b
c ...]`. In the repeated half the only reliable way to predict the next token is
the **induction rule** — look back to the previous occurrence of the current
token and copy whatever followed it. A large vocab makes accidental repeats rare
so the previous occurrence is unique and the induction signal is clean. The task
also exposes next-token targets and a `repeat_mask`, so heads are scored only
where the rule is well-defined.

## Optional: distilgpt2 cross-check

The `@slow` test lazily loads **distilgpt2** (via `transformers`) on a
repeated-token prompt and checks that a real GPT-2-family model has an
induction-like head. `transformers` is optional (`pip install transformers`);
the default path needs only numpy/torch/matplotlib.

> Authorized use only: synthetic data used for education. See
> [../../../ETHICS.md](../../../ETHICS.md).
