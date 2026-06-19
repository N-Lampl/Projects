# data/ (git-ignored)

This project uses **no external dataset** on the default path. The training corpus is
**synthetic** and generated deterministically at runtime by
[`src/llm_privacy/corpus.py`](../src/llm_privacy/corpus.py) (`build_corpus`): thousands of
fake log lines plus a handful of inserted "canary" secrets. Nothing is downloaded and
nothing in this folder is committed.

- **Default corpus:** synthetic, generated in-memory (seed 42). License: N/A (we make it).
- **Download command:** none — just run `make detect`.

## Optional enhanced path (GPT-2)

[`scripts/gpt2_exposure.py`](../scripts/gpt2_exposure.py) downloads pretrained **GPT-2**
weights (~500MB) via `transformers`. That is opt-in only:

```bash
pip install "transformers>=4.40"
python scripts/gpt2_exposure.py
```

Weights cache under `~/.cache/huggingface`, not in this repo.
