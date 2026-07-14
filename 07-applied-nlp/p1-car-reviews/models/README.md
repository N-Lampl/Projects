# Models

No weights are committed here (`models/` is git-ignored). The HuggingFace models
below are downloaded by `transformers` on first use and cached under
`~/.cache/huggingface`.

## Sentiment backends (`--backend hf --model ...`)

| key | model | scale | notes |
|---|---|---|---|
| `nlptown` (default) | `nlptown/bert-base-multilingual-uncased-sentiment` | 1-5 stars | ~420 MB. Native 1-5 output → directly comparable to the ground-truth `Rating`. The headline model. |
| `sst2` | `distilbert-base-uncased-finetuned-sst-2-english` | binary | ~260 MB, ~2× faster on CPU. Used by the `@slow` test. |
| `cardiff` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3-class | ~500 MB. Adds a neutral class. |

`--backend stub` uses a deterministic lexicon scorer (no weights, no network) -
the offline/CI path.

## Optional summarizer (`--summaries abstractive`)

`sshleifer/distilbart-cnn-12-6` (~1.2 GB) - CPU-runnable but slow, so it only ever
summarizes the top-N most-reviewed models. The default `--summaries extractive`
needs no model at all.

> Authorized use only: public pretrained checkpoints used for analysis. See
> [../../../ETHICS.md](../../../ETHICS.md).
