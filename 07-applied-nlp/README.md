# 07 · Applied NLP

The security tracks (00-06) are the pivot; this track is the other half of the
story — **applied NLP / data science**, the strength I'm building the security
half on top of. Same engineering bar as everywhere else: a self-contained
project, a reproducible `make run`, committed figures + `metrics.json`, an offline
fallback, and a short interview story.

Authorized use only — see [../ETHICS.md](../ETHICS.md). Datasets and model weights
are NOT committed; they're downloaded by code. Each project also ships an offline
synthetic fallback so tests / CI need no network.

## Projects — and what each runs on

| Project | What it does | Data |
|---|---|---|
| `p1-car-reviews/` | HuggingFace sentiment over 36,984 Edmunds car reviews, broken down **by brand and specific model**, validated against the 1-5 star rating; plus aspect sentiment, distinctive keywords, topics, and per-model summaries | **REAL** [`florentgbelidji/car-reviews`](https://huggingface.co/datasets/florentgbelidji/car-reviews) (offline synthetic fallback) |

## Notes

- **HuggingFace-first, CPU-only.** A pretrained `transformers` sentiment pipeline
  runs entirely on CPU (batched, length-sorted, truncated). No GPU, no external
  LLM API.
- **Validated against ground truth.** The dataset ships a 1-5 `Rating`, so
  predicted sentiment is checked against a real label (±1 accuracy, MAE, Spearman)
  rather than asserted.
- **Honest small-sample stats.** With 600+ models, rankings use a `min_reviews`
  gate and empirical-Bayes shrinkage toward the global mean so a single 5-star
  review can't top the chart.
