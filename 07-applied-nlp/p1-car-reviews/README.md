# p1 · Car reviews by brand & model — HuggingFace sentiment on CPU

<!-- fullrun-banner -->
> **Real data.** Committed results are the **full run** over all 36,715 Edmunds car
> reviews: `make run-full` (nlptown 1-5 sentiment on CPU). `make run` is the fast
> `--sample 3000` path; `make test` uses an offline lexicon stub (no downloads).

Point a pretrained HuggingFace sentiment model at 36,715 real car reviews, break the
sentiment down **by brand and by specific model**, and — because the dataset ships a
1-5 star `Rating` — **validate the predictions against that ground-truth label** instead
of just asserting them. Then go deeper: aspect-based sentiment (performance, comfort,
reliability, price, fuel economy, safety), the distinctive keywords per model, the
topics reviewers actually talk about, and an extractive summary per model. Everything
runs on a **CPU-only laptop** — no GPU, no external LLM API.

**Authorized use only.** A public dataset and public pretrained checkpoints used for
analysis. See [../../ETHICS.md](../../ETHICS.md).

## The idea

The dataset ([`florentgbelidji/car-reviews`](https://huggingface.co/datasets/florentgbelidji/car-reviews))
is 36,715 usable Edmunds reviews across **3 brands** — Toyota (17,810), Nissan (11,405),
BMW (7,500) — and **116 specific models**, each review carrying free text plus a 1-5 star
`Rating`. Two things make this more than a toy sentiment demo:

1. **Ground-truth validation.** The `Rating` column is a real label, so a 1-5-star
   sentiment model (`nlptown/bert-base-multilingual-uncased-sentiment`) can be scored
   against it directly — exact accuracy, **±1 accuracy**, MAE, Spearman ρ, and a 5×5
   confusion matrix. Car reviews skew positive, so ±1 accuracy and ρ tell the honest
   story better than exact match.
2. **Honest small-sample stats.** With 116 models the long tail is thin, so brand/model
   rankings use a `min_reviews` gate **and** empirical-Bayes shrinkage toward the global
   mean — a model with three glowing reviews can't leapfrog the Camry.

The `Vehicle_Title` field (`"2015 Toyota Camry LE Sedan"`) is parsed into
year/make/model with a curated, multi-word-aware make list (100% coverage here). The
whole pipeline has an **offline path**: a deterministic lexicon "stub" backend + a seeded
synthetic review generator, so tests and CI run with zero downloads.

## Run it

```bash
make run            # fast --sample 3000 (downloads nlptown + dataset on first use)
make run-full       # the committed artifact: ALL 36,715 reviews (~30-40 min, CPU)
make test           # fast offline smoke tests (stub backend, -m 'not slow')
make run ARGS='--model sst2 --summaries abstractive'   # swap model / summarizer
```

Outputs land in [results/](results/):
- `figures/sentiment_vs_rating_confusion.png` — the **money plot**: predicted sentiment
  star vs. actual `Rating` (proves the method against ground truth).
- `figures/brand_sentiment_ranking.png` — Toyota vs. Nissan vs. BMW mean sentiment.
- `figures/model_sentiment_ranking.png` — top models by sentiment (≥30 reviews).
- `figures/aspect_sentiment_heatmap.png` — aspects × brands.
- `figures/topics_prevalence.png` — discovered topics and their share.
- `metrics.json` — validation metrics, rankings, aspects, keywords, topics, summaries.

## What the result shows

On all 36,715 reviews, nlptown's predicted star matches the actual `Rating` **within ±1
for 87% of reviews** (exact 51%, MAE 0.69, Spearman ρ = 0.64) — the sentiment model
tracks how owners actually scored their cars. Exact-match is a coin-flip *because* the
hard call is "is this a 4 or a 5?", and the 5×5 confusion shows exactly that: genuine
3-star reviews are most often read as 4-star and true 4s bleed into 5s — the known
positive-skew of consumer reviews. That's why **±1 accuracy and ρ are the honest
headline**, not exact match.

Across the 3 brands, **BMW edges the top (0.73 mean sentiment), just ahead of Toyota
(0.72), with Nissan a step behind (0.67)** — but the aspect breakdown is the more
interesting story: BMW owners rate **performance, comfort and safety** highest, while
**Toyota wins on reliability, price and fuel economy** (and reliability/price are the
lowest-scoring aspects overall, ~0.56, vs ~0.68 for performance/comfort). At the model
level (74 models with ≥30 reviews), enthusiast cars top the chart (**BMW M5 0.87, Toyota
Supra 0.88**) and high-volume commuters anchor the bottom (**Nissan Rogue 0.57,
Altima/Sentra 0.63**). The distinctive-keyword pass backs this up — *handling*/*fun* for
the BMW 3 Series, *truck*/*trd* for the Tacoma, *transmission* for the Altima — and topic
modeling surfaces what reviewers actually dwell on (mileage/MPG, long-term miles,
fun-to-drive).

## Interview story (3 sentences)

> I ran a pretrained HuggingFace sentiment model over 36,715 real car reviews entirely on
> CPU, broke the sentiment down by brand and by 116 specific models, and — crucially —
> validated the predictions against the dataset's own 1-5 star rating rather than trusting
> them (±1 accuracy ~87%, Spearman ~0.64). I handled the parts that bite in
> practice: parsing messy `Vehicle_Title` strings into make/model, shrinking small-sample
> rankings toward the global mean so a handful of reviews can't top the chart, and keeping
> the whole thing reproducible with an offline stub backend for CI. It shows I can take an
> NLP model from the Hub to a validated, honest, end-to-end analysis — not just call a
> pipeline and eyeball the output.

## Layout

```
src/car_reviews/  utils · data (HF loader + offline fallback) · synthetic · parsing
                  sentiment (stub/HF backends) · aggregate (shrinkage) · aspects
                  keywords (TF-IDF) · topics (NMF) · summarize · evaluate · plots
scripts/          run_analysis.py  (load -> parse -> score -> validate -> figures + metrics.json)
tests/            test_smoke.py  (offline stub + synthetic; one @slow real-model test)
results/          figures/*.png + metrics.json  (committed)
data/ models/     git-ignored (HF dataset + weights downloaded by code)
```

## References

- Dataset: [`florentgbelidji/car-reviews`](https://huggingface.co/datasets/florentgbelidji/car-reviews)
  (Edmunds consumer reviews) — see [data/README.md](data/README.md).
- Model: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
  (1-5 star sentiment); optional `distilbert-sst-2` and `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- Monroe, Colaresi & Quinn. *Fightin' Words: Lexical Feature Selection for Identifying
  Content.* (the distinctive-term intuition behind the keyword pass.)
