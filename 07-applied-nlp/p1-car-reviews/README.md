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
make finetune       # fine-tune DistilBERT on the star labels (~2 h CPU) -> models/ (see below)
make improve        # baseline vs. calibrated vs. fine-tuned on the held-out test set
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
tracks how owners actually scored their cars. Exact-match is only a coin-flip because the
dataset is overwhelmingly positive — **86% of reviews are 4-5 stars** — while this
general-domain model is *more conservative* than car owners: the 5×5 confusion shows it
**under-rates**, pushing genuine 3-star reviews down to 1-2 and spreading true 4s and 5s
rather than matching them. The *ordering* is already decent (that's the ρ = 0.64); the
disagreement is a systematic **calibration gap** — exactly what the improvement work below
closes. That's why **±1 accuracy and ρ are the honest headline**, not exact match.

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

## Making it better: calibrate, then fine-tune

The baseline's weak spot is **calibration, not capability** — ρ = 0.64 says the ordering is
already there, the model is just miscalibrated to this very-positive domain. So the
improvement goes in two measured steps, each scored on the **same held-out 5,000-review test
set** (seeded, never trained on):

**1 · Calibrate (seconds, CPU).** Fit a multinomial-logistic model that maps nlptown's five
class-probabilities to the actual `Rating` on a train split, then apply it to the test set —
i.e. stack a light classifier on the *frozen* model to undo its domain bias
([`improve.py`](src/car_reviews/improve.py)):

| | exact | ±1 acc | MAE | Spearman |
|---|---|---|---|---|
| baseline (nlptown) | 0.50 | 0.87 | 0.70 | 0.635 |
| **+ calibration** | 0.51 | **0.93** | **0.65** | 0.632 |

Calibration lifts **±1 accuracy by 6 points and cuts MAE** — it re-centres the systematic
under-rating for essentially zero compute. But **Spearman doesn't move**: recalibrating
re-weights the model's *existing* signal, it can't manufacture new ranking power. For that
you need better representations.

**2 · Fine-tune DistilBERT (~2 h, CPU).** Fine-tune `distilbert-base-uncased` as a 5-class
head on the star labels (10,000 train reviews, 2 epochs) so the model learns car-review
language end-to-end ([`finetune.py`](src/car_reviews/finetune.py)). Same held-out test set:

| | exact | ±1 acc | MAE | Spearman |
|---|---|---|---|---|
| baseline (nlptown) | 0.50 | 0.87 | 0.70 | 0.635 |
| + calibration | 0.51 | 0.93 | 0.65 | 0.632 |
| **fine-tuned** | **0.68** | **0.97** | **0.42** | **0.666** |

The fine-tune moves **everything, including Spearman**: exact accuracy jumps from 51% to
**68%**, ±1 accuracy to **97%**, MAE nearly halves (0.70 → **0.42**), and ρ rises to
**0.67**. Unlike calibration it improves the *ranking* too — because it learned car-review
representations end-to-end rather than re-weighting a general model's guesses. That's the
payoff of ~2 h of CPU fine-tuning on 10k in-domain labels: a model that agrees with owners
far more often **and** orders them better.

`results/figures/baseline_vs_improved.png` puts all three side by side and
`results/improvement.json` has the full numbers. Reproduce with `make finetune && make improve`
(the calibrated column alone needs no training). Fine-tuned weights land in `models/`
(git-ignored); only the comparison figure + JSON are committed.

## Interview story (3 sentences)

> I ran a pretrained HuggingFace sentiment model over 36,715 real car reviews on CPU, broke
> sentiment down by brand and by 116 specific models, and validated every prediction against
> the dataset's own 1-5 star rating instead of trusting it. Reading the confusion matrix I saw
> the model wasn't weak, it was **miscalibrated** to a very-positive domain — ranking was fine
> (ρ = 0.64) but it systematically under-rated — so I improved it in two measured steps: a
> zero-compute logistic **calibration** that lifted ±1 accuracy from 87% to 93%, then a
> **DistilBERT fine-tune** on the star labels that pushed exact accuracy from 51% to 68% and
> halved MAE while improving the ranking too. It shows I can diagnose *why* a
> model underperforms and improve it deliberately, keep the whole thing reproducible and
> honest (seeded held-out test, offline stub backend for CI) — not just call a pipeline and
> eyeball the output.

## Layout

```
src/car_reviews/  utils · data (HF loader + offline fallback) · synthetic · parsing
                  sentiment (stub/HF backends, predict_proba) · aggregate (shrinkage) · aspects
                  keywords (TF-IDF) · topics (NMF) · summarize · evaluate · plots
                  improve (splits + Calibrator) · finetune (DistilBERT)
scripts/          run_analysis.py (analysis) · improve_model.py (compare) · train_finetune.py
tests/            test_smoke.py  (offline stub + synthetic; @slow real-model + fine-tune tests)
results/          figures/*.png + metrics.json + improvement.json  (committed)
data/ models/     git-ignored (HF dataset + baseline & fine-tuned weights downloaded/trained by code)
```

## References

- Dataset: [`florentgbelidji/car-reviews`](https://huggingface.co/datasets/florentgbelidji/car-reviews)
  (Edmunds consumer reviews) — see [data/README.md](data/README.md).
- Model: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
  (1-5 star sentiment); optional `distilbert-sst-2` and `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- Monroe, Colaresi & Quinn. *Fightin' Words: Lexical Feature Selection for Identifying
  Content.* (the distinctive-term intuition behind the keyword pass.)
