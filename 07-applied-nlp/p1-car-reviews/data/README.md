# Data

This project runs on **real HuggingFace data by default**, with a seeded offline
synthetic fallback so tests and CI need no network. Nothing here is committed
(`data/` is git-ignored except this README).

## Default: `florentgbelidji/car-reviews` (HuggingFace Hub)

- **Dataset:** [`florentgbelidji/car-reviews`](https://huggingface.co/datasets/florentgbelidji/car-reviews)
  — 36,984 consumer car reviews scraped from Edmunds.
- **Columns used:** `Vehicle_Title` (year + make + model + trim, e.g.
  `"1997 Toyota Previa Minivan LE 3dr Minivan AWD"`), `Review` (full text),
  `Review_Title`, `Rating` (1-5, the ground-truth label), `Review_Date`,
  `Author_Name`.
- **Loaded by code** in [`../src/car_reviews/data.py`](../src/car_reviews/data.py)
  via `datasets.load_dataset("florentgbelidji/car-reviews", split="train")`, cached
  under `~/.cache/huggingface`. ~24 MB; never committed to this repo.

```bash
make run        # fast --sample 3000 (downloads the dataset + model on first use)
make run-full   # all 36,984 reviews (the committed metrics.json + figures)
```

## Fallback: seeded synthetic reviews (offline)

If `datasets` is missing or the download fails (offline CI, no network),
[`../src/car_reviews/synthetic.py`](../src/car_reviews/synthetic.py) fabricates a
small review table with the same columns and a **planted, recoverable signal**
(good brands score high, bad brands low; review text sentiment matches the star
rating; aspect keywords injected). This is what the smoke tests and `--backend
stub` use, so the whole pipeline is demonstrable with **zero downloads**.

## Sentiment models (HuggingFace, downloaded on first use)

Also git-ignored (cached by `transformers`, not stored here) — see
[`../models/README.md`](../models/README.md).

> Authorized use only. The dataset is a public dataset used for research/education
> under its stated terms; models are public pretrained checkpoints. See
> [../../../ETHICS.md](../../../ETHICS.md).
