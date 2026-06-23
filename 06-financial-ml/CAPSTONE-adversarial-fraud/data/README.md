# Data

**Default = synthetic, no download.** `scripts/run_capstone.py` calls
`adv_fraud.make_dataset(...)`, a deterministic (seeded) generator that produces a
credit-card-style payments feed with a clearly-injected, learnable fraud signal
and a per-feature **mutability profile** (what a fraudster can vs. cannot change).
This is what makes the project fully self-contained and offline.

The generated arrays are held in memory only; nothing is written here. Anything
that does land in `data/` is git-ignored and must **never** be committed.

## Optional real dataset

To rerun the same pipeline against real labelled fraud, the standard public
benchmark is:

- **Credit Card Fraud Detection** (ULB / Worldline), Kaggle.
  License: **Database Contents License (DbCL) v1.0**. 284,807 transactions,
  ~0.17% fraud (PCA-anonymized features `V1..V28` + `Amount`, `Time`).
  Download (requires a Kaggle account + API token):

  ```bash
  pip install kaggle
  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
  ```

Note: because the real dataset's features are PCA components, the
mutability/feasibility threat model in this repo (which relies on
human-meaningful fields like `amount`, `hour`, `merchant_risk`) maps most
directly onto the synthetic generator. The real set is provided as an optional
PR-AUC sanity check for the classifier, not for the constrained-evasion attack.
