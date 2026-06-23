# Data

This project runs fully **offline** on a seeded **synthetic** transaction stream
generated in-repo by [`src/txn_anomaly/data.py`](../src/txn_anomaly/data.py). No
download is required and **no data is committed** (the `data/` dir is git-ignored).

The synthetic stream injects three anomaly types with a clear, deterministic signal:
- `amount_spike` — amount ~15-40x the account's normal scale
- `off_hours` — activity at 1-4am for accounts that normally transact in daytime
- `velocity` — a burst of many transactions from one account within minutes

The labels are produced only so the **unsupervised** detectors can be graded; the
models never see them at fit time.

## Optional: a real public dataset

To validate on real data, the standard public benchmark is the
**Credit Card Fraud Detection** dataset (Kaggle, ULB Machine Learning Group) —
284,807 European card transactions over two days, 492 fraud cases (0.172%),
features `V1..V28` (PCA-anonymized) + `Time`, `Amount`, `Class`.

- License: Open Database License (ODbL) / database contents under Database Contents License.
- Download (requires a Kaggle account + API token):

  ```bash
  pip install kaggle
  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
  ```

It is ~150MB; **do not commit it**. Adapting `run_detect.py` to score `creditcard.csv`
is left as an exercise: drop `Class` at fit time, score with IsolationForest, then use
`Class` only inside `evaluate_scores`.
