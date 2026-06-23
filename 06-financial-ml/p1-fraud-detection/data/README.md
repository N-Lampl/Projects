# Data

This project runs **fully offline by default** on a seeded synthetic transaction
generator ([../src/fraud_detection/data.py](../src/fraud_detection/data.py)). No
download is required and nothing here is committed (`data/` is git-ignored).

## Default: synthetic transactions (offline)

`make_transactions()` deterministically fabricates a credit-card transaction
table with a ~1% fraud minority and an **injected, learnable** fraud signal:

| feature         | meaning                                   | fraud tilt                  |
|-----------------|-------------------------------------------|-----------------------------|
| `amount`        | log-normal transaction amount             | skews to extreme amounts    |
| `hour`          | hour-of-day (0-23)                         | concentrates in small hours |
| `merchant_cat`  | merchant category (0-7, one-hot encoded)  | cats {2,6} higher-risk      |
| `account_age_d` | days since account opened                 | hits younger accounts       |
| `velocity_1h`   | txns by this account in last hour         | fraud bursts                |
| `velocity_24h`  | txns by this account in last 24h          | fraud bursts                |
| `amount_to_avg` | amount / account running-average amount   | abnormal vs. normal spend   |

The label is sampled from a logistic model of these drivers plus irreducible
noise, so the decision boundary is real but non-trivial.

## Optional: real Kaggle ULB "Credit Card Fraud Detection"

- **Dataset:** Credit Card Fraud Detection (Université Libre de Bruxelles / Worldline).
- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **License:** Database Contents License (DbCL) v1.0 (Open Data Commons).
- **Shape:** 284,807 transactions, 492 fraud (0.172% base rate). Columns:
  `Time`, `V1..V28` (PCA-anonymized), `Amount`, `Class` (1 = fraud). ~144 MB.

Download (requires a Kaggle account + API token at `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
# -> data/creditcard.csv
```

Then load it with `fraud_detection.data.load_creditcard_csv("data/creditcard.csv")`.
The csv exceeds the repo's 50 MB commit limit, so it is **never** committed.

> Authorized use only. We train our own models on either synthetic data or a
> public dataset used under its stated license. See [../../../ETHICS.md](../../../ETHICS.md).
