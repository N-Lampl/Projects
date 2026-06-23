# Data

The default run path needs **no download**: `make run` generates a deterministic,
seeded synthetic borrower table in-memory (`src/credit_risk/data.py`) with a
realistic logistic default-generating process and a baked-in protected-attribute
disparity for the fairness check. Nothing is written to this folder by default,
and any data you add here is **git-ignored** (never commit raw data).

## Optional real datasets (drop-in)

You can run on a real public credit dataset instead. Download it yourself, map
the columns to the schema below, save as `data/credit.csv`, then:

```bash
make run ARGS="--real-csv data/credit.csv"   # or: python3 scripts/run_scoring.py --real-csv data/credit.csv
```

Required columns: `income, dti, utilization, num_delinquencies, emp_length,
loan_amount, default` (plus optional `group` for the fairness check; defaults to
a single group if absent).

- **Give Me Some Credit** (Kaggle, 2011) — 150k borrowers, target
  `SeriousDlqin2yrs`. License: Kaggle competition terms.
  <https://www.kaggle.com/c/GiveMeSomeCredit>
- **Statlog (German Credit)** — UCI ML Repository, 1000 applicants, good/bad
  credit. License: CC BY 4.0.
  <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>

These are documented as options only; the portfolio is self-contained on the
synthetic path so results are reproducible offline.
