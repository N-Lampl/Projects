# data/ (git-ignored)

This project needs **no external dataset**. The model trains on a deterministic
synthetic classification set generated in-process by
`sklearn.datasets.make_classification` (see `src/secure_ml_pipeline/model.py`).

- **Dataset:** synthetic (scikit-learn `make_classification`, seed 42). License: n/a.
- **Download:** none required - just run `make run`.

## `data/poc/` - NEVER COMMITTED
`make poc` writes the benign pickle PoC to `data/poc/malicious_model.pkl`. This is
generated at runtime and is git-ignored. **Do not commit any weaponized or
PoC pickle to the repository.** The payload only writes a benign marker file
(`/tmp/PWNED_DEMO`) and is meant to be detonated solely inside the Docker sandbox
(`make detonate`).
