# models/ (git-ignored)

This project's detectors are **unsupervised and re-fit at run time** — there is no
trained artifact to persist. `make detect` fits a fresh `IsolationForest` on the
engineered features each run (deterministic given `--seed`), so nothing needs to be
saved here. The folder is kept (git-ignored) for parity with the rest of the
portfolio and as a place to cache a model if you later switch to real data.
