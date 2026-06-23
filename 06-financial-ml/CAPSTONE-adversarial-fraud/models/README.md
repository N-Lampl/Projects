# Models

Models are trained in-process by `scripts/run_capstone.py` (a scikit-learn
`StandardScaler` + `LogisticRegression` pipeline, plus an optional adversarially
trained variant). They are cheap to refit, so no weights are committed here.

Any `*.pkl` written to this folder is git-ignored. To reproduce the exact models,
run `make run` (seed = 42).
