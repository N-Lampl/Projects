# Models

This project trains lightweight scikit-learn models (LogisticRegression and
GradientBoosting, each wrapped in `CalibratedClassifierCV`) in seconds, so no
weights are persisted to disk — `make run` fits fresh, deterministic models
every time (seed=42).

Any serialized models you choose to save here are **git-ignored**.
