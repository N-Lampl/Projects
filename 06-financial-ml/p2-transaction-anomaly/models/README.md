# Models

The detectors here are **fit at runtime** and are cheap to train (IsolationForest
fits in a couple of seconds on the synthetic stream), so no model artifacts are
committed. This directory is git-ignored and exists as a place to persist a fitted
detector if you choose to (e.g. `joblib.dump(det.model, "models/iforest.joblib")`).

- Default: `IForestDetector` — `StandardScaler` + `sklearn.ensemble.IsolationForest`.
- Optional: `AutoencoderDetector` — a tiny dense PyTorch autoencoder scored by
  reconstruction MSE; falls back to IsolationForest if torch is not installed.
