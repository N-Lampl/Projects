# models/ (git-ignored)

This project fits detectors **on the fly** inside `make detect` (IsolationForest is
trained in well under a second; the optional autoencoder in a few seconds on CPU), so no
weights are persisted by default. Nothing here is committed.

If you adapt the LANL/LogHub path and want to cache a fitted IsolationForest, persist it
here with `joblib.dump(clf, "models/iforest.joblib")`.
