# models/ (git-ignored)

This baseline trains a scikit-learn `RandomForestClassifier` (from the shared
`ids_pipeline` library) in a couple of seconds, so `make detect` retrains from
scratch every run rather than persisting weights — the run is fully reproducible
from the seed (42). No model artifacts are committed.

If you want to persist a fitted pipeline, `joblib.dump(pipe, "models/nids_rf.joblib")`
after training; it is git-ignored.
