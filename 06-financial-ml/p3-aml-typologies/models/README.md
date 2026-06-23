# models/ (git-ignored)

This project does **not** persist model weights. The detectors are cheap scikit-learn
estimators (IsolationForest, class-weighted RandomForest) that are fit fresh inside
`make detect` on the deterministic synthetic graph, so every run reproduces the same
scores from scratch. There is nothing to download or cache here.
