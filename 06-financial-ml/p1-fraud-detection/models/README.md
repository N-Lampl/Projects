# Models

This directory is git-ignored. The classifiers in this project are cheap to
train (a few seconds on CPU), so `make detect` re-fits them in-memory every run
rather than persisting weights — there is nothing to commit here.

Estimators (see [../src/fraud_detection/models.py](../src/fraud_detection/models.py)):

- **logreg** — `StandardScaler` + `LogisticRegression(class_weight="balanced")`.
- **rf** — `RandomForestClassifier(class_weight="balanced_subsample")`.
- **xgboost** *(optional)* — added only if `xgboost` is importable; otherwise the
  pipeline runs on the two scikit-learn models alone.

The run picks the best estimator by **PR-AUC** on a held-out stratified split.
