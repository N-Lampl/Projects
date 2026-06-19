# models/ (git-ignored)

The served classifier is a tiny scikit-learn `LogisticRegression` trained **in memory**
on synthetic data each time the service starts (`train_model()` in
[`src/api_threat_model/model.py`](../src/api_threat_model/model.py)). Training takes a
fraction of a second on CPU, so no weights are persisted to disk and nothing here is
committed.

If you later want to serve a real persisted model, drop it here and load it inside
`PredictionService.build(...)`; the security controls are model-agnostic.
