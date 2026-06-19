# data/ (git-ignored)

This project needs **no external dataset**. The served model is trained at runtime on
a small **synthetic** two-class tabular dataset generated deterministically in
[`src/api_threat_model/model.py`](../src/api_threat_model/model.py)
(`make_synthetic_dataset`, seeded at 42).

- **Dataset:** synthetic Gaussian blobs (8 features, 2 classes). No download, no license concerns.
- **Why synthetic:** the project's subject is the **security controls around** a serving
  endpoint, not the model. Synthetic data keeps the default path fully offline.
- Nothing in this folder is committed; it is git-ignored at the repo root.
