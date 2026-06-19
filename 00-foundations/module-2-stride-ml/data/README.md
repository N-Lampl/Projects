# data/ (git-ignored)

This project is a **threat-modeling** exercise — it needs **no dataset**. The system
under analysis (an ML inference service) is described in code as a data-flow model
(`src/stride_ml/model.py`), not loaded from disk.

- **Dataset:** none. The "input" is a synthetic, self-authored architecture model.
- **Download:** nothing to download — run `make detect`.
- Anything that ever lands here at runtime is git-ignored and never committed.
