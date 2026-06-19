# models/ (git-ignored)

The capstone trains its models in-memory at run time (the baseline RandomForest,
the hardened model, and the logistic substitute), so no weights need to be
committed. `make attack` rebuilds everything deterministically from `seed=42`.

If you add persistence (e.g. `joblib.dump` of the hardened pipeline), write the
`.pkl` files here — they are git-ignored and `make clean` removes them.
