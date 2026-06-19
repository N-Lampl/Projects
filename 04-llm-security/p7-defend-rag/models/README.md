# models/ (git-ignored)

`make train` / `make defend` write the trained detector here:

- `injection_detector.joblib` -- the TF-IDF + LogisticRegression prompt-injection
  classifier (a few hundred KB). Produced from the synthetic dataset; regenerate
  any time with `make train`.

Nothing in this folder is committed.
