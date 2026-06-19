# models/ (git-ignored)

This library does not persist model artifacts by default -- the demo trains the
pipeline in-process each run (training on the synthetic set takes a few seconds on CPU).
If a downstream project (e.g. CAPSTONE-adversarial-ids) chooses to pickle a fitted
pipeline, write it here. Nothing in this folder is committed.
