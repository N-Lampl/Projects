# models/ (git-ignored)

This project monitors *input distributions*, not a specific trained model, so no
weights are produced by the default path. The folder is kept so the drift
monitor can sit in front of any of the detectors trained elsewhere in this track
(e.g. the tabular IDS) - drop a model artifact here and wire its training-time
feature snapshot in as the PSI/KS `reference`.
