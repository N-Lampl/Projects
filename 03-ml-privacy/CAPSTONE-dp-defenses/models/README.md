# models/ (git-ignored)

This capstone trains all models (DP targets, shared shadows, extraction thieves)
fresh in-memory each run — they are small MLPs that train in seconds on a CPU, so
no checkpoints are persisted or committed. The reproducible evidence lives in
`results/` (figures + `metrics.json`).
