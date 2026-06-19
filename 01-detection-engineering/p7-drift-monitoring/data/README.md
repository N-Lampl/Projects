# data/ (git-ignored)

This project needs **no external dataset**. The monitoring stream is generated
synthetically and deterministically in
[`src/drift_monitoring/stream.py`](../src/drift_monitoring/stream.py) (seeded
NumPy), so the default path runs fully offline.

- **Dataset:** synthetic tabular stream (fraud-/intrusion-detector-style numeric
  features), produced at runtime. License: N/A (generated).
- **Download:** none — just run `make detect`.
- **Optional real-data swap:** to monitor a real tabular feed, point a reference
  snapshot and per-window CSVs into this folder and feed them to
  `drift_monitoring.run_monitor(...)`. Keep any real data git-ignored and use
  only data you are authorized to process.
