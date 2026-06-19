# data/ (git-ignored)

The DEFAULT path needs **no data at all**: the shared `ids_pipeline` library
generates deterministic synthetic network flows in-memory (see
`../shared/ids_pipeline/src/ids_pipeline/data.py`). Nothing in this folder is
committed.

## Optional: real NSL-KDD benchmark

To run the attack against a model trained on real traffic instead of synthetic
flows, download NSL-KDD (public, research use) into this folder:

- **Dataset:** NSL-KDD (Tavallaee et al., 2009) — a cleaned KDD'99 derivative.
- **License:** public, free for research use (University of New Brunswick / CIC).
- **Download (exact command):**

  ```bash
  # ~5 MB; well under the repo's 50 MB limit
  curl -L -o data/KDDTrain+.txt \
    https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
  curl -L -o data/KDDTest+.txt \
    https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
  ```

Then the shared library's `load_data(synthetic=False)` will pick the files up.
The capstone script defaults to synthetic; point it at real data by editing the
`api.load_data(...)` call in `scripts/run_capstone.py`.
