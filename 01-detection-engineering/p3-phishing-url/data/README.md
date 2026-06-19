# data/ (git-ignored)

The **default** run needs **no data download** — `src/phishing_url/data.py` generates a
deterministic SYNTHETIC URL corpus (benign-looking vs phishing-looking) in memory.

## Optional: real PhiUSIIL dataset

- **Dataset:** PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024), UCI ML Repository (id 967).
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Download (lazy, on first call):**

  ```bash
  python3 -m pip install ucimlrepo        # one-time
  make detect ARGS="--data phiusiil"      # fetch_ucirepo(id=967) caches it here
  ```

`load_phiusiil()` flips PhiUSIIL's labels so that `1 = phishing` (matching this project) and
subsamples to `--n` rows. Nothing in this folder is committed.
