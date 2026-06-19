# data/ (git-ignored)

The DEFAULT path needs **no data at all** — `load_data(synthetic=True)` (from the
shared `ids_pipeline` library) generates deterministic synthetic network flows in
memory. Nothing in this folder is committed.

## Optional: real NSL-KDD

To benchmark on the real dataset, download NSL-KDD and drop the two text files here:

- `KDDTrain+.txt`
- `KDDTest+.txt`

- **Dataset:** NSL-KDD (Tavallaee et al., 2009) — a cleaned, de-duplicated version of
  the KDD Cup 1999 intrusion-detection dataset.
- **License:** publicly released for research by the Canadian Institute for
  Cybersecurity (UNB).
- **Download (stable mirror, ~3 MB each, well under the 50 MB limit):**

  ```bash
  curl -L -o data/KDDTrain+.txt \
    https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt
  curl -L -o data/KDDTest+.txt \
    https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt
  ```

  Then run with the real data:

  ```bash
  make detect ARGS=--real
  ```

The shared loader maps NSL-KDD's multiclass attack labels to a binary
`normal`(0)/`attack`(1) target and keeps the feature subset matching the synthetic
schema, so every downstream step is identical for both sources.
