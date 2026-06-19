# Data

## Default: synthetic glyphs (no download, offline)

The default run path uses a **synthetic** MNIST-like dataset of 8x8 grayscale
digit glyphs rendered procedurally in [`src/transfer_blackbox/data.py`](../src/transfer_blackbox/data.py)
(`make_synthetic`). Nothing is downloaded; `make attack` works fully offline.
Pixels live in `[0, 1]` so epsilon values are interpretable as a fraction of full
pixel intensity. This is enough to train two distinct classifiers to high accuracy
and study adversarial transfer + black-box query attacks.

## Optional: real MNIST

- **Dataset:** MNIST handwritten digits (LeCun, Cortes, Burges).
- **License:** Creative Commons Attribution-Share Alike 3.0.
- **Size:** ~11 MB (under the 50 MB limit). Downloaded to `data/MNIST/`.
- **Download / use:** there is no separate download step — pass `--real` and
  torchvision fetches it on first run:

  ```bash
  pip install torchvision           # only needed for the real path
  make attack ARGS=--real           # or: python scripts/run_attacks.py --real
  ```

Everything in `data/` is **git-ignored** and never committed.
