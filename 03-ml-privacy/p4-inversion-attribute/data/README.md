# data/ (git-ignored)

The **default** run uses **no external data at all** — a synthetic 10-class
"digit-like" image set and a synthetic tabular dataset are generated in memory
(see `src/inversion_attribute/data.py` and `attribute.py`). Nothing is downloaded
or committed.

## Synthetic image data (default)
- 10 classes, each with a fixed 28×28 prototype shape + per-pixel noise + small shift.
- Generated deterministically with `make_synthetic(seed=42)`.

## Optional: real MNIST (enhanced path)
- **Dataset:** MNIST handwritten digits (Yann LeCun et al.), public, research use.
- **Download:** automatic when you pass `--mnist` — `torchvision.datasets.MNIST`
  writes to this folder (`data/MNIST/`). ~12 MB. Imported lazily.
- Run: `python3 scripts/run_inversion.py --mnist`

Everything in this folder is git-ignored.
