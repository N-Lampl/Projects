# data/ (git-ignored)

## DEFAULT path -- synthetic, no download
The default `make attack` generates a **synthetic 16x16 image dataset** at runtime
with scikit-learn's `make_classification` (see `src/model_extraction/data.py`).
Nothing is downloaded and nothing is committed.

## OPTIONAL path -- real MNIST
- **Dataset:** MNIST handwritten digits (Yann LeCun et al.), public, research use.
- **Download (~11 MB, automatic):**
  ```bash
  make attack ARGS="--dataset mnist"   # or: python scripts/run_extraction.py --dataset mnist
  ```
  torchvision downloads MNIST into this folder on first run. Requires the optional
  `torchvision` dependency (`pip install torchvision`).

Pixels are kept in `[0, 1]` for both datasets. Everything in this folder is
git-ignored and produced/downloaded at runtime.
