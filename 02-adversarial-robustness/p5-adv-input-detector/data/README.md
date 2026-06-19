# data/ (git-ignored)

Nothing in this folder is committed.

## Default path — synthetic (no download)

`make detect` uses a deterministic **synthetic 28×28 digit** dataset generated in
`src/adv_detector/data.py` (`synthetic_digits`). No network, no files written here.

## Optional path — real MNIST

Enable with `make detect ARGS="--dataset mnist"`. This lazily imports `torchvision` and downloads
MNIST here on first run.

- **Dataset:** MNIST handwritten digits (Yann LeCun et al.), public, research use.
- **Size:** ~11 MB (under the repo's no-large-download policy).
- **Download command:** automatic — just run `make detect ARGS="--dataset mnist"`.
  (Equivalent to `torchvision.datasets.MNIST(root="data", download=True)`.)
- **Pixels are kept in [0, 1]** (ToTensor only, no normalization) so FGSM's ε and bit-depth reduction
  are directly interpretable.
- If the download is unavailable (offline), the code falls back to the synthetic dataset automatically.
