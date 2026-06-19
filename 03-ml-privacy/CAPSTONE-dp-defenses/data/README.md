# data/ (git-ignored)

The DEFAULT run needs **no data download** — it builds a synthetic tabular pool
in-memory with scikit-learn's `make_classification` (see
`src/dp_defenses/data.py`). Nothing in this folder is committed.

## Default (offline)
- **Dataset:** synthetic classification pool, generated deterministically (seed 42).
- **Download:** none. Just run `make run`.
- **Why synthetic:** it is over-parameterised relative to its size, so a
  non-private model memorises training points — the exact leakage DP must suppress
  and that membership inference exploits.

## Optional enhanced path (real data)
- **Dataset:** Fashion-MNIST (Xiao et al., MIT license), flattened to a tabular pool.
- **Download (automatic on first use):**
  ```bash
  python3 -c "from torchvision import datasets; datasets.FashionMNIST('data', train=True, download=True)"
  ```
  Requires `torchvision` (optional). ~30 MB, git-ignored. Switch by calling
  `load_fashion_mnist_pool()` instead of `make_synthetic_pool()`.
