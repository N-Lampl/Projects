# data/ (git-ignored)

The **default** run uses a deterministic **synthetic** dataset generated in
`src/rand_smoothing/data.py` (NumPy/torch only) — no download, fully offline. Each
class is a distinct low-frequency 2D pattern plus Gaussian noise; pixels stay in
`[0, 1]` so the smoothing sigma and the certified L2 radius are in pixel units.

The **optional** path uses real MNIST:

- **Dataset:** MNIST handwritten digits (Yann LeCun et al.), public, research use.
- **Download (automatic on first use):**
  ```bash
  make certify ARGS=--dataset=mnist     # or: python3 scripts/run_certify.py --dataset mnist
  ```
  torchvision fetches MNIST into this folder. Requires `torchvision` installed and
  network access. Nothing in this folder is committed.
