# data/ (git-ignored)

The **default run uses a synthetic, in-memory dataset** — nothing is downloaded
or written here. The optional real-data paths populate this folder.

- **Default:** synthetic images (`src/attack_zoo/data.py:make_synthetic`), generated
  deterministically from a fixed seed. No license, no download.
- **Optional CIFAR-10** (Krizhevsky, public, research use): downloaded automatically by
  `torchvision.datasets.CIFAR10` (~170MB) when you run `make attack ARGS='--source cifar10'`.
- **Optional MNIST** (LeCun et al., public, research use): downloaded by
  `torchvision.datasets.MNIST` when you run `make attack ARGS='--source mnist'`.

All pixels are kept in `[0, 1]` (ToTensor only, no normalization) so the reported
perturbation budgets (L-inf epsilon, mean L2) map directly to pixel intensity.
Nothing in this folder is committed.
