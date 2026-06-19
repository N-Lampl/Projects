# data/ (git-ignored)

Nothing in this folder is committed.

## Default path — no data needed
`make attack` uses **deterministic synthetic 32x32 RGB images** generated in code
(`src/pretrained_foolbox/data.py`, 4 visually-distinct classes). No dataset, no
download, fully offline.

## Optional online path — CIFAR-10
`make pretrained` (or `make attack ARGS=--pretrained`) downloads a few **CIFAR-10**
test images here automatically via `torchvision.datasets.CIFAR10(download=True)`.

- **Dataset:** CIFAR-10 (Krizhevsky, 2009), public, research use.
- **Download:** automatic on first `make pretrained`. Archive is ~170 MB; only a
  handful of images are used. If you are offline / behind a proxy the script
  prints a notice and falls back to the offline synthetic path.

Pixels are kept in **[0, 1]** (ToTensor only); the target model folds its own
normalization in, so the attack's epsilon is a fraction of full pixel intensity.
