# data/ (git-ignored)

MNIST is downloaded here automatically on first run by `torchvision.datasets.MNIST`
(see `src/fgsm_mnist/data.py`). Nothing in this folder is committed.

- **Dataset:** MNIST handwritten digits (Yann LeCun et al.), public, research use.
- **Download:** automatic - just run `make attack` or `make train`.
- **Pixels are kept in [0, 1]** (ToTensor only, no normalization) so FGSM's ε maps directly to a
  fraction of full pixel intensity.
