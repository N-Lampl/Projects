# data/ (git-ignored)

**Default (offline) path uses NO dataset on disk.** The training/test data is
generated in-memory by `src/adv_training/data.py` (`make_synthetic`) — 10 classes
of 28x28 "digit-like" images (fixed per-class Gaussian-blob templates + per-sample
noise), kept in [0, 1] so the L-inf epsilon maps directly to a fraction of full
pixel intensity. Deterministic in the seed, torch-only, no download.

## Optional enhanced path: real MNIST

- **Dataset:** MNIST handwritten digits (Yann LeCun et al.), public, research use.
- **Enable:** `make run ARGS=--real` (needs `torchvision`: `pip install torchvision`).
- **Download:** automatic — torchvision fetches MNIST into this folder on first run.
- Pixels kept in [0, 1] (ToTensor only, no normalization), same as the FGSM project.

Nothing in this folder is committed.
