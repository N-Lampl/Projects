# data/ (git-ignored)

The **default** run needs **no data**: `make attack` synthesises a population pool
with `sklearn.datasets.make_classification` (see `src/lira_mia/data.py`). Nothing
is downloaded or committed.

## Optional enhanced path: Fashion-MNIST

To attack a model trained on real images instead of synthetic tabular data:

```bash
pip install torchvision        # ~optional dependency
make attack ARGS='--dataset fashion_mnist'
```

- **Dataset:** Fashion-MNIST (Zalando Research) - 28x28 grayscale clothing images,
  10 classes. License: MIT. https://github.com/zalandoresearch/fashion-mnist
- **Download:** automatic via `torchvision.datasets.FashionMNIST(download=True)`
  into this folder (~30 MB). Git-ignored, never committed.
- We use a 4000-image subset and flatten each image to a 784-D vector so the same
  MLP + LiRA pipeline applies unchanged.
