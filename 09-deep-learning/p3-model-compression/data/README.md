# Data

This project runs on **synthetic classification data** by default, so tests and
CI need no network. Nothing here is committed (`data/` is git-ignored except this
README).

## Default: synthetic Gaussian blobs (offline, deterministic)

[`../src/compression/data.py`](../src/compression/data.py) draws `k` classes with
random centroids in a moderate-dimensional space, then Gaussian-noised points
around each centroid, and splits deterministically into train / test. The problem
is hard enough that a small student can under-fit but a trained teacher clears
chance by a wide margin - which is what makes the accuracy-vs-size trade-off real.

## Optional: MNIST via torchvision

The `@slow` test lazily imports **torchvision** and loads MNIST (flattened to
784-vectors) through `load_mnist()`. torchvision is optional (`pip install
torchvision`); if it is missing or the download fails, the test skips. The default
path and every fast test need only numpy / torch / matplotlib.

> Authorized use only: synthetic data used for education. See
> [../../../ETHICS.md](../../../ETHICS.md).
