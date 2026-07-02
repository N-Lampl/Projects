# Data

This project runs on **synthetic hierarchical data** by default, so tests and CI
need no network. Nothing here is committed (`data/` is git-ignored except this
README).

## Default: synthetic hierarchical draw (offline, known parameters)

[`../src/bayes_pp/data.py`](../src/bayes_pp/data.py) draws `J` groups whose true
means come from a global `N(mu, tau)`, then a handful of noisy observations per
group. Because the **true group means are stored on the dataset**, every posterior
is scored against ground truth — that is what makes the shrinkage and calibration
claims checkable. Small `n_per_group` makes the per-group MLEs noisy, giving
partial pooling something real to fix.

## Optional: PyMC cross-check

The `@slow` test fits the *same* model with **PyMC** (NUTS) and checks its
posterior means agree with the numpy Gibbs sampler. PyMC is optional (`pip install
pymc arviz`); the default path needs only numpy/scipy.

> Authorized use only: synthetic data used for education. See
> [../../../ETHICS.md](../../../ETHICS.md).
