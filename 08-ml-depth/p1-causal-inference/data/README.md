# Data

This project runs on a **synthetic structural causal model** by default, so tests
and CI need no network. Nothing here is committed (`data/` is git-ignored except
this README).

## Default: synthetic SCM (offline, known ATE)

[`../src/causal_inference/scm.py`](../src/causal_inference/scm.py) draws confounders
`X`, a treatment `T` whose probability depends on `X`, and an outcome `Y` that also
depends on `X`. The treatment adds a constant `tau`, so the **true ATE = tau** is
known exactly — that is what makes the estimators checkable.

## Optional: IHDP benchmark (real, downloaded)

[`../src/causal_inference/data.py`](../src/causal_inference/data.py) can pull the
standard semi-synthetic **IHDP** benchmark (Hill 2011; Shalit et al. simulations)
from `https://www.fredjo.com/files/ihdp_npci_1-100.train.npz`. It ships simulated
potential outcomes, so it too has a known ATE. Used only by the `@slow` test; the
download is cached by nothing and never committed.

> Authorized use only: a public research benchmark used for education. See
> [../../../ETHICS.md](../../../ETHICS.md).
