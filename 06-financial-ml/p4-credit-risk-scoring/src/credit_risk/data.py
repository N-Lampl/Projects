"""Synthetic borrower table with a realistic logistic default-generating process.

No download required. Features mirror a real credit application:
    income            -- annual income (USD)
    dti               -- debt-to-income ratio (0..~1.2)
    utilization       -- revolving credit utilization (0..1+)
    num_delinquencies -- count of past late payments
    emp_length        -- employment length (years)
    loan_amount       -- requested loan principal (USD)

The default label is drawn from a logistic model over these features plus a
small protected-attribute effect, so the data has a genuine, recoverable signal
and a *known* fairness structure we can later measure. Everything is seeded.

A real public dataset (Give-Me-Some-Credit / German Credit) is documented in
data/README.md as an optional drop-in; see `load_german_credit_if_present`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "income",
    "dti",
    "utilization",
    "num_delinquencies",
    "emp_length",
    "loan_amount",
]

# Logistic weights on *standardized* features for the data-generating process.
# Signs match credit intuition: higher income/employment -> lower default;
# higher dti/utilization/delinquencies/loan -> higher default.
_TRUE_WEIGHTS = {
    "income": -0.9,
    "dti": 1.1,
    "utilization": 1.3,
    "num_delinquencies": 1.0,
    "emp_length": -0.6,
    "loan_amount": 0.5,
}
# Bias chosen to land the default rate in a realistic ~10-15% band.
_TRUE_BIAS = -2.1
# Extra log-odds applied to protected group B (a baked-in disparity to *detect*,
# not to endorse — the fairness check exists to surface exactly this).
_GROUP_B_EFFECT = 0.45


def make_credit_data(
    n: int = 12000,
    seed: int = 42,
    group_b_fraction: float = 0.4,
    noise: float = 0.6,
) -> pd.DataFrame:
    """Generate a deterministic synthetic borrower table with a default label.

    Returns a DataFrame with the six FEATURES, a binary ``default`` target and a
    synthetic ``group`` protected attribute ("A"/"B").
    """
    rng = np.random.default_rng(seed)

    # Raw, human-readable feature draws (skewed where it makes sense).
    income = rng.lognormal(mean=10.9, sigma=0.45, size=n)  # ~ median 54k
    dti = np.clip(rng.beta(2.0, 5.0, size=n) * 1.4, 0, 1.4)
    utilization = np.clip(rng.beta(2.0, 3.0, size=n), 0, 1.2)
    num_delinquencies = rng.poisson(0.4, size=n).astype(float)
    emp_length = np.clip(rng.gamma(2.0, 3.0, size=n), 0, 40)
    loan_amount = rng.lognormal(mean=9.4, sigma=0.5, size=n)  # ~ median 12k

    df = pd.DataFrame(
        {
            "income": income,
            "dti": dti,
            "utilization": utilization,
            "num_delinquencies": num_delinquencies,
            "emp_length": emp_length,
            "loan_amount": loan_amount,
        }
    )

    # Protected group; group B borrowers carry a modest extra default risk that
    # is *correlated with* (not independent of) their features, mimicking how
    # real disparities arise from structural factors.
    group = rng.choice(["A", "B"], size=n, p=[1 - group_b_fraction, group_b_fraction])
    df["group"] = group

    # Standardize features for the logistic DGP so weights are comparable.
    z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std(ddof=0)
    logit = np.full(n, _TRUE_BIAS, dtype=float)
    for f, w in _TRUE_WEIGHTS.items():
        logit += w * z[f].to_numpy()
    logit += np.where(group == "B", _GROUP_B_EFFECT, 0.0)
    logit += rng.normal(0.0, noise, size=n)  # irreducible noise

    p_default = 1.0 / (1.0 + np.exp(-logit))
    df["default"] = (rng.uniform(size=n) < p_default).astype(int)
    return df


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified-by-default split returning (train_df, test_df)."""
    rng = np.random.default_rng(seed)
    parts = []
    for label, sub in df.groupby("default"):
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        parts.append((idx[:n_test], idx[n_test:]))
    test_idx = np.concatenate([p[0] for p in parts])
    train_idx = np.concatenate([p[1] for p in parts])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def load_german_credit_if_present(path: str | Path) -> pd.DataFrame | None:
    """Optional real-data path: load a CSV the user dropped into data/.

    Returns None if the file is missing so the default synthetic path always
    works offline. The CSV must already contain the FEATURES + ``default``
    columns (and optionally ``group``); see data/README.md.
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in [*FEATURES, "default"] if c not in df.columns]
    if missing:
        raise ValueError(f"real-data CSV missing required columns: {missing}")
    if "group" not in df.columns:
        df["group"] = "A"
    return df
