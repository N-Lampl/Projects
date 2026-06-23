"""Seeded synthetic credit-card transaction generator.

The DEFAULT (offline) data path. We fabricate a transaction table with a small
fraud minority (~1%) and an *injected, learnable* fraud signal so scikit-learn
can actually separate the classes -- without downloading anything.

Real-world drop-in: the Kaggle ULB "Credit Card Fraud Detection" dataset
(creditcard.csv). See data/README.md. `load_creditcard_csv` reads it if present.

Features (all numeric after encoding):
    amount          log-normal transaction amount; fraud skews to extremes
    hour            hour-of-day 0..23; fraud concentrates in the small hours
    merchant_cat    categorical 0..N; some categories are higher-risk
    account_age_d   days since account opened; fraud hits younger accounts
    velocity_1h     # txns by this account in the last hour; fraud bursts
    velocity_24h    # txns in last 24h
    amount_to_avg   amount / account's running average amount

The label `is_fraud` is drawn from a logistic model of these drivers plus noise,
so the Bayes-optimal boundary is non-trivial but real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "amount",
    "hour",
    "merchant_cat",
    "account_age_d",
    "velocity_1h",
    "velocity_24h",
    "amount_to_avg",
]
LABEL = "is_fraud"
N_MERCHANT_CATS = 8
# A couple of merchant categories carry elevated fraud risk (e.g. online, gift cards).
HIGH_RISK_CATS = {2, 6}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_transactions(
    n: int = 30_000,
    fraud_rate: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic transaction table.

    `fraud_rate` is the *target* base rate; the actual realized rate is close
    (it is sampled from a calibrated logistic, then the intercept is solved so
    the mean fraud probability matches `fraud_rate`).
    """
    rng = np.random.default_rng(seed)

    # --- legitimate-ish base behaviour ---------------------------------------
    amount = rng.lognormal(mean=3.2, sigma=1.0, size=n)  # ~ tens of dollars
    hour = rng.integers(0, 24, size=n)
    merchant_cat = rng.integers(0, N_MERCHANT_CATS, size=n)
    account_age_d = rng.gamma(shape=2.0, scale=300.0, size=n)  # right-skewed
    velocity_1h = rng.poisson(lam=0.6, size=n)
    velocity_24h = velocity_1h + rng.poisson(lam=4.0, size=n)

    # per-account average amount proxy -> amount relative to "normal"
    avg_amount = rng.lognormal(mean=3.2, sigma=0.4, size=n)
    amount_to_avg = amount / np.clip(avg_amount, 1e-3, None)

    # --- injected fraud signal (the part sklearn must learn) ------------------
    night = ((hour <= 4) | (hour >= 23)).astype(float)
    high_risk = np.isin(merchant_cat, list(HIGH_RISK_CATS)).astype(float)
    young = (account_age_d < 120).astype(float)
    log_amt = np.log1p(amount)

    # standardize drivers used in the latent risk score so weights are sane
    def _z(a: np.ndarray) -> np.ndarray:
        return (a - a.mean()) / (a.std() + 1e-9)

    logit = (
        1.10 * _z(log_amt)
        + 1.30 * night
        + 1.20 * high_risk
        + 0.90 * young
        + 0.85 * _z(velocity_1h.astype(float))
        + 0.55 * _z(amount_to_avg)
        + 0.6 * rng.standard_normal(n)  # irreducible noise
    )

    # solve intercept b so that mean(sigmoid(logit + b)) ~= fraud_rate
    b = _solve_intercept(logit, fraud_rate)
    p = _sigmoid(logit + b)
    is_fraud = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "hour": hour,
            "merchant_cat": merchant_cat,
            "account_age_d": np.round(account_age_d, 1),
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "amount_to_avg": np.round(amount_to_avg, 4),
            "is_fraud": is_fraud,
        }
    )
    return df


def _solve_intercept(logit: np.ndarray, target_rate: float, iters: int = 60) -> float:
    """Bisection on the intercept so mean fraud probability == target_rate."""
    lo, hi = -20.0, 20.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rate = _sigmoid(logit + mid).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with merchant_cat one-hot encoded."""
    x = df[FEATURES].copy()
    dummies = pd.get_dummies(x["merchant_cat"].astype(int), prefix="mcat")
    # ensure all categories present (stable columns across splits)
    for c in range(N_MERCHANT_CATS):
        col = f"mcat_{c}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"mcat_{c}" for c in range(N_MERCHANT_CATS)]]
    x = x.drop(columns=["merchant_cat"]).join(dummies)
    return x.astype(float), df[LABEL].astype(int)


def load_creditcard_csv(path: str | Path) -> pd.DataFrame:
    """OPTIONAL real-data path: read the Kaggle ULB creditcard.csv.

    Columns: Time, V1..V28 (PCA), Amount, Class (0/1). Returned as a frame with
    `is_fraud` aliasing `Class`; callers must adapt feature handling. Provided so
    the README's optional path is not vapor.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "is_fraud"})
    return df
