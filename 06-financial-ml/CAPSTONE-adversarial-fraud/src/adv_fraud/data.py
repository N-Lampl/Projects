"""Seeded synthetic credit-card-style fraud data with an injected signal.

No download required. The generator produces a tabular dataset that looks like a
payments feed and bakes in a *learnable* fraud signal so a classifier reaches a
realistic (not perfect) PR-AUC. Crucially, every feature is tagged with an
adversarial **mutability profile** - what a fraudster can actually control at
transaction time vs. immutable account history - which the evasion attack in
``attack.py`` is forced to respect.

Feature dictionary (10 features):
    amount              MUTABLE   transaction value in USD (attacker picks it)
    hour                MUTABLE   hour-of-day 0..23 (attacker times the charge)
    merchant_risk       MUTABLE   risk score of chosen merchant category 0..1
    distance_from_home  MUTABLE   km from cardholder home (attacker location)
    n_items             MUTABLE   basket size (integer)
    account_age_days    IMMUTABLE how long the account has existed
    avg_amount_30d      IMMUTABLE historical average spend (server-side agg)
    txn_count_30d       IMMUTABLE historical transaction count (server-side agg)
    home_country_risk   IMMUTABLE risk score tied to the cardholder, not the txn
    card_present        IMMUTABLE 0/1 chip-present flag set by the terminal

The injected fraud rule (noisy, non-linear, in standardized-ish raw space):
    high amount  +  unusual hour  +  high merchant risk  +  far from home,
    relative to the account's own history -> higher fraud probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES: list[str] = [
    "amount",
    "hour",
    "merchant_risk",
    "distance_from_home",
    "n_items",
    "account_age_days",
    "avg_amount_30d",
    "txn_count_30d",
    "home_country_risk",
    "card_present",
]

# Which features an attacker may modify when crafting an evasion.
MUTABLE: dict[str, bool] = {
    "amount": True,
    "hour": True,
    "merchant_risk": True,
    "distance_from_home": True,
    "n_items": True,
    "account_age_days": False,
    "avg_amount_30d": False,
    "txn_count_30d": False,
    "home_country_risk": False,
    "card_present": False,
}

# Plausible per-feature bounds the attacker must stay inside (lo, hi).
BOUNDS: dict[str, tuple[float, float]] = {
    "amount": (1.0, 5000.0),
    "hour": (0.0, 23.0),
    "merchant_risk": (0.0, 1.0),
    "distance_from_home": (0.0, 8000.0),
    "n_items": (1.0, 30.0),
}

# Features that must hold integer values after perturbation.
INTEGER_FEATURES = {"hour", "n_items"}


@dataclass
class Dataset:
    X: np.ndarray  # (n, 10) raw feature matrix
    y: np.ndarray  # (n,) 0=legit 1=fraud
    feature_names: list[str]


def _mutable_mask() -> np.ndarray:
    return np.array([MUTABLE[f] for f in FEATURES], dtype=bool)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_dataset(
    n: int = 12000,
    fraud_rate: float = 0.04,
    seed: int = 42,
) -> Dataset:
    """Generate a deterministic, class-imbalanced synthetic fraud dataset."""
    rng = np.random.default_rng(seed)

    # --- account-level (immutable) profile ---
    account_age_days = rng.gamma(shape=2.0, scale=400.0, size=n).clip(5, 4000)
    avg_amount_30d = rng.lognormal(mean=3.6, sigma=0.6, size=n).clip(5, 1500)
    txn_count_30d = rng.poisson(lam=25, size=n).clip(1, 300).astype(float)
    home_country_risk = rng.beta(2.0, 8.0, size=n)  # mostly low risk
    card_present = rng.binomial(1, 0.65, size=n).astype(float)

    # --- transaction-level (mutable) features, correlated with the account ---
    amount = (avg_amount_30d * rng.lognormal(0.0, 0.5, size=n)).clip(1, 5000)
    hour = rng.integers(0, 24, size=n).astype(float)
    merchant_risk = rng.beta(2.0, 6.0, size=n)
    distance_from_home = rng.exponential(scale=40.0, size=n).clip(0, 8000)
    n_items = rng.integers(1, 12, size=n).astype(float)

    X = np.column_stack(
        [
            amount,
            hour,
            merchant_risk,
            distance_from_home,
            n_items,
            account_age_days,
            avg_amount_30d,
            txn_count_30d,
            home_country_risk,
            card_present,
        ]
    ).astype(float)

    # --- injected fraud signal (noisy, non-linear) ---
    amount_ratio = amount / (avg_amount_30d + 1.0)  # spend vs. own history
    odd_hour = ((hour < 5) | (hour > 22)).astype(float)
    far = distance_from_home / 500.0
    young_account = np.exp(-account_age_days / 300.0)  # new accounts riskier

    logit = (
        -3.4
        + 1.15 * np.log1p(amount_ratio)
        + 1.30 * odd_hour
        + 2.10 * merchant_risk
        + 0.85 * np.tanh(far)
        + 1.20 * home_country_risk
        + 0.90 * young_account
        - 0.55 * card_present
        + rng.normal(0.0, 0.45, size=n)  # irreducible noise
    )
    p = _sigmoid(logit)

    # Calibrate the threshold so the realized positive rate ~= fraud_rate.
    thresh = np.quantile(p, 1.0 - fraud_rate)
    y = (p >= thresh).astype(int)

    return Dataset(X=X, y=y, feature_names=list(FEATURES))


def mutable_indices() -> np.ndarray:
    """Column indices the attacker is allowed to change."""
    return np.where(_mutable_mask())[0]
