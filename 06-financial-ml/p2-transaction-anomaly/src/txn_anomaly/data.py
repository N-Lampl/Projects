"""Seeded synthetic transaction stream with three injected anomaly types.

The signal is *injected deliberately* so the project produces real, reproducible
detection metrics with no download. The anomaly labels exist ONLY to evaluate the
unsupervised detectors -- the models never see them at fit time.

Injected anomaly types
----------------------
- amount_spike : a transaction whose amount is ~15-40x the customer's normal scale.
- off_hours    : activity at 1am-4am for an account that normally transacts 8am-9pm.
- velocity     : a burst of many transactions from one account within a few minutes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import SEED

FEATURES = [
    "amount_log",
    "hour",
    "is_off_hours",
    "txn_count_1h",
    "secs_since_prev",
    "amount_vs_acct_mean",
]


def make_transactions(
    n: int = 12000,
    n_accounts: int = 400,
    contamination: float = 0.012,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate ``n`` transactions over ``n_accounts`` with injected anomalies.

    Returns a DataFrame with engineered features plus ``is_anomaly`` (eval-only)
    and ``anomaly_type`` columns. Transactions are time-ordered.
    """
    rng = np.random.default_rng(seed)

    # --- normal baseline -------------------------------------------------
    account = rng.integers(0, n_accounts, size=n)
    # each account has its own typical spend scale (lognormal)
    acct_scale = rng.lognormal(mean=3.2, sigma=0.5, size=n_accounts)
    amount = rng.lognormal(mean=0.0, sigma=0.45, size=n) * acct_scale[account]

    # timestamps: a steady stream over ~30 days, with normal daytime hours
    start = pd.Timestamp("2024-01-01")
    secs = np.sort(rng.uniform(0, 30 * 24 * 3600, size=n)).astype(np.int64)
    ts = start + pd.to_timedelta(secs, unit="s")
    # normal hour distribution peaks mid-day (8..21)
    hour = np.clip(rng.normal(loc=14.0, scale=3.5, size=n), 0, 23).astype(int)
    # rebuild timestamp hour so engineered "hour" matches injected events later
    ts = pd.to_datetime(ts.normalize()) + pd.to_timedelta(hour, unit="h")
    ts = ts + pd.to_timedelta(rng.integers(0, 3600, size=n), unit="s")

    df = pd.DataFrame(
        {
            "account": account,
            "timestamp": ts,
            "amount": amount,
        }
    )
    df["is_anomaly"] = False
    df["anomaly_type"] = "normal"

    n_anom = max(3, int(round(n * contamination)))
    # split the budget roughly evenly across the three types
    n_each = n_anom // 3
    idx_pool = rng.permutation(n)
    cursor = 0

    def take(k: int) -> np.ndarray:
        nonlocal cursor
        chosen = idx_pool[cursor : cursor + k]
        cursor += k
        return chosen

    # --- amount spikes ---------------------------------------------------
    spike_idx = take(n_each)
    mult = rng.uniform(15.0, 40.0, size=len(spike_idx))
    df.loc[spike_idx, "amount"] = df.loc[spike_idx, "amount"].to_numpy() * mult
    df.loc[spike_idx, "is_anomaly"] = True
    df.loc[spike_idx, "anomaly_type"] = "amount_spike"

    # --- off-hours -------------------------------------------------------
    off_idx = take(n_each)
    new_hour = rng.integers(1, 5, size=len(off_idx))  # 1..4 am
    base = df.loc[off_idx, "timestamp"].dt.normalize()
    df.loc[off_idx, "timestamp"] = base.to_numpy() + pd.to_timedelta(new_hour, unit="h")
    df.loc[off_idx, "is_anomaly"] = True
    df.loc[off_idx, "anomaly_type"] = "off_hours"

    # --- velocity bursts -------------------------------------------------
    # pick a few seed transactions; clone each into a tight burst on same account
    n_bursts = max(1, (n_anom - 2 * n_each) // 4 + 1)
    burst_seed = take(n_bursts)
    extra_rows = []
    for i in burst_seed:
        acct = int(df.at[i, "account"])
        t0 = df.at[i, "timestamp"]
        amt_scale = float(acct_scale[acct])
        df.at[i, "is_anomaly"] = True
        df.at[i, "anomaly_type"] = "velocity"
        for j in range(1, rng.integers(5, 9)):
            extra_rows.append(
                {
                    "account": acct,
                    "timestamp": t0 + pd.Timedelta(seconds=int(j * rng.integers(20, 90))),
                    "amount": rng.lognormal(0.0, 0.4) * amt_scale,
                    "is_anomaly": True,
                    "anomaly_type": "velocity",
                }
            )
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return _engineer(df)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-transaction features a streaming fraud system could compute live."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_off_hours"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)
    df["amount_log"] = np.log1p(df["amount"].clip(lower=0))

    # per-account running stats (use full-account mean -- eval-only stream)
    acct_mean = df.groupby("account")["amount"].transform("mean")
    df["amount_vs_acct_mean"] = df["amount"] / acct_mean.clip(lower=1e-6)

    # velocity: count of same-account txns in the trailing 1 hour, and gap to prev
    df = df.sort_values(["account", "timestamp"]).reset_index(drop=True)
    secs_prev = df.groupby("account")["timestamp"].diff().dt.total_seconds()
    df["secs_since_prev"] = secs_prev.fillna(86400.0).clip(upper=86400.0)

    counts = []
    for _, g in df.groupby("account", sort=False):
        t = g["timestamp"].to_numpy()
        c = np.zeros(len(t), dtype=int)
        lo = 0
        for hi in range(len(t)):
            while t[hi] - t[lo] > np.timedelta64(1, "h"):
                lo += 1
            c[hi] = hi - lo  # txns in the preceding hour (excl. current)
        counts.append(pd.Series(c, index=g.index))
    df["txn_count_1h"] = pd.concat(counts).sort_index()

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the model feature matrix in a fixed column order."""
    return df[FEATURES].to_numpy(dtype=float)
