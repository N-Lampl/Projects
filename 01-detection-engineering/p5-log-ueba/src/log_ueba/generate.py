"""Synthetic authentication-event generator with injected lateral-movement anomalies.

We model an enterprise where, on a normal day, each user authenticates from a small
"home" set of source hosts to a small set of destination hosts, mostly during business
hours, using interactive/network logons. This is the *benign* behaviour an analyst's
baseline would learn.

We then inject red-team-style anomalies that look like the classic LANL-dataset
lateral movement (Kent, 2015): a compromised credential fans out across many hosts it
has never touched, often off-hours, using service/remote logon types.

Each row is one auth event:
    timestamp, user, src_host, dst_host, logon_type, success, is_anomaly

The generator is the whole reason this project runs offline -- no LANL download, no
LogHub, no network. The real-data path is documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGON_TYPES = ["Interactive", "Network", "Service", "RemoteInteractive", "Batch"]
# Probabilities for a *normal* event: humans log on interactively / over the network.
NORMAL_LOGON_P = np.array([0.45, 0.40, 0.05, 0.05, 0.05])
# An attacker pivoting with stolen creds skews toward service / remote logons.
ATTACK_LOGON_P = np.array([0.05, 0.25, 0.35, 0.30, 0.05])

SECONDS_PER_DAY = 24 * 3600


@dataclass
class GenConfig:
    n_users: int = 60
    n_hosts: int = 120
    n_days: int = 14
    events_per_user_per_day: int = 18
    n_compromised_users: int = 4
    anomaly_fanout: int = 25  # distinct new hosts a compromised cred touches
    seed: int = 42


def _business_hour_seconds(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample times-of-day concentrated in 8:00-18:00 (a working day)."""
    # mixture: most events in business hours, a few scattered
    mu, sigma = 13 * 3600, 2.5 * 3600  # centred ~1pm
    t = rng.normal(mu, sigma, n)
    return np.clip(t, 0, SECONDS_PER_DAY - 1)


def _off_hour_seconds(rng: np.random.Generator, n: int) -> np.ndarray:
    """Attacker times: skewed to nights / early morning (0:00-5:00, 20:00-24:00)."""
    choices = rng.random(n) < 0.5
    early = rng.uniform(0, 5 * 3600, n)
    late = rng.uniform(20 * 3600, SECONDS_PER_DAY, n)
    return np.where(choices, early, late)


def generate_auth_events(cfg: GenConfig | None = None) -> pd.DataFrame:
    """Return a DataFrame of synthetic auth events with a ground-truth `is_anomaly`."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)

    users = [f"U{i:03d}" for i in range(cfg.n_users)]
    hosts = [f"C{i:04d}" for i in range(cfg.n_hosts)]

    # Each user has a small home-host footprint they normally touch.
    home_src = {u: rng.choice(hosts, size=rng.integers(1, 3), replace=False) for u in users}
    home_dst = {u: rng.choice(hosts, size=rng.integers(2, 6), replace=False) for u in users}

    rows: list[dict] = []

    # ---- benign baseline -------------------------------------------------
    for day in range(cfg.n_days):
        day_offset = day * SECONDS_PER_DAY
        for u in users:
            n = rng.poisson(cfg.events_per_user_per_day)
            tod = _business_hour_seconds(rng, n)
            for k in range(n):
                src = rng.choice(home_src[u])
                dst = rng.choice(home_dst[u])
                lt = rng.choice(LOGON_TYPES, p=NORMAL_LOGON_P)
                success = rng.random() > 0.03  # occasional benign failure
                rows.append(
                    {
                        "timestamp": int(day_offset + tod[k]),
                        "user": u,
                        "src_host": src,
                        "dst_host": dst,
                        "logon_type": lt,
                        "success": int(success),
                        "is_anomaly": 0,
                    }
                )

    # ---- injected lateral movement --------------------------------------
    compromised = rng.choice(users, size=cfg.n_compromised_users, replace=False)
    for u in compromised:
        # attack happens on one day, off-hours, fanning out across new hosts
        day = int(rng.integers(cfg.n_days // 2, cfg.n_days))
        day_offset = day * SECONDS_PER_DAY
        new_dsts = rng.choice(hosts, size=cfg.anomaly_fanout, replace=False)
        tod = np.sort(_off_hour_seconds(rng, cfg.anomaly_fanout))
        src = rng.choice(home_src[u])  # often pivots from the user's own box
        for k, dst in enumerate(new_dsts):
            lt = rng.choice(LOGON_TYPES, p=ATTACK_LOGON_P)
            success = rng.random() > 0.25  # lots of failed attempts while probing
            rows.append(
                {
                    "timestamp": int(day_offset + tod[k]),
                    "user": u,
                    "src_host": src,
                    "dst_host": dst,
                    "logon_type": lt,
                    "success": int(success),
                    "is_anomaly": 1,
                }
            )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.attrs["compromised_users"] = list(compromised)
    return df
