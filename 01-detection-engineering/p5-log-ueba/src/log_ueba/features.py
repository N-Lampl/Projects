"""Per-event UEBA feature engineering.

UEBA = User & Entity Behaviour Analytics. The trick is to score each event
*relative to the entity's own history*, not in absolute terms. So we build per-user
running baselines (which hosts they touch, what time they work) and turn every event
into a small numeric vector that captures "how surprising is this for THIS user".

Features (all cheap, all streaming-friendly):
    novel_dst        1 if dst_host never seen for this user before now
    novel_src        1 if src_host never seen for this user before now
    user_dst_card    distinct dst hosts this user has touched so far (fan-out)
    hour_zscore      |event hour - user's mean hour| / user hour-std  (off-hours signal)
    off_hours        1 if outside 7:00-19:00
    is_failure       1 if logon failed
    logon_rarity     1 if a service/remote/batch logon (rarer for humans)
    recent_dst_rate  distinct dst hosts this user touched in a trailing window

These are computed in *event order* so the baseline only ever uses the past --
exactly what a streaming detector (LANL / LogHub filter path) would see.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

RARE_LOGONS = {"Service", "RemoteInteractive", "Batch"}
FEATURE_NAMES = [
    "novel_dst",
    "novel_src",
    "user_dst_card",
    "hour_zscore",
    "off_hours",
    "is_failure",
    "logon_rarity",
    "recent_dst_rate",
]
WINDOW_SECONDS = 3600  # trailing window for the burst-of-new-hosts feature


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stream the events in time order, emitting one feature row per event.

    Returns a DataFrame aligned to `df` with columns FEATURE_NAMES + carried-through
    `is_anomaly`, `timestamp`, `user` (for downstream metrics / plotting).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    seen_dst: dict[str, set] = defaultdict(set)
    seen_src: dict[str, set] = defaultdict(set)
    # Welford running mean/var of event-hour per user.
    cnt: dict[str, int] = defaultdict(int)
    mean: dict[str, float] = defaultdict(float)
    m2: dict[str, float] = defaultdict(float)
    # trailing window of (timestamp, dst) per user for recent fan-out.
    recent: dict[str, deque] = defaultdict(deque)

    out = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        u = row.user
        hour = (row.timestamp % (24 * 3600)) / 3600.0

        novel_dst = float(row.dst_host not in seen_dst[u])
        novel_src = float(row.src_host not in seen_src[u])
        dst_card = float(len(seen_dst[u]))

        # hour z-score against the user's running distribution
        n = cnt[u]
        if n >= 2 and m2[u] > 0:
            std = (m2[u] / n) ** 0.5
            hour_z = abs(hour - mean[u]) / (std + 1e-6)
        else:
            hour_z = 0.0
        hour_z = min(hour_z, 10.0)  # cap to keep IsolationForest well-conditioned

        off_hours = float(hour < 7 or hour >= 19)
        is_failure = float(row.success == 0)
        logon_rarity = float(row.logon_type in RARE_LOGONS)

        # trailing-window distinct-dst rate
        win = recent[u]
        while win and row.timestamp - win[0][0] > WINDOW_SECONDS:
            win.popleft()
        recent_dst_rate = float(len({d for _, d in win}))

        out[i] = [
            novel_dst,
            novel_src,
            dst_card,
            hour_z,
            off_hours,
            is_failure,
            logon_rarity,
            recent_dst_rate,
        ]

        # ---- update baselines AFTER scoring (no peeking at the present) ----
        seen_dst[u].add(row.dst_host)
        seen_src[u].add(row.src_host)
        cnt[u] += 1
        delta = hour - mean[u]
        mean[u] += delta / cnt[u]
        m2[u] += delta * (hour - mean[u])
        win.append((row.timestamp, row.dst_host))

    feats = pd.DataFrame(out, columns=FEATURE_NAMES)
    feats["timestamp"] = df["timestamp"].values
    feats["user"] = df["user"].values
    feats["is_anomaly"] = df["is_anomaly"].values
    return feats
