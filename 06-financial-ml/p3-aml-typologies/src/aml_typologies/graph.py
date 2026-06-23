"""Synthetic AML transaction graph with planted laundering typologies.

Accounts are nodes; transfers are directed, timestamped, valued edges. On top of a
background of benign "normal" payment activity we plant two classic typologies:

* STRUCTURING / smurfing: a *funnel* account receives many sub-threshold deposits
  (just under a reporting threshold, e.g. < $10,000) from a fleet of mule accounts,
  then pushes the aggregate out.
* LAYERING: funds move through a directed *chain* (and sometimes a closed *cycle*)
  of intermediary accounts in rapid succession, each hop passing through most of
  what it received -- obscuring the origin of the money.

Everything is deterministic given a seed so `make detect` reproduces exactly.
The ground-truth label is per-account: 1 if the account participates in any
planted typology, else 0. This is what we score detection against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Currency reporting threshold (e.g. US CTR at $10,000). "Sub-threshold" deposits
# sit just below this to dodge automatic reporting -- the structuring signature.
REPORTING_THRESHOLD = 10_000.0


@dataclass
class AMLGraph:
    """A generated transaction graph plus per-account ground truth."""

    transactions: pd.DataFrame  # columns: src, dst, amount, time, typology
    labels: pd.Series  # index = account id, value in {0,1}
    typology: pd.Series  # index = account id, value in {"none","structuring","layering"}
    rings: list[list[int]] = field(default_factory=list)  # planted layering cycles

    @property
    def accounts(self) -> np.ndarray:
        return self.labels.index.to_numpy()

    @property
    def n_accounts(self) -> int:
        return len(self.labels)


def _normal_edges(rng, account_ids, n_edges):
    """Background benign payments: random pairs, modest log-normal amounts."""
    src = rng.choice(account_ids, size=n_edges)
    dst = rng.choice(account_ids, size=n_edges)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    amount = np.round(rng.lognormal(mean=6.0, sigma=1.0, size=len(src)), 2)
    time = rng.uniform(0, 30 * 24 * 3600, size=len(src))  # 30 days, seconds
    return pd.DataFrame(
        {"src": src, "dst": dst, "amount": amount, "time": time, "typology": "normal"}
    )


def generate_aml_graph(
    n_accounts: int = 1500,
    n_normal_edges: int = 9000,
    n_structuring: int = 12,
    smurfs_per_ring: int = 18,
    n_layering: int = 10,
    chain_len: int = 6,
    seed: int = 42,
) -> AMLGraph:
    """Build a synthetic AML graph with planted structuring + layering typologies.

    Returns an :class:`AMLGraph` whose ``labels`` mark every account that takes part
    in a planted typology (the funnel + its smurfs; every hop in a layering chain).
    """
    rng = np.random.default_rng(seed)
    account_ids = np.arange(n_accounts)

    typology = pd.Series("none", index=account_ids, dtype=object)
    edge_frames = [_normal_edges(rng, account_ids, n_normal_edges)]
    rings: list[list[int]] = []
    # accumulate typology edges as plain rows, build ONE DataFrame at the end
    # (per-edge pd.DataFrame()/concat is quadratic and was the runtime bottleneck)
    rows: list[dict] = []

    # ---- STRUCTURING: a funnel account fed by many sub-threshold deposits --------
    for _ in range(n_structuring):
        funnel = int(rng.integers(n_accounts))
        smurfs = rng.choice(account_ids[account_ids != funnel], size=smurfs_per_ring, replace=False)
        typology.loc[funnel] = "structuring"
        typology.loc[smurfs] = "structuring"
        t0 = rng.uniform(0, 25 * 24 * 3600)
        for i, s in enumerate(smurfs):
            # deposits clustered just under the reporting threshold, in a tight window
            amt = float(np.round(rng.uniform(0.80, 0.985) * REPORTING_THRESHOLD, 2))
            rows.append(
                {
                    "src": int(s),
                    "dst": funnel,
                    "amount": amt,
                    "time": t0 + i * rng.uniform(60, 1800),
                    "typology": "structuring",
                }
            )
        # funnel forwards the aggregate onward (placement -> integration)
        downstream = int(rng.choice(account_ids[account_ids != funnel]))
        rows.append(
            {
                "src": funnel,
                "dst": downstream,
                "amount": float(np.round(0.9 * smurfs_per_ring * REPORTING_THRESHOLD, 2)),
                "time": t0 + smurfs_per_ring * 1800 + 3600,
                "typology": "structuring",
            }
        )

    # ---- LAYERING: funds pass rapidly through a chain, sometimes closing a cycle --
    for j in range(n_layering):
        chain = rng.choice(account_ids, size=chain_len, replace=False).tolist()
        typology.loc[chain] = "layering"
        amount = float(np.round(rng.uniform(40_000, 120_000), 2))
        t = rng.uniform(0, 20 * 24 * 3600)
        close_cycle = j % 2 == 0  # half of the chains loop back into a ring
        hops = list(zip(chain[:-1], chain[1:], strict=True))
        if close_cycle:
            hops.append((chain[-1], chain[0]))
            rings.append([int(a) for a in chain])
        for a, b in hops:
            # rapid pass-through: most of what came in goes straight back out
            passed = float(np.round(amount * rng.uniform(0.90, 0.98), 2))
            t += rng.uniform(120, 3600)  # minutes-to-an-hour between hops
            rows.append(
                {
                    "src": int(a),
                    "dst": int(b),
                    "amount": passed,
                    "time": t,
                    "typology": "layering",
                }
            )
            amount = passed

    if rows:
        edge_frames.append(pd.DataFrame(rows))
    transactions = (
        pd.concat(edge_frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    )
    labels = (typology != "none").astype(int)
    return AMLGraph(transactions=transactions, labels=labels, typology=typology, rings=rings)
