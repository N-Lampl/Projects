"""Per-account graph features for AML typology detection.

networkx is used *opportunistically* for cycle detection when available, but the
module imports and runs without it: a pure-python adjacency/DFS fallback computes
the same cycle-participation flag. Everything else (degrees, fan-in/out, pass-through
ratio, sub-threshold counts) is plain pandas/numpy and never needs networkx.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import REPORTING_THRESHOLD

# Optional dependency: import lazily so the package works without it.
try:  # pragma: no cover - exercised by whichever path the environment provides
    import networkx as nx

    HAVE_NETWORKX = True
except Exception:  # noqa: BLE001
    nx = None
    HAVE_NETWORKX = False

FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "fan_in",
    "fan_out",
    "amount_in",
    "amount_out",
    "passthrough_ratio",
    "rapid_passthrough",
    "sub_threshold_deposits",
    "sub_threshold_ratio",
    "in_out_balance",
    "in_cycle",
]


def _cycle_members_networkx(edges: pd.DataFrame, accounts: np.ndarray) -> set[int]:
    """Cycle members via strongly-connected components (O(V+E)).

    A node participates in a directed cycle iff it sits in an SCC of size > 1 or has a
    self-loop. Using SCCs (rather than enumerating every simple cycle) keeps runtime
    linear even on a dense random background graph, and matches the pure-python
    fallback exactly.
    """
    g = nx.DiGraph()
    g.add_nodes_from(int(a) for a in accounts)
    g.add_edges_from(zip(edges["src"].astype(int), edges["dst"].astype(int), strict=True))
    members: set[int] = set()
    for comp in nx.strongly_connected_components(g):
        if len(comp) > 1:
            members.update(int(n) for n in comp)
    members.update(int(n) for n in nx.nodes_with_selfloops(g))
    return members


def _cycle_members_fallback(edges: pd.DataFrame, accounts: np.ndarray) -> set[int]:
    """Pure-python cycle detection via Tarjan-style SCCs (iterative, no recursion).

    Any node in a strongly-connected component of size > 1, or with a self-loop,
    participates in a cycle. This is the networkx-free path.
    """
    adj: dict[int, list[int]] = {int(a): [] for a in accounts}
    self_loops: set[int] = set()
    for s, d in zip(edges["src"].astype(int), edges["dst"].astype(int), strict=True):
        if s == d:
            self_loops.add(s)
        else:
            adj[s].append(d)

    index_counter = [0]
    indices: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    on_stack: dict[int, bool] = {}
    stack: list[int] = []
    members: set[int] = set(self_loops)

    for root in adj:
        if root in indices:
            continue
        work = [(root, 0)]  # (node, next child index) -> iterative DFS
        while work:
            node, ci = work[-1]
            if ci == 0:
                indices[node] = lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            if ci < len(adj[node]):
                work[-1] = (node, ci + 1)
                child = adj[node][ci]
                if child not in indices:
                    work.append((child, 0))
                elif on_stack.get(child):
                    lowlink[node] = min(lowlink[node], indices[child])
            else:
                if lowlink[node] == indices[node]:  # SCC root
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp.append(w)
                        if w == node:
                            break
                    if len(comp) > 1:
                        members.update(comp)
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
    return members


def cycle_members(edges: pd.DataFrame, accounts: np.ndarray) -> set[int]:
    """Accounts that participate in a directed cycle (networkx if present, else fallback)."""
    if HAVE_NETWORKX:
        return _cycle_members_networkx(edges, accounts)
    return _cycle_members_fallback(edges, accounts)


def build_features(graph, *, threshold: float = REPORTING_THRESHOLD) -> pd.DataFrame:
    """Compute the per-account feature matrix used by the detector.

    Parameters
    ----------
    graph : AMLGraph
        Output of :func:`aml_typologies.graph.generate_aml_graph`.
    threshold : float
        Reporting threshold; deposits in ``[0.5*threshold, threshold)`` count as
        "sub-threshold" (the structuring signature).
    """
    tx = graph.transactions
    accounts = graph.accounts
    idx = pd.Index(accounts, name="account")

    out_g = tx.groupby("src")
    in_g = tx.groupby("dst")

    out_degree = out_g.size().reindex(idx, fill_value=0)
    in_degree = in_g.size().reindex(idx, fill_value=0)
    fan_out = out_g["dst"].nunique().reindex(idx, fill_value=0)
    fan_in = in_g["src"].nunique().reindex(idx, fill_value=0)
    amount_out = out_g["amount"].sum().reindex(idx, fill_value=0.0)
    amount_in = in_g["amount"].sum().reindex(idx, fill_value=0.0)

    # pass-through: a layering hop forwards most of what it receives, fast
    passthrough_ratio = (amount_out / amount_in.replace(0, np.nan)).fillna(0.0).clip(upper=5.0)
    rapid_passthrough = (
        ((passthrough_ratio >= 0.85) & (passthrough_ratio <= 1.15) & (amount_in > 10_000))
        .astype(int)
    )

    # structuring: many incoming deposits just under the reporting threshold
    sub = tx[(tx["amount"] >= 0.5 * threshold) & (tx["amount"] < threshold)]
    sub_threshold_deposits = sub.groupby("dst").size().reindex(idx, fill_value=0)
    sub_threshold_ratio = (sub_threshold_deposits / in_degree.replace(0, np.nan)).fillna(0.0)

    in_out_balance = (amount_in - amount_out).abs() / (amount_in + amount_out + 1.0)

    members = cycle_members(tx, accounts)
    in_cycle = pd.Series(0, index=idx)
    if members:
        in_cycle.loc[list(members)] = 1

    feats = pd.DataFrame(
        {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "amount_in": amount_in,
            "amount_out": amount_out,
            "passthrough_ratio": passthrough_ratio,
            "rapid_passthrough": rapid_passthrough,
            "sub_threshold_deposits": sub_threshold_deposits,
            "sub_threshold_ratio": sub_threshold_ratio,
            "in_out_balance": in_out_balance,
            "in_cycle": in_cycle,
        },
        index=idx,
    )
    return feats[FEATURE_NAMES]
