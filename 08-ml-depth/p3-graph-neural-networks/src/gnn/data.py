"""Data sources: a synthetic SBM by default, the real Cora citation graph as an option.

The default path is fully offline (:func:`gnn.graph.make_sbm`) so tests and CI never
touch the network. :func:`load_cora` pulls the standard **Cora** citation network
(2708 papers, 7 classes, bag-of-words features) from a stable mirror and parses it -
it is exercised only by the ``@slow`` test, which skips on any failure.
"""

from __future__ import annotations

import io
import tarfile

import numpy as np
import torch

from .graph import GraphDataset

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"


def load_cora(url: str = CORA_URL, seed: int = 42) -> GraphDataset:
    """Download Cora and return it as a :class:`GraphDataset`.

    Raises on any network/parse failure - the ``@slow`` test catches and skips.
    """
    from urllib.request import urlopen

    with urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted benchmark host)
        raw = resp.read()

    content = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for name in ("cora/cora.content", "cora/cora.cites"):
            member = tar.extractfile(name)
            if member is None:
                raise RuntimeError(f"missing {name} in Cora archive")
            content[name] = member.read().decode("utf-8")

    # Parse node features + labels.
    rows = [line.split("\t") for line in content["cora/cora.content"].splitlines() if line]
    paper_ids = [r[0] for r in rows]
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    features_np = np.array([[float(v) for v in r[1:-1]] for r in rows], dtype=np.float32)
    class_names = sorted({r[-1] for r in rows})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels_np = np.array([class_to_idx[r[-1]] for r in rows], dtype=np.int64)
    n = len(paper_ids)

    # Parse edges into a dense symmetric adjacency (no self-loops).
    adj_np = np.zeros((n, n), dtype=np.float32)
    for line in content["cora/cora.cites"].splitlines():
        if not line:
            continue
        a, b = line.split("\t")
        if a in id_to_idx and b in id_to_idx:
            i, j = id_to_idx[a], id_to_idx[b]
            if i != j:
                adj_np[i, j] = adj_np[j, i] = 1.0

    n_classes = len(class_names)
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    for c in range(n_classes):
        idx = rng.permutation(np.where(labels_np == c)[0])
        train_mask[idx[:20]] = True
        val_mask[idx[20:50]] = True
    test_mask = ~(train_mask | val_mask)

    return GraphDataset(
        adj=torch.from_numpy(adj_np),
        features=torch.from_numpy(features_np),
        labels=torch.from_numpy(labels_np),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        source=f"Cora citation graph (N={n}, C={n_classes})",
    )
