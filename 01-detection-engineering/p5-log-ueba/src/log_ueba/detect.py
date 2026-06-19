"""Anomaly detectors over the UEBA feature matrix.

Default (always available): scikit-learn IsolationForest -- an unsupervised tree
ensemble that isolates rare points in few splits. No labels used at fit time; this is
how a real UEBA deployment works (you don't have labelled lateral movement up front).

Optional (lazy torch import): a tiny dense autoencoder. It learns to reconstruct
*normal* events; anomalies reconstruct poorly, so reconstruction error is the score.
Imported inside the function so the module loads fine without torch installed.

Both return a 1-D array of anomaly scores where *higher = more anomalous*, so the
downstream metrics (precision@k, time-to-detect) treat them identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES


def _matrix(feats) -> np.ndarray:
    return feats[FEATURE_NAMES].to_numpy(dtype=np.float32)


def isolation_forest_scores(feats, contamination: float = 0.02, seed: int = 42) -> np.ndarray:
    """Fit IsolationForest unsupervised; return per-event anomaly scores (higher=worse)."""
    x = _matrix(feats)
    x = StandardScaler().fit_transform(x)
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(x)
    # decision_function: higher = more normal. Negate so higher = more anomalous.
    return -clf.decision_function(x)


def autoencoder_scores(
    feats,
    epochs: int = 20,
    hidden: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Optional torch autoencoder. Reconstruction error = anomaly score (higher=worse).

    Trains on ALL events (unsupervised); since anomalies are rare the net mostly learns
    the normal manifold, so anomalous rows reconstruct poorly.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x = _matrix(feats)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xn = (x - mu) / sd
    xt = torch.tensor(xn, dtype=torch.float32)

    d = xt.shape[1]
    net = nn.Sequential(
        nn.Linear(d, 8),
        nn.ReLU(),
        nn.Linear(8, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 8),
        nn.ReLU(),
        nn.Linear(8, d),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        recon = net(xt)
        loss = loss_fn(recon, xt)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        err = ((net(xt) - xt) ** 2).mean(dim=1).numpy()
    return err
