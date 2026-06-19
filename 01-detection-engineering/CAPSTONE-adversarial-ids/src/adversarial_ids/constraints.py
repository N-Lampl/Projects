"""Feature-mutability + consistency constraints for adversarial NIDS evasion.

An adversary evading a network IDS is NOT a free L-inf perturbation in pixel
space. They can only change features they actually control at packet-craft time,
and the resulting flow must stay a *valid, consistent* network flow -- otherwise
it is trivially droppable by a sanity check long before it reaches the model.

This module encodes that domain knowledge as:

* a partition of the synthetic-flow schema into MUTABLE vs IMMUTABLE features,
* per-feature numeric bounds (rates live in [0, 1], byte counts are non-negative,
  etc.),
* a direction mask (some features can only be *increased* by an attacker who is
  adding padding / extra requests -- you cannot un-send bytes),
* a ``project`` step that snaps a candidate perturbation back onto the feasible
  set after every attack step.

Only NUMERIC features are attacked here; the categorical fields (protocol /
service / flag) are treated as immutable identity of the connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- Which numeric features can an attacker realistically move? -------------
#
# MUTABLE: things you can pad, repeat, or stretch when crafting traffic.
#   - duration       : hold the connection open longer (increase only)
#   - src_bytes      : add padding to bytes you send       (increase only)
#   - dst_bytes      : induce a larger response            (increase only)
#   - count          : send more connections to same host  (increase only)
#   - srv_count      : more connections to same service    (increase only)
#
# IMMUTABLE: error-/rate-style aggregates the attacker cannot freely dial
# without breaking the very behaviour they need, plus host-count which is a
# defender-side aggregate. Perturbing these would desync the flow.
MUTABLE_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
]
IMMUTABLE_FEATURES = [
    "serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
]

# Per-feature numeric validity bounds (lo, hi). ``None`` = unbounded on that side.
FEATURE_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "duration": (0.0, None),
    "src_bytes": (0.0, None),
    "dst_bytes": (0.0, None),
    "count": (0.0, None),
    "srv_count": (0.0, None),
    "serror_rate": (0.0, 1.0),
    "rerror_rate": (0.0, 1.0),
    "same_srv_rate": (0.0, 1.0),
    "diff_srv_rate": (0.0, 1.0),
    "dst_host_count": (0.0, None),
}

# Features an attacker can only *increase*: you can pad the bytes you send and
# hold a connection open longer, but you cannot "un-pad". The remaining mutable
# features are BIDIRECTIONAL because an attacker can plausibly move them either
# way -- e.g. "low-and-slow" evasion *reduces* connection counts and induced
# response bytes (via connection-splitting / throttling) to mimic benign volume.
INCREASE_ONLY = {"duration", "src_bytes"}


@dataclass
class ConstraintSpec:
    """Compiled, vectorised view of the constraints for a fixed feature order.

    Built once from ``numeric_features`` (the column order the pipeline's scaler
    sees) and then applied to NumPy matrices of raw (unscaled) numeric features.
    """

    numeric_features: list[str]
    mutable_mask: np.ndarray = field(init=False)       # 1.0 where attacker may move
    increase_only_mask: np.ndarray = field(init=False)  # 1.0 where perturbation >= 0
    lo: np.ndarray = field(init=False)
    hi: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        f = self.numeric_features
        self.mutable_mask = np.array(
            [1.0 if c in MUTABLE_FEATURES else 0.0 for c in f], dtype=np.float64
        )
        self.increase_only_mask = np.array(
            [1.0 if c in INCREASE_ONLY else 0.0 for c in f], dtype=np.float64
        )
        self.lo = np.array(
            [FEATURE_BOUNDS[c][0] if FEATURE_BOUNDS[c][0] is not None else -np.inf for c in f],
            dtype=np.float64,
        )
        self.hi = np.array(
            [FEATURE_BOUNDS[c][1] if FEATURE_BOUNDS[c][1] is not None else np.inf for c in f],
            dtype=np.float64,
        )

    @property
    def n_mutable(self) -> int:
        return int(self.mutable_mask.sum())

    def mask_perturbation(self, delta: np.ndarray) -> np.ndarray:
        """Zero out moves on immutable features and clamp increase-only directions.

        ``delta`` is the *raw-space* perturbation (same shape as the feature
        matrix). After masking, immutable features get exactly 0 change and
        increase-only features get ``max(delta, 0)``.
        """
        d = delta * self.mutable_mask  # immutable -> 0
        # increase-only: clamp negative moves to 0 on those columns
        neg_blocked = np.where(self.increase_only_mask > 0, np.maximum(d, 0.0), d)
        return neg_blocked

    def project(self, x_adv: np.ndarray) -> np.ndarray:
        """Snap candidate flows back into the per-feature validity box."""
        return np.clip(x_adv, self.lo, self.hi)

    def is_consistent(self, x: np.ndarray, *, atol: float = 1e-6) -> np.ndarray:
        """Boolean per-row: are all features inside their validity bounds?"""
        within_lo = (x >= self.lo - atol).all(axis=1)
        within_hi = (x <= self.hi + atol).all(axis=1)
        return within_lo & within_hi

    def immutable_preserved(
        self, x_orig: np.ndarray, x_adv: np.ndarray, *, atol: float = 1e-6
    ) -> np.ndarray:
        """Boolean per-row: did every IMMUTABLE feature stay unchanged?"""
        immutable = self.mutable_mask == 0.0
        if not immutable.any():
            return np.ones(len(x_orig), dtype=bool)
        diff = np.abs(x_adv[:, immutable] - x_orig[:, immutable])
        return (diff <= atol).all(axis=1)


def build_constraints(numeric_features: list[str]) -> ConstraintSpec:
    """Public factory: compile a :class:`ConstraintSpec` for a feature order."""
    return ConstraintSpec(numeric_features=list(numeric_features))
