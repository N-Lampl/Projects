"""Framework-agnostic security controls for a model-serving endpoint.

These are plain, dependency-free Python objects so the SAME logic backs both the
stdlib HTTP server (default, always-available path) and the optional FastAPI app.
Each control maps to a row in the threat model (see README):

    - ApiKeyStore            -> A01/Auth  (spoofing, broken access control)
    - TokenBucketLimiter     -> A04/Rate  (denial of service, model extraction by scraping)
    - validate_predict_input -> A03/Input (injection, malformed-input crashes)
    - AuditLog               -> A09/Audit (repudiation, lack of monitoring)

Token-bucket uses an injectable clock so tests are deterministic (no sleeps).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable


# --------------------------------------------------------------------------- auth
def hash_key(raw_key: str) -> str:
    """Store only a salted SHA-256 of the key, never the key itself."""
    return hashlib.sha256(("atm::" + raw_key).encode()).hexdigest()


class ApiKeyStore:
    """API-key auth. Keys are compared in constant time against stored hashes."""

    def __init__(self, key_to_principal: dict[str, str] | None = None) -> None:
        # principal name -> key hash
        self._hashes: dict[str, str] = {}
        if key_to_principal:
            for raw_key, principal in key_to_principal.items():
                self._hashes[principal] = hash_key(raw_key)

    def add(self, principal: str, raw_key: str) -> None:
        self._hashes[principal] = hash_key(raw_key)

    @staticmethod
    def generate_key() -> str:
        return "atm_" + secrets.token_urlsafe(24)

    def authenticate(self, raw_key: str | None) -> str | None:
        """Return the principal for a valid key, else None. Constant-time compare."""
        if not raw_key:
            return None
        candidate = hash_key(raw_key)
        # Always iterate fully and compare every entry to avoid timing leaks.
        matched: str | None = None
        for principal, stored in self._hashes.items():
            if hmac.compare_digest(candidate, stored):
                matched = principal
        return matched


# --------------------------------------------------------------------------- rate limit
@dataclass
class _Bucket:
    tokens: float
    last: float


class TokenBucketLimiter:
    """Per-principal token bucket.

    Each principal gets `capacity` tokens that refill at `refill_per_sec`. A request
    costs one token; if the bucket is empty the request is denied (HTTP 429). This
    bounds both raw DoS and slow model-extraction-by-scraping.
    """

    def __init__(
        self,
        capacity: int = 5,
        refill_per_sec: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def _refill(self, b: _Bucket, now: float) -> None:
        elapsed = max(0.0, now - b.last)
        b.tokens = min(self.capacity, b.tokens + elapsed * self.refill)
        b.last = now

    def allow(self, principal: str, cost: float = 1.0) -> bool:
        with self._lock:
            now = self._clock()
            b = self._buckets.get(principal)
            if b is None:
                b = _Bucket(tokens=self.capacity, last=now)
                self._buckets[principal] = b
            self._refill(b, now)
            if b.tokens >= cost:
                b.tokens -= cost
                return True
            return False

    def tokens_left(self, principal: str) -> float:
        with self._lock:
            b = self._buckets.get(principal)
            if b is None:
                return self.capacity
            self._refill(b, self._clock())
            return b.tokens


# --------------------------------------------------------------------------- input validation
class ValidationError(ValueError):
    """Raised on malformed prediction input -> mapped to HTTP 422."""


def validate_predict_input(payload: object, n_features: int, max_rows: int = 64) -> list[list[float]]:
    """Strictly validate a /predict body and return a clean 2-D float matrix.

    Rejects: missing/oddly-typed fields, wrong feature count, non-finite values,
    oversized batches (resource exhaustion), and out-of-range values that smell like
    probing/overflow attempts. Returns a brand-new nested list (no aliasing).
    """
    import math

    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")
    rows = payload.get("instances")
    if rows is None:
        raise ValidationError("missing required field 'instances'")
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValidationError("'instances' must be a non-empty list")
    if len(rows) > max_rows:
        raise ValidationError(f"too many instances (>{max_rows}); batch size limit exceeded")

    cleaned: list[list[float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValidationError(f"instance {i} must be a list of {n_features} numbers")
        if len(row) != n_features:
            raise ValidationError(
                f"instance {i} has {len(row)} features, expected {n_features}"
            )
        clean_row: list[float] = []
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValidationError(f"instance {i} feature {j} must be a number")
            fv = float(v)
            if not math.isfinite(fv):
                raise ValidationError(f"instance {i} feature {j} is not finite")
            if abs(fv) > 1e6:
                raise ValidationError(f"instance {i} feature {j} out of allowed range")
            clean_row.append(fv)
        cleaned.append(clean_row)
    return cleaned


# --------------------------------------------------------------------------- audit log
@dataclass
class AuditEvent:
    ts: float
    principal: str
    action: str
    outcome: str  # "ok" | "denied" | "error"
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "ts": round(self.ts, 6),
            "principal": self.principal,
            "action": self.action,
            "outcome": self.outcome,
            "detail": self.detail,
        }


class AuditLog:
    """In-memory ring buffer of security-relevant events (auth, rate-limit, predict).

    Real deployments would ship these to a SIEM; here it is queryable so abuse tests
    can assert that denials were actually recorded (non-repudiation).
    """

    def __init__(self, maxlen: int = 10000, clock: Callable[[], float] = time.time) -> None:
        self._events: deque[AuditEvent] = deque(maxlen=maxlen)
        self._clock = clock
        self._lock = Lock()

    def record(self, principal: str, action: str, outcome: str, detail: str = "") -> AuditEvent:
        ev = AuditEvent(self._clock(), principal or "-", action, outcome, detail)
        with self._lock:
            self._events.append(ev)
        return ev

    def events(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def count(self, outcome: str | None = None, action: str | None = None) -> int:
        return sum(
            1
            for e in self.events()
            if (outcome is None or e.outcome == outcome)
            and (action is None or e.action == action)
        )


# --------------------------------------------------------------------------- bundle
@dataclass
class SecurityControls:
    """Bundle wiring all controls together for one app instance."""

    keys: ApiKeyStore
    limiter: TokenBucketLimiter
    audit: AuditLog = field(default_factory=AuditLog)
