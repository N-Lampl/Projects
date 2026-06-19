"""The model-serving logic, independent of any web framework.

`PredictionService.handle(...)` takes the already-parsed pieces of an HTTP request
(api key, path, parsed JSON body) and returns a (status_code, response_dict) pair,
applying the full security pipeline in order:

    1. authenticate  -> 401 on bad/missing key
    2. rate limit    -> 429 when the principal's token bucket is empty
    3. validate body -> 422 on malformed input
    4. predict       -> 200 with class + probabilities

Every step writes an audit event. Both the stdlib server and the FastAPI app call
this, so the security behaviour and tests are identical across transports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ServedModel, train_model
from .security import (
    ApiKeyStore,
    AuditLog,
    SecurityControls,
    TokenBucketLimiter,
    ValidationError,
    validate_predict_input,
)


@dataclass
class Response:
    status: int
    body: dict


class PredictionService:
    def __init__(self, model: ServedModel, controls: SecurityControls) -> None:
        self.model = model
        self.controls = controls

    # -- factory used by both servers and tests -------------------------------
    @classmethod
    def build(
        cls,
        api_keys: dict[str, str] | None = None,
        capacity: int = 5,
        refill_per_sec: float = 1.0,
        clock=None,
    ) -> PredictionService:
        if api_keys is None:
            api_keys = {"demo-secret-key": "demo-client"}
        keys = ApiKeyStore(api_keys)
        if clock is not None:
            limiter = TokenBucketLimiter(capacity, refill_per_sec, clock=clock)
        else:
            limiter = TokenBucketLimiter(capacity, refill_per_sec)
        controls = SecurityControls(keys=keys, limiter=limiter, audit=AuditLog())
        model = train_model()
        return cls(model, controls)

    # -- the single entry point used by every transport -----------------------
    def handle(self, api_key: str | None, path: str, body: object) -> Response:
        c = self.controls

        if path == "/health":
            return Response(200, {"status": "ok"})

        if path != "/predict":
            c.audit.record("-", path, "error", "not found")
            return Response(404, {"error": "not found"})

        # 1. authentication
        principal = c.keys.authenticate(api_key)
        if principal is None:
            c.audit.record("-", "/predict", "denied", "auth: invalid or missing api key")
            return Response(401, {"error": "invalid or missing API key"})

        # 2. rate limiting
        if not c.limiter.allow(principal):
            c.audit.record(principal, "/predict", "denied", "rate limit exceeded")
            return Response(
                429,
                {"error": "rate limit exceeded", "retry_after_sec": 1},
            )

        # 3. input validation
        try:
            rows = validate_predict_input(body, self.model.n_features)
        except ValidationError as e:
            c.audit.record(principal, "/predict", "error", f"validation: {e}")
            return Response(422, {"error": str(e)})

        # 4. inference
        x = np.asarray(rows, dtype=np.float64)
        proba = self.model.predict_proba(x)
        preds = self.model.predict(x)
        results = [
            {
                "prediction": int(preds[i]),
                "probabilities": {
                    str(self.model.classes[k]): float(proba[i, k])
                    for k in range(len(self.model.classes))
                },
            }
            for i in range(len(rows))
        ]
        c.audit.record(principal, "/predict", "ok", f"{len(rows)} instance(s)")
        return Response(200, {"results": results})
