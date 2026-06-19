"""Fast smoke tests (run in CI) for every security control, exercised through the
framework-agnostic PredictionService and the default stdlib transport. One slow
end-to-end test (marked @slow) spins up the real FastAPI TestClient if available.
"""

from __future__ import annotations

import pytest

from api_threat_model import (
    ApiKeyStore,
    PredictionService,
    TokenBucketLimiter,
    ValidationError,
    set_seed,
    validate_predict_input,
)
from api_threat_model.security import hash_key

N_FEATURES = 8
GOOD_KEY = "demo-secret-key"


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _body(n: int = 1) -> dict:
    return {"instances": [[0.1 * (j + 1) for j in range(N_FEATURES)] for _ in range(n)]}


def _service(clock=None) -> PredictionService:
    return PredictionService.build(capacity=5, refill_per_sec=1.0, clock=clock or FakeClock())


# --------------------------------------------------------------------- reproducibility
def test_set_seed_is_deterministic():
    import numpy as np

    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert (a == b).all()


# --------------------------------------------------------------------- auth
def test_api_key_never_stored_in_plaintext():
    store = ApiKeyStore({GOOD_KEY: "client"})
    assert GOOD_KEY not in store._hashes.values()
    assert store._hashes["client"] == hash_key(GOOD_KEY)


def test_auth_accepts_valid_rejects_invalid():
    store = ApiKeyStore({GOOD_KEY: "client"})
    assert store.authenticate(GOOD_KEY) == "client"
    assert store.authenticate("nope") is None
    assert store.authenticate(None) is None
    assert store.authenticate("") is None


def test_predict_requires_auth():
    svc = _service()
    assert svc.handle(None, "/predict", _body()).status == 401
    assert svc.handle("wrong", "/predict", _body()).status == 401
    assert svc.handle(GOOD_KEY, "/predict", _body()).status == 200


# --------------------------------------------------------------------- rate limit
def test_token_bucket_blocks_after_capacity():
    clock = FakeClock()
    lim = TokenBucketLimiter(capacity=3, refill_per_sec=1.0, clock=clock)
    assert [lim.allow("p") for _ in range(3)] == [True, True, True]
    assert lim.allow("p") is False  # 4th in same instant -> denied


def test_token_bucket_refills_over_time():
    clock = FakeClock()
    lim = TokenBucketLimiter(capacity=2, refill_per_sec=1.0, clock=clock)
    assert lim.allow("p") and lim.allow("p")
    assert lim.allow("p") is False
    clock.advance(1.0)  # one token back
    assert lim.allow("p") is True
    assert lim.allow("p") is False


def test_rate_limit_is_per_principal():
    clock = FakeClock()
    lim = TokenBucketLimiter(capacity=1, refill_per_sec=1.0, clock=clock)
    assert lim.allow("a") is True
    assert lim.allow("a") is False
    assert lim.allow("b") is True  # b has its own bucket


def test_burst_returns_429_then_recovers():
    clock = FakeClock()
    svc = _service(clock)
    statuses = [svc.handle(GOOD_KEY, "/predict", _body()).status for _ in range(8)]
    assert statuses.count(200) == 5
    assert statuses.count(429) == 3
    clock.advance(2.0)
    assert svc.handle(GOOD_KEY, "/predict", _body()).status == 200


# --------------------------------------------------------------------- input validation
def test_validate_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        validate_predict_input({"instances": "x"}, N_FEATURES)
    with pytest.raises(ValidationError):
        validate_predict_input({"instances": [[1.0, 2.0]]}, N_FEATURES)
    with pytest.raises(ValidationError):
        validate_predict_input({"instances": [["a"] * N_FEATURES]}, N_FEATURES)
    with pytest.raises(ValidationError):
        validate_predict_input({"instances": [[float("inf")] * N_FEATURES]}, N_FEATURES)
    with pytest.raises(ValidationError):
        validate_predict_input({"instances": [[1.0] * N_FEATURES] * 1000}, N_FEATURES)
    with pytest.raises(ValidationError):
        validate_predict_input({}, N_FEATURES)
    with pytest.raises(ValidationError):
        validate_predict_input({"instances": [[True] * N_FEATURES]}, N_FEATURES)  # bool != number


def test_validate_returns_clean_copy():
    out = validate_predict_input({"instances": [[1, 2, 3, 4, 5, 6, 7, 8]]}, N_FEATURES)
    assert out == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
    assert all(isinstance(v, float) for v in out[0])


def test_predict_returns_422_on_malformed_body():
    svc = _service()
    assert svc.handle(GOOD_KEY, "/predict", {"instances": [[1.0, 2.0]]}).status == 422


# --------------------------------------------------------------------- audit log
def test_audit_records_outcomes():
    svc = _service()
    svc.handle(None, "/predict", _body())  # denied
    svc.handle(GOOD_KEY, "/predict", {"instances": [[1.0]]})  # error (validation)
    svc.handle(GOOD_KEY, "/predict", _body())  # ok
    audit = svc.controls.audit
    assert audit.count(outcome="denied") >= 1
    assert audit.count(outcome="error") >= 1
    assert audit.count(outcome="ok") >= 1
    assert audit.count(action="/predict") == len(audit.events())


# --------------------------------------------------------------------- happy path shape
def test_successful_prediction_shape():
    svc = _service()
    resp = svc.handle(GOOD_KEY, "/predict", _body(2))
    assert resp.status == 200
    assert len(resp.body["results"]) == 2
    r0 = resp.body["results"][0]
    assert r0["prediction"] in (0, 1)
    assert abs(sum(r0["probabilities"].values()) - 1.0) < 1e-6


def test_health_needs_no_auth():
    svc = _service()
    assert svc.handle(None, "/health", None).status == 200


# --------------------------------------------------------------------- stdlib transport (in-proc)
def test_stdlib_transport_end_to_end():
    """Boot the real stdlib HTTP server on an ephemeral port and hit it with urllib."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from api_threat_model.stdlib_server import make_handler

    svc = _service()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/predict"
        data = json.dumps(_body()).encode()
        # no key -> 401
        req = urllib.request.Request(
            url, data=data, headers={"content-type": "application/json"}
        )
        try:
            urllib.request.urlopen(req)
            assert False, "expected 401"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        # good key -> 200
        req = urllib.request.Request(
            url,
            data=data,
            headers={"content-type": "application/json", "x-api-key": GOOD_KEY},
        )
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        payload = json.loads(resp.read())
        assert "results" in payload
    finally:
        server.shutdown()


# --------------------------------------------------------------------- optional FastAPI
@pytest.mark.slow
def test_fastapi_transport_with_testclient():
    """End-to-end through the optional FastAPI app + TestClient (skipped if missing)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api_threat_model import create_app

    app = create_app(capacity=5, refill_per_sec=1.0)
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    # missing key
    assert client.post("/predict", json=_body()).status_code == 401
    # valid
    r = client.post("/predict", json=_body(), headers={"x-api-key": GOOD_KEY})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1
    # malformed
    bad = client.post("/predict", json={"instances": [[1.0]]}, headers={"x-api-key": GOOD_KEY})
    assert bad.status_code == 422
    # burst -> 429 eventually
    codes = [
        client.post("/predict", json=_body(), headers={"x-api-key": GOOD_KEY}).status_code
        for _ in range(10)
    ]
    assert 429 in codes
