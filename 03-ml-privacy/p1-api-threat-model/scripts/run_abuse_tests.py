#!/usr/bin/env python3
"""Drive the PredictionService through a battery of abuse cases, prove each control
fires, and write results/figures/*.png + results/metrics.json. Run via `make demo`.

Default path needs NO web server or FastAPI: it calls PredictionService.handle()
directly (the same code both transports run), with an injected clock so the
token-bucket sweep is deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_threat_model import PredictionService, set_seed  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"

GOOD_KEY = "demo-secret-key"
BAD_KEY = "totally-wrong-key"
N_FEATURES = 8


class FakeClock:
    """Manually advanced monotonic clock for deterministic rate-limit tests."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def valid_body(n: int = 1) -> dict:
    return {"instances": [[0.1 * (j + 1) for j in range(N_FEATURES)] for _ in range(n)]}


def run_abuse_suite() -> dict:
    """Returns a structured result for every abuse case + the bucket sweep."""
    clock = FakeClock()
    # Audit-accumulating service used for the auth/audit cases.
    svc = PredictionService.build(capacity=5, refill_per_sec=1.0, clock=clock)
    cases: list[dict] = []

    def case(name, control, status, expected):
        cases.append(
            {
                "case": name,
                "control": control,
                "status": status,
                "expected_status": expected,
                "passed": status == expected,
            }
        )

    def fresh():
        # Each validation case gets a full bucket so we isolate the control under
        # test (the pipeline runs auth -> rate-limit -> validation by design).
        return PredictionService.build(capacity=5, refill_per_sec=1.0, clock=FakeClock())

    # --- happy path -------------------------------------------------------
    r = svc.handle(GOOD_KEY, "/predict", valid_body())
    case("authorized valid request", "baseline", r.status, 200)

    # --- auth: missing & wrong key ---------------------------------------
    r = svc.handle(None, "/predict", valid_body())
    case("missing API key", "auth", r.status, 401)
    r = svc.handle(BAD_KEY, "/predict", valid_body())
    case("wrong API key", "auth", r.status, 401)

    # --- input validation: malformed bodies ------------------------------
    r = fresh().handle(GOOD_KEY, "/predict", {"instances": "not-a-list"})
    case("instances not a list", "input-validation", r.status, 422)
    r = fresh().handle(GOOD_KEY, "/predict", {"instances": [[1.0, 2.0]]})
    case("wrong feature count", "input-validation", r.status, 422)
    r = fresh().handle(GOOD_KEY, "/predict", {"instances": [["x"] * N_FEATURES]})
    case("non-numeric feature", "input-validation", r.status, 422)
    r = fresh().handle(
        GOOD_KEY, "/predict", {"instances": [[float("nan")] + [0.0] * (N_FEATURES - 1)]}
    )
    case("non-finite (NaN) feature", "input-validation", r.status, 422)
    r = fresh().handle(GOOD_KEY, "/predict", {"instances": [[1.0] * N_FEATURES] * 999})
    case("oversized batch (DoS)", "input-validation", r.status, 422)
    r = fresh().handle(GOOD_KEY, "/predict", {})
    case("missing 'instances' field", "input-validation", r.status, 422)

    # --- rate limiting: burst past the bucket ----------------------------
    # Fresh principal so the bucket starts full at `capacity`.
    burst_svc = PredictionService.build(capacity=5, refill_per_sec=1.0, clock=clock)
    statuses = [burst_svc.handle(GOOD_KEY, "/predict", valid_body()).status for _ in range(10)]
    allowed = sum(1 for s in statuses if s == 200)
    throttled = sum(1 for s in statuses if s == 429)
    cases.append(
        {
            "case": "burst of 10 requests, capacity 5",
            "control": "rate-limit",
            "status": f"{allowed} allowed / {throttled} throttled",
            "expected_status": "5 allowed / 5 throttled",
            "passed": allowed == 5 and throttled == 5,
        }
    )
    # after refill (advance clock 3s @ 1 tok/s) more requests succeed
    clock.advance(3.0)
    refill_allowed = sum(
        1 for _ in range(3) if burst_svc.handle(GOOD_KEY, "/predict", valid_body()).status == 200
    )
    cases.append(
        {
            "case": "after 3s refill, 3 more requests",
            "control": "rate-limit",
            "status": f"{refill_allowed} allowed",
            "expected_status": "3 allowed",
            "passed": refill_allowed == 3,
        }
    )

    # --- audit log: denials were recorded --------------------------------
    audit = svc.controls.audit
    denied = svc.controls.audit.count(outcome="denied")
    errors = svc.controls.audit.count(outcome="error")
    cases.append(
        {
            "case": "audit log recorded auth denials",
            "control": "audit",
            "status": f"{denied} denied events",
            "expected_status": ">=2 denied events",
            "passed": denied >= 2,
        }
    )

    # bucket-depletion sweep for the figure (fresh principal/clock)
    sweep_clock = FakeClock()
    sweep_svc = PredictionService.build(capacity=5, refill_per_sec=1.0, clock=sweep_clock)
    tokens_trace = []
    status_trace = []
    for _ in range(12):
        tokens_trace.append(sweep_svc.controls.limiter.tokens_left(GOOD_KEY))
        status_trace.append(sweep_svc.handle(GOOD_KEY, "/predict", valid_body()).status)

    return {
        "cases": cases,
        "tokens_trace": tokens_trace,
        "status_trace": status_trace,
        "audit_total": len(audit.events()),
        "audit_denied": denied,
        "audit_errors": errors,
    }


def plot_rate_limit(tokens_trace, status_trace) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(1, len(tokens_trace) + 1))
    ax.step(xs, tokens_trace, where="post", color="#2c7fb8", linewidth=2, label="tokens available")
    for i, (x, s) in enumerate(zip(xs, status_trace)):
        ax.scatter(
            x,
            -0.4,
            marker="o",
            s=80,
            color="#2ca25f" if s == 200 else "#de2d26",
            zorder=3,
        )
    ax.scatter([], [], color="#2ca25f", label="200 allowed")
    ax.scatter([], [], color="#de2d26", label="429 throttled")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("request # (no time elapsed -> no refill)")
    ax.set_ylabel("tokens in bucket")
    ax.set_title("Token-bucket rate limiting: capacity 5, then throttle", pad=12)
    ax.set_ylim(-1, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "rate_limit_bucket.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_controls(cases) -> Path:
    by_control: dict[str, list[bool]] = {}
    for c in cases:
        by_control.setdefault(c["control"], []).append(c["passed"])
    controls = list(by_control)
    passed = [sum(v) for v in by_control.values()]
    totals = [len(v) for v in by_control.values()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(controls, totals, color="#dddddd", label="abuse cases")
    ax.bar(controls, passed, color="#2ca25f", label="blocked as expected")
    for i, (p, t) in enumerate(zip(passed, totals)):
        ax.annotate(f"{p}/{t}", (i, t), textcoords="offset points", xytext=(0, 4), ha="center")
    ax.set_ylabel("number of abuse cases")
    ax.set_title("Security controls vs abuse tests (blocked / total)", pad=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(totals) + 1)
    fig.tight_layout()
    out = FIG_DIR / "controls_coverage.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the API abuse-test suite + write artifacts.")
    ap.add_argument("--json-only", action="store_true", help="skip figures")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    res = run_abuse_suite()
    cases = res["cases"]
    n_pass = sum(1 for c in cases if c["passed"])
    n_total = len(cases)

    print("abuse-test results:")
    for c in cases:
        flag = "PASS" if c["passed"] else "FAIL"
        print(f"  [{flag}] {c['control']:<17} {c['case']:<42} -> {c['status']}")
    print(f"\n{n_pass}/{n_total} controls behaved as expected")

    figures = []
    if not args.json_only:
        figures.append(plot_rate_limit(res["tokens_trace"], res["status_trace"]))
        figures.append(plot_controls(cases))

    metrics = {
        "project": "p1-api-threat-model",
        "summary": (
            f"Hardened model-serving endpoint: {n_pass}/{n_total} abuse cases blocked by "
            "API-key auth, token-bucket rate limiting, strict input validation, and audit logging."
        ),
        "seed": 42,
        "transport_default": "stdlib http.server (no fastapi required)",
        "transport_optional": "fastapi + uvicorn",
        "controls": {
            "api_key_auth": "salted SHA-256 hashes, constant-time compare",
            "rate_limit": "per-principal token bucket (capacity 5, 1 tok/s refill)",
            "input_validation": "type/shape/finiteness/range/batch-size checks -> 422",
            "audit_log": "in-memory ring buffer of auth/rate/predict events",
        },
        "abuse_tests": {
            "total": n_total,
            "blocked_as_expected": n_pass,
            "cases": cases,
        },
        "audit": {
            "total_events": res["audit_total"],
            "denied_events": res["audit_denied"],
            "error_events": res["audit_errors"],
        },
        "figures": [str(p.relative_to(PROJECT)) for p in figures],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    for p in figures:
        print(f"wrote {p.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
