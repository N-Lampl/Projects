"""Deterministic STRIDE analysis of the ML inference service.

Two layers of findings:

1. **Per-flow rule engine** — generic STRIDE threats derived from each data flow's
   attributes (does it cross a boundary? authenticated? encrypted? PII?).
2. **ML-specific threats** — curated threats that classic STRIDE under-covers but
   matter for an inference service (model theft, poisoning, evasion/adversarial
   examples, membership inference, prompt-style abuse). These are mapped onto the
   STRIDE categories so the counts stay consistent.

The output is a flat list of `Threat` records; `summarize()` rolls them up into the
per-category counts written to results/metrics.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .model import STRIDE, System


@dataclass(frozen=True)
class Threat:
    id: str
    category: str  # one of STRIDE keys
    target: str  # element or flow the threat applies to
    title: str
    description: str
    mitigation: str
    severity: str  # "Low" | "Medium" | "High"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _flow_threats(system: System) -> list[Threat]:
    """Generic STRIDE threats inferred from data-flow attributes."""
    out: list[Threat] = []
    n = 0

    def add(cat: str, flow, title: str, desc: str, mit: str, sev: str) -> None:
        nonlocal n
        n += 1
        out.append(Threat(f"F{n:02d}", cat, f"{flow.source} -> {flow.sink}", title, desc, mit, sev))

    for f in system.flows:
        # Spoofing: any flow that is not authenticated can be impersonated.
        if not f.authenticated:
            add("Spoofing", f, f"Unauthenticated flow '{f.data}'",
                "Source identity is not verified; an attacker can impersonate the sender.",
                "Require mTLS / signed service tokens (OIDC) on this flow.", "High")
        # Tampering: anything crossing a boundary can be modified in transit if not encrypted.
        if f.crosses_boundary and not f.encrypted:
            add("Tampering", f, f"In-transit tampering of '{f.data}'",
                f"Flow crosses '{f.crosses_boundary}' without encryption; payload can be altered.",
                "Enforce TLS 1.2+ and HMAC/signature on the payload.", "High")
        # Information Disclosure: PII crossing a boundary is a confidentiality risk.
        if f.carries_pii and f.crosses_boundary:
            add("Information Disclosure", f, f"PII exposure on '{f.data}'",
                f"Personally identifiable data crosses '{f.crosses_boundary}'.",
                "Encrypt in transit + at rest; minimize/redact PII; scope access.", "Medium")
        # Denial of Service: every externally reachable flow can be flooded.
        if f.crosses_boundary == "Internet / DMZ":
            add("Denial of Service", f, f"Request flooding of '{f.data}'",
                "Public-facing flow can be saturated, exhausting inference capacity.",
                "Rate-limit per principal, autoscale, request quotas, WAF.", "Medium")
        # Repudiation: state-changing flows without logging can be denied later.
        if "deploy" in f.data or f.sink == "Prediction Log":
            add("Repudiation", f, f"Action on '{f.data}' may be repudiated",
                "State-changing or audited action lacks tamper-evident attribution.",
                "Append-only signed audit log with actor identity + timestamp.", "Low")
    return out


def _ml_threats(system: System) -> list[Threat]:
    """ML-specific threats that generic STRIDE rules miss (curated, mapped to STRIDE)."""
    return [
        Threat("ML01", "Tampering", "Model Registry",
               "Model / supply-chain poisoning",
               "A tampered or backdoored artifact is promoted into the registry and served.",
               "Sign artifacts (Sigstore), verify signature on load, pin hashes, gate promotion.",
               "High"),
        Threat("ML02", "Tampering", "Inference Service",
               "Adversarial-example evasion",
               "Crafted inputs (e.g. FGSM/PGD) flip predictions at inference time.",
               "Adversarial training, input sanitization, detect OOD/adversarial inputs.",
               "Medium"),
        Threat("ML03", "Information Disclosure", "Inference Service",
               "Model extraction / theft",
               "Repeated queries reconstruct a surrogate of the proprietary model.",
               "Rate-limit + watermark, restrict confidence outputs, monitor query patterns.",
               "Medium"),
        Threat("ML04", "Information Disclosure", "Prediction Log",
               "Membership / training-data inference",
               "Confidence scores leak whether a record was in the training set.",
               "Round/cap confidences, differential privacy, limit output granularity.",
               "Medium"),
        Threat("ML05", "Denial of Service", "Inference Service",
               "Sponge / heavy-input resource exhaustion",
               "Maliciously large or pathological inputs blow up latency and memory.",
               "Strict input-size/shape validation, timeouts, per-request resource caps.",
               "Medium"),
        Threat("ML06", "Spoofing", "Feature Store",
               "Feature poisoning via spoofed upstream",
               "An attacker spoofs a feature source to bias online features.",
               "Authenticate feature producers, validate ranges, anomaly-detect drift.",
               "Medium"),
        Threat("ML07", "Elevation of Privilege", "Inference Service",
               "Unsafe artifact deserialization (pickle RCE)",
               "Loading an untrusted pickle/torch artifact executes arbitrary code.",
               "Use safe formats (safetensors), load in a sandbox, verify signatures first.",
               "High"),
        Threat("ML08", "Elevation of Privilege", "API Gateway",
               "Lateral movement after token compromise",
               "A stolen service token grants broad access across the mesh.",
               "Least-privilege scopes, short-lived tokens, mesh-level network policy.",
               "Medium"),
        Threat("ML09", "Repudiation", "MLOps Operator",
               "Unattributed model deployment",
               "A model is promoted with no tamper-evident record of who/when/what.",
               "Signed deployment audit trail tied to operator identity + change ticket.",
               "Low"),
    ]


def analyze(system: System) -> list[Threat]:
    """Return all threats (generic flow-derived + ML-specific), deterministically ordered."""
    threats = _flow_threats(system) + _ml_threats(system)
    # stable order: by STRIDE category order, then id
    order = {cat: i for i, cat in enumerate(STRIDE)}
    return sorted(threats, key=lambda t: (order[t.category], t.id))


def summarize(threats: list[Threat]) -> dict[str, int]:
    """Count threats per STRIDE category (every category present, even if zero)."""
    counts = dict.fromkeys(STRIDE, 0)
    for t in threats:
        counts[t.category] += 1
    return counts
