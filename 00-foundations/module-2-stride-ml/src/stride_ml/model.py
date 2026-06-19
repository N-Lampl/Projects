"""Data-flow model of the ML inference service we threat-model.

Plain dataclasses (no pytm dependency) describe a small but realistic system:

    Client -> API Gateway -> Inference Service -> Model Artifact (loaded from a
    registry) -> Feature/Logging stores.

This same structure is what `threatmodel.py` walks to emit STRIDE findings, and
what `pytm` (optional path) is configured to mirror.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# STRIDE: the six threat categories and the security property each one violates.
STRIDE = {
    "Spoofing": "Authentication",
    "Tampering": "Integrity",
    "Repudiation": "Non-repudiation",
    "Information Disclosure": "Confidentiality",
    "Denial of Service": "Availability",
    "Elevation of Privilege": "Authorization",
}


@dataclass(frozen=True)
class Element:
    """An external entity, process, or datastore in the DFD."""

    name: str
    kind: str  # "actor" | "process" | "datastore"
    description: str = ""


@dataclass(frozen=True)
class Boundary:
    """A trust boundary crossing in the system (where privilege/trust changes)."""

    name: str
    description: str = ""


@dataclass(frozen=True)
class DataFlow:
    """A directed edge: data moving from source to sink, optionally crossing a boundary."""

    source: str
    sink: str
    data: str
    protocol: str = "HTTPS"
    crosses_boundary: str | None = None
    authenticated: bool = False
    encrypted: bool = True
    carries_pii: bool = False
    ml_specific: bool = False  # flows that touch model artifacts / inference I/O


@dataclass
class System:
    """The full ML inference service model: elements, boundaries and data flows."""

    name: str
    elements: list[Element] = field(default_factory=list)
    boundaries: list[Boundary] = field(default_factory=list)
    flows: list[DataFlow] = field(default_factory=list)


def build_ml_inference_service() -> System:
    """Return the reference ML inference service used throughout this project."""
    sys_ = System(name="ML Inference Service")

    sys_.elements = [
        Element("Client App", "actor", "Untrusted external caller submitting prediction requests."),
        Element("API Gateway", "process", "TLS termination, authN/Z, rate limiting, request routing."),
        Element("Inference Service", "process", "Loads the model, runs preprocessing + forward pass."),
        Element("Model Registry", "datastore", "Stores versioned, signed model artifacts."),
        Element("Feature Store", "datastore", "Online features joined into requests at inference time."),
        Element("Prediction Log", "datastore", "Append-only log of inputs, outputs and metadata."),
        Element("MLOps Operator", "actor", "Internal user who promotes/deploys models."),
    ]

    sys_.boundaries = [
        Boundary("Internet / DMZ", "Public network -> gateway: fully untrusted input."),
        Boundary("Service Mesh", "Gateway -> internal services: authenticated but lateral-movement risk."),
        Boundary("Data Tier", "Services -> datastores: confidential model + feature data."),
    ]

    sys_.flows = [
        DataFlow("Client App", "API Gateway", "prediction request + auth token",
                 crosses_boundary="Internet / DMZ", authenticated=True, carries_pii=True),
        DataFlow("API Gateway", "Inference Service", "validated request",
                 crosses_boundary="Service Mesh", authenticated=True, carries_pii=True,
                 ml_specific=True),
        DataFlow("Inference Service", "Feature Store", "feature lookup keys",
                 crosses_boundary="Data Tier", authenticated=True),
        DataFlow("Feature Store", "Inference Service", "online feature vector",
                 crosses_boundary="Data Tier", authenticated=True, carries_pii=True),
        DataFlow("Model Registry", "Inference Service", "signed model artifact",
                 crosses_boundary="Data Tier", authenticated=True, ml_specific=True),
        DataFlow("Inference Service", "API Gateway", "prediction + confidence",
                 crosses_boundary="Service Mesh", authenticated=True, ml_specific=True),
        DataFlow("API Gateway", "Client App", "prediction response",
                 crosses_boundary="Internet / DMZ", authenticated=True, carries_pii=True),
        DataFlow("Inference Service", "Prediction Log", "inputs + outputs + request id",
                 crosses_boundary="Data Tier", authenticated=True, carries_pii=True),
        DataFlow("MLOps Operator", "Model Registry", "new model version (deploy)",
                 crosses_boundary="Service Mesh", authenticated=True, ml_specific=True),
    ]
    return sys_
