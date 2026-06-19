"""api-threat-model: a hardened model-serving endpoint with security controls.

Public API:
    set_seed, get_device            -- reproducibility helpers
    train_model, ServedModel        -- the tiny sklearn model being served
    ApiKeyStore                     -- API-key auth (salted-hash, constant-time)
    TokenBucketLimiter              -- per-principal rate limiting (DoS / scraping)
    validate_predict_input          -- strict input validation
    AuditLog, AuditEvent            -- security event logging (non-repudiation)
    SecurityControls                -- bundle of the above
    PredictionService               -- framework-agnostic request handler
    create_app                      -- optional FastAPI transport (lazy import)
    make_handler, serve             -- default stdlib HTTP transport
"""

from .app import create_app
from .model import ServedModel, make_synthetic_dataset, train_model
from .security import (
    ApiKeyStore,
    AuditEvent,
    AuditLog,
    SecurityControls,
    TokenBucketLimiter,
    ValidationError,
    validate_predict_input,
)
from .service import PredictionService, Response
from .stdlib_server import make_handler, serve
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "ServedModel",
    "make_synthetic_dataset",
    "train_model",
    "ApiKeyStore",
    "TokenBucketLimiter",
    "validate_predict_input",
    "ValidationError",
    "AuditLog",
    "AuditEvent",
    "SecurityControls",
    "PredictionService",
    "Response",
    "create_app",
    "make_handler",
    "serve",
]
