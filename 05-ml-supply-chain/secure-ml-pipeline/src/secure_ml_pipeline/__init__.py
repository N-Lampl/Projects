"""secure-ml-pipeline: a supply-chain hardening demo for ML model artifacts.

THREAT  -> a benign pickle-deserialization PoC built at runtime (poc.py).
FIX     -> train + serialize as safetensors, statically scan, sign + verify,
           with a CI gate that fails closed on unsigned/tampered artifacts.

Public API:
    set_seed, get_device          -- reproducibility helpers
    TinyMLP, make_synthetic_data  -- the small model + offline synthetic data
    save_safetensors, load_safetensors, sha256_file  -- safe (de)serialization
    scan_pickle_file, scan_pickle_bytes, ScanResult  -- offline opcode scanner
    build_benign_poc, write_poc                       -- the runtime threat PoC
    run_modelscan, cosign_sign_blob,                  -- optional external tools
    local_sign, local_verify                          -- offline sign/verify gate
"""

from .integrity import (
    cosign_sign_blob,
    have_tool,
    local_sign,
    local_verify,
    run_modelscan,
)
from .model import TinyMLP, make_synthetic_data
from .pickle_scan import ScanResult, scan_pickle_bytes, scan_pickle_file
from .poc import build_benign_poc, marker_exists, write_poc
from .serialize import load_safetensors, save_safetensors, sha256_file
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "TinyMLP",
    "make_synthetic_data",
    "save_safetensors",
    "load_safetensors",
    "sha256_file",
    "scan_pickle_file",
    "scan_pickle_bytes",
    "ScanResult",
    "build_benign_poc",
    "write_poc",
    "marker_exists",
    "run_modelscan",
    "cosign_sign_blob",
    "have_tool",
    "local_sign",
    "local_verify",
]
