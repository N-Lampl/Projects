"""Safe serialization: train a model and round-trip it via safetensors.

safetensors is a *data-only* format (a JSON header + raw tensor bytes). Unlike
pickle, loading a .safetensors file CANNOT execute arbitrary code, because the
format has no opcode for "call this function". That single property removes the
entire pickle-deserialization attack surface. This module is the "FIX" side of
the project: produce the artifact a secure pipeline should actually ship.

If `safetensors` is not installed we fall back to a tiny pure-Python writer that
emits the *identical* file layout (8-byte little-endian header length, JSON
header, raw bytes), so the default offline path still runs.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

# safetensors ships in the default env here, but import lazily so the module
# imports even if it is missing - the pure-python fallback covers that case.
try:  # optional, preferred
    from safetensors.torch import load_file as _st_load
    from safetensors.torch import save_file as _st_save

    _HAVE_SAFETENSORS = True
except Exception:  # pragma: no cover - exercised only when lib absent
    _HAVE_SAFETENSORS = False


_DTYPE_MAP = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
}
_NP_MAP = {"F32": np.float32, "F64": np.float64, "I64": np.int64, "I32": np.int32}


def _fallback_save(state: dict[str, torch.Tensor], path: Path) -> None:
    """Pure-python safetensors writer (used only if the lib is unavailable)."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, tensor in state.items():
        t = tensor.contiguous().cpu()
        raw = t.numpy().tobytes()
        header[name] = {
            "dtype": _DTYPE_MAP[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _fallback_load(path: Path) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
        body = f.read()
    out: dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        start, end = meta["data_offsets"]
        arr = np.frombuffer(body[start:end], dtype=_NP_MAP[meta["dtype"]])
        out[name] = torch.from_numpy(arr.reshape(meta["shape"]).copy())
    return out


def save_safetensors(model: torch.nn.Module, path: str | Path) -> Path:
    """Serialize model weights to a .safetensors file. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    if _HAVE_SAFETENSORS:
        _st_save(state, str(path))
    else:
        _fallback_save(state, path)
    return path


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load weights from a .safetensors file (no code execution possible)."""
    path = Path(path)
    if _HAVE_SAFETENSORS:
        return _st_load(str(path))
    return _fallback_load(path)


def sha256_file(path: str | Path) -> str:
    """Content hash - the basis for tamper detection / signing."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
