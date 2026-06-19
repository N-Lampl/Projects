"""Fast smoke tests (run in CI). One slow end-to-end test is marked @slow.

These encode the security INVARIANTS of the project:
  * the runtime PoC is detected by the opcode scanner (REDUCE/dangerous global)
  * a safetensors round-trip preserves weights and is NOT flagged
  * the integrity gate fails closed on tampered + unsigned artifacts
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from secure_ml_pipeline import (
    TinyMLP,
    build_benign_poc,
    load_safetensors,
    local_sign,
    local_verify,
    save_safetensors,
    scan_pickle_bytes,
    set_seed,
    sha256_file,
)
from secure_ml_pipeline.pickle_scan import DANGEROUS_OPCODES


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_opcode_scanner_flags_benign_poc():
    """The runtime PoC must be detected as MALICIOUS (it uses REDUCE)."""
    data = build_benign_poc("/tmp/PWNED_DEMO")  # noqa: S108
    res = scan_pickle_bytes(data, path="<poc>")
    assert res.malicious is True
    assert res.verdict == "MALICIOUS"
    assert any(f.opcode in DANGEROUS_OPCODES for f in res.findings)


def test_opcode_scanner_passes_clean_pickle():
    """A plain data pickle with no callables/globals must scan clean."""
    import pickle

    data = pickle.dumps({"a": 1, "b": [1, 2, 3], "c": "hello"})
    res = scan_pickle_bytes(data)
    assert res.malicious is False
    assert res.verdict == "clean"


def test_safetensors_roundtrip_preserves_weights():
    set_seed(0)
    model = TinyMLP(in_features=8)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.safetensors"
        save_safetensors(model, path)
        loaded = load_safetensors(path)
        for k, v in model.state_dict().items():
            assert torch.allclose(loaded[k], v)
        # The safetensors file is not a pickle => opcode scan finds nothing risky.
        res = scan_pickle_bytes(path.read_bytes(), path=str(path))
        assert res.malicious is False


def test_integrity_gate_fails_closed():
    set_seed(0)
    model = TinyMLP(in_features=8)
    with tempfile.TemporaryDirectory() as td:
        artifact = Path(td) / "m.safetensors"
        sig = Path(td) / "m.safetensors.sig"
        save_safetensors(model, artifact)
        local_sign(artifact, sig)

        assert local_verify(artifact, sig) is True  # genuine -> pass

        original = artifact.read_bytes()
        flipped = bytearray(original)
        flipped[-1] ^= 0x01
        artifact.write_bytes(bytes(flipped))
        assert local_verify(artifact, sig) is False  # tampered -> blocked
        artifact.write_bytes(original)

        assert local_verify(artifact, Path(td) / "missing.sig") is False  # unsigned -> blocked


def test_sha256_changes_on_tamper():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.bin"
        p.write_bytes(b"abc")
        h1 = sha256_file(p)
        p.write_bytes(b"abd")
        assert sha256_file(p) != h1


@pytest.mark.slow
def test_full_pipeline_end_to_end():
    """Train, serialize, scan PoC vs safetensors, sign, verify, tamper-fail."""
    import torch as _torch

    from secure_ml_pipeline import make_synthetic_data

    set_seed()
    x, y = make_synthetic_data(n=400, in_features=10)
    model = TinyMLP(in_features=10)
    opt = _torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = _torch.nn.CrossEntropyLoss()
    for _ in range(50):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    with tempfile.TemporaryDirectory() as td:
        art = Path(td) / "model.safetensors"
        sig = Path(td) / "model.safetensors.sig"
        save_safetensors(model, art)
        local_sign(art, sig)

        # safetensors clean, PoC malicious
        st_res = scan_pickle_bytes(art.read_bytes())
        poc_res = scan_pickle_bytes(build_benign_poc("/tmp/PWNED_DEMO"))  # noqa: S108
        assert st_res.malicious is False
        assert poc_res.malicious is True

        # gate correctness
        assert local_verify(art, sig) is True
        bad = bytearray(art.read_bytes())
        bad[10] ^= 0xFF
        art.write_bytes(bytes(bad))
        assert local_verify(art, sig) is False
