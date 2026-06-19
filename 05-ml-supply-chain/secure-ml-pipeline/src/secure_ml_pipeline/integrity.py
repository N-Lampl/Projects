"""Optional external integrity tools (modelscan, Sigstore cosign) with graceful,
logged fallbacks. Every wrapper returns a structured dict and NEVER raises if the
tool is absent — it returns {"available": False, "skipped": True, ...} so the
pipeline keeps running offline.

The offline fallback for signing is a local HMAC over the artifact's sha256
(`local_sign` / `local_verify`). It is NOT a substitute for Sigstore's
keyless, transparency-logged signatures — it only demonstrates the
"fail closed on tampered/unsigned artifact" gate without network or cosign.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("secure_ml_pipeline.integrity")

# Demo-only HMAC key. In a real pipeline this would be a Sigstore keyless
# identity (OIDC) — never a hardcoded secret. Kept here purely so the offline
# "verify" gate is self-contained.
_DEMO_HMAC_KEY = b"secure-ml-pipeline-demo-key-not-a-real-secret"


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


# --------------------------------------------------------------------------- #
# modelscan (protectai) — optional static scanner for model files
# --------------------------------------------------------------------------- #
def run_modelscan(path: str | Path) -> dict:
    """Invoke `modelscan -p <path>` if installed; else skip with a warning."""
    if not have_tool("modelscan"):
        log.warning("modelscan not installed - SKIPPING (offline opcode scanner still runs)")
        return {"available": False, "skipped": True, "tool": "modelscan"}
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["modelscan", "-p", str(path), "--reporting-format", "json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        report = None
        if proc.stdout.strip():
            try:
                report = json.loads(proc.stdout)
            except json.JSONDecodeError:
                report = {"raw": proc.stdout[:4000]}
        # modelscan exits non-zero when issues are found.
        return {
            "available": True,
            "skipped": False,
            "tool": "modelscan",
            "exit_code": proc.returncode,
            "issues_found": proc.returncode != 0,
            "report": report,
        }
    except Exception as exc:  # pragma: no cover - depends on external tool
        log.warning("modelscan invocation failed: %s", exc)
        return {"available": True, "skipped": True, "tool": "modelscan", "error": str(exc)}


# --------------------------------------------------------------------------- #
# Sigstore cosign — optional keyless sign/verify
# --------------------------------------------------------------------------- #
def cosign_sign_blob(path: str | Path, sig_out: str | Path) -> dict:
    """Sign a blob with `cosign sign-blob` if installed; else skip."""
    if not have_tool("cosign"):
        log.warning("cosign not installed - SKIPPING Sigstore signing (local HMAC demo used instead)")
        return {"available": False, "skipped": True, "tool": "cosign"}
    try:  # pragma: no cover - requires cosign + network OIDC
        proc = subprocess.run(  # noqa: S603
            ["cosign", "sign-blob", "--yes", "--output-signature", str(sig_out), str(path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return {
            "available": True,
            "skipped": False,
            "tool": "cosign",
            "exit_code": proc.returncode,
            "signature": str(sig_out),
        }
    except Exception as exc:
        log.warning("cosign signing failed: %s", exc)
        return {"available": True, "skipped": True, "tool": "cosign", "error": str(exc)}


# --------------------------------------------------------------------------- #
# Offline fallback signing: HMAC over the artifact sha256
# --------------------------------------------------------------------------- #
def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def local_sign(path: str | Path, sig_out: str | Path) -> dict:
    """Write a detached HMAC 'signature' over the artifact's content hash."""
    digest = _sha256(path)
    sig = hmac.new(_DEMO_HMAC_KEY, digest.encode(), hashlib.sha256).hexdigest()
    record = {"scheme": "hmac-sha256-demo", "sha256": digest, "signature": sig}
    Path(sig_out).write_text(json.dumps(record, indent=2) + "\n")
    return record


def local_verify(path: str | Path, sig_in: str | Path) -> bool:
    """Verify the artifact matches its signature. Fails closed on tamper."""
    if not Path(sig_in).exists():
        log.error("no signature file at %s - FAIL CLOSED (treat as unsigned)", sig_in)
        return False
    try:
        record = json.loads(Path(sig_in).read_text())
    except Exception as exc:
        log.error("unreadable signature: %s - FAIL CLOSED", exc)
        return False
    current = _sha256(path)
    expected_sig = hmac.new(_DEMO_HMAC_KEY, current.encode(), hashlib.sha256).hexdigest()
    if current != record.get("sha256"):
        log.error("artifact sha256 changed (TAMPERED) - FAIL CLOSED")
        return False
    if not hmac.compare_digest(expected_sig, record.get("signature", "")):
        log.error("signature mismatch - FAIL CLOSED")
        return False
    return True
