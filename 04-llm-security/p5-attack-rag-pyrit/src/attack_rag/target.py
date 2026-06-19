"""Adapter that loads the authorized target: ../p4-vulnerable-rag.

The attacks in this project run against the *deliberately-vulnerable* RAG lab
built in p4. We import it lazily by path so this package still imports if p4 is
absent, and so a future reorg of p4 raises a clear, single error.

Authorized use only: the target is a self-built lab app on synthetic data with a
mock LLM. See ../../ETHICS.md.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# ../p4-vulnerable-rag/src on the path so `import vulnerable_rag` resolves.
_P4_SRC = Path(__file__).resolve().parents[3] / "p4-vulnerable-rag" / "src"


def _ensure_target_on_path() -> None:
    if _P4_SRC.is_dir() and str(_P4_SRC) not in sys.path:
        sys.path.insert(0, str(_P4_SRC))


@lru_cache(maxsize=1)
def load_target():
    """Return the p4 `vulnerable_rag` module, or raise a clear error."""
    _ensure_target_on_path()
    try:
        import vulnerable_rag  # type: ignore
    except ImportError as exc:  # pragma: no cover - only if p4 is missing
        raise ImportError(
            "Could not import the target p4-vulnerable-rag. Expected it at "
            f"{_P4_SRC}. Build p4 first; this project attacks it."
        ) from exc
    return vulnerable_rag


def build_target(k: int = 3):
    """Construct a fresh VulnerableRAG instance over the *clean* p4 corpus."""
    vr = load_target()
    return vr.VulnerableRAG(documents=vr.build_corpus(), k=k)


def clean_corpus() -> list:
    """Return a copy of the target's pristine knowledge base."""
    vr = load_target()
    return list(vr.build_corpus())


def make_document(doc_id: str, title: str, text: str, sensitivity: str = "public"):
    """Build a p4 Document (the attacker's poisoned doc uses the same type)."""
    vr = load_target()
    return vr.Document(doc_id=doc_id, title=title, text=text, sensitivity=sensitivity)


def planted_api_key() -> str:
    """The fake secret planted in the target corpus (for success checks)."""
    return load_target().PLANTED_API_KEY
