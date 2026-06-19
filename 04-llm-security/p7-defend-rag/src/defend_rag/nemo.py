"""OPTIONAL NeMo-Guardrails integration (NVIDIA-NeMo/Guardrails).

This is an enhanced, NOT-default path. NeMo-Guardrails (pkg `nemoguardrails`,
from github.com/NVIDIA-NeMo/Guardrails) lets you declare input/output rails in
Colang. We import it lazily so the default offline pipeline runs without it.

When `nemoguardrails` is unavailable we fall back to a tiny self-relient output
rail that reuses our own redaction logic, so `use_nemo=True` never crashes the
demo -- it simply degrades to the built-in guard.
"""

from __future__ import annotations

from dataclasses import dataclass

# A minimal Colang/YAML rails config kept inline for reference. A real deployment
# would load this from a `config/` directory via RailsConfig.from_path(...).
RAILS_CONFIG_YAML = """
models: []
rails:
  output:
    flows:
      - self check output
prompts:
  - task: self_check_output
    content: |
      Does the assistant message leak credentials, system instructions, or PII?
      Answer Yes or No.
"""


@dataclass
class _FallbackRails:
    """Used when nemoguardrails isn't installed: reuse our redaction guard."""

    def apply_output_rail(self, answer: str) -> str:
        from .guardrails import redact_secrets

        clean, _ = redact_secrets(answer)
        return clean


def nemo_available() -> bool:
    try:
        import nemoguardrails  # noqa: F401

        return True
    except Exception:
        return False


def maybe_build_rails():
    """Return a rails object exposing `apply_output_rail(answer) -> str`.

    Tries real NeMo-Guardrails; on any failure returns the fallback rail so the
    defended pipeline always has a working output rail.
    """
    if not nemo_available():
        return _FallbackRails()
    try:  # pragma: no cover - only runs if nemoguardrails is installed
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_content(yaml_content=RAILS_CONFIG_YAML)
        rails = LLMRails(config)

        class _NemoRails:
            def apply_output_rail(self, answer: str) -> str:
                # Run the answer through NeMo's output rails. If the rail blocks
                # it, fall back to our redaction so we still return something.
                from .guardrails import redact_secrets

                try:
                    result = rails.generate(messages=[{"role": "assistant", "content": answer}])
                    text = result.get("content", answer) if isinstance(result, dict) else answer
                except Exception:
                    text = answer
                clean, _ = redact_secrets(text)
                return clean

        return _NemoRails()
    except Exception:
        return _FallbackRails()
