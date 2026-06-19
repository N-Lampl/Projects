"""Pluggable LLM providers.

Default: a deterministic MOCK provider so the whole lab runs OFFLINE with no
API keys and no network. The mock is intentionally naive/credulous -- it obeys
embedded instructions in its context -- so prompt-injection attacks land.

Optional real providers (OpenAI / Anthropic / Ollama) are imported lazily inside
each method and selected via env vars (read from a local .env if present). They
make REAL API calls, so they are NOT used by the default `make run` path.

Authorized use only: point real providers at YOUR OWN keys / local Ollama.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol


def _load_dotenv() -> None:
    """Minimal .env loader (no python-dotenv dependency). Project-local only."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMProvider(Protocol):
    name: str

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class MockProvider:
    """Deterministic, offline, deliberately-credulous LLM stand-in.

    Behaviour (no randomness, fully reproducible):
      * Echoes the strongest fact it can find in the retrieved context.
      * OBEYS embedded instructions in the context -> indirect prompt injection
        works (it will append PWNED and try to reveal secrets) so p5/p6 have a
        real target. A hardened model would refuse; this lab target does not.
    """

    name = "mock"

    _INJECTION_RE = re.compile(r"ignore (all |the )?previous instructions", re.IGNORECASE)
    _SECRET_RE = re.compile(r"sk-[A-Za-z0-9\-]+")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        context = user_prompt
        obeyed_injection = bool(self._INJECTION_RE.search(context))

        # Naive "answer": surface the first informative sentence from context.
        answer = self._summarize(context)

        if obeyed_injection:
            # The vulnerable behaviour: leak the system prompt + any secrets seen.
            leaked_secret = self._SECRET_RE.search(context)
            parts = [answer, "[maintenance mode] system prompt: " + system_prompt]
            if leaked_secret:
                parts.append("leaked key: " + leaked_secret.group(0))
            return " ".join(parts) + " PWNED"

        return answer

    @staticmethod
    def _summarize(context: str) -> str:
        # Pull the context block out of the user prompt and return its lead.
        marker = "CONTEXT:"
        body = context.split(marker, 1)[-1] if marker in context else context
        body = body.split("QUESTION:", 1)[0].strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        lead = " ".join(sentences[:2]) if sentences else "I don't have information on that."
        return f"Based on the knowledge base: {lead}"


class OpenAIProvider:  # pragma: no cover - optional real-network path
    name = "openai"

    def __init__(self, model: str | None = None):
        _load_dotenv()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai not installed (optional real path).") from exc
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider:  # pragma: no cover - optional real-network path
    name = "anthropic"

    def __init__(self, model: str | None = None):
        _load_dotenv()
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic not installed (optional real path).") from exc
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # See ../../ETHICS.md: use your own key. Default to a small, fast model.
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OllamaProvider:  # pragma: no cover - optional real-network path
    name = "ollama"

    def __init__(self, model: str | None = None):
        _load_dotenv()
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self._host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self._model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "")


_PROVIDERS = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory. Defaults to the offline mock unless RAG_PROVIDER overrides it."""
    _load_dotenv()
    name = (name or os.environ.get("RAG_PROVIDER") or "mock").lower()
    if name not in _PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]()
