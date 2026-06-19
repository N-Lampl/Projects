# models/ (git-ignored)

This project trains **no model**. The "model under test" is the deliberately-
vulnerable RAG app in `../p4-vulnerable-rag`, which itself defaults to a
deterministic offline **mock** LLM provider (no weights, no API).

If you wire up a real provider (e.g. local Ollama) via p4's `RAG_PROVIDER` env
var, that is your own infrastructure — nothing is stored here. Not committed.
