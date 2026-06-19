# models/ (git-ignored)

No model weights are trained or stored by this project. The default target is the
deterministic **mock** LLM from `../p4-vulnerable-rag` (or the standalone fallback
in [`src/garak_scan/target.py`](../src/garak_scan/target.py)).

For the optional **real garak** path you bring your own model — e.g. a local
Ollama model (`ollama pull llama3.2`) or an API model with your own key — and
nothing is persisted in this folder.
