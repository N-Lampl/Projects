# Models

**No model weights are trained or stored by this project.**

The capstone is an orchestration / CI layer: it drives the OFFLINE red-team
harnesses (`../p2-garak-scan`, `../p3-promptfoo-redteam`) against the lab RAG
targets (`../p4-vulnerable-rag` vulnerable, `../p7-defend-rag` remediated). The
default targets use a deterministic **mock LLM provider** (no real model), so
there is nothing to download or train here.

The optional enhanced paths (real garak / promptfoo against your OWN local model
via Ollama or an API key) are documented in the README and `requirements.txt`;
any model used there lives outside this repo.
