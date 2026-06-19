# Authorization (template)

> Reusable scope statement for the LLM-security track. Each attack project links here
> (and to the repo-wide [../../ETHICS.md](../../ETHICS.md)) so authorization is explicit.

**Engagement:** Personal security-research lab (educational portfolio).
**Authorized by:** The repository owner, against their own systems only.

## In scope
- The deliberately-vulnerable lab app in `04-llm-security/p4-vulnerable-rag` (self-built, synthetic data).
- LLM endpoints accessed with **my own API keys** on my own account.
- Local models I run myself (e.g. Ollama on `localhost`).

## Out of scope
- Any third-party, shared, or production LLM/app I do not own or lack **written** permission to test.
- Exfiltrating real personal data. All "secrets" and "PII" used here are synthetic and planted.
- Real outbound exfiltration. Exfil channels are **mocked** — nothing leaves the process.

## Rules of engagement
- Default runs are **offline + deterministic** (mock model). Real-model runs are opt-in via `.env`.
- Multi-turn attacks are **rate/turn-capped** (`MAX_TURNS`, `MAX_BRANCHES`) to bound cost & blast radius.
- Findings against real third-party software follow coordinated disclosure (see [../../SECURITY.md](../../SECURITY.md)).
