# Ethics & Authorized Use

This repository is a **defensive / research-oriented security learning portfolio**. Many of the
techniques here are *dual-use*: the same method that measures a model's weakness can be misused
against systems you don't own. This file is the hard rule every project in this repo follows.

## The rule

> **I only run these techniques against models, data, and applications that I trained myself, am
> licensed to use, or am explicitly authorized to test. Never against third-party or production
> systems I do not own.**

## Authorized scope (what's allowed here)

- Models I trained from scratch in this repo.
- Public, openly-licensed datasets and **open-weight** models, used within their licenses.
- LLM endpoints accessed with **my own API keys**, on my own account.
- Local, throwaway lab targets I stand up myself (e.g. Docker containers on `localhost`).

## Out of scope (never done here)

- Attacking any deployed system, API, or model I do not own or lack written authorization to test.
- Executing live malware. Malware work is **feature-based only** (e.g. EMBER feature vectors); no
  detonation of real samples.
- Committing weaponized artifacts. Proof-of-concept payloads (e.g. the pickle-deserialization PoC)
  are **generated at runtime** by a local script and only ever loaded inside an isolated container
  (`docker run --network none`), with a deliberately benign payload.
- Publishing or redistributing safety-stripped ("abliterated") copies of other parties' models. The
  refusal-direction project (`04-llm-security/p8`) ships an **interpretability analysis**, not a
  modified model.

## Responsible disclosure

If a project here ever surfaces a real vulnerability in third-party software, follow that project's
coordinated-disclosure process. See `SECURITY.md`.

## Frameworks this repo maps to

- **MITRE ATLAS** — adversarial threats against AI systems.
- **MITRE ATT&CK** — adversary tactics & techniques (for the detection-engineering track).
- **OWASP Top 10 for LLM Applications (2025)** — for the LLM-security track.
