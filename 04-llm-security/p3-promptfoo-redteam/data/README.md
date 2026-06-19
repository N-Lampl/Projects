# data/ (git-ignored)

This project needs **no external dataset**. The "data" is:

- The **probe library** (injection / jailbreak / PII / system-prompt-leak prompts)
  defined in code at `src/promptfoo_redteam/probes.py`. All prompts are written
  by hand; none are downloaded.
- The **target knowledge base** lives in the sibling lab project
  `../p4-vulnerable-rag` (planted, fully synthetic PII/secret/injection bait — a
  target range, not real data). This project imports that target at runtime.

Nothing is committed here. There is no download step.
