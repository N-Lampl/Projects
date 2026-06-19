# data/ (git-ignored)

This project needs **no external dataset to download**. The detector's training
data is a **synthetic prompt-injection corpus generated in-process** by
[`src/defend_rag/dataset.py`](../src/defend_rag/dataset.py).

- **Dataset:** ~1,200 balanced examples (injection / jailbreak vs. benign
  questions + knowledge-base sentences), built deterministically from small
  hand-written templates (seed 42).
- **License:** authored for this lab; public-domain-equivalent.
- **Download command:** none required. Just run `make defend`.

The structure mirrors public injection corpora (e.g. the deepset
`prompt-injections` set and Lakera's Gandalf data) without shipping them. To use
a real public set instead, drop it here and adapt `dataset.py` to read it; this
folder is git-ignored and never committed.

The attack targets (planted API key, synthetic SSNs/cards/emails, the leaky
system prompt) all come from the **../p4-vulnerable-rag** corpus and are fake
lab bait, not real data.
