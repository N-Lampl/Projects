# data/ (git-ignored)

This project ships its own **tiny in-repo corpus** in code
(`src/vulnerable_rag/corpus.py`) -- there is **no external dataset to download**.

- **Corpus:** ~9 short synthetic documents about a fictional "AcmeCloud" product.
- **License:** authored for this lab; public-domain-equivalent.
- **Download command:** none required. Just run `make run`.

Everything PII-shaped in the corpus (names, emails, SSNs, card numbers, the
`sk-LAB-FAKE-...` API key) is **invented and fake**, planted on purpose so the
attack projects (p5/p6) have concrete targets. It is a target range, not a leak.

This `data/` folder exists only for optional larger corpora you might add later;
anything dropped here is git-ignored and never committed.
