# Data

**This project needs NO downloaded dataset.** The red-team operates on:

- the **synthetic, planted corpus** built by `../p4-vulnerable-rag` (fake support
  docs containing fake PII / a fake `sk-...` key / a planted injection), and
- a **fixed, hand-written probe set** reused from `../p2-garak-scan` and
  `../p3-promptfoo-redteam`.

Everything is generated in-memory at runtime; nothing is committed and there is
nothing to download. If `../p4-vulnerable-rag` is unavailable, the harness falls
back to a self-contained mock target with the same planted artifacts.

## License / authorization

All "secrets" and "PII" are fabricated lab values (e.g. `HUNTER2-LAB`,
`sk-LAB-FAKE-...`, `SSN 521-08-4417`). They are not real. Authorized-use-only:
the target is a self-built, deliberately vulnerable app. See
[../../../ETHICS.md](../../../ETHICS.md).
