# data/ (git-ignored)

The DEFAULT path needs **no data at all** — `generate_aml_graph(seed=42)` builds a
deterministic synthetic transaction graph in memory, with planted **structuring**
and **layering** typologies (and known ground-truth labels). Nothing in this folder
is committed.

## Optional: real IBM synthetic AML dataset (AMLSim / "AML World")

To benchmark on a published transaction-graph dataset, the standard public option is
the **IBM Transactions for Anti-Money Laundering** data (a.k.a. AMLSim / "AML World"),
released by IBM Research alongside the Multi-GNN AML work.

- **Dataset:** IBM Transactions for Anti-Money Laundering (synthetic, labeled with
  laundering typologies including fan-in/fan-out, cycles, and stacking).
- **License:** released by IBM for research (CDLA-Sharing / community use; check the
  Kaggle page terms before redistributing).
- **Source:** Kaggle — search "IBM Transactions for Anti Money Laundering (AML)".
  The smaller "LI-Small" / "HI-Small" splits are the most laptop-friendly; the full
  splits are multi-GB, so **do not** commit them here.

  ```bash
  # requires a configured Kaggle API token (~/.kaggle/kaggle.json)
  kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
    -f HI-Small_Trans.csv -p data/
  ```

The loader in this repo expects an edge table with `src, dst, amount, time`; adapting
the IBM CSV (account ids, timestamp, amount, is-laundering flag) to that shape is the
only wiring needed. The default synthetic path is what `make detect` runs.

## Why synthetic by default

Real AML data is sensitive and access-controlled; planted typologies give us exact
ground truth to measure precision@k and recall against. See
[../../../ETHICS.md](../../../ETHICS.md).
