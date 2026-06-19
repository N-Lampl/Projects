# data/ (git-ignored)

This project needs **no external dataset**. The "attack corpus" is the built-in
probe set in [`src/garak_scan/probes.py`](../src/garak_scan/probes.py) and the
target is the sibling [`../p4-vulnerable-rag`](../p4-vulnerable-rag) mock app (or
a self-contained fallback). Nothing is downloaded and nothing is committed here.

- **Probes:** small, hand-written, allowlisted in
  [`configs/probe_allowlist.yaml`](../configs/probe_allowlist.yaml); modelled on
  garak's probe taxonomy (promptinject / leakreplay / mitigation).
- **Optional real garak:** if you run real garak (see the README), it produces
  its own `report.jsonl` under garak's run dir — point `make parse REPORT=...`
  at it. That file is git-ignored.
