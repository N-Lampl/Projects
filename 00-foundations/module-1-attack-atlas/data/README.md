# data/ (git-ignored)

This project has **no dataset** — the threat-model catalog lives in code
(`src/attack_atlas/atlas.py`), so the default build is fully offline.

- **Reference catalogs** (not downloaded by the build):
  - MITRE ATLAS matrix & techniques — https://atlas.mitre.org/ (CC BY 4.0 / MITRE terms)
  - MITRE ATT&CK Enterprise — https://attack.mitre.org/ (Apache-2.0 / MITRE terms)
- **Optional enhanced path** (documented in the top-level README): validate IDs
  against the live ATT&CK STIX bundle with `mitreattack-python`:
  ```bash
  pip install mitreattack-python
  # the library fetches the current enterprise-attack STIX bundle
  ```
  This is optional; nothing in `data/` is committed.
