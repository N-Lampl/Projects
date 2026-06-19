# data/ (git-ignored)

This project needs **no external dataset**. Its inputs are the Sigma rules in
[`../rules/`](../rules) (committed) and the embedded offline ATT&CK catalog in
`src/threat_informed/attack.py`.

- **"Dataset":** the Sigma rule files + a minimal embedded MITRE ATT&CK (Enterprise) slice.
- **Download:** none. `make detect` runs fully offline.
- **Optional enhanced path:** to validate against the *full* ATT&CK matrix you could fetch the
  official STIX bundle from <https://github.com/mitre-attack/attack-stix-data> (CC-BY 4.0) and load
  it instead of the embedded catalog. Keep any download under this git-ignored folder; never commit it.
