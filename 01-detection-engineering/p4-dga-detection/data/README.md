# data/ (git-ignored)

This project needs **no external dataset** — it is fully self-contained.

- **Dataset:** synthetic, generated in-process by `src/dga_detection/data.py`.
  - *Benign* domains are pronounceable: dictionary words and consonant-vowel
    syllables (e.g. `cloudshop.com`, `toveranu.io`).
  - *DGA* domains imitate three malware families: uniform-random alnum
    (Cryptolocker-style), long hex strings (Necurs-style), and concatenated
    dictionary words with no separators (Matsnu-style).
- **License:** none required — the data is produced at runtime and never written
  to disk or committed.
- **Download / generate:** nothing to download. Just run:

  ```bash
  make detect
  ```

If you later want to validate against *real* DGA feeds, the public
[Bambenek OSINT DGA feeds](https://osint.bambenek.com/feeds/) and the benign
[Tranco list](https://tranco-list.eu/) are the usual references — both are an
optional, manual step and are NOT used by the default path.
