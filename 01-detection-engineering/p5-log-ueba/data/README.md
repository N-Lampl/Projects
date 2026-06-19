# data/ (git-ignored)

The **default path generates its own data** -- a synthetic enterprise auth log -- so
this project runs fully offline with no download. Nothing here is committed.

## Synthetic generator (default)

`src/log_ueba/generate.py` produces ~15k auth events for ~60 users over 14 days, with
red-team-style lateral movement injected for 4 "compromised" users (credential fans out
across many never-before-seen hosts, off-hours, using service/remote logon types). The
schema mirrors a Windows security-log / LANL auth feed:

    timestamp, user, src_host, dst_host, logon_type, success, is_anomaly

Run `make detect` -- no dataset needed.

## Real-data path (optional, documented for parity)

The same feature/detector code runs on real auth logs. The two standard public sources:

- **LANL Comprehensive Cyber Security Events** (Kent, 2015) -- `auth.txt.gz`, ~58 days
  of real enterprise auth events with labelled red-team activity in `redteam.txt`.
  License: public, research use. Large (>10 GB) -- **do not download here**; stream it.
  Source: <https://csr.lanl.gov/data/cyber1/>
  ```bash
  # streaming filter so you never materialize the whole file (illustrative):
  zcat auth.txt.gz | awk -F, '{print $1","$2","$4","$5","$8","$9}' > auth_subset.csv
  ```
  Map columns to our schema (time,user,src_host,dst_host,logon_type,success), join
  `redteam.txt` on (time,user,src,dst) to set `is_anomaly`, then feed to
  `build_features` + `isolation_forest_scores` unchanged.

- **LogHub** (`https://github.com/logpai/loghub`) -- parsed system/auth logs (e.g. the
  Linux/OpenSSH sets). License: per-dataset (research). Use Drain/Spell to template the
  lines, then derive the same per-user features.

Because the feature builder is **streaming and causal** (baselines only ever use the
past), the exact same code is a real-time SIEM filter, not just a batch script.
