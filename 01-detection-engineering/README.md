# 01 · Detection Engineering (ML for security)

Your data-science strength, pointed at security telemetry. **The highest-job-probability track** —
SOC / detection-engineer / security-data-scientist roles. Mostly classical ML → CPU-friendly.

Authorized use only — see [../ETHICS.md](../ETHICS.md). Datasets are public; downloaded by code.

## Project

| Project | Build | Status |
|---|---|---|
| `p7-drift-monitoring/` | Concept-drift / model monitoring **as a security control** | |

## Notes

- **Metrics that matter to a SOC:** precision@k, alert volume, time-to-detect — not just ROC-AUC,
  which is misleading under heavy class imbalance.
- Model monitoring reframed **as a security control** — drift as an early-warning signal that a
  detector is degrading or being evaded.
