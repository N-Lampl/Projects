# 06 · Financial Crime & Risk ML

ML pointed at **financial crime and risk** - fraud, money-laundering, market manipulation, and credit
risk. It's the same "ML for security" mindset as the detection track, applied to money: heavy class
imbalance, anomaly detection, adversaries who adapt, and metrics that actually matter to a risk team
(PR-AUC, precision@k, calibration) rather than raw accuracy.

Authorized use only - see [../ETHICS.md](../ETHICS.md). Datasets are NOT committed; they're
downloaded by code. The project also ships an offline synthetic generator as a fallback.

## Project

| Project | Real result | Data |
|---|---|---|
| `CAPSTONE-adversarial-fraud/` | Evasion ASR **100% → 0%** after adversarial training | Synthetic - *the feature-mutability attack needs interpretable features; real fraud's PCA features can't express "a fraudster changes the amount"* |

## Notes

- **Imbalanced metrics, not accuracy.** Fraud is needle-in-haystack - report PR-AUC, precision@k,
  recall at a fixed false-positive budget, and calibration. A 99.9%-accurate fraud model is useless.
- **Adversaries adapt.** The capstone treats fraud as adversarial (a fraudster mutates amount/timing
  to slip past the model): evade the baseline, harden with adversarial training, re-measure.
- **CPU-friendly.** Classical ML (sklearn) by default; xgboost / a small torch autoencoder are
  optional enhancements imported lazily.

The capstone falls back to its offline synthetic generator (`make run` / no flag) so tests and CI
need no network. Datasets are git-ignored, never committed.
