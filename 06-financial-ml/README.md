# 06 · Financial Crime & Risk ML

ML pointed at **financial crime and risk** — fraud, money-laundering, market manipulation, and credit
risk. It's the same "ML for security" mindset as the detection track, applied to money: heavy class
imbalance, anomaly detection, adversaries who adapt, and metrics that actually matter to a risk team
(PR-AUC, precision@k, calibration) rather than raw accuracy.

⚠️ Authorized use only — see [../ETHICS.md](../ETHICS.md). All data is **synthetic** by default
(generated in-repo); real public datasets are documented per project and never committed.

## Projects

| Project | Build | Status |
|---|---|---|
| `p1-fraud-detection/` | Supervised credit-card fraud on imbalanced data; PR-AUC, precision@k, threshold tuning | ✅ |
| `p2-transaction-anomaly/` | Unsupervised anomaly detection (IsolationForest + autoencoder) on a transaction stream | ✅ |
| `p3-aml-typologies/` | Money-laundering detection on a transaction **graph**: structuring/smurfing + layering | ✅ |
| `p4-credit-risk-scoring/` | Default prediction with **calibration** (reliability, Brier) + a fairness gap check | ✅ |
| `p5-market-manipulation/` | Time-series anomaly detection: pump-and-dump / spoofing on synthetic OHLCV | ✅ |
| ★ `CAPSTONE-adversarial-fraud/` | Evade your **own** fraud model under feature-mutability constraints → harden → re-measure | ✅ |

## Notes

- **Imbalanced metrics, not accuracy.** Fraud/AML are needle-in-haystack — report PR-AUC, precision@k,
  recall at a fixed false-positive budget, and calibration. A 99.9%-accurate fraud model is useless.
- **Adversaries adapt.** The capstone treats fraud as adversarial (a fraudster mutates amount/timing
  to slip past the model), mirroring the adversarial-IDS capstone in the detection track.
- **CPU-friendly.** Classical ML (sklearn) by default; xgboost / a small torch autoencoder / networkx
  are optional enhancements imported lazily.

## Real datasets (documented per project; optional)

Credit-card fraud (ULB, Kaggle) · Give Me Some Credit / German Credit (credit risk) · IBM AML synthetic
transactions · LOBSTER / public tick data (market microstructure). Each project ships a synthetic
generator so it runs offline; swap in the real dataset to make the numbers benchmark-grade.
