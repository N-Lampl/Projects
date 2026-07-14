# Fraud Model Robustness Report Card

**Project:** CAPSTONE-adversarial-fraud  ·  **Baseline:** logreg  ·  **Hardened:** gboost (3-round adv-training)  ·  **Data:** synthetic (seed 42)

## Overall grade: A

Logistic fraud model (PR-AUC=0.415). A feasibility-constrained greedy evasion (only mutable txn fields, in-bounds, consistency-preserving) achieved ASR=100% against the baseline; adversarial training cut ASR to 0% (grade A) while keeping clean PR-AUC at 0.645.

## Clean performance (no attacker)

| Metric | Baseline | Hardened |
|---|---|---|
| PR-AUC (primary) | 0.415 | 0.645 |
| ROC-AUC | 0.921 | 0.974 |
| Recall @ 5% FPR budget | 0.625 | 0.847 |
| Precision @ threshold | 0.342 | - |
| p@100 | 0.490 | - |

## Adversarial robustness (feasibility-constrained evasion)

| Metric | Value |
|---|---|
| Frauds attacked (baseline) | 90 |
| **Attack Success Rate - before** | **100.0%** |
| **Attack Success Rate - after hardening** | **0.0%** |
| ASR reduction | +100.0% |
| Evasion feasibility rate | 100.0% |
| Mean P(fraud) drop under attack | 0.160 |
| Old evasions still fooling hardened model | 13.3% |

## Threat model

- **Mutable** (attacker-controlled): amount, hour, merchant_risk, distance_from_home, n_items
- **Immutable** (server-side / account history): account_age_days, avg_amount_30d, txn_count_30d, home_country_risk, card_present
- **Constraints enforced:** per-feature plausibility bounds; integer fields rounded; amount kept >= 5% of the account's 30-day average (consistency). Feasibility rate above audits that every counted evasion obeys the contract.

## Figures

- `results/figures/robustness_before_after.png`
- `results/figures/score_shift.png`

*Synthetic data and self-trained models only. Authorized use only - see ../../ETHICS.md.*
