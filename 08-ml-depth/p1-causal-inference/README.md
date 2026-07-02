# p1 · Causal inference — recover a known treatment effect on CPU

> **Synthetic-by-default, known answer.** Committed results come from a structural
> causal model whose true ATE is exactly `2.0`, so every estimator can be *scored*,
> not asserted. `make run` regenerates them; `make test` runs offline. A `@slow`
> test recovers the effect on the real IHDP benchmark.

The question causal inference answers isn't "are treatment and outcome correlated"
— it's "what would the outcome have been *had we intervened*". Here a treatment `T`
and an outcome `Y` share a common cause `X`, so the naive treated-minus-control
difference is **confounded**. This project estimates the average treatment effect
(ATE) four ways — naive, regression adjustment, inverse-propensity weighting, and
doubly-robust AIPW — and, because the data is simulated with a *known* effect,
shows exactly which methods recover it and how well their confidence intervals
behave. Everything runs on a **CPU-only laptop** with scikit-learn linear models.

**Authorized use only.** Synthetic data plus a public research benchmark, used for
education. See [../../ETHICS.md](../../ETHICS.md).

## The idea

The data-generating process ([`scm.py`](src/causal_inference/scm.py)) draws
confounders `X`, assigns treatment with `P(T=1 | X) = sigmoid(X·α)`, and generates
`Y = X·β + τ·T + ε`. Because `α` and `β` point the same way, treated units really
do differ in `X`, and that same `X` drives `Y` — a textbook confound. The true ATE
is the constant `τ = 2.0`.

Four estimators ([`estimators.py`](src/causal_inference/estimators.py)):

1. **Naive** — treated mean minus control mean. Ignores `X`; wrong on purpose.
2. **Regression adjustment (G-computation)** — fit outcome models `μ₀, μ₁` and
   average `μ₁(X) − μ₀(X)`. Unbiased *if the outcome model is right*.
3. **IPW** — reweight by the inverse estimated propensity. Unbiased *if the
   propensity model is right*.
4. **AIPW (doubly robust)** — combine both; consistent if *either* model is right,
   and its efficient influence function gives a standard error, hence a real
   confidence interval whose coverage we can measure.

Two things are then reported: point estimates on one dataset, and **CI coverage**
across 200 re-draws (how often each 95% interval actually contains the truth).

## Run it

```bash
make run     # SCM -> four estimates + coverage + balance -> figures + metrics.json
make test    # fast offline smoke tests (-m 'not slow')
make run ARGS='--tau 3 --confounding 2 --n-sims 500'   # crank up the confounding
```

Outputs land in [results/](results/):
- `figures/ate_by_method.png` — the **money plot**: each estimate vs the dashed true
  ATE, with the AIPW 95% interval drawn in.
- `figures/covariate_balance.png` — love plot: `|SMD|` per covariate before vs after
  IPW weighting.
- `metrics.json` — estimates, bias per method, AIPW SE/CI, coverage, balance.

## What the result shows

With the true ATE fixed at **2.00**:

| method | estimate | bias |
|---|---|---|
| naive (diff in means) | 6.29 | **+4.29** |
| regression adjustment | 2.04 | +0.04 |
| IPW | 2.20 | +0.20 |
| **AIPW (doubly robust)** | **2.00** | **−0.00** |

The naive difference is off by more than **200%** — it reads a `+4.29` confound on
top of the real `+2.00` effect. Every adjusted estimator collapses that bias;
doubly-robust AIPW nails it. The interval story is just as sharp: across 200
re-draws the **AIPW 95% interval covers the truth 94% of the time** (near its
nominal rate) while the naive interval — correctly sized but centred on a biased
estimate — covers **0%**. And the diagnostic that explains it: inverse-propensity
weighting cuts the mean absolute standardized mean difference from **0.47 to 0.03**,
i.e. the weighted treated/control groups become comparable in `X`. Adjusting for the
confounder is what turns correlation into a defensible causal estimate.

## Interview story (3 sentences)

> I built a confounded structural causal model with a known treatment effect and
> estimated it four ways — naive, regression adjustment, IPW, and doubly-robust
> AIPW — so the methods could be scored against ground truth rather than trusted.
> The naive difference was off by 200% (it read the confound as effect), while AIPW
> recovered the true ATE with a 95% interval that actually covered 94% of the time,
> and an inverse-propensity love plot showed *why* — the confounders went from
> `|SMD|` 0.47 to 0.03 after weighting. It shows I understand identification, the
> double-robustness property, and how to validate a causal claim with coverage and
> balance diagnostics instead of a single point estimate.

## Layout

```
src/causal_inference/  utils · scm (confounded data-generating process)
                       estimators (naive · regression · IPW · AIPW + SMD balance)
                       experiment (point estimates · coverage study · balance table)
                       data (real IHDP loader) · plots
scripts/               run_analysis.py  -> results/figures + metrics.json
tests/                 test_smoke.py  (offline SCM; @slow real-IHDP recovery)
results/               figures/*.png + metrics.json  (committed)
data/ models/          git-ignored (IHDP downloaded by code; no weights persisted)
```

## References

- Hill (2011), *Bayesian Nonparametric Modeling for Causal Inference* — the IHDP
  benchmark ([data/README.md](data/README.md)).
- Robins, Rotnitzky & Zhao (1994) — augmented IPW / double robustness.
- Chernozhukov et al. (2018), *Double/Debiased Machine Learning* — the influence-
  function estimator and its inference.
