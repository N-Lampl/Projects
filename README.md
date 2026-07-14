# ml-security-portfolio

> A data scientist's pivot into ML security: a CPU-only monorepo that **attacks _and_ defends** real
> ML systems - drift monitoring, adversarial evasion, model-privacy attacks, LLM interpretability &
> prompt-injection defense, and ML supply-chain security - every project self-contained, reproducible,
> and grounded in **MITRE ATLAS / ATT&CK** and the **OWASP LLM Top 10**.

**Authorized use only.** Every technique here runs against models, data, and apps I own or am
licensed to test. See **[ETHICS.md](ETHICS.md)**.

**▶ Live Playground - attack these models in your browser.** The [`dashboard/`](dashboard/)
(React/Vite) opens with two interactive demos that run the **real trained models entirely
client-side** (no backend): a prompt-injection detector you can try to sneak past, and a fraud model
you can watch a fraudster evade - before the adversarially-hardened version shuts the attack down.
Weights are exported from the Python projects and the scoring math is reproduced in JavaScript to
~1e-6 parity. Below the Playground: headline before/after results, every project's metrics +
figures, and a build-on-this roadmap. See **[docs/PLAYGROUND.md](docs/PLAYGROUND.md)** for the story;
`cd dashboard && npm install && npm run dev` to run locally. Deploys to GitHub Pages.

---

## Why this repo exists

I'm a data scientist moving into a security-focused role. ML is my strength; this portfolio builds
the security half - the attacker mindset, the frameworks, and the defensive engineering - by
*doing*. Each project ships a strong README, a reproducible run (`make attack`), a committed
"money plot" + `metrics.json`, and a short interview story. Everything is built to run on a
**CPU-only laptop** (small models, pretrained targets, classical ML, data subsets, and API/small-Ollama
for LLMs).

## Project index

**10 flagship projects across 9 tracks - all built, every one passing its fast test suite.** Each runs
**offline & deterministically** out of the box (synthetic data / mock LLM fallbacks); real datasets,
LLM API keys, or a GPU *enhance* specific projects but are never required to see them work. This is a
deliberately curated cut of a larger body of work - one strong, fully-owned piece per theme.

| Track | Project | What it demonstrates | Maps to |
|---|---|---|---|
| **01 detection** | [p7-drift-monitoring](01-detection-engineering/p7-drift-monitoring) | Monitoring a deployed detector for data/concept drift (PSI/KS) and alerting before it silently degrades | MLOps · ATT&CK |
| **02 adversarial** | [p1-fgsm-mnist](02-adversarial-robustness/p1-fgsm-mnist) | FGSM adversarial examples collapse a 99% MNIST CNN with an imperceptible perturbation | ATLAS AML.T0043 |
| **03 privacy** | [p3-membership-inference](03-ml-privacy/p3-membership-inference) | LiRA likelihood-ratio membership inference: was this record in the training set? | ATLAS AML.T0024 |
| **04 llm-security** | [p8-refusal-direction-interp](04-llm-security/p8-refusal-direction-interp) | **Abliteration** - refusal lives on one direction; locate it, ablate it, keep capability | interp · LLM safety |
| **05 supply-chain** | [secure-ml-pipeline](05-ml-supply-chain/secure-ml-pipeline) | pickle-RCE PoC → safetensors → ModelScan → Sigstore signing → CI gate | ATLAS AML.T0010 |
| **06 financial** | [CAPSTONE-adversarial-fraud](06-financial-ml/CAPSTONE-adversarial-fraud) | Evade my **own** fraud model under feature-mutability constraints → harden → re-measure | ATLAS AML.T0015 |
| **07 applied-nlp** | [p1-car-reviews](07-applied-nlp/p1-car-reviews) | HuggingFace sentiment over 36,984 car reviews **by brand & model**, validated vs. 1-5 star ratings | applied NLP · sentiment |
| **08 ml-depth** | [p3-graph-neural-networks](08-ml-depth/p3-graph-neural-networks) | A from-scratch, pure-PyTorch GCN scored vs. known ground truth | ML depth |
| **09 deep-learning** | [p1-transformer-interp](09-deep-learning/p1-transformer-interp) | Mechanistic interpretability: induction heads + logit lens + activation patching | interp · modern DL |
| **09 deep-learning** | [p3-model-compression](09-deep-learning/p3-model-compression) | Pruning / quantization / distillation Pareto - size vs. accuracy trade-offs | efficient DL |

The interactive **dashboard** (React/Vite) opens with two live in-browser demos: a **prompt-injection
detector** you can try to sneak past (real TF-IDF + LogisticRegression weights exported from the
LLM-security work) and the **adversarial-fraud** model you can watch a fraudster evade before the
hardened version shuts it down. The original 30-week learning roadmap lives in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Quickstart

```bash
# 1) (recommended) install uv - https://docs.astral.sh/uv/  - or just use system python3
make setup                      # ruff + pytest + pre-commit

# 2) run the seed project end-to-end (CPU, a couple of minutes)
cd 02-adversarial-robustness/p1-fgsm-mnist
make attack                     # trains a small MNIST CNN, runs FGSM, writes results/figures + metrics.json
```

Every project follows the **same skeleton** (`src/`, `scripts/`, `tests/`, `results/figures/`,
git-ignored `data/` + `models/`), so once you've read one you can read them all.

## Repo layout

```
01-detection-engineering/   # ML on security telemetry → drift monitoring in production
02-adversarial-robustness/  # evasion attacks (FGSM)
03-ml-privacy/              # training-data leakage (membership inference / LiRA)
04-llm-security/            # LLM interpretability (abliteration) + the prompt-injection detector demo
05-ml-supply-chain/         # pickle-RCE → safetensors → signing → CI gate
06-financial-ml/            # adversarial fraud: evade my own model, then harden it
07-applied-nlp/             # applied NLP / data science (the ML strength the security half builds on)
08-ml-depth/               # graph neural networks (from scratch)
09-deep-learning/          # transformer interpretability, model compression
docs/                       # ROADMAP.md, shared ATLAS/ATT&CK navigator layers, top-level plots
```

## Conventions

- **Reproducible:** every project pins deps and seeds RNGs (`set_seed(42)`); `make attack` regenerates
  the committed figure + `metrics.json`.
- **No data/weights in git:** datasets and model weights are downloaded by code (`src/data.py`,
  SHA-256 verified) and git-ignored. Only small figures + `metrics.json` are committed as evidence.
- **CI never trains:** GitHub Actions runs lint + fast smoke tests; heavy work is `@pytest.mark.slow`.
- **Ethics first:** see [ETHICS.md](ETHICS.md) and [SECURITY.md](SECURITY.md).

## License

Code: [MIT](LICENSE). Datasets and pretrained weights retain their own licenses (see each
`data/README.md`).
