# 02 · Adversarial Robustness (evasion & defenses)

Attacking and defending models at inference time — start with the **quick win**: implement the
gradient-sign attack by hand and watch it flip a 99%-accurate classifier.

Authorized use only — see [../ETHICS.md](../ETHICS.md). All targets are self-trained or public
pretrained weights.

## Project

| Project | Build | Status |
|---|---|---|
| **`p1-fgsm-mnist/`** | **SEED** — FGSM from scratch (~30 lines); accuracy-vs-ε curve | done |

## Start with `p1-fgsm-mnist`

It's the quick win: implement the gradient-sign attack by hand, see it flip a 99%-accurate classifier,
and prove the whole repo skeleton (seeds, Makefile, figures, metrics, CI) on a tiny CPU project before
the flagships depend on it. → [`p1-fgsm-mnist/`](p1-fgsm-mnist/)
