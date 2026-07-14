# 03 · ML Privacy (extraction, inference & defenses)

Stealing models and recovering training data - and defending with differential privacy. Research-depth
signal for AI-security / trust-&-safety roles. **CPU-heaviest track → sequenced last; run heavy bits as
overnight batch.**

Authorized use only - see [../ETHICS.md](../ETHICS.md). Attacks target models you trained.

## Project

| Project | Build | Status |
|---|---|---|
| `p3-membership-inference/` | Shokri + **LiRA** (online, warm-started shadows); report **TPR@1%FPR + AUC** | |

## Notes

- LiRA at this scale measures **methodology / curve shape**, not paper-matching low-FPR numbers - be
  honest about that. Cross-check with Privacy Meter's built-in LiRA.
- Use small targets (MNIST/Fashion-MNIST CNN, or logistic/MLP on UCI Adult) to keep CPU time sane.
