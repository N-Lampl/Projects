"""Perplexity-based memorization detection: the *canary exposure* metric.

Method (Carlini, Liu, Erlingsson, Kos, Song — "The Secret Sharer", USENIX 2019):

A canary is a phrase with a high-entropy secret slot, e.g.
    "user alice secret code is ________"  (the blank is 10 random digits).

For a trained model we compute the per-canary *log-perplexity* (negative average
log-likelihood) of completing the secret slot. The randomness space R has |R|
possible secrets. If the model had NOT memorized, the true secret's rank among all
|R| candidates (ordered by perplexity, lowest = most likely) would be uniform.
Memorization makes the true secret far more likely than the alternatives.

We summarize this with **exposure**:

    exposure(s) = log2(|R|)  -  log2(rank of s among all candidates)

- rank = 1 (the single most-likely secret of all 10**10) -> exposure = log2(|R|),
  the maximum. The model has essentially leaked the secret.
- rank ~ |R|/2 (no preference) -> exposure ~ 1, i.e. baseline / no memorization.

Estimating the exact rank over 10**10 candidates is infeasible, so we use the
paper's distributional estimate: fit a (skew-)normal to the perplexities of a sample
of random candidates and read off where the real secret falls in that distribution
(Section 4 of the paper). We implement the normal-CDF version with scikit-learn /
scipy-free math (erf from torch) so it runs on the default stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .corpus import (
    SECRET_ALPHABET,
    SECRET_LEN,
    Canary,
    encode,
    make_canary,
)
from .utils import get_device

# Size of the secret randomness space R = |alphabet| ** length.
RANDOMNESS_SPACE = len(SECRET_ALPHABET) ** SECRET_LEN
LOG2_R = SECRET_LEN * math.log2(len(SECRET_ALPHABET))


def sequence_log_perplexity(
    model: nn.Module, text: str, device: torch.device | None = None
) -> float:
    """Negative mean log-likelihood (natural log) of `text` under the model.

    Lower = the model finds the text more probable. Computed teacher-forced over the
    full string; this is the standard char-LM perplexity (in log space).
    """
    device = device or get_device()
    model.eval()
    idxs = encode(text)
    if len(idxs) < 2:
        return float("inf")
    x = torch.tensor(idxs[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(idxs[1:], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (1, T)
    return float(-token_ll.mean().item())


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class ExposureResult:
    name: str
    secret: str
    real_perplexity: float
    mean_random_perplexity: float
    std_random_perplexity: float
    estimated_rank: float
    exposure: float


def estimate_exposure(
    model: nn.Module,
    canary: Canary,
    n_samples: int = 2000,
    seed: int = 0,
    device: torch.device | None = None,
) -> ExposureResult:
    """Estimate exposure for one canary via the Gaussian rank estimate.

    We sample `n_samples` random secrets, measure their perplexities, fit a normal,
    and estimate the real secret's rank = |R| * P(perp < real_perp).
    """
    device = device or get_device()
    rng = np.random.default_rng(seed)

    real_text = canary.text
    real_perp = sequence_log_perplexity(model, real_text, device)

    perps = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        secret = "".join(rng.choice(list(SECRET_ALPHABET), size=SECRET_LEN))
        text = make_canary(canary.name, secret)
        perps[i] = sequence_log_perplexity(model, text, device)

    mean, std = float(perps.mean()), float(perps.std() + 1e-12)
    # Fraction of the full space expected to be MORE likely (lower perp) than real.
    z = (real_perp - mean) / std
    frac_below = _normal_cdf(z)
    # rank in [1, |R|]; clamp so log2(rank) is finite.
    rank = max(frac_below * RANDOMNESS_SPACE, 1.0)
    exposure = LOG2_R - math.log2(rank)

    return ExposureResult(
        name=canary.name,
        secret=canary.secret,
        real_perplexity=real_perp,
        mean_random_perplexity=mean,
        std_random_perplexity=std,
        estimated_rank=rank,
        exposure=exposure,
    )


def empirical_rank(
    model: nn.Module,
    canary: Canary,
    n_samples: int = 2000,
    seed: int = 0,
    device: torch.device | None = None,
) -> int:
    """Count how many of `n_samples` random secrets are at least as likely as the
    real one (a direct, distribution-free check that complements the Gaussian fit).
    """
    device = device or get_device()
    rng = np.random.default_rng(seed)
    real_perp = sequence_log_perplexity(model, canary.text, device)
    n_better = 1  # the real secret itself
    for _ in range(n_samples):
        secret = "".join(rng.choice(list(SECRET_ALPHABET), size=SECRET_LEN))
        text = make_canary(canary.name, secret)
        if sequence_log_perplexity(model, text, device) <= real_perp:
            n_better += 1
    return n_better
