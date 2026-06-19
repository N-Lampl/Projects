"""Online LiRA (Likelihood Ratio Attack) -- Carlini et al., S&P 2022.

The question membership inference answers: *was this exact example in the target
model's training set?* LiRA answers it per-example with a calibrated hypothesis
test instead of a single global threshold.

For each query example z = (x, y):

  1. Train many SHADOW models on random subsets of a population pool. By design,
     z is IN roughly half of those subsets and OUT of the other half ("online"
     LiRA -- both worlds are estimated empirically, no analytic OUT model).
  2. For each shadow, record the logit-confidence phi (see model.logit_confidence)
     that the shadow assigns to z's true label, split into:
        IN  distribution  ~ N(mu_in,  s_in^2)
        OUT distribution  ~ N(mu_out, s_out^2)
  3. Score the TARGET model's confidence phi* on z by the likelihood ratio

        Lambda(z) = N(phi*; mu_in, s_in^2) / N(phi*; mu_out, s_out^2)

     A high ratio => phi* looks like an IN model => predict "member".

Aggregating Lambda over many z and sweeping the threshold gives an ROC; the
security-relevant number is TPR at a low FPR (e.g. 1%), because a real attacker
only cares about confident hits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .scipy_stub import norm_logpdf  # local stable Gaussian log-pdf (no scipy dep)


@dataclass
class ShadowSignals:
    """Per-example shadow confidences, masked by membership.

    `phi[i, s]` = logit-confidence shadow s gave example i.
    `member[i, s]` = True if example i was IN shadow s's training set.
    """

    phi: np.ndarray  # (n_examples, n_shadows) float64
    member: np.ndarray  # (n_examples, n_shadows) bool


def fit_in_out_gaussians(sig: ShadowSignals) -> tuple[np.ndarray, ...]:
    """Per-example mean/std of the IN and OUT shadow-confidence distributions.

    Uses a small variance floor so examples that happen to land in few shadows of
    one world still get a finite, well-behaved likelihood.
    """
    n_ex = sig.phi.shape[0]
    mu_in = np.zeros(n_ex)
    mu_out = np.zeros(n_ex)
    s_in = np.zeros(n_ex)
    s_out = np.zeros(n_ex)
    var_floor = 1e-3

    for i in range(n_ex):
        in_vals = sig.phi[i, sig.member[i]]
        out_vals = sig.phi[i, ~sig.member[i]]
        # global fallbacks when a world is empty for this example
        mu_in[i] = in_vals.mean() if in_vals.size else sig.phi[i].mean()
        mu_out[i] = out_vals.mean() if out_vals.size else sig.phi[i].mean()
        s_in[i] = max(in_vals.std(), 0.0) if in_vals.size > 1 else 1.0
        s_out[i] = max(out_vals.std(), 0.0) if out_vals.size > 1 else 1.0

    s_in = np.sqrt(s_in**2 + var_floor)
    s_out = np.sqrt(s_out**2 + var_floor)
    return mu_in, s_in, mu_out, s_out


def lira_scores(
    phi_target: np.ndarray,
    sig: ShadowSignals,
) -> np.ndarray:
    """Per-example LiRA membership score = log-likelihood ratio (IN vs OUT).

    Higher => more likely a training member. We return the log ratio (a monotone
    transform of Lambda) for numerical stability; thresholding it gives the ROC.
    """
    mu_in, s_in, mu_out, s_out = fit_in_out_gaussians(sig)
    ll_in = norm_logpdf(phi_target, mu_in, s_in)
    ll_out = norm_logpdf(phi_target, mu_out, s_out)
    return ll_in - ll_out


def roc_from_scores(
    scores: np.ndarray,
    is_member: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC (fpr, tpr, thresholds), sorted by descending score. Pure numpy."""
    order = np.argsort(-scores)
    s = scores[order]
    m = is_member[order].astype(np.int64)

    p = int(m.sum())
    n = int((1 - m).sum())
    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    tpr = tp / max(p, 1)
    fpr = fp / max(n, 1)
    # prepend the (0,0) origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thr = np.concatenate([[np.inf], s])
    return fpr, tpr, thr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC."""
    return float(np.trapz(tpr, fpr))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float = 0.01) -> float:
    """TPR at the largest operating point with fpr <= target_fpr.

    This low-FPR number is the headline security metric for MIA: it measures how
    many members an attacker can flag while almost never crying wolf.
    """
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())
