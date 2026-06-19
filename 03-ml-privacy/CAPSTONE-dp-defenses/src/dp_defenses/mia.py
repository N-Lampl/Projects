"""Online LiRA membership inference, reused against every DP target.

This is the same likelihood-ratio attack as track-03 p3 (Carlini et al., S&P
2022), condensed here so the capstone is self-contained. The key design choice
for the privacy-utility study: ONE shared shadow set is trained once (non-private,
on the population pool) and scored against ALL targets -- eps=inf, eps=3, eps=1.
That isolates the variable we care about: the only thing changing between rows of
the tradeoff table is the *target's* training procedure, not the attacker.

Per query example z = (x, y):
  * IN  shadows give confidences  ~ N(mu_in,  s_in^2)
  * OUT shadows give confidences  ~ N(mu_out, s_out^2)
  * score(z) = logN(phi_target; in) - logN(phi_target; out)   (log likelihood ratio)
Sweeping the threshold gives an ROC; the headline security number is TPR @ 1% FPR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scipy_stub import norm_logpdf


@dataclass
class ShadowSignals:
    """Per-example shadow confidences masked by membership.

    `phi[i, s]` = logit-confidence shadow s gave query example i.
    `member[i, s]` = True if example i was IN shadow s's training set.
    """

    phi: np.ndarray  # (n_examples, n_shadows) float64
    member: np.ndarray  # (n_examples, n_shadows) bool


def fit_in_out_gaussians(sig: ShadowSignals) -> tuple[np.ndarray, ...]:
    """Per-example mean/std of the IN and OUT shadow-confidence distributions."""
    n_ex = sig.phi.shape[0]
    mu_in = np.zeros(n_ex)
    mu_out = np.zeros(n_ex)
    s_in = np.zeros(n_ex)
    s_out = np.zeros(n_ex)
    var_floor = 1e-3

    for i in range(n_ex):
        in_vals = sig.phi[i, sig.member[i]]
        out_vals = sig.phi[i, ~sig.member[i]]
        mu_in[i] = in_vals.mean() if in_vals.size else sig.phi[i].mean()
        mu_out[i] = out_vals.mean() if out_vals.size else sig.phi[i].mean()
        s_in[i] = max(in_vals.std(), 0.0) if in_vals.size > 1 else 1.0
        s_out[i] = max(out_vals.std(), 0.0) if out_vals.size > 1 else 1.0

    s_in = np.sqrt(s_in**2 + var_floor)
    s_out = np.sqrt(s_out**2 + var_floor)
    return mu_in, s_in, mu_out, s_out


def lira_scores(phi_target: np.ndarray, sig: ShadowSignals) -> np.ndarray:
    """Per-example LiRA membership score = log likelihood ratio (IN vs OUT)."""
    mu_in, s_in, mu_out, s_out = fit_in_out_gaussians(sig)
    ll_in = norm_logpdf(phi_target, mu_in, s_in)
    ll_out = norm_logpdf(phi_target, mu_out, s_out)
    return ll_in - ll_out


def roc_from_scores(
    scores: np.ndarray, is_member: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC (fpr, tpr, thresholds), sorted by descending score. Pure numpy."""
    order = np.argsort(-scores)
    s = scores[order]
    m = is_member[order].astype(np.int64)
    p = int(m.sum())
    n = int((1 - m).sum())
    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    tpr = np.concatenate([[0.0], tp / max(p, 1)])
    fpr = np.concatenate([[0.0], fp / max(n, 1)])
    thr = np.concatenate([[np.inf], s])
    return fpr, tpr, thr


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC."""
    return float(np.trapz(tpr, fpr))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float = 0.01) -> float:
    """TPR at the largest operating point with fpr <= target_fpr.

    The low-FPR number is the headline MIA security metric: how many members an
    attacker flags while almost never crying wolf. DP should drive this toward the
    1% chance baseline.
    """
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())
