"""Feature squeezing — the signal the runtime detector is built on.

Idea (Xu, Evans, Qi 2018, "Feature Squeezing: Detecting Adversarial Examples in
Deep Neural Networks", NDSS): adversarial perturbations live in the high-precision,
high-frequency corners of input space that don't matter for clean classification.
If you "squeeze" the input to a coarser representation and the model's prediction
*changes a lot*, the input was probably adversarial.

Two classic squeezers, both implemented with plain torch (no extra deps):

  1. bit-depth reduction  -- round each pixel to `bits` bits of precision.
  2. median blur (k x k)  -- replace each pixel by the median of its neighborhood,
                             which erases isolated high-frequency FGSM speckle.

The detection score for an input is the L1 distance between the model's softmax
on the original vs. the squeezed version (max over squeezers). Large score =>
fragile under squeezing => suspicious.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def bit_depth_reduce(x: torch.Tensor, bits: int = 2) -> torch.Tensor:
    """Reduce pixels in [0,1] to `bits` bits of precision (e.g. bits=1 -> {0,1})."""
    levels = float(2**bits - 1)
    return torch.round(x * levels) / levels


def median_blur(x: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """k x k median filter over each image (x: [N,1,H,W]), reflect-padded."""
    pad = kernel // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = xp.unfold(2, kernel, 1).unfold(3, kernel, 1)  # [N,1,H,W,k,k]
    patches = patches.contiguous().view(*patches.shape[:4], -1)
    return patches.median(dim=-1).values


@torch.no_grad()
def _softmax(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(model(x), dim=1)


@torch.no_grad()
def squeeze_scores(
    model: nn.Module,
    x: torch.Tensor,
    *,
    bits: int = 2,
    kernel: int = 3,
) -> torch.Tensor:
    """Per-example feature-squeezing scores -> tensor [N, 3].

    Columns:
      0: L1(softmax(x), softmax(bit_depth_reduce(x)))
      1: L1(softmax(x), softmax(median_blur(x)))
      2: max(col0, col1)   -- the canonical Feature-Squeezing detector statistic.
    """
    model.eval()
    p = _softmax(model, x)
    p_bit = _softmax(model, bit_depth_reduce(x, bits))
    p_med = _softmax(model, median_blur(x, kernel))
    d_bit = (p - p_bit).abs().sum(dim=1)
    d_med = (p - p_med).abs().sum(dim=1)
    d_max = torch.maximum(d_bit, d_med)
    return torch.stack([d_bit, d_med, d_max], dim=1)


@torch.no_grad()
def statistical_features(x: torch.Tensor) -> torch.Tensor:
    """Cheap input-only statistics that also separate clean vs FGSM -> [N, 4].

    FGSM adds a near-uniform +-epsilon speckle, which inflates total variation and
    the count of mid-gray pixels relative to a clean (mostly black/white) digit.
      0: mean absolute total variation (high-frequency energy)
      1: std of pixel values
      2: fraction of "mid-gray" pixels in (0.1, 0.9)
      3: mean pixel value
    """
    n = x.shape[0]
    flat = x.view(n, -1)
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv = tv_h + tv_w
    std = flat.std(dim=1)
    midgray = ((flat > 0.1) & (flat < 0.9)).float().mean(dim=1)
    mean = flat.mean(dim=1)
    return torch.stack([tv, std, midgray, mean], dim=1)


@torch.no_grad()
def detector_features(
    model: nn.Module, x: torch.Tensor, *, bits: int = 2, kernel: int = 3
) -> torch.Tensor:
    """Full feature vector fed to the sklearn detector: squeeze + statistical -> [N, 7]."""
    sq = squeeze_scores(model, x, bits=bits, kernel=kernel)
    st = statistical_features(x)
    return torch.cat([sq, st], dim=1)


FEATURE_NAMES = [
    "squeeze_bitdepth_L1",
    "squeeze_median_L1",
    "squeeze_max_L1",
    "stat_total_variation",
    "stat_pixel_std",
    "stat_midgray_frac",
    "stat_pixel_mean",
]
