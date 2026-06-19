"""Model inversion by gradient ascent (Fredrikson et al., CCS 2015).

Goal: given only *query access* to a trained classifier f, reconstruct an input
that is representative of a target class c -- recovering information about the
private training data the model memorised.

Method: start from a blank/noisy image x, then iteratively descend the loss

    L(x) = CrossEntropy(f(x), c) + lambda * || x ||^2

i.e. ascend the model's confidence in class c (with a small L2 prior to keep the
image from blowing up). This is the mirror image of FGSM: FGSM perturbs the input
to *break* a prediction; inversion optimises a *whole* input to *maximise* a
class score. Both use gradients w.r.t. the input pixels, not the weights.

    x <- clip( x - step * sign( d/dx [ L(x) ] ), 0, 1 )

We use the sign of the gradient (an L-inf style step) for stability on CPU, plus
optional Gaussian blur every few steps as an image prior -- a standard trick that
makes reconstructions far more recognisable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# fixed 3x3 Gaussian-ish blur kernel used as an image prior during inversion
_BLUR = torch.tensor(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32
)
_BLUR = (_BLUR / _BLUR.sum()).view(1, 1, 3, 3)


def _blur(x: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, _BLUR, padding=1)


def invert_class(
    model: nn.Module,
    target_class: int,
    *,
    steps: int = 300,
    step_size: float = 0.1,
    l2_lambda: float = 0.01,
    blur_every: int = 20,
    img_shape: tuple[int, int, int] = (1, 28, 28),
    device: torch.device | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Reconstruct a class-representative image for `target_class`.

    Returns (image in [0,1] with shape (1, *img_shape), final confidence in [0,1]).
    """
    device = device or torch.device("cpu")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    x = 0.5 + 0.05 * torch.randn(1, *img_shape, generator=gen, device=device)
    x = x.clamp(0.0, 1.0).requires_grad_(True)

    target = torch.tensor([target_class], device=device)
    opt = torch.optim.Adam([x], lr=step_size)

    for t in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, target) + l2_lambda * (x**2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
            if blur_every and (t + 1) % blur_every == 0:
                x.copy_(_blur(x).clamp_(0.0, 1.0))

    with torch.no_grad():
        conf = F.softmax(model(x), dim=1)[0, target_class].item()
    return x.detach(), conf


def invert_all_classes(
    model: nn.Module,
    n_classes: int = 10,
    *,
    device: torch.device | None = None,
    seed: int = 42,
    **kwargs,
) -> tuple[torch.Tensor, list[float]]:
    """Run inversion for every class. Returns (images (n_classes, 1, H, W), confidences)."""
    imgs, confs = [], []
    for c in range(n_classes):
        img, conf = invert_class(model, c, device=device, seed=seed + c, **kwargs)
        imgs.append(img)
        confs.append(conf)
    return torch.cat(imgs, dim=0), confs


def reconstruction_quality(recon: torch.Tensor, prototypes: torch.Tensor) -> dict[str, float]:
    """Measure how well reconstructions match the true class prototypes.

    Returns the mean correlation of each reconstruction with its own-class
    prototype, and the "top-1 match rate": fraction of reconstructions whose
    best-correlating prototype is the correct class (a leakage score in [0,1]).
    """
    n = recon.shape[0]
    r = recon.view(n, -1)
    p = prototypes.view(prototypes.shape[0], -1)
    r = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-8)
    p = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-8)
    corr = (r @ p.t()) / r.shape[1]  # (n, n_classes) correlation matrix
    own = corr.diag()
    top1 = (corr.argmax(1) == torch.arange(n)).float().mean().item()
    return {
        "mean_own_class_correlation": own.mean().item(),
        "top1_match_rate": top1,
    }
