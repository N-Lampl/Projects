"""Benchmark harness: run each attack over a loader, measure the metrics that
matter for an attack comparison table.

For every attack we report, over images the model originally classified correctly:
  success_rate -- fraction pushed to a wrong prediction
  mean_l2      -- mean L2 norm of the perturbation (over successful samples)
  mean_linf    -- mean L-infinity norm of the perturbation (successful samples)
  runtime_s    -- wall-clock seconds for the whole attack
"""

from __future__ import annotations

import time
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def _norms(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = delta.flatten(1)
    return flat.norm(dim=1), flat.abs().max(dim=1).values


@torch.no_grad()
def _predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).argmax(1)


def run_attack(
    model: nn.Module,
    loader: DataLoader,
    attack_fn: Callable[..., torch.Tensor],
    *,
    device: torch.device | None = None,
    **attack_kwargs,
) -> dict:
    """Run one attack over the loader and return its metrics dict."""
    device = device or torch.device("cpu")
    model.to(device).eval()

    n_correct = 0
    n_success = 0
    l2_sum = 0.0
    linf_sum = 0.0
    t0 = time.perf_counter()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        clean_pred = _predict(model, x)
        correct = clean_pred == y
        if not correct.any():
            continue
        xc, yc = x[correct], y[correct]

        x_adv = attack_fn(model, xc, yc, **attack_kwargs)
        adv_pred = _predict(model, x_adv)
        flipped = adv_pred != yc

        delta = (x_adv - xc)[flipped]
        if delta.shape[0] > 0:
            l2, linf = _norms(delta)
            l2_sum += float(l2.sum())
            linf_sum += float(linf.sum())
        n_correct += int(correct.sum())
        n_success += int(flipped.sum())

    runtime = time.perf_counter() - t0
    success_rate = n_success / max(n_correct, 1)
    return {
        "n_correct": n_correct,
        "n_success": n_success,
        "success_rate": success_rate,
        "mean_l2": (l2_sum / n_success) if n_success else None,
        "mean_linf": (linf_sum / n_success) if n_success else None,
        "runtime_s": runtime,
    }
