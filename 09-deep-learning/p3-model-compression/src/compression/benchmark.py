"""Measure the accuracy / size / latency / sparsity of a model variant."""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .compress import sparsity as _sparsity
from .data import ClassDataset


@dataclass
class VariantMetrics:
    """The four Pareto axes for one model variant."""

    accuracy: float
    size_mb: float
    latency_ms: float
    sparsity: float


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 accuracy on ``(x, y)``."""
    model.eval()
    preds = model(x).argmax(dim=1)
    return float((preds == y).float().mean())


def size_mb(model: nn.Module) -> float:
    """Serialized state_dict size in megabytes (what you'd ship to disk)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 * 1024)


@torch.no_grad()
def latency_ms(model: nn.Module, x: torch.Tensor, n_reps: int = 30, batch: int = 128) -> float:
    """Median per-forward-pass latency in ms over ``n_reps`` runs."""
    model.eval()
    sample = x[:batch]
    model(sample)  # warm-up (kernels, quant dispatch)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model(sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return float(times[len(times) // 2])


def benchmark(
    model: nn.Module,
    data: ClassDataset,
    n_reps: int = 30,
    batch: int = 128,
) -> VariantMetrics:
    """Benchmark a variant on the test split -> accuracy / size / latency / sparsity."""
    return VariantMetrics(
        accuracy=accuracy(model, data.x_test, data.y_test),
        size_mb=size_mb(model),
        latency_ms=latency_ms(model, data.x_test, n_reps=n_reps, batch=batch),
        sparsity=_sparsity(model),
    )


def metrics_dict(m: VariantMetrics) -> dict:
    """VariantMetrics -> plain JSON-serializable dict."""
    return {k: float(v) for k, v in asdict(m).items()}
