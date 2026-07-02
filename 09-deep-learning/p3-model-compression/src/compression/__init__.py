"""Model compression & efficient inference: pruning · quantization · distillation."""

from __future__ import annotations

from .benchmark import (
    VariantMetrics,
    accuracy,
    benchmark,
    latency_ms,
    metrics_dict,
    size_mb,
)
from .compress import distill, dynamic_quantize, magnitude_prune, sparsity
from .data import ClassDataset, load_mnist, make_blobs
from .models import Student, Teacher, count_params
from .train import train_classifier
from .utils import get_device, set_seed

__all__ = [
    "ClassDataset",
    "Student",
    "Teacher",
    "VariantMetrics",
    "accuracy",
    "benchmark",
    "count_params",
    "distill",
    "dynamic_quantize",
    "get_device",
    "latency_ms",
    "load_mnist",
    "magnitude_prune",
    "make_blobs",
    "metrics_dict",
    "set_seed",
    "size_mb",
    "sparsity",
    "train_classifier",
]
