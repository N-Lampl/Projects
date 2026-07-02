"""Fast, offline, deterministic tests for the compression project.

They train a small teacher on synthetic blobs, apply each compression technique,
and assert the real trade-offs hold: pruning keeps accuracy at high sparsity, the
distilled student has fewer params but stays well above chance, quantization
shrinks the serialized model, and the benchmark returns finite numbers. No
network, no torchvision. The one MNIST cross-check is marked ``@slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from compression import (
    Student,
    Teacher,
    benchmark,
    count_params,
    distill,
    dynamic_quantize,
    magnitude_prune,
    make_blobs,
    set_seed,
    size_mb,
    sparsity,
    train_classifier,
)
from compression.benchmark import accuracy


def _trained_teacher(seed: int = 0, epochs: int = 8):
    data = make_blobs(n_samples=2000, n_features=32, n_classes=5, seed=seed)
    teacher = Teacher(data.n_features, data.n_classes, hidden=128)
    train_classifier(teacher, data, epochs=epochs, seed=seed)
    return teacher, data


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    t_a = torch.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    t_b = torch.randn(5)
    assert np.array_equal(a, b)
    assert torch.equal(t_a, t_b)


def test_baseline_learns_above_chance():
    teacher, data = _trained_teacher()
    acc = accuracy(teacher, data.x_test, data.y_test)
    assert acc > 0.85  # 5 classes -> chance is 0.2


def test_pruning_yields_sparsity_and_keeps_accuracy():
    teacher, data = _trained_teacher()
    base_acc = accuracy(teacher, data.x_test, data.y_test)
    pruned = magnitude_prune(teacher, fraction=0.7)
    s = sparsity(pruned)
    pruned_acc = accuracy(pruned, data.x_test, data.y_test)
    assert 0.6 < s < 0.8  # ~70% of Linear weights zeroed
    assert pruned_acc >= base_acc - 0.05  # accuracy within tolerance of baseline


def test_pruning_fraction_zero_is_identity():
    teacher, _ = _trained_teacher()
    pruned = magnitude_prune(teacher, fraction=0.0)
    assert sparsity(pruned) == pytest.approx(0.0, abs=1e-6)


def test_distilled_student_smaller_and_accurate():
    teacher, data = _trained_teacher()
    student = Student(data.n_features, data.n_classes, hidden=16)
    distill(teacher, student, data, epochs=25, seed=0)
    assert count_params(student) < count_params(teacher)  # fewer params
    acc = accuracy(student, data.x_test, data.y_test)
    assert acc > 0.6  # well above 0.2 chance for 5 classes


def test_quantization_shrinks_size():
    teacher, data = _trained_teacher()
    q = dynamic_quantize(teacher)
    # int8 dynamic quantization must shrink the serialized fp32 model.
    assert size_mb(q) < size_mb(teacher)
    # quantized model still classifies (finite, above chance).
    q_acc = accuracy(q, data.x_test, data.y_test)
    assert q_acc > 0.2


def test_benchmark_returns_finite_expected_keys():
    teacher, data = _trained_teacher()
    m = benchmark(teacher, data, n_reps=5, batch=64)
    for field in ("accuracy", "size_mb", "latency_ms", "sparsity"):
        v = getattr(m, field)
        assert np.isfinite(v)
    assert m.latency_ms > 0.0
    assert m.size_mb > 0.0
    assert 0.0 <= m.sparsity <= 1.0


def test_training_is_reproducible():
    t1, d = _trained_teacher(seed=3)
    t2, _ = _trained_teacher(seed=3)
    a1 = accuracy(t1, d.x_test, d.y_test)
    a2 = accuracy(t2, d.x_test, d.y_test)
    assert a1 == a2


@pytest.mark.slow
def test_mnist_baseline_reaches_reasonable_accuracy():
    """Train the teacher on real MNIST via torchvision and check it learns."""
    try:
        from compression import load_mnist

        data = load_mnist(n_train=6000, n_test=1500, seed=0)
    except Exception as exc:  # torchvision missing / download failed
        pytest.skip(f"MNIST/torchvision unavailable: {type(exc).__name__}")

    teacher = Teacher(data.n_features, data.n_classes, hidden=128)
    train_classifier(teacher, data, epochs=5, seed=0)
    acc = accuracy(teacher, data.x_test, data.y_test)
    assert acc > 0.9  # a plain MLP clears 90% on MNIST easily
