"""The three compression techniques: pruning, quantization, distillation.

All three operate on the plain ``nn.Linear`` stacks defined in ``models.py`` and
are deterministic on CPU given a seed.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn.utils import prune

from .data import ClassDataset


def magnitude_prune(model: nn.Module, fraction: float) -> nn.Module:
    """Zero the smallest-|weight| ``fraction`` of every Linear layer's weights.

    Returns a *copy* with the pruning masks made permanent (so the zeros persist
    in the state_dict and sparsity is real, not a reparametrisation hook).
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    pruned = copy.deepcopy(model)
    for module in pruned.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=fraction)
            prune.remove(module, "weight")  # bake the mask into weight
    pruned.eval()
    return pruned


def sparsity(model: nn.Module) -> float:
    """Fraction of weight entries (Linear layers) that are exactly zero."""
    zeros = 0
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.detach()
            zeros += int((w == 0).sum())
            total += w.numel()
    return zeros / total if total else 0.0


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Post-training dynamic quantization of Linear layers to int8 (CPU)."""
    model = copy.deepcopy(model).eval()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def distill(
    teacher: nn.Module,
    student: nn.Module,
    data: ClassDataset,
    epochs: int = 20,
    lr: float = 1e-2,
    batch_size: int = 256,
    temperature: float = 3.0,
    alpha: float = 0.7,
    seed: int = 0,
) -> nn.Module:
    """Train ``student`` on the teacher's soft targets (knowledge distillation).

    Loss = ``alpha`` * soft KL(student || teacher, T) * T^2
         + (1 - ``alpha``) * hard cross-entropy against the true labels.
    Trains the student in place and returns it.
    """
    torch.manual_seed(seed)
    teacher.eval()
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    x, y = data.x_train, data.y_train
    n = len(x)

    with torch.no_grad():
        teacher_logits = teacher(x)

    t = temperature
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            s_logits = student(x[idx])
            soft = kl(
                torch.log_softmax(s_logits / t, dim=1),
                torch.softmax(teacher_logits[idx] / t, dim=1),
            ) * (t * t)
            hard = ce(s_logits, y[idx])
            (alpha * soft + (1.0 - alpha) * hard).backward()
            opt.step()
    student.eval()
    return student
