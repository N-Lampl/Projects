"""Three classic mechanistic-interpretability analyses on the tiny model.

1. ``induction_head_score`` - how much attention mass each head puts on the
   ``prev-occurrence + 1`` position (the induction source) in the repeated half.
2. ``logit_lens`` - project each layer's residual stream through the unembedding
   and measure next-token accuracy; it should sharpen with depth.
3. ``activation_patching`` - splice a clean residual activation into a corrupted
   run and measure how much of the correct-token logit is recovered.
"""

from __future__ import annotations

import torch

from .model import TinyTransformer
from .task import InductionBatch, prev_occurrence_plus_one


def induction_head_score(model: TinyTransformer, batch: InductionBatch) -> dict:
    """Per-head attention mass on the induction source position.

    For every (layer, head) we average, over positions in the repeated half that
    have a defined previous occurrence, the attention weight placed on the
    ``prev-occurrence + 1`` token. A true induction head scores near 1.0.
    """
    model.eval()
    with torch.no_grad():
        _, cache = model(batch.tokens, return_cache=True)

    src = prev_occurrence_plus_one(batch.tokens)  # (B, T), -1 where undefined
    valid = batch.repeat_mask & (src >= 0)  # (B, T)

    per_layer: list[list[float]] = []
    for attn in cache.attn:  # attn: (B, H, T, T)
        b, h, t, _ = attn.shape
        head_scores: list[float] = []
        for head in range(h):
            aw = attn[:, head, :, :]  # (B, T, T)
            src_clamped = src.clamp(min=0)  # (B, T)
            mass = torch.gather(aw, 2, src_clamped[:, :, None]).squeeze(-1)  # (B, T)
            sel = mass[valid]
            head_scores.append(float(sel.mean().item()) if sel.numel() else 0.0)
        per_layer.append(head_scores)

    flat = [s for layer in per_layer for s in layer]
    best = max(flat)
    best_idx = flat.index(best)
    n_heads = len(per_layer[0])
    return {
        "per_head": per_layer,  # [layer][head]
        "max_score": best,
        "best_layer": best_idx // n_heads,
        "best_head": best_idx % n_heads,
    }


def logit_lens(model: TinyTransformer, batch: InductionBatch) -> dict:
    """Next-token accuracy from each layer's residual stream via the unembedding.

    Applies the final LayerNorm + unembedding to every cached residual stream
    (post-embedding, then after each block) and measures accuracy on the
    induction-half positions. Accuracy should increase with depth.
    """
    model.eval()
    with torch.no_grad():
        _, cache = model(batch.tokens, return_cache=True)

    valid = batch.repeat_mask  # (B, T)
    targets = batch.targets

    accuracies: list[float] = []
    correct_logits: list[float] = []
    for resid in cache.resid:  # (B, T, d_model)
        logits = model.resid_to_logits(resid)  # (B, T, V)
        preds = logits.argmax(dim=-1)  # (B, T)
        acc = (preds[valid] == targets[valid]).float().mean().item()
        accuracies.append(float(acc))
        tgt_logit = torch.gather(logits, 2, targets[:, :, None]).squeeze(-1)  # (B, T)
        correct_logits.append(float(tgt_logit[valid].mean().item()))

    return {
        "accuracy_by_layer": accuracies,  # index 0 = embedding, then per block
        "correct_logit_by_layer": correct_logits,
        "n_layers": len(accuracies) - 1,
    }


def activation_patching(
    model: TinyTransformer,
    clean: InductionBatch,
    corrupt: InductionBatch,
    layer: int,
    position: int,
) -> dict:
    """Patch the clean residual at ``(layer, position)`` into the corrupt run.

    We measure the correct-token logit at ``position`` (target = the clean
    next token) in three conditions: clean run, corrupt run, and corrupt run
    with the clean residual spliced in. The patching effect is the fraction of
    the clean-minus-corrupt gap recovered by the splice - near 1.0 when that
    activation carries the information the model needs.
    """
    model.eval()
    tgt = clean.targets[:, position]  # (B,) correct next token from the clean run

    def correct_logit(run_batch: InductionBatch, patch=None) -> torch.Tensor:
        with torch.no_grad():
            logits = model(run_batch.tokens, resid_patch=patch)
        return torch.gather(logits[:, position, :], 1, tgt[:, None]).squeeze(-1)

    # Cache the clean residual stream to donate its activation.
    with torch.no_grad():
        _, clean_cache = model(clean.tokens, return_cache=True)
    donor = clean_cache.resid[layer][:, position, :]  # (B, d_model)

    clean_logit = correct_logit(clean)
    corrupt_logit = correct_logit(corrupt)
    patched_logit = correct_logit(corrupt, patch=(layer, position, donor))

    gap = (clean_logit - corrupt_logit).mean()
    recovered = (patched_logit - corrupt_logit).mean()
    effect = float((recovered / gap).item()) if abs(float(gap.item())) > 1e-6 else 0.0

    return {
        "layer": layer,
        "position": position,
        "clean_logit": float(clean_logit.mean().item()),
        "corrupt_logit": float(corrupt_logit.mean().item()),
        "patched_logit": float(patched_logit.mean().item()),
        "recovered": float(recovered.item()),
        "gap": float(gap.item()),
        "effect": effect,  # fraction of the clean-corrupt gap recovered
    }


def patching_sweep(
    model: TinyTransformer,
    clean: InductionBatch,
    corrupt: InductionBatch,
    position: int,
) -> dict:
    """Run ``activation_patching`` across every residual layer at one position."""
    n_resid = model.cfg.n_layers + 1
    effects = [
        activation_patching(model, clean, corrupt, layer=i, position=position)["effect"]
        for i in range(n_resid)
    ]
    return {"position": position, "effect_by_layer": effects}
