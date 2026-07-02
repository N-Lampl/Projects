"""Synthetic induction task: sequences with a repeated subsequence.

Each example is a random sequence over a small vocab that is then *duplicated*:
``[a b c d ... | a b c d ...]``. In the second half, the optimal next-token
predictor is the classic **induction** rule: when you see a token, look back to
the previous occurrence of that same token and copy whatever came *after* it.
An induction head implements exactly ``[A][B] ... [A] -> [B]``.

Everything is deterministic given a seed, and we expose next-token targets plus
a boolean mask marking the positions where the induction rule is well-defined
(the second, repeated half) so the interp code can score heads only there.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InductionBatch:
    """A batch of repeated-sequence examples with next-token targets."""

    tokens: torch.Tensor  # (B, T) int64 input ids
    targets: torch.Tensor  # (B, T) int64 next-token targets
    repeat_mask: torch.Tensor  # (B, T) bool: positions in the repeated half
    half: int  # length of one copy (T == 2 * half)


def make_induction_batch(
    batch_size: int = 64,
    half: int = 16,
    vocab_size: int = 64,
    seed: int = 0,
) -> InductionBatch:
    """Build a batch of ``[seq | seq]`` sequences over a small vocab.

    The first token of the vocab (``0``) is reserved as a low-frequency filler so
    that repeats within a single random half are rare, making the induction rule
    the *only* reliable way to predict the second half.
    """
    gen = torch.Generator().manual_seed(seed)
    # Draw the first half; sample from ids [1, vocab_size) to keep 0 as filler.
    first = torch.randint(1, vocab_size, (batch_size, half), generator=gen)
    tokens = torch.cat([first, first], dim=1)  # (B, 2*half)

    # Next-token targets: predict tokens[:, i+1]; last position has no target, so
    # we point it at itself and it is masked out of every loss/metric below.
    targets = torch.empty_like(tokens)
    targets[:, :-1] = tokens[:, 1:]
    targets[:, -1] = tokens[:, -1]

    # The induction rule is well-defined only in the repeated (second) half, and
    # only up to the second-to-last position (the last has no next token).
    repeat_mask = torch.zeros_like(tokens, dtype=torch.bool)
    repeat_mask[:, half:-1] = True

    return InductionBatch(tokens=tokens, targets=targets, repeat_mask=repeat_mask, half=half)


def prev_occurrence_plus_one(tokens: torch.Tensor) -> torch.Tensor:
    """For each position, the index of ``prev-occurrence-of-this-token + 1``.

    This is the source position an induction head should attend to. Returns
    ``-1`` where no previous occurrence exists (so no induction target). Shape
    ``(B, T)``, dtype int64.
    """
    b, t = tokens.shape
    out = torch.full((b, t), -1, dtype=torch.long)
    for bi in range(b):
        last_seen: dict[int, int] = {}
        for i in range(t):
            tok = int(tokens[bi, i].item())
            if tok in last_seen:
                src = last_seen[tok] + 1
                if src < t:
                    out[bi, i] = src
            last_seen[tok] = i
    return out
