"""Transformer internals & mechanistic interpretability (tiny model, CPU)."""

from __future__ import annotations

from .interp import (
    activation_patching,
    induction_head_score,
    logit_lens,
    patching_sweep,
)
from .model import ModelConfig, RunCache, TinyTransformer
from .plots import (
    plot_activation_patching,
    plot_attention_induction,
    plot_logit_lens,
)
from .task import (
    InductionBatch,
    make_induction_batch,
    prev_occurrence_plus_one,
)
from .train import TrainResult, masked_next_token_loss, train_induction
from .utils import get_device, set_seed

__all__ = [
    "InductionBatch",
    "ModelConfig",
    "RunCache",
    "TinyTransformer",
    "TrainResult",
    "activation_patching",
    "get_device",
    "induction_head_score",
    "logit_lens",
    "make_induction_batch",
    "masked_next_token_loss",
    "patching_sweep",
    "plot_activation_patching",
    "plot_attention_induction",
    "plot_logit_lens",
    "prev_occurrence_plus_one",
    "set_seed",
    "train_induction",
]
