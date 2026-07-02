"""Fast, offline, deterministic tests for the transformer-interp project.

They train a tiny decoder-only transformer from scratch on the synthetic
induction task (numpy/torch only, no network, no transformers) and assert real
mechanistic behaviour: training drives next-token loss to ~0, an induction head
emerges, the logit lens sharpens with depth, and activation patching at the
right layer recovers the correct-token logit. The one transformers/distilgpt2
cross-check is marked ``@slow`` and imports transformers *lazily*.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from transformer_interp import (
    ModelConfig,
    TinyTransformer,
    activation_patching,
    induction_head_score,
    logit_lens,
    make_induction_batch,
    patching_sweep,
    prev_occurrence_plus_one,
    set_seed,
    train_induction,
)

HALF = 16
VOCAB = 64


@pytest.fixture(scope="module")
def trained():
    """Train once (deterministically) and reuse across tests (~7s on CPU)."""
    return train_induction(steps=200, batch_size=48, half=HALF, seed=42)


@pytest.fixture(scope="module")
def eval_batch():
    return make_induction_batch(batch_size=96, half=HALF, vocab_size=VOCAB, seed=7)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.randn(5)
    ta = torch.randn(5)
    set_seed(123)
    b = np.random.randn(5)
    tb = torch.randn(5)
    assert np.array_equal(a, b)
    assert torch.equal(ta, tb)


def test_model_forward_shapes_and_cache():
    cfg = ModelConfig(vocab_size=VOCAB, n_ctx=2 * HALF, n_heads=4, n_layers=2)
    model = TinyTransformer(cfg)
    batch = make_induction_batch(batch_size=8, half=HALF, vocab_size=VOCAB, seed=0)
    logits, cache = model(batch.tokens, return_cache=True)
    assert logits.shape == (8, 2 * HALF, VOCAB)
    # One attention tensor per layer, (B, H, T, T).
    assert len(cache.attn) == cfg.n_layers
    assert cache.attn[0].shape == (8, cfg.n_heads, 2 * HALF, 2 * HALF)
    # Residual stream after embedding + each layer.
    assert len(cache.resid) == cfg.n_layers + 1
    assert cache.resid[-1].shape == (8, 2 * HALF, cfg.d_model)


def test_attention_is_causal_and_normalized():
    cfg = ModelConfig(vocab_size=VOCAB, n_ctx=2 * HALF)
    model = TinyTransformer(cfg)
    batch = make_induction_batch(batch_size=4, half=HALF, vocab_size=VOCAB, seed=1)
    _, cache = model(batch.tokens, return_cache=True)
    attn = cache.attn[0]  # (B, H, T, T)
    # Rows sum to 1 (softmax over keys).
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)
    # Upper triangle (future keys) must be zero (causal mask).
    t = attn.shape[-1]
    future = torch.triu(torch.ones(t, t), diagonal=1).bool()
    assert attn[..., future].abs().max().item() < 1e-6


def test_prev_occurrence_matches_repeat_structure():
    batch = make_induction_batch(batch_size=4, half=HALF, vocab_size=VOCAB, seed=2)
    src = prev_occurrence_plus_one(batch.tokens)
    # Every repeated-half token has a defined induction source (its identical
    # copy sits exactly ``half`` positions earlier).
    assert torch.all(src[:, HALF:] >= 0)
    # The source is always "one position after an earlier occurrence of the same
    # token": tokens[src - 1] must equal the current token wherever defined.
    for bi in range(batch.tokens.shape[0]):
        for pos in range(HALF, 2 * HALF):
            s = int(src[bi, pos].item())
            if s >= 0:
                assert batch.tokens[bi, s - 1] == batch.tokens[bi, pos]


def test_training_drives_loss_low(trained):
    # From-scratch training solves the induction task almost perfectly.
    assert trained.val_loss < 0.1
    assert trained.final_loss < trained.losses[0]


def test_induction_head_emerges(trained, eval_batch):
    head = induction_head_score(trained.model, eval_batch)
    # A head devotes most of its attention to the previous-occurrence+1 token.
    assert head["max_score"] > 0.5
    assert 0.0 <= head["max_score"] <= 1.0
    # per_head is [n_layers][n_heads].
    assert len(head["per_head"]) == trained.model.cfg.n_layers
    assert len(head["per_head"][0]) == trained.model.cfg.n_heads


def test_logit_lens_sharpens_with_depth(trained, eval_batch):
    lens = logit_lens(trained.model, eval_batch)
    acc = lens["accuracy_by_layer"]
    # The final layer predicts far better than the raw embedding.
    assert acc[-1] > acc[0]
    assert acc[-1] > 0.9
    assert acc[0] < 0.2
    # Correct-token logit also rises with depth.
    assert lens["correct_logit_by_layer"][-1] > lens["correct_logit_by_layer"][0]


def test_activation_patching_recovers_signal(trained, eval_batch):
    corrupt = make_induction_batch(batch_size=96, half=HALF, vocab_size=VOCAB, seed=13)
    pos = trained.model.cfg.n_ctx - 2
    # Patching the final-layer residual should recover ~all of the logit gap.
    last = activation_patching(
        trained.model, eval_batch, corrupt, layer=trained.model.cfg.n_layers, position=pos
    )
    assert last["gap"] > 0.0  # clean beats corrupt on the correct token
    assert last["effect"] > 0.8
    # Patching the pre-computation embedding recovers almost nothing.
    first = activation_patching(trained.model, eval_batch, corrupt, layer=0, position=pos)
    assert first["effect"] < 0.3


def test_patching_effect_increases_with_layer(trained, eval_batch):
    corrupt = make_induction_batch(batch_size=96, half=HALF, vocab_size=VOCAB, seed=13)
    pos = trained.model.cfg.n_ctx - 2
    sweep = patching_sweep(trained.model, eval_batch, corrupt, position=pos)
    effects = sweep["effect_by_layer"]
    assert len(effects) == trained.model.cfg.n_layers + 1
    # Later residual streams carry more of the answer than the embedding.
    assert effects[-1] > effects[0]


def test_training_is_reproducible():
    r1 = train_induction(steps=60, batch_size=32, half=HALF, seed=7)
    r2 = train_induction(steps=60, batch_size=32, half=HALF, seed=7)
    assert r1.losses == r2.losses
    w1 = r1.model.unembed.weight.detach()
    w2 = r2.model.unembed.weight.detach()
    assert torch.equal(w1, w2)


@pytest.mark.slow
def test_distilgpt2_has_induction_like_head():
    """Real GPT-2-family model should attend back to a repeated token's successor."""
    try:
        import transformers  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # transformers not installed
        pytest.skip(f"transformers unavailable: {type(exc).__name__}")

    try:
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2", output_attentions=True)
    except Exception as exc:  # offline / weights not cached
        pytest.skip(f"distilgpt2 weights unavailable: {type(exc).__name__}")

    model.eval()
    # A repeated random-ish sequence so induction is the way to predict the copy.
    text = " apple banana cherry apple banana cherry apple banana"
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(ids)
    attns = out.attentions  # tuple[layer] of (1, n_heads, T, T)

    seq = ids[0].tolist()
    t = len(seq)
    best = 0.0
    for attn in attns:
        for head in range(attn.shape[1]):
            aw = attn[0, head]  # (T, T)
            scores = []
            last_seen: dict[int, int] = {}
            for i in range(t):
                tokid = seq[i]
                if tokid in last_seen and last_seen[tokid] + 1 < t:
                    scores.append(float(aw[i, last_seen[tokid] + 1].item()))
                last_seen[tokid] = i
            if scores:
                best = max(best, sum(scores) / len(scores))
    # Some head should place clearly non-trivial mass on the induction source.
    assert best > 0.15
