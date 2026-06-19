"""Fast smoke tests (run in CI). The one slow test that trains is marked @slow
and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import math

import pytest
import torch

from llm_privacy import (
    CharLM,
    Canary,
    build_corpus,
    estimate_exposure,
    make_canary,
    sequence_log_perplexity,
    set_seed,
)
from llm_privacy.exposure import LOG2_R, RANDOMNESS_SPACE


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_corpus_contains_inserted_canaries():
    corpus, canaries = build_corpus(n_background=200, n_canaries=3, canary_repeats=5, seed=1)
    assert len(canaries) == 3
    for c in canaries:
        # each canary appears exactly canary_repeats times
        assert corpus.count(c.text) == 5
        assert c.secret in c.text


def test_model_output_shape():
    model = CharLM().eval()
    x = torch.randint(0, 5, (2, 16))
    logits, _ = model(x)
    assert logits.shape == (2, 16, model.head.out_features)


def test_perplexity_is_finite_and_positive():
    set_seed(0)
    model = CharLM().eval()
    p = sequence_log_perplexity(model, make_canary("alice", "1234567890"))
    assert math.isfinite(p)
    assert p > 0  # untrained log-perplexity is roughly log(vocab) > 0


def test_randomness_space_and_max_exposure_consistent():
    # |R| = 10**10, log2(|R|) ~ 33.2 bits
    assert RANDOMNESS_SPACE == 10**10
    assert abs(LOG2_R - math.log2(RANDOMNESS_SPACE)) < 1e-9


def test_exposure_near_baseline_on_untrained_model():
    """An untrained model has no memorization, so exposure should be small
    (well below the ~33-bit maximum)."""
    set_seed(0)
    model = CharLM().eval()
    c = Canary(name="alice", secret="1357913579",
               text=make_canary("alice", "1357913579"))
    res = estimate_exposure(model, c, n_samples=200, seed=3)
    assert math.isfinite(res.exposure)
    assert res.exposure < LOG2_R * 0.5  # not memorized


@pytest.mark.slow
def test_training_memorizes_canary_end_to_end():
    """Train on a corpus with a heavily-repeated canary and confirm its exposure
    rises well above baseline (the whole point of the project)."""
    from llm_privacy import train

    set_seed(42)
    corpus, canaries = build_corpus(n_background=1500, n_canaries=2, canary_repeats=24, seed=42)
    model = CharLM()
    train(model, corpus, epochs=8, log_every=0)

    exposures = [estimate_exposure(model, c, n_samples=500, seed=5).exposure for c in canaries]
    # a never-inserted secret as control
    control = Canary(name=canaries[0].name, secret="0000000001",
                     text=make_canary(canaries[0].name, "0000000001"))
    control_exp = estimate_exposure(model, control, n_samples=500, seed=6).exposure

    assert max(exposures) > 8.0  # clear memorization signal
    assert max(exposures) > control_exp  # inserted secrets beat the control
