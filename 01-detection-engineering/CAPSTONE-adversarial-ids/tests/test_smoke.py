"""Fast smoke tests (run in CI). The one slow end-to-end test is marked @slow
and excluded from CI via ``-m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from adversarial_ids import (
    build_constraints,
    constrained_fgsm,
    fit_surrogate,
    get_pipeline_api,
    set_seed,
)
from adversarial_ids.constraints import IMMUTABLE_FEATURES, MUTABLE_FEATURES
from adversarial_ids.surrogate import GradientSurrogate


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_shared_pipeline_imports():
    """The capstone must be able to load the shared ids_pipeline by path."""
    api = get_pipeline_api()
    for fn in ("load_data", "build_pipeline", "train", "evaluate"):
        assert hasattr(api, fn)


def test_constraints_partition_is_disjoint_and_complete():
    api = get_pipeline_api()
    num = api.data.NUMERIC_FEATURES
    cspec = build_constraints(num)
    assert set(MUTABLE_FEATURES).isdisjoint(IMMUTABLE_FEATURES)
    # every numeric feature is classified as mutable XOR immutable
    assert set(MUTABLE_FEATURES) | set(IMMUTABLE_FEATURES) == set(num)
    assert int(cspec.mutable_mask.sum()) == len(MUTABLE_FEATURES)


def test_mask_perturbation_zeroes_immutable_and_blocks_decreases():
    # 2 mutable (duration, src_bytes), 1 immutable (serror_rate)
    num = ["duration", "src_bytes", "serror_rate"]
    spec = build_constraints(num)
    delta = np.array([[-5.0, 3.0, 0.9]])  # decrease duration, increase bytes, move rate
    masked = spec.mask_perturbation(delta)
    # immutable serror_rate -> 0 change
    assert masked[0, 2] == 0.0
    # duration is increase-only -> negative move blocked to 0
    assert masked[0, 0] == 0.0
    # src_bytes increase preserved
    assert masked[0, 1] == 3.0


def _toy_surrogate(d: int) -> GradientSurrogate:
    return GradientSurrogate(
        w=np.ones(d),
        b=0.0,
        mean=np.zeros(d),
        scale=np.ones(d),
    )


def test_constrained_fgsm_respects_mutability_and_bounds():
    api = get_pipeline_api()
    num = api.data.NUMERIC_FEATURES
    cspec = build_constraints(num)
    d = len(num)
    sur = _toy_surrogate(d)
    set_seed(0)
    # start from VALID flows: volumetric features large, bounded rates in [0,1]
    X = np.abs(np.random.rand(16, d)) + 1.0
    bounded = np.isfinite(cspec.hi)
    X[:, bounded] = np.random.rand(16, int(bounded.sum()))  # rates in [0,1]
    y = np.ones(16)

    X_adv = constrained_fgsm(sur, cspec, X, y, epsilon=0.5, steps=5)

    assert X_adv.shape == X.shape
    # immutable features must be untouched
    immutable = cspec.mutable_mask == 0.0
    assert np.allclose(X_adv[:, immutable], X[:, immutable])
    # all features (incl. bounded rates) stay valid after the attack
    assert cspec.is_consistent(X_adv).all()
    # something actually moved on the mutable features
    assert not np.allclose(X_adv, X)


def test_increase_only_features_never_decrease():
    api = get_pipeline_api()
    num = api.data.NUMERIC_FEATURES
    cspec = build_constraints(num)
    d = len(num)
    # surrogate whose gradient pushes everything down (sign negative)
    sur = GradientSurrogate(w=-np.ones(d), b=0.0, mean=np.zeros(d), scale=np.ones(d))
    X = np.full((8, d), 10.0)
    y = np.ones(8)
    X_adv = constrained_fgsm(sur, cspec, X, y, epsilon=0.5, steps=5)
    inc = cspec.increase_only_mask == 1.0
    assert (X_adv[:, inc] >= X[:, inc] - 1e-9).all()


def test_surrogate_gradient_shape_and_proba_range():
    api = get_pipeline_api()
    ds = api.load_data(synthetic=True, n_samples=600)
    num = ds.numeric_features
    base = api.build_pipeline(ds)
    api.train(base, ds)
    X = ds.X_train[num].to_numpy(dtype=float)
    sur = fit_surrogate(base, X, num)
    p = sur.proba(X[:20])
    assert p.shape == (20,)
    assert (p >= 0).all() and (p <= 1).all()
    g = sur.loss_gradient_raw(X[:20], np.ones(20))
    assert g.shape == (20, len(num))


@pytest.mark.slow
def test_capstone_end_to_end_hardening_reduces_asr():
    """Full loop: train -> attack -> harden -> re-attack. ASR should not rise."""
    from adversarial_ids import (
        adversarially_train,
        attack_success_rate,
        craft_adversarial,
        refit_surrogate_for,
    )

    set_seed(42)
    api = get_pipeline_api()
    ds = api.load_data(synthetic=True, n_samples=4000)
    num = ds.numeric_features
    cspec = build_constraints(num)
    base = api.build_pipeline(ds)
    api.train(base, ds)

    X_test = ds.X_test[num].to_numpy(dtype=float)
    y_test = ds.y_test.to_numpy()
    sur = fit_surrogate(base, ds.X_train[num].to_numpy(dtype=float), num)
    X_atk = X_test[y_test == 1]
    y_atk = y_test[y_test == 1]

    X_adv, _ = craft_adversarial(sur, cspec, X_atk, y_atk, epsilon=0.5, steps=10)
    before = attack_success_rate(base, cspec, num, X_atk, X_adv, y_atk)

    hardened, _ = adversarially_train(api, ds, cspec, sur, epsilon=0.5, steps=10)
    sur2 = refit_surrogate_for(api, hardened, ds)
    X_adv2, _ = craft_adversarial(sur2, cspec, X_atk, y_atk, epsilon=0.5, steps=10)
    after = attack_success_rate(hardened, cspec, num, X_atk, X_adv2, y_atk)

    # the attack is real (constraints satisfied) and hardening does not make it worse
    assert before["immutable_preserved_rate"] == 1.0
    assert before["consistency_rate"] == 1.0
    assert after["attack_success_rate"] <= before["attack_success_rate"] + 1e-9
