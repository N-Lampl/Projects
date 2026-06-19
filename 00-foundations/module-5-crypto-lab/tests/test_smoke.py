"""Fast smoke tests (run in CI). One slow end-to-end test is marked @slow
and excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from crypto_lab import (
    decrypt_bytes,
    ecb_encrypt,
    encrypt_bytes,
    encrypt_image_ecb,
    gen_key,
    gen_salt,
    generate_keypair,
    make_penguin,
    salted_hash,
    set_seed,
    sign,
    unsalted_hash,
    verify,
)


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


# --- hashing invariants ----------------------------------------------------- #


def test_unsalted_reuse_collides_salted_does_not():
    pw = "hunter2"
    assert unsalted_hash(pw) == unsalted_hash(pw)  # deterministic, no salt
    s1, s2 = gen_salt(), gen_salt()
    assert salted_hash(pw, s1) != salted_hash(pw, s2)  # salt breaks the collision


def test_salted_hash_is_reproducible_with_same_salt():
    salt = gen_salt()
    assert salted_hash("abc", salt) == salted_hash("abc", salt)


# --- AES / ECB invariants --------------------------------------------------- #


def test_ecb_leaks_identical_blocks():
    """Two identical 16-byte plaintext blocks -> identical ciphertext blocks."""
    key = gen_key()
    block = b"YELLOW SUBMARINE"  # exactly 16 bytes
    ct = ecb_encrypt(block * 2, key)
    assert ct[:16] == ct[16:32]  # the ECB flaw, made explicit


def test_encrypt_image_preserves_shape_and_changes_pixels():
    img = make_penguin(64)
    key = gen_key()
    enc = encrypt_image_ecb(img, key)
    assert enc.shape == img.shape
    assert enc.dtype == np.uint8
    assert not np.array_equal(enc, img)


def test_penguin_has_large_flat_regions():
    """The demo only works if the image has big constant areas (few unique blocks)."""
    img = make_penguin(128)
    b = img.tobytes()
    b = b[: (len(b) // 16) * 16]
    blocks = {b[i : i + 16] for i in range(0, len(b), 16)}
    uniqueness = len(blocks) / (len(b) // 16)
    assert uniqueness < 0.5  # lots of repeated blocks


# --- RSA + signatures invariants -------------------------------------------- #


def test_rsa_encrypt_decrypt_roundtrip():
    set_seed(7)
    key = generate_keypair(bits=512)  # small + fast for CI
    msg = b"hi"
    ct = encrypt_bytes(msg, key.public)
    assert decrypt_bytes(ct, key, len(msg)) == msg


def test_rsa_signature_verifies_and_rejects_tampering():
    set_seed(7)
    key = generate_keypair(bits=512)
    msg = b"transfer $100"
    sig = sign(msg, key)
    assert verify(msg, sig, key.public)
    assert not verify(b"transfer $900", sig, key.public)


# --- one slow end-to-end test ----------------------------------------------- #


@pytest.mark.slow
def test_full_pipeline_end_to_end(tmp_path):
    """Run the lab script's logic end to end and assert the headline results."""
    from crypto_lab import simulate_breach

    set_seed(42)
    users = ["password", "123456", "password", "qwerty"]
    wordlist = ["password", "123456", "qwerty", "letmein"]
    res = simulate_breach(users, wordlist)
    assert res.n_cracked_unsalted == 4  # all in the wordlist -> all cracked
    assert res.n_cracked_salted == 0  # salting defeats the table

    img = make_penguin(96)
    key = gen_key()
    enc = encrypt_image_ecb(img, key)
    # ECB ciphertext must still be blockier than random (structure leaked).
    eb = enc.tobytes()
    eb = eb[: (len(eb) // 16) * 16]
    ecb_uniqueness = len({eb[i : i + 16] for i in range(0, len(eb), 16)}) / (len(eb) // 16)
    assert ecb_uniqueness < 1.0
