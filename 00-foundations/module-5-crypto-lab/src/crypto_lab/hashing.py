"""Password hashing: why a salt matters.

Default path uses only `hashlib` (stdlib). The misuse here is *unsalted*
hashing: identical passwords -> identical digests, so a single precomputed
"rainbow table" cracks every matching account at once. A per-user random salt
breaks that: the same password yields a different digest for each user.

NOTE: SHA-256 (even salted) is *fast*, which makes it a poor choice for real
password storage — use a slow KDF (scrypt/argon2/PBKDF2). We expose a
PBKDF2 helper to show the right primitive while keeping the salt demo simple.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


def sha256_hex(data: bytes) -> str:
    """Plain SHA-256 hex digest (the *unsalted* baseline)."""
    return hashlib.sha256(data).hexdigest()


def unsalted_hash(password: str) -> str:
    """Hash a password with no salt — the misuse we are demonstrating."""
    return sha256_hex(password.encode("utf-8"))


def salted_hash(password: str, salt: bytes) -> str:
    """Hash a password with a per-user salt prepended to the input."""
    return sha256_hex(salt + password.encode("utf-8"))


def gen_salt(n_bytes: int = 16) -> bytes:
    """Cryptographically random salt."""
    return os.urandom(n_bytes)


def pbkdf2_hash(password: str, salt: bytes, iterations: int = 200_000) -> str:
    """The *correct* primitive: a slow, salted KDF (PBKDF2-HMAC-SHA256)."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return dk.hex()


@dataclass
class CrackResult:
    """Outcome of running a precomputed-table attack against a user list."""

    n_users: int
    n_cracked_unsalted: int
    n_cracked_salted: int


def rainbow_table(candidate_passwords: list[str]) -> dict[str, str]:
    """Precompute unsalted SHA-256 digests for a wordlist (digest -> password)."""
    return {unsalted_hash(p): p for p in candidate_passwords}


def simulate_breach(
    user_passwords: list[str],
    candidate_passwords: list[str],
) -> CrackResult:
    """Steal a hashed user DB, then attack it with a precomputed table.

    - Unsalted: one table cracks every user whose password is in the wordlist.
    - Salted: the table is useless because each digest was computed over a
      unique salt the attacker did not know in advance.
    """
    table = rainbow_table(candidate_passwords)

    # Build the two "stolen" databases.
    unsalted_db = [unsalted_hash(p) for p in user_passwords]
    salts = [gen_salt() for _ in user_passwords]
    salted_db = [salted_hash(p, s) for p, s in zip(user_passwords, salts, strict=True)]

    cracked_unsalted = sum(1 for h in unsalted_db if h in table)
    # The naive table lookup fails on salted hashes (different digests).
    cracked_salted = sum(1 for h in salted_db if h in table)

    return CrackResult(
        n_users=len(user_passwords),
        n_cracked_unsalted=cracked_unsalted,
        n_cracked_salted=cracked_salted,
    )
