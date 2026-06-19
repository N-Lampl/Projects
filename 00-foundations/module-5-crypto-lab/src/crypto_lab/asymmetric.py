"""Public-key crypto: RSA encryption and digital signatures (textbook).

DEFAULT PATH: a small, self-contained "textbook RSA" built on Python's
big-int support and `secrets` (stdlib). It demonstrates the core ideas:

    keygen:  n = p*q,  e=65537,  d = e^-1 mod phi(n)
    encrypt: c = m^e mod n          decrypt: m = c^d mod n
    sign:    s = H(msg)^d mod n     verify:  s^e mod n == H(msg)

Textbook RSA has no padding and is NOT secure for real use (it is
deterministic and malleable). The README points at OAEP/PSS as the fix.

ENHANCED PATH: if `cryptography` is installed, `oaep_roundtrip` and
`pss_sign_verify` exercise the padded, production-grade primitives. Lazy
imports keep this module loadable without that dependency.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass


def _is_probable_prime(n: int, rounds: int = 40) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _gen_prime(bits: int) -> int:
    while True:
        cand = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(cand):
            return cand


@dataclass
class RSAKey:
    """A textbook-RSA keypair. (e, n) is public; d is the private exponent."""

    n: int
    e: int
    d: int

    @property
    def public(self) -> tuple[int, int]:
        return self.e, self.n


def generate_keypair(bits: int = 1024, e: int = 65537) -> RSAKey:
    """Generate a textbook-RSA keypair (default 1024-bit modulus, demo only)."""
    half = bits // 2
    while True:
        p = _gen_prime(half)
        q = _gen_prime(half)
        if p == q:
            continue
        n = p * q
        phi = (p - 1) * (q - 1)
        if phi % e == 0:
            continue
        d = pow(e, -1, phi)
        return RSAKey(n=n, e=e, d=d)


def encrypt_int(m: int, pub: tuple[int, int]) -> int:
    e, n = pub
    if m >= n:
        raise ValueError("message integer must be < n")
    return pow(m, e, n)


def decrypt_int(c: int, key: RSAKey) -> int:
    return pow(c, key.d, key.n)


def encrypt_bytes(data: bytes, pub: tuple[int, int]) -> int:
    """Encrypt a short message (must be < modulus). Textbook, no padding."""
    m = int.from_bytes(data, "big")
    return encrypt_int(m, pub)


def decrypt_bytes(c: int, key: RSAKey, length: int) -> bytes:
    m = decrypt_int(c, key)
    return m.to_bytes(length, "big")


def _digest_int(message: bytes, n: int) -> int:
    """SHA-256 digest reduced into the modulus (toy 'hash-then-sign')."""
    return int.from_bytes(hashlib.sha256(message).digest(), "big") % n


def sign(message: bytes, key: RSAKey) -> int:
    """Sign = encrypt the message hash with the PRIVATE exponent."""
    return pow(_digest_int(message, key.n), key.d, key.n)


def verify(message: bytes, signature: int, pub: tuple[int, int]) -> bool:
    """Verify = decrypt the signature with the PUBLIC exponent, compare hash."""
    e, n = pub
    recovered = pow(signature, e, n)
    return recovered == _digest_int(message, n)


# --------------------------------------------------------------------------- #
# Enhanced path (optional `cryptography`)                                      #
# --------------------------------------------------------------------------- #


def cryptography_available() -> bool:
    try:
        import cryptography.hazmat.primitives.asymmetric.rsa  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def oaep_roundtrip(message: bytes) -> bool:
    """Real RSA-OAEP encrypt/decrypt round trip (ENHANCED path, lazy import)."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pad = padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None,
    )
    ct = key.public_key().encrypt(message, pad)
    return key.decrypt(ct, pad) == message


def pss_sign_verify(message: bytes) -> bool:
    """Real RSA-PSS sign/verify round trip (ENHANCED path, lazy import)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pad = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
    sig = key.sign(message, pad, hashes.SHA256())
    try:
        key.public_key().verify(sig, message, pad, hashes.SHA256())
        return True
    except InvalidSignature:
        return False
