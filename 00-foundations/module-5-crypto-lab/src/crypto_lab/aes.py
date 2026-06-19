"""Symmetric encryption: ECB (bad) vs GCM (good), the classic "penguin" demo.

ECB encrypts each 16-byte block independently, so identical plaintext blocks
map to identical ciphertext blocks. Encrypt an image and the outline survives
in the ciphertext (Tux the penguin is the canonical example). GCM adds a
random nonce and chains state, so equal plaintext blocks produce unrelated
ciphertext — plus it authenticates, detecting tampering.

DEFAULT PATH: a small, self-contained pure-Python AES-128 block cipher
(`_AES`) drives a stdlib-only ECB demo. No third-party deps required.

ENHANCED PATH: if the `cryptography` package is importable we additionally
run real AES-GCM (authenticated encryption) via `gcm_encrypt`. It is imported
lazily so this module loads without it.
"""

from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------- #
# A compact, dependency-free AES-128 (ECB single block) for the default demo.  #
# Educational implementation — NOT for production use.                         #
# --------------------------------------------------------------------------- #

_SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]
_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def _xtime(a: int) -> int:
    a <<= 1
    if a & 0x100:
        a ^= 0x11B
    return a & 0xFF


def _mul(a: int, b: int) -> int:
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        a = _xtime(a)
        b >>= 1
    return p & 0xFF


class _AES:
    """Minimal AES-128 single-block encryptor (ECB primitive)."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 16:
            raise ValueError("default AES demo requires a 16-byte key")
        self._round_keys = self._expand(key)

    @staticmethod
    def _expand(key: bytes) -> list[list[int]]:
        words = [list(key[i : i + 4]) for i in range(0, 16, 4)]
        for i in range(4, 44):
            tmp = list(words[i - 1])
            if i % 4 == 0:
                tmp = tmp[1:] + tmp[:1]  # RotWord
                tmp = [_SBOX[b] for b in tmp]  # SubWord
                tmp[0] ^= _RCON[i // 4 - 1]
            words.append([words[i - 4][j] ^ tmp[j] for j in range(4)])
        # group into eleven 16-byte round keys (column-major state)
        return [sum(words[r * 4 : r * 4 + 4], []) for r in range(11)]

    def encrypt_block(self, block: bytes) -> bytes:
        s = list(block)
        self._add_round_key(s, self._round_keys[0])
        for rnd in range(1, 10):
            s = [_SBOX[b] for b in s]
            self._shift_rows(s)
            self._mix_columns(s)
            self._add_round_key(s, self._round_keys[rnd])
        s = [_SBOX[b] for b in s]
        self._shift_rows(s)
        self._add_round_key(s, self._round_keys[10])
        return bytes(s)

    @staticmethod
    def _add_round_key(s: list[int], rk: list[int]) -> None:
        for i in range(16):
            s[i] ^= rk[i]

    @staticmethod
    def _shift_rows(s: list[int]) -> None:
        # state is column-major: s[r + 4*c]
        for r in range(1, 4):
            row = [s[r + 4 * c] for c in range(4)]
            row = row[r:] + row[:r]
            for c in range(4):
                s[r + 4 * c] = row[c]

    @staticmethod
    def _mix_columns(s: list[int]) -> None:
        for c in range(4):
            col = [s[4 * c + r] for r in range(4)]
            s[4 * c + 0] = _mul(col[0], 2) ^ _mul(col[1], 3) ^ col[2] ^ col[3]
            s[4 * c + 1] = col[0] ^ _mul(col[1], 2) ^ _mul(col[2], 3) ^ col[3]
            s[4 * c + 2] = col[0] ^ col[1] ^ _mul(col[2], 2) ^ _mul(col[3], 3)
            s[4 * c + 3] = _mul(col[0], 3) ^ col[1] ^ col[2] ^ _mul(col[3], 2)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def ecb_encrypt(data: bytes, key: bytes) -> bytes:
    """AES-128-ECB with zero-padding to a 16-byte multiple (DEFAULT path).

    Pure-Python. ECB is intentionally insecure — used only to expose the
    pattern-leakage flaw visually.
    """
    cipher = _AES(key)
    if len(data) % 16:
        data = data + b"\x00" * (16 - len(data) % 16)
    out = bytearray()
    for i in range(0, len(data), 16):
        out += cipher.encrypt_block(data[i : i + 16])
    return bytes(out)


def encrypt_image_ecb(img: np.ndarray, key: bytes) -> np.ndarray:
    """Encrypt a grayscale uint8 image block-by-block with AES-ECB.

    Returns a uint8 array of the SAME shape, so it can be displayed: the
    ciphertext "image" still shows the original outline because identical
    plaintext blocks become identical ciphertext blocks.
    """
    if img.dtype != np.uint8:
        raise ValueError("expected a uint8 grayscale image")
    h, w = img.shape
    flat = img.tobytes()
    ct = ecb_encrypt(flat, key)[: h * w]
    return np.frombuffer(ct, dtype=np.uint8).reshape(h, w)


def gen_key() -> bytes:
    """Random 16-byte (AES-128) key."""
    return os.urandom(16)


def gcm_available() -> bool:
    """True if the optional `cryptography` library is importable."""
    try:
        import cryptography.hazmat.primitives.ciphers.aead  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def gcm_encrypt(plaintext: bytes, key: bytes, aad: bytes = b"") -> tuple[bytes, bytes]:
    """AES-GCM authenticated encryption (ENHANCED path, lazy import).

    Returns (nonce, ciphertext_with_tag). Requires `cryptography`.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    return nonce, ct


def gcm_decrypt(nonce: bytes, ciphertext: bytes, key: bytes, aad: bytes = b"") -> bytes:
    """Verify-and-decrypt an AES-GCM ciphertext. Raises on tampering."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key).decrypt(nonce, ciphertext, aad)
