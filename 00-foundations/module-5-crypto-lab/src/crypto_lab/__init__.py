"""crypto_lab: hands-on crypto basics + common misuse demos.

Default path = stdlib + numpy/matplotlib only (no third-party crypto needed).
Optional enhanced path uses `cryptography` via lazy imports.

Public API:
    set_seed, get_device                      -- reproducibility helpers
    unsalted_hash, salted_hash, gen_salt,
    pbkdf2_hash, simulate_breach              -- hashing: salt vs no salt
    ecb_encrypt, encrypt_image_ecb, gen_key,
    gcm_encrypt, gcm_decrypt, gcm_available   -- AES ECB (bad) vs GCM (good)
    generate_keypair, encrypt_bytes,
    decrypt_bytes, sign, verify               -- textbook RSA + signatures
    make_penguin                              -- synthetic image for ECB demo
"""

from .aes import (
    ecb_encrypt,
    encrypt_image_ecb,
    gcm_available,
    gcm_decrypt,
    gcm_encrypt,
    gen_key,
)
from .asymmetric import (
    RSAKey,
    decrypt_bytes,
    encrypt_bytes,
    generate_keypair,
    sign,
    verify,
)
from .hashing import (
    CrackResult,
    gen_salt,
    pbkdf2_hash,
    salted_hash,
    simulate_breach,
    unsalted_hash,
)
from .image import make_penguin
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "unsalted_hash",
    "salted_hash",
    "gen_salt",
    "pbkdf2_hash",
    "simulate_breach",
    "CrackResult",
    "ecb_encrypt",
    "encrypt_image_ecb",
    "gen_key",
    "gcm_encrypt",
    "gcm_decrypt",
    "gcm_available",
    "generate_keypair",
    "RSAKey",
    "encrypt_bytes",
    "decrypt_bytes",
    "sign",
    "verify",
    "make_penguin",
]
