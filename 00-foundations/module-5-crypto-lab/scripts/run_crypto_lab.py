"""Run the whole crypto lab: hashing, AES ECB-vs-GCM, RSA + signatures.

Default path uses only stdlib + numpy/matplotlib. If `cryptography` is present
we also exercise the real AES-GCM / OAEP / PSS primitives and record it.

Outputs:
    results/figures/ecb_penguin.png        -- the money plot (plaintext vs ECB vs GCM)
    results/figures/hash_salt_demo.png     -- cracked-accounts bar chart
    results/metrics.json                   -- machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from crypto_lab import (  # noqa: E402
    asymmetric,
    decrypt_bytes,
    encrypt_bytes,
    encrypt_image_ecb,
    gcm_available,
    gcm_decrypt,
    gcm_encrypt,
    gen_key,
    gen_salt,
    generate_keypair,
    make_penguin,
    salted_hash,
    set_seed,
    sign,
    simulate_breach,
    unsalted_hash,
    verify,
)

FIG_DIR = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics.json"

# A few common-but-weak passwords used by several "users". Reuse is the point:
# identical passwords -> identical unsalted hashes.
USER_PASSWORDS = [
    "password", "123456", "password", "qwerty", "letmein",
    "123456", "dragon", "password", "monkey", "abc123",
    "qwerty", "iloveyou", "admin", "123456", "welcome",
]
WORDLIST = [
    "password", "123456", "qwerty", "letmein", "dragon",
    "monkey", "abc123", "iloveyou", "admin", "welcome", "football", "111111",
]


def run_hashing() -> dict:
    """Show that one rainbow table cracks unsalted hashes but not salted ones."""
    res = simulate_breach(USER_PASSWORDS, WORDLIST)

    # Illustrate digest collisions on reused passwords (unsalted) vs salts.
    pw = "password"
    same_unsalted = unsalted_hash(pw) == unsalted_hash(pw)
    s1, s2 = gen_salt(), gen_salt()
    diff_salted = salted_hash(pw, s1) != salted_hash(pw, s2)

    return {
        "n_users": res.n_users,
        "cracked_unsalted": res.n_cracked_unsalted,
        "cracked_salted": res.n_cracked_salted,
        "unsalted_reuse_collides": bool(same_unsalted),
        "salted_breaks_reuse": bool(diff_salted),
    }


def plot_hashing(stats: dict) -> str:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    labels = ["Unsalted\nSHA-256", "Salted\nSHA-256"]
    vals = [stats["cracked_unsalted"], stats["cracked_salted"]]
    bars = ax.bar(labels, vals, color=["#c0392b", "#27ae60"])
    ax.set_ylabel("accounts cracked by one rainbow table")
    ax.set_title(f"Salting defeats precomputed tables (n={stats['n_users']} users)")
    ax.set_ylim(0, stats["n_users"])
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", va="bottom")
    fig.tight_layout()
    path = FIG_DIR / "hash_salt_demo.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def run_aes(img: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """Encrypt the penguin with ECB; measure how much structure leaks."""
    key = gen_key()
    ecb_img = encrypt_image_ecb(img, key)

    # Quantify leakage: fraction of unique 16-byte blocks. Low uniqueness in the
    # ECB ciphertext == repeated plaintext blocks survived == structure leaked.
    def block_uniqueness(arr: np.ndarray) -> float:
        b = arr.tobytes()
        b = b[: (len(b) // 16) * 16]
        blocks = {b[i : i + 16] for i in range(0, len(b), 16)}
        total = len(b) // 16
        return len(blocks) / total if total else 1.0

    stats = {
        "plaintext_block_uniqueness": round(block_uniqueness(img), 4),
        "ecb_block_uniqueness": round(block_uniqueness(ecb_img), 4),
        "gcm_used": False,
        "gcm_nonce_randomizes": None,
        "gcm_detects_tamper": None,
    }

    gcm_render = None
    if gcm_available():
        # GCM on the same bytes: equal plaintext -> unrelated ciphertext.
        nonce1, ct1 = gcm_encrypt(img.tobytes(), key)
        nonce2, ct2 = gcm_encrypt(img.tobytes(), key)
        # Render GCM ciphertext (drop the 16-byte tag) as a same-shape image.
        h, w = img.shape
        body = ct1[: h * w]
        if len(body) < h * w:
            body = body + bytes(h * w - len(body))
        gcm_render = np.frombuffer(body, dtype=np.uint8).reshape(h, w)

        # Tamper detection: flip a ciphertext byte, decryption must fail.
        tampered = bytearray(ct1)
        tampered[0] ^= 0x01
        try:
            gcm_decrypt(nonce1, bytes(tampered), key)
            detects = False
        except Exception:  # noqa: BLE001 - any auth failure means it caught it
            detects = True

        stats.update(
            gcm_used=True,
            gcm_nonce_randomizes=bool(ct1 != ct2),
            gcm_detects_tamper=bool(detects),
            gcm_block_uniqueness=round(block_uniqueness(gcm_render), 4),
        )

    return stats, ecb_img, gcm_render


def plot_aes(img, ecb_img, gcm_img) -> str:
    n = 3 if gcm_img is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("plaintext")
    axes[1].imshow(ecb_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("AES-ECB (bad)\noutline leaks")
    if gcm_img is not None:
        axes[2].imshow(gcm_img, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("AES-GCM (good)\nlooks like noise")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("The ECB penguin: encryption mode matters", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "ecb_penguin.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def run_rsa() -> dict:
    """Textbook RSA encrypt/decrypt + sign/verify, with a forgery check."""
    key = generate_keypair(bits=1024)
    msg = b"attack at dawn"

    c = encrypt_bytes(msg, key.public)
    recovered = decrypt_bytes(c, key, len(msg))
    enc_ok = recovered == msg

    sig = sign(msg, key)
    verify_ok = verify(msg, sig, key.public)
    # A modified message must NOT verify against the original signature.
    forgery_rejected = not verify(b"attack at dusk!", sig, key.public)

    stats = {
        "modulus_bits": key.n.bit_length(),
        "encrypt_decrypt_roundtrip": bool(enc_ok),
        "signature_verifies": bool(verify_ok),
        "tampered_message_rejected": bool(forgery_rejected),
        "oaep_pss_used": False,
    }
    if asymmetric.cryptography_available():
        stats.update(
            oaep_pss_used=True,
            real_oaep_roundtrip=bool(asymmetric.oaep_roundtrip(msg)),
            real_pss_sign_verify=bool(asymmetric.pss_sign_verify(msg)),
        )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the crypto basics + misuse lab.")
    ap.add_argument("--size", type=int, default=128, help="penguin image size (px)")
    args = ap.parse_args()

    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    img = make_penguin(args.size)

    hashing = run_hashing()
    aes_stats, ecb_img, gcm_img = run_aes(img)
    rsa_stats = run_rsa()

    fig_hash = plot_hashing(hashing)
    fig_aes = plot_aes(img, ecb_img, gcm_img)

    enhanced = bool(aes_stats["gcm_used"] and rsa_stats["oaep_pss_used"])
    summary = (
        f"Unsalted SHA-256 let one rainbow table crack "
        f"{hashing['cracked_unsalted']}/{hashing['n_users']} accounts; salting cut that to "
        f"{hashing['cracked_salted']}. AES-ECB leaked image structure "
        f"(block uniqueness {aes_stats['ecb_block_uniqueness']} vs plaintext "
        f"{aes_stats['plaintext_block_uniqueness']}). Textbook RSA "
        f"({rsa_stats['modulus_bits']}-bit) round-tripped and verified signatures, "
        f"rejecting a tampered message. "
        + ("Enhanced path (cryptography: GCM/OAEP/PSS) ran." if enhanced
           else "Enhanced path skipped (cryptography not installed).")
    )

    metrics = {
        "project": "module-5-crypto-lab",
        "summary": summary,
        "enhanced_path_ran": enhanced,
        "hashing": hashing,
        "aes": aes_stats,
        "rsa": rsa_stats,
        "figures": [fig_aes, fig_hash],
    }
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(summary)
    print(f"wrote {METRICS.relative_to(ROOT)} and {len(metrics['figures'])} figures")


if __name__ == "__main__":
    main()
