"""Synthetic domain generation: pronounceable BENIGN vs high-entropy DGA-style.

Everything here is self-contained — no datasets are downloaded. Benign domains
are built from English-like syllables (consonant-vowel patterns) so they read as
pronounceable. DGA domains imitate real malware families:

  * ``random``  -- uniform random letters/digits (e.g. Cryptolocker-style)
  * ``hexish``  -- long hex-ish strings (e.g. some Necurs variants)
  * ``dict``    -- concatenated dictionary words with no separators (matsnu-style)

The ``dict`` family is deliberately *harder*: it has lower character entropy than
the random families, which is exactly why we need n-gram features and not just an
entropy threshold.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

# small, fixed pools keep generation deterministic and dependency-free
_CONSONANTS = "bcdfghjklmnprstvw"
_VOWELS = "aeiou"
_TLDS = ["com", "net", "org", "io", "co", "info", "biz"]
_HEX = "0123456789abcdef"
_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"

# pronounceable benign roots
_WORDS = [
    "cloud", "shop", "data", "green", "swift", "north", "media", "prime",
    "logic", "vault", "spark", "river", "stone", "bright", "quick", "tiger",
    "maple", "ocean", "pixel", "nova", "atlas", "delta", "orbit", "ember",
    "frost", "glow", "harbor", "jet", "koala", "lunar", "mango", "noble",
]

# a DISJOINT word pool for the dictionary-DGA family. Same length/entropy
# profile as benign, but a different (rarer) vocabulary -> only character
# n-gram transitions, not entropy, can tell them apart. This is the realistic
# matsnu/suppobox case that defeats a simple entropy threshold.
_DGA_WORDS = [
    "anvil", "brick", "crypt", "dwarf", "fjord", "glyph", "hymn", "ivory",
    "jolt", "knack", "lymph", "myrrh", "nymph", "plumb", "quartz", "rhythm",
    "sphinx", "twelfth", "vex", "waltz", "xylem", "yacht", "zealot", "blitz",
    "crux", "flux", "gnarl", "khaki", "mauve", "ochre", "psalm", "wrath",
]


def _syllable(rng: random.Random) -> str:
    """A simple consonant-vowel(-consonant) syllable -> pronounceable."""
    s = rng.choice(_CONSONANTS) + rng.choice(_VOWELS)
    if rng.random() < 0.4:
        s += rng.choice(_CONSONANTS)
    return s


def gen_benign(rng: random.Random) -> str:
    """A pronounceable, human-readable second-level domain label."""
    if rng.random() < 0.5:
        # word (+ optional word/short suffix) -> "cloudshop", "dataprime7"
        label = rng.choice(_WORDS)
        if rng.random() < 0.5:
            label += rng.choice(_WORDS)
        elif rng.random() < 0.3:
            label += str(rng.randint(1, 99))
    else:
        # 2-3 syllables -> "toveranu", "kelibo"
        label = "".join(_syllable(rng) for _ in range(rng.randint(2, 3)))
    return label


def gen_dga(rng: random.Random, family: str | None = None) -> tuple[str, str]:
    """A DGA-style label. Returns (label, family)."""
    family = family or rng.choice(["random", "hexish", "dict"])
    if family == "random":
        n = rng.randint(12, 22)
        label = "".join(rng.choice(_ALNUM) for _ in range(n))
    elif family == "hexish":
        n = rng.randint(16, 32)
        label = "".join(rng.choice(_HEX) for _ in range(n))
    else:  # dict: jam 2-3 words together, no separators -> low-entropy but unusual
        # overlaps benign in length/entropy on purpose: only n-gram transitions
        # (rare word-boundary char pairs) reliably separate it.
        label = "".join(rng.choice(_DGA_WORDS) for _ in range(rng.randint(2, 3)))
    return label, family


def make_dataset(n_per_class: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Build a balanced benign/DGA dataset.

    Columns: domain (with TLD), label (0=benign, 1=dga), family.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n_per_class):
        label = gen_benign(rng)
        rows.append((f"{label}.{rng.choice(_TLDS)}", 0, "benign"))
    for _ in range(n_per_class):
        label, fam = gen_dga(rng)
        rows.append((f"{label}.{rng.choice(_TLDS)}", 1, fam))

    df = pd.DataFrame(rows, columns=["domain", "label", "family"])
    # shuffle deterministically
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_test_split_df(
    df: pd.DataFrame, test_frac: float = 0.25, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic split that preserves class balance well enough at this size."""
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(df)) >= test_frac
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)
