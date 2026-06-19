"""Synthetic corpus generation with inserted CANARIES.

The corpus mimics a log / record dataset: lots of structurally similar lines so a
char-level LM can actually learn the format. Into this background we insert a small
number of *canaries* — secret-looking strings of the form

    user alice secret code is 8401739265

A canary is a known phrase with a high-entropy random *secret* slot. To measure
memorization we compare the trained model's likelihood of the REAL inserted secret
against the likelihood of many random alternatives drawn from the same space
(Carlini et al., "The Secret Sharer", 2019).
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

# The vocabulary the char-LM sees. Fixed so secret indices are stable across runs.
VOCAB = sorted(set(string.ascii_lowercase + string.digits + " ."))
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}

# The secret is a fixed-length digit string -> randomness space = 10**SECRET_LEN.
SECRET_LEN = 10
SECRET_ALPHABET = string.digits

_NAMES = [
    "alice",
    "bob",
    "carol",
    "dave",
    "erin",
    "frank",
    "grace",
    "heidi",
    "ivan",
    "judy",
]
_ACTIONS = ["login", "logout", "upload", "download", "delete", "create", "update"]
_STATUS = ["ok", "fail", "retry", "queued", "done"]

# The canary template. "{secret}" is the high-entropy slot we will probe.
CANARY_TEMPLATE = "user {name} secret code is {secret}"


@dataclass
class Canary:
    """A single inserted secret and the line it lived in."""

    name: str
    secret: str
    text: str


def random_secret(rng: random.Random) -> str:
    return "".join(rng.choice(SECRET_ALPHABET) for _ in range(SECRET_LEN))


def make_canary(name: str, secret: str) -> str:
    return CANARY_TEMPLATE.format(name=name, secret=secret)


def _background_line(rng: random.Random) -> str:
    name = rng.choice(_NAMES)
    action = rng.choice(_ACTIONS)
    status = rng.choice(_STATUS)
    n = rng.randint(0, 9999)
    return f"event {name} {action} id {n:04d} status {status}."


def build_corpus(
    n_background: int = 4000,
    n_canaries: int = 4,
    canary_repeats: int = 16,
    seed: int = 42,
) -> tuple[str, list[Canary]]:
    """Build the training text and return (corpus_text, inserted_canaries).

    Each canary is inserted `canary_repeats` times (memorization grows with the
    number of insertions — that is exactly the effect we want to measure).
    """
    rng = random.Random(seed)
    lines = [_background_line(rng) for _ in range(n_background)]

    canaries: list[Canary] = []
    chosen_names = rng.sample(_NAMES, n_canaries)
    for name in chosen_names:
        secret = random_secret(rng)
        text = make_canary(name, secret)
        canaries.append(Canary(name=name, secret=secret, text=text))
        for _ in range(canary_repeats):
            lines.append(text)

    rng.shuffle(lines)
    corpus = "\n".join(lines) + "\n"
    return corpus, canaries


def encode(text: str) -> list[int]:
    """Map text -> indices, dropping any out-of-vocab characters."""
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode(idxs: list[int]) -> str:
    return "".join(IDX_TO_CHAR[i] for i in idxs)
