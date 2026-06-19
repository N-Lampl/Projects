"""URL data sources.

Default (offline, deterministic): a SYNTHETIC generator that builds benign-looking
vs phishing-looking URLs from controllable distributions, so the whole pipeline runs
with zero downloads. The phishing class is biased toward the lexical "tells" the
literature flags (IP hosts, long hostnames, many dots/hyphens, @, punycode, deep
paths, suspicious TLDs, brand-keyword stuffing) but with overlap so the task is real.

Optional (enhanced): PhiUSIIL via ucimlrepo, imported lazily — see load_phiusiil().
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

# --- vocabulary for the synthetic generator -------------------------------------------------

_BENIGN_BRANDS = [
    "google", "amazon", "wikipedia", "github", "stackoverflow", "nytimes",
    "reddit", "microsoft", "apple", "netflix", "spotify", "linkedin",
    "dropbox", "shopify", "cloudflare", "mozilla", "python", "ubuntu",
]
_BENIGN_TLDS = ["com", "org", "net", "io", "edu", "gov", "co.uk", "de"]
_BENIGN_WORDS = [
    "about", "help", "docs", "blog", "news", "login", "account", "search",
    "products", "support", "pricing", "careers", "contact", "home", "user",
]

# brands abused by phishers + suspicious infrastructure tokens
_TARGET_BRANDS = ["paypal", "apple", "microsoft", "amazon", "netflix", "bankofamerica", "wellsfargo"]
_PHISH_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "click", "zip", "review"]
_PHISH_WORDS = [
    "secure", "verify", "update", "account", "login", "confirm", "signin",
    "webscr", "billing", "suspended", "unlock", "validation", "support-team",
]
_HOST_FILLER = "abcdefghijklmnopqrstuvwxyz0123456789"


def _rand_token(rng: random.Random, lo: int, hi: int) -> str:
    n = rng.randint(lo, hi)
    return "".join(rng.choice(_HOST_FILLER) for _ in range(n))


def _make_benign(rng: random.Random) -> str:
    scheme = rng.choice(["https", "https", "https", "http"])  # mostly https
    brand = rng.choice(_BENIGN_BRANDS)
    sub = rng.choice(["", "", "www.", "www.", "docs.", "blog.", "api."])
    tld = rng.choice(_BENIGN_TLDS)
    host = f"{sub}{brand}.{tld}"
    n_path = rng.randint(0, 2)
    path = "/".join(rng.choice(_BENIGN_WORDS) for _ in range(n_path))
    url = f"{scheme}://{host}/{path}".rstrip("/")
    if rng.random() < 0.15:  # occasional benign query string
        url += f"?id={rng.randint(1, 9999)}"
    # ~12% of benign URLs legitimately look a bit "phishy" (long hyphenated host,
    # a security word, a deep path) so the classes overlap and AUC isn't a trivial 1.0.
    if rng.random() < 0.12:
        url += f"/{rng.choice(['account', 'login', 'secure', 'update'])}-{_rand_token(rng, 4, 10)}"
    return url


def _make_phish(rng: random.Random) -> str:
    scheme = rng.choice(["http", "http", "https"])  # phishing skews http
    brand = rng.choice(_TARGET_BRANDS)
    style = rng.random()

    if style < 0.22:  # raw IP host
        host = ".".join(str(rng.randint(1, 255)) for _ in range(4))
        if rng.random() < 0.3:
            host += f":{rng.choice([8080, 8000, 443, 8888])}"
    elif style < 0.45:  # brand stuffed into a long hyphenated subdomain on a junk domain
        host = f"{brand}-{rng.choice(_PHISH_WORDS)}.{_rand_token(rng, 6, 14)}.{rng.choice(_PHISH_TLDS)}"
    elif style < 0.62:  # punycode / homograph-ish
        host = f"xn--{_rand_token(rng, 6, 12)}.{rng.choice(_PHISH_TLDS)}"
    elif style < 0.78:  # @ trick: looks like a brand but routes to attacker host
        host = f"{brand}.com@{_rand_token(rng, 8, 16)}.{rng.choice(_PHISH_TLDS)}"
    else:  # many dots / deep subdomains
        host = ".".join([brand] + [_rand_token(rng, 3, 8) for _ in range(rng.randint(2, 4))])
        host += f".{rng.choice(_PHISH_TLDS)}"

    # ~10% of phishing URLs are "stealthy": short, https, a clean-ish host on a
    # normal TLD — these overlap with benign and keep the task from being trivial.
    if rng.random() < 0.10:
        scheme = "https"
        host = f"{brand}.{rng.choice(_BENIGN_TLDS)}"

    n_path = rng.randint(1, 5)
    path = "/".join(rng.choice(_PHISH_WORDS + [brand]) for _ in range(n_path))
    url = f"{scheme}://{host}/{path}"
    if rng.random() < 0.55:  # phishing loves long query strings / tokens
        url += f"?{rng.choice(_PHISH_WORDS)}={_rand_token(rng, 12, 28)}&id={rng.randint(1, 99999)}"
    return url


def make_synthetic(n: int = 4000, phish_frac: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Generate a labeled DataFrame of synthetic URLs.

    Returns columns: url (str), label (int; 1 = phishing, 0 = benign).
    Deterministic given `seed`.
    """
    rng = random.Random(seed)
    n_phish = int(round(n * phish_frac))
    rows = []
    for _ in range(n_phish):
        rows.append((_make_phish(rng), 1))
    for _ in range(n - n_phish):
        rows.append((_make_benign(rng), 0))
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=["url", "label"])
    # drop any accidental dup (url, label) pairs for a cleaner signal
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df


def train_test_split_df(
    df: pd.DataFrame, test_frac: float = 0.25, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified-ish split (shuffle then slice)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    train = df.iloc[idx[:cut]].reset_index(drop=True)
    test = df.iloc[idx[cut:]].reset_index(drop=True)
    return train, test


def load_phiusiil(max_rows: int | None = 8000, seed: int = 42) -> pd.DataFrame:
    """OPTIONAL real-data path: PhiUSIIL Phishing URL dataset via ucimlrepo.

    Requires `pip install ucimlrepo` and network access on first call. Imported
    lazily so this module still loads offline. Returns the same (url, label) shape.
    Raises ImportError/RuntimeError with a helpful message if unavailable.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:  # pragma: no cover - optional path
        raise ImportError(
            "ucimlrepo not installed. `pip install ucimlrepo` for the real-data path."
        ) from e

    ds = fetch_ucirepo(id=967)  # PhiUSIIL Phishing URL Dataset
    X = ds.data.features
    y = ds.data.targets
    if "URL" not in X.columns:  # pragma: no cover - schema guard
        raise RuntimeError("Expected a 'URL' column in PhiUSIIL features.")
    label_col = y.columns[0]
    df = pd.DataFrame({"url": X["URL"].astype(str), "label": y[label_col].astype(int)})
    # PhiUSIIL uses 1=legit, 0=phishing; flip so 1=phishing to match this project.
    df["label"] = 1 - df["label"]
    df = df.dropna().drop_duplicates(subset=["url"]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df
