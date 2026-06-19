"""Lexical URL feature extraction — no network lookups, pure string analysis.

These are the classic, cheap, explainable signals used by URL-based phishing
detectors (Ma et al. 2009; Sahingoz et al. 2019): length, character counts,
host structure, suspicious tokens, and entropy. Everything is computed from the
URL string alone, so the detector works at wire speed with no DNS/WHOIS calls.
"""

from __future__ import annotations

import math
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SUSPICIOUS_WORDS = (
    "secure", "account", "update", "login", "verify", "confirm", "signin",
    "webscr", "banking", "billing", "suspended", "unlock", "validation",
)
_SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd")

# stable, ordered feature names (so the model's coefficients are interpretable)
FEATURE_NAMES = [
    "url_len",
    "host_len",
    "path_len",
    "query_len",
    "n_dots",
    "n_hyphens",
    "n_slashes",
    "n_digits",
    "n_special",
    "n_subdomains",
    "has_ip_host",
    "has_at",
    "has_punycode",
    "is_https",
    "has_port",
    "digit_ratio",
    "host_entropy",
    "n_suspicious_words",
    "is_shortener",
    "tld_len",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_one(url: str) -> dict[str, float]:
    """Compute the lexical feature dict for a single URL string."""
    url = url.strip()
    if "://" not in url:  # urlparse needs a scheme to populate netloc
        parsed = urlparse("http://" + url)
        is_https = 0
    else:
        parsed = urlparse(url)
        is_https = int(parsed.scheme.lower() == "https")

    netloc = parsed.netloc
    host = netloc.split("@")[-1].split(":")[0]  # strip userinfo + port
    path = parsed.path
    query = parsed.query

    digits = sum(c.isdigit() for c in url)
    special = sum(not c.isalnum() and c not in "/:.?=&-_" for c in url)
    tld = host.rsplit(".", 1)[-1] if "." in host else ""

    return {
        "url_len": float(len(url)),
        "host_len": float(len(host)),
        "path_len": float(len(path)),
        "query_len": float(len(query)),
        "n_dots": float(url.count(".")),
        "n_hyphens": float(url.count("-")),
        "n_slashes": float(url.count("/")),
        "n_digits": float(digits),
        "n_special": float(special),
        "n_subdomains": float(max(host.count(".") - 1, 0)),
        "has_ip_host": float(bool(_IP_RE.match(host))),
        "has_at": float("@" in netloc),
        "has_punycode": float("xn--" in host.lower()),
        "is_https": float(is_https),
        "has_port": float(":" in netloc.split("@")[-1]),
        "digit_ratio": float(digits / len(url)) if url else 0.0,
        "host_entropy": _entropy(host),
        "n_suspicious_words": float(sum(w in url.lower() for w in _SUSPICIOUS_WORDS)),
        "is_shortener": float(any(s in url.lower() for s in _SHORTENERS)),
        "tld_len": float(len(tld)),
    }


def extract_features(urls: list[str] | pd.Series) -> np.ndarray:
    """Vectorize a list/Series of URLs into an (n, n_features) float array."""
    rows = [extract_one(u) for u in urls]
    return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=np.float64)
