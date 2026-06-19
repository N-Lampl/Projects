"""A tiny dependency-free HTTP client + a deterministic MOCK Juice Shop.

Two ways to run every exploit in this lab:

* **Real path** (needs docker): point at the live Juice Shop container on localhost:3000.
  Uses `requests` if installed, else falls back to stdlib `urllib`.
* **Offline path** (the DEFAULT, no docker): a `MockJuiceShop` that reproduces the *vulnerable
  behaviour* of the relevant endpoints — SQL-injection login bypass, broken access control,
  weak JWT, reflected XSS echo. It lets the exploit scripts and CI tests run and prove their
  logic with ZERO infrastructure, then the identical code runs unchanged against the container.

The mock is intentionally faithful to Juice Shop's documented flaws so the writeups stay honest.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Response envelope (so real + mock paths share one shape)
# ---------------------------------------------------------------------------


@dataclass
class Response:
    status: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.body)


# ---------------------------------------------------------------------------
# Real client (urllib; optional requests upgrade)
# ---------------------------------------------------------------------------


class HttpClient:
    """Minimal GET/POST over the live container. No third-party deps required."""

    def __init__(self, base_url: str, timeout: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _do(self, method: str, path: str, body: Any, headers: dict[str, str]) -> Response:
        url = self.base_url + path
        data = None
        hdrs = dict(headers)
        if body is not None:
            data = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (local target)
                return Response(r.status, r.read().decode("utf-8", "replace"), dict(r.headers))
        except urllib.error.HTTPError as e:
            return Response(e.code, e.read().decode("utf-8", "replace"), dict(e.headers))

    def get(self, path: str, headers: dict[str, str] | None = None) -> Response:
        return self._do("GET", path, None, headers or {})

    def post(self, path: str, body: Any, headers: dict[str, str] | None = None) -> Response:
        return self._do("POST", path, body, headers or {})


# ---------------------------------------------------------------------------
# Deterministic offline MOCK of the vulnerable endpoints
# ---------------------------------------------------------------------------


def _b64url(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def _make_jwt(payload: dict, alg: str = "HS256", sig: str = "s3cr3t") -> str:
    header = _b64url({"typ": "JWT", "alg": alg})
    body = _b64url(payload)
    if alg == "none":
        return f"{header}.{body}."
    return f"{header}.{body}.{base64.urlsafe_b64encode(sig.encode()).decode().rstrip('=')}"


class MockJuiceShop:
    """Reproduces the *exploitable behaviour* of select Juice Shop endpoints, offline.

    It is deliberately vulnerable in the same ways the real app is, so the exploit scripts
    succeed here and (unchanged) against the container. It is NOT a full reimplementation.
    """

    # A toy user table. The admin row is the SQLi prize.
    USERS = [
        {"id": 1, "email": "admin@juice-sh.op", "password": "admin123", "role": "admin"},
        {"id": 2, "email": "jim@juice-sh.op", "password": "ncc-1701", "role": "customer"},
    ]

    def __init__(self) -> None:
        self.base_url = "mock://juice-shop"
        # baskets[basket_id] -> owner user id  (A01 broken access control)
        self.baskets = {1: 1, 2: 2}

    # -- routing -----------------------------------------------------------
    def get(self, path: str, headers: dict[str, str] | None = None) -> Response:
        headers = headers or {}
        if path.startswith("/rest/basket/"):
            return self._basket(path, headers)
        if path == "/ftp/" or path.startswith("/ftp"):
            return self._ftp(path)
        if path.startswith("/search?") or path.startswith("/rest/products/search"):
            return self._search(path)
        return Response(404, json.dumps({"error": "not found"}))

    def post(self, path: str, body: Any, headers: dict[str, str] | None = None) -> Response:
        if path == "/rest/user/login":
            return self._login(body or {})
        return Response(404, json.dumps({"error": "not found"}))

    # -- A03: SQL injection login bypass -----------------------------------
    def _login(self, body: dict) -> Response:
        email = str(body.get("email", ""))
        password = str(body.get("password", ""))
        # Juice Shop builds: SELECT * FROM Users WHERE email='<email>' AND password='<hash>'
        # A trailing "' OR 1=1--" comments out the password check and returns the first row.
        injection = re.search(r"'\s*or\s*1\s*=\s*1\s*(--|#)", email, re.IGNORECASE)
        if injection:
            user = self.USERS[0]  # first row == admin, classic Juice Shop bypass
            token = _make_jwt({"id": user["id"], "email": user["email"], "role": user["role"]})
            return Response(200, json.dumps({"authentication": {"token": token, "umail": user["email"]}}))
        for u in self.USERS:
            if u["email"] == email and u["password"] == password:
                token = _make_jwt({"id": u["id"], "email": u["email"], "role": u["role"]})
                return Response(200, json.dumps({"authentication": {"token": token, "umail": u["email"]}}))
        return Response(401, json.dumps({"error": "Invalid email or password."}))

    # -- A01: broken access control (IDOR on basket) -----------------------
    def _basket(self, path: str, headers: dict[str, str]) -> Response:
        try:
            basket_id = int(path.rsplit("/", 1)[-1])
        except ValueError:
            return Response(400, json.dumps({"error": "bad id"}))
        # The vuln: the server NEVER checks that the JWT subject owns this basket.
        owner = self.baskets.get(basket_id)
        if owner is None:
            return Response(404, json.dumps({"error": "no basket"}))
        return Response(200, json.dumps({"data": {"id": basket_id, "userId": owner, "items": []}}))

    # -- A05: security misconfiguration (directory listing / sensitive files)
    def _ftp(self, path: str) -> Response:
        if path == "/ftp/" or path == "/ftp":
            listing = ["acquisitions.md", "coupons_2013.md.bak", "package.json.bak"]
            return Response(200, json.dumps({"listing": listing}))
        if path.endswith("acquisitions.md"):
            return Response(200, "CONFIDENTIAL acquisitions roadmap ...")
        # Poison-null-byte trick historically bypassed the extension whitelist.
        return Response(403, json.dumps({"error": "Only .md and .pdf allowed"}))

    # -- A03: reflected XSS (search echoes the q parameter unescaped) -------
    def _search(self, path: str) -> Response:
        q = ""
        if "?" in path:
            qs = path.split("?", 1)[1]
            for pair in qs.split("&"):
                if pair.startswith("q="):
                    q = urllib.parse.unquote_plus(pair[2:])
        # Vulnerable: query is reflected into the HTML body verbatim.
        html = f"<h2>Search results for: {q}</h2>"
        return Response(200, html, {"Content-Type": "text/html"})


def make_client(base_url: str, offline: bool):
    """Return a live HttpClient or the offline MockJuiceShop, behind one interface."""
    return MockJuiceShop() if offline else HttpClient(base_url)


__all__ = ["Response", "HttpClient", "MockJuiceShop", "make_client", "_make_jwt"]
