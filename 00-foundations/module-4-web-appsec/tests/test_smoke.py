"""Fast smoke tests (run in CI) for the OWASP Top 10 lab.

The mock reproduces the vulnerable behaviour, so these tests exercise the REAL exploit logic
with no docker. One @slow test runs the end-to-end script and checks the artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from juiceshop_lab import (
    MockJuiceShop,
    exposed_ftp_listing,
    forged_jwt_none_alg,
    idor_basket,
    reflected_xss_search,
    run_all,
    set_seed,
    sqli_login_bypass,
)
from juiceshop_lab.utils import require_local_target

PROJECT = Path(__file__).resolve().parents[1]


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_require_local_target_blocks_remote():
    require_local_target("http://localhost:3000")  # ok
    require_local_target("http://127.0.0.1:3000")  # ok
    with pytest.raises(SystemExit):
        require_local_target("http://example.com")


def test_sqli_login_bypass_returns_admin():
    f = sqli_login_bypass(MockJuiceShop())
    assert f["success"] is True
    assert f["evidence"]["obtained_role"] == "admin"
    assert "OR 1=1" in f["payload"]["email"]


def test_login_rejects_wrong_password():
    # The mock must NOT be a pushover: real creds work, bad ones fail (no false positives).
    c = MockJuiceShop()
    bad = c.post("/rest/user/login", {"email": "jim@juice-sh.op", "password": "wrong"})
    assert bad.status == 401
    good = c.post("/rest/user/login", {"email": "jim@juice-sh.op", "password": "ncc-1701"})
    assert good.status == 200


def test_reflected_xss_is_unescaped():
    f = reflected_xss_search(MockJuiceShop())
    assert f["success"] is True
    assert f["evidence"]["payload_reflected_unescaped"] is True


def test_idor_reads_other_users_basket():
    f = idor_basket(MockJuiceShop(), victim_basket_id=1, attacker_id=2)
    assert f["success"] is True
    assert f["evidence"]["leaked_owner_userId"] == 1  # not the attacker (2)


def test_exposed_ftp_lists_sensitive_files():
    f = exposed_ftp_listing(MockJuiceShop())
    assert f["success"] is True
    assert any(name.endswith(".bak") for name in f["evidence"]["files"])


def test_forged_jwt_grants_admin():
    f = forged_jwt_none_alg(MockJuiceShop())
    assert f["success"] is True
    assert f["evidence"]["decoded_payload"]["role"] == "admin"


def test_run_all_covers_expected_categories():
    findings = run_all(MockJuiceShop())
    cats = {f["owasp"].split(":")[0] for f in findings}
    assert {"A01", "A03", "A05", "A07"} <= cats
    assert all("fix" in f and f["fix"] for f in findings)  # every break has a fix


@pytest.mark.slow
def test_end_to_end_script_writes_artifacts(tmp_path):
    """Run scripts/run_lab.py (offline) and confirm findings.json + metrics.json appear."""
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts" / "run_lab.py")],
        check=True,
        cwd=PROJECT,
    )
    metrics = json.loads((PROJECT / "results" / "metrics.json").read_text())
    findings = json.loads((PROJECT / "results" / "findings.json").read_text())
    assert metrics["n_confirmed"] == 5
    assert metrics["n_categories"] >= 4
    assert len(findings["findings"]) == 5
