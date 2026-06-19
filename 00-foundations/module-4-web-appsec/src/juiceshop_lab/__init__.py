"""OWASP Top 10 break-and-fix lab against a LOCAL OWASP Juice Shop container.

Public API:
    set_seed, get_device, require_local_target   -- portfolio-standard helpers + guardrail
    make_client, HttpClient, MockJuiceShop        -- live target or deterministic offline mock
    run_all                                       -- run all five OWASP probes
    individual probes                             -- sqli_login_bypass, reflected_xss_search,
                                                     idor_basket, exposed_ftp_listing,
                                                     forged_jwt_none_alg
"""

from .client import HttpClient, MockJuiceShop, Response, make_client
from .exploits import (
    SQLI_LOGIN_PAYLOAD,
    XSS_PAYLOAD,
    exposed_ftp_listing,
    forged_jwt_none_alg,
    idor_basket,
    reflected_xss_search,
    run_all,
    sqli_login_bypass,
)
from .utils import get_device, require_local_target, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "require_local_target",
    "make_client",
    "HttpClient",
    "MockJuiceShop",
    "Response",
    "run_all",
    "sqli_login_bypass",
    "reflected_xss_search",
    "idor_basket",
    "exposed_ftp_listing",
    "forged_jwt_none_alg",
    "SQLI_LOGIN_PAYLOAD",
    "XSS_PAYLOAD",
]
