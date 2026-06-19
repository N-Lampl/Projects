"""Tiny self-contained Gaussian log-pdf (so we don't depend on scipy).

LiRA only needs the *log* density of a normal for its likelihood ratio:

    log N(x; mu, sigma) = -0.5*((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
"""

from __future__ import annotations

import numpy as np

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def norm_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Elementwise log of the Gaussian density. `sigma` must be > 0."""
    x = np.asarray(x, dtype=np.float64)
    z = (x - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - _LOG_SQRT_2PI
