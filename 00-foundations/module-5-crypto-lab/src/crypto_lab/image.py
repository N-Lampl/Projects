"""Generate the small synthetic image used in the ECB-penguin demo.

We build a blocky "penguin/logo"-style grayscale image as a numpy array with
large flat regions. Flat regions are exactly what makes ECB leak: many
identical 16-byte plaintext blocks -> many identical ciphertext blocks, so the
silhouette survives encryption.

No PIL required for the default path — the array is rendered with matplotlib in
the script. (PIL is an optional convenience, never imported here.)
"""

from __future__ import annotations

import numpy as np


def make_penguin(size: int = 128) -> np.ndarray:
    """Return a (size, size) uint8 grayscale silhouette with flat regions.

    Deterministic given `size`. Big constant-color areas are intentional.
    """
    img = np.full((size, size), 230, dtype=np.uint8)  # light background

    cy, cx = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]

    # Body: a filled ellipse (one big flat dark region).
    body = ((yy - cy) / (size * 0.34)) ** 2 + ((xx - cx) / (size * 0.26)) ** 2 <= 1.0
    img[body] = 30

    # Belly: a lighter ellipse inside the body (another flat region).
    belly = ((yy - (cy + size * 0.05)) / (size * 0.24)) ** 2 + (
        (xx - cx) / (size * 0.16)
    ) ** 2 <= 1.0
    img[belly] = 235

    # Head: a flat dark circle on top.
    head = ((yy - (cy - size * 0.30)) ** 2 + (xx - cx) ** 2) <= (size * 0.14) ** 2
    img[head] = 30

    # Eyes: two small light dots.
    for dx in (-int(size * 0.05), int(size * 0.05)):
        eye = ((yy - (cy - size * 0.32)) ** 2 + (xx - (cx + dx)) ** 2) <= (size * 0.018) ** 2
        img[eye] = 245

    # Beak: small flat triangle-ish block.
    beak = (np.abs(yy - (cy - size * 0.27)) < size * 0.025) & (
        np.abs(xx - cx) < size * 0.05
    )
    img[beak] = 150

    return img
