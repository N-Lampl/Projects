"""Seeded synthetic car-review generator — the DEFAULT offline / CI data path.

Real-world drop-in: the HuggingFace dataset ``florentgbelidji/car-reviews`` (see
``data.py`` / ``data/README.md``). This module fabricates a small review table
with the *same columns* and a **planted, recoverable signal** so tests can assert
the pipeline works with zero downloads:

    * each (make, model) has a hidden "quality" in [0, 1];
    * ``Rating`` (1-5) is drawn to correlate with that quality;
    * the ``Review`` text is assembled from positive/negative templates that also
      inject aspect keywords (performance, comfort, reliability, price, fuel
      economy, safety) so aspect extraction and sentiment both have signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["Review_Date", "Author_Name", "Vehicle_Title", "Review_Title", "Review", "Rating"]

# (make, model, body/trim, hidden quality). Good brands score high, bad low.
_FLEET: list[tuple[str, str, str, float]] = [
    ("Lexus", "ES", "Sedan", 0.92),
    ("Toyota", "Camry", "Sedan LE", 0.86),
    ("Toyota", "Prius", "Hatchback", 0.83),
    ("Honda", "Accord", "Sedan EX", 0.80),
    ("Honda", "Civic", "Coupe", 0.72),
    ("Subaru", "Outback", "Wagon", 0.70),
    ("Ford", "Focus", "Sedan SE", 0.55),
    ("Chevrolet", "Malibu", "Sedan LT", 0.52),
    ("Dodge", "Journey", "SUV", 0.40),
    ("Fiat", "500", "Hatchback", 0.33),
    ("Suzuki", "Forenza", "Sedan", 0.28),
    ("Land Rover", "Range Rover", "SUV", 0.48),
]

# Aspect -> (positive clause, negative clause). Keywords match aspects.ASPECT_LEXICON.
_ASPECT_TEMPLATES: dict[str, tuple[str, str]] = {
    "performance": (
        "The engine has plenty of power and acceleration feels quick.",
        "The engine is sluggish and acceleration is weak.",
    ),
    "comfort": (
        "The seats are comfortable and the ride is quiet on long trips.",
        "The seats are cramped and the ride is noisy and harsh.",
    ),
    "reliability": (
        "It has been reliable with no problems and dependable service.",
        "It is unreliable and broke down with an expensive repair.",
    ),
    "price": (
        "Great value for the price and the resale value held up well.",
        "It felt overpriced and was not worth the money.",
    ),
    "fuel_economy": (
        "The fuel economy is excellent and the gas mileage saves money.",
        "The fuel economy is terrible and the gas mileage is disappointing.",
    ),
    "safety": (
        "It feels safe with strong brakes and good stability.",
        "The brakes are poor and it feels unsafe in an emergency.",
    ),
}

_POS_OPENERS = [
    "I absolutely love this car.",
    "This has been a fantastic vehicle.",
    "Excellent car, highly recommend it.",
    "Very happy with this purchase.",
]
_NEG_OPENERS = [
    "I regret buying this car.",
    "This has been a terrible vehicle.",
    "Awful experience, would not recommend.",
    "Very disappointed with this purchase.",
]


def make_reviews(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic review table with a planted signal."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    aspects = list(_ASPECT_TEMPLATES)

    for i in range(n):
        make, model, trim, quality = _FLEET[rng.integers(0, len(_FLEET))]
        # Rating correlates with hidden quality (+ noise), clipped to 1..5.
        rating = int(np.clip(round(1 + 4 * quality + rng.normal(0, 0.5)), 1, 5))
        positive = rating >= 4 or (rating == 3 and quality >= 0.5)

        opener = (_POS_OPENERS if positive else _NEG_OPENERS)[rng.integers(0, len(_POS_OPENERS))]
        # Mention 3 random aspects, phrased by sentiment.
        chosen = rng.choice(aspects, size=3, replace=False)
        clauses = [_ASPECT_TEMPLATES[a][0 if positive else 1] for a in chosen]
        review = " ".join([opener, *clauses])

        year = int(rng.integers(2005, 2020))
        rows.append(
            {
                "Review_Date": f"{year}-{int(rng.integers(1, 13)):02d}-01",
                "Author_Name": f"user_{i:04d}",
                "Vehicle_Title": f"{year} {make} {model} {trim}",
                "Review_Title": opener,
                "Review": review,
                "Rating": rating,
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)
