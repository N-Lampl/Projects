"""Parse the free-text ``Vehicle_Title`` into year / make / model / trim.

The dataset's ``Vehicle_Title`` looks like ``"1997 Toyota Previa Minivan LE 3dr
Minivan AWD"`` - i.e. ``<year> <make> <model...> <trim...>``. The two reliable
anchors are (a) the leading 4-digit year and (b) a curated, multi-word-aware
list of car makes. We do NOT rely on "first token after the year is the make",
because that breaks on ``Land Rover``, ``Alfa Romeo``, ``Mercedes-Benz``, etc.

Grouping keys the rest of the project uses:
    * brand-level stats  -> ``make``      (canonical spelling)
    * model-level stats  -> ``model_key`` = ``"{make} {model}"`` (e.g. "Toyota
      Previa") - the bare model token collides across makes ("S", "GT", "3"),
      so it is always namespaced by the make.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

UNKNOWN = "UNKNOWN"

# Canonical makes present in this 1990s-2010s US-market dataset. Multi-word makes
# MUST be matched before single-word ones (longest-prefix wins), so the matcher
# below sorts by token count then length.
CANONICAL_MAKES: list[str] = [
    "Alfa Romeo",
    "Aston Martin",
    "Land Rover",
    "Mercedes-Benz",
    "Rolls-Royce",
    "Am General",
    "Acura",
    "Audi",
    "Bentley",
    "BMW",
    "Buick",
    "Cadillac",
    "Chevrolet",
    "Chrysler",
    "Daewoo",
    "Dodge",
    "Eagle",
    "Ferrari",
    "Fiat",
    "Fisker",
    "Ford",
    "Genesis",
    "Geo",
    "GMC",
    "Honda",
    "Hummer",
    "Hyundai",
    "Infiniti",
    "Isuzu",
    "Jaguar",
    "Jeep",
    "Kia",
    "Lamborghini",
    "Lexus",
    "Lincoln",
    "Lotus",
    "Maserati",
    "Maybach",
    "Mazda",
    "McLaren",
    "Mercury",
    "Mini",
    "Mitsubishi",
    "Nissan",
    "Oldsmobile",
    "Panoz",
    "Plymouth",
    "Pontiac",
    "Porsche",
    "Ram",
    "Saab",
    "Saturn",
    "Scion",
    "Smart",
    "Subaru",
    "Suzuki",
    "Tesla",
    "Toyota",
    "Volkswagen",
    "Volvo",
]

# Common alternative spellings folded into the canonical make above.
ALIASES: dict[str, str] = {
    "mercedes": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "benz": "Mercedes-Benz",
    "vw": "Volkswagen",
    "chevy": "Chevrolet",
    "rolls royce": "Rolls-Royce",
    "landrover": "Land Rover",
    "land-rover": "Land Rover",
    "alfa": "Alfa Romeo",
    "range rover": "Land Rover",  # occasionally listed without the "Land Rover" prefix
}

# Tokens that end the model span and begin the trim: body styles, drivetrain,
# door counts, and common trim markers.
STOP_TOKENS: set[str] = {
    # body styles
    "sedan",
    "coupe",
    "convertible",
    "hatchback",
    "wagon",
    "minivan",
    "van",
    "suv",
    "pickup",
    "truck",
    "crew",
    "cab",
    "extended",
    "regular",
    "quad",
    "club",
    # truck cab-style qualifiers (consolidate Tacoma/Tundra/Frontier variants)
    "double",
    "access",
    "xtracab",
    "crewmax",
    "supercrew",
    "supercab",
    "king",
    "mega",
    "crosscabriolet",
    "sport",
    "utility",
    "hardtop",
    "roadster",
    "fastback",
    "liftback",
    # drivetrain
    "awd",
    "fwd",
    "rwd",
    "4wd",
    "4x4",
    "2wd",
    "quattro",
    "4matic",
    "xdrive",
    # trims
    "le",
    "xle",
    "se",
    "lt",
    "ls",
    "ex",
    "dx",
    "gt",
    "gls",
    "glx",
    "limited",
    "base",
    "premium",
    "touring",
    "lx",
    "sl",
    "sle",
    "slt",
    "laramie",
}

_YEAR_RE = re.compile(r"^\s*(\d{4})\b")
_DOOR_RE = re.compile(r"^\d+dr$", re.IGNORECASE)

# Build the matcher: (lowercased spelling, canonical), longest first so multi-word
# and hyphenated makes win before their single-word prefixes.
_MATCHERS: list[tuple[str, str]] = sorted(
    [(m.lower(), m) for m in CANONICAL_MAKES] + list(ALIASES.items()),
    key=lambda kv: (kv[0].count(" "), len(kv[0])),
    reverse=True,
)


@dataclass(frozen=True)
class ParsedTitle:
    """Structured view of a ``Vehicle_Title`` string."""

    year: int | None
    make: str  # canonical make, or UNKNOWN
    model: str  # bare model span, e.g. "Previa" ("" if unknown)
    trim: str
    raw: str

    @property
    def model_key(self) -> str:
        """Make-namespaced model, e.g. ``"Toyota Previa"`` (empty if no model)."""
        if self.make == UNKNOWN or not self.model:
            return self.make
        return f"{self.make} {self.model}"


def _match_make(rest: str) -> tuple[str, str]:
    """Return (canonical_make, remainder_after_make). UNKNOWN if no make matches."""
    low = rest.lower()
    for matcher, canonical in _MATCHERS:
        if low == matcher or low.startswith(matcher + " "):
            return canonical, rest[len(matcher) :].strip()
    return UNKNOWN, rest


def _split_model_trim(after_make: str, max_model_tokens: int = 2) -> tuple[str, str]:
    """Take up to ``max_model_tokens`` leading tokens as the model, stopping at
    the first body-style / drivetrain / door / trim token; the rest is trim.
    """
    tokens = after_make.split()
    model: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in STOP_TOKENS or _DOOR_RE.match(tok):
            break
        model.append(tok)
        if len(model) >= max_model_tokens:
            break
    trim = after_make[len(" ".join(model)) :].strip()
    return " ".join(model), trim


def parse_vehicle_title(title: str) -> ParsedTitle:
    """Parse one ``Vehicle_Title`` into a :class:`ParsedTitle`."""
    raw = "" if title is None else str(title)
    s = raw.strip()

    m = _YEAR_RE.match(s)
    if m:
        year: int | None = int(m.group(1))
        rest = s[m.end() :].strip()
    else:
        year = None
        rest = s

    make, after_make = _match_make(rest)
    if make == UNKNOWN:
        return ParsedTitle(year=year, make=UNKNOWN, model="", trim=after_make, raw=raw)

    model, trim = _split_model_trim(after_make)
    return ParsedTitle(year=year, make=make, model=model, trim=trim, raw=raw)


def add_parsed_columns(df: pd.DataFrame, title_col: str = "Vehicle_Title") -> pd.DataFrame:
    """Return a copy of ``df`` with year / make / model / model_key / trim columns."""
    out = df.copy()
    parsed = [parse_vehicle_title(t) for t in out[title_col].astype(str)]
    out["year"] = [p.year for p in parsed]
    out["make"] = [p.make for p in parsed]
    out["model"] = [p.model for p in parsed]
    out["model_key"] = [p.model_key for p in parsed]
    out["trim"] = [p.trim for p in parsed]
    return out


def make_coverage(df: pd.DataFrame, make_col: str = "make") -> float:
    """Fraction of rows whose make was resolved to a canonical brand."""
    if len(df) == 0:
        return 0.0
    return float((df[make_col] != UNKNOWN).mean())


def unmatched_make_examples(
    df: pd.DataFrame, title_col: str = "Vehicle_Title", make_col: str = "make", k: int = 15
) -> list[str]:
    """Top leading tokens (after the year) for rows we could NOT map to a make -
    an auditable list for extending :data:`CANONICAL_MAKES`.
    """
    unmatched = df.loc[df[make_col] == UNKNOWN, title_col].astype(str)
    counts: dict[str, int] = {}
    for title in unmatched:
        s = _YEAR_RE.sub("", title).strip()
        lead = s.split()[0] if s.split() else ""
        if lead:
            counts[lead] = counts.get(lead, 0) + 1
    return [tok for tok, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]]
