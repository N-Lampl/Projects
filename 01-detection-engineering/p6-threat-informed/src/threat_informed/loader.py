"""Load + validate Sigma rules and extract their ATT&CK technique ids.

PyYAML is used when available; otherwise we fall back to a tiny YAML-subset parser
that handles exactly the structure our Sigma rules use (scalars, nested maps, and
``- `` block lists). This keeps the DEFAULT path dependency-free: it works with
nothing but the Python standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .attack import is_known_technique, normalize_technique_id, tactics_for_technique

# ---------------------------------------------------------------------------
# YAML parsing (PyYAML if present, else a minimal fallback)
# ---------------------------------------------------------------------------
try:  # optional enhanced path
    import yaml as _yaml

    _HAVE_YAML = True
except Exception:  # pragma: no cover - exercised only without pyyaml
    _yaml = None
    _HAVE_YAML = False


def _strip_comment(line: str) -> str:
    # Remove trailing comments that are not inside quotes (our rules quote values).
    if "'" in line or '"' in line:
        return line
    idx = line.find("#")
    return line if idx == -1 else line[:idx]


def _coerce(value: str):
    value = value.strip()
    if not value:
        return None
    if (value[0] == value[-1]) and value[0] in "'\"":
        return value[1:-1]
    return value


def _minimal_yaml_load(text: str) -> dict:
    """Parse the small YAML subset used by our Sigma rules.

    Supports: ``key: value``, nested maps via indentation, and block sequences
    (``- item``). Indentation is two spaces per level (as written in rules/).
    """
    root: dict = {}
    # stack of (indent, container) where container is a dict
    stack: list[tuple[int, dict]] = [(-1, root)]
    last_list: list | None = None
    last_list_indent = -1

    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        # block-sequence item
        if stripped.startswith("- "):
            item = _coerce(stripped[2:])
            if last_list is None or indent < last_list_indent:
                continue
            last_list.append(item)
            continue

        # key: value  (pop deeper scopes)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            # could open a map or a list; peek handled lazily
            new_map: dict = {}
            parent[key] = new_map
            stack.append((indent, new_map))
            # prepare a list in case the children are "- " items
            last_list = []
            last_list_indent = indent
            parent[key] = last_list if False else new_map  # default map; swapped below
            # We don't know yet; store both behaviours: use a list holder.
            parent[key] = _Pending(new_map, last_list)
        else:
            parent[key] = _coerce(rest)

    return _resolve_pending(root)


@dataclass
class _Pending:
    as_map: dict
    as_list: list


def _resolve_pending(obj):
    if isinstance(obj, _Pending):
        if obj.as_list:
            return obj.as_list
        return _resolve_pending(obj.as_map)
    if isinstance(obj, dict):
        return {k: _resolve_pending(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_pending(v) for v in obj]
    return obj


def load_yaml(text: str) -> dict:
    """Parse a YAML document, using PyYAML when installed."""
    if _HAVE_YAML:
        return _yaml.safe_load(text)
    return _minimal_yaml_load(text)


# ---------------------------------------------------------------------------
# Sigma rule model + validation
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ("title", "logsource", "detection", "level")
_TECHNIQUE_TAG = re.compile(r"^attack\.t(\d{4}(?:\.\d{3})?)$", re.IGNORECASE)


@dataclass
class SigmaRule:
    path: Path
    title: str
    rule_id: str
    level: str
    logsource: dict
    techniques: list[str] = field(default_factory=list)
    tactics: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _extract_techniques(tags) -> list[str]:
    out: list[str] = []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        m = _TECHNIQUE_TAG.match(tag.strip())
        if m:
            out.append(normalize_technique_id("T" + m.group(1)))
    return out


def parse_rule(path: Path) -> SigmaRule:
    """Parse one Sigma rule file into a validated :class:`SigmaRule`."""
    data = load_yaml(Path(path).read_text())
    errors: list[str] = []
    if not isinstance(data, dict):
        return SigmaRule(
            path=Path(path),
            title="<unparsable>",
            rule_id="",
            level="",
            logsource={},
            errors=["file did not parse to a mapping"],
        )

    for fld in REQUIRED_FIELDS:
        if fld not in data:
            errors.append(f"missing required field: {fld}")

    logsource = data.get("logsource") or {}
    if isinstance(logsource, dict) and not (logsource.get("category") or logsource.get("service")):
        errors.append("logsource has neither 'category' nor 'service'")

    detection = data.get("detection") or {}
    if isinstance(detection, dict) and "condition" not in detection:
        errors.append("detection block has no 'condition'")

    techniques = _extract_techniques(data.get("tags"))
    if not techniques:
        errors.append("no ATT&CK technique tag (attack.tXXXX) found")
    for tid in techniques:
        if not is_known_technique(tid):
            errors.append(f"unknown ATT&CK technique id: {tid}")

    tactics: list[str] = []
    for tid in techniques:
        for t in tactics_for_technique(tid):
            if t not in tactics:
                tactics.append(t)

    return SigmaRule(
        path=Path(path),
        title=str(data.get("title", "")),
        rule_id=str(data.get("id", "")),
        level=str(data.get("level", "")),
        logsource=logsource if isinstance(logsource, dict) else {},
        techniques=techniques,
        tactics=tactics,
        errors=errors,
    )


def load_rules(rules_dir: str | Path) -> list[SigmaRule]:
    """Parse + validate every ``*.yml`` / ``*.yaml`` rule in a directory (sorted)."""
    rules_dir = Path(rules_dir)
    files = sorted(p for p in rules_dir.iterdir() if p.suffix in (".yml", ".yaml"))
    return [parse_rule(p) for p in files]
