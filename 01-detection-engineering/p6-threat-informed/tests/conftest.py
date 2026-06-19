"""Ensure src/ is importable when pytest is invoked from the tests/ dir too."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
