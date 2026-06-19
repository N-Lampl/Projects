"""Ensure src/ is on the path when pytest is invoked from inside tests/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
