"""Insert src/ on the path for tests (mirrors the project-root conftest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
