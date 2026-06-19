"""Make the scripts/ directory importable in tests without installing the project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
