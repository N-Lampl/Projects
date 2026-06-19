"""Make src/ (and the shared ids_pipeline) importable in tests without installing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# the shared library is imported by path; expose it for tests too
sys.path.insert(0, str((ROOT / ".." / "shared" / "ids_pipeline" / "src").resolve()))
