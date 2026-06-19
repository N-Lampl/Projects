"""Insert src/ on path so `import log_ueba` works under pytest from any cwd."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
