"""Controller tests: the fixture module (harness) importable under any pytest import mode."""

import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
