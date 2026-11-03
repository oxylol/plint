"""The coolant tests import their fixtures module ``coolfix`` directly; make that work under both
pytest import modes (prepend and importlib)."""

import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
