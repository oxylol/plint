"""Where the simulator's data lives: the repository root (db/, web/, examples/, validation/, docs/).

Set CNCTWIN_ROOT to override (the Docker image uses /app). Never hard-code /opt/plint paths in code.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CNCTWIN_ROOT", Path(__file__).resolve().parent.parent))
DB = ROOT / "db"
WEB = ROOT / "web"
EXAMPLES = ROOT / "examples"
VALIDATION = ROOT / "validation"
DOCS = ROOT / "docs"


def data_dir() -> Path:
    """Projects and runs of the service (DATA_DIR), default ROOT/runs-data for local use."""
    return Path(os.environ.get("DATA_DIR", ROOT / "runs-data"))
