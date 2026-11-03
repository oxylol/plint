import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba-cache"))
