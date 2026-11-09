"""Which cnctwin tree the convergence tools run (imported first, before NumPy, Numba or cnctwin).

By default the tree the tools sit in (``tools/convergence/../..``), with the Numba cache bin/tsafe chose for it. With
``--root DIR`` (a copy of the tree: cnctwin/, db/, examples/) the studies run that copy and keep its compiled kernels
in ``DIR/.numba-cache-convergence`` (or ``--cache DIR``): the copy's sources never change, so its cache stays valid
while the working tree is being edited (bin/tsafe keys the working tree's cache by every kernel source, and an edit
of any of them means a cold compile of about 12 CPU minutes).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]


def pop_option(argv: list[str], name: str) -> str | None:
    """Remove ``--name VALUE`` / ``--name=VALUE`` from argv and return VALUE."""
    for i, arg in enumerate(list(argv)):
        if arg == name and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if arg.startswith(name + "="):
            del argv[i]
            return arg.split("=", 1)[1]
    return None


_CHOSEN: list[Path] = []


def setup(argv: list[str] | None = None) -> Path:
    """Put the chosen tree first on sys.path (and its own Numba cache in the environment). Returns the root. Idempotent:
    a later call (a module that imports study.py after its script chose a tree) keeps the first choice."""
    if _CHOSEN:
        return _CHOSEN[0]
    argv = sys.argv if argv is None else argv
    root = pop_option(argv, "--root")
    cache = pop_option(argv, "--cache")
    root_path = Path(root).resolve() if root else DEFAULT_ROOT
    if root:
        os.environ["NUMBA_CACHE_DIR"] = str(Path(cache).resolve() if cache else root_path / ".numba-cache-convergence")
        os.environ["CNCTWIN_ROOT"] = str(root_path)
    elif cache:
        os.environ["NUMBA_CACHE_DIR"] = str(Path(cache).resolve())
    for p in (str(root_path), str(HERE)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(root_path))
    _CHOSEN.append(root_path)
    return root_path
