"""The GPU tests run the GPU paths on a real CUDA device when one is usable, otherwise in numba-cuda's CUDA
simulator (docs/models/gpu.md "Tests"). The choice is made before numba.cuda is first imported: the
simulator can only be selected then."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# pytest runs with --import-mode=importlib (pyproject.toml): test directories are not on sys.path, so the
# helper module of these tests (gpufix.py, a unique name) is made importable here
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _select_backend() -> None:
    if "numba.cuda" in sys.modules or os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        return
    from cnctwin.gpu import device

    if not device.probe(0).usable:
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
    device.forget()


_select_backend()


@pytest.fixture(scope="session")
def gpu_kind() -> str:
    """'cuda' or 'simulator' (the GPU tests skip when neither is usable)."""
    from cnctwin.gpu import device

    found = device.info()
    if not found.usable:
        pytest.skip(f"no CUDA device and no CUDA simulator: {found.reason}")
    return found.kind
