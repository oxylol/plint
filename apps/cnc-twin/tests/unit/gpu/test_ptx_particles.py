"""The particle flight kernel compiles for a real device (sm_52) with FMA contraction off, without a device
(cnctwin.gpu.ptxcheck)."""

from __future__ import annotations

import pytest

from cnctwin.gpu import ptxcheck


def test_compiles_for_sm52():
    report = ptxcheck.check((5, 2), groups=("particles",))
    if "fma" not in report:
        pytest.skip(f"no CUDA toolkit (NVVM) to compile with: {report.get('error', '')[:300]}")
    assert report["ok"], report
    flight = report["kernels"]["particles.k_flight"]
    assert flight["target"] == "sm_52" and not flight["local_memory"]
