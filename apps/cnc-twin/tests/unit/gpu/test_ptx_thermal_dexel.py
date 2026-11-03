"""The thermal and dexel kernels compile for a real device (sm_52, the GTX 950) with FMA contraction off,
without a device: NVVM of the installed CUDA toolkit in a child process (cnctwin.gpu.ptxcheck)."""

from __future__ import annotations

import pytest

from cnctwin.gpu import ptxcheck


def test_compiles_for_sm52():
    report = ptxcheck.check((5, 2), groups=("thermal", "dexel"))
    if "fma" not in report:
        pytest.skip(f"no CUDA toolkit (NVVM) to compile with: {report.get('error', '')[:300]}")
    assert report["fma"]["ok"], report["fma"]
    assert report["ok"], report
    assert set(report["kernels"]) == {"thermal.k_cells", "thermal.k_sweep_x", "thermal.k_sweep_y",
                                      "thermal.k_sweep_z", "thermal.k_update", "dexel.k_rays"}
    assert all(r["target"] == "sm_52" for r in report["kernels"].values())
