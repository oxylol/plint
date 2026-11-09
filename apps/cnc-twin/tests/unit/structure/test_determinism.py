"""Deterministic eigenvectors: two separate processes compute the same workpiece modes (direct and multigrid
solvers, Ritz refresh) and tool modes; the SHA-256 of the result arrays must be equal."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

CHILD = r"""
import hashlib, sys
import numpy as np
sys.path.insert(0, ROOT)
from tests.unit.structure import structfix as fx
from tests.unit.tooling.fixtures import HOLDER, TOOLS
from cnctwin.db.simap import SIMap
from cnctwin.structure import toolmodel as TM
from cnctwin.structure import voxel as VX
from cnctwin.structure.schema import settings
from cnctwin.tooling import build_tool

h = hashlib.sha256()
grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (12, 8, 5))
active = np.ones(grid.shape, bool)
model = VX.VoxelModel(grid, active, 70e9, 0.33, 2700.0)
pts = model.node_xyz()[model.node_ijk[:, 2] == 0][::3]
springs = model.spring_data(pts, np.tile(np.eye(3) * 1e8, (len(pts), 1, 1)))
for direct in (100000, 0):
    system = VX.System(model, springs, None, settings({"direct_max_dof": direct, "gmg_coarse_dof": 200}))
    lam, V = system.eigen(8, 1e-10)
    h.update(lam.tobytes()); h.update(np.ascontiguousarray(V).tobytes())
cut = active.copy(); cut[3:9, 2:6, 4] = False
model2 = VX.VoxelModel(grid, cut, 70e9, 0.33, 2700.0)
system2 = VX.System(model2, model2.spring_data(pts, np.tile(np.eye(3) * 1e8, (len(pts), 1, 1))), None, settings({}))
guess = np.stack([model2.from_lattice(model.to_lattice(V[:, j])) for j in range(8)], axis=1)
lam2, V2, res, ok = system2.ritz(guess, lam, 8, 1e-3, 6)
h.update(lam2.tobytes()); h.update(np.ascontiguousarray(V2).tobytes())
geo = build_tool(TOOLS["square_endmill"], HOLDER, {"number": 1, "stickout_mm": 35})
res = TM.tool_modes(geo, SIMap(fx.MACHINE), settings({}))
for a in (res.modes.freq_Hz, res.modes.zeta, res.modes.shape, res.modes.static_residual_m_per_N):
    h.update(np.ascontiguousarray(a).tobytes())
print(h.hexdigest())
"""


def _run() -> str:
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1"})
    out = subprocess.run([sys.executable, "-c", f"ROOT = {str(ROOT)!r}\n" + CHILD], cwd=str(ROOT), env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_two_processes_give_identical_eigenvectors():
    first = _run()
    second = _run()
    assert len(first) == 64 and first == second
