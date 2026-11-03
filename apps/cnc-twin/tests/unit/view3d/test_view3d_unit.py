"""The 3D view's pure modules in node (view3d_unit.mjs) against a synthetic run read with the run
infrastructure's own readers (view3d_expect.py): frames.bin, brick records, keyframe GLBs, bricks.idx and
the API's time rule, the brick loader and its cache, level of detail, poses and chains (against
cnctwin.geom.transform), stock motion, .npz reading, field grids and their orientation, normalised
convolution, float16 rounding as numpy, measuring, ballistic jets, the cantilever bend, hashing, colour
lookups against the shell's colour maps, the api adapter, the parameters."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(HERE))

import make_synthetic_run as synthetic  # noqa: E402
import view3d_expect  # noqa: E402


def test_view3d_modules_in_node(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node (18 or later) is not installed")
    run = tmp_path / "run"
    synthetic.make_synthetic_run(run, seed=4, media=False, view3d=True)
    expect = tmp_path / "expect" / "expect.json"
    view3d_expect.main([str(run), str(expect)])
    result = subprocess.run([node, str(HERE / "view3d_unit.mjs"), str(ROOT / "web"), str(expect)],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    passed, failed = result.stdout.strip().splitlines()[-1].split(" passed, ")
    assert int(failed.split()[0]) == 0 and int(passed) >= 150, result.stdout[-2000:]
