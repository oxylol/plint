"""Projects and helpers of the run-loop tests (docs/models/loop.md "Tests"; no tests here).

Every project is a small variant of an example written by cnctwin.core.drive.project_variant into a temporary
folder: short programs, small stocks and coarse settings for the slow domains' heavy L2 analyses (the lobe
diagrams, the structure's voxel model, the coefficient pipeline), every domain on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from cnctwin.core.drive import project_variant
from cnctwin.core.warmup import WARM_PHYSICS, mill_project
from cnctwin.paths import EXAMPLES

FLIP = Path(EXAMPLES) / "two-setup-flip"

# every derived output, small enough for a test (the pipeline is the same at full size;
# tests/integration/test_outputs_*.py); run options over the project's outputs section, which warmup's projects
# switch off
SMALL_OUTPUTS = {"report": True, "images": True, "audio": True, "image_size": [320, 180, 1],
                 "video": {"size": [192, 108, 1], "speed": 4, "fps": 8}, "memory_mode": "rlimit"}

FLIP_OP10 = """%
O01010 (LOOP TEST OP10: SIDE STEP ALONG THE FRONT TOP EDGE)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T2 M06
G90 G54 G00 X-8. Y-4.5 S10000 M03
G43 H02 Z25. M08
Z5.
G01 Z1.5 F1000.
G01 X10. F1600.
G01 Y-8.
G00 Z25.
M09
M05
G53 G49 Z0.
M30
%
"""

FLIP_OP20 = """%
O01020 (LOOP TEST OP20: SIDE STEP ALONG THE FRONT TOP EDGE OF THE FLIPPED PART)
G90 G17 G21 G40 G49 G80 G94 G55
G53 G00 Z0.
T2 M06
G90 G55 G00 X-8. Y-4.5 S10000 M03
G43 H02 Z25. M08
Z8.
G01 Z4. F1000.
G01 X10. F1600.
G01 Y-8.
G00 Z25.
M09
M05
G53 G49 Z0.
M30
%
"""


def tiny_mill(tmp: Path, parts: int = 1, physics: dict | None = None) -> Path:
    """cnctwin.core.warmup's mill project (a side step 0.5 mm wide and 1 mm deep along 12 mm of the front face of
    a 30 x 20 x 10 mm Al block, D10 at 10 000 rpm), ``parts`` parts, ``physics`` settings merged per domain."""
    project = mill_project(tmp)
    if parts != 1 or physics:
        data = yaml.safe_load(project.read_text())
        data["production"]["parts"] = int(parts)
        for domain, settings in (physics or {}).items():
            data["physics"].setdefault(domain, {}).update(settings)
        project.write_text(yaml.safe_dump(data, sort_keys=False))
    return project


def flip_project(tmp: Path) -> Path:
    """examples/two-setup-flip (80 x 60 x 25 Al blank, OP20 flipped 180 deg about X from OP10) with one short side
    step per operation (the D10 at 10 000 rpm, 0.5 mm wide and 1 mm deep along 10 mm of the front top edge: OP10's
    on the blank's y 0 face at its top, OP20's on the old y 60 face at the old bottom), lattice 0.25 mm, no CAD.
    OP10 does not face the top, so OP20's stock_pose sets the blank's full 25 mm (not the example's faced 22.5 mm)
    on the jaw steps: its top is at G55 Z5."""
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "op20.nc").write_text(FLIP_OP20)

    def edit(data):
        data["cad"] = None
        data["physics"].update({k: dict(v) for k, v in WARM_PHYSICS.items()})
        data["physics"]["lattice_mm"] = 0.25
        data["operations"][1]["program"] = "op20.nc"
        data["operations"][1]["setup"]["stock_pose"]["translate_mm"] = [-40, -30, 0]
        data["production"]["parts"] = 1
        data["outputs"] = {"report": [], "images": False, "video": False, "audio": False}
    return project_variant(FLIP, tmp, program_text=FLIP_OP10, program_name="op10.nc", edits=edit)


def swept_volume_m3(run_dir: Path, top_z_m: float, box_m, radius_m: float = 5e-3, ticks=None,
                    h: float = 10e-6) -> float:
    """Volume a flat end mill of ``radius_m`` sweeps out of a block along the tool path recorded in frames.bin
    (``tip_S_m`` per L1 tick, the stock frame S): the union of the tool's discs at the tip positions below the
    block's top ``top_z_m``, within the block's xy box ``((x0, y0), (x1, y1))``, rastered at ``h``, times the
    depth below the top (the median tip height). ``ticks``: (tick0, tick1) of the records taken (one operation).
    The controller's look-ahead rounds the corners of the program within its tolerance, so the path the tool
    took, not the programmed one, is the reference of the removed volume."""
    from scipy.spatial import cKDTree

    from cnctwin.core.rundir import read_frames
    records, _ = read_frames(run_dir)
    tip = np.asarray(records["tip_S_m"][:, 0, :], float)
    take = tip[:, 2] < top_z_m - 1e-5
    if ticks is not None:
        take &= (records["tick"] >= ticks[0]) & (records["tick"] < ticks[1])
    centres = tip[take]
    if not len(centres):
        return 0.0
    depth = top_z_m - float(np.median(centres[:, 2]))
    lo = np.maximum(centres[:, :2].min(axis=0) - radius_m, np.asarray(box_m[0], float))
    hi = np.minimum(centres[:, :2].max(axis=0) + radius_m, np.asarray(box_m[1], float))
    if np.any(hi <= lo):
        return 0.0
    xs = np.arange(lo[0], hi[0], h) + 0.5 * h
    ys = np.arange(lo[1], hi[1], h) + 0.5 * h
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    distance, _ = cKDTree(centres[:, :2]).query(grid, k=1)
    return float(np.count_nonzero(distance <= radius_m)) * h * h * depth


def read(run_dir: Path, name: str):
    return json.loads((Path(run_dir) / name).read_text())


def conservation_ok(run: dict) -> dict:
    """{balance: ok} of run.json summary.conservation (mass, energy, time) and the end-of-run mass check."""
    cons = run["summary"]["conservation"]
    out = {name: bool(cons[name]["ok"]) for name in ("mass", "energy", "time")}
    out["final_mass"] = bool((run["summary"].get("final_mass") or {}).get("ok"))
    return out


__all__ = ["tiny_mill", "flip_project", "swept_volume_m3", "read", "conservation_ok", "SMALL_OUTPUTS", "FLIP_OP10",
           "FLIP_OP20"]
