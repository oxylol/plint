"""Projects of the convergence studies (docs/convergence.md).

Every milling case is a variant of examples/pocket-islands-al6061 (Haas VF-2SS, Al 6061-T6, emulsion) with a small
block stock (or a meshed stock with a pocket), no CAD and a short program of its own; the turning case is a variant of
examples/lathe-shaft-4140 (Haas ST-20Y, 4140). Projects are written by cnctwin.core.drive.project_variant. Frames: G54
X0 Y0 at the stock's lower-left corner, Z0 on its top (milling); Z0 on the bar's free end (turning).

The settings a study varies are written into the project's ``physics`` section (``physics_edits``); nothing else is
changed between the levels of a study.
"""

from __future__ import annotations

import copy
import math
import struct
from pathlib import Path

from cnctwin.core.drive import project_variant
from cnctwin.paths import EXAMPLES

POCKET = Path(EXAMPLES) / "pocket-islands-al6061"
LATHE = Path(EXAMPLES) / "lathe-shaft-4140"

END_MILL_D10 = "helical-59482-mhsv-020-41000"        # Helical MHSV-4: D10, 4 flutes, 35 deg helix, variable pitch
BALL_D10 = "tungaloy-teb100e4l-20-30c10cfh72"         # Tungaloy TEB100E4L: D10 ball nose, 4 flutes, 38 deg helix
TURN_TOOL = "tungaloy-cnmg120408-tm-t9225-pclnr2525m12e"   # CNMG120408-TM T9225 in PCLNR2525M12E (r 0.8 mm)

RPM = 10000.0
FEED_MM_MIN = 1600.0                                  # f_z = 0.04 mm (4 flutes, 10 000 rpm)

# dynamics' L2 stability analysis on coarse grids (lobe diagrams and SD checks are report artefacts: they feed no
# quantity of these studies; the defaults cost minutes per new engagement). The time-domain chatter detection and
# the hot loop's modal dynamics are unchanged by these settings.
DYNAMICS_FAST_L2 = {"sld_n_freq": 64, "sld_n_speeds": 8, "sld_n_lobes": 2, "sd_check": 0, "sd_max_points": 0,
                    "sld_refresh_macro": 1000, "engagement_tol": 0.5, "pd_max_iter": 2, "pd_tol": 0.05}


# ------------------------------------------------------------------------------------------------ programs
def _mill_head(rpm: float, x_mm: float, y_mm: float) -> list[str]:
    return ["%", "O07001 (CONVERGENCE STUDY)", "G90 G17 G21 G40 G49 G80 G94 G54", "G53 G00 Z0.", "T1 M06",
            f"G90 G54 G00 X{x_mm:.4f} Y{y_mm:.4f} S{rpm:.0f} M03", "G43 H01 Z25. M08", "Z2."]


def _mill_tail() -> list[str]:
    return ["G00 Z25.", "M09", "M05", "G53 G49 Z0.", "M30", "%"]


def line_program(ap_mm: float, y_mm: float, x0_mm: float, x1_mm: float, feed_mm_min: float = FEED_MM_MIN,
                 rpm: float = RPM, lead_out_y_mm: float | None = None) -> str:
    """Plunge outside the stock at (x0, y), one straight cut along +x to x1 at depth ap, an optional lead-out
    move in y (sideways out of the cut before the retract), retract."""
    body = _mill_head(rpm, x0_mm, y_mm) + [f"G01 Z{-ap_mm:.4f} F1000.", f"G01 X{x1_mm:.4f} F{feed_mm_min:.1f}"]
    if lead_out_y_mm is not None:
        body.append(f"G01 Y{lead_out_y_mm:.4f}")
    return "\n".join(body + _mill_tail()) + "\n"


def corner_program(ap_mm: float, path_mm: list[tuple[float, float]], feed_mm_min: float = FEED_MM_MIN,
                   rpm: float = RPM) -> str:
    """Plunge at the first point of ``path_mm`` (outside the material), G01 through the others at depth ap."""
    x0, y0 = path_mm[0]
    body = _mill_head(rpm, x0, y0) + [f"G01 Z{-ap_mm:.4f} F1000."]
    for i, (x, y) in enumerate(path_mm[1:]):
        body.append(f"G01 X{x:.4f} Y{y:.4f}" + (f" F{feed_mm_min:.1f}" if i == 0 else ""))
    return "\n".join(body + _mill_tail()) + "\n"


def passes_program(ap_mm: float, passes: list[tuple[float, float, float]], feed_mm_min: float = FEED_MM_MIN,
                   rpm: float = RPM) -> str:
    """Several straight cuts at depth ap, each (y, x0, x1) from x0 to x1 (outside the stock at both ends): the
    tool goes up and back over the part between them (every pass the same climb or conventional direction)."""
    y, x0, _ = passes[0]
    body = _mill_head(rpm, x0, y)
    for y, x0, x1 in passes:
        body += [f"G00 X{x0:.4f} Y{y:.4f}", "Z2.", f"G01 Z{-ap_mm:.4f} F1000.", f"G01 X{x1:.4f} F{feed_mm_min:.1f}",
                 "G00 Z2."]
    return "\n".join(body + _mill_tail()) + "\n"


def turn_program(d_mm: float, ap_mm: float, z1_mm: float, f_mm_rev: float, rpm: float) -> str:
    """One straight OD pass at constant speed (G97, M04: the direction tooling's right-hand tool cuts along its
    rake face under motion's ISO spindle sign, docs/models/hotloop.md "Open problems") from Z2 to z1."""
    x = d_mm - 2.0 * ap_mm
    return f"""%
O07002 (CONVERGENCE STUDY: OD TURNING PASS)
N10 G21 G18 G40 G80 G99
N20 G54
N30 G50 S2500
N100 G53 G00 X0
N110 T0101
N120 G97 S{rpm:.0f} M04
N130 M08
N140 G00 X{d_mm + 4.0:.3f} Z2.0
N150 G00 X{x:.3f}
N160 G01 Z{z1_mm:.3f} F{f_mm_rev:.3f}
N170 G00 X{d_mm + 4.0:.3f}
N180 G53 G00 X0
N190 M05
N200 M30
%
"""


# ------------------------------------------------------------------------------------------------ edits
def _merge(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def block_edits(size_mm, *, tool: str = END_MILL_D10, physics: dict | None = None, runout_um: float | None = None,
                machine=None, stock_mesh: str | None = None):
    """Edits of the pocket example: the block (or meshed) stock centred on the vise seat, one tool, no CAD, no
    outputs; ``physics`` merged into the physics section (the study's settings)."""
    size_mm = [float(v) for v in size_mm]

    def edit(data):
        data["cad"] = None
        data["stock"]["size_mm"] = list(size_mm)
        if stock_mesh is not None:
            data["stock"]["kind"] = "mesh"
            data["stock"]["mesh"] = stock_mesh
        data["physics"]["dynamics"] = dict(DYNAMICS_FAST_L2)
        _merge(data["physics"], physics or {})
        entry = data["tools"]["list"][0]
        entry["tool"] = tool
        if runout_um is not None:
            entry["runout_um"] = float(runout_um)
        data["tools"]["list"] = [entry]
        data["production"]["parts"] = 1
        data["outputs"] = {"report": [], "images": False, "video": False, "audio": False}
        if machine is not None:
            data["machine"] = machine
        op = data["operations"][0]
        op["setup"]["stock_pose"] = {"translate_mm": [-size_mm[0] / 2, -size_mm[1] / 2, 0], "rotate_deg": [0, 0, 0]}
        op["setup"]["work_offsets_mm"] = {"G54": [-381.0 - size_mm[0] / 2, -203.2 - size_mm[1] / 2,
                                                  -510.0 + size_mm[2]]}
        op["setup"]["workholding"][0]["jaw_opening_mm"] = size_mm[1]
    return edit


def bar_edits(d_mm: float, length_mm: float, *, physics: dict | None = None):
    """Edits of the lathe example: a plain d x length bar of 4140 chucked 30 mm deep in its soft jaws (no
    tailstock, no CAD), the CNMG roughing tool alone. G54 Z0 on the free end."""
    def edit(data):
        data["cad"] = None
        data["physics"]["dynamics"] = dict(DYNAMICS_FAST_L2)
        _merge(data["physics"], physics or {})
        data["stock"] = {"kind": "cylinder", "size_mm": [d_mm, length_mm], "tolerance_mm": None,
                         "non_squareness_deg": None, "form": "cold_drawn_bar", "mesh": None}
        data["tools"]["list"] = [data["tools"]["list"][0]]
        data["tools"]["list"][0]["tool"] = TURN_TOOL
        data["production"]["parts"] = 1
        data["outputs"] = {"report": [], "images": False, "video": False, "audio": False}
        op = data["operations"][0]
        op["setup"]["workholding"] = op["setup"]["workholding"][:2]
        op["setup"]["stock_pose"] = {"translate_mm": [0, 0, 99], "rotate_deg": [0, 0, 0]}
        op["setup"]["work_offsets_mm"] = {"G54": [-426.0, 0.0, 99.0 + length_mm - 662.0]}
    return edit


def one_mode_machine(db, fn_Hz: float, zeta: float, k_N_um: float, direction: str = "x") -> dict:
    """The VF-2SS with one flexible mode: a spindle mode carrying the rigid tool (no frame, table or rotary modes),
    written inline with ``extends`` (the 1-DOF regenerative system of tests/integration/test_chatter.py)."""
    obj = db.object("machine", "haas-vf2ss")
    raw = copy.deepcopy(obj.merged if obj.merged is not None else obj.raw)
    spindles = raw["spindles"]
    spindles[0]["frf_modes"] = [{"fn_Hz": float(fn_Hz), "zeta": float(zeta), "k_N_um": float(k_N_um),
                                 "direction": direction, "src": "project"}]
    structure = dict(raw.get("structure") or {})
    structure.update({"modes": [], "table_modes": [], "rotary_modes": []})
    return {"extends": "haas-vf2ss", "spindles": spindles, "structure": structure}


# ------------------------------------------------------------------------------------------------ meshes
def write_pocket_stl(path: Path, size_mm, pocket_mm) -> None:
    """Binary STL of a block [0, sx] x [0, sy] x [0, sz] with a rectangular pocket [x0, x1] x [y0, y1] open at the
    top down to z_floor (sharp internal corners). ``pocket_mm`` = (x0, x1, y0, y1, z_floor); x0 <= 0 opens the pocket
    to the left face. The solid is the union of the cells of the grid through every breakpoint less the pocket's
    cell; each boundary face of a cell is written as two triangles with an outward normal, so the mesh is conforming
    (no T-junctions) and watertight."""
    sx, sy, sz = (float(v) for v in size_mm)
    x0, x1, y0, y1, zf = (float(v) for v in pocket_mm)
    xs = [0.0] + ([x0] if x0 > 0.0 else []) + [x1, sx]
    ys = [0.0, y0, y1, sy]
    zs = [0.0, zf, sz]
    xa = max(x0, 0.0)

    def filled(i, j, k):
        if i < 0 or j < 0 or k < 0 or i >= len(xs) - 1 or j >= len(ys) - 1 or k >= len(zs) - 1:
            return False
        inside_pocket = xs[i] >= xa and xs[i + 1] <= x1 and ys[j] >= y0 and ys[j + 1] <= y1 and zs[k] >= zf
        return not inside_pocket

    tris = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                if not filled(i, j, k):
                    continue
                X = (xs[i], xs[i + 1])
                Y = (ys[j], ys[j + 1])
                Z = (zs[k], zs[k + 1])
                for axis in range(3):
                    for side in (0, 1):
                        nb = [i, j, k]
                        nb[axis] += 1 if side else -1
                        if filled(*nb):
                            continue
                        # the face's four corners, counter-clockwise seen from outside
                        c = [X, Y, Z]
                        u, v = [a for a in range(3) if a != axis]
                        fixed = c[axis][side]
                        pts = []
                        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                            p = [0.0, 0.0, 0.0]
                            p[axis] = fixed
                            p[u] = c[u][a]
                            p[v] = c[v][b]
                            pts.append(tuple(p))
                        # (u, v, axis) is a right-handed cyclic order for axis = 0 (y, z), 1 (x, z: left), 2 (x, y)
                        right_handed = axis != 1
                        outward_positive = side == 1
                        if right_handed != outward_positive:
                            pts = pts[::-1]
                        tris.append((pts[0], pts[1], pts[2]))
                        tris.append((pts[0], pts[2], pts[3]))
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(b"cnctwin convergence pocket stock".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            ux, uy, uz = (b[i] - a[i] for i in range(3))
            vx, vy, vz = (c[i] - a[i] for i in range(3))
            n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
            m = math.sqrt(sum(w * w for w in n)) or 1.0
            fh.write(struct.pack("<3f", *(w / m for w in n)))
            for p in (a, b, c):
                fh.write(struct.pack("<3f", *(float(w) for w in p)))
            fh.write(struct.pack("<H", 0))


# ------------------------------------------------------------------------------------------------ projects
def mill_project(directory, program: str, edits) -> Path:
    return project_variant(POCKET, directory, program_text=program, program_name="study.nc", edits=edits)


def turn_project(directory, program: str, edits) -> Path:
    return project_variant(LATHE, directory, program_text=program, program_name="study.nc", edits=edits)
