"""Projects and references of the hot-loop integration tests (docs/models/hotloop.md "Tests").

Every project is a variant of examples/pocket-islands-al6061 (Haas VF-2SS, Al 6061-T6, emulsion) with a small
block stock, no CAD and a short program of its own, written to a temporary directory by
cnctwin.core.drive.project_variant. Frames: G54 X0 Y0 at the stock's lower-left corner, Z0 on its top.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from cnctwin.core.drive import OperationDriver, project_variant
from cnctwin.paths import EXAMPLES

POCKET = Path(EXAMPLES) / "pocket-islands-al6061"
END_MILL_D10_4FL = "helical-59482-mhsv-020-41000"     # MHSV-4: D10, 4 flutes, 35 deg helix, LOC 20, variable pitch
DRILL_D6_8 = "osg-8690680-ado-3d-6.8"                  # OSG ADO-3D carbide drill D6.8
SPEED_RPM = 10000.0
OMEGA = -SPEED_RPM * 2.0 * math.pi / 60.0              # M03
SLOT_FEED_MM_MIN = 1600.0                              # f_z = 0.04 mm


# dynamics' L2 stability analysis (lobe diagrams per new engagement, SD checks) on coarse grids: the tests check
# the hot loop, not the lobes (cnctwin.dynamics.schema; the defaults cost minutes per new engagement)
DYNAMICS_FAST_L2 = {"sld_n_freq": 64, "sld_n_speeds": 8, "sld_n_lobes": 2, "sd_check": 0, "sd_max_points": 0,
                    "sld_refresh_macro": 1000, "engagement_tol": 0.5}


# cutting's coefficient pipeline every 0.5 s: during a spindle run-up it otherwise rebuilds its Oxley rows at every
# speed change (minutes for a drill's lip, chisel and margin groups); the cut itself runs at constant speed
CUTTING_EVERY_N_MACRO = 50


def _block(data, size_mm=(40.0, 30.0, 10.0), lattice_mm=0.1, tool=END_MILL_D10_4FL, element_mm=None):
    data["cad"] = None
    data["physics"]["dynamics"] = dict(DYNAMICS_FAST_L2)
    data["physics"]["cutting"] = {"every_n_macro": CUTTING_EVERY_N_MACRO}
    data["stock"]["size_mm"] = list(size_mm)
    data["physics"]["lattice_mm"] = lattice_mm
    if element_mm is not None:
        data["physics"]["edge_element_mm"] = element_mm
    entry = data["tools"]["list"][0]
    entry["tool"] = tool
    data["tools"]["list"] = [entry]
    op = data["operations"][0]
    # the stock centred on the vise seat; G54 at its lower-left corner and top (fixture seat 100 mm above the table)
    op["setup"]["stock_pose"] = {"translate_mm": [-size_mm[0] / 2, -size_mm[1] / 2, 0], "rotate_deg": [0, 0, 0]}
    op["setup"]["work_offsets_mm"] = {"G54": [-381.0 - size_mm[0] / 2, -203.2 - size_mm[1] / 2,
                                              -510.0 + size_mm[2]]}
    op["setup"]["workholding"][0]["jaw_opening_mm"] = size_mm[1]


def slot_program(ap_mm: float, y_mm: float = 15.0, x0_mm: float = -8.0, x1_mm: float = 30.0,
                 feed_mm_min: float = SLOT_FEED_MM_MIN, rpm: float = SPEED_RPM) -> str:
    return f"""%
O01002 (FULL SLOT)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T1 M06
G90 G54 G00 X{x0_mm:.3f} Y{y_mm:.3f} S{rpm:.0f} M03
G43 H01 Z25. M08
Z2.
G01 Z-{ap_mm:.4f} F1000.
G01 X{x1_mm:.3f} F{feed_mm_min:.1f}
G00 Z25.
M05
M30
%
"""


def slot_project(tmp: Path, ap_mm: float, *, lattice_mm: float = 0.1, size_mm=(40.0, 30.0, 10.0), **program) -> Path:
    return project_variant(POCKET, tmp, program_text=slot_program(ap_mm, **program), program_name="slot.nc",
                           edits=lambda d: _block(d, size_mm, lattice_mm))


def drill_program(depth_mm: float, peck_mm: float, feed_mm_min: float, rpm: float, x_mm: float = 20.0,
                  y_mm: float = 15.0) -> str:
    return f"""%
O01003 (G83 PECK DRILL)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T1 M06
G90 G54 G00 X{x_mm:.3f} Y{y_mm:.3f} S{rpm:.0f} M03
G43 H01 Z25. M08
G98 G83 X{x_mm:.3f} Y{y_mm:.3f} Z-{depth_mm:.3f} R2. Q{peck_mm:.3f} F{feed_mm_min:.1f}
G80
G00 Z25.
M05
M30
%
"""


def drill_project(tmp: Path, depth_mm: float, peck_mm: float, feed_mm_min: float, rpm: float, *,
                  lattice_mm: float = 0.1) -> Path:
    return project_variant(POCKET, tmp, program_text=drill_program(depth_mm, peck_mm, feed_mm_min, rpm),
                           program_name="drill.nc", edits=lambda d: _block(d, (40.0, 30.0, 12.0), lattice_mm,
                                                                           DRILL_D6_8))


def station_ap_mm(tool, n_stations: int) -> float:
    """Depth of cut placing the stock top midway between peripheral station n_stations - 1 and n_stations, so the
    top station lies well inside a lattice cell from the surface (docs/models/hotloop.md "Field resolution")."""
    from cnctwin.cutting.memory import station_geometry
    r, z, _, _ = station_geometry(tool)
    R = float(np.max(r))
    zs = np.sort(z[r > 0.98 * R])
    return 1e3 * 0.5 * (float(zs[n_stations - 1]) + float(zs[n_stations]))


def kienzle_model(slot, geometry):
    """Verification coefficients for the drilling and turning tests (cutting's LinearModel): Kienzle with the
    al6061-t6 database entry (kc1.1 = 800 N/mm2, mc = 0.25; examples/pocket-islands-al6061/EXPECTED.md), thrust
    and passive ratios and edge forces of the order of the aluminium values in Altintas 2012 (estimates; the
    tests compare the loop with cutting's stand-alone kernel on the same table, not with measurements)."""
    from cnctwin.core.abi import EDGE_CHISEL, EDGE_MARGIN
    from cnctwin.cutting.coefficients import LinearModel
    base = {"kc11": 800e6, "mc": 0.25, "ratio_r": 0.45, "ratio_a": 0.15, "Kte": 15e3, "Kre": 25e3, "Kae": 0.0,
            "h_min_m": 1e-6}
    return LinearModel({"default": base, EDGE_CHISEL: {"Kte": 30e3, "Kre": 90e3, "h_min_m": 1e-6},
                        EDGE_MARGIN: {"Kte": 0.0, "Kre": 0.0, "h_min_m": 1e-6}})


def run_until_done(drv: OperationDriver, max_steps: int = 5000):
    n = 0
    while not drv.feeder.finished() and n < max_steps:
        drv.step()
        n += 1
    return drv.result(drv.feeder.finished())


def steady_steps(drv: OperationDriver, x_lo: float, x_hi: float):
    """Macro steps entirely in cut with the tool tip between x_lo and x_hi (S) during the whole step."""
    return [st for st in drv.steps if st.in_cut_s >= 0.0099 and getattr(st, "x_range", (np.inf, -np.inf))[0] > x_lo
            and getattr(st, "x_range", (np.inf, -np.inf))[1] < x_hi]


class Recorder:
    """Per-step tool tip range (S x) next to the driver's StepRecord (the steady window of a straight cut)."""

    def __init__(self, drv: OperationDriver):
        self.drv = drv

    def run(self, max_steps: int = 5000):
        drv = self.drv
        n = 0
        while not drv.feeder.finished() and n < max_steps:
            rec = drv.step()
            frame = drv.state.bus_prev.motion.paths[0]
            xs = np.asarray(frame.tip_S_m)[:, 0]
            rec.x_range = (float(xs.min()), float(xs.max()))
            tips = np.asarray(frame.tip_S_m)
            rec.tip_yz = (float(np.mean(tips[:, 1])), float(np.mean(tips[:, 2])))
            rec.z_range = (float(tips[:, 2].min()), float(tips[:, 2].max()))
            rec.z_first_last = (float(tips[0, 2]), float(tips[-1, 2]))
            rec.omega = float(np.mean(frame.spindle_omega_rad_s))
            n += 1
        return drv.result(drv.feeder.finished())


def slot_reference(drv: OperationDriver, ap_m: float, omega: float, vf_m_s: float, dt_s: float, revs: int = 6):
    """cutting's stand-alone Driver with the run's own tool and table: straight full slot along +x (T0 = S axes),
    the peripheral stations below ap engaged over the front half, memory prefilled on the steady path. Returns
    (mean force T0 [3], peak |F| over the revolutions, chip volume per second)."""
    from cnctwin.cutting import abi as CA
    from cnctwin.cutting.drive import Driver, straight_immersion
    from cnctwin.cutting.memory import station_geometry
    state = drv.state
    tc = state.cutting.tools[0]
    tool, table = tc.tool, tc.table
    nt, ne = np.asarray(state.tooling.effective[0].elem_r_m).shape
    ref = Driver(tool, table, nt, ne, memory_bins=int(state.cutting.settings.memory_bins), rho_kg_m3=drv.rho)
    r, z, _, _ = station_geometry(tool)
    R = float(np.max(r))
    stations = [s for s in range(len(z)) if z[s] < ap_m and r[s] > 0.98 * R]
    lo, hi = straight_immersion(0.0, math.pi)
    ref.set_interval(lo, hi, stations)
    tip = lambda t: np.array([vf_m_s * t, 0.0, 0.0])      # noqa: E731
    ref.prefill_memory(tip, 0.0, omega)
    n = int(round(2.0 * math.pi / abs(omega) / dt_s))
    ref.run_rotating(2 * n, omega, dt_s, tip, record=False)
    t0 = 2 * n * dt_s
    v0 = float(ref.acc.scalars[CA.ACC_CHIP_VOLUME_M3])
    rec = ref.run_rotating(revs * n, omega, dt_s, tip, t0_s=t0)
    W = np.array(rec.wrench)
    rate = (float(ref.acc.scalars[CA.ACC_CHIP_VOLUME_M3]) - v0) / (revs * n * dt_s)
    return W[:, :3].mean(axis=0), float(np.max(np.linalg.norm(W[:, :3], axis=1))), rate


def lattice_slot_volume(field, x0: float, x1: float, y: float, radius: float, depth: float, top: float) -> float:
    """z-family volume of a straight slot on the field's level-0 lattice: every column whose node lies within
    ``radius`` of the tool path segment (and inside the stock) loses ``depth`` (the field's own column model)."""
    a = float(field.lv_cell_m[0])
    o = field.lv_origin_m[0]
    nx, ny = int(field.lv_dims[0, 0]), int(field.lv_dims[0, 1])
    xs = o[0] + a * np.arange(nx)
    ys = o[1] + a * np.arange(ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    lo, hi = field.lv_box_lo_m[0], field.lv_box_hi_m[0]
    inside = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
    t = np.clip(X, x0, x1)
    d = np.hypot(X - t, Y - y)
    return float(np.count_nonzero(inside & (d <= radius))) * a * a * depth


def profile_height(prof, r: float, dr: float) -> float:
    """Lowest height z of a removal Profile (lines only) where R(z) + dr reaches radius r (inf: never)."""
    for i in range(len(prof.z0_m)):
        z0, z1 = float(prof.z0_m[i]), float(prof.z1_m[i])
        c0, c1 = float(prof.c0[i]), float(prof.c1[i])
        R0 = c0 + c1 * z0 + dr
        R1 = c0 + c1 * z1 + dr
        if r <= R0:
            return z0
        if r <= R1 and c1 > 0.0:
            return (r - dr - c0) / c1
    return math.inf


def lattice_hole_volume(field, prof, centre_xy, dr: float, z_tip: float, top: float) -> float:
    """z-family volume of a drilled hole on the field's level-0 lattice: every column within the drill's envelope
    loses the length from the stock top down to the envelope's surface at its radius."""
    a = float(field.lv_cell_m[0])
    o = field.lv_origin_m[0]
    nx, ny = int(field.lv_dims[0, 0]), int(field.lv_dims[0, 1])
    total = 0.0
    for i in range(nx):
        x = o[0] + a * i
        for j in range(ny):
            y = o[1] + a * j
            r = math.hypot(x - centre_xy[0], y - centre_xy[1])
            zb = profile_height(prof, r, dr)
            if zb < math.inf:
                total += max(0.0, top - (z_tip + zb))
    return total * a * a


def air_program(dwell_s: float = 2.0, x0_mm: float = -60.0, x1_mm: float = 140.0, rpm: float = SPEED_RPM) -> str:
    return f"""%
O01004 (SPINDLE AT SPEED IN AIR, THEN A LONG RAPID)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T1 M06
G90 G54 G00 X{x0_mm:.3f} Y15. S{rpm:.0f} M03
G43 H01 Z50.
G04 P{dwell_s:.3f}
G00 X{x1_mm:.3f}
G04 P0.3
M05
M30
%
"""


def air_project(tmp: Path, **program) -> Path:
    return project_variant(POCKET, tmp, program_text=air_program(**program), program_name="air.nc",
                           edits=lambda d: _block(d, (40.0, 30.0, 10.0), 0.25))


def one_mode_machine(db, fn_Hz: float, zeta: float, k_N_um: float, direction: str = "x") -> dict:
    """The VF-2SS with one flexible mode only: a spindle FRF mode carrying the rigid tool (no frame, table or
    rotary modes): the 1-DOF regenerative system of the chatter test, written inline with extends."""
    import copy
    obj = db.object("machine", "haas-vf2ss")
    raw = copy.deepcopy(obj.merged if obj.merged is not None else obj.raw)
    spindles = raw["spindles"]
    spindles[0]["frf_modes"] = [{"fn_Hz": fn_Hz, "zeta": zeta, "k_N_um": k_N_um, "direction": direction,
                                 "src": "project"}]
    structure = dict(raw.get("structure") or {})
    structure.update({"modes": [], "table_modes": [], "rotary_modes": []})
    return {"extends": "haas-vf2ss", "spindles": spindles, "structure": structure}


def chatter_project(tmp: Path, db, ap_mm: float, fn_Hz: float, zeta: float, k_N_um: float, **program) -> Path:
    def edit(d):
        _block(d, (40.0, 30.0, 10.0), 0.1)
        d["machine"] = one_mode_machine(db, fn_Hz, zeta, k_N_um)
    return project_variant(POCKET, tmp, program_text=slot_program(ap_mm, **program), program_name="slot.nc",
                           edits=edit)


# ------------------------------------------------------------------------------------------------- turning
LATHE = Path(EXAMPLES) / "lathe-shaft-4140"
TURN_TOOL = "tungaloy-cnmg120408-tm-t9225-pclnr2525m12e"   # CNMG120408-TM T9225 in PCLNR2525M12E: kappa_r 95, r 0.8
TURN_RPM = 1400.0                                           # G97: vc 198 m/min on d45
TURN_FEED_MM_REV = 0.25


def turn_program(d_mm: float, ap_mm: float, z1_mm: float, f_mm_rev: float = TURN_FEED_MM_REV,
                 rpm: float = TURN_RPM, spindle: str = "M04") -> str:
    """One straight OD pass at constant speed (G97) from Z2 to z1 at the diameter d - 2 ap (diameter words).

    M04 by default: with motion's ISO 841 sign (M03 turns the work counter-clockwise seen from the tailstock)
    tooling's right-hand turning tool (hand sign -1, intended cutting velocity h y_T) cuts along its rake face
    only in M04 on a turret at +X (docs/models/hotloop.md "Open problems": the lathe example's M03)."""
    x = d_mm - 2.0 * ap_mm
    return f"""%
O01005 (OD TURNING PASS)
N10 G21 G18 G40 G80 G99
N20 G54
N30 G50 S2500
N100 G53 G00 X0
N110 T0101
N120 G97 S{rpm:.0f} {spindle}
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


def _bar(data, d_mm: float, length_mm: float, lattice_mm: float):
    """The lathe example with a plain d x length bar chucked 30 mm deep in the soft jaws (no tailstock, no CAD),
    the CNMG roughing tool alone. G54 Z0 on the free end (z 99 + length on the spindle face frame)."""
    data["cad"] = None
    data["physics"]["dynamics"] = dict(DYNAMICS_FAST_L2)
    data["physics"]["cutting"] = {"every_n_macro": CUTTING_EVERY_N_MACRO}
    data["physics"]["lattice_mm"] = lattice_mm
    data["stock"] = {"kind": "cylinder", "size_mm": [d_mm, length_mm], "tolerance_mm": None,
                     "non_squareness_deg": None, "form": "cold_drawn_bar", "mesh": None}
    data["tools"]["list"] = [data["tools"]["list"][0]]
    data["tools"]["list"][0]["tool"] = TURN_TOOL
    op = data["operations"][0]
    op["setup"]["workholding"] = op["setup"]["workholding"][:2]
    op["setup"]["stock_pose"] = {"translate_mm": [0, 0, 99], "rotate_deg": [0, 0, 0]}
    op["setup"]["work_offsets_mm"] = {"G54": [-426.0, 0.0, 99.0 + length_mm - 662.0]}


def turn_project(tmp: Path, d_mm: float, length_mm: float, ap_mm: float, z1_mm: float, *, lattice_mm: float = 0.5,
                 **program) -> Path:
    return project_variant(LATHE, tmp, program_text=turn_program(d_mm, ap_mm, z1_mm, **program),
                           program_name="turn.nc", edits=lambda d: _bar(d, d_mm, length_mm, lattice_mm))


# ------------------------------------------------------------------------------------------------- pocket-islands
def pocket_program(n_arcs: int = 40, helix_turns: int | None = None) -> str:
    """examples/pocket-islands-al6061/op10.nc without the probe section (N10: the probe stylus is not modelled in
    the hot loop yet, docs/models/hotloop.md "Open problems"): T1's helical entry (all of it, or its first
    ``helix_turns`` turns) and the first ``n_arcs`` trochoidal arcs, then the tool change position and M30."""
    lines = (POCKET / "op10.nc").read_text().splitlines()
    head = lines[:lines.index("G53 G00 Z0.") + 1]
    i20 = next(i for i, ln in enumerate(lines) if ln.startswith("N20"))
    i_troch = next(i for i, ln in enumerate(lines) if ln.startswith("(TROCHOIDAL LOOPS"))
    body = lines[i20:i_troch + 1]
    if helix_turns is not None:
        i_helix = next(i for i, ln in enumerate(body) if ln.startswith("(HELICAL ENTRY"))
        helix = [ln for ln in body[i_helix + 1:] if ln.startswith("G03") and " Z" in ln]
        keep = set(helix[helix_turns:])
        body = [ln for ln in body if ln not in keep and not ln.startswith("G03 X50. Y13.3 I0. J2.7")]
    arcs = [ln for ln in lines[i_troch + 1:] if ln.startswith("G03")][:n_arcs]
    tail = ["G00 Z25.", "M09", "M05", "G53 G49 Z0.", "M30", "%"]
    return "\n".join(head + body + arcs + tail) + "\n"


def pocket_project(tmp: Path, *, n_arcs: int = 40, helix_turns: int | None = None, fast_l2: bool = True) -> Path:
    """The example itself (stock, CAD, vise, lattice 0.075 mm, both end mills) with :func:`pocket_program`; G54 Z on
    the nominal top (the probe of N10 would set it), the dynamics' L2 lobes on coarse grids."""
    def edit(d):
        d["operations"][0]["setup"]["work_offsets_mm"]["G54"][2] = -480.0
        if fast_l2:
            d["physics"]["dynamics"] = dict(DYNAMICS_FAST_L2)
            d["physics"]["cutting"] = {"every_n_macro": CUTTING_EVERY_N_MACRO}
    return project_variant(POCKET, tmp, program_text=pocket_program(n_arcs, helix_turns), program_name="op10.nc",
                           edits=edit)
