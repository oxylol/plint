"""Run the levels of a convergence study and write tools/convergence/results/<study>.json (docs/convergence.md).

    bin/tsafe python tools/convergence/study.py <study> [--root DIR] [--levels i,j,...] [--fresh]

Studies (``--list``): the L0 stride on a slot and on a side-milling finish, the dexel lattice on the side finish and
on a pocket corner, the edge element size for a flat end mill and a ball nose, the surface-memory bins (forces with
runout and regeneration; the chatter onset of a 1-DOF system), the polar field of a turning pass (theta bins; row
height), and with every domain on the macro step, the coupling lag and the workpiece thermal grid (cell; cadence).
Levels already in the results file are kept (``--fresh`` recomputes them); each level is one run of the simulator,
deterministic, in this process. One study at a time (the box is shared; bin/tsafe caps the memory).
"""

from __future__ import annotations

import sys

import env

ROOT = env.setup()

import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import cases as C  # noqa: E402
import measure as M  # noqa: E402
from analysis import analyse  # noqa: E402

RESULTS = env.HERE / "results"
WORK = Path("/tmp") / "cnctwin-convergence"
TEETH_D10 = 4
IPO_S = 1e-3
DEG_PER_TICK = 360.0 * C.RPM / 60.0 * 1e-6          # spindle angle per base tick at 10 000 rpm


# ------------------------------------------------------------------------------------------------ tolerances
# The documented tolerance of each quantity (docs/convergence.md "Tolerances"): one tenth of the validation budget of
# the quantity (validation/budgets.yaml) where there is one; (value, relative?, noise floor).
TOL = {
    "F_mean_N": (0.015, True, 1e-4),            # cutting_force main 15 %
    "Fx_N": (0.015, True, 1e-4), "Fy_N": (0.015, True, 1e-4), "Fz_N": (0.015, True, 1e-4),
    "torque_Nm": (0.015, True, 1e-4),           # torque 15 %
    "F_peak_N": (0.02, True, 1e-4),             # milling peak force 20 %
    "F_peak_corner_N": (0.02, True, 1e-4),
    "chip_volume_field_mm3": (0.01, True, 1e-5),     # half the 2 % volume-consistency band
    "chip_volume_analytic_mm3": (0.01, True, 1e-5),
    "removed_mm3": (0.01, True, 1e-6),
    "engaged_deg": (0.01, True, 1e-5),
    "engaged_max_deg": (0.01, True, 1e-5),
    "wall_dev_um": (1.0, False, 0.02),          # 1 um: a tenth of a +-10 um finishing tolerance
    "wall_lo_dev_um": (1.0, False, 0.02), "wall_hi_dev_um": (1.0, False, 0.02),
    "wall_bottom_dev_um": (1.0, False, 0.02), "wall_right_dev_um": (1.0, False, 0.02),
    "chip_tooth_spread": (0.01, False, 1e-4),   # absolute share of the mean chip
    "vib_pp_um": (0.025, True, 1e-4),           # wall_error 25 %
    "onset_depth_mm": (0.025, True, 1e-4),      # stability_lobe 25 %
    "Ra_um": (0.03, True, 1e-4), "Rt_um": (0.03, True, 1e-4),   # roughness 30 %
    "tool_rise_K": (0.015, True, 1e-4), "edge_rise_K": (0.015, True, 1e-4), "rake_rise_K": (0.015, True, 1e-4),
    "VB_growth_um": (0.05, True, 1e-4),         # taylor_life / VB(t) 50 %
    "bulk_rise_K": (0.015, True, 1e-4), "bulk_max_rise_K": (0.015, True, 1e-4),
    "surface_rise_K": (0.015, True, 1e-4), "preheat_rise_K": (0.015, True, 1e-4),
    "rake_rise_mean_K": (0.015, True, 1e-4), "edge_rise_mean_K": (0.015, True, 1e-4),
    "preheat_rise_mean_K": (0.015, True, 1e-4),
    "expansion_um": (0.1, False, 0.002),        # 0.1 um: a hundredth of a +-10 um tolerance (the expansion is small)
}


# ------------------------------------------------------------------------------------------------ helpers
def template_physics() -> dict:
    """The shipped defaults: the physics section of db/_templates/project.yaml of the tree studied (without
    ``resolved``). Every case starts from it; a study changes only its own settings (and states any other
    exception in its note)."""
    import yaml
    from cnctwin.paths import DB
    data = yaml.safe_load((Path(DB) / "_templates" / "project.yaml").read_text())
    phys = dict(data["physics"])
    phys.pop("resolved", None)
    return phys


def level_physics(level, **case) -> dict:
    """Template defaults, then the case's own settings, then the level's."""
    phys = template_physics()
    C._merge(phys, {k: v for k, v in case.items() if v is not None})
    C._merge(phys, level.physics)
    return phys


def _driver(project, db, *, dynamics: bool, dynamics_l2: bool = False):
    from cnctwin.core.drive import OperationDriver
    return OperationDriver(project, db=db, use_dynamics=dynamics, dynamics_l2=dynamics_l2)


def _tick_mask_x(run, x_lo_mm, x_hi_mm, y_mm=None):
    a = run.tick_arrays()
    x = a["tip"][:, 0] * 1e3
    m = (x >= x_lo_mm) & (x <= x_hi_mm)
    if y_mm is not None:
        m &= np.abs(a["tip"][:, 1] * 1e3 - y_mm) < 0.05
    return m


def profile_radius(prof, z: float) -> float:
    """R(z) of a removal Profile (0 outside it)."""
    from cnctwin.removal import abi as RA
    for i in range(len(prof.z0_m)):
        z0, z1 = float(prof.z0_m[i]), float(prof.z1_m[i])
        if z0 - 1e-15 <= z <= z1 + 1e-15:
            c0, c1, c2, c3 = (float(prof.c0[i]), float(prof.c1[i]), float(prof.c2[i]), float(prof.c3[i]))
            if int(prof.kind[i]) == RA.PK_LINE:
                return c0 + c1 * z
            return c1 + c3 * math.sqrt(max(c2 * c2 - (z - c0) ** 2, 0.0))
    return 0.0


def envelope_radius(shape, z: float) -> float:
    """Radius of the swept envelope of all teeth at height z above the tip: R(z) + the largest tooth offset of the
    slab holding z (runout, removal's ToothSlabs)."""
    R = profile_radius(shape.cut, z)
    sl = shape.slabs
    best = -math.inf
    for j in range(sl.z.shape[0]):
        n = int(sl.n[j])
        for k in range(n):
            if sl.z[j, k] - 1e-12 <= z <= sl.z[j, k + 1] + 1e-12:
                best = max(best, float(sl.dr[j, k]))
    return R + (best if math.isfinite(best) else 0.0)


def _cap(R: float, d: float) -> float:
    """Area of the circular segment of a disc of radius R that reaches d beyond a straight wall at distance R - d
    from its centre, on one side of the centre's normal to the wall: integral_0^u (sqrt(R^2 - t^2) - (R - d)) dt."""
    if d <= 0.0:
        return 0.0
    u = math.sqrt(max(R * R - (R - d) ** 2, 0.0))
    return 0.5 * R * R * math.asin(min(u / R, 1.0)) - 0.5 * u * (R - d)


def corner_area(R: float, d_bottom: float, d_right: float, xc: float, yc: float, ye: float) -> float:
    """Area a disc of radius R removes along (x < 0) -> (xc, yc) -> (xc, ye) from a stock whose pocket walls lie
    d_bottom below the disc on the first segment and d_right to its right on the second (the bottom strip from x = 0,
    the corner below the turn, the right strip, the corner beside the turn and the end cap)."""
    return (max(d_bottom, 0.0) * xc + _cap(R, d_bottom) + max(d_right, 0.0) * (ye - yc) + 2.0 * _cap(R, d_right))


def swept_wall_volume(path_xy_mm, shape, ap_mm: float, y_wall_mm: float, x_wall_mm: float,
                      pixel_mm: float = 0.002, band_mm: float = 0.6) -> float:
    """Volume (m3) the envelope of every tooth (radius R(z) + the largest tooth offset, envelope_radius) removes when
    its axis follows the polyline ``path_xy_mm`` (S, mm) through the walls below y_wall and right of x_wall of the
    corner stock: per distinct radius the wall bands are sampled on a pixel_mm grid and a pixel is removed when its
    distance to the path (resampled every 5 um) is within the radius; integrated over the depth from the tip."""
    from scipy.spatial import cKDTree
    P = np.asarray(path_xy_mm, float)
    seg = np.diff(P, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    pts = [P[:1]]
    for i in range(len(seg)):
        n = max(int(math.ceil(length[i] / 0.005)), 1)
        t = (np.arange(1, n + 1) / n)[:, None]
        pts.append(P[i] + t * seg[i])
    tree = cKDTree(np.concatenate(pts, axis=0))
    xb = np.arange(0.0, x_wall_mm + band_mm, pixel_mm) + 0.5 * pixel_mm
    yb = np.arange(y_wall_mm - band_mm, y_wall_mm, pixel_mm) + 0.5 * pixel_mm
    xr = np.arange(x_wall_mm, x_wall_mm + band_mm, pixel_mm) + 0.5 * pixel_mm
    yr = np.arange(y_wall_mm, P[:, 1].max() + 6.0, pixel_mm) + 0.5 * pixel_mm
    g1 = np.stack(np.meshgrid(xb, yb, indexing="ij"), axis=-1).reshape(-1, 2)
    g2 = np.stack(np.meshgrid(xr, yr, indexing="ij"), axis=-1).reshape(-1, 2)
    g2 = g2[g2[:, 1] >= y_wall_mm]
    d = np.concatenate([tree.query(g1)[0], tree.query(g2)[0]])
    d.sort()
    zs = np.linspace(0.0, ap_mm, 501)
    radii = np.array([1e3 * envelope_radius(shape, z * 1e-3) for z in zs])
    areas = np.searchsorted(d, radii, side="right") * pixel_mm * pixel_mm
    return float(np.trapezoid(areas, zs)) * 1e-9


# ------------------------------------------------------------------------------------------------ cases
@dataclass
class Level:
    label: str
    h: float                       # step size of the refined setting (smaller = finer)
    physics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def mill_case(study, level: Level, db, *, kind: str, dynamics: bool, lattice_mm: float | None = None,
              tool: str = C.END_MILL_D10, ap_mm: float = 3.0, machine=None, runout_um=None, ae_mm: float = 0.5):
    """One run of a milling case; returns (quantities, cost, details)."""
    phys = level_physics(level, lattice_mm=lattice_mm)
    work = WORK / study / level.label
    if work.exists():
        shutil.rmtree(work)
    if kind == "slot":
        size = (25.0, 30.0, 10.0)
        program = C.line_program(ap_mm, 15.0, -8.0, 17.0)
        window = (1.5, 16.0)
    elif kind == "side":
        size = (25.0, 12.0, 10.0)
        program = C.line_program(ap_mm, size[1] + 5.0 - ae_mm, -8.0, 30.0)
        half = math.sqrt(25.0 - (5.0 - ae_mm) ** 2)         # the engaged arc spans x_c .. x_c + half (mm)
        window = (1.0, size[0] - half - 0.5)
    elif kind == "chatter":
        size = (25.0, 30.0, 10.0)
        program = C.line_program(ap_mm, 15.0, -8.0, 20.0)
        window = (2.0, 19.0)
    else:
        raise ValueError(kind)
    project = C.mill_project(work, program, C.block_edits(size, tool=tool, physics=phys, machine=machine,
                                                          runout_um=runout_um))
    drv = _driver(project, db, dynamics=dynamics, dynamics_l2=(kind == "chatter"))
    v0 = M.field_volume_m3(drv.state.removal.workpieces[0].field)
    run = M.DriverRun(drv, dynamics=dynamics).run()
    mask = _tick_mask_x(run, *window)
    q = M.steady_means(run, mask, teeth=TEETH_D10, rho=drv.rho, ipo_s=IPO_S)
    field_ = run.result.field
    top = float(field_.lv_box_hi_m[0, 2])
    shape = drv.state.removal.tools[0]
    details = {"removed_total_mm3": 1e9 * run.result.removed_volume_m3,
               "analytic_total_mm3": 1e9 * run.result.analytic_chip_volume_m3,
               "finished": bool(run.result.finished)}
    z_lo, z_hi = top - ap_mm * 1e-3 + 0.35e-3, top - 0.35e-3
    if kind == "slot":
        R = envelope_radius(shape, 1.5e-3)
        lo = M.wall_samples(field_, 1, 15e-3 - R, (3e-3, 14e-3), (z_lo, z_hi))
        hi = M.wall_samples(field_, 1, 15e-3 + R, (3e-3, 14e-3), (z_lo, z_hi))
        if lo.size:
            q["wall_lo_dev_um"] = float(1e6 * (np.mean(lo) - (15e-3 - R)))
        if hi.size:
            q["wall_hi_dev_um"] = float(1e6 * (np.mean(hi) - (15e-3 + R)))
        details["slot_width_mm"] = float(1e3 * (np.mean(hi) - np.mean(lo))) if lo.size and hi.size else None
    elif kind == "side":
        R = envelope_radius(shape, 1.5e-3)
        nominal = (size[1] + 5.0 - ae_mm) * 1e-3 - R
        s = M.wall_samples(field_, 1, nominal, (3e-3, 20e-3), (z_lo, z_hi))
        q.update({k: v for k, v in M.wall_stats(s, nominal).items() if k in ("wall_dev_um",)})
        details.update(M.wall_stats(s, nominal))
        # the removed volume against the swept envelope (rigid: exact geometry; with the modes: the deflection too)
        zs = np.linspace(0.0, ap_mm * 1e-3, 4001)
        d = np.array([max(envelope_radius(shape, z) - 5e-3 + ae_mm * 1e-3, 0.0) for z in zs])
        V = size[0] * 1e-3 * float(np.trapezoid(d, zs))
        q["removed_mm3"] = 1e9 * run.result.removed_volume_m3
        details["removed_analytic_mm3"] = 1e9 * V
        details["removed_xyz_mm3"] = [1e9 * (a - b) for a, b in zip(v0, M.field_volume_m3(field_))]
        # every y ray across the wall loses exactly the cut depth at its (x, z): the y family is the exact removed
        # volume of this wall up to its (x, z) sampling (the reference at the finest lattice)
        details["removed_y_mm3"] = details["removed_xyz_mm3"][1]
    details["cost"] = run.cost()
    details["steps_cut"] = sum(1 for s in run.steps if s["in_cut_s"] > 0)
    tool = drv.state.cutting.tools[0].tool if drv.state.cutting.tools else None
    if tool is not None:
        # the cutting edge's stations around the depth of cut: an element is engaged whole or not at all, so the
        # engaged edge ends at a station boundary, not at the stock's top
        from cnctwin.cutting.memory import station_geometry
        r_st, z_st, _, _ = station_geometry(tool)
        zs = np.sort(np.asarray(z_st, float)[np.asarray(r_st) > 0.3 * float(np.max(r_st))]) * 1e3
        below = zs[zs < ap_mm]
        above = zs[zs >= ap_mm]
        details["stations_around_ap_mm"] = [float(below[-1]) if below.size else None,
                                            float(above[0]) if above.size else None]
        details["stations_below_ap"] = int(below.size)
    if kind == "chatter":
        steady = [s for s in run.steps if s["in_cut_s"] > 0.0 and s["tip_lo"][0] * 1e3 > window[0]
                  and s["tip_hi"][0] * 1e3 < window[1]]
        late = steady[len(steady) // 2:]
        details["chatter_share"] = float(np.mean([bool(s.get("chatter")) for s in late])) if late else 0.0
        details["chatter_index"] = float(np.median([s.get("chatter_index", 0.0) for s in late])) if late else 0.0
        details["vib_pp_um_late"] = float(1e6 * np.median([s.get("vib_pp_m", 0.0) for s in late])) if late else 0.0
        details["depth_limit_mm"] = float(1e3 * np.median([s.get("depth_limit_m", np.inf) for s in late])) \
            if late else None
    return q, run.cost(), details


def corner_case(study, level: Level, db, *, ap_mm: float = 5.0, ae_mm: float = 0.3):
    """The pocket corner (rigid): a meshed stock 36 x 26 x 8 mm with a channel open to the left face (walls at
    y = 5 and x = 30, floor 6 mm deep); a finishing pass ae into the bottom wall, through the internal corner and
    ae into the right wall, climb milling, ap 5 mm."""
    phys = level_physics(level)
    work = WORK / study / level.label
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    size = (36.0, 26.0, 8.0)
    y0, x1, y1, floor = 5.0, 30.0, 21.0, 2.0
    C.write_pocket_stl(work / "stock.stl", size, (-1.0, x1, y0, y1, floor))
    R0 = 5.0
    yc, xc, ye = y0 + R0 - ae_mm, x1 - R0 + ae_mm, 15.5
    program = C.corner_program(ap_mm, [(-8.0, yc), (xc, yc), (xc, ye)])
    project = C.mill_project(work / "project", program,
                             C.block_edits(size, physics=phys, stock_mesh=str(work / "stock.stl")))
    drv = _driver(project, db, dynamics=False)
    v0 = M.field_volume_m3(drv.state.removal.workpieces[0].field)
    run = M.DriverRun(drv, dynamics=False).run()
    a = run.tick_arrays()
    tip = a["tip"] * 1e3
    # steady part of the bottom wall: x in [2, xc - 6] on y = yc
    mask = (tip[:, 0] >= 2.0) & (tip[:, 0] <= xc - 6.0) & (np.abs(tip[:, 1] - yc) < 0.05)
    q = M.steady_means(run, mask, teeth=TEETH_D10, rho=drv.rho, ipo_s=IPO_S)
    q = {k: q[k] for k in ("F_mean_N", "F_peak_N", "engaged_deg") if k in q}
    # the corner: the tool within 6 mm of the turn
    near = (np.hypot(tip[:, 0] - xc, tip[:, 1] - yc) < 6.0)
    if near.any():
        q["F_peak_corner_N"] = float(np.max(np.linalg.norm(a["P"][near][:, :3], axis=1)))
    steps = [s for s in run.steps if s["in_cut_s"] > 0.0]
    q["engaged_max_deg"] = float(np.degrees(max(s["engaged_rad"] for s in steps)))
    field_ = run.result.field
    top = float(field_.lv_box_hi_m[0, 2])
    shape = drv.state.removal.tools[0]
    z_lo, z_hi = top - ap_mm * 1e-3 + 0.35e-3, top - 0.35e-3
    Rw = envelope_radius(shape, 1.5e-3)
    nb = (yc * 1e-3) - Rw
    nr = (xc * 1e-3) + Rw
    sb = M.wall_samples(field_, 1, nb, (3e-3, (xc - 6.0) * 1e-3), (z_lo, z_hi))
    sr = M.wall_samples(field_, 0, nr, ((yc + 3.0) * 1e-3, (ye - 1.0) * 1e-3), (z_lo, z_hi))
    if sb.size:
        q["wall_bottom_dev_um"] = float(1e6 * (np.mean(sb) - nb))
    if sr.size:
        q["wall_right_dev_um"] = float(1e6 * (np.mean(sr) - nr))
    # the exact removed volume: the tool's envelope swept along the path it actually took (the tick positions: the
    # control rounds the corner by up to ~0.45 mm and the machine's errors offset the lines by a few um), intersected
    # with the stock's walls (swept_wall_volume); the programmed sharp-cornered path for comparison
    cut = (tip[:, 2] < (top * 1e3 - ap_mm + 0.05)) & (tip[:, 0] > -6.0)
    Vp = swept_wall_volume(tip[cut, :2], shape, ap_mm, y0, x1)
    zs = np.linspace(0.0, ap_mm * 1e-3, 2001)
    A = []
    for z in zs:
        R = envelope_radius(shape, z)
        A.append(corner_area(R, R - (yc - y0) * 1e-3, R - (x1 - xc) * 1e-3, xc * 1e-3, yc * 1e-3, ye * 1e-3))
    V = float(np.trapezoid(np.asarray(A), zs))
    v1 = M.field_volume_m3(field_)
    q["removed_mm3"] = 1e9 * run.result.removed_volume_m3
    details = {"removed_analytic_mm3": 1e9 * V, "removed_path_mm3": 1e9 * Vp,
               "removed_xyz_mm3": [1e9 * (b - c) for b, c in zip(v0, v1)],
               "finished": bool(run.result.finished), "cost": run.cost(),
               "wall_bottom_n": int(sb.size), "wall_right_n": int(sr.size)}
    return q, run.cost(), details


def turn_case(study, level: Level, db, *, d_mm: float = 45.0, length_mm: float = 40.0, ap_mm: float = 1.5,
              f_mm_rev: float = 0.25, rpm: float = 1400.0):
    """The OD turning pass (rigid): 4140 bar d45 x 40 mm, CNMG120408 in PCLNR, 1400 rpm M04, f 0.25 mm/rev,
    ap 1.5 mm, from Z2 to Z-5 (5 mm of cut)."""
    phys = level_physics(level, edge_element_mm=0.1)
    work = WORK / study / level.label
    if work.exists():
        shutil.rmtree(work)
    z1 = -5.0
    project = C.turn_project(work, C.turn_program(d_mm, ap_mm, z1, f_mm_rev, rpm),
                             C.bar_edits(d_mm, length_mm, physics=phys))
    drv = _driver(project, db, dynamics=False)
    run = M.DriverRun(drv, dynamics=False).run()
    a = run.tick_arrays()
    z = a["tip"][:, 2] * 1e3
    lo, hi = length_mm - 4.0, length_mm - 1.0          # tip z in S: 1 mm into the cut to 1 mm before its end
    mask = (z >= lo) & (z <= hi)
    q = M.steady_means(run, mask, teeth=1, rho=drv.rho, ipo_s=IPO_S)
    q = {k: q[k] for k in ("F_mean_N", "Fx_N", "Fy_N", "Fz_N", "torque_Nm") if k in q}
    pf = drv.state.removal.workpieces[0].polar
    zc, r = M.turned_profile(pf, (length_mm - 4.0) * 1e-3, (length_mm - 1.5) * 1e-3)
    rough = M.roughness(zc, r)
    q.update({k: rough[k] for k in ("Ra_um", "Rt_um") if k in rough})
    from cnctwin.removal import abi as RA
    g = pf.geom
    r_nose = 0.8e-3
    f = f_mm_rev * 1e-3
    rt_exact = r_nose - math.sqrt(r_nose ** 2 - (f / 2) ** 2)
    u = np.linspace(-f / 2, f / 2, 200001)
    prof = r_nose - np.sqrt(r_nose ** 2 - u ** 2)
    ra_exact = float(np.mean(np.abs(prof - prof.mean())))
    details = {"n_theta": int(g[RA.PG_NTHETA]), "dz_um": float(1e6 * g[RA.PG_DZ]), "nz": int(g[RA.PG_NZ]),
               "Ra_ideal_um": 1e6 * ra_exact, "Rt_ideal_um": 1e6 * rt_exact, "r_mean_mm": rough.get("r_mean_mm"),
               "notes": list(drv.state.removal.workpieces[0].notes), "finished": bool(run.result.finished),
               "removed_total_mm3": 1e9 * run.result.removed_volume_m3, "cost": run.cost()}
    return q, run.cost(), details


# ------------------------------------------------------------------------------------------------ chatter onset
def onset_case(study, level: Level, db, *, ap_mm: float, k_lo: float = 4.0, k_hi: float = 16.0, iterations: int = 7,
               fn_Hz: float = 900.0, zeta: float = 0.02, k_ref: float = 5.0):
    """Chatter onset of the 1-DOF slot (tests/integration/test_chatter.py's system: one spindle mode in x, 900 Hz,
    zeta 0.02) at a fixed depth ap, found by bisection over the modal stiffness k (log scale) between a chattering
    k_lo and a stable k_hi. For a linear regenerative system the limit depth is proportional to k, so the onset depth
    of the reference system (k_ref = 5 N/um) is ap k_ref / k_onset: a continuous measure that does not snap to the
    edge stations, as a bisection over ap would (the engaged edge changes by whole elements)."""
    probes = []
    total_cost = {"cpu_total_s": 0.0, "cpu_in_cut_s": 0.0, "runs": 0}

    def chatters(k):
        lv = Level(label=f"{level.label}-k{k:.4f}", h=level.h, physics=level.physics)
        machine = C.one_mode_machine(db, fn_Hz, zeta, k)
        q, cost, det = mill_case(study, lv, db, kind="chatter", dynamics=True, ap_mm=ap_mm, machine=machine,
                                 lattice_mm=None)
        total_cost["cpu_total_s"] += cost["cpu_total_s"]
        total_cost["cpu_in_cut_s"] += cost["cpu_in_cut_s"]
        total_cost["runs"] += 1
        c = det["chatter_share"] >= 0.5
        probes.append({"k_N_um": k, "chatter": c, "share": det["chatter_share"], "index": det["chatter_index"],
                       "vib_pp_um": det["vib_pp_um_late"], "depth_limit_mm": det["depth_limit_mm"],
                       "F_mean_N": q.get("F_mean_N")})
        shutil.rmtree(WORK / study / lv.label, ignore_errors=True)
        return c

    lo, hi = k_lo, k_hi
    if not chatters(lo) or chatters(hi):
        return {}, total_cost, {"probes": probes, "error": "the bracket does not straddle the onset"}
    for _ in range(iterations):
        mid = math.sqrt(lo * hi)
        if chatters(mid):
            lo = mid
        else:
            hi = mid
    k_on = math.sqrt(lo * hi)
    q = {"onset_depth_mm": ap_mm * k_ref / k_on}
    return q, total_cost, {"probes": probes, "k_onset_N_um": k_on, "k_bracket": [lo, hi],
                           "onset_bracket_mm": [ap_mm * k_ref / hi, ap_mm * k_ref / lo]}


# ------------------------------------------------------------------------------------------------ studies
def stride_levels(rigid: bool):
    rules = [(4.0, 24), (2.0, 48), (1.0, 96), (0.5, 96), (0.25, 96), (0.125, 96), (0.0625, 96)] if rigid else \
        [(2.0, 48), (1.0, 96), (0.5, 96), (0.25, 96), (0.125, 96), (0.0625, 96)]
    out = []
    for deg, spt in rules:
        out.append(Level(label=f"deg{deg:g}-spt{spt}", h=deg,
                         physics={"max_deg_per_step": deg, "steps_per_tooth": spt}))
    return out


@dataclass
class Study:
    name: str
    title: str
    setting: str
    h_name: str
    levels: list
    default_label: str
    quantities: list
    run: object
    note: str = ""
    h_from_cost: bool = False       # replace h by the stride actually used (degrees per L0 step)
    exact: dict = field(default_factory=dict)   # quantity -> exact value, or the details key holding it
    test_labels: list | None = None  # the levels tests/convergence runs (default: the shipped default and every finer)


def _stride_h(det_cost):
    st = det_cost.get("strides") or []
    return DEG_PER_TICK * max(st) if st else None


STUDIES = {
    "stride_slot": Study(
        "stride_slot", "L0 stride: D10 slot (rigid)", "physics.max_deg_per_step, physics.steps_per_tooth",
        "spindle angle per L0 step (deg)", stride_levels(True), "deg0.5-spt96",
        ["F_mean_N", "Fx_N", "Fy_N", "F_peak_N", "torque_Nm", "chip_volume_analytic_mm3", "chip_volume_field_mm3",
         "wall_lo_dev_um", "wall_hi_dev_um"],
        lambda st, lv, db: mill_case(st, lv, db, kind="slot", dynamics=False, ap_mm=3.0),
        note="D10 4FL (MHSV), Al 6061-T6, 10 000 rpm, f_z 0.04 mm, full slot ap 3.0 mm, lattice 0.25 mm (default), "
             "rigid tool and work", h_from_cost=True),
    "stride_side": Study(
        "stride_side", "L0 stride: side-milling finish (VF-2SS modes)",
        "physics.max_deg_per_step, physics.steps_per_tooth",
        "spindle angle per L0 step (deg)", stride_levels(False), "deg0.5-spt96",
        ["F_mean_N", "Fy_N", "F_peak_N", "torque_Nm", "chip_volume_analytic_mm3", "chip_volume_field_mm3",
         "wall_dev_um", "vib_pp_um"],
        lambda st, lv, db: mill_case(st, lv, db, kind="side", dynamics=True, ap_mm=5.0, lattice_mm=0.125),
        note="D10 4FL, Al 6061-T6, 10 000 rpm, f_z 0.04 mm, climb, ae 0.5 mm, ap 5 mm, lattice 0.125 mm (= ae / 4, "
             "the twin check rule), the VF-2SS machine's modes in the hot loop (dynamics L2 off)", h_from_cost=True),
    "lattice_side": Study(
        "lattice_side", "Dexel lattice: side-milling finish (rigid)", "physics.lattice_mm", "lattice spacing (mm)",
        [Level(f"lat{a:g}", a, {"lattice_mm": a}) for a in (0.5, 0.25, 0.125, 0.0625, 0.03125)], "lat0.25",
        ["removed_mm3", "wall_dev_um", "engaged_deg", "F_mean_N", "F_peak_N", "chip_volume_analytic_mm3"],
        lambda st, lv, db: mill_case(st, lv, db, kind="side", dynamics=False, ap_mm=5.0, ae_mm=0.3),
        note="D10 4FL, Al 6061-T6, 10 000 rpm, f_z 0.04 mm, climb, ae 0.3 mm, ap 5 mm, rigid; the template default "
             "0.25 mm is above the twin check limit ae / 4 = 0.075 mm for this pass",
        exact={"removed_mm3": "finest:removed_y_mm3"}),
    "lattice_corner": Study(
        "lattice_corner", "Dexel lattice: pocket corner (rigid)", "physics.lattice_mm", "lattice spacing (mm)",
        [Level(f"lat{a:g}", a, {"lattice_mm": a}) for a in (0.5, 0.25, 0.125, 0.0625, 0.03125)], "lat0.25",
        ["removed_mm3", "wall_bottom_dev_um", "wall_right_dev_um", "engaged_max_deg", "F_mean_N",
         "F_peak_corner_N"],
        lambda st, lv, db: corner_case(st, lv, db),
        note="meshed stock with a channel (walls at y = 5 and x = 30 mm), D10 4FL climb finishing 0.3 mm into both "
             "walls through the internal corner, ap 5 mm, 10 000 rpm, f_z 0.04 mm, rigid",
        exact={"removed_mm3": "finest:removed_path_mm3"}),
    "element_flat": Study(
        "element_flat", "Edge element size: D10 flat end mill, slot (rigid)", "physics.edge_element_mm",
        "edge element length (mm)",
        [Level(f"el{e:g}", e, {"edge_element_mm": e}) for e in (1.0, 0.5, 0.25, 0.125, 0.0625)], "el0.5",
        ["F_mean_N", "Fx_N", "Fy_N", "Fz_N", "F_peak_N", "torque_Nm", "chip_volume_analytic_mm3",
         "chip_volume_field_mm3"],
        lambda st, lv, db: mill_case(st, lv, db, kind="slot", dynamics=False, ap_mm=3.0, lattice_mm=0.0625),
        note="D10 4FL (MHSV), Al 6061-T6, 10 000 rpm, f_z 0.04 mm, full slot ap 3.0 mm, lattice 0.0625 mm, rigid"),
    "element_ball": Study(
        "element_ball", "Edge element size: D10 ball nose, slot (rigid)", "physics.edge_element_mm",
        "edge element length (mm)",
        [Level(f"el{e:g}", e, {"edge_element_mm": e}) for e in (1.0, 0.5, 0.25, 0.125, 0.0625)], "el0.5",
        ["F_mean_N", "Fx_N", "Fy_N", "Fz_N", "F_peak_N", "torque_Nm", "chip_volume_analytic_mm3",
         "chip_volume_field_mm3"],
        lambda st, lv, db: mill_case(st, lv, db, kind="slot", dynamics=False, ap_mm=2.0, lattice_mm=0.0625,
                                     tool=C.BALL_D10),
        note="D10 4FL ball nose (Tungaloy TEB100E4L), Al 6061-T6, 10 000 rpm, f_z 0.04 mm, slot ap 2.0 mm (the ball "
             "only), lattice 0.0625 mm, rigid"),
}


def _bins_runout(st, lv, db):
    machine = C.one_mode_machine(db, 900.0, 0.02, 5.0)
    return mill_case(st, lv, db, kind="chatter", dynamics=True, ap_mm=0.6, machine=machine, runout_um=5.0)


STUDIES["bins_runout"] = Study(
    "bins_runout", "Surface-memory bins alone, stride fixed (1-DOF)", "physics.memory_bins",
    "angle per memory bin (deg)",
    [Level(f"bins{b}", 360.0 / b, {"memory_bins": b}) for b in (90, 180, 360, 720, 1440, 2880)], "bins720",
    ["F_mean_N", "Fx_N", "Fy_N", "F_peak_N", "torque_Nm", "chip_volume_analytic_mm3", "chip_tooth_spread", "vib_pp_um"],
    _bins_runout,
    note="D10 4FL, Al 6061-T6, 10 000 rpm, f_z 0.04 mm, full slot ap 0.6 mm, runout 5 um TIR, the VF-2SS with one "
         "spindle mode in x (900 Hz, zeta 0.02, 5 N/um; stable: forced vibration imprinted on the surface)")


STUDIES["bins_stride"] = Study(
    "bins_stride", "Surface-memory bins with the stride (1-DOF, runout, regeneration)",
    "physics.memory_bins with physics.max_deg_per_step = 360 / memory_bins", "angle per memory bin (deg)",
    [Level(f"bins{b}", 360.0 / b, {"memory_bins": b, "max_deg_per_step": 360.0 / b, "steps_per_tooth": 24})
     for b in (180, 360, 720, 1440, 2880)], "bins720",
    ["F_mean_N", "Fx_N", "Fy_N", "F_peak_N", "torque_Nm", "chip_volume_analytic_mm3", "chip_tooth_spread", "vib_pp_um"],
    _bins_runout,
    note="the case of bins_runout with the L0 stride refined with the bins (at most one memory bin per L0 step: "
         "max_deg_per_step = 360 / memory_bins; steps_per_tooth 24 so that the angle rule binds)")


def _bins_onset(st, lv, db):
    return onset_case(st, lv, db, ap_mm=2.4)


STUDIES["bins_chatter"] = Study(
    "bins_chatter", "Surface-memory bins: chatter onset depth (1-DOF)", "physics.memory_bins",
    "angle per memory bin (deg)",
    [Level(f"bins{b}", 360.0 / b, {"memory_bins": b}) for b in (180, 360, 720, 1440)], "bins720",
    ["onset_depth_mm"], _bins_onset,
    note="full slot, D10 4FL, Al 6061-T6, 10 000 rpm, f_z 0.04 mm, ap 2.4 mm; one spindle mode in x (900 Hz, "
         "zeta 0.02); onset by bisection over the modal stiffness, reported as the onset depth at 5 N/um")

STUDIES["turn_theta"] = Study(
    "turn_theta", "Turning: polar theta bins", "physics.polar_theta_bins", "angle per theta bin (deg)",
    [Level(f"th{n}", 360.0 / n, {"polar_theta_bins": n, "removal": {"polar_min_feed_mm_rev": 0.2,
                                                                     "polar_max_mb": 400}})
     for n in (180, 360, 720, 1440, 2880)], "th720",
    ["F_mean_N", "Fx_N", "Fy_N", "Fz_N", "Ra_um", "Rt_um"], lambda st, lv, db: turn_case(st, lv, db),
    note="OD pass, 4140 d45, CNMG120408 / PCLNR, 1400 rpm, f 0.25 mm/rev, ap 1.5 mm, rigid; rows 25 um "
         "(polar_min_feed_mm_rev 0.2), polar_max_mb 400 (no coarsening)")

STUDIES["turn_dz"] = Study(
    "turn_dz", "Turning: polar row height (axial resolution)", "physics.removal.polar_min_feed_mm_rev (dz = f_min / 8)",
    "polar row height dz (um)",
    [Level(f"fmin{f:g}", f / 8.0 * 1e3, {"removal": {"polar_min_feed_mm_rev": f, "polar_max_mb": 400}})
     for f in (0.4, 0.2, 0.1, 0.05, 0.025)], "fmin0.05",
    ["F_mean_N", "Fx_N", "Fy_N", "Fz_N", "Ra_um", "Rt_um"], lambda st, lv, db: turn_case(st, lv, db),
    note="OD pass, 4140 d45, CNMG120408 / PCLNR, 1400 rpm, f 0.25 mm/rev, ap 1.5 mm, rigid, 720 theta bins; "
         "dz = min(f_min / 8, r_nose / 16) with f_min = polar_min_feed_mm_rev (the default 0.05 mm gives 6.25 um), "
         "polar_max_mb 400 (no coarsening)", test_labels=["fmin0.2", "fmin0.1", "fmin0.05"])


# ------------------------------------------------------------------------------------------------ whole loop
LOOP_SIZE = (150.0, 20.0, 15.0)
LOOP_AE, LOOP_AP = 0.5, 1.0
# one side step of 118 mm (about 4.4 s in cut, 9 s simulated). Two passes of 166 mm (11 s in cut, the 10-20 s cut the
# study asked for) ran too: the tool's flank wear and temperature run away together after about 4.5 s of cutting (VB
# 106 um at 8 s, 592 um at 9 s, 1.2 mm at 14 s, then no cut), so the comparison stops before that point
# (docs/convergence.md 5)
LOOP_PASSES = [(-5.0 + LOOP_AE, -8.0, 110.0)]
LOOP_EVAL_X_MM = 100.0             # quantities at the moment the tip passes x = 100 mm on the (last) pass
LOOP_FORCE_X_MM = (20.0, 100.0)    # forces: mean over the pass between these tip positions
# the coupling temperatures also as time means over this span before the evaluation moment (5 solves of the tool
# field, 25 edge evaluations, about 80 revolutions): their step-to-step scatter (the passes' heat, the solve instants)
# is a few per cent, which a value at one instant compares as much as the step's effect
LOOP_MEAN_S = 0.5
# structure's work model at 3000 DOF, as in the stored results (its 50 000 DOF default cost ~1 s per macro step on
# this stock until 2026-11-09, 0.09 s since: docs/performance.md 11)
LOOP_CASE = {"structure": {"max_dof": 3000}}


def loop_case(study, level: Level, db):
    """A side step 0.5 mm wide and 1 mm deep along the front face of a 150 x 20 x 15 mm Al 6061-T6 block in the vise
    of the pocket example, from x = -8 to 110 mm (about 4.4 s in cut), D10 4FL at 10 000 rpm, f_z 0.04 mm, flood
    coolant (M08), every domain on (run_project's loop); structure's work model at 3000 DOF, dynamics' L2 on coarse
    grids."""
    phys = level_physics(level, **LOOP_CASE)
    work = WORK / study / level.label
    if work.exists():
        shutil.rmtree(work)
    program = C.passes_program(LOOP_AP, LOOP_PASSES)
    project = C.mill_project(work / "project", program, C.block_edits(LOOP_SIZE, physics=phys))
    rows = []
    led = {"worst": 0.0}

    def observe(run):
        st = run.state
        b = st.bus_prev
        fr = b.motion.paths[0] if b.motion.paths else None
        if fr is None:
            return
        tips = np.asarray(fr.tip_S_m, float)
        r = {"t": int(st.clock.tick) * 1e-6, "x": float(tips[-1, 0]) * 1e3, "y": float(tips[-1, 1]) * 1e3,
             "xs": (tips[:, 0] * 1e3).tolist(), "ys": (tips[:, 1] * 1e3).tolist()}
        for slot, h in (getattr(b.thermal, "tools", {}) or {}).items():
            r["body_K"] = float(h.body_temp_K)
            ep = np.asarray(h.edge_peak_q_K, float)
            r["edge_K"] = float(np.nanmax(ep)) if ep.size and np.isfinite(ep).any() else math.nan
            rk = np.asarray(h.rake_temp_mean_K, float)
            r["rake_K"] = float(np.nanmax(rk)) if rk.size and np.isfinite(rk).any() else math.nan
        th = b.thermal
        if th is not None:
            r["bulk_K"] = float((th.work_bulk_mean_K or {}).get(0, math.nan))
            r["bulk_max_K"] = float((th.work_bulk_max_K or {}).get(0, math.nan))
            r["surface_K"] = float((th.contact_surface_temp_K or {}).get(0, math.nan))
            r["preheat_K"] = float((th.contact_preheat_K or {}).get(0, math.nan))
        for slot, tw in (getattr(b.wear, "tools", {}) or {}).items():
            r["VB_m"] = float(tw.VB_max_m)
        cc = b.cutting.contacts
        if cc:
            W = np.asarray(cc[0].wrench_tool_T0_mean, float)
            r["F"] = (W[:, :3]).tolist() if W.size else []
            r["in_cut"] = float(cc[0].in_cut_s)
        rc = b.removal.contacts
        disp = (getattr(b.structure, "displacement", {}) or {}).get(0)
        if rc and disp is not None and rc[0].geometry.in_cut:
            g = rc[0].geometry
            p = np.asarray(g.point_S_m, float)
            n = np.asarray(g.normal_M, float)
            d = _trilinear(disp, p)
            if d is not None:
                r["expansion_um"] = float(1e6 * np.dot(d, n / (np.linalg.norm(n) or 1.0)))
                r["disp_um"] = (1e6 * d).tolist()
        lw = getattr(run.ledger, "worst", None)
        if lw is not None:
            led["worst"] = float(lw.get("energy", 0.0))
            r["energy_rel"] = float((getattr(run.ledger, "final", {}) or {}).get("energy", 0.0))
        rows.append(r)

    t0 = time.process_time()
    result, run = M.run_loop(project, work / "run", observe=observe)
    cpu = time.process_time() - t0
    q, det = loop_quantities(rows)
    summary = result.get("summary", {})
    cons = json.loads((work / "run" / "run.json").read_text())["summary"].get("conservation", {}) \
        if (work / "run" / "run.json").exists() else {}
    q["energy_worst_rel"] = led["worst"]
    keys = ("t", "x", "y", "in_cut", "body_K", "edge_K", "rake_K", "VB_m", "bulk_K", "bulk_max_K", "surface_K",
            "preheat_K", "expansion_um", "energy_rel")
    series = {k: [r.get(k) for r in rows[::5]] for k in keys}
    cut = [r for r in rows if r.get("in_cut", 0.0) > 0.0 and r.get("energy_rel") is not None]
    det.update({"complete": bool(result.get("complete")), "verdict": result.get("verdict"),
                "sim_time_s": summary.get("sim_time_s"), "conservation": cons, "cpu_s": result.get("cpu_s"),
                "digest": _rows_digest(rows), "series": series,
                "energy_rel_above_1e-6": int(sum(1 for r in rows if (r.get("energy_rel") or 0.0) > 1e-6)),
                "findings": summary.get("findings")})
    cost = {"cpu_total_s": cpu, "sim_s": summary.get("sim_time_s"), "cpu_phases_s": result.get("cpu_phases_s"),
            "cpu_hot_s": (result.get("cpu_s") or {}).get("hot_s"), "cpu_l2_s": (result.get("cpu_s") or {}).get("l2_s"),
            "l2_by_domain_s": (result.get("cpu_s") or {}).get("l2_by_domain_s")}
    return q, cost, det


def _trilinear(disp, p):
    """thermal_m of structure's Displacement grid (node values at origin + i cell) at the point p (S)."""
    grid = np.asarray(disp.thermal_m, float)
    if grid.ndim != 4 or grid.shape[-1] != 3 or not disp.cell_m:
        return None
    u = (np.asarray(p, float) - np.asarray(disp.origin_S_m, float)) / float(disp.cell_m)
    i0 = np.floor(u).astype(int)
    f = u - i0
    n = np.array(grid.shape[:3])
    out = np.zeros(3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = np.clip(i0 + np.array([dx, dy, dz]), 0, n - 1)
                w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                out += w * grid[idx[0], idx[1], idx[2]]
    return out


def _rows_digest(rows) -> str:
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, default=_jsonable).encode())
    return h.hexdigest()[:16]


def loop_quantities(rows):
    """Quantities of the loop case at a common moment (the tip at LOOP_EVAL_X_MM on the second pass, interpolated
    linearly in time between macro steps) and over a common window (the second pass between LOOP_FORCE_X_MM)."""
    y2 = LOOP_PASSES[-1][0]
    t_eval = None
    prev = None
    for r in rows:
        on2 = abs(r["y"] - y2) < 0.05 and r.get("in_cut", 0.0) > 0.0
        if on2 and r["x"] >= LOOP_EVAL_X_MM and prev is not None and abs(prev["y"] - y2) < 0.05 and \
                prev.get("in_cut", 0.0) > 0.0:
            # the tick inside this step where the tip passes the evaluation x
            xs = np.asarray(r["xs"])
            k = int(np.argmax(xs >= LOOP_EVAL_X_MM))
            n = len(xs)
            t_eval = r["t"] - (n - 1 - k) * 1e-3
            break
        prev = r
    q, det = {}, {"t_eval_s": t_eval}
    if t_eval is None:
        return q, det
    ts = np.array([r["t"] for r in rows])

    def at(key):
        vs = np.array([r.get(key, math.nan) for r in rows], float)
        ok = np.isfinite(vs)
        return float(np.interp(t_eval, ts[ok], vs[ok])) if ok.sum() >= 2 else math.nan

    T0 = 293.15
    q["tool_rise_K"] = at("body_K") - T0
    q["edge_rise_K"] = at("edge_K") - T0
    q["rake_rise_K"] = at("rake_K") - T0
    q["VB_growth_um"] = 1e6 * (at("VB_m") - next(r["VB_m"] for r in rows if "VB_m" in r))
    q["bulk_rise_K"] = at("bulk_K") - T0
    q["bulk_max_rise_K"] = at("bulk_max_K") - T0
    q["surface_rise_K"] = at("surface_K") - T0
    q["preheat_rise_K"] = at("preheat_K") - T0

    def mean_before(key):
        vals = [r.get(key) for r in rows if t_eval - LOOP_MEAN_S < r["t"] <= t_eval]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.mean(vals)) if vals else math.nan

    q["rake_rise_mean_K"] = mean_before("rake_K") - T0
    q["edge_rise_mean_K"] = mean_before("edge_K") - T0
    q["preheat_rise_mean_K"] = mean_before("preheat_K") - T0
    q["expansion_um"] = at("expansion_um")
    F = []
    for r in rows:
        if abs(r["y"] - y2) < 0.05 and r.get("F") and r.get("in_cut", 0.0) > 0.0:
            for x, f in zip(r["xs"], r["F"]):
                if LOOP_FORCE_X_MM[0] <= x <= LOOP_FORCE_X_MM[1]:
                    F.append(f)
    if F:
        F = np.asarray(F)
        q["F_mean_N"] = float(np.linalg.norm(F.mean(axis=0)))
        q["Fy_N"] = float(F.mean(axis=0)[1])
    det["eval_samples"] = len(rows)
    return q, det


LOOP_NOTE = ("a side step 0.5 mm wide and 1 mm deep along the front face of a 150 x 20 x 15 mm Al 6061-T6 block, "
             "x = -8 to 110 mm (4.4 s in cut), D10 4FL, 10 000 rpm, f_z 0.04 mm, flood coolant, every domain on; "
             "structure at 3000 DOF; quantities when the tip passes x = 100 mm, forces over x = 20-100 mm")


def _macro_level(m):
    return Level(f"macro{m}", float(m), {"macro_ipo": m})


STUDIES["macro"] = Study(
    "macro", "Macro step (every domain on)", "physics.macro_ipo", "macro step (ms)",
    [_macro_level(m) for m in (5, 10, 20, 50, 2)], "macro10",
    ["tool_rise_K", "edge_rise_K", "rake_rise_K", "VB_growth_um", "F_mean_N", "bulk_rise_K", "preheat_rise_K",
     "expansion_um", "rake_rise_mean_K", "edge_rise_mean_K", "preheat_rise_mean_K"], loop_case, note=LOOP_NOTE,
    test_labels=["macro10", "macro5"])  # 2 ms: over 1.8 GB until convergence.md 9.10

STUDIES["lag"] = Study(
    "lag", "Coupling lag (every domain on)", "physics.lag_macro", "lag (macro steps)",
    [Level("lag1", 1.0, {"lag_macro": 1}), Level("lag2", 2.0, {"lag_macro": 2})], "lag1",
    ["tool_rise_K", "edge_rise_K", "VB_growth_um", "F_mean_N", "bulk_rise_K", "expansion_um"], loop_case,
    note="the macro-step case (" + LOOP_NOTE + ") at the default macro step", test_labels=["lag1", "lag2"])

STUDIES["thermal_cell"] = Study(
    "thermal_cell", "Workpiece thermal grid: cell size", "physics.thermal.cell_mm", "cell (mm)",
    [Level(f"cell{c:g}", c, {"thermal": {"cell_mm": c}}) for c in (8.0, 4.0, 2.0, 1.0, 0.5)], "cell2",
    ["bulk_rise_K", "bulk_max_rise_K", "surface_rise_K", "preheat_rise_K", "expansion_um", "tool_rise_K",
     "VB_growth_um", "preheat_rise_mean_K", "rake_rise_mean_K"], loop_case,
    note="the macro-step case (" + LOOP_NOTE + ") at the default macro step")

STUDIES["thermal_every"] = Study(
    "thermal_every", "Workpiece thermal grid: cadence", "physics.thermal.every_n_macro", "FV step (ms)",
    [Level(f"every{n}", 10.0 * n, {"thermal": {"every_n_macro": n}}) for n in (20, 10, 5, 2, 1)], "every5",
    ["bulk_rise_K", "bulk_max_rise_K", "surface_rise_K", "expansion_um", "tool_rise_K", "preheat_rise_K",
     "preheat_rise_mean_K"], loop_case,
    note="the macro-step case (" + LOOP_NOTE + ") at the default macro step (10 ms): the finite-volume step "
         "is every_n_macro x 10 ms")


# ------------------------------------------------------------------------------------------------ driver
def results_path(name: str) -> Path:
    return RESULTS / f"{name}.json"


def load(name: str) -> dict:
    p = results_path(name)
    return json.loads(p.read_text()) if p.exists() else {}


def save(name: str, doc: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = results_path(name)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=_jsonable) + "\n")
    tmp.replace(p)


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, set):
        return sorted(v)
    raise TypeError(type(v))


def analyse_study(study: Study, doc: dict) -> dict:
    levels = [lv for lv in doc.get("levels", []) if lv.get("quantities")]
    by_label = {lv["label"]: lv for lv in levels}
    default = by_label.get(study.default_label)
    out = {}
    for qname in study.quantities:
        tol, rel, noise = TOL.get(qname, (0.01, True, 1e-6))
        hs = [lv["h"] for lv in levels if qname in lv["quantities"]]
        vs = [lv["quantities"][qname] for lv in levels if qname in lv["quantities"]]
        if default is None or qname not in default["quantities"] or len(hs) < 2:
            continue
        exact = study.exact.get(qname)
        if isinstance(exact, str):
            src = default
            key = exact
            if exact.startswith("finest:"):
                key = exact.split(":", 1)[1]
                src = min((lv for lv in levels if key in lv.get("details", {})), key=lambda lv: lv["h"], default=None)
            exact = (src or {}).get("details", {}).get(key)
        out[qname] = analyse(hs, vs, default_h=default["h"], tol=tol, relative=rel, noise=noise, exact=exact)
    return out


def run_study(name: str, only=None, fresh: bool = False) -> dict:
    from cnctwin.db import Database
    study = STUDIES[name]
    doc = load(name)
    old = {lv["label"]: lv for lv in doc.get("levels", [])} if not fresh else {}
    db = Database.load()
    levels = []
    for i, level in enumerate(study.levels):
        if level.label in old and (only is None or i not in only):
            levels.append(old[level.label])
            continue
        if only is not None and i not in only:
            continue
        t0 = time.process_time()
        print(f"[{name}] level {level.label} ...", flush=True)
        try:
            q, cost, details = study.run(name, level, db)
            error = None
        except Exception as exc:          # a level the simulator stops is recorded with the reason
            import traceback
            q, cost, details = {}, {}, {}
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        h = level.h
        if study.h_from_cost and cost:
            h = _stride_h(cost) or h
        entry = {"label": level.label, "h": h, "physics": level.physics, "quantities": q, "cost": cost,
                 "details": details, "error": error, "cpu_level_s": time.process_time() - t0}
        print(f"[{name}] {level.label}: h {h:g} {json.dumps(q, default=_jsonable)[:600]} "
              f"cpu {entry['cpu_level_s']:.1f} s{' ERROR ' + error if error else ''}", flush=True)
        levels.append(entry)
        doc = {"study": name, "title": study.title, "setting": study.setting, "h_name": study.h_name,
               "default_label": study.default_label, "note": study.note, "root": str(ROOT),
               "levels": levels + [old[lv.label] for lv in study.levels[i + 1:] if lv.label in old]}
        save(name, doc)
        shutil.rmtree(WORK / name / level.label, ignore_errors=True)
    doc = {"study": name, "title": study.title, "setting": study.setting, "h_name": study.h_name,
           "default_label": study.default_label, "note": study.note, "root": str(ROOT), "levels": levels}
    doc["analysis"] = analyse_study(study, doc)
    save(name, doc)
    return doc


def evaluate(name: str, labels=None) -> dict:
    """Run levels of a study afresh without touching its results file (tests/convergence): ``labels`` (default: the
    shipped default and every finer level, the levels the study's reference rests on). Returns the study document
    with its analysis."""
    from cnctwin.db import Database
    study = STUDIES[name]
    default = next(lv for lv in study.levels if lv.label == study.default_label)
    if labels is None:
        labels = study.test_labels or [lv.label for lv in study.levels if lv.h <= default.h + 1e-12]
    db = Database.load()
    levels = []
    for level in study.levels:
        if level.label not in labels:
            continue
        q, cost, details = study.run(name, level, db)
        h = level.h
        if study.h_from_cost and cost:
            h = _stride_h(cost) or h
        levels.append({"label": level.label, "h": h, "physics": level.physics, "quantities": q, "cost": cost,
                       "details": details, "error": None})
        shutil.rmtree(WORK / name / level.label, ignore_errors=True)
    doc = {"study": name, "title": study.title, "default_label": study.default_label, "levels": levels}
    doc["analysis"] = analyse_study(study, doc)
    return doc


def main(argv) -> int:
    if "--list" in argv:
        for name, st in STUDIES.items():
            print(f"{name:16s} {st.title} [{st.setting}] levels: {', '.join(lv.label for lv in st.levels)}")
        return 0
    if "--analyse" in argv:
        # recompute the analysis of stored results (no simulation)
        for name in [a for a in argv if not a.startswith("-")]:
            doc = load(name)
            if doc:
                st = STUDIES[name]
                doc.update(title=st.title, setting=st.setting, h_name=st.h_name, note=st.note)
                doc["analysis"] = analyse_study(st, doc)
                save(name, doc)
                print(f"{name}: analysis of {len(doc['levels'])} levels")
        return 0
    fresh = "--fresh" in argv
    argv = [a for a in argv if a != "--fresh"]
    levels = env.pop_option(argv, "--levels")
    only = {int(v) for v in levels.split(",")} if levels else None
    names = [a for a in argv if not a.startswith("-")]
    for name in names:
        run_study(name, only, fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
