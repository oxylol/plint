"""Prescribed-motion cuts through the cutting domain's coefficient pipeline and force kernels (shared by
the cutting_force, torque, drilling and wall_error harnesses; docs/validation.md section 9).

* **Coefficients**: :class:`cnctwin.cutting.coefficients.MechanicsModel`, the model the cutting domain
  runs: per table row (tooling's coefficient groups, :func:`cnctwin.cutting.geometry.row_plan`) an Oxley /
  Johnson-Cook process map of cnctwin.mechanics, the oblique transformation, the edge, ploughing,
  indentation and flank-land models, and - where the material file has measured Kienzle or mechanistic
  data for the pairing (tool class, coating, process) - the calibration of those forces to them
  (``build_calibration``; docs/models/mechanics.md 7.3). Without such data the rows are the mechanics
  prediction alone. Every database value of that chain is recorded (``Collector.used``), including the
  calibration entries, so the runner can mark the case calibration-dependent.
* **Slow state**: the tool body and the work at the ambient temperature (a test cut of seconds; the
  rake-face flash of the contact is part of the mechanics partition), friction factor 1 (dry) unless the
  harness gives the coolant's, the wear land VB of the case.
* **Kernels**: :class:`cnctwin.cutting.drive.Driver` - ``cut_step`` for rotating tools on a steady
  straight path with analytic engaged intervals (straight-wall, slot and face cuts: the circle of every
  station inside the case's strip of material and below its depth of cut, on the feed side), the surface
  memory prefilled with the steady previous passes; ``turn_step`` on a polar field for lathe tools.
  Step: at most 0.5 deg of spindle rotation and at least 96 steps per tooth period (architecture 5.1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...cutting import abi as CA
from ...cutting import analytic as AN
from ...cutting import coefficients as CO
from ...cutting import geometry as GE
from ...cutting import schema as CS
from ...cutting.drive import Driver
from ...db.models import params
from ...mechanics.coefficients import select_entries
from ...mechanics.contact import min_shear_speed
from . import common as H

TWO_PI = 2.0 * math.pi
MAX_STEP_RAD = math.radians(0.5)        # physics.max_deg_per_step default (architecture 5.1)
STEPS_PER_TOOTH = 96                    # physics.steps_per_tooth default (architecture 5.1)
EDGE_ELEMENT_M = 0.25e-3                # edge element length used for the validation tools (default 0.5 mm)


def _text(value) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    return "" if value is None else str(value)


# ================================================================================================ pipeline
class Pipeline:
    """The coefficient pipeline of one case: work material, tool side, MechanicsModel for ``process``."""

    def __init__(self, db, case, col: H.Collector, process: str, *, tool_kind: str, tool_mat: str | None = None,
                 coat: str | None = None, insert: dict | None = None, ambient_K: float | None = None):
        self.case = case
        self.col = col
        self.work, self.raw = H.work_material(db, case, col)
        self.tool, self.tool_mat, self.coating = H.tool_side(db, case, col, tool_mat=tool_mat, coat=coat,
                                                              insert=insert, kind=tool_kind)
        self.ambient_K = H.ambient(case, col) if ambient_K is None else float(ambient_K)
        self.process = process
        self.settings = CS.settings(None)
        self.params = dict(params("cutting-contact"))
        self.model = CO.MechanicsModel(self.work, self.tool, process, contact_params=self.params)
        self.friction = self.work.friction_for(self.tool.tool_class, self.tool.coating)
        H.material_provenance(self.raw, col, friction_entry=self.friction,
                              extra=("cutting.indentation_coeff_N_mm3", "cutting.bue"))
        self.v_min = min_shear_speed()
        self.calibration_kind = "none"
        self.calibration_sources: tuple = ()
        self._recorded = False
        col.assume("tool and work temperature", f"{self.ambient_K - 273.15:.0f} degC",
                   "a test cut of seconds: tool body and work at ambient (the contact flash is in the mechanics "
                   "heat partition)", "est")

    # ---------------------------------------------------------------------------------------- rows
    def state(self, speed_m_s: float, *, VB_m: float = 0.0, friction_factor: float = 1.0) -> CO.RowState:
        T = self.ambient_K
        return CO.RowState(speed_m_s=max(float(speed_m_s), 1e-6), tool_temp_K=T, flank_temp_K=T, work_temp_K=T,
                           friction_factor=float(friction_factor), VB_m=float(VB_m))

    def h_grid(self) -> np.ndarray:
        st = self.settings
        return CO.log_grid(st.h_log_lo_m, max(st.h_hi_m, 2e-3), st.log_nodes)

    def rows(self, key: tuple, geom: CO.RowGeometry, state: CO.RowState) -> CO.Rows:
        rows = self.model.rows(key, geom, state, self.h_grid())
        if not (geom.forming or geom.low_speed):
            self._record_calibration(key, geom)
        return rows

    def _record_calibration(self, key, geom) -> None:
        cal = self.model.calibration(key, self.model._group(geom))
        if cal is None or cal.kind == "none":
            if not self._recorded:
                self.col.note(f"Coefficients: the mechanics prediction alone (no measured Kienzle or mechanistic data "
                              f"in the material file for {self.tool.tool_class}/{self.tool.coating}, {self.process}).")
            self._recorded = True
            return
        self.calibration_kind = cal.kind
        self.calibration_sources = tuple(sorted(set(cal.sources)))
        if self._recorded:
            return
        self._recorded = True
        kind, entries = select_entries(self.work, self.tool, self.process)
        raw_list = list(((self.raw.get("cutting") or {}).get(kind) or []))
        objects = list(self.work.kienzle if kind == "kienzle" else self.work.mechanistic)
        mid = self.raw.get("id")
        for entry in entries:
            raw = None
            idx = next((i for i, e in enumerate(objects) if e is entry), None)
            if idx is not None and len(raw_list) == len(objects):
                raw = raw_list[idx]
            if raw is None:
                raw = next((r for r in raw_list if str(r.get("src") or "") == entry.src), {})
            notes = [str(raw.get("note") or "")]
            for k, v in sorted(raw.items()):
                if isinstance(v, Mapping) and v.get("note"):
                    notes.append(f"{k}: {v.get('note')}")
            self.col.used(f"{mid} cutting.{kind} ({entry.tool_class}, {entry.coating}, {entry.process}) "
                          f"calibrating the forces", entry.src, " | ".join(n for n in notes if n))
        self.col.note(f"Coefficients calibrated to the material's {kind} data ({', '.join(self.calibration_sources)}) "
                      f"for {self.tool.tool_class}/{self.tool.coating}, {self.process} (cnctwin.mechanics "
                      "build_calibration; Oxley carries the dependence away from the reference conditions).")

    def settings_dict(self) -> dict:
        return {"tool_material": self.tool_mat, "coating": self.coating, "process": self.process,
                "mu_sliding": self.friction.mu, "h_min_ratio": self.work.min_chip_thickness_ratio,
                "calibration": self.calibration_kind,
                "calibration_sources": list(self.calibration_sources)}


# ================================================================================================ layout
@dataclass
class Layout:
    geo: object
    rows_geom: list
    elem_row: np.ndarray
    rows: list
    table: CA.CutTable
    tool: CA.CutTool
    nt: int
    ne: int


def layout(pipe: Pipeline, geo, omega_rad_s: float, *, speed_m_s: float | None = None, VB_m: float = 0.0,
           friction_factor: float = 1.0, h_hi_m: float | None = None, key: str = "t") -> Layout:
    """Rows, kernel table and CutTool of a tool: each row at the cutting speed of its elements
    (|omega| x mean element radius for rotating tools, ``speed_m_s`` for lathe tools)."""
    rows_geom, elem_row = GE.row_plan(geo, geo.edge, omega_rad_s, pipe.v_min)
    rows = []
    for r, g in enumerate(rows_geom):
        speed = speed_m_s if speed_m_s is not None else abs(omega_rad_s) * max(g.radius_m, 1e-4)
        rows.append(pipe.rows((key, r), g, pipe.state(speed, VB_m=VB_m, friction_factor=friction_factor)))
    st = pipe.settings
    table = CO.build_table(rows, max(st.h_hi_m, h_hi_m or 0.0), st.n_lo, st.n_hi, pipe.v_min)
    shape = np.asarray(geo.edge.elem_r_m).shape
    VB = np.full(shape, float(VB_m)) if VB_m > 0.0 else None
    options = GE.ToolOptions(tangent_vn=st.tangent_vn, turn_colwell=bool(st.turn_colwell),
                             surface_event_rad=st.surface_event_rad, h_cap_m=st.h_cap_m,
                             recovery_ratio=float(pipe.params["recovery_transition_ratio"]),
                             pd_max_flank_m=float(pipe.params["pd_max_flank_m"]),
                             pd_edge_land_per_radius=float(pipe.params["pd_edge_land_per_radius"]))
    tool = GE.build_cut_tool(geo, geo.edge, geo.frames, rows, elem_row, VB, options)
    return Layout(geo, rows_geom, elem_row, rows, table, tool, int(shape[0]), int(shape[1]))


def rows_force_per_width(rows: CO.Rows, h_m: float) -> tuple[float, float, float]:
    """(F_t, F_r, F_a) per unit width of a row at chip thickness h (the row's own h grid, linear in log h):
    the closed form the kernels integrate, for the report."""
    lg = np.log(rows.h_grid_m)
    lh = math.log(max(h_m, float(rows.h_grid_m[0])))
    k = [float(np.interp(lh, lg, getattr(rows, name))) for name in ("Kt_Pa", "Kr_Pa", "Ka_Pa")]
    return (k[0] * h_m + rows.Kte_N_per_m, k[1] * h_m + rows.Kre_N_per_m, k[2] * h_m + rows.Kae_N_per_m)


# ================================================================================================ tools
def build_geometry(item: dict, *, stickout_m: float | None = None, edge_element_m: float = EDGE_ELEMENT_M):
    """tooling.build_tool of an inline tool item (db/_templates/tool.yaml keys), no holder, no runout."""
    from ...tooling import build_tool
    from ...tooling.schema import ToolingSettings
    inst = {"number": 1, "runout_um": 0}
    if stickout_m is not None:
        inst["stickout_mm"] = stickout_m * 1e3
    return build_tool(item, None, inst, ToolingSettings(edge_element_m=edge_element_m))


def tool_item(kind: str, tool_class: str, tool_mat: str, coating: str, geometry: dict, tool_id: str) -> dict:
    return {"id": tool_id, "type": kind, "hand": "right", "tool_material": {"class": tool_class, "id": tool_mat},
            "coating": {"id": coating or "none"}, "geometry": geometry}


# ================================================================================================ milling
def strip_arc(r: float, y_lo: float, y_hi: float) -> tuple[float, float] | None:
    """Immersion arc [phi_st, phi_ex] (Altintas: clockwise from +Y, the feed along +X) of a circle of radius r
    within the strip y_lo <= y <= y_hi on the feed side (phi in [0, pi])."""
    if r <= 0.0:
        return None
    c_hi = min(1.0, y_hi / r)
    c_lo = max(-1.0, y_lo / r)
    if c_hi <= c_lo:
        return None
    return math.acos(c_hi), math.acos(c_lo)


def strip(direction: str, ae_m: float, R_m: float) -> tuple[float, float]:
    """The material strip of a straight cut in Altintas' frame: down milling y in [-R, -R + ae] (the tooth
    exits tangentially at phi = pi), up milling [R - ae, R], a slot [-R, R] (sld.immersion_angles)."""
    if direction == "slot" or ae_m >= 2.0 * R_m * (1.0 - 1e-12):
        return -R_m, R_m
    if direction == "up":
        return R_m - ae_m, R_m
    return -R_m, -R_m + ae_m


@dataclass
class MillRun:
    t_s: np.ndarray
    force_T0_N: np.ndarray            # [n, 3] force on the tool: X feed, Y = feed turned +90 deg, Z tool axis
    torque_Nm: np.ndarray             # [n] process torque about the tool axis (on the rotating body)
    omega_rad_s: float
    n_teeth: int
    dt_s: float
    engaged_stations: int
    station_force_T0_N: np.ndarray | None = None   # [n, nst, 3] force on the tool per station (T0)

    def per_tooth_peaks(self, values: np.ndarray) -> np.ndarray:
        steps = max(int(round(TWO_PI / (self.n_teeth * abs(self.omega_rad_s)) / self.dt_s)), 1)
        n = (len(values) // steps) * steps
        return np.asarray(values[:n]).reshape(-1, steps).max(axis=1) if n else np.asarray([np.max(values)])


def time_step(omega_rad_s: float, n_teeth: int) -> float:
    step = min(MAX_STEP_RAD, TWO_PI / (max(n_teeth, 1) * STEPS_PER_TOOTH))
    return step / abs(omega_rad_s)


def mill(lay: Layout, omega_rad_s: float, fz_m: float, ap_m: float, y_lo: float, y_hi: float, *,
         rho_kg_m3: float, z_floor_m: float = 0.0, revs_warm: int = 2, revs: int = 1,
         keep_stations: bool = False) -> MillRun:
    """Steady straight milling (tool along +x) through cut_step: stations at 0 <= z - z_floor <= ap whose
    circle meets the strip [y_lo, y_hi] on the feed side are engaged over that arc."""
    geo = lay.geo
    drv = Driver(lay.tool, lay.table, lay.nt, lay.ne, rho_kg_m3=rho_kg_m3)
    st_z = np.asarray(lay.tool.st_z_m, float)
    st_r = np.asarray(geo.edge.station_r_m, float)
    engaged = 0
    for s in range(st_z.shape[0]):
        z = st_z[s] - z_floor_m
        if z < -1e-12 or z > ap_m * (1.0 + 1e-9):
            continue
        arc = strip_arc(abs(float(st_r[s])), y_lo, y_hi)
        if arc is None:
            continue
        lo = 0.5 * math.pi - arc[1]
        hi = 0.5 * math.pi - arc[0]
        drv.iv.lo_rad[s, 0] = lo % TWO_PI
        drv.iv.hi_rad[s, 0] = lo % TWO_PI + (hi - lo)
        drv.iv.count[s] = 1
        engaged += 1
    nt = lay.nt
    vf = fz_m * nt * abs(omega_rad_s) / TWO_PI

    def tip(t):
        return np.array([vf * t, 0.0, 0.0])
    drv.prefill_memory(tip, 0.0, omega_rad_s)
    dt = time_step(omega_rad_s, nt)
    n_rev = int(round(TWO_PI / abs(omega_rad_s) / dt))
    drv.run_rotating(revs_warm * n_rev, omega_rad_s, dt, tip, record=False)
    rec = drv.run_rotating(revs * n_rev, omega_rad_s, dt, tip, t0_s=revs_warm * n_rev * dt)
    W = np.array(rec.wrench)
    stations = np.array(rec.station)[:, :, :3].copy() if keep_stations else None
    return MillRun(np.array(rec.t_s), W[:, :3].copy(), np.array(rec.torque), float(omega_rad_s), nt, dt, engaged,
                   stations)


# ================================================================================================ drilling
@dataclass
class DrillRun:
    thrust_N: float                   # mean axial force on the tool, positive pushing the tool back (+z_T)
    torque_Nm: float                  # mean process torque magnitude
    thrust_series: np.ndarray
    torque_series: np.ndarray
    engaged_stations: int


def drill(lay: Layout, omega_rad_s: float, f_rev_m: float, *, rho_kg_m3: float, r_min_m: float = 0.0,
          revs_warm: int = 2, revs: int = 1) -> DrillRun:
    """Steady full-diameter drilling (tool along -z) through cut_step: every station of radius > r_min (a
    pilot hole) is engaged over the full circle; the memory holds the steady previous passes."""
    drv = Driver(lay.tool, lay.table, lay.nt, lay.ne, rho_kg_m3=rho_kg_m3)
    st_r = np.abs(np.asarray(lay.geo.edge.station_r_m, float))
    engaged = []
    for s in range(st_r.shape[0]):
        if st_r[s] > r_min_m:
            engaged.append(s)
    drv.set_full_circle(engaged)
    vz = f_rev_m * abs(omega_rad_s) / TWO_PI

    def tip(t):
        return np.array([0.0, 0.0, -vz * t])
    drv.prefill_memory(tip, 0.0, omega_rad_s)
    dt = time_step(omega_rad_s, lay.nt)
    n_rev = int(round(TWO_PI / abs(omega_rad_s) / dt))
    drv.run_rotating(revs_warm * n_rev, omega_rad_s, dt, tip, record=False)
    rec = drv.run_rotating(revs * n_rev, omega_rad_s, dt, tip, t0_s=revs_warm * n_rev * dt)
    W = np.array(rec.wrench)
    torque = np.array(rec.torque)
    return DrillRun(float(np.mean(W[:, 2])), float(abs(np.mean(torque))), W[:, 2].copy(), torque, len(engaged))


# ================================================================================================ turning
@dataclass
class TurnRun:
    Fc_N: float                       # against the cutting velocity
    Ff_N: float                       # along the work axis, positive pushing the tool back against its feed
    Fp_N: float                       # radial (passive), positive pushing the tool away from the axis
    direct_per_width: tuple           # (F_t, F_r, F_a) N/m from the rows at h (closed form, cross-check)


def tube_turning(pipe: Pipeline, *, speed_m_s: float, f_m: float, width_m: float, rake_rad: float,
                 clearance_rad: float, edge_radius_m: float, R_out_m: float, VB_m: float = 0.0,
                 friction_factor: float = 1.0, n_el: int = 40, revs: float = 3.0) -> TurnRun:
    """Orthogonal tube-end (or disc) turning through turn_step: a sharp straight edge at kappa_r = 90 deg,
    inclination 0, across the wall (edge 1.5 x the width, elements outside the wall see no material), feed
    f per revolution along the axis, the polar field of the tube updated after the tool has left a bin."""
    itool = AN.straight_turning_tool(0.5 * math.pi, 1.5 * width_m, n_el, rake_rad=rake_rad, clear_rad=clearance_rad)
    itool.edge = itool.edge._replace(elem_edge_radius_m=np.full_like(itool.edge.elem_edge_radius_m, edge_radius_m))
    R_in = R_out_m - width_m
    omega_c = -speed_m_s / (R_out_m - 0.5 * width_m)
    lay = layout(pipe, itool, omega_c, speed_m_s=speed_m_s, VB_m=VB_m, friction_factor=friction_factor,
                 h_hi_m=2.0 * f_m, key=f"turn{rake_rad:.4f}")
    nz = 96
    zr = 6.0 * f_m
    polar = CA.empty_polar(720, nz)
    polar.r_out_m[:] = R_out_m
    polar.geom[:] = (TWO_PI / 720, -zr + f_m, zr / nz, 0.0)
    drv = Driver(lay.tool, lay.table, 1, n_el, rotating=False, polar=polar,
                 rho_kg_m3=float(pipe.work.rho_kg_m3) if getattr(pipe.work, "rho_kg_m3", None) else 7850.0)
    drv.minor_side = 1
    vf = f_m * abs(omega_c) / TWO_PI
    z0 = f_m

    def pose(t):
        phi = -omega_c * t
        c, s = math.cos(phi), math.sin(phi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        tip = Rz @ np.array([R_in, 0.0, z0 - vf * t])
        w = np.array([0.0, 0.0, -omega_c])
        v = np.cross(w, tip) + Rz @ np.array([0.0, 0.0, -vf])
        return Rz @ np.diag([-1.0, -1.0, 1.0]), tip, v, w
    dt = MAX_STEP_RAD / abs(omega_c)
    n_rev = int(round(TWO_PI / abs(omega_c) / dt))
    drv.run_turning(int(revs * n_rev), omega_c, dt, pose)
    t = np.array(drv.record.t_s)[-n_rev:]
    fc, ff, fp = [], [], []
    for k in range(len(t)):
        _, tip, v, _ = pose(t[k])
        F = np.array(drv.record.force_S[-n_rev + k])
        fc.append(-F @ v / np.linalg.norm(v))
        ff.append(F[2])
        fp.append(F @ (tip / np.linalg.norm(tip)))
    direct = rows_force_per_width(lay.rows[0], f_m)
    return TurnRun(float(np.mean(fc)), float(np.mean(ff)), float(np.mean(fp)), direct)


__all__ = ["Pipeline", "Layout", "layout", "build_geometry", "tool_item", "strip", "strip_arc", "mill", "MillRun",
           "drill", "DrillRun", "tube_turning", "TurnRun", "rows_force_per_width", "time_step", "EDGE_ELEMENT_M"]
