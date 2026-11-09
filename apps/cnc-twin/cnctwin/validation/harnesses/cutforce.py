"""Harnesses for cutting_force and torque: the cutting domain's force model at the experiment's engagement.

Every case runs the coefficient pipeline the cutting domain uses (cnctwin.cutting.coefficients.MechanicsModel:
Oxley / Johnson-Cook maps of cnctwin.mechanics, calibrated to the material's measured Kienzle or mechanistic
data where the database has them for the pairing, else the mechanics prediction alone) and the domain's
force kernels through cnctwin.cutting.drive.Driver (``harnesses.cutsim``):

* **orthogonal and tube turning** (``orthogonal_turning``): ``turn_step`` with a sharp straight edge at
  kappa_r = 90 deg across the stated width, feed per revolution = uncut chip thickness, the polar field of
  the tube; Fc against the cutting velocity, the feed (thrust, back) force along the axis. The closed form
  of the rows at h (F = (K h + Ke) b) is listed next to it as a check of the kernel;
* **milling** (peripheral, slot, face, ball-end): ``cut_step`` with the tool built by cnctwin.tooling from
  the stated geometry, on a steady straight path, engaged intervals of the straight cut (the strip of the
  stated radial engagement, below the axial depth); each reference statistic (mean over a revolution, peak,
  most negative value, RMS, mean of per-tooth peaks) is computed from the simulated force on the tool in
  Altintas' frame (X feed, Y feed turned +90 deg, Z tool axis), per the case's statement. Where a source's
  frame orientation or force sign is not fully defined, magnitudes are compared (said per value).

The milling cases each need their own reading of the source (frames, statistics, which values are
fresh-tool references): ``MILLING`` maps a case id to its reader, with the case text it follows. Torque
cases use the same readers and compare the process torque about the tool axis.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from ...core.abi import EDGE_CORNER, EDGE_END
from ..cases import from_si, name_tokens, to_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction, not_compared
from . import assumptions as AS
from . import common as H
from . import cutsim as CSIM

MAIN = "main_force_pct"
THRUST = "thrust_force_pct"
MILL = "milling_peak_force_pct"
X_KEYS = ("h_mm", "feed_mm_rev", "uncut_chip_thickness_um", "f_mm", "vc_m_min", "cutting_speed_m_min")
ID_WORDS = ("test", "case", "trial", "run")
FLUIDS = (("semi-synthetic", "semi-synthetic-6pct"), ("emulsion", "emulsion-8pct"), ("oemeta", "emulsion-8pct"),
          ("mql", "mql-ester"), ("canola", "mql-ester"), ("oil", "straight-oil"))
MODEL_CHAIN_TURN = ["cnctwin.mechanics.materials (databases)", "cnctwin.cutting.coefficients.MechanicsModel "
                    "(Oxley / Johnson-Cook maps, edge and ploughing forces, calibration to measured Kienzle / "
                    "mechanistic data of the pairing)", "cnctwin.cutting.coefficients.build_table",
                    "cnctwin.cutting.kernels.turn_step (cnctwin.cutting.drive.Driver, polar field)"]
MODEL_CHAIN_MILL = ["cnctwin.mechanics.materials (databases)", "cnctwin.tooling.build_tool (edge model)",
                    "cnctwin.cutting.coefficients.MechanicsModel (rows per coefficient group)",
                    "cnctwin.cutting.kernels.cut_step (cnctwin.cutting.drive.Driver, analytic engaged intervals, "
                    "prefilled surface memory)"]


def _text(value) -> str:
    return H._text(value)


# ================================================================================================ coolant
def coolant_factor(db, case, col: H.Collector, mode: str, work, tool, speed_m_s: float, ambient_K: float) -> float:
    """Lubrication friction factor of the case's coolant (cnctwin.coolant.lubrication.friction, full reach of
    the jet), 1 for dry cutting."""
    if mode in ("", "dry"):
        return 1.0
    text = " ".join(_text(case.section("conditions").get(k)) for k in ("coolant", "flood_fluid", "fluid")).lower()
    text += " " + mode.lower()
    fluid_id = next((fid for key, fid in FLUIDS if key in text), "emulsion-8pct")
    return _fluid_factor(db, fluid_id, tool.tool_class, work.family, speed_m_s, ambient_K, col)


def _fluid_factor(db, fluid_id: str, tool_class: str, family: str, v_m_s: float, ambient_K: float, col) -> float:
    """cnctwin.coolant.lubrication.friction of a database fluid at full reach of the jet (the helper the coolant
    harness used to provide; it now traces each jet through its scene, which the force cases do not describe)."""
    from ...coolant import fluid as F
    from ...coolant import lubrication as L
    fluid = F.fluid_from_db(db.coolant(fluid_id))
    lub = L.friction(fluid, fluid.props(ambient_K), 1.0, tool_class, family, v_m_s, None, None, 1e-3, fluid.is_mql,
                     fluid.dosage_m3_s or 0.0)
    col.assume(f"coolant fluid {fluid_id}", "database entry", "the case's fluid is not in the database: the closest "
               "database fluid, full reach of the jet", "db/coolants")
    return float(lub.factor)


def coolant_mode(case, name: str = "") -> str:
    low = name.lower()
    for word in ("cryomql", "mql", "flood", "hpc", "dry"):
        if re.search(rf"(^|[\s_]){word}([\s_]|$)", low):
            return word
    text = _text(case.section("conditions").get("coolant")).lower()
    if not text or text.startswith("dry"):
        return "dry"
    if "mql" in text:
        return "mql"
    return "flood"


# ================================================================================================ orthogonal
def _row_of(rows, entry, x_keys=X_KEYS):
    """The test of a reference: a test id after 'test', 'case', 'trial' or 'run' (or a non-numeric id token),
    else cnctwin.validation.cases.find_test (x against the per-test values)."""
    ids = {r.test_id: r for r in rows}
    tokens = name_tokens(entry.get("name", ""))
    low = [t.lower() for t in tokens]
    if len(rows) > 1:
        for k, token in enumerate(tokens):
            if token in ids and (not token.isdigit() or (k > 0 and low[k - 1] in ID_WORDS)):
                return ids[token]
    return H.find_test(rows, entry, x_keys)


def _token_value(name: str, letters: tuple[str, ...]) -> float | None:
    """A number written after one of ``letters`` in a reference or curve name ('f=0.05', 'f0.1', 'vc361')."""
    for token in re.split(r"[\s(),;:_]+", str(name)):
        for letter in letters:
            m = re.fullmatch(rf"{letter}=?([0-9]*\.?[0-9]+)", token)
            if m:
                return float(m.group(1))
    return None


@dataclass
class OrthPoint:
    h_m: float
    speed_m_s: float
    rake_rad: float
    edge_radius_m: float
    width_m: float
    coolant: str
    label: str


def _scalar(row, *keys):
    value = row.si(*keys) if row is not None else None
    return float(value) if isinstance(value, (int, float)) else None


def _orth_point(case, rows, col, entry, *, x=None, x_name="", x_unit="", curve_name="") -> OrthPoint:
    name = str(entry.get("name") or curve_name)
    row = _row_of(rows, entry) if entry.get("name") else (rows[0] if len(rows) == 1 else None)
    xn = (x_name or "").lower()
    speed = _scalar(row, "vc_m_min", "cutting_speed_m_min")
    h = _scalar(row, "h_mm", "feed_mm_rev", "uncut_chip_thickness_um", "f_mm", "h_um")
    if x is not None and xn in ("vc", "v", "cutting_speed", "speed"):
        speed = to_si(float(x), x_unit)
    if x is not None and xn in ("f", "h", "feed", "uncut_chip_thickness", "t0"):
        h = to_si(float(x), x_unit)
    if speed is None:
        v = _token_value(name, ("vc",))
        speed = to_si(v, "m/min") if v is not None else None
    if h is None:
        v = _token_value(name, ("f", "h"))
        if v is not None:
            h = to_si(v, "um" if v > 5.0 else "mm")
    if speed is None or h is None:
        raise ValueError(f"{case.id}: no speed or uncut thickness for {name!r}")
    rake = H.rake_rad(case, row, col)
    r_e = H.edge_radius_m(case, row, col, "insert")
    width = H.width_m(case, row)
    if width is None:
        width = _scalar(row, "depth_of_cut_mm")
    if width is None:
        units = {str(e.get("unit") or "") for e in case.reference_values} | \
            {str(c.get("y_unit") or "") for c in case.reference_curves}
        if not all("/mm" in u for u in units):
            raise ValueError(f"{case.id}: no width of cut")
        width = 1e-3
        col.assume("width of cut", "1 mm", "not stated: the references are forces per unit width (plane strain)", "est")
    return OrthPoint(h, speed, rake, r_e, width, coolant_mode(case, name), row.test_id if row is not None else "")


def _tube_radius(case, col) -> float:
    stock = case.section("stock")
    form = _text(stock.get("form")).lower()
    od = case.si("stock", "outer_diameter_mm")
    if isinstance(od, (int, float)) and "disc" not in form:
        return 0.5 * float(od)
    if "disc" in form or "planer" in _text(case.section("machine").get("type")).lower():
        col.assume("cut geometry", "straight edge on a 1 m radius", "disc periphery or planing: the cutting speed is "
                   "uniform across the width", "est")
        return 1.0
    col.assume("tube outer diameter", f"{AS.TUBE_DIAMETER.value * 1e3:g} mm", AS.TUBE_DIAMETER.reason,
               "harnesses/assumptions.py")
    return 0.5 * AS.TUBE_DIAMETER.value


def _component(name: str) -> tuple[str, str]:
    """(budget component, force kind) of an orthogonal reference name: Fc main; Ff, Ft, Fa, Fp thrust/feed."""
    head = str(name).split()[0].lower()
    if head.startswith("fc"):
        return MAIN, "Fc"
    return THRUST, "Ff"


def orthogonal(case, db) -> HarnessOutput:
    col = H.Collector()
    pipe = CSIM.Pipeline(db, case, col, "turning", tool_kind="turning_insert")
    rows = H.test_rows(case)
    clearance = H.clearance_rad(case, col)
    R_out = _tube_radius(case, col)
    cache: dict = {}

    def simulate(pt: OrthPoint) -> CSIM.TurnRun:
        key = (round(pt.h_m, 12), round(pt.speed_m_s, 9), round(pt.rake_rad, 9), round(pt.edge_radius_m, 12),
               round(pt.width_m, 12), pt.coolant)
        if key not in cache:
            factor = coolant_factor(db, case, col, pt.coolant, pipe.work, pipe.tool, pt.speed_m_s, pipe.ambient_K)
            cache[key] = CSIM.tube_turning(pipe, speed_m_s=pt.speed_m_s, f_m=pt.h_m, width_m=pt.width_m,
                                           rake_rad=pt.rake_rad, clearance_rad=clearance,
                                           edge_radius_m=pt.edge_radius_m, R_out_m=R_out, friction_factor=factor)
        return cache[key]

    def predict(pt: OrthPoint, name: str, unit: str) -> Prediction:
        run = simulate(pt)
        comp, kind = _component(name)
        force = run.Fc_N if kind == "Fc" else run.Ff_N
        direct = run.direct_per_width[0 if kind == "Fc" else 1] * pt.width_m
        per_width = "/mm" in unit or "/m" in unit
        value_si = force / pt.width_m if per_width else force
        return Prediction(from_si(value_si, unit), COMPARED,
                          f"h {pt.h_m * 1e3:.3f} mm, vc {pt.speed_m_s * 60:.0f} m/min, r_e {pt.edge_radius_m * 1e6:.0f} um"
                          f", b {pt.width_m * 1e3:g} mm" + (f", {pt.coolant}" if pt.coolant != "dry" else ""),
                          component=comp,
                          extra={"closed_form": from_si(direct / pt.width_m if per_width else direct, unit),
                                 "Fp_N": run.Fp_N})

    values = []
    for entry in case.reference_values:
        pt = _orth_point(case, rows, col, entry, x=entry.get("x"), x_name=str(entry.get("x_name") or ""),
                         x_unit=str(entry.get("x_unit") or ""))
        values.append(predict(pt, str(entry.get("name")), str(entry.get("unit") or "")))
    curves = []
    for curve in case.reference_curves:
        preds = []
        for x, _y in curve.get("points") or []:
            pt = _orth_point(case, rows, col, {"name": curve.get("name")}, x=x, x_name=str(curve.get("x_name") or ""),
                             x_unit=str(curve.get("x_unit") or ""), curve_name=str(curve.get("name")))
            preds.append(predict(pt, str(curve.get("name")), str(curve.get("y_unit") or "")))
        curves.append(preds)
    col.note("The kernel value (turn_step) is compared; 'closed_form' in the extras is the rows' (K h + Ke) b at the "
             "same h (a check of the table and the element integration, normally within 1-2 %).")
    settings = pipe.settings_dict()
    settings.update({"tube_outer_radius_m": R_out, "edge_elements": 40, "step_deg": 0.5})
    return HarnessOutput(values, curves, MODEL_CHAIN_TURN, settings, col.assumptions, col.provenance, col.notes,
                         None)


# ================================================================================================ milling
@dataclass
class MillTool:
    kind: str
    geometry: dict
    tool_class: str
    stickout_m: float | None = None
    insert_stations_only: bool = False      # round-insert cutters: only the insert (corner) edge cuts


@dataclass
class MillCond:
    rpm: float
    fz_m: float
    ap_m: float
    ae_m: float
    direction: str
    VB_m: float = 0.0
    coolant: str = "dry"
    y_shift_m: float = 0.0                  # strip offset (centred face cuts)

    def key(self):
        return (round(self.rpm, 6), round(self.fz_m, 12), round(self.ap_m, 12), round(self.ae_m, 12), self.direction,
                round(self.VB_m, 12), self.coolant, round(self.y_shift_m, 12))


@dataclass
class MillRequest:
    """One reference value or curve point: the condition and the statistic to take from the simulation."""

    cond: MillCond | None
    stat: str = "mean"                      # mean | peak | min | max | rms3 | meanxy | toothpeakxy | plateau | torque
    axis: str = "x"                         # x feed | y normal | z axial (Altintas frame, force on the tool)
    on_workpiece: bool = False              # the source measured the force on the workpiece (sign flipped)
    magnitude: bool = True                  # compare |value| (frame sign not defined by the source)
    status: str = COMPARED
    reason: str = ""


@dataclass
class MillPlan:
    tool: MillTool
    requests_values: list
    requests_curves: list
    notes: list = field(default_factory=list)


AXES = {"x": 0, "y": 1, "z": 2}


def statistic(run: CSIM.MillRun, req: MillRequest) -> float:
    F = run.force_T0_N
    if req.on_workpiece:
        F = -F
    if req.stat == "torque":
        return float(abs(np.mean(run.torque_Nm)))
    if req.stat == "rms3":
        rms = np.sqrt(np.mean(F ** 2, axis=0))
        return float(math.sqrt(float(np.sum(rms ** 2)) / 3.0))
    if req.stat == "meanxy":
        return float(np.mean(np.hypot(F[:, 0], F[:, 1])))
    if req.stat == "toothpeakxy":
        return float(np.mean(run.per_tooth_peaks(np.hypot(F[:, 0], F[:, 1]))))
    comp = F[:, AXES[req.axis]]
    if req.stat == "mean":
        value = float(np.mean(comp))
    elif req.stat == "peak":
        value = float(comp[np.argmax(np.abs(comp))])
    elif req.stat == "min":
        value = float(np.min(comp))
    elif req.stat == "max":
        value = float(np.max(comp))
    elif req.stat == "plateau":
        mag = np.linalg.norm(F, axis=1)
        engaged = mag > 0.05 * float(np.max(mag)) if np.max(mag) > 0 else np.ones(len(mag), bool)
        value = float(np.mean(comp[engaged]))
    else:
        raise ValueError(f"unknown statistic {req.stat!r}")
    return abs(value) if req.magnitude else value


def _rpm(case, col, D_m: float) -> float:
    rpm = case.value("conditions", "spindle_rpm")
    if isinstance(rpm, (int, float)):
        return float(rpm)
    vc = case.si("conditions", "vc_m_min", "cutting_speed_m_min")
    if isinstance(vc, (int, float)):
        return float(vc) / (math.pi * D_m) * 60.0
    raise ValueError(f"{case.id}: no spindle speed")


def _geometry(case, keys: dict) -> dict:
    """Tooling geometry keys from the case's tool section (values as written, sources stripped)."""
    out = {}
    for target, source in keys.items():
        if isinstance(source, tuple):
            value = None
            for s in source:
                value = case.value("tool", s)
                if value is not None:
                    break
        else:
            value = case.value("tool", source)
        if value is not None:
            out[target] = value
    return out


def _endmill_tool(case, col, kind: str, extra: dict | None = None, stickout_default: float | None = None) -> MillTool:
    g = _geometry(case, {"diameter_mm": ("diameter_mm", "D_mm"), "flutes": "flutes", "helix_deg": "helix_deg",
                         "rake_deg": ("normal_rake_deg", "rake_deg", "radial_rake_deg"),
                         "edge_radius_um": "edge_radius_um", "corner_radius_mm": ("corner_radius_mm", "R_mm")})
    clear = case.value("tool", "clearance_deg", "first_clearance_deg", "relief_deg")
    if clear is None:
        clear = AS.ENDMILL_CLEARANCE.value
        col.assume("clearance angle", f"{clear:g} deg", AS.ENDMILL_CLEARANCE.reason, "harnesses/assumptions.py")
    g["clearance_deg"] = {"primary": clear}
    D = float(g["diameter_mm"])
    L = case.value("tool", "flute_length_mm", "cutting_length_mm")
    ap = case.value("conditions", "ap_mm")
    ap_max = max(float(v) for v in ap) if isinstance(ap, (list, tuple)) else float(ap or 0.0)
    g["cutting_length_mm"] = float(L) if L is not None else max(2.0 * D, 1.2 * ap_max)
    g.setdefault("overall_length_mm", g["cutting_length_mm"] + 2.0 * D)
    g["shank"] = {"diameter_mm": D, "length_mm": 2.0 * D}
    if extra:
        g.update(extra)
    if "edge_radius_um" not in g:
        r = H.edge_radius_m(case, None, col, "solid")
        g["edge_radius_um"] = r * 1e6
    tool_class = _text(case.section("tool").get("tool_class")) or "carbide"
    stick = case.si("tool", "stickout_mm", "overhang_mm", "gauge_length_mm")
    return MillTool(kind, g, tool_class, float(stick) if isinstance(stick, (int, float)) else stickout_default)


def _strip_of(cond: MillCond, R: float) -> tuple[float, float]:
    if cond.direction == "centred":
        return -0.5 * cond.ae_m + cond.y_shift_m, 0.5 * cond.ae_m + cond.y_shift_m
    lo, hi = CSIM.strip(cond.direction, cond.ae_m, R)
    return lo + cond.y_shift_m, hi + cond.y_shift_m


class MillSim:
    """Simulations of one milling case, one per distinct condition (cached)."""

    def __init__(self, db, case, col, tool: MillTool):
        self.db, self.case, self.col, self.mt = db, case, col, tool
        self.pipe = CSIM.Pipeline(db, case, col, "milling", tool_kind=tool.kind)
        item = CSIM.tool_item(tool.kind, tool.tool_class, self.pipe.tool_mat, self.pipe.coating, tool.geometry,
                              f"validation-{case.id}")
        self.geo = CSIM.build_geometry(item, stickout_m=tool.stickout_m)
        self.R = 0.5 * float(self.geo.props.diameter_m)
        for note in self.geo.notes:
            col.note(f"Tool model: {note}")
        self.runs: dict = {}
        self.layouts: dict = {}

    def run(self, cond: MillCond) -> CSIM.MillRun:
        key = cond.key()
        if key in self.runs:
            return self.runs[key]
        omega = -2.0 * math.pi * cond.rpm / 60.0
        factor = coolant_factor(self.db, self.case, self.col, cond.coolant, self.pipe.work, self.pipe.tool,
                                abs(omega) * self.R, self.pipe.ambient_K)
        lkey = (round(cond.rpm, 6), round(cond.VB_m, 12), cond.coolant)
        lay = self.layouts.get(lkey)
        if lay is None:
            lay = CSIM.layout(self.pipe, self.geo, omega, VB_m=cond.VB_m, friction_factor=factor,
                              key=f"m-{cond.coolant}")
            if self.mt.insert_stations_only:
                lay = _insert_only(lay)
            self.layouts[lkey] = lay
        y_lo, y_hi = _strip_of(cond, self.R)
        run = CSIM.mill(lay, omega, cond.fz_m, cond.ap_m, y_lo, y_hi, rho_kg_m3=float(self.pipe.work.rho_kg_m3))
        self.runs[key] = run
        return run


def _insert_only(lay: CSIM.Layout) -> CSIM.Layout:
    """Round-insert cutters built as toroidal end mills: the end-edge elements (inside the insert's lowest
    point, the cutter body) are made invalid so only the insert's arc cuts. The kernel visits the elements of
    the tool's station lists (``st_ptr`` / ``st_elem``, built from the valid flags), so the lists and the packed
    element rows are rebuilt from the new flags (flipping ``valid`` alone left the end edges cutting)."""
    from dataclasses import replace
    from ...cutting.abi import with_packed
    kind = np.asarray(lay.geo.edge.elem_kind)
    tool = lay.tool
    valid = tool.valid.copy()
    valid[np.asarray(kind).reshape(-1) == EDGE_END] = 0
    station = np.asarray(tool.station, np.int64)
    tooth = np.asarray(tool.tooth, np.int64)
    order = [int(e) for e in np.lexsort((tooth, station)) if valid[e] != 0]
    nst = int(tool.st_z_m.shape[0])
    st_ptr = np.zeros(nst + 1, np.int64)
    for e in order:
        st_ptr[station[e] + 1] += 1
    st_ptr = np.ascontiguousarray(np.cumsum(st_ptr), dtype=np.int32)
    st_elem = np.ascontiguousarray(order if order else [0], dtype=np.int32)
    return replace(lay, tool=with_packed(tool._replace(valid=valid, st_ptr=st_ptr, st_elem=st_elem)))


def milling(case, db) -> HarnessOutput:
    reader = MILLING.get(case.id)
    col = H.Collector()
    plan = reader(case, col) if reader is not None else generic_plan(case, col)
    sim = MillSim(db, case, col, plan.tool)
    for note in plan.notes:
        col.note(note)

    def predict(req: MillRequest, unit: str, name: str, reference=None) -> Prediction:
        if req.cond is None or req.status != COMPARED:
            value = None
            if req.cond is not None:
                value = from_si(statistic(sim.run(req.cond), MillRequest(req.cond, req.stat, req.axis,
                                                                         req.on_workpiece, req.magnitude)), unit)
            return not_compared(req.reason, value=value)
        run = sim.run(req.cond)
        value_si = statistic(run, req)
        if req.magnitude and isinstance(reference, (int, float)) and reference < 0.0:
            value_si = -abs(value_si)          # magnitudes compared: the simulated value takes the reference's sign
        c = req.cond
        note = (f"{req.stat} {req.axis.upper() if req.stat not in ('rms3', 'meanxy', 'toothpeakxy', 'torque') else ''}"
                f"{' (on the workpiece)' if req.on_workpiece else ''}{', magnitude' if req.magnitude else ''}; "
                f"{c.rpm:.0f} rpm, fz {c.fz_m * 1e3:g}, ap {c.ap_m * 1e3:g}, ae {c.ae_m * 1e3:g} mm, {c.direction}"
                + (f", VB {c.VB_m * 1e3:.2f} mm" if c.VB_m > 0 else "") + (f", {c.coolant}" if c.coolant != "dry" else ""))
        mean = run.force_T0_N.mean(axis=0)
        return Prediction(from_si(value_si, unit), COMPARED, note.strip(), component=None if req.stat == "torque" else MILL,
                          extra={"mean_tool_XYZ_N": [float(v) for v in mean], "engaged_stations": run.engaged_stations})

    values = [predict(req, str(entry.get("unit") or ""), str(entry.get("name")), entry.get("value"))
              for req, entry in zip(plan.requests_values, case.reference_values)]
    curves = []
    for reqs, curve in zip(plan.requests_curves, case.reference_curves):
        curves.append([predict(req, str(curve.get("y_unit") or ""), str(curve.get("name")), pt[1])
                       for req, pt in zip(reqs, curve.get("points") or [])])
    settings = sim.pipe.settings_dict()
    settings.update({"tool_type": plan.tool.kind, "edge_element_m": CSIM.EDGE_ELEMENT_M,
                     "steps": "<= 0.5 deg, >= 96 per tooth", "runout": 0.0})
    col.assume("runout", "0", "not stated in the case: an ideal concentric tool", "est")
    return HarnessOutput(values, curves, MODEL_CHAIN_MILL, settings, col.assumptions, col.provenance, col.notes, None)


# ------------------------------------------------------------------------------------------------ readers
def _direction(case, default: str = "down") -> str:
    text = _text(case.section("conditions").get("direction")).lower()
    if "slot" in text or "full" in _text(case.section("conditions").get("radial_engagement")).lower():
        return "slot"
    if text.startswith("up"):
        return "up"
    if text.startswith("down") or "climb" in text:
        return "down"
    return default


def _num(case, key, default=None):
    value = case.si("conditions", key)
    return float(value) if isinstance(value, (int, float)) else default


def generic_plan(case, col) -> MillPlan:
    """A milling case without a reader: one condition from the case's scalar keys, every value the mean of
    the component its name starts with (Fx feed, Fy normal, Fz axial, M/T torque)."""
    kind = _text(case.section("tool").get("type")) or "square_endmill"
    tool = _endmill_tool(case, col, kind)
    D = float(tool.geometry["diameter_mm"]) * 1e-3
    cond = MillCond(_rpm(case, col, D), _num(case, "fz_mm_tooth"), _num(case, "ap_mm"),
                    _num(case, "ae_mm", D), _direction(case), coolant=coolant_mode(case))

    def req(name):
        head = str(name).lower()
        if head.startswith(("m", "t")):
            return MillRequest(cond, "torque")
        axis = "x" if head.startswith("fx") else ("y" if head.startswith("fy") else "z")
        return MillRequest(cond, "mean", axis)
    return MillPlan(tool, [req(v.get("name")) for v in case.reference_values],
                    [[req(c.get("name")) for _ in c.get("points") or []] for c in case.reference_curves],
                    ["Generic reading: one condition from the case's scalar keys; mean force components by name."])


def _plodzien(case, col) -> MillPlan:
    """Round-insert (R300, 4 mm insert radius) face mill, one insert, as a toroidal end mill D 42 mm with a
    4 mm corner radius; down milling, ae 20 mm from the exit side; references are per-pass maxima (the case:
    'taken as per-pass maxima: compare with the simulated peak of each component')."""
    D = case.si("tool", "diameter_mm")
    rc = case.value("tool", "insert_radius_mm")
    tool = MillTool("toroidal_endmill", {"diameter_mm": D * 1e3, "flutes": int(case.value("tool", "inserts_used") or 1),
                                         "helix_deg": 0.0, "rake_deg": case.value("tool", "rake_deg"),
                                         "clearance_deg": {"primary": case.value("tool", "relief_deg")},
                                         "edge_radius_um": case.value("tool", "edge_radius_um"),
                                         "corner_radius_mm": rc, "cutting_length_mm": rc, "overall_length_mm": 60,
                                         "shank": {"diameter_mm": 22, "length_mm": 40}, "centre_cutting": False},
                    "carbide", insert_stations_only=True)
    rpm = _rpm(case, col, D)
    ae = case.si("conditions", "ae_mm")
    axis = {"FfN": "y", "Ff": "x", "Fa": "z"}
    curves = []
    for c in case.reference_curves:
        name = str(c["name"])
        comp, fz = name.split("_fz")
        reqs = [MillRequest(MillCond(rpm, float(fz) * 1e-3, float(x) * 1e-3, ae, "down"), "peak", axis[comp])
                for x, _ in c["points"]]
        curves.append(reqs)
    return MillPlan(tool, [], curves, ["Ff along the feed = X, FfN normal to the feed in the face plane = Y, Fa axial "
                                       "= Z; peak magnitudes (source frame signs not compared)."])


def _reznicek(case, col) -> MillPlan:
    """High-feed face mill (SECO, D 20, 3 inserts, entering angle 15 deg est., corner radius 1 mm, face angle
    11 deg as the insert rake), centred over the 14 mm surface; each value is the mean of ten maxima of the
    record (reference note): the peak of each component. X taken along the feed, Y normal (the source's
    frame is the dynamometer's; magnitudes)."""
    D = case.si("tool", "diameter_mm") * 1e3
    tool = MillTool("face_mill", {"diameter_mm": D, "cutting_length_mm": case.value("tool", "max_ap_mm") or 1.0,
                                  "overall_length_mm": case.value("tool", "functional_length_mm") or 30,
                                  "inserts": {"count": int(case.value("tool", "inserts")), "ic_mm": 6.4,
                                              "lead_angle_deg": case.value("tool", "entering_angle_deg"),
                                              "nose_radius_mm": case.value("tool", "corner_radius_mm"),
                                              "chip_breaker": {"rake_deg": case.value("tool", "rake_deg")},
                                              "edge_length_mm": 6.4},
                                  "shank": {"diameter_mm": 16, "length_mm": 40}}, "carbide")
    rpm = _rpm(case, col, D * 1e-3)
    ae = case.si("conditions", "ae_mm")
    curves = []
    for c in case.reference_curves:
        comp, ap = str(c["name"]).split("_ap")
        reqs = [MillRequest(MillCond(rpm, float(x) * 1e-3, float(ap) * 1e-3, ae, "centred", y_shift_m=0.0), "peak",
                            comp[1].lower(), on_workpiece=True) for x, _ in c["points"]]
        curves.append(reqs)
    col.assume("insert edge length", "6.4 mm", "the insert width LPHT0603 (case text: width 6.4 mm)", "case")
    return MillPlan(tool, [], curves, ["Centred cut: strip y in [-ae/2, ae/2]."])


def _villarrazo(case, col) -> MillPlan:
    """Indexable shoulder mill D 16 mm, 2 AOMT inserts (entering 90 deg, corner 1.2 mm est., rake 10 deg est.);
    F_RMS = sqrt((Fx^2 + Fy^2 + Fz^2) / 3) of the RMS values over a pass (Eqs. 1-2); only the 630 mm values are
    fresh-tool references (case notes); the stated rpm is primary (case conditions)."""
    D = case.si("tool", "diameter_mm") * 1e3
    tool = MillTool("shoulder_mill", {"diameter_mm": D, "cutting_length_mm": 11.0, "overall_length_mm": 85,
                                      "inserts": {"count": int(case.value("tool", "inserts")), "ic_mm": 6.8,
                                                  "lead_angle_deg": case.value("tool", "entering_angle_deg"),
                                                  "nose_radius_mm": case.value("tool", "corner_radius_mm"),
                                                  "chip_breaker": {"rake_deg": case.value("tool", "rake_deg")},
                                                  "clearance_deg": case.value("tool", "clearance_deg"),
                                                  "edge_length_mm": 11.0},
                                      "shank": {"diameter_mm": 16, "length_mm": 40}}, "carbide")
    col.assume("insert edge length", "11 mm", "AOMT 12 insert (12 mm class): usable edge about 11 mm", "est")
    rpm = float((case.value("conditions", "spindle_rpm") or [6000])[0])
    cond = {mode: MillCond(rpm, case.si("conditions", "fz_mm_tooth"), case.si("conditions", "ap_mm"),
                           case.si("conditions", "ae_mm"), _direction(case), coolant="mql") for mode in ("mql",)}
    reqs = []
    for v in case.reference_values:
        name = str(v["name"])
        if name.endswith("_630mm"):
            reqs.append(MillRequest(cond["mql"], "rms3", magnitude=True))
        else:
            reqs.append(MillRequest(cond["mql"], "rms3", status="not_compared",
                                    reason="worn tool after 5040+ mm of cut, wear not stated (case: only the 630 mm "
                                           "values are fresh-tool references)"))
    return MillPlan(tool, reqs, [], ["MQL and CryoMQL get the same prediction: the CO2 of CryoMQL is not modelled; "
                                     "the MQL oil enters through the lubrication friction factor."])


def _zylka(case, col) -> MillPlan:
    """Solid carbide end mill D 16, 3 flutes, 45 deg helix; FXY = sqrt(Fx^2 + Fy^2); MeanF = mean of a window,
    MeanUpF = mean of per-tooth peaks after a 1150 Hz low-pass filter (case force_definition). The x of each
    curve is the edge radius; MAR 40-110 um tools carry a cylindrical margin the tool model cannot represent."""
    tool = _endmill_tool(case, col, "square_endmill")
    D = float(tool.geometry["diameter_mm"]) * 1e-3
    rpm = _rpm(case, col, D)
    base = MillCond(rpm, case.si("conditions", "fz_mm_tooth"), case.si("conditions", "ap_mm"),
                    case.si("conditions", "ae_mm"), _direction(case), coolant="flood")
    curves = []
    for c in case.reference_curves:
        name = str(c["name"])
        margin_ok = "MAR0-30" in name
        reqs = []
        for x, _ in c["points"]:
            cond = MillCond(**{**base.__dict__})
            cond_key = ("edge", float(x))
            reqs.append(_ZylkaRequest(cond, float(x) * 1e-6, "meanxy" if name.startswith("MeanF") else "toothpeakxy",
                                      margin_ok, cond_key))
        curves.append(reqs)
    return MillPlan(tool, [], curves, ["Edge radius per point from the curve's x; mean FXY is compared; MeanUpF (peaks "
                                       "of the 1150 Hz low-pass filtered signal, filter order not stated) is listed "
                                       "unfiltered, not counted."])


class _ZylkaRequest(MillRequest):
    """A Zylka point: its edge radius and statistic (the curve's x is the edge radius)."""

    def __init__(self, cond, edge_radius_m, stat_name, margin_ok, cond_key):
        status = COMPARED if (margin_ok and stat_name == "meanxy") else "not_compared"
        reason = ("" if status == COMPARED else
                  ("the cylindrical margin (MAR 40-110 um) of this tool cannot be represented: tooling's end mill has "
                   "no margin land (the case directs a model without margin contact to the MAR 0-30 curves)"
                   if not margin_ok else "peak of a 1150 Hz low-pass filtered signal (filter order not stated): the "
                   "simulated unfiltered mean per-tooth peak is listed"))
        super().__init__(cond, stat_name, "x", False, True, status, reason)
        self.edge_radius_m = edge_radius_m
        self.stat_name = stat_name
        self.margin_ok = margin_ok
        self.cond_key = cond_key


def _lin(case, col) -> MillPlan:
    """Solid carbide end mill D 6, 2 flutes, 35 deg helix, normal rake 12.29 deg, TiAlN; average forces over a
    revolution (case notes), the test of each value from its label (a1 ... d3) and the noted ae, ap, fz."""
    tool = _endmill_tool(case, col, "square_endmill", stickout_default=18e-3)
    rpm = _rpm(case, col, float(tool.geometry["diameter_mm"]) * 1e-3)
    tests = {}
    for v in case.reference_values:
        note = str(v.get("note") or "")
        label = str(v["name"]).split("_")[1]
        m = re.search(r"ae ([0-9.]+), ap ([0-9.]+), fz ([0-9.]+)", note)
        if m:
            tests[label] = MillCond(rpm, float(m.group(3)) * 1e-3, float(m.group(2)) * 1e-3, float(m.group(1)) * 1e-3,
                                    _direction(case))
    reqs = []
    for v in case.reference_values:
        comp, label = str(v["name"]).split("_")
        reqs.append(MillRequest(tests[label], "mean", comp[1].lower(), on_workpiece=True))
    return MillPlan(tool, reqs, [], ["Mean forces; the dynamometer's X/Y orientation relative to the feed is taken as "
                                     "X feed, Y normal (magnitudes compared, signs as printed are not)."])


def _wang(case, col) -> MillPlan:
    """HSS end mill D 8, 4 flutes, 30 deg helix; mean forces over a revolution; X normal to the feed, Y feed
    (operation_note); tests 1-3 at 830 rpm, fz 0.05, ap 3.5, ae 0.6 mm and 4-6 at 900 rpm, fz 0.03, ap 4.0,
    ae 0.7 mm with the flank wear VB in the names."""
    tool = _endmill_tool(case, col, "square_endmill", stickout_default=30e-3)
    rpms = case.value("conditions", "spindle_rpm")
    fzs = case.value("conditions", "fz_mm_tooth")
    aps = case.value("conditions", "ap_mm")
    aes = case.value("conditions", "ae_mm")
    reqs = []
    for v in case.reference_values:
        name = str(v["name"])
        test = int(re.search(r"test([0-9]+)", name).group(1))
        vb = float(re.search(r"VB([0-9.]+)", name).group(1))
        i = 0 if test <= 3 else 1
        cond = MillCond(float(rpms[i]), float(fzs[i]) * 1e-3, float(aps[i]) * 1e-3, float(aes[i]) * 1e-3, "down",
                        VB_m=vb * 1e-3)
        axis = {"Fx": "y", "Fy": "x", "Fz": "z"}[name[:2]]
        reqs.append(MillRequest(cond, "mean", axis, on_workpiece=True))
    return MillPlan(tool, reqs, [], ["Source X (normal to the feed) = Y here, source Y (feed) = X; magnitudes."])


def _zheng(case, col) -> MillPlan:
    """Ball end mill D 8 (flutes 4, helix 30, rake 0: est. in the case), full slot ap 0.5 mm at 1000 rpm; mean
    forces in the source's f-n-z frame (x feed, y normal, z axial)."""
    tool = _endmill_tool(case, col, "ball_endmill", stickout_default=32e-3)
    rpm = float(case.value("conditions", "spindle_rpm"))
    ap = case.si("conditions", "ap_mm")
    D = float(tool.geometry["diameter_mm"]) * 1e-3
    curves = []
    for c in case.reference_curves:
        axis = str(c["name"])[1].lower()
        curves.append([MillRequest(MillCond(rpm, float(x) * 1e-3, ap, D, "slot"), "mean", axis, on_workpiece=True)
                       for x, _ in c["points"]])
    return MillPlan(tool, [], curves, ["Mean forces in the source's frame (x feed, y normal, z axial), magnitudes."])


def _engin_bullnose(case, col) -> MillPlan:
    """Bull-nose cutter D 31.75 with round inserts (R 4.7625 mm, centre at 11.1125 mm), one insert, as a
    toroidal end mill; slot ap 4 mm, 217 rpm, 0.1 mm/tooth; forces on the tool in Altintas' frame (the
    references' signs: Fx most negative, Fy and Fz peaks)."""
    D = case.value("tool", "D_mm")
    R = case.value("tool", "R_mm")
    tool = MillTool("toroidal_endmill", {"diameter_mm": D, "flutes": int(case.value("tool", "inserts_used") or 1),
                                         "helix_deg": 0.0, "rake_deg": case.value("tool", "rake_deg"),
                                         "clearance_deg": {"primary": 11.0}, "corner_radius_mm": R,
                                         "cutting_length_mm": R, "overall_length_mm": 80,
                                         "shank": {"diameter_mm": 25, "length_mm": 50}, "centre_cutting": False},
                    "carbide", insert_stations_only=True)
    col.assume("insert clearance", "11 deg", "P-type round insert (case: 'a P-type (11 deg clearance) round insert')",
               "case")
    tool.geometry["edge_radius_um"] = H.edge_radius_m(case, None, col, "insert") * 1e6
    cond = MillCond(float(case.value("conditions", "spindle_rpm")), case.si("conditions", "fz_mm_tooth"),
                    case.si("conditions", "ap_mm"), float(D) * 1e-3, "slot")
    stat = {"Fx_min": ("min", "x"), "Fy_max": ("max", "y"), "Fz_max": ("max", "z")}
    reqs = [MillRequest(cond, stat[v["name"]][0], stat[v["name"]][1], magnitude=False) for v in case.reference_values]
    return MillPlan(tool, reqs, [], ["Signed extremes of the force on the tool (Altintas frame)."])


def _engin_taperball(case, col) -> MillPlan:
    """One-flute tapered ball end mill (R 3 mm, 4 deg taper, lead 105 mm, rake 10, relief 10) in a 10 mm slot
    at 500 rpm, 0.05 mm/tooth: built as a ball end mill with a cylindrical flank (tooling has no taper: the
    radius at the top of the cut is 3.0 instead of 3.49 mm) and the helix of the stated lead at the ball
    radius."""
    R = float(case.value("tool", "R_mm"))
    lead = float(case.value("tool", "lead_mm"))
    helix = math.degrees(math.atan(2.0 * math.pi * R / lead))
    tool = MillTool("ball_endmill", {"diameter_mm": 2 * R, "flutes": int(case.value("tool", "flutes")),
                                     "helix_deg": helix, "rake_deg": case.value("tool", "rake_deg"),
                                     "clearance_deg": {"primary": case.value("tool", "relief_deg")},
                                     "cutting_length_mm": float(case.value("tool", "h_mm")), "overall_length_mm": 70,
                                     "shank": {"diameter_mm": 6, "length_mm": 30}}, "carbide")
    tool.geometry["edge_radius_um"] = H.edge_radius_m(case, None, col, "solid") * 1e6
    col.assume("flute helix", f"{helix:.2f} deg", "from the stated lead 105 mm at the ball radius (constant helix)",
               "case")
    col.assume("taper", "none (cylindrical flank)", "tooling's ball end mill has no taper angle", "est")
    cond = MillCond(float(case.value("conditions", "spindle_rpm")), case.si("conditions", "fz_mm_tooth"),
                    case.si("conditions", "ap_mm"), 2 * R * 1e-3, "slot")
    stat = {"Fy_max": ("max", "y"), "Fx_min": ("min", "x"), "Fz_plateau": ("plateau", "z")}
    reqs = [MillRequest(cond, stat[v["name"]][0], stat[v["name"]][1], magnitude=False) for v in case.reference_values]
    return MillPlan(tool, reqs, [], ["Signed extremes of the force on the tool (Altintas frame); Fz plateau = mean of "
                                     "Fz while the flute is in the cut."])


def _zawada(case, col) -> MillPlan:
    """Thin-wall finishing end mill D 12, 4 flutes, 41 deg helix (Kyocera SGS 44748, TiB2), ap 48, ae 0.2 mm,
    fz 0.025, down; the peak force normal to the wall (Fx1) of the first pass against vc; rigid wall (the
    wall thickness of each curve does not enter a rigid-workpiece model)."""
    tool = _endmill_tool(case, col, "square_endmill", stickout_default=60e-3)
    D = float(tool.geometry["diameter_mm"]) * 1e-3
    curves = []
    for c in case.reference_curves:
        reqs = []
        for x, _ in c["points"]:
            rpm = to_si(float(x), "m/min") / (math.pi * D) * 60.0
            reqs.append(MillRequest(MillCond(rpm, case.si("conditions", "fz_mm_tooth"), case.si("conditions", "ap_mm"),
                                             case.si("conditions", "ae_mm"), "down"), "peak", "y"))
        curves.append(reqs)
    return MillPlan(tool, [], curves, ["Rigid wall and tool: the measured peaks include the wall's compliance and "
                                       "dynamics (case notes), which this force harness does not couple."])


MILLING = {
    "force-facemill-42crmo4-plodzien2023-toroidal": _plodzien,
    "force-highfeed-ss1.4541-reznicek2023": _reznicek,
    "force-mill-aisi1045-villarrazo2023-mql": _villarrazo,
    "force-sidemill-al7075-zylka2023-edge-microgeometry": _zylka,
    "force-sidemill-in718-lin2021-annealed": _lin,
    "force-sidemill-ss304-wang2025-hss": _wang,
    "force-slot-in718-zheng2023-ballend": _zheng,
    "force-slot-ti64-engin-bullnose": _engin_bullnose,
    "force-slot-ti64-engin-taperball": _engin_taperball,
    "force-thinwall-al7050-zawada2025-finish": _zawada,
}


def _zylka_milling(case, db) -> HarnessOutput:
    col = H.Collector()
    plan = _zylka(case, col)
    for note in plan.notes:
        col.note(note)
    sims: dict = {}
    curves = []
    for reqs, curve in zip(plan.requests_curves, case.reference_curves):
        unit = str(curve.get("y_unit") or "")
        preds = []
        for req in reqs:
            r_um = round(req.edge_radius_m * 1e6, 6)
            if r_um not in sims:
                geometry = dict(plan.tool.geometry)
                geometry["edge_radius_um"] = r_um
                sims[r_um] = MillSim(db, case, col if not sims else H.Collector(), MillTool(plan.tool.kind, geometry,
                                                                                         plan.tool.tool_class,
                                                                                         plan.tool.stickout_m))
            sim = sims[r_um]
            run = sim.run(req.cond)
            value = from_si(statistic(run, MillRequest(req.cond, req.stat_name, "x", False, True)), unit)
            c = req.cond
            note = (f"{req.stat_name}; edge radius {r_um:g} um; {c.rpm:.0f} rpm, fz {c.fz_m * 1e3:g}, ap {c.ap_m * 1e3:g}"
                    f", ae {c.ae_m * 1e3:g} mm, {c.direction}, flood")
            if req.status == COMPARED:
                preds.append(Prediction(value, COMPARED, note, component=MILL))
            else:
                preds.append(not_compared(req.reason + f" ({note})", value=value))
        curves.append(preds)
    first = next(iter(sims.values()))
    settings = first.pipe.settings_dict()
    settings.update({"tool_type": plan.tool.kind, "edge_element_m": CSIM.EDGE_ELEMENT_M, "runout": 0.0})
    col.assume("runout", "0", "not stated in the case: an ideal concentric tool", "est")
    return HarnessOutput([], curves, MODEL_CHAIN_MILL, settings, col.assumptions, col.provenance, col.notes, None)


MILLING_OPERATIONS = ("peripheral_milling", "slotting", "face_milling", "ball_end_milling")


@harness("cutting_force", models=("cnctwin.cutting.coefficients", "cnctwin.cutting.kernels", "cnctwin.tooling",
                                  "cnctwin.mechanics.coefficients"),
         description="The cutting domain's force model at the case's engagement: rows of the coefficient pipeline "
                     "(MechanicsModel: Oxley / Johnson-Cook, calibrated to the material's measured Kienzle or "
                     "mechanistic data of the pairing where the database has them) integrated by the domain's "
                     "kernels over a prescribed steady cut (turn_step on a polar field for orthogonal tube turning, "
                     "cut_step with analytic straight-cut engagement for milling); each reference statistic (mean, "
                     "peak, extreme, RMS) taken from the simulated force on the tool.")
def cutting_force(case, db) -> HarnessOutput:
    if case.operation in ("orthogonal_turning", "oblique_turning", "grooving"):
        return orthogonal(case, db)
    if case.id == "force-sidemill-al7075-zylka2023-edge-microgeometry":
        return _zylka_milling(case, db)
    if case.operation in MILLING_OPERATIONS:
        return milling(case, db)
    raise ValueError(f"{case.id}: operation {case.operation!r} is not handled by the cutting_force harness")


@harness("torque", models=("cnctwin.cutting.kernels", "cnctwin.cutting.coefficients", "cnctwin.tooling"),
         description="Milling spindle torque: the process torque about the tool axis of the same simulation as "
                     "cutting_force (cut_step over a steady straight cut), mean over a revolution unless the case's "
                     "reader says otherwise.")
def torque(case, db) -> HarnessOutput:
    if case.operation not in MILLING_OPERATIONS:
        raise ValueError(f"{case.id}: torque cases are milling cases")
    return milling(case, db)


# ================================================================================================ discussion
def _mean_signed(comps) -> float | None:
    vals = [c.error_rel for c in comps if c.status == COMPARED and c.error_rel is not None and math.isfinite(c.error_rel)]
    return sum(vals) / len(vals) if vals else None


@discussion("cutting_force")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        comps = [c for c in r.comparisons if c.counted]
        misses = [c for c in comps if c.within_tolerance is False]
        if not misses:
            continue
        by_comp = {}
        for c in comps:
            by_comp.setdefault(c.component or "-", []).append(c)
        parts = []
        for comp, cs in sorted(by_comp.items()):
            m = _mean_signed(cs)
            inside = sum(1 for c in cs if c.within_tolerance)
            parts.append(f"{comp}: {inside}/{len(cs)} within, mean signed error {m * 100:+.0f} %" if m is not None
                         else f"{comp}: {inside}/{len(cs)}")
        cal = r.output.settings.get("calibration")
        text = "; ".join(parts) + f". Coefficients: {'calibrated to ' + cal + ' data' if cal not in (None, 'none') else 'mechanics alone (no measured data of the pairing)'}."
        small_h = [c for c in misses if c.x is not None and c.x_name in ("f", "h") and c.x is not None and
                   c.x_unit in ("mm/rev", "mm") and float(c.x) <= 0.03]
        if small_h:
            text += (f" {len(small_h)} miss(es) at h <= 0.03 mm, where the edge (ploughing) force dominates: the "
                     "(2 + pi) k contact pressure below the stagnation point is an upper-bound estimate "
                     "(docs/models/mechanics.md 4).")
        if r.case.material_id == "aisi1045":
            text += " Ck45 forms a built-up edge below about 70 m/min (database BUE window); the force rows carry no BUE."
        if r.case.material_id == "ti6al4v":
            text += " Ti6Al4V forms segmented chips; Oxley's continuous-chip solution has no cyclic shear."
        if r.case.operation in MILLING_OPERATIONS:
            text += (" Milling statistics are taken from a rigid tool and workpiece without runout; dynamometer "
                     "dynamics, runout and the unstated frame orientations of some sources are not modelled.")
        out[r.id] = text
    indep = [c for r in q.cases if r.status == "ok" and not r.calibration_dependent for c in r.comparisons if c.counted]
    if indep:
        parts = []
        for comp in (MAIN, THRUST, MILL):
            cs = [c for c in indep if c.component == comp]
            if cs:
                m = _mean_signed(cs)
                inside = sum(1 for c in cs if c.within_tolerance)
                parts.append(f"{comp} {inside}/{len(cs)} values within tolerance"
                             + (f", mean signed error {m * 100:+.0f} %" if m is not None else ""))
        out["summary"] = (H.budget_summary(q) + "Independent values: " + "; ".join(parts) + ". The thrust (feed) force "
                          "is the weak part of the orthogonal model: it is set by the edge force below the stagnation "
                          "point and the friction estimate, while the main force follows the flow stress. Milling cases "
                          "add the tool geometry the sources leave open (rake, edge radius, entering angle: listed as "
                          "assumptions per case). Cases whose material file holds Kienzle data of the same experiments "
                          "are calibration-dependent and reported apart. No coefficient was changed for this report.")
    return out


__all__ = ["cutting_force", "torque", "orthogonal", "milling", "MILLING", "statistic", "MillCond", "MillRequest",
           "coolant_factor", "coolant_mode"]
