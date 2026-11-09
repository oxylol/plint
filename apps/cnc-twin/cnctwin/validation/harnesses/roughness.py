"""Harness for roughness: cnctwin.surface's kinematic profile model with the case's feeds and tools.

The profile is the lower envelope of the generating edge placed at each pass (cnctwin.surface.kernels.
envelope_profile: the passes in time order, each lowering the surface only where the layer it removes reaches
the minimum chip thickness h_min = ratio x edge radius, Brammertz 1961; docs/models/surface.md 2), evaluated by
cnctwin.surface.roughness (ISO 4287 parameters with the ISO 4288 cut-off chosen from the feed spacing, Gaussian
filter of ISO 16610-21). The generating edge in the profile plane, per process:

* **turning**: the insert's nose radius with the major and minor edges (entering angle kappa_r, minor angle
  kappa_r' from the insert's corner angle: kappa_r' = 180 deg - corner - kappa_r), one pass per revolution at
  the feed;
* **face and end-mill floors**: the cutter's corner (corner radius, or a sharp corner) with the minor (end) edge
  at its dish angle, one pass per tooth at the feed per tooth (ideal cutter: equal teeth, no runout);
* **ball-end milling**: the scallop (cross-feed) profile of the ball (radius R, any lead or tilt: the ball's
  silhouette stays a circle of radius R) at the step-over, and the feed-mark profile (the swept sphere, radius
  R) at the feed per tooth; the roughest direction is reported (ISO 4288: perpendicular to the lay).

The h_min ratio is the material file's (surface's material props), the edge radius the case's or the tooling
default of the edge kind; the recovery ratio is the surface model's (db/models/surface-roughness.yaml). Built-up
edge, vibration, runout and side flow are not in the kinematic profile: they are named in the report where the
measured roughness lies far above it.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ...surface import abi as SA
from ...surface import kernels as SK
from ...surface import params as SP
from ...surface.lines import N_PARABOLA
from ...surface.roughness import IsoRules, evaluate, primary_params
from ..cases import from_si, to_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import common as H

DEG = math.pi / 180.0
PASSES = 60                       # passes of a profile (the ISO evaluation takes up to 5 cut-offs inside them)
WINDOW_SPACINGS = 6.0             # half-width of a pass's template window in feed spacings (surface half_spacings)
# ISO 1832 insert shape letter -> corner angle (deg); ISO 5608 holder style letter -> entering angle kappa_r (deg)
INSERT_CORNER = {"C": 80.0, "D": 55.0, "R": 180.0, "S": 90.0, "T": 60.0, "V": 35.0, "W": 80.0, "K": 55.0}
HOLDER_KAPPA = {"A": 90.0, "B": 75.0, "D": 45.0, "F": 90.0, "G": 90.0, "J": 93.0, "K": 75.0, "L": 95.0, "N": 63.0,
                "Q": 107.5, "S": 45.0, "V": 72.5, "X": 117.5}
MODEL_CHAIN = ["generating edge in the profile plane (case tool)", "cnctwin.surface.kernels.envelope_profile "
               "(lower envelope with the minimum chip thickness)", "cnctwin.surface.roughness.evaluate (ISO 4287, "
               "ISO 4288 cut-off, ISO 16610-21 filter)"]


# ================================================================================================ profile
def corner_template(r_m: float, kappa_r: float, kappa_minor: float, depth_m: float, du: float) -> np.ndarray:
    """(u, w) of a corner of radius r between the major edge (entering angle kappa_r, feed side u < 0) and the
    minor edge (kappa_r', u > 0), w = height above the lowest point, up to the depth of cut (u along the feed
    direction of the profile; r = 0: a sharp corner)."""
    lo = -(depth_m / max(math.tan(kappa_r), 1e-9) + r_m + 4.0 * du) if kappa_r < 0.5 * math.pi - 1e-9 \
        else -(r_m + 4.0 * du)
    hi = r_m * math.sin(kappa_minor) + depth_m / max(math.tan(kappa_minor), 1e-9) if kappa_minor > 1e-9 else 1e-2
    hi = min(hi, 5e-3)
    u = np.arange(lo, hi + du, du)
    zt_major = -r_m * math.sin(kappa_r)
    wt_major = r_m - r_m * math.cos(kappa_r)
    zt_minor = r_m * math.sin(kappa_minor)
    wt_minor = r_m - r_m * math.cos(kappa_minor)
    nose = r_m - np.sqrt(np.clip(r_m * r_m - u * u, 0.0, None))
    major = wt_major + (zt_major - u) / max(math.tan(kappa_r), 1e-12) if kappa_r < 0.5 * math.pi - 1e-9 \
        else np.where(u < zt_major, np.inf, 0.0)
    minor = wt_minor + (u - zt_minor) * math.tan(kappa_minor)
    w = np.where(u <= zt_major, major, np.where(u >= zt_minor, minor, nose))
    keep = np.isfinite(w) & (w <= depth_m)
    return np.stack([u[keep], w[keep]], axis=1)


def circle_template(R_m: float, half_width_m: float, du: float) -> np.ndarray:
    u = np.arange(-half_width_m, half_width_m + du, du)
    return np.stack([u, R_m - np.sqrt(np.clip(R_m * R_m - u * u, 0.0, None))], axis=1)


def profile_roughness(template: np.ndarray, spacing_m: float, h_min_m: float, recovery: float, rules: IsoRules):
    """(Roughness, kinematic Pt) of the envelope of ``template`` repeated every ``spacing`` (passes in order of
    increasing position, i.e. the tool moving along +u), through the surface model's envelope kernel."""
    du = rules.spacing_for(spacing_m)
    # window: the feed spacings of the surface model and the distance over which the generating edge rises by the
    # thickest layer the minimum chip thickness can leave (so that layer stays inside the passes that remove it)
    rise = (1.0 + recovery) * h_min_m
    near = np.abs(template[template[:, 1] <= 1.5 * rise, 0]) if rise > 0.0 else np.zeros(0)
    half = max(WINDOW_SPACINGS * spacing_m, 20.0 * du, 1.5 * float(np.max(near)) if len(near) else 0.0)
    cutoff = rules.cutoff_for_rsm(spacing_m)
    n = max(PASSES, int(math.ceil((4.0 * half + 7.0 * cutoff) / spacing_m)) + 10)
    n = min(n, int(1.5e6 * du / spacing_m))
    P = SA.new_passes(n)
    P.kind[:] = SA.KIND_CUT
    P.mode[:] = SA.MODE_DIRECT
    P.s[:] = np.arange(n) * spacing_m
    P.w[:] = 0.0
    P.tpl[:] = 0
    P.hmin[:] = h_min_m
    P.half[:] = half
    P.t_max[:] = 10.0 * half + 1e-3
    T = SA.new_templates([template])
    s_lo = float(P.s[0]) - half - 1e-4
    s_hi = float(P.s[-1]) + half + 1e-4
    count = int(math.ceil((s_hi - s_lo) / du)) + 1
    G = SA.new_grid(s_lo, du, count, max(len(template), N_PARABOLA, 2))
    SK.envelope_profile(P, T, G, N_PARABOLA, float(recovery))
    z = G.z
    # between the lowest points of the first and the last passes, one window trimmed at each end (lines.py)
    i_lo = int(math.ceil((float(P.s[0]) + 2.0 * half - s_lo) / du))
    i_hi = int(math.floor((float(P.s[-1]) - 2.0 * half - s_lo) / du))
    seg = z[i_lo:i_hi + 1]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 32:
        return None, None
    rough = evaluate(seg.copy(), du, rules, hint_rsm_m=spacing_m)
    kin = primary_params(seg.copy(), du, rules)
    return rough, kin


# ================================================================================================ inputs
def _min_chip(db, case, col) -> float:
    raw = H.raw_object(db, "material", case.material_id)
    props = SP.material_props(raw)
    ratio = float(props.min_chip_ratio)
    cut = raw.get("cutting") or {}
    entry = cut.get("min_chip_thickness_ratio")
    if entry is not None:
        src, note = H.leaf_source(entry)
        col.used(f"{case.material_id} cutting.min_chip_thickness_ratio", src, note)
    else:
        col.assume("minimum chip thickness ratio", f"{ratio:g}", "not in the material file: the surface model's "
                   "default", "db/models/surface-roughness.yaml")
    return ratio


def _edge_radius(case, col, kind: str) -> float:
    return H.edge_radius_m(case, None, col, kind)


def _iso_angles(case, col) -> tuple[float, float]:
    """(kappa_r, kappa_r') of a turning insert: the stated entering angle, else the holder's ISO 5608 style
    letter; the minor angle from the insert's ISO 1832 corner angle."""
    tool = case.section("tool")
    designation = H._text(tool.get("iso_designation") or tool.get("iso_code"))
    holder = H._text(tool.get("holder"))
    kr = case.value("tool", "cutting_edge_angle_deg", "entering_angle_deg")
    if not isinstance(kr, (int, float)):
        style = holder[2:3].upper() if len(holder) >= 3 else ""
        kr = HOLDER_KAPPA.get(style)
        if kr is None:
            raise ValueError(f"{case.id}: no entering angle and no ISO holder style to derive it")
        col.assume("entering angle", f"{kr:g} deg", f"ISO 5608 holder style letter {style!r} of {holder}", "case")
    corner = INSERT_CORNER.get(designation[:1].upper())
    if corner is None:
        raise ValueError(f"{case.id}: no ISO insert shape to derive the minor cutting edge angle")
    minor = max(180.0 - corner - float(kr), 0.5)
    col.assume("minor cutting edge angle", f"{minor:g} deg", f"180 deg - corner {corner:g} deg (ISO 1832 shape "
               f"{designation[:1]}) - kappa_r {float(kr):g} deg", "case")
    return float(kr) * DEG, minor * DEG


def _note_value(note: str, key: str) -> float | None:
    m = re.search(rf"{key} ([0-9.]+)", note or "")
    return float(m.group(1)) if m else None


def _predict(rough, kin, name: str, unit: str, text: str, extra: dict | None = None) -> Prediction:
    if rough is None:
        return no_prediction("the profile is too short to evaluate")
    param = "Rz" if name.startswith("Rz") else "Ra"
    value = rough.Rz_m if param == "Rz" else rough.Ra_m
    ex = {"Ra_um": rough.Ra_m * 1e6, "Rz_um": rough.Rz_m * 1e6, "kinematic_Pt_um": kin.Rt_m * 1e6 if kin else None,
          "cutoff_mm": rough.cutoff_m * 1e3, "method": rough.method}
    if extra:
        ex.update(extra)
    return Prediction(from_si(value, unit), COMPARED, text, extra=ex)


# ================================================================================================ cases
def _turning(case, db, col, rules, ratio, recovery) -> list:
    kr, kr2 = _iso_angles(case, col)
    r_e = _edge_radius(case, col, "chamfer" if "T0" in H._text(case.section("tool").get("iso_designation")) else
                       "insert")
    h_min = ratio * r_e
    noses = case.value("tool", "nose_radius_mm_levels")
    nose_default = case.value("tool", "nose_radius_mm")
    ap_default = case.value("conditions", "ap_mm")
    out = []
    for v in case.reference_values:
        name = str(v["name"])
        nose = _note_value(name.replace("_r", " r "), "r") if noses else None
        nose = (nose if nose is not None else float(nose_default)) * 1e-3
        f = to_si(float(v["x"]), str(v.get("x_unit") or "mm/rev"))
        ap = _note_value(str(v.get("note") or ""), "ap")
        ap = (ap if ap is not None else float(ap_default)) * 1e-3
        du = rules.spacing_for(f)
        tpl = corner_template(nose, kr, kr2, ap, du)
        rough, kin = profile_roughness(tpl, f, h_min, recovery, rules)
        out.append(_predict(rough, kin, name, str(v.get("unit") or "um"),
                            f"turning: r {nose * 1e3:g} mm, f {f * 1e3:g} mm/rev, kappa_r {kr / DEG:g}, kappa_r' "
                            f"{kr2 / DEG:g} deg, h_min {h_min * 1e6:.1f} um",
                            {"theoretical_Ra_um": f * f / (32.0 * nose) * 1e6}))
    return out


def _ball(case, db, col, rules, ratio, recovery) -> list:
    R = 0.5 * case.si("tool", "diameter_mm")
    flutes = int(case.value("tool", "flutes"))
    rpm = float(case.value("conditions", "spindle_rpm"))
    r_e = _edge_radius(case, col, "solid")
    h_min = ratio * r_e
    ae_runs = case.value("conditions", "run_ae_mm")
    feed_runs = case.value("conditions", "run_feed_mm_min")
    col.note("Ball-end milling: the scallop profile across the feed at the step-over and the feed-mark profile at "
             "the feed per tooth (radius R both); the rougher is compared. Lead and tilt angles do not change either "
             "kinematic profile (the ball's silhouette is a circle of radius R), only the cutting speed at the "
             "contact, which the kinematic model does not use.")
    out = []
    for v in case.reference_values:
        run = int(v["x"]) - 1
        ae = float(ae_runs[run]) * 1e-3
        fz = float(feed_runs[run]) * 1e-3 / 60.0 / (rpm / 60.0 * flutes)
        cross = profile_roughness(circle_template(R, min(R, 3.0 * ae + 2e-5), rules.spacing_for(ae)), ae, 0.0,
                                  recovery, rules)
        feed = profile_roughness(circle_template(R, min(R, 3.0 * fz + 2e-5), rules.spacing_for(fz)), fz, h_min,
                                 recovery, rules)
        best = cross if (cross[0] is not None and (feed[0] is None or cross[0].Ra_m >= feed[0].Ra_m)) else feed
        which = "scallops (cross-feed)" if best is cross else "feed marks"
        out.append(_predict(best[0], best[1], str(v["name"]), str(v.get("unit") or "um"),
                            f"run {run + 1}: {which}, ae {ae * 1e3:g} mm, fz {fz * 1e3:.4f} mm",
                            {"cross_Ra_um": cross[0].Ra_m * 1e6 if cross[0] else None,
                             "feed_Ra_um": feed[0].Ra_m * 1e6 if feed[0] else None}))
    return out


def _floor(case, db, col, rules, ratio, recovery) -> list:
    """Floors of face and end mills: the corner and the minor (end) edge of every tooth at fz."""
    from ...db.models import params
    tool = case.section("tool")
    corner = case.value("tool", "corner_radius_mm")
    kr = case.value("tool", "entering_angle_deg")
    kind = H._text(tool.get("type"))
    if kind == "face_mill":
        dish = float(params("tooling")["insert_minor_dish_rad"])
        col.assume("minor edge angle", f"{dish / DEG:.2f} deg", "the tooling model's minor-edge dish of milling "
                   "inserts (no wiper stated)", "db/models/tooling.yaml")
        r_e = _edge_radius(case, col, "insert")
        col.note("Ideal cutter: every insert at the same axial and radial position. The case's insert loads (Table 1 "
                 "of the source) show axial runout of the individual inserts, whose offsets are not given.")
    else:
        dish = float(params("tooling")["end_dish_rad"])
        col.assume("end-edge dish", f"{dish / DEG:.2f} deg", "the tooling model's end-tooth dish (no corner radius "
                   "stated: a sharp corner)", "db/models/tooling.yaml")
        r_e = _edge_radius(case, col, "solid")
        col.assume("measured surface", "the slot floor", "the case: not stated ('Ra was measured for each slot'); the "
                   "wall's peripheral feed marks are listed with each value", "case")
    h_min = ratio * r_e
    r = float(corner) * 1e-3 if isinstance(corner, (int, float)) else 0.0
    kr_rad = float(kr) * DEG if isinstance(kr, (int, float)) else 0.5 * math.pi
    ap = case.si("conditions", "ap_mm")
    R = 0.5 * case.si("tool", "diameter_mm")
    out = []
    for v in case.reference_values:
        fz = to_si(float(v["x"]), str(v.get("x_unit") or "mm"))
        du = rules.spacing_for(fz)
        tpl = corner_template(r, kr_rad, dish, min(ap, 1e-3), du)
        rough, kin = profile_roughness(tpl, fz, h_min, recovery, rules)
        extra = {}
        if kind != "face_mill":
            wall = profile_roughness(circle_template(R, min(R, 3.0 * fz + 2e-5), rules.spacing_for(fz)), fz, h_min,
                                     recovery, rules)
            extra["wall_Ra_um"] = wall[0].Ra_m * 1e6 if wall[0] else None
        out.append(_predict(rough, kin, str(v["name"]), str(v.get("unit") or "um"),
                            f"floor: corner r {r * 1e3:g} mm, fz {fz * 1e3:g} mm, minor angle {dish / DEG:.2f} deg, "
                            f"h_min {h_min * 1e6:.1f} um", extra))
    return out


@harness("roughness", models=("cnctwin.surface.kernels", "cnctwin.surface.roughness"),
         description="cnctwin.surface's kinematic profile: the lower envelope of the case's generating edge (insert "
                     "nose and edges, cutter corner and end edge, ball) at the case's feed or step-over, with the "
                     "minimum chip thickness rule (Brammertz), evaluated to ISO 4287 / 4288 (Gaussian filter, cut-off "
                     "from the feed spacing).")
def roughness(case, db) -> HarnessOutput:
    col = H.Collector()
    rules = IsoRules()
    ratio = _min_chip(db, case, col)
    recovery = float(SP.roughness()["recovery_ratio"])
    kind = H._text(case.section("tool").get("type"))
    if case.operation in ("longitudinal_turning", "facing", "boring") or kind == "turning_insert":
        values = _turning(case, db, col, rules, ratio, recovery)
    elif kind == "ball_endmill" or case.operation == "ball_end_milling":
        values = _ball(case, db, col, rules, ratio, recovery)
    else:
        values = _floor(case, db, col, rules, ratio, recovery)
    if not H.dry(case) or "hpc" in " ".join(str(v.get("name")) for v in case.reference_values).lower():
        col.note("Coolant does not enter the kinematic profile (it acts on built-up edge and friction, not modelled "
                 "in the profile).")
    settings = {"min_chip_ratio": ratio, "recovery_ratio": recovery, "passes": PASSES,
                "window_spacings": WINDOW_SPACINGS}
    return HarnessOutput(values, [[] for _ in case.reference_curves], MODEL_CHAIN, settings, col.assumptions,
                         col.provenance, col.notes, None)


@discussion("roughness")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        comps = [c for c in r.comparisons if c.counted and c.error_rel is not None]
        if not comps:
            continue
        mean = sum(c.error_rel for c in comps) / len(comps)
        inside = sum(1 for c in comps if c.within_tolerance)
        text = f"{inside}/{len(comps)} within tolerance, mean signed error {mean * 100:+.0f} %"
        if mean < -0.3:
            text += ("; the kinematic profile is well below the measured roughness: built-up edge, vibration, runout, "
                     "side flow and the stylus/filter of the instrument add to the feed marks and are not in the "
                     "kinematic model")
        out[r.id] = text + "."
    out["summary"] = (H.budget_summary(q) + "Where the feed marks dominate (turning at larger feeds) the kinematic "
                      "profile with the minimum chip thickness follows the measurements; at small feeds and step-overs "
                      "the non-kinematic contributions decide the measured value.")
    return out


__all__ = ["roughness", "profile_roughness", "corner_template", "circle_template"]
