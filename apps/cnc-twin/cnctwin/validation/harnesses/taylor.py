"""Harness for taylor_life: the wear model integrated to the case's wear criterion (cnctwin.wear).

Per case the wear law is built exactly as the wear domain builds it at init (``cnctwin.wear.build.wear_law``:
published Usui constants of the tool material or the work material when the pairing has them, else Usui
constants calibrated to the work material's Taylor entry with the default ``MechanicsResponse``), for
the case's tool material, coating and edge geometry. At each tested condition the response
(``MechanicsResponse``: Oxley + Smithey's land stresses + Blok's flank partition, the same equivalent
continuous orthogonal cut the calibration uses, h = feed, engaged depth = ap) gives sigma_n V_s and the
flank temperature at every land width; the time to grow the land from 0 to the criterion is the
integral of dVB / (dVB/dt) (Gauss-Legendre in VB, split where the coating wears through), the
rate being the wear kernel's (Usui branch at each temperature, coating factor while it covers the
contact, Archard floor). A notch criterion (VN) uses the wear model's notch excess at the depth-of-cut
line, VB_N = VB (1 + dH)^q. Wear curves are the inverse: VB(t) at the reference times.

Coolant other than dry: the flood/MQL friction factor from cnctwin.coolant.lubrication enters the
response's friction; the convective cooling of the flank is not part of the wear response (noted per
value). Fluids the database does not have (cryogenic sprays) give no prediction.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ...db.simap import wrap
from ...mechanics.materials import ToolSide
from ...wear import build as WB
from ...wear import calibration as CAL
from ...wear import kernels as WK
from ...wear import params as WP
from ...wear.abi import LS_ABR, LS_NOTCH_EXCESS
from ...wear.schema import settings as wear_settings
from ..cases import from_si, to_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction, not_compared
from . import common as H

FLUIDS = {"flood": "emulsion-5pct", "mql": "mql-ester"}
VB_CURVE_NODES = 48


class _Law:
    """The case's wear law and the pieces to integrate it."""

    def __init__(self, db, case, col: H.Collector, coating_text: str | None = None):
        self.case = case
        self.col = col
        self.work_mech, self.raw = H.work_material(db, case, col)
        self.work = WP.work_data(db.material(case.material_id))
        self.tool_mat_id = H.tool_material_id(case, col)
        self.coating_id = H.coating_id(case, col, coating_text)
        tm_obj = db.tool_material(self.tool_mat_id)
        self.tool_mat = WP.tool_material_data(tm_obj)
        coat_obj = None if self.coating_id == "none" else db.coating(self.coating_id)
        self.coating = WP.coating_data(coat_obj, None) if coat_obj is not None else None
        item = {"type": "turning_insert", "coating": {"id": self.coating_id}}
        self.side = ToolSide.from_db(wrap(item), tm_obj, coat_obj)
        self.substrate = ToolSide.from_db(wrap({"type": "turning_insert", "coating": {"id": "none"}}), tm_obj, None)
        self.ambient = H.ambient(case, col)
        self.rake = H.rake_rad(case, None, col)
        self.clearance = H.clearance_rad(case, col)
        self.incl = (case.value("tool", "inclination_deg") or 0.0) * H.DEG
        self.edge_radius = H.edge_radius_m(case, None, col, "chamfer" if self._chamfered() else "insert")
        fl = WP.get(WP.FLANK)
        a = max(self.clearance, fl["min_clearance_rad"])
        g = 1.0 / math.tan(a) - math.tan(self.rake)
        g = g if g > 1e-3 else 1.0 / math.tan(a)
        c = math.cos(a) * (1.0 - math.tan(a) * math.tan(self.rake))
        c = c if c > 1e-3 else math.cos(a)
        self.fg = CAL.FlankGeometry(self.rake, self.clearance, self.edge_radius, self.incl, g, c)
        self.settings = wear_settings(None)
        self.estimated: list = []
        self.friction_factor = 1.0
        self.response = CAL.MechanicsResponse(self.work_mech, self.side, self.ambient, 1.0, substrate=self.substrate)
        self.law, self.report, self.taylor = WB.wear_law(
            slot=0, work=self.work, tool_mat=self.tool_mat, coating=self.coating, coating_id=self.coating_id,
            process="turning", fg=self.fg, response=self.response, vb0_m=0.0, settings=self.settings,
            estimated=self.estimated, flow_stress=self.work_mech.jc is not None)
        for text in self.estimated:
            col.note(f"Wear model: {text}.")
        self._provenance()
        hs = self.tool_mat.hardness_table()
        hc = self.coating.hardness_table() if self.coating is not None else (None, None)
        self.abrasion = CAL.Abrasion(float(self.law.scal[LS_ABR]), hs[0], hs[1], hc[0], hc[1])
        self.notch_factor = 1.0 + float(self.law.scal[LS_NOTCH_EXCESS])

    def _chamfered(self) -> bool:
        prep = str(self.case.value("tool", "edge_preparation") or "").lower()
        return "chamfer" in prep or "t-land" in prep

    def _provenance(self) -> None:
        flank = self.report.get("flank", {})
        source = flank.get("source")
        if source == "published":
            for src, origin in zip(flank.get("src") or [], flank.get("origin") or []):
                self.col.used(f"Usui flank constants ({origin})", src)
        elif source == "taylor" and self.taylor is not None:
            H.material_provenance(self.raw, self.col, flow=True, chip_thickness=True, physical=True,
                                  tool_life_entry=self.taylor.index,
                                  friction_entry=self.work_mech.friction_for(self.side.tool_class, self.side.coating))
            return
        H.material_provenance(self.raw, self.col, friction_entry=self.work_mech.friction_for(self.side.tool_class,
                                                                                              self.side.coating))

    def with_friction(self, factor: float) -> "_Law":
        """The same law with a coolant friction factor in the response (the law itself stays as built)."""
        clone = object.__new__(_Law)
        clone.__dict__.update(self.__dict__)
        clone.friction_factor = float(factor)
        clone.response = CAL.MechanicsResponse(self.work_mech, self.side, self.ambient, float(factor),
                                               substrate=self.substrate)
        return clone

    def _rate(self, node) -> float:
        law = self.law
        coat = self.coating.factor if self.coating is not None else 1.0
        usui = 0.0
        for temp, weight, ok in zip(node.temps, node.weights, node.coat_ok):
            usui += weight * (coat if ok else 1.0) * WK.usui_rate(law.fl_A, law.fl_B, law.fl_lo, float(temp))
        return self.fg.g_vb * (usui + node.abr) * node.sv

    def time_to(self, speed_m_s: float, feed_m: float, ap_m: float, vb_lim_m: float) -> float:
        """Cutting time (s) to grow the flank land from 0 to ``vb_lim_m`` at constant conditions."""
        coated = self.coating is not None

        def response(v, vb):
            cond = CAL.Condition(v, feed_m, ap_m, vb, self.rake, self.clearance, self.edge_radius, self.incl, coated)
            return self.response.response(0, cond)
        nodes = CAL._nodes(response, speed_m_s, 0.0, vb_lim_m, self.fg, self.coating, self.abrasion,
                           int(self.settings.calib_nodes))
        total = 0.0
        for node in nodes:
            rate = self._rate(node)
            if not rate > 0.0:
                return math.inf
            total += node.weight_m / rate
        return total

    def flank_state(self, speed_m_s: float, feed_m: float, ap_m: float, vb_m: float) -> tuple[float, float]:
        """(flank temperature, flank normal stress) of the response at a land width."""
        cond = CAL.Condition(speed_m_s, feed_m, ap_m, vb_m, self.rake, self.clearance, self.edge_radius, self.incl,
                             self.coating is not None)
        r = self.response.response(0, cond)
        return float(np.sum(r.weights * r.flank_temp_K)), float(r.flank_sigma_Pa)

    def vb_curve(self, speed_m_s: float, feed_m: float, ap_m: float, vb_max_m: float) -> tuple[np.ndarray, np.ndarray]:
        """(t, VB) of the land growth from 0 to vb_max (cumulative integral on a VB grid)."""
        vbs = np.linspace(0.0, vb_max_m, VB_CURVE_NODES + 1)
        times = [0.0]
        for lo, hi in zip(vbs[:-1], vbs[1:]):
            times.append(times[-1] + self._segment(speed_m_s, feed_m, ap_m, float(lo), float(hi)))
        return np.array(times), vbs

    def _segment(self, v, f, ap, lo, hi) -> float:
        coated = self.coating is not None

        def response(vv, vb):
            cond = CAL.Condition(vv, f, ap, vb, self.rake, self.clearance, self.edge_radius, self.incl, coated)
            return self.response.response(0, cond)
        nodes = CAL._nodes(response, v, lo, hi, self.fg, self.coating, self.abrasion, 3)
        total = 0.0
        for node in nodes:
            rate = self._rate(node)
            if not rate > 0.0:
                return math.inf
            total += node.weight_m / rate
        return total

    def settings_dict(self) -> dict:
        flank = self.report.get("flank", {})
        out = {"tool_material": self.tool_mat_id, "coating": self.coating_id, "flank_law": flank.get("source"),
               "rake_deg": math.degrees(self.rake), "clearance_deg": math.degrees(self.clearance),
               "edge_radius_um": self.edge_radius * 1e6, "VB_per_VN": self.fg.g_vb, "notch_factor": self.notch_factor,
               "B_K": list(map(float, self.law.fl_B)), "A_per_Pa": list(map(float, self.law.fl_A))}
        if flank.get("source") == "taylor" and self.taylor is not None:
            cal = flank.get("calibration") or {}
            out.update({"taylor_entry": f"{self.case.material_id} tool_life[{self.taylor.index}] "
                                        f"({self.taylor.tool_class}, {self.taylor.coating or 'any'})",
                        "taylor_src": self.taylor.src, "taylor_relation": flank.get("relation"),
                        "calibration_residuals": cal.get("residual_rel")})
        return out


def _feed_from(text: str) -> float | None:
    m = re.search(r"(?<![a-z])f([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) * 1e-3 if m else None


def _vb_from(text: str) -> float | None:
    m = re.search(r"VB_?([0-9]+(?:\.[0-9]+)?)(?:mm)?", text)
    return float(m.group(1)) * 1e-3 if m else None


def _speeds(case) -> list[float]:
    """Tested cutting speeds (m/s)."""
    v = case.si("conditions", "cutting_speed_m_min")
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) for x in (v or [])]


def _criteria(case) -> tuple[float | None, float | None]:
    vb = case.si("conditions", "wear_criterion_VB_mm", "wear_criterion_VBB_mm")
    vn = case.si("conditions", "wear_criterion_VN_mm")
    if isinstance(vb, (list, tuple)):
        vb = None
    return (float(vb) if isinstance(vb, (int, float)) else None), (float(vn) if isinstance(vn, (int, float)) else None)


@harness("taylor_life", models=("cnctwin.wear.build.wear_law", "cnctwin.wear.calibration", "cnctwin.wear.kernels.usui_rate"),
         description="The wear law as the wear domain builds it (published Usui constants, else calibrated to the "
                     "database Taylor entry with the mechanics response), integrated from a fresh edge to the case's "
                     "VB (or VN) criterion at each tested speed, feed and depth (equivalent orthogonal cut, h = f).")
def taylor_life(case, db) -> HarnessOutput:
    col = H.Collector()
    base = _Law(db, case, col)
    laws = {"base": base}
    vb_crit, vn_crit = _criteria(case)
    feed_case = case.si("conditions", "feed_mm_rev")
    ap = case.si("conditions", "ap_mm")
    if not isinstance(ap, (int, float)):
        raise ValueError(f"{case.id}: no depth of cut")
    preds = []
    for entry in case.reference_values:
        name = str(entry.get("name"))
        unit = str(entry.get("unit") or "min")
        law = base
        note_extra = ""
        if "_TiAlN" in name or "_uncoated" in name:
            key = "tialn" if "_TiAlN" in name else "none"
            if key not in laws:
                laws[key] = _Law(db, case, col, coating_text=("TiAlN" if key == "tialn" else "uncoated"))
            law = laws[key]
        coolant = next((c for c in ("dry", "flood", "mql", "coldspray") if f"_{c}_" in name), "dry")
        if coolant == "coldspray":
            preds.append(no_prediction("cold-spray (cryogenic) jet: the database has no such fluid"))
            continue
        if coolant in FLUIDS:
            factor = _coolant_factor(db, FLUIDS[coolant], law, col)
            law = law.with_friction(factor)
            note_extra = f"; {coolant}: friction factor {factor:.2f} (cnctwin.coolant.lubrication), flank cooling not in the response"
        if entry.get("x") is not None and str(entry.get("x_name")) == "cutting_speed":
            speed = to_si(entry["x"], entry.get("x_unit"))
        else:
            speed = _speeds(case)[0]
        feed = _feed_from(name) or (float(feed_case) if isinstance(feed_case, (int, float)) else None)
        if feed is None:
            raise ValueError(f"{case.id}: no feed for {name}")
        vb_lim = _vb_from(name) or vb_crit
        if name.startswith("test_end_time"):
            limits = [x for x in (vb_crit, vn_crit / base.notch_factor if vn_crit else None) if x]
            vb_lim = min(limits)
            which = "VN" if vn_crit and vn_crit / base.notch_factor < (vb_crit or math.inf) else "VB"
            note_extra += f"; end by the first criterion ({which}: VB {vb_lim * 1e3:.3f} mm, notch factor {base.notch_factor:.2f})"
        if vb_lim is None:
            raise ValueError(f"{case.id}: no wear criterion for {name}")
        life = law.time_to(speed, feed, ap, vb_lim)
        if not math.isfinite(life):
            preds.append(no_prediction("the wear law gives no wear at this condition"))
            continue
        flank = law.flank_state(speed, feed, ap, 0.5 * vb_lim)
        preds.append(Prediction(from_si(life, unit), COMPARED,
                                f"vc {speed * 60:.0f} m/min, f {feed * 1e3:g} mm, ap {ap * 1e3:g} mm, VB {vb_lim * 1e3:g} "
                                f"mm; flank {flank[0] - 273.15:.0f} degC, sigma_n {flank[1] / 1e6:.0f} MPa at VB "
                                f"{0.5 * vb_lim * 1e3:g} mm{note_extra}",
                                extra={"flank_temp_degC": flank[0] - 273.15, "flank_sigma_MPa": flank[1] / 1e6}))
    curves = []
    for curve in case.reference_curves:
        name = str(curve.get("name"))
        m = re.search(r"vc([0-9]+)", name)
        if not m:
            raise ValueError(f"{case.id}: curve {name}: no speed in the name")
        speed = float(m.group(1)) / 60.0
        feed = float(feed_case) if isinstance(feed_case, (int, float)) else None
        y_name = str(curve.get("y_name"))
        factor = base.notch_factor if y_name.startswith("VN") else 1.0
        pts = curve.get("points") or []
        y_max = max(to_si(p[1], curve.get("y_unit")) for p in pts) / factor
        t_last = max(to_si(p[0], curve.get("x_unit")) for p in pts)
        vb_top = max(1.6 * y_max, 1e-5)
        t, vb = base.vb_curve(speed, feed, ap, vb_top)
        for _ in range(4):
            if t[-1] >= t_last:
                break
            vb_top *= 2.0
            t, vb = base.vb_curve(speed, feed, ap, vb_top)
        preds_c = []
        for x, y in pts:
            tx = to_si(x, curve.get("x_unit"))
            if y == 0 or tx <= 0.0:
                preds_c.append(not_compared("fresh edge (t = 0)", value=0.0))
                continue
            if tx > t[-1]:
                preds_c.append(no_prediction(f"time beyond the simulated range ({t[-1]:.0f} s)"))
                continue
            vb_t = float(np.interp(tx, t, vb)) * factor
            preds_c.append(Prediction(from_si(vb_t, curve.get("y_unit")), COMPARED,
                                      f"vc {speed * 60:.0f} m/min{', VN = notch factor x VB' if factor != 1.0 else ''}"))
        curves.append(preds_c)
    out = HarnessOutput(preds, curves, ["cnctwin.wear.build.wear_law", "cnctwin.wear.calibration.MechanicsResponse "
                                        "(cnctwin.mechanics Oxley + Smithey land + Blok partition)",
                                        "life integral with cnctwin.wear.kernels.usui_rate"],
                        base.settings_dict(), col.assumptions, col.provenance, col.notes, base.ambient)
    if len(laws) > 1:
        out.settings["coated_variant"] = laws.get("tialn").settings_dict() if "tialn" in laws else None
    return out


def _coolant_factor(db, fluid_id: str, law: _Law, col: H.Collector) -> float:
    from ...coolant import fluid as F
    from ...coolant import lubrication as L
    fluid = F.fluid_from_db(db.coolant(fluid_id))
    props = fluid.props(law.ambient)
    lub = L.friction(fluid, props, 1.0, law.side.tool_class, law.work.family, 2.5, None, None, 1e-3,
                     fluid.is_mql, fluid.dosage_m3_s or 0.0)
    col.assume(f"coolant fluid {fluid_id}", f"friction factor {lub.factor:.2f}", "the case's fluid is not in the "
               f"database: the database {fluid_id} entry, full reach", "db/coolants")
    return lub.factor


@discussion("taylor_life")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok" or r.stats is None:
            continue
        errs = [c.error_rel for c in r.comparisons if c.where == "value" and c.status == COMPARED and c.error_rel
                is not None and math.isfinite(c.error_rel)]
        if not errs:
            continue
        mean = sum(errs) / len(errs)
        law = (r.output.settings or {}).get("flank_law")
        src = (r.output.settings or {}).get("taylor_src")
        text = f"mean signed error of the lives {mean * 100:+.0f} %; flank law: {law}"
        if law == "taylor":
            text += f" (calibrated to {r.output.settings.get('taylor_entry')}, src {src or 'est'})"
        elif law == "abrasive_only":
            text += (" (no Usui constants for the pairing and no Taylor entry for this tool class in the material: "
                     "only the Archard floor, K_ref 2e-8 est, wears the tool)")
        elif law == "published":
            text += (" (published Usui constants fitted with the source's own FE temperatures; the mechanics "
                     "response's flank temperatures differ, and the life moves by exp(B dT / T^2))")
        curves = [c.error_rel for c in r.comparisons if c.where == "curve" and c.status == COMPARED and
                  c.error_rel is not None]
        if curves:
            text += (f"; VB(t) curves: mean signed error {sum(curves) / len(curves) * 100:+.0f} % (the model's land "
                     "grows slowly while it is small: the flank temperature rises with the land)")
        out[r.id] = text + "."
    out["summary"] = (H.budget_summary(q) + "Where the wear law is calibrated to a database Taylor "
                      "entry, the prediction can be no better than that entry for the case's tool and material (most "
                      "entries are estimates); where the same paper supplied the entry the case is "
                      "calibration-dependent and shows only that the integration reproduces its own fit. Ceramic, "
                      "cermet and PCBN pairings rest on estimated Taylor entries or published Usui sets from other "
                      "tests.")
    return out


__all__ = ["taylor_life"]
