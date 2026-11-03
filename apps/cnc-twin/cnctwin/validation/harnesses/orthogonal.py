"""Harnesses for chip_ratio and shear_angle: Oxley's orthogonal cutting theory at the case's conditions.

Per test of a case (uncut chip thickness h, cutting speed, rake, edge radius, width of cut):
``cnctwin.mechanics.oxley.solve`` at t1 = h - h_min (h_min = the material's minimum-chip-thickness
ratio x the edge radius, ``cnctwin.mechanics.contact.min_chip_thickness``: the layer below the
stagnation point is ploughed, not cut), work and tool body at ambient (a short orthogonal test). What
the experiments measured, and so what is compared:

* chip ratio: the measured chip thickness h_c of the whole cut, ratio h / h_c, so the simulated ratio is
  h / t2 with t2 Oxley's chip thickness; the chip-thickness references are listed (simulated t2) but
  not counted twice (the budget is defined on the ratio);
* shear angle: the sources compute phi from their measured chip ratio with tan phi = r cos a /
  (1 - r sin a); the simulated phi is the same formula on the simulated ratio of the same kind: h / t2
  for thickness-based references, the chip flow speed ratio V_c / V for flow-speed-based ones (equal to
  Oxley's own shear angle).

Below h_min the model forms no chip (no prediction).
"""

from __future__ import annotations

import math

from ...mechanics import contact as C
from ...mechanics import oxley
from ..cases import from_si
from ..registry import (COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction, not_compared)
from . import common as H

X_KEYS = ("h_mm", "feed_mm_rev", "uncut_chip_thickness_um", "vc_m_min", "cutting_speed_m_min")
MODEL_CHAIN = ["cnctwin.mechanics.materials.WorkMaterial/ToolSide (databases)",
               "cnctwin.mechanics.contact.min_chip_thickness", "cnctwin.mechanics.oxley.solve (Oxley 1989 with "
               "Johnson-Cook, Coulomb-Tresca interface, rake heat partition at the tool temperature)"]


class _Tests:
    """Solves each test of a case once."""

    def __init__(self, db, case, col: H.Collector):
        self.case = case
        self.col = col
        self.work, self.raw = H.work_material(db, case, col)
        self.tool, self.tool_mat, self.coating = H.tool_side(db, case, col)
        self.ambient = H.ambient(case, col)
        self.rows = H.test_rows(case)
        self.friction = self.work.friction_for(self.tool.tool_class, self.tool.coating)
        H.material_provenance(self.raw, col, friction_entry=self.friction)
        if not H.dry(case):
            col.note("The case is not dry; the dry friction of the database is used (no coolant model at this level).")
        self.cache: dict[int, dict] = {}

    def solve(self, row) -> dict:
        if row.index in self.cache:
            return self.cache[row.index]
        case, col = self.case, self.col
        h = H.uncut_thickness_m(row)
        v = H.speed_m_s(row)
        if h is None or v is None:
            raise ValueError(f"{case.id}: test {row.test_id} has no uncut chip thickness or speed")
        rake = H.rake_rad(case, row, col)
        r_e = H.edge_radius_m(case, row, col, "insert")
        width = H.width_m(case, row)
        h_min = C.min_chip_thickness(self.work, r_e)
        out = {"h_m": h, "speed_m_s": v, "rake_rad": rake, "edge_radius_m": r_e, "h_min_m": h_min, "width_m": width}
        if h <= h_min:
            out["status"] = "no chip"
        else:
            res = oxley.solve(self.work, h - h_min, v, rake, work_temp_K=self.ambient, tool=self.tool,
                              tool_temp_K=self.ambient, contact_width_m=width)
            out["status"] = res.status
            if res.status in ("oxley", "merchant"):
                out.update(t2_m=res.chip_thickness_m, phi_rad=res.shear_angle_rad, chip_speed_m_s=res.chip_speed_m_s,
                           T_int_K=res.T_interface_K, T_chip_K=res.T_chip_K, regime=res.regime)
        self.cache[row.index] = out
        return out

    def settings(self) -> dict:
        return {"tool_material": self.tool_mat, "coating": self.coating,
                "mu_sliding": self.friction.mu, "h_min_ratio": self.work.min_chip_thickness_ratio,
                "tool_temperature": "ambient (tool body at the start of a short test)",
                "work_temperature_K": self.ambient}


def _apparent_phi(r: float, rake: float) -> float:
    return math.atan2(r * math.cos(rake), 1.0 - r * math.sin(rake))


def _no_chip(state: dict) -> Prediction:
    if state["status"] == "no chip":
        return no_prediction(f"no chip: h = {state['h_m'] * 1e6:.1f} um <= h_min = {state['h_min_m'] * 1e6:.1f} um "
                             f"(edge radius {state['edge_radius_m'] * 1e6:.1f} um)")
    return no_prediction(f"Oxley/Merchant found no solution ({state['status']})")


def _note(state: dict) -> str:
    text = f"h {state['h_m'] * 1e3:.3f} mm, vc {state['speed_m_s'] * 60:.0f} m/min, r_e {state['edge_radius_m'] * 1e6:.0f} um"
    if state.get("status") == "merchant":
        text += ", Merchant fallback"
    return text


def _paired_uncertainty(entries: list[dict], index: int) -> float | None:
    """Uncertainty of a derived chip ratio from the uncertainty of the chip thickness it was derived
    from (same test label): u_r = r u_hc / h_c."""
    target = entries[index]
    lab = H.label(target)
    for e in entries:
        if e is target or e.get("uncertainty") is None or H.label(e) != lab:
            continue
        if str(e.get("unit")) in ("um", "mm", "m") and e.get("value"):
            return abs(float(target["value"])) * float(e["uncertainty"]) / float(e["value"])
    return None


@harness("chip_ratio", models=("cnctwin.mechanics.oxley", "cnctwin.mechanics.contact"),
         description="Oxley's orthogonal theory (cnctwin.mechanics.oxley.solve) per test at t1 = h - h_min, tool and "
                     "work at ambient; simulated ratio h / t2 against the measured h / h_chip.")
def chip_ratio(case, db) -> HarnessOutput:
    col = H.Collector()
    tests = _Tests(db, case, col)
    entries = case.reference_values
    preds = []
    for i, entry in enumerate(entries):
        row = H.find_test(tests.rows, entry, X_KEYS)
        if row is None:
            raise ValueError(f"{case.id}: no test matches reference {entry.get('name')!r}")
        state = tests.solve(row)
        unit = str(entry.get("unit") or "")
        is_ratio = str(entry.get("name", "")).startswith("chip_ratio")
        if "t2_m" not in state:
            preds.append(_no_chip(state) if is_ratio else not_compared("thickness listed with its ratio; " +
                                                                      _no_chip(state).note))
            continue
        if is_ratio:
            preds.append(Prediction(state["h_m"] / state["t2_m"], COMPARED, _note(state),
                                    uncertainty=_paired_uncertainty(entries, i),
                                    extra={"t2_um": state["t2_m"] * 1e6, "T_interface_K": state["T_int_K"]}))
        else:
            preds.append(not_compared("chip thickness: listed, the budget applies to the derived ratio",
                                      value=from_si(state["t2_m"], unit)))
    return HarnessOutput(preds, [[] for _ in case.reference_curves], MODEL_CHAIN, tests.settings(), col.assumptions,
                         col.provenance, col.notes, tests.ambient)


@harness("shear_angle", models=("cnctwin.mechanics.oxley", "cnctwin.mechanics.contact"),
         description="Oxley's orthogonal theory per test; the simulated shear angle is computed from the simulated "
                     "chip ratio of the same kind the source used (h / t2 or chip flow speed / vc) with tan phi = "
                     "r cos a / (1 - r sin a).")
def shear_angle(case, db) -> HarnessOutput:
    col = H.Collector()
    tests = _Tests(db, case, col)
    entries = case.reference_values
    flow_based = any(str(e.get("name", "")).startswith("v_chip") for e in entries)
    col.note("Shear angle compared as the source computed it: from the chip " +
             ("flow speed ratio V_chip / vc" if flow_based else "thickness ratio h / h_c") + ".")
    preds = []
    for entry in entries:
        row = H.find_test(tests.rows, entry, X_KEYS)
        if row is None:
            raise ValueError(f"{case.id}: no test matches reference {entry.get('name')!r}")
        state = tests.solve(row)
        name = str(entry.get("name", ""))
        unit = str(entry.get("unit") or "")
        if name.startswith("v_chip"):
            if "chip_speed_m_s" in state:
                preds.append(not_compared("chip flow speed: listed, the budget applies to the shear angle",
                                          value=from_si(state["chip_speed_m_s"], unit)))
            else:
                preds.append(not_compared("chip flow speed: " + _no_chip(state).note))
            continue
        if "t2_m" not in state:
            preds.append(_no_chip(state))
            continue
        r = state["chip_speed_m_s"] / state["speed_m_s"] if flow_based else state["h_m"] / state["t2_m"]
        phi = _apparent_phi(r, state["rake_rad"])
        preds.append(Prediction(from_si(phi, unit), COMPARED, _note(state) + f", r {r:.3f}",
                                extra={"oxley_phi_deg": math.degrees(state["phi_rad"]), "chip_ratio": r}))
    return HarnessOutput(preds, [[] for _ in case.reference_curves], MODEL_CHAIN, tests.settings(), col.assumptions,
                         col.provenance, col.notes, tests.ambient)


# ------------------------------------------------------------------------------------------ discussion
def _compared(result):
    return [c for c in result.comparisons if c.status == COMPARED and c.error_rel is not None]


def _range(values) -> str:
    return f"{min(values):.3g}-{max(values):.3g}" if values else "-"


def _case_text(result, what: str) -> str | None:
    comps = _compared(result)
    none = [c for c in result.comparisons if c.status == "no_prediction"]
    if not comps and not none:
        return None
    misses = [c for c in result.comparisons if c.counted and c.within_tolerance is False]
    if not misses:
        return None
    mean = sum(c.error_rel for c in comps) / len(comps) if comps else 0.0
    sim = [float(c.simulated) for c in comps]
    ref = [float(c.reference) for c in comps]
    settings = result.output.settings
    parts = [f"simulated {what} {_range(sim)} against measured {_range(ref)} (mean signed error {mean * 100:+.0f} %)",
             f"database sliding friction mu = {settings.get('mu_sliding'):.2g}, h_min ratio "
             f"{settings.get('h_min_ratio')}"]
    if none:
        parts.append(f"{len(none)} test(s) below the model's minimum chip thickness (h <= ratio x edge radius) although "
                     "the experiment formed a chip")
    return "; ".join(parts) + "."


SLIDING = ("With the Coulomb-Tresca interface in the sliding regime, Oxley's shear angle is set mainly by the friction "
           "angle (lambda = atan mu) and the strain-hardening term, so the simulated ratio barely changes with speed "
           "and feed while the measured one varies.")


@discussion("chip_ratio")
def _discuss_chip_ratio(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        text = _case_text(r, "ratio")
        if text:
            work = r.case.material_id
            if work == "ti6al4v":
                text += (" Ti6Al4V forms segmented chips (database tendency 'segmented'); Oxley's continuous-chip "
                         "solution over-predicts the thickness ratio of segmented chips (mechanics.md 13).")
            if work == "aisi1045":
                text += (" Ck45 has a built-up edge at 10-50 m/min (database BUE window) that thickens the chip; the "
                         "orthogonal solution has no BUE. The ETH data give an Albrecht friction coefficient of 0.78; "
                         "the database keeps the estimate 0.5.")
            out[r.id] = text + " " + SLIDING
    comps = [c for r in q.cases if r.status == "ok" and not r.calibration_dependent for c in _compared(r)]
    if comps:
        mean = sum(c.error_rel for c in comps) / len(comps)
        out["summary"] = (H.budget_summary(q) + f"Over the independent cases the simulated chip ratio is on average "
                          f"{mean * 100:+.0f} % off (positive: the simulated chip is thinner than measured). " + SLIDING + " The friction coefficients of the material files are "
                          "estimates for Ck45 and 304; BUE (Ck45 below 70 m/min) and the ploughed layer h_min (an "
                          "estimated ratio x the edge radius) decide the low-speed and small-h tests, where the model "
                          "forms no chip at h = 10 um although the experiment did. The two Ti6Al4V cases are "
                          "calibration-dependent (Johnson-Cook set chosen against the Ducobu experiments; friction from "
                          "the ETH tests) and are reported apart. No coefficient was changed for this report.")
    return out


@discussion("shear_angle")
def _discuss_shear(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        text = _case_text(r, "shear angle (deg)")
        if text:
            out[r.id] = text + " " + SLIDING
    comps = [c for r in q.cases if r.status == "ok" and not r.calibration_dependent for c in _compared(r)]
    if comps:
        mean = sum(c.error_rel for c in comps) / len(comps)
        out["summary"] = (H.budget_summary(q) + f"Mean signed error {mean * 100:+.0f} % over the independent values. The shear angle "
                          "inherits the chip-ratio behaviour (budgets.yaml: d phi / phi = 0.8-0.95 x the chip-ratio "
                          "error): almost constant in the model (set by the database friction, Ti6Al4V mu = 0.3 from the "
                          "ETH Albrecht analysis, 304 mu = 0.6), rising with the uncut thickness in the measurements.")
    return out


__all__ = ["chip_ratio", "shear_angle"]
