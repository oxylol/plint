"""Harness for wall_error: the surface error a deflecting end mill leaves on a milled wall.

Three kinds of case:

* **surface error profiles and maxima of peripheral milling** (Budak 1994): the cutting domain's force model at
  the case's engagement (``cutforce.MillSim``: the coefficient pipeline and cut_step over a steady straight cut,
  rigid tool) gives the force on every station of the edge at every step; the tool is a cantilever of the stated
  gauge length with tooling's fluted section (``ToolProps.fluted``: second moment of the flute cross-section) and
  the tool material's Young's modulus, on the stated collet stiffness (a translational spring at the clamp, the
  source's model); quasi-static (tooth frequencies far below the tool's first mode). The surface error at a height
  z is the deflection normal to the wall at the instant the edge element at z passes the surface-generating angle
  (phi = 0 up milling, pi down milling), overcut positive (Budak's sign: up milling pushes the cutter into the
  wall); the maximum normal force is the largest force normal to the wall over a revolution;
* **static tip stiffness of end mills** (Budak 1994, Sect. 3.3): k = k_c k_b / (k_c + k_b), k_b = 3 E I / l^3 with
  tooling's fluted section and the stated modulus, gauge length and collet stiffness;
* **surface location error** (Insperger 2006): the steady forced vibration of the case's tool modes under the
  case's linear cutting force, through the dynamics domain's time-domain benchmark (``dynamics.drive.
  run_benchmark``: the domain's modal kernels, straight teeth, the tooth leaving the cut when h < 0); SLE is the
  tool's displacement normal to the wall at the surface-generating instants of the last periods, positive when
  material is left (Insperger's definition). A speed where the model's cut is unstable gives no prediction.
"""

from __future__ import annotations

import math

import numpy as np

from ...dynamics.drive import BenchModes, run_benchmark
from ..cases import from_si, to_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import assumptions as AS
from . import common as H
from . import cutsim as CSIM
from .cutforce import MillCond, MillSim, MillTool, _strip_of
from .stability import coefficients, immersion, modes

TWO_PI = 2.0 * math.pi
SLE_PERIODS = 400               # tooth periods of the SLE time-domain run; the SLE is read over the last SLE_READ
SLE_READ = 40
CHAIN_PROFILE = ["cnctwin.cutting.coefficients.MechanicsModel + cnctwin.cutting.kernels.cut_step (station forces)",
                 "cnctwin.tooling (fluted section ToolProps.fluted)", "cantilever on the stated collet spring "
                 "(quasi-static deflection at the surface-generating instants)"]
CHAIN_SLE = ["cnctwin.dynamics.drive.run_benchmark (case modes, linear force with loss of contact)",
             "surface location error at the generating instants (Insperger 2006)"]


def _youngs(db, tool_mat: str, col: H.Collector) -> float:
    raw = H.raw_object(db, "tool_material", tool_mat)
    entry = raw.get("youngs_modulus_GPa") or raw.get("elastic_modulus_GPa") or raw.get("E_GPa")
    value = entry.get("value") if isinstance(entry, dict) else entry
    if not isinstance(value, (int, float)):
        raise ValueError(f"tool material {tool_mat} has no Young's modulus")
    src = (entry.get("src") if isinstance(entry, dict) else None) or raw.get("src") or ""
    col.used(f"tool material {tool_mat} youngs_modulus_GPa", str(src), str(entry.get("note") or "")
             if isinstance(entry, dict) else "")
    return float(value) * 1e9


def _section_I(geo) -> float:
    f = geo.props.fluted
    return 0.5 * (float(f.I_min_m4) + float(f.I_max_m4))


def cantilever(L_m: float, EI: float, k_clamp: float, z_obs: np.ndarray, z_load: np.ndarray) -> np.ndarray:
    """Compliance [n_obs, n_load] of a cantilever clamped at L from the tip (heights z from the tip) on a
    translational clamp spring: point load P at a = L - z_load from the clamp, deflection at x = L - z_obs."""
    x = np.clip(L_m - np.asarray(z_obs, float), 0.0, L_m)[:, None]
    a = np.clip(L_m - np.asarray(z_load, float), 0.0, L_m)[None, :]
    near = x <= a
    c = np.where(near, x * x * (3.0 * a - x), a * a * (3.0 * x - a)) / (6.0 * EI)
    return c + (1.0 / k_clamp if k_clamp > 0.0 else 0.0)


class Profile:
    """Station forces of one cut and the wall error they imprint."""

    def __init__(self, sim: MillSim, cond: MillCond, L_m: float, EI: float, k_clamp: float):
        self.sim, self.cond = sim, cond
        omega = -TWO_PI * cond.rpm / 60.0
        lkey = (round(cond.rpm, 6), round(cond.VB_m, 12), cond.coolant)
        lay = sim.layouts.get(lkey)
        if lay is None:
            lay = CSIM.layout(sim.pipe, sim.geo, omega, VB_m=cond.VB_m, key=f"m-{cond.coolant}")
            sim.layouts[lkey] = lay
        y_lo, y_hi = _strip_of(cond, sim.R)
        self.run = CSIM.mill(lay, omega, cond.fz_m, cond.ap_m, y_lo, y_hi, rho_kg_m3=float(sim.pipe.work.rho_kg_m3),
                             keep_stations=True)
        self.lay = lay
        self.omega = omega
        self.z_st = np.asarray(lay.tool.st_z_m, float)
        self.L, self.EI, self.k = L_m, EI, k_clamp
        self.phi_g = 0.0 if cond.direction == "up" else math.pi

    def error(self, z_m: float) -> float:
        """Wall error at height z (overcut positive), mean over the teeth's generating instants."""
        tool = self.lay.tool
        valid = np.nonzero(tool.valid != 0)[0]
        s = int(np.argmin(np.abs(self.z_st - z_m)))
        elems = [e for e in valid if int(tool.station[e]) == s]
        if not elems:
            return math.nan
        t = self.run.t_s
        F = self.run.station_force_T0_N[:, :, 1]                 # y (normal to the wall), on the tool
        comp = cantilever(self.L, self.EI, self.k, np.array([z_m]), self.z_st)[0]
        u = F @ comp                                               # deflection along +Y at height z, per step
        alpha_g = 0.5 * math.pi - self.phi_g
        out = []
        for e in elems:
            a0 = float(tool.alpha0_rad[e])
            # omega t + a0 = alpha_g + 2 pi k  ->  t = (alpha_g - a0 + 2 pi k) / omega, inside the record
            k_lo = math.ceil((self.omega * t[-1] + a0 - alpha_g) / TWO_PI)
            k_hi = math.floor((self.omega * t[0] + a0 - alpha_g) / TWO_PI)
            for k in range(int(k_lo), int(k_hi) + 1):
                tg = (alpha_g - a0 + TWO_PI * k) / self.omega
                if t[0] <= tg <= t[-1]:
                    out.append(float(np.interp(tg, t, u)))
        if not out:
            return math.nan
        uy = float(np.mean(out))
        return uy if self.phi_g == 0.0 else -uy

    def max_error(self, ap_m: float) -> float:
        zs = self.z_st[(self.z_st >= 0.0) & (self.z_st <= ap_m)]
        errs = [abs(self.error(float(z))) for z in zs]
        errs = [e for e in errs if math.isfinite(e)]
        return max(errs) if errs else math.nan

    def max_normal_force(self) -> float:
        return float(np.max(np.abs(self.run.force_T0_N[:, 1])))


def _tool(case, col, kind: str = "square_endmill") -> MillTool:
    D = case.value("tool", "diameter_mm")
    rake = case.value("tool", "rake_deg")
    if rake is None:
        rake = AS.ENDMILL_RAKE.value
        col.assume("radial rake", f"{rake:g} deg", AS.ENDMILL_RAKE.reason, "harnesses/assumptions.py")
    clear = case.value("tool", "clearance_deg")
    if clear is None:
        clear = AS.ENDMILL_CLEARANCE.value
        col.assume("clearance angle", f"{clear:g} deg", AS.ENDMILL_CLEARANCE.reason, "harnesses/assumptions.py")
    ap_raw = case.value("conditions", "ap_mm")
    ap = max(float(v) for v in ap_raw) if isinstance(ap_raw, (list, tuple)) else float(ap_raw or 0.0)
    L = case.value("tool", "gauge_length_mm")
    g = {"diameter_mm": D, "flutes": int(case.value("tool", "flutes")), "helix_deg": case.value("tool", "helix_deg"),
         "rake_deg": rake, "clearance_deg": {"primary": clear}, "cutting_length_mm": max(1.2 * float(ap), 2.0 * D),
         "overall_length_mm": float(L) + 40.0, "shank": {"diameter_mm": D, "length_mm": 40.0},
         "corner_radius_mm": 0.0, "edge_radius_um": H.edge_radius_m(case, None, col, "solid") * 1e6}
    tool_class = H._text(case.section("tool").get("tool_class")) or "carbide"
    return MillTool(kind, g, tool_class, float(L) * 1e-3)


def _feed(name: str) -> float:
    import re
    m = re.search(r"st([0-9.]+)", name)
    if not m:
        raise ValueError(f"no feed per tooth in {name!r}")
    return float(m.group(1).rstrip(".")) * 1e-3


def _profiles(case, db) -> HarnessOutput:
    col = H.Collector()
    mt = _tool(case, col)
    sim = MillSim(db, case, col, mt)
    E = _youngs(db, sim.pipe.tool_mat, col)
    I = _section_I(sim.geo)
    EI = E * I
    L = mt.stickout_m
    k_clamp = case.si("tool", "clamping_stiffness_N_mm")
    k_clamp = float(k_clamp) if isinstance(k_clamp, (int, float)) else 0.0
    d_eq = (64.0 * I / math.pi) ** 0.25
    D = float(mt.geometry["diameter_mm"]) * 1e-3
    col.note(f"Tool: cantilever of {L * 1e3:g} mm, E {E / 1e9:.0f} GPa, fluted-section I {I * 1e12:.0f} mm4 (equivalent "
             f"diameter {d_eq / D:.3f} D from tooling; the source's model uses 0.8 D), collet {k_clamp / 1e3:.0f} N/mm.")
    rpm = float(case.value("conditions", "spindle_rpm"))
    direction = "up" if H._text(case.section("conditions").get("direction")).lower().startswith("up") else "down"
    ap = case.si("conditions", "ap_mm")
    profiles: dict = {}

    def profile(fz: float, ae: float) -> Profile:
        key = (round(fz, 9), round(ae, 9))
        if key not in profiles:
            profiles[key] = Profile(sim, MillCond(rpm, fz, ap, ae, direction), L, EI, k_clamp)
        return profiles[key]

    curves = []
    ae_case = case.si("conditions", "ae_mm")
    for c in case.reference_curves:
        fz = _feed(str(c["name"]))
        unit = str(c.get("y_unit") or "")
        preds = []
        for x, _ in c.get("points") or []:
            z = to_si(float(x), str(c.get("x_unit") or "mm"))
            p = profile(fz, ae_case)
            e = p.error(z)
            if not math.isfinite(e):
                preds.append(no_prediction(f"no edge element generates the wall at z = {z * 1e3:g} mm"))
                continue
            preds.append(Prediction(from_si(e, unit), COMPARED, f"z {z * 1e3:g} mm, fz {fz * 1e3:g} mm, {direction}, "
                                    f"ae {ae_case * 1e3:g}, ap {ap * 1e3:g} mm",
                                    extra={"max_normal_force_N": p.max_normal_force()}))
        curves.append(preds)
    values = []
    for v in case.reference_values:
        name = str(v["name"])
        unit = str(v.get("unit") or "")
        fz = _feed(name)
        ae = to_si(float(v["x"]), str(v.get("x_unit") or "mm"))
        p = profile(fz, ae)
        if name.startswith("max_normal_force"):
            values.append(Prediction(from_si(p.max_normal_force(), unit), COMPARED,
                                     f"largest force normal to the wall over a revolution; fz {fz * 1e3:g}, ae "
                                     f"{ae * 1e3:g} mm"))
        else:
            e = p.max_error(ap)
            values.append(Prediction(from_si(e, unit), COMPARED, f"largest |error| over the engaged height; fz "
                                     f"{fz * 1e3:g}, ae {ae * 1e3:g} mm, {direction}"))
    settings = sim.pipe.settings_dict()
    settings.update({"E_Pa": E, "I_m4": I, "gauge_length_m": L, "clamp_stiffness_N_per_m": k_clamp,
                     "rpm": rpm, "direction": direction})
    col.assume("runout", "0", "not stated in the case: an ideal concentric tool", "est")
    return HarnessOutput(values, curves, CHAIN_PROFILE, settings, col.assumptions, col.provenance, col.notes, None)


def _static(case, db) -> HarnessOutput:
    col = H.Collector()
    D_list = case.value("tool", "diameter_mm_per_tool")
    L_list = case.value("tool", "gauge_length_mm_per_tool")
    E_list = case.value("tool", "youngs_modulus_GPa_per_tool")
    k_list = case.value("tool", "collet_stiffness_N_mm_per_tool")
    cls = case.value("tool", "tool_class_per_tool")
    values = []
    for v in case.reference_values:
        D = float(v["x"])
        i = int(np.argmin([abs(float(d) - D) for d in D_list]))
        g = {"diameter_mm": D, "flutes": int(case.value("tool", "flutes")), "helix_deg": case.value("tool", "helix_deg"),
             "rake_deg": AS.ENDMILL_RAKE.value, "clearance_deg": {"primary": AS.ENDMILL_CLEARANCE.value},
             "cutting_length_mm": 2.0 * D, "overall_length_mm": float(L_list[i]) + 40.0,
             "shank": {"diameter_mm": D, "length_mm": 40.0}, "corner_radius_mm": 0.0}
        tm = "hss-m2" if str(cls[i]).lower() == "hss" else "carbide-k20-6co"
        geo = CSIM.build_geometry(CSIM.tool_item("square_endmill", str(cls[i]), tm, "none", g, f"validation-{case.id}-{i}"))
        I = _section_I(geo)
        E = float(E_list[i]) * 1e9
        L = float(L_list[i]) * 1e-3
        kc = float(k_list[i]) * 1e3
        kb = 3.0 * E * I / L ** 3
        kT = kc * kb / (kc + kb)
        d_eq = (64.0 * I / math.pi) ** 0.25
        values.append(Prediction(from_si(kT, str(v.get("unit") or "N/mm")), COMPARED,
                                 f"tool {'AB'[i]}: kb {kb / 1e3:.0f} N/mm (fluted section from tooling, equivalent "
                                 f"diameter {d_eq / (D * 1e-3):.3f} D), collet {kc / 1e3:.0f} N/mm",
                                 extra={"kb_N_per_mm": kb / 1e3, "d_eq_over_D": d_eq / (D * 1e-3)}))
    col.note("The fluted section is tooling's (no core or gullet data in the case: the Kops & Vo type mean with the "
             "sector model's anisotropy); the stated moduli, gauge lengths and collet stiffnesses are used.")
    col.assume("tool cross-section", "tooling's fluted-section model", "the case gives no core diameter or flute shape",
               "cnctwin.tooling")
    return HarnessOutput(values, [], ["cnctwin.tooling (fluted section)", "cantilever on a collet spring "
                                      "(k = kc kb / (kc + kb), kb = 3 E I / l^3)"], {}, col.assumptions,
                         col.provenance, col.notes, None)


def _sle(case, db) -> HarnessOutput:
    col = H.Collector()
    Kt, Kr = coefficients(case)
    phi_st, phi_ex, direction = immersion(case)
    md = modes(case)
    bench = BenchModes([(f, z, m, d) for f, z, m, d in md.modes])
    N = int(case.value("tool", "flutes"))
    fz = case.si("conditions", "feed_mm_tooth")
    ap = case.si("conditions", "ap_mm")
    phi_g = phi_ex if direction == "down" else phi_st
    n_g = np.array([math.sin(phi_g), math.cos(phi_g)])
    curves = []
    for c in case.reference_curves:
        unit = str(c.get("y_unit") or "")
        preds = []
        for x, _ in c.get("points") or []:
            rpm = float(x)
            w = TWO_PI * rpm / 60.0
            res = run_benchmark(bench, n_teeth=N, Kt_Pa=Kt, Kr_Pa=Kr, depth_m=ap, fz_m=fz, phi_st_rad=phi_st,
                                phi_ex_rad=phi_ex, speed_rad_s=w, steps_per_tooth=96, periods=SLE_PERIODS,
                                linear=False, kick_m_s=0.0)
            if res.growth_per_period > 0.0:
                preds.append(no_prediction(f"{rpm:g} rpm: the model's cut is unstable at ap {ap * 1e3:g} mm (growth "
                                           f"{res.growth_per_period:.3g} per period): no steady surface"))
                continue
            t = res.t_s
            u = res.u_m
            tau = TWO_PI / (N * w)
            t_end = float(t[-1])
            sle = []
            from ...dynamics.drive import NODE_PHASE_STEPS
            dt = float(t[1] - t[0])
            phase0 = NODE_PHASE_STEPS * w * dt
            for k in range(SLE_READ):
                # tooth 0 at the generating angle: w t + phase0 = phi_g + 2 pi m
                m = math.floor((w * t_end + phase0 - phi_g) / TWO_PI) - k
                tg = (phi_g + TWO_PI * m - phase0) / w
                if tg < t[0]:
                    break
                ux = float(np.interp(tg, t, u[:, 0]))
                uy = float(np.interp(tg, t, u[:, 1]))
                sle.append(-(n_g[0] * ux + n_g[1] * uy))
            value = float(np.mean(sle)) if sle else math.nan
            if not math.isfinite(value):
                preds.append(no_prediction("no generating instant in the record"))
                continue
            preds.append(Prediction(from_si(value, unit), COMPARED, f"{rpm:g} rpm: steady SLE over {len(sle)} periods "
                                    f"(growth {res.growth_per_period:.3g}); tau {tau * 1e3:.3g} ms",
                                    extra={"growth_per_period": res.growth_per_period}))
        curves.append(preds)
    col.assume("cutter", "straight teeth", "the time-domain benchmark has no helix (the case's tool: 45 deg)", "est")
    settings = {"Kt_Pa": Kt, "Kr_Pa": Kr, "ap_m": ap, "fz_m": fz, "direction": direction, "periods": SLE_PERIODS,
                "read_periods": SLE_READ, "modes": [[f, z, m] for f, z, m, _ in md.modes]}
    return HarnessOutput([], curves, CHAIN_SLE, settings, col.assumptions, col.provenance, col.notes, None)


@harness("wall_error", models=("cnctwin.cutting.kernels", "cnctwin.tooling", "cnctwin.dynamics.drive"),
         description="Surface error of a milled wall: the cutting domain's station forces over a steady cut, imprinted "
                     "through a cantilever of the stated tool (tooling's fluted section, stated collet stiffness) at "
                     "the surface-generating instants (Budak), the static tip stiffness of that model, and the "
                     "surface location error of the steady forced vibration of the case's tool modes through the "
                     "dynamics domain's time-domain benchmark (Insperger).")
def wall_error(case, db) -> HarnessOutput:
    if "static-stiffness" in case.id or case.value("tool", "diameter_mm_per_tool") is not None:
        return _static(case, db)
    if case.section("dynamics").get("modal_mass_kg") is not None:
        return _sle(case, db)
    return _profiles(case, db)


@discussion("wall_error")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        comps = [c for c in r.comparisons if c.counted and c.error_rel is not None and math.isfinite(c.error_rel)]
        if not comps:
            continue
        mean = sum(c.error_rel for c in comps) / len(comps)
        inside = sum(1 for c in r.comparisons if c.counted and c.within_tolerance)
        n = sum(1 for c in r.comparisons if c.counted)
        text = f"{inside}/{n} within tolerance, mean signed error {mean * 100:+.0f} %"
        cal = r.output.settings.get("calibration")
        if cal is not None:
            text += (f"; forces from the coefficient pipeline ({'calibrated to ' + cal if cal != 'none' else 'mechanics '
                     'alone'}), so force errors (cutting_force) enter directly")
        none = [c for c in r.comparisons if c.status == "no_prediction"]
        if none:
            text += f"; {len(none)} speed(s) unstable in the model where the experiment measured a steady SLE"
        out[r.id] = text + "."
    out["summary"] = (H.budget_summary(q) + "The wall error is the product of the force normal to the wall at the "
                      "generating instant and the tool compliance: errors of the force model and of the tool section "
                      "(no core data in the cases) multiply; the SLE cases depend on the dynamic amplification near "
                      "the resonant speeds and on the stability of the cut.")
    return out


__all__ = ["wall_error", "cantilever", "Profile"]
