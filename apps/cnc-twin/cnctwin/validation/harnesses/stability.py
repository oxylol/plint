"""Harness for stability_lobe: the dynamics domain's stability limit at the case's speeds.

What is simulated (cnctwin.dynamics, the methods the domain uses for its lobe diagrams and operating points):

* the linearised cut: a straight-tooth cutter (``sld.classic_milling``) for the analytical benchmarks, else the
  cutter built by cnctwin.tooling (diameter, flutes, helix of the case) turned into the domain's CutModel by
  ``dynamics.operating.cut_model`` over the engaged peripheral elements at the depth (helix lag included),
  with the case's linear cutting coefficients (Kt and Kn = Kr: the experiment's own identified inputs, like its
  modal parameters);
* the relative tool-workpiece FRF of the case's modes (``sld.ModalFRF``: modal mass, stiffness or frequency,
  damping ratio per direction; complex-residue modes (Engin 1999) through their own FRF, ZOA only);
* the limit at each speed: zero-order (``sld.limit_at_speed``, ZOA) and semi-discretisation
  (``semidisc.limit_at_speed``, first order, m and points per mode period as the domain's settings); the
  simulator's limit is the smaller of the two, as the domain's operating-point verdict (ZOA margin and the SD
  check) uses both;
* measured test points: the tested depth bounds the limit (a stable test from below, a chatter test from
  above: ``Prediction.bound``), border tests and chatter frequencies are two-sided; the dynamics benchmark
  time-domain integration (``dynamics.drive.run_benchmark``, the domain's modal kernels, straight teeth) at the
  tested depth and speed is listed with each test (stable when the regenerative difference decays).

Process damping is not included: the cases state no indentation coefficient and the tests are at speeds where
it is small (listed as an assumption).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...dynamics import schema as DS
from ...dynamics import semidisc as SD
from ...dynamics import sld as SLD
from ...dynamics.drive import BenchModes, run_benchmark
from ..cases import from_si
from ..registry import COMPARED, HarnessOutput, Prediction, discussion, harness, no_prediction
from . import common as H

TWO_PI = 2.0 * math.pi
MODEL_CHAIN = ["cnctwin.tooling.build_tool (cutter edge, helix)", "cnctwin.dynamics.operating.cut_model / "
               "sld.classic_milling (linearised cut)", "cnctwin.dynamics.sld.ModalFRF (case modes)",
               "cnctwin.dynamics.sld.limit_at_speed (ZOA)", "cnctwin.dynamics.semidisc.limit_at_speed (SD)",
               "cnctwin.dynamics.drive.run_benchmark (time domain, test points)"]
TD_PERIODS = 240                  # tooth periods of a time-domain check (the growth rate uses the late half)
TD_STEPS = 96                     # steps per tooth period (physics.steps_per_tooth default)
# semi-discretisation cost guard: a monodromy of more intervals than this per tooth period (long delays against the
# mode period, where ZOA is accurate: no flip lobes at many vibration periods per tooth) is not built; the limit
# is then ZOA's (said per value)
SD_MAX_INTERVALS = 120
# analytical benchmarks computed by their source with a stated discretisation: the same m is used (no mode-period
# criterion), so the solver is compared with the published chart at equal resolution
BENCHMARK_SD_M = {"insperger-2004-updated-sd": 40}   # 'updated semi-discretisation, m = 40' (case file headers)
UNSTABLE = ("chatter", "light_chatter", "period_doubling")


# ================================================================================================ inputs
def _f(value, default=None):
    if isinstance(value, dict):
        value = value.get("value")
    return float(value) if isinstance(value, (int, float)) else default


def coefficients(case) -> tuple[float, float]:
    cc = (case.section("dynamics").get("cutting_coefficients") or {})
    Kt = _f(cc.get("Kt_N_mm2"), _f(cc.get("Ktc_N_mm2")))
    if Kt is None:
        raise ValueError(f"{case.id}: the case states no cutting coefficient")
    Kr = _f(cc.get("Kn_N_mm2"), _f(cc.get("Krc_N_mm2")))
    if Kr is None and _f(cc.get("Kr")) is not None:
        Kr = _f(cc.get("Kr")) * Kt
    if Kr is None:
        raise ValueError(f"{case.id}: no radial cutting coefficient")
    return Kt * 1e6, Kr * 1e6


@dataclass
class Modes:
    modes: list                   # (freq_Hz, zeta, mass_kg, (dx, dy))
    residue: list | None = None   # complex-residue modes: (freq_Hz, zeta, sigma, upsilon, axis)

    def frf(self):
        if self.residue:
            return ResidueFRF(self.residue)
        frfs = [SLD.ModalFRF.single(f, z, m, (d[0], d[1], 0.0)) for f, z, m, d in self.modes]
        return SLD.ModalFRF.combine(*frfs)


def modes(case) -> Modes:
    dyn = case.section("dynamics")
    out = []
    if dyn.get("natural_freq_Hz") is not None:
        f = _f(dyn.get("natural_freq_Hz"))
        z = _f(dyn.get("damping_ratio"))
        m = _f(dyn.get("modal_mass_kg"))
        if m is None and _f(dyn.get("stiffness_N_um")) is not None:
            m = _f(dyn.get("stiffness_N_um")) * 1e6 / (TWO_PI * f) ** 2
        out.append((f, z, m, (1.0, 0.0)))
        return Modes(out)
    if isinstance(dyn.get("modal_mass_kg"), list):
        masses = [float(v) for v in dyn["modal_mass_kg"]]
        ks = [float(v) for v in dyn["stiffness_N_per_m"]]
        zs = [float(v) for v in dyn["damping_ratio"]]
        for i, (m, k, z) in enumerate(zip(masses, ks, zs)):
            out.append((math.sqrt(k / m) / TWO_PI, z, m, (1.0, 0.0) if i == 0 else (0.0, 1.0)))
        return Modes(out)
    residue = []
    for axis, key in ((0, "modes_x"), (1, "modes_y")):
        for mode in dyn.get(key) or []:
            f = _f(mode.get("natural_freq_Hz"))
            z = _f(mode.get("damping_ratio"))
            if z is None and _f(mode.get("damping_pct")) is not None:
                z = _f(mode.get("damping_pct")) / 100.0
            if mode.get("residue_e6") is not None:
                sigma, upsilon = (float(v) * 1e-6 for v in mode["residue_e6"])
                residue.append((f, z, sigma, upsilon, axis))
                continue
            m = _f(mode.get("modal_mass_kg"))
            if m is None and _f(mode.get("stiffness_N_mm")) is not None:
                m = _f(mode.get("stiffness_N_mm")) * 1e3 / (TWO_PI * f) ** 2
            out.append((f, z, m, (1.0, 0.0) if axis == 0 else (0.0, 1.0)))
    return Modes(out, residue or None)


class ResidueFRF:
    """Tool-point FRF from complex residues sigma +- i upsilon per mode and direction (Engin 1999, Table 1):
    G(w) = (R1 + R2 i w) / (wn^2 - w^2 + 2 i zeta wn w), R1 = 2 (zeta wn sigma - wd upsilon), R2 = 2 sigma."""

    def __init__(self, residue: list):
        self.residue = residue
        self.omega_rad_s = np.array([TWO_PI * r[0] for r in residue])
        self.zeta = np.array([r[1] for r in residue])
        self.static = np.zeros((3, 3))

    def __call__(self, omega):
        w = np.atleast_1d(np.asarray(omega, float))
        G = np.zeros((len(w), 3, 3), complex)
        for f, z, sigma, upsilon, axis in self.residue:
            wn = TWO_PI * f
            wd = wn * math.sqrt(max(1.0 - z * z, 0.0))
            R1 = 2.0 * (z * wn * sigma - wd * upsilon)
            R2 = 2.0 * sigma
            G[:, axis, axis] += (R1 + R2 * 1j * w) / (wn * wn - w * w + 2j * z * wn * w)
        return G[0] if np.ndim(omega) == 0 else G

    def frequency_grid(self, n: int, lo_factor: float, hi_factor: float) -> np.ndarray:
        return SLD.ModalFRF.frequency_grid(self, n, lo_factor, hi_factor)


def immersion(case) -> tuple[float, float, str]:
    cond = case.section("conditions")
    direction = H._text(cond.get("direction")).lower()
    ratio = _f(cond.get("radial_immersion_ratio"))
    if ratio is None:
        ratio = case.si("conditions", "ae_mm") / case.si("tool", "diameter_mm")
    if direction.startswith("slot") or ratio >= 1.0:
        return 0.0, math.pi, "slot"
    down = not direction.startswith("up")
    lo, hi = SLD.immersion_angles(ratio, down)
    return lo, hi, "down" if down else "up"


# ================================================================================================ cut model
class Cut:
    """The linearised cut of a case: straight teeth (benchmarks) or the tooling-built helical cutter."""

    def __init__(self, case, col: H.Collector, Kt: float, Kr: float, phi_st: float, phi_ex: float):
        self.case = case
        self.col = col
        self.Kt, self.Kr = Kt, Kr
        self.phi_st, self.phi_ex = phi_st, phi_ex
        self.N = int(_f(case.value("tool", "flutes"), 2))
        self.helix = _f(case.value("tool", "helix_deg"), 0.0)
        self.geo = None
        if self.helix > 0.0:
            from . import cutsim as CSIM
            D = _f(case.value("tool", "diameter_mm"))
            g = {"diameter_mm": D, "flutes": self.N, "helix_deg": self.helix, "rake_deg": 0.0,
                 "clearance_deg": {"primary": 10.0}, "cutting_length_mm": max(2.0 * D, 10.0),
                 "overall_length_mm": 3.0 * D + 20.0, "shank": {"diameter_mm": D, "length_mm": D},
                 "corner_radius_mm": 0.0}
            item = CSIM.tool_item("square_endmill", "carbide", "carbide-k20-6co", "none", g, f"validation-{case.id}")
            self.geo = CSIM.build_geometry(item, edge_element_m=0.1e-3)
            col.note(f"Cut model: the {self.N}-flute cutter with a {self.helix:g} deg helix built by cnctwin.tooling "
                     "(dynamics.operating.cut_model over the peripheral elements below the depth); rake and clearance "
                     "do not enter the linearised cut.")
        self.cache: dict = {}

    def model(self, depth_m: float) -> SLD.CutModel:
        if self.geo is None:
            return SLD.classic_milling(self.N, self.Kt, self.Kr, self.phi_st, self.phi_ex)
        key = round(depth_m, 9)
        if key in self.cache:
            return self.cache[key]
        from types import SimpleNamespace

        from ...core.abi import EDGE_PERIPHERAL
        from ...dynamics.operating import cut_model
        edge = self.geo.edge
        engaged = (np.asarray(edge.elem_kind) == EDGE_PERIPHERAL) & (np.asarray(edge.elem_z_m) <= depth_m) & \
            (np.asarray(edge.elem_valid) != 0)
        if not engaged.any():
            engaged = (np.asarray(edge.elem_kind) == EDGE_PERIPHERAL) & (np.asarray(edge.elem_z_m) <=
                                                                         float(np.min(edge.elem_z_m[edge.elem_kind ==
                                                                                                    EDGE_PERIPHERAL])))
        ng = int(np.asarray(self.geo.groups.kind).shape[0])
        grid = np.array([1e-6, 1e-3])
        coeffs = SimpleNamespace(h_grid_m=grid, Kt_Pa=np.full((ng, 2), self.Kt), Kr_Pa=np.full((ng, 2), self.Kr),
                                 Ka_Pa=np.zeros((ng, 2)))
        arc = (0.5 * math.pi - self.phi_ex, 0.5 * math.pi - self.phi_st)
        z_top = float(np.max(np.asarray(edge.elem_z_m)[engaged] + 0.5 * np.asarray(edge.elem_len_m)[engaged]))
        cm = cut_model(edge, self.geo.frames, coeffs, engaged=engaged, ap_m=max(min(depth_m, z_top), 1e-6),
                       h_mean_m=1e-4, arc=arc, rotating=True, spin=-1.0, K_sp_scale=0.0, land_per_radius=0.0,
                       VB_m=None, work_radius_m=0.0)
        self.cache[key] = cm
        return cm


# ================================================================================================ limits
@dataclass
class Limit:
    speed_rpm: float
    zoa_m: float
    sd_m: float | None
    chatter_Hz: float
    kind: str

    @property
    def depth_m(self) -> float:
        return min(self.zoa_m, self.sd_m) if self.sd_m is not None else self.zoa_m


class Limits:
    def __init__(self, cut: Cut, frf, settings, real_modes: bool, a_max_m: float, benchmark_m: int | None = None):
        self.cut, self.frf, self.st = cut, frf, settings
        self.real = real_modes
        self.benchmark_m = benchmark_m          # the source's own discretisation (m intervals, no mode criterion)
        self.a_max = a_max_m
        self.omegas = frf.frequency_grid(settings.sld_n_freq, settings.sld_freq_lo_factor, settings.sld_freq_hi_factor)
        self.cache: dict = {}

    def at(self, rpm: float, depth_m: float | None = None) -> Limit:
        model_depth = depth_m if (depth_m is not None and self.cut.geo is not None) else 1e-3
        key = (round(rpm, 6), round(model_depth, 9) if self.cut.geo is not None else 0.0)
        if key in self.cache:
            return self.cache[key]
        cm = self.cut.model(model_depth)
        w = TWO_PI * rpm / 60.0
        zoa = SLD.limit_at_speed(cm, self.frf, w, self.omegas, method="zoa")
        sd = None
        kind = "hopf"
        st = self.st
        m, ppp = (self.benchmark_m, 0) if self.benchmark_m else (st.sd_m, st.sd_points_per_period)
        if self.real and SD.intervals(cm, self.frf, w, m, ppp) <= SD_MAX_INTERVALS:
            a, rho = SD.limit_at_speed(cm, self.frf, w, m=m, a_max=self.a_max, n_scan=st.sd_scan,
                                       tol_rel=st.sd_tol, points_per_period=ppp)
            sd = a
            if math.isfinite(a):
                mm = SD.intervals(cm, self.frf, w, m, ppp)
                mu = SD.multipliers(cm, self.frf, w, a, mm)[0]
                kind = SD.bifurcation_kind(complex(mu))
        lim = Limit(rpm, float(zoa.depth_limit_m), sd, float(zoa.chatter_freq_Hz), kind)
        self.cache[key] = lim
        return lim


def _label(name: str) -> str:
    return str(name).rsplit("_test", 1)[0] if str(name).endswith("_test") else str(name)


@harness("stability_lobe", models=("cnctwin.dynamics.sld", "cnctwin.dynamics.semidisc", "cnctwin.dynamics.drive"),
         description="The dynamics domain's stability limit at the case's speeds: ZOA (Altintas & Budak) and "
                     "first-order semi-discretisation (Insperger & Stepan) of the linearised cut (straight teeth for "
                     "the analytical benchmarks, the tooling-built helical cutter otherwise) with the case's modes and "
                     "linear cutting coefficients; the simulator's limit is the smaller of the two. Measured tests "
                     "bound the limit from one side (stable from below, chatter from above); the time-domain "
                     "benchmark of the domain's modal kernels at each tested depth and speed is listed.")
def stability_lobe(case, db) -> HarnessOutput:
    col = H.Collector()
    st = DS.settings(None)
    Kt, Kr = coefficients(case)
    phi_st, phi_ex, direction = immersion(case)
    md = modes(case)
    frf = md.frf()
    real = md.residue is None
    depths = [float(v["value"]) * 1e-3 for v in case.reference_values if v.get("unit") == "mm"]
    depths += [float(p[1]) * 1e-3 for c in case.reference_curves for p in c.get("points") or []]
    a_max = max(st.sd_a_max_m, 10.0 * max(depths)) if depths else st.sd_a_max_m
    cut = Cut(case, col, Kt, Kr, phi_st, phi_ex)
    benchmark_m = BENCHMARK_SD_M.get(case.src) if not case.measured else None
    if benchmark_m:
        col.note(f"Analytical benchmark: semi-discretisation with the source's m = {benchmark_m} intervals per tooth "
                 "period.")
    limits = Limits(cut, frf, st, real, a_max, benchmark_m)
    col.assume("process damping", "none", "the case states no indentation coefficient; the linear lobes", "est")
    if not real:
        col.note("Complex-residue modes: ZOA only (semi-discretisation and the time-domain benchmark need real "
                 "mass-normalised modes).")
    bench = BenchModes([(f, z, m, d) for f, z, m, d in md.modes]) if real else None
    fz = _f(case.si("conditions", "feed_mm_tooth"))
    if fz is None and _f(case.si("conditions", "feed_mm_rev")) is not None:
        fz = case.si("conditions", "feed_mm_rev") / cut.N
    td_agree = [0, 0]

    def td(rpm: float, depth_m: float) -> dict:
        if bench is None or fz is None:
            return {}
        res = run_benchmark(bench, n_teeth=cut.N, Kt_Pa=Kt, Kr_Pa=Kr, depth_m=depth_m, fz_m=fz, phi_st_rad=phi_st,
                            phi_ex_rad=phi_ex, speed_rad_s=TWO_PI * rpm / 60.0, steps_per_tooth=TD_STEPS,
                            periods=TD_PERIODS)
        return {"td_stable": bool(res.growth_per_period <= 0.0), "td_growth_per_period": float(res.growth_per_period)}

    values = []
    for v in case.reference_values:
        name = str(v.get("name"))
        unit = str(v.get("unit") or "")
        rpm = float(v.get("x"))
        if name == "chatter_frequency":
            lim = limits.at(rpm)
            if lim.chatter_Hz <= 0.0:
                values.append(no_prediction("no ZOA crossing at this speed"))
            else:
                values.append(Prediction(lim.chatter_Hz, COMPARED, f"ZOA chatter frequency at the limit, {rpm:g} rpm",
                                         extra={"limit_mm": from_si(lim.depth_m, "mm")}))
            continue
        depth = float(v["value"]) * 1e-3
        lim = limits.at(rpm, depth)
        label = _label(name)
        bound = "lower" if label == "stable" else ("upper" if label in UNSTABLE else "none")
        a = lim.depth_m
        extra = {"zoa_mm": from_si(lim.zoa_m, "mm") if math.isfinite(lim.zoa_m) else "inf",
                 "sd_mm": (from_si(lim.sd_m, "mm") if lim.sd_m is not None and math.isfinite(lim.sd_m) else
                           ("inf" if lim.sd_m is not None else None)),
                 "chatter_Hz": lim.chatter_Hz, "boundary": lim.kind,
                 "predicted": "stable" if depth < a else "chatter", "measured": label}
        extra.update(td(rpm, depth))
        if "td_stable" in extra and label in ("stable",) + UNSTABLE:
            td_agree[1] += 1
            td_agree[0] += int(extra["td_stable"] == (label == "stable"))
        note = (f"limit {('%.3g mm' % (a * 1e3)) if math.isfinite(a) else 'none up to %.3g mm' % (a_max * 1e3)} at "
                f"{rpm:g} rpm ({'min(ZOA, SD)' if lim.sd_m is not None else ('ZOA; SD not built, long delay' if real else 'ZOA')}"
                f", {lim.kind}); measured {label}")
        value = from_si(a if math.isfinite(a) else a_max, unit)
        values.append(Prediction(value, COMPARED, note, bound=bound, extra=extra))
    curves = []
    for c in case.reference_curves:
        unit = str(c.get("y_unit") or "")
        preds = []
        for x, y in c.get("points") or []:
            lim = limits.at(float(x), float(y) * 1e-3)
            a = lim.depth_m
            preds.append(Prediction(from_si(a if math.isfinite(a) else a_max, unit), COMPARED,
                                    f"{float(x):g} rpm: ZOA {lim.zoa_m * 1e3:.3g} mm, SD "
                                    f"{(lim.sd_m or math.inf) * 1e3:.3g} mm ({lim.kind})",
                                    extra={"zoa_mm": from_si(lim.zoa_m, "mm") if math.isfinite(lim.zoa_m) else "inf",
                                           "sd_mm": from_si(lim.sd_m, "mm") if lim.sd_m is not None and
                                           math.isfinite(lim.sd_m) else "inf"}))
        curves.append(preds)
    if td_agree[1]:
        col.note(f"Time-domain benchmark (straight teeth, linear, {TD_STEPS} steps per tooth, {TD_PERIODS} periods) "
                 f"agrees with the measured stable/chatter label in {td_agree[0]} of {td_agree[1]} tests.")
    settings = {"Kt_Pa": Kt, "Kr_Pa": Kr, "phi_st_deg": math.degrees(phi_st), "phi_ex_deg": math.degrees(phi_ex),
                "direction": direction, "flutes": cut.N, "helix_deg": cut.helix, "sd_m": st.sd_m,
                "sd_points_per_period": st.sd_points_per_period, "sld_n_freq": st.sld_n_freq,
                "modes": [[f, z, m] for f, z, m, _ in md.modes] if real else "complex residues",
                "td_agreement": td_agree}
    return HarnessOutput(values, curves, MODEL_CHAIN, settings, col.assumptions, col.provenance, col.notes, None)


@discussion("stability_lobe")
def _discuss(q) -> dict:
    out = {}
    for r in q.cases:
        if r.status != "ok":
            continue
        comps = [c for c in r.comparisons if c.counted]
        misses = [c for c in comps if c.within_tolerance is False]
        if not misses:
            continue
        low = [c for c in misses if c.error_rel is not None and c.error_rel < 0]
        high = [c for c in misses if c.error_rel is not None and c.error_rel > 0]
        text = f"{len(comps) - len(misses)}/{len(comps)} within tolerance"
        if low:
            text += f"; {len(low)} predicted limits below stable tests or the reference curve (conservative)"
        if high:
            text += f"; {len(high)} above chatter tests or the reference curve"
        flank = [c for c in misses if c.where == "curve"]
        if flank:
            text += ("; curve misses sit mostly on lobe flanks, where a 50-90 rpm shift of a steep boundary changes "
                     "the limit by tens of per cent (docs/models/dynamics.md 5.3)")
        td = r.output.settings.get("td_agreement")
        if td and td[1]:
            text += f"; time domain agrees with {td[0]}/{td[1]} measured labels"
        out[r.id] = text + ". Inputs (modes, coefficients) are the case's; damping and the helix (where est.) " \
            "set the lobe depth and the flip lobes."
    out["summary"] = (H.budget_summary(q) + "The analytical benchmarks test the solver against the published "
                      "semi-discretisation; the measured cases test the linear model with the experiment's own modal "
                      "and cutting-coefficient identification, where the border is resolved only to the depth step of "
                      "the test grid.")
    return out


__all__ = ["stability_lobe", "Modes", "modes", "coefficients", "ResidueFRF", "Cut", "Limits"]
