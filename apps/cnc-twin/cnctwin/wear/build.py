"""Building the wear state of one tool slot at init (docs/models/wear.md "Per tool").

* element geometry from the nominal EdgeModel: VB / VN = cot(a) - tan(g), the coating depth at the edge
  per unit VN cos(a)(1 - tan(a) tan(g)) (yen-2004-tool-wear-fem land geometry), wedge angle, kappa shares;
* the wear law: published Usui constants of the tool material or the work material when they match
  (converted explicitly to SI rates; with a Taylor entry for the pairing their A is anchored to it at the
  15-min speed, B as published: calibration.anchor_published), else Usui constants calibrated to the work material's Taylor
  life (calibration.py), else the abrasive (Archard) term alone; the rake uses its own published set,
  else the flank law (Usui et al. 1984: one characteristic equation for both faces);
* hardness tables (the database's, hot hardness when tabulated), the Basquin curve, the initial flaw;
* the chipping law (Weibull reference volume and strength, friction, contact and crack constants);
* critical sections with their effective volumes and instance strengths (median or a hash draw);
* the fallback response table (temperatures and stresses vs speed, used only when a thermal or
  cutting record is missing).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.abi import EDGE_CHISEL, EDGE_MARGIN, EDGE_WIPER, EdgeModel
from ..core.errors import InputError
from ..core.rng import hash_uniform, stream_key
from ..mechanics.materials import TOOL_PROCESS, ToolSide, coating_matches
from ..units import REFERENCE_TEMP_K
from . import abi as A
from . import calibration as CAL
from . import fracture as FR
from . import params as P
from .state import FALLBACK_ARRAYS, SECTION_ARRAYS, SlotWear

STRENGTH_STREAM = "wear.strength"
_CALIBRATIONS: dict = {}           # in-process cache of calibrations (key: inputs that determine the result)


def clear_cache() -> None:
    _CALIBRATIONS.clear()


def _pieces(settings) -> int:
    """physics.wear.calib_pieces (settings objects without it: the schema default)."""
    from .schema import WearSettings
    return int(getattr(settings, "calib_pieces", WearSettings.calib_pieces))


# ------------------------------------------------------------------------------------------ geometry
def element_geometry(edge: EdgeModel, edge_radius_m: float | None) -> A.ElemGeo:
    fl = P.get(P.FLANK)
    fr = P.get(P.FRACTURE)
    valid = np.asarray(edge.elem_valid, dtype=np.int8)
    nt, ne = valid.shape
    geo = A.new_geo(nt, ne)
    rake = np.asarray(edge.elem_rake_rad, dtype=float)
    clear = np.asarray(edge.elem_clear_rad, dtype=float)
    kappa = np.asarray(edge.elem_kappa_rad, dtype=float)
    a = np.maximum(clear, fl["min_clearance_rad"])
    g = 1.0 / np.tan(a) - np.tan(rake)
    g = np.where(g > 1e-3, g, 1.0 / np.tan(a))
    c_coat = np.cos(a) * (1.0 - np.tan(a) * np.tan(rake))
    c_coat = np.where(c_coat > 1e-3, c_coat, np.cos(a))
    beta = np.clip(0.5 * math.pi - rake - clear, fr["min_wedge_angle_rad"], fr["max_wedge_angle_rad"])
    geo.valid[:] = valid
    geo.kind[:] = np.asarray(edge.elem_kind, dtype=np.int64)
    geo.sin_k[:] = np.sin(kappa)
    geo.cos_k[:] = np.cos(kappa)
    geo.rake[:] = rake
    geo.clear[:] = clear
    geo.g_vb[:] = g
    geo.c_coat[:] = c_coat
    geo.beta[:] = beta
    geo.r_e0[:] = float(edge_radius_m) if edge_radius_m else np.asarray(edge.elem_edge_radius_m, dtype=float)
    geo.db[:] = np.asarray(edge.elem_db_m, dtype=float)
    return geo


def main_elements(edge: EdgeModel) -> np.ndarray:
    """Valid elements that form chips (no margins, chisels, wipers): the calibration's edge."""
    valid = np.asarray(edge.elem_valid) != 0
    kind = np.asarray(edge.elem_kind)
    main = valid & (kind != EDGE_MARGIN) & (kind != EDGE_CHISEL) & (kind != EDGE_WIPER)
    return main if main.any() else valid


def flank_geometry(edge: EdgeModel, geo: A.ElemGeo) -> CAL.FlankGeometry:
    sel = main_elements(edge)
    if not sel.any():
        raise InputError("wear: the tool has no valid edge element")
    return CAL.FlankGeometry(rake_rad=float(np.median(geo.rake[sel])), clearance_rad=float(np.median(geo.clear[sel])),
                             edge_radius_m=float(np.median(geo.r_e0[sel])),
                             incl_rad=float(np.median(np.asarray(edge.elem_incl_rad)[sel])),
                             g_vb=float(np.median(geo.g_vb[sel])), c_coat=float(np.median(geo.c_coat[sel])))


# ------------------------------------------------------------------------------------------ wear law
def _branches(entries: list, coating: P.CoatingData | None, tool_coating: str) -> tuple[np.ndarray, ...]:
    """(A, B, lo) arrays of published branches; a set for the tool's own coating is turned into the
    substrate law (A / coating factor) because the coating phase is modelled separately."""
    A_, B_, lo = [], [], []
    for e in entries:
        a_value = e.A_per_Pa
        if coating is not None and e.coating and coating_matches(e.coating, tool_coating) == 3:
            a_value = a_value / coating.factor
        A_.append(a_value)
        B_.append(e.B_K)
        lo.append(e.T_lo_K)
    if lo:
        lo[0] = -math.inf
    return np.array(A_, dtype=float), np.array(B_, dtype=float), np.array(lo, dtype=float)


def _taylor_condition(entry: P.TaylorEntry, fg: CAL.FlankGeometry, speed: float, vb: float,
                      coated: bool = True) -> CAL.Condition:
    cp = P.get(P.CALIBRATION)
    h = entry.feed_ref_m if entry.feed_ref_m else cp["default_feed_m"]
    depth = entry.ap_ref_m if entry.ap_ref_m else cp["default_depth_m"]
    return CAL.Condition(speed, h, depth, vb, fg.rake_rad, fg.clearance_rad, fg.edge_radius_m, fg.incl_rad, coated)


def taylor_check(taylor: P.TaylorEntry, response, slot: int, fg: CAL.FlankGeometry, abrasion: CAL.Abrasion,
                 vb0_m: float, nodes: int, pieces: int = 1) -> dict:
    """Edge contact lives the abrasive-only law predicts at the calibration and check speeds against the
    Taylor lives converted to contact time (reported, not fitted)."""
    out = {"speeds_m_s": [], "taylor_life_s": [], "contact_life_s": [], "model_life_s": [], "residual_rel": [],
           "life_basis": taylor.life_basis, "contact_fraction": taylor.contact_fraction}
    for speed in CAL.calibration_speeds(taylor):
        target = taylor.contact_life_s(speed, taylor.feed_ref_m, taylor.ap_ref_m)
        f = lambda v, vb: response.response(slot, _taylor_condition(taylor, fg, v, vb, False))
        ns = CAL._nodes(f, speed, vb0_m, taylor.VB_limit_m, fg, None, abrasion, nodes, pieces)
        life = CAL.life_s(ns, 0.0, 1.0, 1.0, fg.g_vb, None, 1.0)
        out["speeds_m_s"].append(speed)
        out["taylor_life_s"].append(taylor.life_s(speed, taylor.feed_ref_m, taylor.ap_ref_m))
        out["contact_life_s"].append(target)
        out["model_life_s"].append(life if math.isfinite(life) else None)
        out["residual_rel"].append(life / target - 1.0 if math.isfinite(life) else None)
    return out


def _anchored(ref: P.TaylorEntry, relation: str, fA: np.ndarray, fB: np.ndarray, flo: np.ndarray, report: dict,
              coating: P.CoatingData | None, tool_mat: P.ToolMaterialData, work: P.WorkData, abrasive: float,
              hs: tuple, fg: CAL.FlankGeometry, response, slot: int, vb0_m: float, settings,
              estimated: list) -> tuple[np.ndarray, P.TaylorEntry | None]:
    """The published flank set with its A anchored to the pairing's Taylor entry (calibration.anchor_published):
    (A array, the entry used) or the set unchanged and None when it cannot be anchored (noted in estimated)."""
    phase = coating if relation == "same" else (
        P.generic_coating(ref.coating, coating, tool_mat) if relation == "other" else None)
    abrasion = CAL.Abrasion(abrasive, hs[0], hs[1], *(phase.hardness_table() if phase is not None else (None, None)))
    key = ("anchor", work.id, tool_mat.id, tuple(report["flank"]["src"]), tuple(fA),
           tuple(fB), tuple(flo), ref, relation, round(fg.rake_rad, 9), round(fg.clearance_rad, 9),
           round(fg.edge_radius_m, 12), round(fg.incl_rad, 9), round(vb0_m, 12), int(settings.calib_nodes),
           _pieces(settings), getattr(response, "cache_key", id(response)), abrasive,
           None if phase is None else (phase.id, phase.factor, phase.thickness_m))
    try:
        anchor = _CALIBRATIONS.get(key)
        if anchor is None:
            coated = relation == "same"
            anchor = CAL.anchor_published(ref, lambda v, vb: response.response(
                slot, _taylor_condition(ref, fg, v, vb, coated)), fg, phase, abrasion, vb0_m, settings.calib_nodes,
                fA, fB, flo, _pieces(settings))
            _CALIBRATIONS[key] = anchor
    except (InputError, ValueError, LookupError, ZeroDivisionError) as error:
        estimated.append(f"tool slot {slot}: published Usui set used as published, not anchored to {work.id} "
                         f"tool_life[{ref.index}] ({error})")
        report["flank"]["anchor"] = {"error": str(error)}
        return fA, None
    report["flank"]["anchor"] = {**anchor.report, "relation": relation, "A_published_per_Pa": fA.tolist()}
    report["flank"]["A_per_Pa"] = (fA * anchor.factor).tolist()
    estimated.append(f"tool slot {slot}: published Usui set anchored to {work.id} tool_life[{ref.index}] at the "
                     f"15-min speed: A x {anchor.factor:.3g} for this model's flank temperatures (B as published)")
    return fA * anchor.factor, ref


def wear_law(*, slot: int, work: P.WorkData, tool_mat: P.ToolMaterialData, coating: P.CoatingData | None,
             coating_id: str, process: str, fg: CAL.FlankGeometry, response, vb0_m: float, settings,
             estimated: list, flow_stress: bool = True) -> tuple[A.Law, dict, P.TaylorEntry | None]:
    """Law arrays, a JSON report of the sources, and the Taylor entry used (None: not calibrated).

    Materials without a flow-stress law (composites, polymers, wood, foam) wear by the abrasive term with
    the database abrasiveness factor; their Taylor data are compared with it (``taylor_check``), not fitted:
    the default response's temperatures for them come from Merchant's geometry on a material that forms no
    chip and would not match the run's thermal temperatures."""
    fl = P.get(P.FLANK)
    family = work.family
    entries = list(tool_mat.usui) + list(work.usui)
    flank_pub = P.select_usui(entries, "flank", family, tool_mat.tool_class, coating_id)
    rake_pub = P.select_usui(entries, "rake", family, tool_mat.tool_class, coating_id)
    report: dict = {"family": family, "tool_class": tool_mat.tool_class}
    abrasive = fl["abrasive_coeff_ref"] * work.abrasive_factor
    abr_scale = 1.0
    taylor = None
    hs = tool_mat.hardness_table()
    hc = coating.hardness_table() if coating is not None else hs
    if flank_pub:
        fA, fB, flo = _branches(flank_pub, coating, coating_id)
        report["flank"] = {"source": "published", "src": [e.src for e in flank_pub],
                           "origin": [e.origin for e in flank_pub], "A_per_Pa": fA.tolist(), "B_K": fB.tolist()}
        ref, relation = P.select_taylor(work.taylor, tool_mat.tool_class, coating_id, process, tool_mat)
        if ref is not None and response is not None and flow_stress:
            fA, taylor = _anchored(ref, relation, fA, fB, flo, report, coating, tool_mat, work, abrasive, hs, fg,
                                   response, slot, vb0_m, settings, estimated)
    else:
        taylor, relation = P.select_taylor(work.taylor, tool_mat.tool_class, coating_id, process, tool_mat)
        fA = fB = flo = None
        if taylor is not None and not flow_stress:
            report["flank"] = {"source": "abrasive_only", "reason": "no flow-stress law: abrasive wear with the "
                               "database abrasiveness factor"}
            if response is not None:
                try:
                    report["flank"]["taylor_check"] = taylor_check(
                        taylor, response, slot, fg, CAL.Abrasion(abrasive, hs[0], hs[1]), vb0_m, settings.calib_nodes,
                        _pieces(settings))
                except (InputError, ValueError, LookupError, ZeroDivisionError) as error:
                    report["flank"]["taylor_check"] = {"error": str(error)}
            estimated.append(f"tool slot {slot}: {work.id} has no flow-stress law: abrasive (Archard) wear with "
                             f"abrasiveness {work.abrasive_factor:g}; Taylor data reported as a check")
            taylor = None
        elif taylor is not None and response is not None:
            phase = coating if relation == "same" else (
                P.generic_coating(taylor.coating, coating, tool_mat) if relation == "other" else None)
            abrasion = CAL.Abrasion(abrasive, hs[0], hs[1], *(phase.hardness_table() if phase is not None
                                                              else (None, None)))
            key = (work.id, tool_mat.id, coating_id, taylor, relation, round(fg.rake_rad, 9),
                   round(fg.clearance_rad, 9), round(fg.edge_radius_m, 12), round(fg.incl_rad, 9), round(vb0_m, 12),
                   int(settings.calib_nodes), _pieces(settings), getattr(response, "cache_key", id(response)),
                   abrasive, None if phase is None else (phase.id, phase.factor, phase.thickness_m))
            try:
                cal = _CALIBRATIONS.get(key)
                if cal is None:
                    coated = relation == "same"
                    cal = CAL.calibrate(taylor, lambda v, vb: response.response(
                        slot, _taylor_condition(taylor, fg, v, vb, coated)), fg, phase, abrasion, vb0_m,
                        settings.calib_nodes, _pieces(settings))
                    _CALIBRATIONS[key] = cal
                fA, fB, flo = np.array([cal.A_per_Pa]), np.array([cal.B_K]), np.array([-math.inf])
                abr_scale = cal.abrasive_scale
                report["flank"] = {"source": "taylor", "relation": relation, "A_per_Pa": [cal.A_per_Pa],
                                   "B_K": [cal.B_K], "coating_phase": None if phase is None else phase.id,
                                   "calibration": cal.report}
                if not cal.report.get("bracketed", True):
                    estimated.append(f"tool slot {slot}: Taylor calibration not bracketed (B = {cal.B_K:.4g} K at the "
                                     f"end of its range, residual at the upper speed {cal.report['residual_rel'][1]:+.1%}): "
                                     f"the response's flank temperature changes too little between the calibration "
                                     f"speeds for the entry's speed exponent")
                if relation == "other":
                    estimated.append(f"tool slot {slot}: Taylor entry for coating {taylor.coating!r} calibrated with a "
                                     f"generic coating (factor {phase.factor:g}, {phase.thickness_m * 1e6:.1f} um)")
                if taylor.engagement == "default":
                    estimated.append(f"tool slot {slot}: milling Taylor entry {taylor.index} of {work.id} states no "
                                     f"reference engagement: its cutting-time life is read at the ISO 8688-2 "
                                     f"reference arc ({math.degrees(2.0 * math.pi * taylor.contact_fraction):.1f} deg, "
                                     f"contact fraction {taylor.contact_fraction:.3f})")
            except (InputError, ValueError, LookupError, ZeroDivisionError) as error:
                estimated.append(f"tool slot {slot}: Taylor calibration impossible ({error}); abrasive wear only")
                report["flank"] = {"source": "abrasive_only", "reason": str(error)}
                taylor = None
        elif taylor is not None:
            estimated.append(f"tool slot {slot}: no response model for the Taylor calibration; abrasive wear only")
            report["flank"] = {"source": "abrasive_only", "reason": "no response model"}
            taylor = None
        else:
            estimated.append(f"tool slot {slot}: no Usui constants and no Taylor data for {tool_mat.tool_class} on "
                             f"{work.id}; abrasive (Archard) wear only")
            report["flank"] = {"source": "abrasive_only", "reason": "no Usui constants and no Taylor data"}
        if fA is None:
            fA, fB, flo = np.zeros(1), np.ones(1), np.array([-math.inf])
    if rake_pub:
        rA, rB, rlo = _branches(rake_pub, coating, coating_id)
        report["rake"] = {"source": "published", "src": [e.src for e in rake_pub], "A_per_Pa": rA.tolist(),
                          "B_K": rB.tolist()}
    else:
        rA, rB, rlo = fA.copy(), fB.copy(), flo.copy()
        report["rake"] = {"source": "flank law"}
    law = A.new_law(fA.size, rA.size, hs[0].size, hc[0].size)
    for target, values in ((law.fl_A, fA), (law.fl_B, fB), (law.fl_lo, flo), (law.rk_A, rA), (law.rk_B, rB),
                           (law.rk_lo, rlo), (law.hs_T, hs[0]), (law.hs_Pa, hs[1]), (law.hc_T, hc[0]),
                           (law.hc_Pa, hc[1])):
        target[:] = values
    s = law.scal
    s[A.LS_ABR] = abrasive * abr_scale
    s[A.LS_COAT_FACTOR] = coating.factor if coating is not None else 1.0
    s[A.LS_COAT_TMAX_K] = coating.max_service_K if coating is not None else 0.0
    s[A.LS_NOTCH_EXCESS] = (1.0 + work.hardening_frac) ** fl["notch_hardness_exponent"] - 1.0
    s[A.LS_ROUNDING] = fl["edge_rounding_ratio"]
    s[A.LS_BUE_PROTECT] = fl["bue_protection"]
    grain = tool_mat.grain_m if tool_mat.grain_m else fl["default_grain_size_m"]
    if not tool_mat.grain_m:
        estimated.append(f"tool slot {slot}: grain size {grain * 1e6:g} um assumed for BUE attrition")
    s[A.LS_PLUCK_M] = grain * fl["bue_pluck_fraction"]
    s[A.LS_SKIN_FACTOR] = work.skin_factor
    s[A.LS_DOC_RATIO] = fl["doc_h_ratio"]
    s[A.LS_MIN_CLEAR] = fl["min_clearance_rad"]
    s[A.LS_CRACK_GROWTH] = P.get(P.FRACTURE)["crack_growth_damage"]
    report["abrasive_coeff"] = float(s[A.LS_ABR])
    report["abrasive_scale"] = float(abr_scale)
    return law, report, taylor


def fatigue_constants(law: A.Law, tool_mat: P.ToolMaterialData, sigma0_Pa: float | None) -> dict:
    """Basquin constants and the initial flaw into the law; returns them for the report."""
    out = {"basquin_sigma_f_Pa": None, "basquin_b": None, "basquin_src": "", "thermal_sigma_f_Pa": None,
           "thermal_b": None, "initial_flaw_m": None}
    if tool_mat.trs_Pa:
        e20 = float(tool_mat.youngs(REFERENCE_TEMP_K))
        bq = FR.basquin(tool_mat.tool_class, tool_mat.trs_Pa, e20)
        th = FR.thermal_basquin(tool_mat.tool_class, tool_mat.trs_Pa, e20)
        law.scal[A.LS_FAT_SIGMA_PA] = th.sigma_f_Pa
        law.scal[A.LS_FAT_B] = th.b
        out.update(basquin_sigma_f_Pa=bq.sigma_f_Pa, basquin_b=bq.b, basquin_src=bq.src, thermal_sigma_f_Pa=th.sigma_f_Pa,
                   thermal_b=th.b)
    if tool_mat.kic_Pa_sqrt_m and sigma0_Pa:
        a0 = FR.initial_flaw_m(tool_mat.kic_Pa_sqrt_m, sigma0_Pa)
        law.scal[A.LS_CRACK_A0_M] = a0
        out["initial_flaw_m"] = a0
    return out


def chip_law(tool_mat: P.ToolMaterialData, mu: float, settings, estimated: list, slot: int) -> tuple[A.ChipLaw, dict]:
    fr = P.get(P.FRACTURE)
    fl = P.get(P.FLANK)
    m = tool_mat.weibull_m if tool_mat.weibull_m else fr["default_weibull_modulus"]
    if not tool_mat.weibull_m:
        estimated.append(f"tool slot {slot}: Weibull modulus {m:g} assumed (tool material gives none)")
    if m <= 2.0:
        raise InputError(f"tool material {tool_mat.id}: Weibull modulus {m:g} must exceed 2 for the chipping model "
                         "(the effective stressed volume of the edge wedge diverges)")
    chip = A.new_chip_law()
    c = chip.scal
    c[A.CL_M] = m
    if tool_mat.trs_Pa:
        sigma0 = FR.characteristic_strength_Pa(tool_mat.trs_Pa, m)
    else:
        sigma0 = math.inf
        estimated.append(f"tool slot {slot}: tool material {tool_mat.id} gives no TRS: chipping and breakage are "
                         "not evaluated")
    c[A.CL_SIGMA0_PA] = sigma0
    c[A.CL_VREF_M3] = FR.reference_volume_m3(m)
    c[A.CL_KIC] = tool_mat.kic_Pa_sqrt_m or 0.0
    if not tool_mat.kic_Pa_sqrt_m:
        estimated.append(f"tool slot {slot}: no fracture toughness: comb cracks do not lower the chipping strength")
    c[A.CL_Y] = fr["crack_shape_factor"]
    c[A.CL_LC_RATIO] = fr["entry_contact_ratio"]
    c[A.CL_CHIP_RATIO] = fr["chip_size_ratio"]
    c[A.CL_EXIT_FACTOR] = fr["exit_stress_factor"]
    c[A.CL_MU] = mu
    c[A.CL_COAT_SIGMA0_PA] = fr["coating_impact_strength_Pa"]
    c[A.CL_COAT_M] = fr["coating_weibull_modulus"]
    c[A.CL_BETA_MIN] = fr["min_wedge_angle_rad"]
    c[A.CL_BETA_MAX] = fr["max_wedge_angle_rad"]
    c[A.CL_NTHETA] = float(settings.chip_theta_nodes)
    c[A.CL_ROUNDING] = fl["edge_rounding_ratio"]
    # the edge's strength under repeated entries follows the tool material's Basquin curve (the b of the section
    # fatigue: hardmetal-type classes Nakajima et al. 2006, HSS the universal slopes)
    bq = FR.basquin(tool_mat.tool_class, tool_mat.trs_Pa or 1.0, float(tool_mat.youngs(REFERENCE_TEMP_K)))
    c[A.CL_FATIGUE_B] = bq.b
    return chip, {"weibull_m": m, "sigma0_Pa": sigma0 if math.isfinite(sigma0) else None,
                  "vref_m3": float(c[A.CL_VREF_M3]), "mu": mu, "edge_fatigue_b": bq.b, "edge_fatigue_src": bq.src}


# ------------------------------------------------------------------------------------------ sections
def sections(info, slot: int, edge: EdgeModel, chip: A.ChipLaw, seed: int, mode: str) -> tuple[dict, list]:
    secs = info.sections(slot)
    n = len(secs)
    out = {name: np.zeros(n) for name in SECTION_ARRAYS}
    names = []
    valid = np.asarray(edge.elem_valid) != 0
    z = np.asarray(edge.elem_z_m, dtype=float)
    lengths = np.asarray(edge.elem_len_m, dtype=float)
    z_load = float(np.sum(z[valid] * lengths[valid]) / np.sum(lengths[valid])) if valid.any() and \
        np.sum(lengths[valid]) > 0 else 0.0
    m = float(chip.scal[A.CL_M])
    sigma0 = float(chip.scal[A.CL_SIGMA0_PA])
    vref = float(chip.scal[A.CL_VREF_M3])
    sid = stream_key(STRENGTH_STREAM)
    for i, s in enumerate(secs):
        arm = max(s.z_m - z_load, s.d_eq_m)
        vb_ = FR.section_volume_m3(s.d_eq_m, arm, m, torsion=False)
        vt_ = FR.section_volume_m3(s.d_eq_m, arm, m, torsion=True)
        quantile = 0.5 if mode == "median" else float(hash_uniform(int(seed), sid, int(slot), i, 0))
        out["z_m"][i] = s.z_m
        out["Zb_m3"][i] = s.Z_bending_m3
        out["Zp_m3"][i] = s.Z_polar_m3
        out["d_m"][i] = s.d_eq_m
        out["volume_b_m3"][i] = vb_
        out["volume_t_m3"][i] = vt_
        if math.isfinite(sigma0):
            out["strength_b_Pa"][i] = FR.strength_Pa(vb_, sigma0, vref, m, quantile)
            out["strength_t_Pa"][i] = FR.strength_Pa(vt_, sigma0, vref, m, quantile)
        else:
            out["strength_b_Pa"][i] = math.inf
            out["strength_t_Pa"][i] = math.inf
        names.append(s.name)
    return out, names


# ------------------------------------------------------------------------------------------ fallback table
def fallback_table(response, slot: int, taylor: P.TaylorEntry | None, fg: CAL.FlankGeometry, settings) -> dict:
    cp = P.get(P.CALIBRATION)
    empty = {name: np.zeros(0) for name in FALLBACK_ARRAYS}
    if response is None:
        return empty
    if taylor is not None:
        v1, v2, _ = CAL.calibration_speeds(taylor)
        lo, hi = min(v1, v2) / cp["fallback_speed_span"], max(v1, v2) * cp["fallback_speed_span"]
        reference = taylor
    else:
        lo, hi = cp["table_speed_lo_m_s"], cp["table_speed_hi_m_s"]
        reference = P.TaylorEntry("", "", "", 1.0, 1.0, 0.0, 0.0, 1.0, cp["default_feed_m"], cp["default_depth_m"],
                                  1.0, None, "", -1)
    speeds = np.exp(np.linspace(math.log(lo), math.log(hi), int(settings.fallback_speeds)))
    rows = {name: [] for name in FALLBACK_ARRAYS}
    try:
        for v in speeds:
            r = response.response(slot, _taylor_condition(reference, fg, float(v), cp["reference_land_m"]))
            rows["speed_m_s"].append(float(v))
            rows["flank_T_K"].append(float(np.sum(np.asarray(r.weights) * np.asarray(r.flank_temp_K))))
            rows["rake_T_K"].append(float(np.sum(np.asarray(r.weights) * np.asarray(r.rake_temp_K))))
            rows["flank_sigma_Pa"].append(float(r.flank_sigma_Pa))
            rows["rake_sigma_Pa"].append(float(r.rake_sigma_Pa))
            rows["chip_ratio"].append(float(r.chip_speed_ratio))
    except (InputError, ValueError, LookupError, ZeroDivisionError):
        return empty
    return {name: np.ascontiguousarray(values, dtype=float) for name, values in rows.items()}


# ------------------------------------------------------------------------------------------ the slot
def tool_process(tool_type: str) -> str:
    return TOOL_PROCESS.get(tool_type, "milling")


def initial_state(geo: A.ElemGeo, vb0_m: float, coating: P.CoatingData | None) -> A.ElemState:
    nt, ne = geo.valid.shape
    st = A.new_state(nt, ne)
    valid = geo.valid != 0
    vn0 = np.where(valid, vb0_m / geo.g_vb, 0.0) if vb0_m > 0.0 else np.zeros((nt, ne))
    st.vn[:] = vn0
    if coating is not None:
        st.coat_f[:] = np.where(valid, np.maximum(coating.thickness_m - vn0 * geo.c_coat, 0.0), 0.0)
        st.coat_r[:] = np.where(valid, coating.thickness_m, 0.0)
    return st


def default_response(work_mech, tool_side: ToolSide | None, ambient_K: float, substrate: ToolSide | None = None):
    if work_mech is None:
        return None
    return CAL.MechanicsResponse(work_mech, tool_side, ambient_K, substrate=substrate)


def friction(work_mech, tool_class: str, coating_id: str, estimated: list, slot: int) -> float:
    if work_mech is not None:
        try:
            return float(work_mech.friction_for(tool_class, coating_id or "none").mu)
        except LookupError:
            pass
    mu = P.get(P.FLANK)["default_friction"]
    estimated.append(f"tool slot {slot}: no friction entry for {tool_class}: mu = {mu:g} for the entry force direction")
    return mu


def build_slot(*, slot, tool_spec, instance, tool_mat: P.ToolMaterialData, coating: P.CoatingData | None,
               coating_id: str, work: P.WorkData, work_mech, info, response, settings, seed: int,
               estimated: list) -> SlotWear:
    """The SlotWear of one tool slot (see the module doc)."""
    edge = info.edge_model(slot.slot)
    geo = element_geometry(edge, instance.initial_edge_radius_m)
    vb0 = float(instance.initial_VB_m or 0.0)
    process = tool_process(tool_spec.type)
    if not np.any(geo.valid != 0):
        estimated.append(f"tool slot {slot.slot}: no cutting-edge elements: usage counters only, no wear model")
        fg = CAL.FlankGeometry(0.0, P.get(P.FLANK)["min_clearance_rad"], 0.0, 0.0, 1.0, 1.0)
        response = None
    else:
        fg = flank_geometry(edge, geo)
    law, report, taylor = wear_law(slot=slot.slot, work=work, tool_mat=tool_mat, coating=coating,
                                   coating_id=coating_id, process=process, fg=fg, response=response, vb0_m=vb0,
                                   settings=settings, estimated=estimated,
                                   flow_stress=work_mech is not None and work_mech.jc is not None)
    mu = friction(work_mech, tool_mat.tool_class, coating_id, estimated, slot.slot)
    chip, chip_info = chip_law(tool_mat, mu, settings, estimated, slot.slot)
    fat = fatigue_constants(law, tool_mat, chip_info["sigma0_Pa"])
    secs, names = sections(info, slot.slot, edge, chip, seed, settings.strength)
    fallback = fallback_table(response, slot.slot, taylor, fg, settings)
    if fallback["speed_m_s"].size == 0:
        estimated.append(f"tool slot {slot.slot}: no fallback response table (temperatures default to the body "
                         "temperature when thermal gives none)")
    lf = P.get(P.LIFE)
    limits = {"vb": taylor.VB_limit_m if taylor is not None else lf["vb_limit_default_m"],
              "vb_max": lf["vb_max_limit_m"], "notch": lf["notch_limit_m"], "kt_base": lf["kt_limit_base_m"],
              "kt_feed_ratio": lf["kt_limit_feed_ratio"]}
    props = {"tool_material": tool_mat.id, "coating": coating.id if coating is not None else "none",
             "coating_thickness_m": coating.thickness_m if coating is not None else 0.0,
             "trs_Pa": tool_mat.trs_Pa, "kic_Pa_sqrt_m": tool_mat.kic_Pa_sqrt_m, "poisson": tool_mat.poisson,
             "density_kg_m3": tool_mat.density_kg_m3, "process": process, "tool_type": tool_spec.type,
             "limits": limits, **chip_info, **fat,
             "tables": {"E_T_K": tool_mat.E_T_K.tolist(), "E_Pa": tool_mat.E_Pa.tolist(),
                        "alpha_T_K": tool_mat.alpha_T_K.tolist(), "alpha_1_K": tool_mat.alpha_1_K.tolist(),
                        "k_T_K": tool_mat.k_T_K.tolist(), "k_W_mK": tool_mat.k_W_mK.tolist(),
                        "cp_T_K": tool_mat.cp_T_K.tolist(), "cp_J_kgK": tool_mat.cp_J_kgK.tolist()},
             "life_limit_s": instance.life_limit_s, "finish_radius_m": float(info.finish_radius_m(slot.slot)),
             "tool_radius_m": float(info.tool_radius_m(slot.slot)), "initial_VB_m": vb0,
             "max_service_K": tool_mat.max_service_K}
    st = initial_state(geo, vb0, coating)
    nt, ne = geo.valid.shape
    sw = SlotWear(slot=slot.slot, number=slot.number, tool_id=slot.tool_id, holder_id=slot.holder_id,
                  tool_class=tool_mat.tool_class, rotating=bool(info.rotating(slot.slot)), elem=st, geo=geo, law=law,
                  chip=chip, sections=secs, section_names=names, fallback=fallback,
                  finishes=np.zeros((nt, ne), np.int8), props=props, info={"law": report})
    used = float(instance.life_used_s or 0.0)
    sw.counters = {"in_cut_s": used, "carried_in_s": used, "cutting_distance_m": 0.0, "cut_path_m": 0.0, "uses": 0,
                   "parts": 0, "last_part": -1, "chip_events": 0, "chip_trials": 0, "spalls": 0, "bue_breakoffs": 0,
                   "op_events": 0, "max_flank_K": 0.0, "max_rake_K": 0.0, "max_section_stress_Pa": 0.0,
                   "max_failure_probability": 0.0, "active_s": 0.0, "above_service_s": 0.0}
    sw.life = {"expired": False, "cause": "", "expired_tick": -1, "expired_time_s": None, "broken": False,
               "broken_tick": -1, "broken_cause": "", "broken_section": "", "broken_z_m": 0.0, "rates": {},
               "duty": [0.0, 0.0], "values": {}, "limits": {}, "remaining_s": math.inf, "replacement_s": math.inf,
               "remaining_cause": "", "used_fraction": 0.0, "feed": [0.0, 0.0], "feed_m": 0.0, "Ra_m": 0.0,
               "Rt_m": 0.0, "drift_m": 0.0}
    sw.flags = {"coating_through": False, "rake_through": False, "crater_onset": False, "comb_cracks": False,
                "risk_warned_op": -1, "temperature_fallback": False, "stress_fallback": False, "chipped_any": False}
    sw.part_start = {"used_fraction": 0.0, "in_cut_s": used}
    return sw


__all__ = ["build_slot", "element_geometry", "flank_geometry", "main_elements", "wear_law", "chip_law", "sections",
           "fallback_table", "initial_state", "default_response", "friction", "tool_process", "fatigue_constants",
           "clear_cache", "STRENGTH_STREAM"]
