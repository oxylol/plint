"""One macro step of one tool slot (docs/models/wear.md "Macro step"): the bus inputs of the slot, the
kernel inputs with their fallbacks, BUE dynamics (cnctwin.mechanics.chipform.bue_advance), the wear,
chipping and output kernels, section breakage and fatigue, life criteria, the published ToolWear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.bus import ToolWear
from ..core.errors import TwinError
from ..core.findings import Category, Severity
from ..core.rng import stream_key
from ..mechanics.chipform import bue_advance
from . import abi as A
from . import fracture as FR
from . import kernels as K
from . import life as L
from . import params as P

CHIP_STREAM = "wear.chipping"
COAT_STREAM = "wear.coating"
BUE_STREAM = "wear.bue_breakoff"
_CHIP_KEY = stream_key(CHIP_STREAM)
_COAT_KEY = stream_key(COAT_STREAM)
ELEM_FIELDS = ("in_cut_time_s", "flank_sliding_m", "rake_sliding_m", "int_sigma_v_flank", "int_sigma_v_rake",
               "int_h_dt", "entries")


@dataclass
class Gathered:
    """The bus inputs of one tool slot for one step (cutting record summed over its contacts)."""

    contacts: list = field(default_factory=list)
    in_cut_s: float = 0.0
    elem: dict | None = None                 # ELEM_FIELDS -> [nt, ne]
    shocks: np.ndarray = field(default_factory=lambda: np.zeros((0, A.SHOCK_WIDTH)))
    section_moment_Nm: np.ndarray | None = None
    section_stress_Pa: np.ndarray | None = None
    torque_peak_Nm: float = 0.0
    force_peak_N: float = 0.0
    peak_row: int = -1                       # per-tick row of the largest force (-1: unknown)
    surface: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))   # (tooth, elem) of events
    friction_factor: float = 1.0

    @property
    def cutting(self) -> bool:
        return self.elem is not None and bool(np.any(self.elem["in_cut_time_s"] > 0.0))


def _array(value, shape, name, slot):
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    if arr.shape != shape:
        raise TwinError(f"wear: tool slot {slot}: cutting's {name} has shape {arr.shape}, the EdgeModel is {shape}")
    return arr


def gather(prev, slot: int, shape: tuple, nsec: int) -> Gathered:
    g = Gathered()
    for cut in sorted(prev.cutting.contacts, key=lambda c: c.contact):
        if int(cut.tool_slot) != slot:
            continue
        g.contacts.append(int(cut.contact))
        g.in_cut_s += float(cut.in_cut_s)
        arrays = {name: _array(getattr(cut, name), shape, name, slot) for name in ELEM_FIELDS}
        if arrays["in_cut_time_s"] is not None:
            if g.elem is None:
                g.elem = {name: np.zeros(shape) for name in ELEM_FIELDS}
            for name, arr in arrays.items():
                if arr is not None:
                    g.elem[name] += arr
        rows = np.asarray(cut.shocks, dtype=float).reshape(-1, A.SHOCK_WIDTH)
        if rows.size:
            g.shocks = np.concatenate([g.shocks, rows])
        for attr, target in (("section_moment_peak_Nm", "section_moment_Nm"),
                             ("section_stress_peak_Pa", "section_stress_Pa")):
            values = np.asarray(getattr(cut, attr), dtype=float).reshape(-1)
            if values.size == nsec and nsec > 0:
                old = getattr(g, target)
                setattr(g, target, values.copy() if old is None else np.maximum(old, values))
        g.torque_peak_Nm = max(g.torque_peak_Nm, abs(float(cut.torque_peak_Nm)))
        g.force_peak_N = max(g.force_peak_N, float(np.linalg.norm(np.asarray(cut.force_peak_M_N, dtype=float))))
        peaks = np.asarray(cut.wrench_tool_T0_peak, dtype=float).reshape(-1, 6)
        if peaks.shape[0]:
            g.peak_row = int(np.argmax(np.linalg.norm(peaks[:, :3], axis=1)))
        events = np.asarray(cut.surface_events, dtype=float).reshape(-1, 16)
        if events.size:
            own = events[events[:, 15].astype(np.int64) == slot]
            if own.size:
                g.surface = np.concatenate([g.surface, own[:, 1:3].astype(np.int64)])
    if g.shocks.shape[0] > 1:
        order = np.lexsort((g.shocks[:, A.SH_ELEM], g.shocks[:, A.SH_TOOTH], g.shocks[:, A.SH_TICK]))
        g.shocks = np.ascontiguousarray(g.shocks[order])
    return g


# ------------------------------------------------------------------------------------------ runtime
@dataclass
class Runtime:
    """What a slot needs beyond its state (rebuilt after a resume): geometry provider objects, the
    mechanics work material (BUE window), the skin query, the tolerances, element keys."""

    edge: object
    keys: np.ndarray                      # [nt * ne] int64 hash keys
    work_mech: object = None
    skin: object = None
    skin_active: bool = False
    tolerances: object = None
    h_min_ratio: float = 0.3
    last: ToolWear | None = None          # last published record (republished unchanged when idle)
    out: A.ElemOut | None = None
    z_load_m: float = 0.0                 # length-weighted mean height of the cutting edge (load point)
    cut: bool = False                     # the slot cut in the last step
    summary: dict | None = None           # extremes of the last published state (step.summary)


def _interp_log(speeds, table, v):
    v = np.maximum(v, 1e-12)
    return np.interp(np.log(v), np.log(speeds), table)


def kernel_input(sw, rt: Runtime, g: Gathered, heat, ambient_K: float, dt: float, pose_S_T, notes: list) -> A.ElemIn:
    nt, ne = sw.geo.valid.shape
    e = g.elem
    in_cut = e["in_cut_time_s"]
    speed = np.where(in_cut > 0.0, e["flank_sliding_m"] / np.where(in_cut > 0.0, in_cut, 1.0), 0.0)
    fb = sw.fallback
    flank_q = rake_q = weights = None
    if heat is not None:
        fq = np.asarray(heat.flank_temp_q_K, dtype=float)
        rq = np.asarray(heat.edge_temp_q_K, dtype=float)
        w = np.asarray(heat.q_weights, dtype=float)
        if fq.ndim == 3 and fq.shape[:2] == (nt, ne) and rq.shape == fq.shape and w.shape == (fq.shape[2],) \
                and fq.shape[2] > 0:
            flank_q, rake_q, weights = fq, rq, w
        elif fq.size:
            raise TwinError(f"wear: tool slot {sw.slot}: thermal's edge temperatures have shape {fq.shape}, "
                            f"the EdgeModel is {(nt, ne)}")
    body = None
    if heat is not None:
        b = np.asarray(heat.body_temp_at_elem_K, dtype=float)
        body = b if b.shape == (nt, ne) else np.full((nt, ne), float(heat.body_temp_K))
    if body is None:
        body = np.full((nt, ne), ambient_K)
    if flank_q is None:
        weights = np.ones(1)
        if fb["speed_m_s"].size:
            tf = np.where(in_cut > 0.0, _interp_log(fb["speed_m_s"], fb["flank_T_K"], speed), body)
            tr = np.where(in_cut > 0.0, _interp_log(fb["speed_m_s"], fb["rake_T_K"], speed), body)
            notes.append("temperature_fallback")
        else:
            tf = tr = body
            notes.append("temperature_body")
        flank_q, rake_q = tf[:, :, None], tr[:, :, None]
    inp = A.new_input(nt, ne, weights.shape[0])
    inp.in_cut_s[:] = in_cut
    sv_f = e["int_sigma_v_flank"]
    sv_r = e["int_sigma_v_rake"]
    if np.sum(sv_f) <= 0.0 and np.sum(e["flank_sliding_m"]) > 0.0 and fb["speed_m_s"].size:
        sv_f = e["flank_sliding_m"] * _interp_log(fb["speed_m_s"], fb["flank_sigma_Pa"], speed)
        notes.append("stress_fallback")
    if np.sum(sv_r) <= 0.0 and np.sum(e["flank_sliding_m"]) > 0.0 and fb["speed_m_s"].size:
        sliding = e["rake_sliding_m"] if np.sum(e["rake_sliding_m"]) > 0.0 else \
            e["flank_sliding_m"] * _interp_log(fb["speed_m_s"], fb["chip_ratio"], speed)
        sv_r = sliding * _interp_log(fb["speed_m_s"], fb["rake_sigma_Pa"], speed)
        notes.append("stress_fallback")
    if np.sum(sv_f) <= 0.0 and np.sum(e["flank_sliding_m"]) > 0.0 and not fb["speed_m_s"].size:
        notes.append("stress_missing")
    inp.sv_flank[:] = sv_f
    inp.sv_rake[:] = sv_r
    inp.h_int[:] = e["int_h_dt"]
    inp.entries[:] = e["entries"]
    if rt.skin_active and rt.skin is not None and pose_S_T is not None:
        inp.skin_frac[:] = np.clip(rt.skin.skin_fraction(sw.slot, rt.edge, sw.rotating, pose_S_T, in_cut), 0.0, 1.0)
    inp.flank_T[:] = flank_q
    inp.rake_T[:] = rake_q
    inp.w_q[:] = weights
    if heat is not None:
        amp = np.asarray(heat.edge_cycle_amp_K, dtype=float)
        if amp.shape == (nt, ne):
            inp.cycle_dT[:] = amp
    t = sw.props["tables"]
    E = np.interp(body, t["E_T_K"], t["E_Pa"])
    alpha = np.interp(body, t["alpha_T_K"], t["alpha_1_K"])
    nu = float(sw.props["poisson"])
    inp.therm_coef[:] = alpha * E / (2.0 * (1.0 - nu))
    k = np.interp(body, t["k_T_K"], t["k_W_mK"])
    cp = np.interp(body, t["cp_T_K"], t["cp_J_kgK"])
    inp.diffusivity[:] = k / (float(sw.props["density_kg_m3"]) * cp)
    inp.scal[A.IS_DT_S] = dt
    return inp


def bue_step(sw, rt: Runtime, g: Gathered, inp: A.ElemIn, seed: int) -> int:
    """Advance the BUE of the elements in cut by their in-cut time (mechanics' relaxation and break-off
    hazard); the coolant friction factor scales the BUE's saturation height. Returns break-offs."""
    window = getattr(rt.work_mech, "bue", None) if rt.work_mech is not None else None
    st = sw.elem
    in_cut = inp.in_cut_s
    cutting = (in_cut > 0.0) & (sw.geo.valid != 0)
    if window is None and not np.any(st.bue_m > 0.0):
        return 0
    inside = np.zeros(in_cut.shape, bool)
    if window is not None:
        speed = np.where(cutting, g.elem["flank_sliding_m"] / np.where(cutting, in_cut, 1.0), 0.0)
        lo, hi = window.speed_m_s
        inside = cutting & (speed >= lo) & (speed <= hi)
        if window.temp_K is not None and inside.any():
            rake_mean = np.einsum("ijq,q->ij", inp.rake_T, inp.w_q)
            inside &= (rake_mean >= window.temp_K[0]) & (rake_mean <= window.temp_K[1])
    candidates = np.flatnonzero((cutting & (inside | (st.bue_m > 0.0))).reshape(-1))
    if candidates.size == 0:
        return 0
    ne = in_cut.shape[1]
    heights = st.bue_m.reshape(-1)
    counters = st.n_bue.reshape(-1)
    flat_inside = inside.reshape(-1)
    broke_total = 0
    for flat in candidates:
        j, k = divmod(int(flat), ne)
        t_in = float(in_cut[j, k])
        uncut = float(inp.h_int[j, k]) / t_in * g.friction_factor
        flags = bue_advance(heights[flat:flat + 1], np.array([1.0 if flat_inside[flat] else 0.0]), np.array([uncut]),
                            t_in, int(seed), rt.keys[flat:flat + 1], counters[flat:flat + 1], BUE_STREAM)
        if flags[0]:
            inp.bue_broke[j, k] = 1.0
            broke_total += 1
    return broke_total


def chip_step(sw, inp: A.ElemIn, g: Gathered, seed: int) -> np.ndarray:
    rows = np.ascontiguousarray(g.shocks, dtype=np.float64)
    out = np.zeros((rows.shape[0], A.N_CO))
    if rows.shape[0] == 0:
        return out
    nt = sw.geo.valid.shape[0]
    b_eng = np.ascontiguousarray(np.sum(np.where(inp.in_cut_s > 0.0, sw.geo.db, 0.0), axis=1), dtype=np.float64)
    if b_eng.shape[0] != nt:
        b_eng = np.zeros(nt)
    keys = np.array([int(seed), _CHIP_KEY, _COAT_KEY, int(sw.slot)], dtype=np.int64)
    K.chip_trials(sw.geo, sw.elem, sw.chip, keys, rows, b_eng, out)
    return out


# ------------------------------------------------------------------------------------------ breakage
@dataclass
class SectionCheck:
    index: int = -1
    name: str = ""
    z_m: float = 0.0
    sigma_b_Pa: float = 0.0
    tau_Pa: float = 0.0
    sigma_eq_Pa: float = 0.0
    strength_Pa: float = math.inf
    probability: float = 0.0
    cause: str = ""
    ratio: float = 0.0
    fatigue: float = 0.0
    cutting_vm_Pa: float = 0.0
    packing: bool = False


def check_sections(sw, g: Gathered, packing_torque_Nm: float, packing_force_N: float, omega_abs: float,
                   z_load_m: float) -> tuple[SectionCheck | None, SectionCheck | None]:
    """(worst section by stress / strength, the section that fails this step or None). Updates the
    sections' fatigue sums."""
    secs = sw.sections
    n = secs["z_m"].shape[0]
    if n == 0 or not math.isfinite(float(sw.chip.scal[A.CL_SIGMA0_PA])):
        return None, None
    m = float(sw.chip.scal[A.CL_M])
    sigma0 = float(sw.chip.scal[A.CL_SIGMA0_PA])
    vref = float(sw.chip.scal[A.CL_VREF_M3])
    sf, b = sw.props.get("basquin_sigma_f_Pa"), sw.props.get("basquin_b")
    basquin = FR.Basquin(float(sf), float(b), "") if sf and b else None
    entries = g.elem["entries"] if g.elem is not None else None
    interrupted = entries is not None and np.any(entries > 0.0)
    if interrupted:
        cycles = float(np.sum(np.max(entries, axis=1)))
    elif sw.rotating and g.in_cut_s > 0.0:
        cycles = g.in_cut_s * omega_abs / (2.0 * math.pi)
    else:
        cycles = 0.0
    worst, failed = None, None
    for i in range(n):
        arm = max(float(secs["z_m"][i]) - z_load_m, 0.0)
        m_cut = float(g.section_moment_Nm[i]) if g.section_moment_Nm is not None else g.force_peak_N * arm
        m_pack = packing_force_N * arm
        moment = max(m_cut, m_pack)
        torque = max(g.torque_peak_Nm, packing_torque_Nm)
        sigma_b = moment / float(secs["Zb_m3"][i]) if secs["Zb_m3"][i] > 0 else 0.0
        tau = torque / float(secs["Zp_m3"][i]) if secs["Zp_m3"][i] > 0 else 0.0
        sigma_eq = FR.equivalent_stress_Pa(sigma_b, tau, sw.tool_class)
        torsion = 2.0 * tau > sigma_b
        volume = float(secs["volume_t_m3"][i] if torsion else secs["volume_b_m3"][i])
        strength = float(secs["strength_t_Pa"][i] if torsion else secs["strength_b_Pa"][i])
        prob = FR.failure_probability(sigma_eq, volume, sigma0, vref, m)
        packing = m_pack > m_cut or packing_torque_Nm > g.torque_peak_Nm
        cause = "packing" if packing else ("torsion" if torsion else "bending")
        if basquin is not None and cycles > 0.0 and sigma_eq > 0.0:
            amp = FR.swt_amplitude(sigma_eq, rotating_bending=not interrupted)
            secs["fatigue"][i] += cycles / basquin.cycles(amp)
        check = SectionCheck(i, sw.section_names[i] if i < len(sw.section_names) else str(i),
                             float(secs["z_m"][i]), sigma_b, tau, sigma_eq, strength, prob, cause,
                             sigma_eq / strength if strength > 0 else math.inf, float(secs["fatigue"][i]),
                             float(g.section_stress_Pa[i]) if g.section_stress_Pa is not None else 0.0, packing)
        if worst is None or check.ratio > worst.ratio:
            worst = check
        overload = sigma_eq >= strength
        fatigue = secs["fatigue"][i] >= 1.0
        if overload or fatigue:
            if not overload:
                check.cause = "fatigue"
            if failed is None or check.ratio > failed.ratio:
                failed = check
    return worst, failed


# ------------------------------------------------------------------------------------------ life
def criteria(sw, out: A.ElemOut, rt: Runtime, tolerances, feed_m: float, notes: list) -> tuple[dict, dict, dict]:
    """(values, limits, extra) of the life criteria (life.CRITERIA)."""
    st = sw.elem
    geo = sw.geo
    valid = geo.valid != 0
    used = (st.used != 0) & valid
    lim = sw.props["limits"]
    values, limits, extra = {}, {}, {}
    counts = used.sum(axis=1)
    sums = np.where(used, out.VB, 0.0).sum(axis=1)
    vb_b = float(np.max(np.where(counts > 0, sums / np.maximum(counts, 1), 0.0))) if counts.size else 0.0
    summ = rt.summary if rt.summary is not None else summary(sw, out)
    values["vb"] = vb_b
    limits["vb"] = float(lim["vb"])
    values["vb_max"] = summ["VB_max"]
    limits["vb_max"] = float(lim["vb_max"])
    values["crater"] = summ["KT_max"]
    limits["crater"] = float(lim["kt_base"]) + float(lim["kt_feed_ratio"]) * max(feed_m, 0.0)
    values["notch"] = summ["notch_max"]
    limits["notch"] = float(lim["notch"])
    finishing = (sw.finishes != 0) & valid
    applies = tolerances is not None and tolerances.applies(sw.slot, bool(finishing.any()))
    recession = st.vn + st.vn_notch + st.chip_d
    if applies and finishing.any():
        allowed = tolerances.surface_tolerance_m(sw.slot)
        values["dimension"] = float(np.max(recession[finishing]))
        limits["dimension"] = None if allowed is None else allowed * float(sw.props.get("dimensional_share", 1.0))
        ra_spec, rt_spec = tolerances.finish_spec(sw.slot)
        radius = float(sw.props["finish_radius_m"]) or float(sw.props["tool_radius_m"])
        edge_r = float(np.max(out.edge_r[finishing]))
        chip = float(np.max(st.chip_d[finishing]))
        ra, rt_ = L.roughness_m(feed_m, radius, rt.h_min_ratio * edge_r, chip)
        ra0, rt0 = L.roughness_m(feed_m, radius, rt.h_min_ratio * float(np.min(geo.r_e0[finishing])), 0.0)
        extra.update(Ra_m=ra, Rt_m=rt_)
        ratios, ratios0 = [], []
        if ra_spec:
            ratios.append(ra / ra_spec)
            ratios0.append(ra0 / ra_spec)
        if rt_spec:
            ratios.append(rt_ / rt_spec)
            ratios0.append(rt0 / rt_spec)
        if ratios and feed_m > 0.0:
            if max(ratios0) >= 1.0:
                notes.append("finish_unreachable")
                values["finish"], limits["finish"] = 0.0, None
            else:
                values["finish"], limits["finish"] = max(ratios), 1.0
        extra["drift_m"] = values["dimension"]
    values["fatigue"] = float(np.max(sw.sections["fatigue"])) if sw.sections["fatigue"].size else 0.0
    limits["fatigue"] = 1.0
    values["breakage"] = 1.0 if sw.life["broken"] else 0.0
    limits["breakage"] = 1.0
    return values, limits, extra


def summary(sw, out: A.ElemOut) -> dict:
    """Per-step extremes of one tool (computed once; used by the record, the criteria and the channels)."""
    st = sw.elem
    valid = sw.geo.valid != 0
    used = (st.used != 0) & valid
    if not valid.any():
        return {"VB_max": 0.0, "VB_tooth": np.zeros(valid.shape[0]), "KT_max": 0.0, "notch_max": 0.0,
                "edge_r_max": 0.0, "radial_max": 0.0, "axial_max": 0.0, "coat_min": float("nan"), "bue_max": 0.0,
                "chipped": 0, "D_max": 0.0}
    vb = np.where(valid, out.VB, 0.0)
    return {"VB_max": float(vb.max()), "VB_tooth": vb.max(axis=1), "KT_max": float(np.where(valid, st.kt, 0.0).max()),
            "notch_max": float(np.where(valid, out.VB_N, 0.0).max()),
            "edge_r_max": float(np.where(valid, out.edge_r, 0.0).max()),
            "radial_max": float(np.where(valid, out.radial, 0.0).max()),
            "axial_max": float(np.where(valid, out.axial, 0.0).max()),
            "coat_min": float(st.coat_f[used].min()) if used.any() else float("nan"),
            "bue_max": float(st.bue_m.max()), "chipped": int(np.count_nonzero(st.chipped)),
            "D_max": float(st.D_th.max())}


def publish(sw, out: A.ElemOut, summ: dict | None = None) -> ToolWear:
    st = sw.elem
    life = sw.life
    summ = summ if summ is not None else summary(sw, out)
    axial = summ["axial_max"]
    return ToolWear(
        VB_m=out.VB.copy(), KT_m=st.kt.copy(), notch_m=out.VB_N.copy(), edge_radius_m=out.edge_r.copy(),
        radial_wear_m=out.radial.copy(), axial_wear_m=out.axial.copy(), coating_remaining_m=st.coat_f.copy(),
        chipped=st.chipped.astype(bool), bue_height_m=st.bue_m.copy(), broken=bool(life["broken"]),
        life_used_fraction=float(life["used_fraction"]), rake_coating_remaining_m=st.coat_r.copy(),
        chip_depth_m=st.chip_d.copy(), thermal_damage=st.D_th.copy(), comb_crack_depth_m=st.crack_m.copy(),
        section_fatigue_damage=sw.sections["fatigue"].copy(),
        VB_max_m=summ["VB_max"], broken_tick=int(life["broken_tick"]),
        broken_section_z_m=float(life["broken_z_m"]),
        length_loss_m=float(life["broken_z_m"]) if life["broken"] else axial, expired=bool(life["expired"]),
        life_cause=str(life["cause"]), remaining_life_s=float(life["remaining_s"]),
        replacement_time_s=float(life["replacement_s"]), in_cut_s=float(sw.counters["in_cut_s"]),
        cutting_distance_m=float(sw.counters["cutting_distance_m"]), uses=int(sw.counters["uses"]),
        parts=int(sw.counters["parts"]))


def finding(ctx, state, category, severity, message, tick, path, quantities, parents):
    return ctx.finding(state, "wear", category, severity, message, int(tick), path=int(path),
                       quantities=quantities, parents=parents)


__all__ = ["Gathered", "gather", "Runtime", "kernel_input", "bue_step", "chip_step", "SectionCheck", "check_sections",
           "criteria", "summary", "publish", "finding", "CHIP_STREAM", "COAT_STREAM", "BUE_STREAM", "Category",
           "Severity"]
