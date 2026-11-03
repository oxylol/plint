"""Chip form (docs/models/mechanics.md, "Chip form").

Per tooth pass (milling, drilling) or per continuous cut (turning): chip thickness, width and
length from the shear angle and the chip flow rule; up-curl radius (natural curl on a flat rake
face, the geometric curl of a groove/step breaker after Nakayama and Jawahir & van Luttervelt,
the flute of a milling cutter or drill); breaking by Nakayama's criterion (bending strain
t/(2R) against the remaining fracture strain of the chip) inside or outside the insert's
chip-breaker window; segmentation from the database critical speed or Recht's thermal
instability ratio; built-up edge windows and dynamics; polymers smearing above their smear
temperature, composites and powder-forming materials as dust; the ISO 3685 form class
(codes in cnctwin.core.bus.CHIP_FORMS); chip colour; ejection velocity and kinetic energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.bus import CHIP_FORMS
from ..core.rng import stream_key
from . import abi
from . import colour as COL
from . import oblique as O
from . import params as P
from .materials import ToolSide, WorkMaterial

BUE_STREAM = "mechanics.bue_breakoff"
NO_CHIP = 0                 # form code when no chip forms (ChipFormationContact.form_code default)


@dataclass(frozen=True)
class ChipState:
    """Mechanics state of the chip-forming zone for one pass (from a coefficient row or an
    orthogonal solution)."""

    shear_angle_rad: float
    chip_ratio: float                   # t1 / t2 in the normal plane
    contact_length_m: float
    T_chip_K: float
    T_interface_K: float
    T_shear_K: float
    shear_strain: float                 # total primary shear strain gamma = cos a / (sin phi cos(phi - a))
    shear_strain_rate_1_s: float
    tau_s_Pa: float = 0.0
    beta_to_work: float = 0.0
    c0: float = 0.0

    @classmethod
    def from_orthogonal(cls, res) -> "ChipState":
        v = res.vector
        return cls(float(v[abi.OX_PHI]), float(v[abi.OX_CHIP_RATIO]), float(v[abi.OX_LC]), float(v[abi.OX_T_CHIP]),
                   float(v[abi.OX_T_INT]), float(v[abi.OX_T_AB]), 2.0 * math.sqrt(3.0) * float(v[abi.OX_EPS_AB]),
                   math.sqrt(3.0) * float(v[abi.OX_EPSDOT_AB]), float(v[abi.OX_K_AB]), float(v[abi.OX_BETA]),
                   float(v[abi.OX_C0]))

    @classmethod
    def from_group(cls, gc, h_m: float) -> "ChipState":
        """Interpolate a GroupCoefficients row at h (linear in log h)."""
        from .kernels import interp_log_h

        def at(row):
            return float(interp_log_h(gc.h_grid_m, np.ascontiguousarray(row, dtype=np.float64), float(h_m)))

        return cls(at(gc.shear_angle_rad), at(gc.chip_ratio), at(gc.contact_length_m), at(gc.T_chip_K),
                   at(gc.T_interface_K), at(gc.T_shear_K), at(gc.shear_strain), at(gc.shear_strain_rate_1_s))


@dataclass(frozen=True)
class ChipConditions:
    """Geometry and kinematics of the pass."""

    h_m: float                          # mean uncut chip thickness
    width_m: float                      # chip width b
    speed_m_s: float
    rake_rad: float                     # normal rake
    incl_rad: float = 0.0
    cut_length_m: float | None = None   # uncut chip length per tooth pass (None: continuous cut)
    feed_m: float | None = None         # feed per revolution (turning, breaker window)
    depth_m: float | None = None        # depth of cut (turning, breaker window)
    nose_radius_m: float = 0.0          # engaged nose / corner radius
    process: str = "turning"            # turning | milling | drilling | tapping


@dataclass(frozen=True)
class ChipForm:
    thickness_m: float
    width_m: float
    length_m: float                     # length of one chip piece
    curl_radius_m: float
    curl_mode: str                      # natural | breaker | flute | none
    breaker_state: str                  # none | in_window | below_window | above_window
    turns: float
    broken: bool
    bend_strain: float
    fracture_strain: float
    form_code: int
    form_name: str
    segmented: bool
    segment_pitch_m: float
    recht_ratio: float
    bue_window: bool
    molten: bool
    dust_fraction: float
    chip_flow_angle_rad: float
    equivalent_rake_rad: float
    velocity_rel_tool_m_s: np.ndarray   # (t, r, a) frame of the edge element
    velocity_rel_work_m_s: np.ndarray
    kinetic_energy_J_per_kg: float      # relative to the work material it came from
    temp_K: float
    colour_rgb: tuple[int, int, int]
    colour_name: str


# ------------------------------------------------------------------------------ curl and breaking
def natural_curl_radius(contact_length_m: float) -> float:
    return P.number(P.CHIP, "natural_curl_ratio") * max(contact_length_m, 0.0)


def breaker_curl_radius(breaker, contact_length_m: float, natural_m: float) -> float:
    """Curl radius imposed by a chip breaker: a groove of width W and depth d makes the chip
    follow a circle through its edges and bottom, R = (W^2 + 4 d^2) / (8 d); a chip whose contact
    ends before the groove meets its back wall at distance W_land + W and height d,
    R = ((W_land + W - l_c)^2 + d^2) / (2 d) (Nakayama 1962). The tighter of breaker and
    natural curl wins."""
    width = breaker.groove_width_m
    depth = breaker.groove_depth_m
    if width <= 0.0 or depth <= 0.0:
        return natural_m
    if contact_length_m >= breaker.land_m:
        radius = (width * width + 4.0 * depth * depth) / (8.0 * depth)
    else:
        run = breaker.land_m + width - contact_length_m
        radius = (run * run + depth * depth) / (2.0 * depth)
    return min(radius, natural_m)


def breaker_window_state(breaker, cond: ChipConditions) -> str:
    if breaker is None:
        return "none"
    feed = cond.feed_m if cond.feed_m is not None else cond.h_m
    state = "in_window"
    if breaker.feed_window_m is not None:
        lo, hi = breaker.feed_window_m
        if feed < lo:
            return "below_window"
        if feed > hi:
            state = "above_window"
    if breaker.depth_window_m is not None and cond.depth_m is not None:
        lo, hi = breaker.depth_window_m
        if cond.depth_m < lo:
            return "below_window"
        if cond.depth_m > hi:
            state = "above_window"
    return state


def flute_curl_radius(tool: ToolSide | None) -> float | None:
    if tool is None:
        return None
    ratio = P.number(P.CHIP, "flute_curl_ratio")
    if tool.gullet_area_m2:
        return ratio * math.sqrt(tool.gullet_area_m2 / math.pi)
    if tool.diameter_m:
        return ratio * 0.25 * tool.diameter_m
    return None


def chip_fracture_strain(work: WorkMaterial) -> float:
    """Bending strain the formed chip withstands: a fraction of the bulk fracture strain,
    scaled by the material's breakability rating."""
    p = P.get(P.CHIP)
    bulk = work.fracture_strain if work.fracture_strain is not None else 0.0
    strain = max(float(p["break_strain_ratio"]) * bulk, float(p["min_break_strain"]))
    if work.breakability == "good":
        strain *= float(p["breakability_factor_good"])
    elif work.breakability == "poor":
        strain *= float(p["breakability_factor_poor"])
    return strain


# ------------------------------------------------------------------------------ segmentation
def recht_ratio(work: WorkMaterial, state: ChipState, t1_m: float, speed_m_s: float) -> float:
    """Recht's ratio (dtau/dgamma) / (-(dtau/dT) dT/dgamma) at the shear-plane state with the
    Johnson-Cook law. dT/dgamma = f tau / (rho c): the heat retained in the primary band of
    thickness Delta = l_AB / C0 during the shear time t_s = gamma_AB / gamma_dot,
    f = 1 / (1 + 2 sqrt(kappa t_s) / Delta) (thermal diffusion out of the band; f -> 1 is
    Recht's adiabatic limit). Below 1: catastrophic (localised) shear, segmented chip."""
    if work.jc is None or state.shear_angle_rad <= 0.0 or state.shear_strain_rate_1_s <= 0.0:
        return math.inf
    jc = work.jc
    a, b, n, c, m = jc[abi.MAT_A_PA], jc[abi.MAT_B_PA], jc[abi.MAT_N], jc[abi.MAT_C], jc[abi.MAT_M]
    eps = 0.5 * state.shear_strain / math.sqrt(3.0)
    epsdot = state.shear_strain_rate_1_s / math.sqrt(3.0)
    rate = 1.0 + c * math.log(max(epsdot / jc[abi.MAT_EPS0_1_S], 1.0))
    span = jc[abi.MAT_TMELT_K] - jc[abi.MAT_TREF_K]
    ts = min(max((state.T_shear_K - jc[abi.MAT_TREF_K]) / span, 1e-6), 0.99)
    therm = 1.0 - ts ** m
    dsig_deps = n * b * eps ** (n - 1.0) * rate * therm if eps > 0.0 else math.inf
    dtau_dgamma = dsig_deps / 3.0
    dtau_dT = -(a + b * eps ** n) * rate * m * ts ** (m - 1.0) / span / math.sqrt(3.0)
    tau = (a + b * eps ** n) * rate * therm / math.sqrt(3.0)
    rho_c = work.rho_kg_m3 * work.cp(state.T_shear_K)
    kappa = work.diffusivity(state.T_shear_K)
    c0 = state.c0 if state.c0 > 0.0 else P.number(P.CONTACT, "merchant_C0")
    band = t1_m / math.sin(state.shear_angle_rad) / c0
    t_shear = 0.5 * state.shear_strain / state.shear_strain_rate_1_s
    retained = 1.0 / (1.0 + 2.0 * math.sqrt(kappa * t_shear) / band)
    dT_dgamma = retained * tau / rho_c
    denom = -dtau_dT * dT_dgamma
    return dtau_dgamma / denom if denom > 0.0 else math.inf


def is_segmented(work: WorkMaterial, state: ChipState, cond: ChipConditions) -> tuple[bool, float]:
    """(segmented, Recht ratio). Database rule first: critical speed (and onset thickness) when
    given; materials whose tendency is 'segmented' without an onset segment always; otherwise
    (thermal_instability) Recht's criterion."""
    ratio = recht_ratio(work, state, cond.h_m, cond.speed_m_s)
    seg = work.segmentation
    if work.tendency in ("discontinuous", "powder", "molten"):
        return False, ratio
    if seg is not None and seg.criterion == "critical_speed":
        if seg.speed_onset_m_s is None:
            return work.tendency == "segmented", ratio
        above = cond.speed_m_s >= seg.speed_onset_m_s
        if seg.h_onset_m is not None:
            above = above and cond.h_m >= seg.h_onset_m
        return above, ratio
    if seg is not None and seg.temp_onset_K is not None:
        return state.T_shear_K >= seg.temp_onset_K, ratio
    if seg is None and work.tendency != "segmented":
        return False, ratio
    return ratio < P.number(P.CHIP, "recht_threshold"), ratio


# ------------------------------------------------------------------------------ BUE
def bue_in_window(work: WorkMaterial, speed_m_s: float, interface_temp_K: float) -> bool:
    window = work.bue
    if window is None:
        return False
    lo, hi = window.speed_m_s
    if not (lo <= speed_m_s <= hi):
        return False
    if window.temp_K is not None:
        tlo, thi = window.temp_K
        return tlo <= interface_temp_K <= thi
    return True


def bue_advance(height_m: np.ndarray, in_window: np.ndarray, uncut_m: np.ndarray, dt_s: float, seed: int,
                element_keys: np.ndarray, counters: np.ndarray, stream: str = BUE_STREAM) -> np.ndarray:
    """Advance BUE heights in place over dt (kernels.bue_step); counters are the owner's
    per-element draw counters (incremented). Returns the 0/1 break-off flags."""
    broke = np.zeros(height_m.shape[0], dtype=np.int64)
    from .kernels import bue_step
    bue_step(height_m, np.ascontiguousarray(in_window, dtype=np.float64), np.ascontiguousarray(uncut_m, np.float64),
             float(dt_s), P.bue_vector(), int(seed), int(stream_key(stream)), element_keys, counters, broke)
    return broke


# ------------------------------------------------------------------------------ form class
def classify(curl_radius_m: float, thickness_m: float, width_m: float, turns: float, broken: bool,
             flow_angle_rad: float, conical: bool, process: str, segmented_loose: bool = False) -> int:
    """ISO 3685 form code from the curl geometry: ribbon (straight), arc (< 1 turn), spiral
    (planar coil), helical: washer-type when the helix pitch 2 pi R tan(eta_c) does not exceed
    the chip width (coils side by side), tubular otherwise, conical when the chip radius varies
    across the width; subclass long/short/snarled by the number of turns."""
    p = P.get(P.CHIP)
    long_turns = float(p["max_turns_short"])
    snarl = float(p["min_turns_snarled"])
    side = abs(flow_angle_rad) >= float(p["min_side_curl_rad"])
    if not math.isfinite(curl_radius_m) or curl_radius_m >= float(p["ribbon_curl_ratio"]) * thickness_m:
        if broken:
            return 12
        return 13 if turns >= snarl else 11
    if turns < 1.0:
        return 62 if (broken or segmented_loose) else 61
    if not side:
        return 32 if conical else 31
    pitch = 2.0 * math.pi * curl_radius_m * math.tan(abs(flow_angle_rad))
    if conical or process == "drilling":
        return 52 if (broken or turns <= long_turns) else 51
    tight = pitch <= width_m
    base = 40 if tight else 20
    if broken or turns <= long_turns:
        return base + 2
    return base + (3 if turns >= snarl else 1)


def chip_form(work: WorkMaterial, tool: ToolSide | None, cond: ChipConditions, state: ChipState, *,
              h_W_m2K: float | None = None, ambient_K: float = 293.15) -> ChipForm:
    """Chip of one pass (see the module doc). ``h_W_m2K`` (film coefficient around the chip:
    coolant or air) enables the colour from contact time and cooling; without it the colour is
    taken from a hold at the chip temperature for the contact time only."""
    eta = O.stabler_flow_angle(cond.incl_rad)
    alpha_e = O.equivalent_rake(cond.rake_rad, cond.incl_rad, eta)
    forms_chip = state.chip_ratio > 0.0 and state.contact_length_m > 0.0 and cond.h_m > 0.0
    r_c = state.chip_ratio if forms_chip else 1.0
    thickness = cond.h_m / r_c
    width = float(O.chip_width(cond.width_m, cond.incl_rad, eta))
    v_ratio = float(O.chip_velocity_ratio(r_c, cond.incl_rad, eta))
    chip_speed = cond.speed_m_s * v_ratio
    direction = O.chip_flow_direction(cond.rake_rad, cond.incl_rad, eta)
    v_tool = chip_speed * direction
    v_work = v_tool + np.array([cond.speed_m_s, 0.0, 0.0])
    ke = 0.5 * float(v_work @ v_work)
    contact_time = state.contact_length_m / chip_speed if chip_speed > 0.0 else 0.0
    # colour
    if h_W_m2K is not None:
        rgb, cname, _ = COL.emitted_chip_colour(work, state.T_chip_K, contact_time, thickness, h_W_m2K, ambient_K)
    else:
        rgb, cname, _ = COL.chip_colour(work, COL.exposure_hold(state.T_chip_K, contact_time,
                                                                COL.activation_temperature(work)))
    segmented, ratio = is_segmented(work, state, cond)
    pitch = P.number(P.CHIP, "segment_pitch_ratio") * cond.h_m if segmented else 0.0
    bue = bue_in_window(work, cond.speed_m_s, state.T_interface_K)
    length_piece = math.inf if cond.cut_length_m is None else cond.cut_length_m * v_ratio
    smear = work.melt_smear_K if work.melt_smear_K is not None else work.glass_transition_K
    molten = (work.family == "polymer" or work.tendency == "molten") and smear is not None and \
        state.T_chip_K >= smear
    common = dict(thickness_m=thickness, width_m=width, segmented=segmented, segment_pitch_m=pitch,
                  recht_ratio=ratio, bue_window=bue, chip_flow_angle_rad=eta, equivalent_rake_rad=alpha_e,
                  velocity_rel_tool_m_s=v_tool, velocity_rel_work_m_s=v_work, kinetic_energy_J_per_kg=ke,
                  temp_K=state.T_chip_K, colour_rgb=rgb, colour_name=cname)
    if not forms_chip:      # ploughing below h_min or rubbing: no chip leaves the edge
        return ChipForm(length_m=0.0, curl_radius_m=math.inf, curl_mode="none", breaker_state="none", turns=0.0,
                        broken=False, bend_strain=0.0, fracture_strain=0.0, form_code=NO_CHIP,
                        form_name="none (ploughing, no chip)", molten=False, dust_fraction=0.0, **common)
    # materials that do not form coherent chips
    if work.family in ("composite", "foam") or (work.tendency == "powder" and work.family in ("wood", "composite",
                                                                                                "foam")):
        return ChipForm(length_m=thickness, curl_radius_m=math.inf, curl_mode="none", breaker_state="none", turns=0.0,
                        broken=True, bend_strain=0.0, fracture_strain=0.0, form_code=90, form_name=CHIP_FORMS[90],
                        molten=False, dust_fraction=max(work.dust_mass_fraction, 1.0 if work.family == "composite"
                                                        else work.dust_mass_fraction), **common)
    if molten:
        return ChipForm(length_m=length_piece if math.isfinite(length_piece) else width, curl_radius_m=math.inf,
                        curl_mode="none", breaker_state="none", turns=0.0, broken=False, bend_strain=0.0,
                        fracture_strain=0.0, form_code=95, form_name=CHIP_FORMS[95], molten=True,
                        dust_fraction=work.dust_mass_fraction, **common)
    if work.tendency in ("discontinuous", "powder"):
        piece = max(P.number(P.CHIP, "segment_pitch_ratio") * cond.h_m, thickness)
        code = 70
        return ChipForm(length_m=piece, curl_radius_m=math.inf, curl_mode="none", breaker_state="none", turns=0.0,
                        broken=True, bend_strain=0.0, fracture_strain=0.0, form_code=code, form_name=CHIP_FORMS[code],
                        molten=False, dust_fraction=work.dust_mass_fraction, **common)
    # curl
    natural = natural_curl_radius(state.contact_length_m)
    radius = natural
    mode = "natural"
    breaker = tool.chip_breaker if tool is not None else None
    window = breaker_window_state(breaker, cond)
    if breaker is not None and window in ("in_window", "above_window"):
        radius = breaker_curl_radius(breaker, state.contact_length_m, natural)
        mode = "breaker" if radius < natural else "natural"
    if cond.process in ("milling", "drilling", "tapping"):
        flute = flute_curl_radius(tool)
        if flute is not None and flute < radius:
            radius = flute
            mode = "flute"
    radius = max(radius, thickness)          # a chip cannot coil tighter than about its own thickness
    bend = thickness / (2.0 * radius)
    fracture = chip_fracture_strain(work)
    breaks = bend >= fracture
    conical = cond.nose_radius_m > P.number(P.CHIP, "nose_conical_ratio") * cond.width_m or cond.process == "drilling"
    side = abs(eta) >= P.number(P.CHIP, "min_side_curl_rad") or conical
    circumference = 2.0 * math.pi * radius
    if breaks:
        turns_break = P.number(P.CHIP, "turns_helical_break" if side else "turns_upcurl_break")
        piece = min(turns_break * circumference, length_piece)
    else:
        piece = length_piece
    turns = piece / circumference if math.isfinite(piece) else math.inf
    broken = breaks or math.isfinite(length_piece)
    code = classify(radius, thickness, width, turns, broken, eta, conical, cond.process, segmented)
    return ChipForm(length_m=piece, curl_radius_m=radius, curl_mode=mode, breaker_state=window, turns=turns,
                    broken=broken, bend_strain=bend, fracture_strain=fracture, form_code=code,
                    form_name=CHIP_FORMS[code], molten=False, dust_fraction=work.dust_mass_fraction, **common)


__all__ = ["ChipState", "ChipConditions", "ChipForm", "chip_form", "classify", "natural_curl_radius",
           "breaker_curl_radius", "breaker_window_state", "flute_curl_radius", "chip_fracture_strain", "recht_ratio",
           "is_segmented", "bue_in_window", "bue_advance", "BUE_STREAM", "NO_CHIP"]
