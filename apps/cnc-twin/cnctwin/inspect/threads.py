"""Thread class decision: limits of ISO metric (ISO 965-1) and Unified inch (ASME B1.1) screw threads and the
GO / NOT GO gauge judgement of a measured thread (docs/models/inspect.md "Threads").

**ISO 965-1** (P and d in mm, tolerances in um; d is the geometric mean of the basic major diameter range):

* pitch diameter of external threads ``T_d2(6) = 90 P^0.4 d^0.1``, grades 3-9 by the factors 0.5 ... 2.0;
  internal ``T_D2(n) = k(n) T_d2(6)`` with k = 0.85, 1.06, 1.32, 1.7, 2.12 for grades 4-8;
* major diameter of external threads ``T_d(6) = 180 P^(2/3) - 3.15 P^(-1/2)`` (grades 4, 6, 8: 0.63, 1, 1.6);
  minor diameter of internal threads ``T_D1(6) = 433 P - 190 P^1.22`` (P <= 0.8) or ``230 P^0.7`` (grades 4-8:
  0.63, 0.8, 1, 1.25, 1.6);
* fundamental deviations: es = -(50 + 11 P) e, -(30 + 11 P) f, -(15 + 11 P) g, 0 h; EI = +(15 + 11 P) G, 0 H;
* every value is rounded to the nearest R40 number (ISO 3; ties to the larger) and then to a whole um (half to
  even): this reproduces the tables (es(g) = -28 / -32 um, T_d2(6) = 118 / 132 um, T_D2(6) = 160 / 170 um,
  T_d(6) = 212 / 236 um, T_D1(6) = 265 / 300 um for M8 x 1.25 / M10 x 1.5).

Limits: external ``d_max = d + es``, ``d_min = d_max - T_d``, ``d2_max = d2 + es``, ``d2_min = d2_max - T_d2``;
internal ``D_min = D + EI``, ``D2_min = D2 + EI``, ``D2_max = D2_min + T_D2``, ``D1_min = D1 + EI``,
``D1_max = D1_min + T_D1``, with the ISO 68-1 basic d2 = d - 0.75 H, d1 = d - 1.25 H, H = (sqrt 3 / 2) P.

**ASME B1.1** (inch): class 2A pitch diameter tolerance ``0.0015 D^(1/3) + 0.0015 LE^(1/2) + 0.015 P^(2/3)``
(LE = D for UNC, UNF, 4/6/8-UN; 9 P otherwise), class factors 1A 1.5, 3A 0.75, 1B 1.95, 2B 1.3, 3B 0.975; allowance
0.3 x the 2A tolerance for 1A and 2A; external major tolerance 0.060 P^(2/3) (2A, 3A) or 0.090 P^(2/3) (1A);
internal minor tolerance 0.25 P - 0.4 P^2 (1B, 2B); limits rounded to 0.0001 in (1B/2B minor: 0.001 in).

**Gauging** (ISO 1502): the GO gauge checks the virtual pitch diameter, the simple pitch diameter plus the
diameter equivalents of the cumulative pitch deviation over the length of engagement,
``f_P = |dP_sum| cot(alpha / 2)``, and of the flank angle deviations,
``f_alpha = sum 2 h |d alpha_i| / sin(alpha)`` with h the engaged flank height on the interfering side of the
pitch line (H/4 below it, 3H/8 above it: K = 2 and 3 in the ISO 1502 form ``0.073 P (K1 |da1| + K2 |da2|)``);
the NOT GO gauge checks the simple pitch diameter. External: GO accepts ``d2_v <= d2_max``, NOT GO accepts
``d2 >= d2_min``; internal: GO accepts ``D2_v >= D2_min``, NOT GO accepts ``D2 <= D2_max``. Major and minor
diameters are checked against their limits directly (plain gauges).
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from ..db.models import params as model_params
from ..units import INCH_M

MODEL = "inspect-threads"
ISO_DESIGNATION = re.compile(r"^\s*M\s*(\d+(?:\.\d+)?)(?:\s*[xX×]\s*(\d+(?:\.\d+)?))?(?:\s*-\s*([0-9a-hA-H]+))?"
                             r"\s*$")
UTS_DESIGNATION = re.compile(r"^\s*(#?\d+(?:/\d+)?(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(UNC|UNF|UNEF|UNS|UN|UNR|UNJ)?"
                             r"(?:\s*-\s*([123][AB]))?\s*$", re.IGNORECASE)
ISO_CLASS = re.compile(r"^(\d)([a-hA-H])(?:(\d)([a-hA-H]))?$")
UTS_CLASS = re.compile(r"^([123])([AB])$")


class ThreadSpecError(InputError):
    """A thread designation or class this module cannot evaluate."""


@functools.lru_cache(maxsize=1)
def _p() -> dict:
    return dict(model_params(MODEL))


def round_r40_um(value_um: float) -> float:
    """Nearest R40 number (ISO 3; ties to the larger), then to a whole um (half to even)."""
    if value_um == 0.0:
        return 0.0
    sign = -1.0 if value_um < 0.0 else 1.0
    x = abs(float(value_um))
    k = math.floor(math.log10(x))
    series = np.asarray(_p()["r40_series"], dtype=np.float64)
    cands = np.concatenate([series[-1:] * 10.0 ** (k - 1), series * 10.0 ** k, [10.0 ** (k + 1)]])
    diff = np.abs(cands - x)
    best = float(cands[np.flatnonzero(diff <= diff.min() * (1 + 1e-12) + 1e-12)[-1]])
    return sign * float(round(best))


@dataclass
class ThreadSpec:
    system: str                  # iso | uts
    internal: bool
    d_m: float                   # basic major diameter
    pitch_m: float
    pd_class: str                # pitch diameter class (6g, 6H, 2A ...)
    crest_class: str             # major (external) / minor (internal) diameter class
    series: str = ""             # UTS series (UNC, UNF ...)
    engagement_m: float | None = None

    @property
    def designation(self) -> str:
        if self.system == "iso":
            cls = self.pd_class if self.pd_class == self.crest_class else self.pd_class + self.crest_class
            return f"M{self.d_m * 1e3:g}x{self.pitch_m * 1e3:g}-{cls}"
        tpi = INCH_M / self.pitch_m
        return f"{self.d_m / INCH_M:g}-{tpi:g} {self.series or 'UN'}-{self.pd_class}"


@dataclass
class ThreadLimits:
    spec: ThreadSpec
    d2_min_m: float
    d2_max_m: float
    major_min_m: float | None
    major_max_m: float | None
    minor_min_m: float | None
    minor_max_m: float | None
    basic: dict = field(default_factory=dict)
    tolerances_m: dict = field(default_factory=dict)
    deviation_m: float = 0.0     # es (external) or EI (internal)
    notes: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"designation": self.spec.designation, "class": self.spec.pd_class if self.spec.pd_class ==
                self.spec.crest_class else self.spec.pd_class + self.spec.crest_class, "system": self.spec.system,
                "internal": self.spec.internal, "pitch_m": self.spec.pitch_m, "d_m": self.spec.d_m,
                "pitch_diameter_m": {"min": self.d2_min_m, "max": self.d2_max_m},
                "major_diameter_m": {"min": self.major_min_m, "max": self.major_max_m},
                "minor_diameter_m": {"min": self.minor_min_m, "max": self.minor_max_m},
                "fundamental_deviation_m": self.deviation_m, "tolerances_m": dict(self.tolerances_m),
                "basic_m": dict(self.basic), "notes": list(self.notes)}


# ============================================================================ parsing

def coarse_pitch_m(d_m: float) -> float:
    p = _p()
    ds = np.asarray(p["iso261_coarse_diameters_m"])
    k = int(np.argmin(np.abs(ds - d_m)))
    if abs(ds[k] - d_m) > 1e-9:
        raise ThreadSpecError(f"M{d_m * 1e3:g} is not in the ISO 261 coarse series: give the pitch (M{d_m * 1e3:g}xP)")
    return float(p["iso261_coarse_pitches_m"][k])


def _uts_size(text: str) -> float:
    text = text.strip()
    if text.startswith("#"):
        return (0.060 + 0.013 * int(text[1:])) * INCH_M
    if "/" in text:
        a, b = text.split("/")
        return float(a) / float(b) * INCH_M
    return float(text) * INCH_M


def parse(designation: str, internal: bool | None = None, d_m: float | None = None,
          pitch_m: float | None = None, engagement_m: float | None = None) -> ThreadSpec:
    """A thread spec from a designation (``M8x1.25-6g``, ``M10-6H``, ``1/4-20 UNC-2A``) or a bare class
    (``6H``, ``5g6g``, ``2B``) with the nominal diameter and pitch given; ``internal`` from the class letter case
    (ISO) or A/B (UTS) when not given."""
    text = str(designation).strip()
    m = ISO_DESIGNATION.match(text)
    if m:
        d = float(m.group(1)) * 1e-3
        pitch = float(m.group(2)) * 1e-3 if m.group(2) else coarse_pitch_m(d)
        cls = m.group(3) or ("6H" if internal else "6g")
        return _iso_spec(d, pitch, cls, internal, engagement_m)
    m = UTS_DESIGNATION.match(text)
    if m:
        d = _uts_size(m.group(1))
        pitch = INCH_M / float(m.group(2))
        cls = (m.group(4) or ("2B" if internal else "2A")).upper()
        return _uts_spec(d, pitch, cls, (m.group(3) or "UN").upper(), internal, engagement_m)
    if d_m is None or pitch_m is None:
        raise ThreadSpecError(f"{designation!r}: a bare thread class needs the nominal diameter and the pitch")
    if UTS_CLASS.match(text.upper()):
        return _uts_spec(float(d_m), float(pitch_m), text.upper(), "UN", internal, engagement_m)
    return _iso_spec(float(d_m), float(pitch_m), text, internal, engagement_m)


def _iso_spec(d, pitch, cls, internal, engagement):
    m = ISO_CLASS.match(cls)
    if not m:
        raise ThreadSpecError(f"{cls!r} is not an ISO 965-1 tolerance class (6g, 6H, 5g6g ...)")
    pd = m.group(1) + m.group(2)
    crest = (m.group(3) + m.group(4)) if m.group(3) else pd
    upper = pd[1].isupper()
    if crest[1].isupper() != upper:
        raise ThreadSpecError(f"{cls!r} mixes internal (upper case) and external (lower case) positions")
    if internal is not None and bool(internal) != upper:
        raise ThreadSpecError(f"{cls!r} is an {'internal' if upper else 'external'} class but the feature is "
                              f"{'internal' if internal else 'external'}")
    return ThreadSpec("iso", upper, d, pitch, pd, crest, "", engagement)


def _uts_spec(d, pitch, cls, series, internal, engagement):
    m = UTS_CLASS.match(cls)
    if not m:
        raise ThreadSpecError(f"{cls!r} is not an ASME B1.1 class (1A, 2A, 3A, 1B, 2B, 3B)")
    is_internal = m.group(2) == "B"
    if internal is not None and bool(internal) != is_internal:
        raise ThreadSpecError(f"{cls!r} is an {'internal' if is_internal else 'external'} class but the feature is "
                              f"{'internal' if internal else 'external'}")
    return ThreadSpec("uts", is_internal, d, pitch, cls, cls, series, engagement)


# ============================================================================ limits

def _grade_factor(grades_key: str, factors_key: str, grade: int, what: str) -> float:
    p = _p()
    grades = [int(g) for g in p[grades_key]]
    if grade not in grades:
        raise ThreadSpecError(f"ISO 965-1 has no {what} tolerance grade {grade} (grades {grades})")
    return float(p[factors_key][grades.index(grade)])


def _d_mean_mm(d_m: float) -> float:
    lim = np.asarray(_p()["iso_d_ranges_m"]) * 1e3
    d = d_m * 1e3
    if not lim[0] < d <= lim[-1]:
        raise ThreadSpecError(f"ISO 965-1 covers basic major diameters from {lim[0]:g} to {lim[-1]:g} mm, not {d:g}")
    k = int(np.searchsorted(lim, d, side="left")) - 1
    return math.sqrt(lim[k] * lim[k + 1])


def _iso_deviation_um(position: str, pitch_mm: float) -> float:
    p = _p()
    positions = list(p["iso_deviation_positions"])
    if position not in positions:
        raise ThreadSpecError(f"ISO 965-1 tolerance position {position!r} is not encoded ({', '.join(positions)})")
    a = float(p["iso_deviation_constants"][positions.index(position)])
    if position in ("h", "H"):
        return 0.0
    value = a + float(p["iso_deviation_pitch_coefficient"]) * pitch_mm
    return -round_r40_um(value) if position.islower() else round_r40_um(value)


TABLE_TD2, TABLE_TD2_INT, TABLE_TD, TABLE_TD1 = 0, 1, 2, 3


def _tabulated_um(quantity: int, d_m: float, pitch_m: float, grade: int, formula_um: float) -> tuple[float, bool]:
    """The ISO 965-1 table value where it differs from the rounded formula (db iso_table_values), else the
    rounded formula value; (um, tabulated)."""
    d = d_m * 1e3
    P = pitch_m * 1e3
    for row in _p().get("iso_table_values") or []:
        lo, hi, pitch, g, q, value = (float(v) for v in row)
        if int(q) == quantity and int(g) == grade and lo < d <= hi and abs(pitch - P) < 1e-9:
            return value, True
    return round_r40_um(formula_um), False


def iso_limits(spec: ThreadSpec) -> ThreadLimits:
    p = _p()
    P = spec.pitch_m * 1e3
    dmean = _d_mean_mm(spec.d_m)
    H = float(p["iso_h_per_pitch"]) * spec.pitch_m
    crest_ext = float(p["iso_external_crest_truncation"])
    crest_int = float(p["iso_internal_crest_truncation"])
    d2 = spec.d_m - 2.0 * (0.5 - crest_ext) * H            # d - 0.75 H
    d1 = spec.d_m - 2.0 * (1.0 - crest_ext - crest_int) * H   # d - 1.25 H
    td2_6 = float(p["iso_Td2_coefficient"]) * P ** float(p["iso_Td2_pitch_exponent"]) * \
        dmean ** float(p["iso_Td2_diameter_exponent"])
    pd_grade, pd_pos = int(spec.pd_class[0]), spec.pd_class[1]
    cr_grade, cr_pos = int(spec.crest_class[0]), spec.crest_class[1]
    notes = []
    if spec.internal:
        TD2, tab2 = _tabulated_um(TABLE_TD2_INT, spec.d_m, spec.pitch_m, pd_grade,
                                  _grade_factor("iso_TD2_grades", "iso_TD2_grade_factors", pd_grade, "D2") * td2_6)
        if P <= float(p["iso_TD1_fine_max_pitch_m"]) * 1e3:
            td1_6 = float(p["iso_TD1_fine_a"]) * P - float(p["iso_TD1_fine_b"]) * P ** float(p["iso_TD1_fine_exponent"])
        else:
            td1_6 = float(p["iso_TD1_coarse_coefficient"]) * P ** float(p["iso_TD1_coarse_exponent"])
        TD1, tab1 = _tabulated_um(TABLE_TD1, spec.d_m, spec.pitch_m, cr_grade,
                                  _grade_factor("iso_TD1_grades", "iso_TD1_grade_factors", cr_grade, "D1") * td1_6)
        EI = _iso_deviation_um(pd_pos, P)
        notes += ["T_D2 is the ISO 965-1 table value (the rounded formula differs)"] if tab2 else []
        notes += ["T_D1 is the ISO 965-1 table value (the rounded formula differs)"] if tab1 else []
        if cr_pos != pd_pos:
            notes.append(f"minor diameter position {cr_pos} differs from the pitch diameter's {pd_pos}: "
                         f"its own deviation is used")
        EI_c = _iso_deviation_um(cr_pos, P)
        return ThreadLimits(spec, d2 + EI * 1e-6, d2 + (EI + TD2) * 1e-6, spec.d_m + EI * 1e-6, None,
                            d1 + EI_c * 1e-6, d1 + (EI_c + TD1) * 1e-6, {"d2_m": d2, "d1_m": d1, "H_m": H},
                            {"T_D2_m": TD2 * 1e-6, "T_D1_m": TD1 * 1e-6}, EI * 1e-6, notes)
    Td2, tab2 = _tabulated_um(TABLE_TD2, spec.d_m, spec.pitch_m, pd_grade,
                              _grade_factor("iso_Td2_grades", "iso_Td2_grade_factors", pd_grade, "d2") * td2_6)
    td_6 = float(p["iso_Td_coefficient_a"]) * P ** (2.0 / 3.0) - float(p["iso_Td_coefficient_b"]) * P ** -0.5
    Td, tab = _tabulated_um(TABLE_TD, spec.d_m, spec.pitch_m, cr_grade,
                            _grade_factor("iso_Td_grades", "iso_Td_grade_factors", cr_grade, "d") * td_6)
    notes += ["T_d2 is the ISO 965-1 table value (the rounded formula differs)"] if tab2 else []
    notes += ["T_d is the ISO 965-1 table value (the rounded formula differs)"] if tab else []
    es = _iso_deviation_um(pd_pos, P)
    es_c = _iso_deviation_um(cr_pos, P) if cr_pos != pd_pos else es
    return ThreadLimits(spec, d2 + (es - Td2) * 1e-6, d2 + es * 1e-6, spec.d_m + (es_c - Td) * 1e-6,
                        spec.d_m + es_c * 1e-6, None, None, {"d2_m": d2, "d1_m": d1, "H_m": H},
                        {"T_d2_m": Td2 * 1e-6, "T_d_m": Td * 1e-6}, es * 1e-6, notes)


def _round_in(value_in: float, step_in: float) -> float:
    return round(value_in / step_in) * step_in


def uts_limits(spec: ThreadSpec) -> ThreadLimits:
    p = _p()
    D = spec.d_m / INCH_M
    P = spec.pitch_m / INCH_M
    H = float(p["iso_h_per_pitch"]) * P
    E = D - 0.75 * H
    D1 = D - 1.25 * H
    coarse = spec.series in ("UNC", "UNF", "UNR", "UNJ") or spec.series in ("4-UN", "6-UN", "8-UN")
    le = (spec.engagement_m / INCH_M) if spec.engagement_m else (D if coarse else
                                                                 float(p["asme_engagement_pitches"]) * P)
    tol2a = float(p["asme_pd_diameter_coefficient"]) * D ** (1.0 / 3.0) + \
        float(p["asme_pd_engagement_coefficient"]) * math.sqrt(le) + float(p["asme_pd_pitch_coefficient"]) * P ** (2 / 3)
    classes = list(p["asme_classes"])
    k = classes.index(spec.pd_class)
    tol = float(p["asme_pd_class_factors"][k]) * tol2a
    allowance = float(p["asme_allowance_factors"][k]) * tol2a
    step = float(p["asme_rounding_m"]) / INCH_M
    notes = [f"pitch diameter tolerance for a length of engagement of {le:.4f} in"]
    # as the standard's tables: internal limits are the basic size plus the tolerance, rounded; external limits
    # use the rounded allowance and tolerances (maximum = basic - allowance, minimum = maximum - tolerance)
    if spec.internal:
        d2_min = _round_in(E, step)
        d2_max = _round_in(E + tol, step)
        minor_max = None
        if spec.pd_class in ("1B", "2B"):
            mstep = float(p["asme_minor_rounding_m"]) / INCH_M
            minor_max = _round_in(D1 + float(p["asme_minor_tolerance_a"]) * P - float(p["asme_minor_tolerance_b"]) * P * P,
                                  mstep)
            minor_min = _round_in(D1, mstep)
        else:
            minor_min = _round_in(D1, step)
            notes.append("class 3B minor diameter tolerance not encoded: its maximum is not checked")
        return ThreadLimits(spec, d2_min * INCH_M, d2_max * INCH_M, D * INCH_M, None, minor_min * INCH_M,
                            None if minor_max is None else minor_max * INCH_M, {"d2_m": E * INCH_M, "d1_m": D1 * INCH_M},
                            {"T_D2_m": tol * INCH_M}, 0.0, notes)
    tol, allowance = _round_in(tol, step), _round_in(allowance, step)
    major_tol = _round_in(float(p["asme_major_tolerance_coefficients"][k]) * P ** (2.0 / 3.0), step)
    major_max = _round_in(D - allowance, step)
    d2_max = _round_in(E - allowance, step)
    return ThreadLimits(spec, (d2_max - tol) * INCH_M, d2_max * INCH_M, (major_max - major_tol) * INCH_M,
                        major_max * INCH_M, None, None, {"d2_m": E * INCH_M, "d1_m": D1 * INCH_M},
                        {"T_d2_m": tol * INCH_M, "T_d_m": major_tol * INCH_M}, -allowance * INCH_M, notes)


def limits(spec: ThreadSpec) -> ThreadLimits:
    return iso_limits(spec) if spec.system == "iso" else uts_limits(spec)


def normal_engagement_m(spec: ThreadSpec) -> float:
    """ISO 965-1 upper limit of the normal length of engagement N (6.7 P d^0.2), or the ASME basis."""
    p = _p()
    if spec.system == "uts":
        return spec.d_m
    return float(p["iso_le_normal_max_coefficient"]) * spec.pitch_m * 1e3 * \
        _d_mean_mm(spec.d_m) ** float(p["iso_le_diameter_exponent"]) * 1e-3


# ============================================================================ judgement

def virtual_pitch_diameter(spec: ThreadSpec, simple_m: float, pitch_error_m: float, half_angle_errors_rad,
                           engagement_m: float | None = None, form_angle_rad: float = math.pi / 3.0) -> dict:
    """Virtual (functional) pitch diameter from the simple one and the diameter equivalents (module doc)."""
    p = _p()
    le = engagement_m if engagement_m else (spec.engagement_m or normal_engagement_m(spec))
    n_pitches = le / spec.pitch_m
    dp_sum = abs(pitch_error_m) * n_pitches
    half = 0.5 * form_angle_rad
    f_p = dp_sum / math.tan(half)
    H = float(p["iso_h_per_pitch"]) * spec.pitch_m
    below = (0.5 - float(p["iso_internal_crest_truncation"])) * H      # pitch line to the internal crest: H/4
    above = (0.5 - float(p["iso_external_crest_truncation"])) * H      # pitch line to the external crest: 3H/8
    f_a = 0.0
    for err in half_angle_errors_rad:
        if err is None or not math.isfinite(err):
            continue
        # external: a larger half angle interferes below the pitch line, a smaller one above (internal: reversed)
        larger = err > 0.0
        h = (below if larger else above) if not spec.internal else (above if larger else below)
        f_a += 2.0 * h * abs(err) / math.sin(form_angle_rad)
    sign = -1.0 if spec.internal else 1.0
    return {"virtual_m": simple_m + sign * (f_p + f_a), "f_pitch_m": f_p, "f_flank_m": f_a,
            "engagement_m": le, "cumulative_pitch_error_m": dp_sum}


def judge(limits_: ThreadLimits, measured: dict, engagement_m: float | None = None) -> dict:
    """GO / NOT GO and diameter checks of a measured thread (keys of surface.json ``threads`` entries:
    ``simple_pitch_diameter_m`` or ``pitch_diameter_3wire_m``, ``major_diameter_m``, ``minor_diameter_m``,
    ``pitch_error_m``, ``flank_half_angles_rad``, ``form_angle_nominal_rad``)."""
    spec = limits_.spec
    simple = measured.get("simple_pitch_diameter_m")
    if simple is None or not math.isfinite(float(simple)):
        simple = measured.get("pitch_diameter_3wire_m")
    if simple is None:
        raise ThreadSpecError("the thread measurement has no pitch diameter")
    simple = float(simple)
    alpha = float(measured.get("form_angle_nominal_rad") or math.pi / 3.0)
    halves = measured.get("flank_half_angles_rad") or []
    errors = [float(h) - 0.5 * alpha for h in halves if h is not None and math.isfinite(float(h))]
    virt = virtual_pitch_diameter(spec, simple, float(measured.get("pitch_error_m") or 0.0), errors, engagement_m,
                                  alpha)
    v = virt["virtual_m"]
    checks = {}
    if spec.internal:
        gauges = {"GO": bool(v >= limits_.d2_min_m), "NOT GO": bool(simple <= limits_.d2_max_m)}
    else:
        gauges = {"GO": bool(v <= limits_.d2_max_m), "NOT GO": bool(simple >= limits_.d2_min_m)}
    major = measured.get("major_diameter_m")
    minor = measured.get("minor_diameter_m")
    if major is not None:
        major = float(major)
        lo, hi = limits_.major_min_m, limits_.major_max_m
        checks["major"] = bool((lo is None or major >= lo) and (hi is None or major <= hi))
    if minor is not None:
        minor = float(minor)
        lo, hi = limits_.minor_min_m, limits_.minor_max_m
        checks["minor"] = bool((lo is None or minor >= lo) and (hi is None or minor <= hi))
    ok = all(gauges.values()) and all(checks.values())
    out = limits_.to_json()
    out["pitch_diameter_m"] = {"measured": simple, "virtual": v, "min": limits_.d2_min_m, "max": limits_.d2_max_m}
    if major is not None:
        out["major_diameter_m"] = {"measured": major, "min": limits_.major_min_m, "max": limits_.major_max_m}
    if minor is not None:
        out["minor_diameter_m"] = {"measured": minor, "min": limits_.minor_min_m, "max": limits_.minor_max_m}
    out.update({"gauges": gauges, "diameter_checks": checks, "virtual": virt, "pass": bool(ok)})
    return out


def clear_cache() -> None:
    _p.cache_clear()


__all__ = ["ThreadSpec", "ThreadLimits", "ThreadSpecError", "parse", "limits", "iso_limits", "uts_limits", "judge",
           "virtual_pitch_diameter", "round_r40_um", "coarse_pitch_m", "normal_engagement_m", "clear_cache"]
