"""Unbalance of a tool-holder assembly and speed limits (docs/models/holding.md section 3).

ISO 21940-11: the balance quality grade G is the product of the permissible specific unbalance and the
speed, ``G = e_per Omega``; the permissible residual unbalance of a rotor of mass m is ``U_per = m e_per =
G m / Omega`` (in the standard's units ``U_per [g mm] = 9549 G [mm/s] m [kg] / n [1/min]``).

Sources of the assembly's unbalance, in the rotating tool frame T (origin at the tip, +z_T toward the
spindle), each a static vector U (kg m) acting at a height z and a couple C (kg m^2):

* **mounting eccentricity of the tool**: the runout offset o and tilt (a_x, a_y) of the EdgeModel move a
  point p of the axisymmetric tool to ``p + o + a x p``; with the tool's mass m, centre of mass z_c and
  inertias I_a (axial), I_t (transverse, about the centre of mass) this gives
  ``U = m (o_x + a_y z_c, o_y - a_x z_c)`` at z_c and the couple about z_c ``C = (a_y, -a_x) (I_t - I_a)``
  (product of inertia of a tilted symmetric body);
* **residual unbalance of the holder** (and of the tool when it publishes a grade): ``|U| = G m / Omega_G``
  at the grade's speed (else the body's maximum speed), at its centre of mass, in a direction drawn once
  per tool instance from the counter-based stream ``holding.unbalance_phase`` (the phase of a residual
  unbalance is unknown);
* the assembly vector ``U = sum U_i`` acts at the assembly centre of mass z_U, the couple is
  ``C = sum U_i (z_i - z_U) + sum C_i``. The rotor force on the spindle at speed omega is
  ``omega^2 U`` (rotating with the tool) and the moment ``omega^2 e_z x C`` about z_U.

The check uses the worst case of the unknown phases (magnitudes added): the effective quality of the
assembly at the step's speed ``G_eff = U_worst omega / m_assembly`` against the required grade (the
holder's or tool's published grade, else G 2.5 of machine-tool drives).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.rng import hash_uniform, stream_key
from . import params as P

STREAM_PHASE = "holding.unbalance_phase"
ISO_UNBALANCE_CONSTANT = 30.0e3 / math.pi    # 9549.3: U [g mm] = 9549 G [mm/s] m [kg] / n [1/min]


def permissible_unbalance_kg_m(grade_m_s: float, mass_kg: float, omega_rad_s: float) -> float:
    """U_per = G m / Omega (ISO 21940-11)."""
    if omega_rad_s <= 0.0:
        return math.inf
    return grade_m_s * mass_kg / omega_rad_s


def permissible_unbalance_g_mm(grade_mm_s: float, mass_kg: float, speed_rpm: float) -> float:
    """The standard's form: U_per [g mm] = 9549 G [mm/s] m [kg] / n [1/min]."""
    return ISO_UNBALANCE_CONSTANT * grade_mm_s * mass_kg / speed_rpm


def specific_unbalance_m(grade_m_s: float, omega_rad_s: float) -> float:
    """e_per = G / Omega."""
    return grade_m_s / omega_rad_s if omega_rad_s > 0.0 else math.inf


def effective_grade_m_s(unbalance_kg_m: float, mass_kg: float, omega_rad_s: float) -> float:
    """Balance quality of a rotor with residual unbalance U at speed omega: G = U omega / m."""
    return unbalance_kg_m * abs(omega_rad_s) / mass_kg if mass_kg > 0.0 else 0.0


def tool_eccentricity(mass_kg: float, com_z_m: float, I_axial: float, I_trans: float, runout_T_m, tilt_rad):
    """(U [3], z, C [3]) of an axisymmetric tool displaced by the runout offset and tilted (module doc)."""
    o = np.asarray(runout_T_m, dtype=float).reshape(3)
    ax, ay = (float(v) for v in np.asarray(tilt_rad, dtype=float).reshape(2))
    U = mass_kg * np.array([o[0] + ay * com_z_m, o[1] - ax * com_z_m, 0.0])
    K = I_trans - I_axial
    C = np.array([ay * K, -ax * K, 0.0])
    return U, float(com_z_m), C


@dataclass
class Unbalance:
    """Unbalance of one tool-holder assembly in T (static vector, its height, couple) and its check data."""

    static_T_kg_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    z_T_m: float = 0.0
    couple_T_kg_m2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass_kg: float = 0.0
    worst_kg_m: float = 0.0                  # magnitudes added (unknown phases)
    tool_kg_m: float = 0.0                   # mounting eccentricity of the tool
    holder_kg_m: float = 0.0                 # residual of the holder (grade)
    tool_residual_kg_m: float = 0.0          # residual of the tool (its own grade)
    required_grade_m_s: float = 0.0
    required_source: str = ""
    max_rad_s: float = math.inf              # the lower of the tool's and holder's speed limits
    max_rad_s_tool: float = math.inf
    max_rad_s_holder: float = math.inf
    notes: list = field(default_factory=list)

    def force_N(self, omega_rad_s: float) -> float:
        return float(np.linalg.norm(self.static_T_kg_m)) * omega_rad_s ** 2

    def to_dict(self) -> dict:
        def f(x):
            return float(x) if math.isfinite(float(x)) else None
        return {"static_T_kg_m": [float(v) for v in self.static_T_kg_m], "z_T_m": self.z_T_m,
                "couple_T_kg_m2": [float(v) for v in self.couple_T_kg_m2], "mass_kg": self.mass_kg,
                "worst_kg_m": self.worst_kg_m, "tool_kg_m": self.tool_kg_m, "holder_kg_m": self.holder_kg_m,
                "tool_residual_kg_m": self.tool_residual_kg_m, "required_grade_m_s": self.required_grade_m_s,
                "required_source": self.required_source, "max_rad_s": f(self.max_rad_s),
                "max_rad_s_tool": f(self.max_rad_s_tool), "max_rad_s_holder": f(self.max_rad_s_holder),
                "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, d: dict) -> "Unbalance":
        def g(x):
            return math.inf if x is None else float(x)
        return cls(np.array(d["static_T_kg_m"], dtype=float), float(d["z_T_m"]),
                   np.array(d["couple_T_kg_m2"], dtype=float), float(d["mass_kg"]), float(d["worst_kg_m"]),
                   float(d["tool_kg_m"]), float(d["holder_kg_m"]), float(d["tool_residual_kg_m"]),
                   float(d["required_grade_m_s"]), str(d["required_source"]), g(d["max_rad_s"]),
                   g(d["max_rad_s_tool"]), g(d["max_rad_s_holder"]), list(d["notes"]))


def _residual(grade, mass_kg: float, speed_limit: float | None, default_grade: float | None, what: str,
              notes: list) -> tuple[float, float | None]:
    """(|U|, G) of a body's residual unbalance from its grade (or the default grade)."""
    if mass_kg <= 0.0:
        return 0.0, None
    g, speed = (grade if grade is not None else (None, None))
    if g is None:
        if default_grade is None:
            return 0.0, None
        g = default_grade
        notes.append(f"{what}: no balance grade published: G {g * 1e3:.1f} taken at its maximum speed")
    if speed is None or speed <= 0.0:
        speed = speed_limit
    if speed is None or speed <= 0.0 or not math.isfinite(speed):
        notes.append(f"{what}: grade G {g * 1e3:.1f} without a speed and no speed limit: residual unbalance not "
                     f"evaluated")
        return 0.0, g
    return permissible_unbalance_kg_m(g, mass_kg, speed), g


def assembly_unbalance(td, seed: int) -> Unbalance:
    """Unbalance of the tool and holder of one slot (module doc); deterministic phases per slot."""
    notes: list = []
    out = Unbalance()
    if not td.rotating:
        return out
    key = stream_key(STREAM_PHASE)
    m_t, m_h = float(td.tool_mass_kg), float(td.holder_mass_kg)
    U_t, z_t, C_t = tool_eccentricity(m_t, td.tool_com_z_m, td.tool_I_axial_kgm2, td.tool_I_trans_kgm2, td.runout_T_m,
                                      td.tilt_rad)
    vmax_t = td.max_rad_s_tool if td.max_rad_s_tool else None
    vmax_h = td.max_rad_s_holder if td.max_rad_s_holder else None
    vmax = min([v for v in (vmax_t, vmax_h) if v is not None], default=None)
    default_holder = P.p(P.BALANCE, "default_holder_grade_m_s") if m_h > 0.0 else None
    u_h, g_h = _residual(td.holder_grade, m_h, vmax_h if vmax_h is not None else vmax, default_holder,
                         f"holder of T{td.number}", notes)
    u_tg, g_t = _residual(td.tool_grade, m_t, vmax_t if vmax_t is not None else vmax, None, f"T{td.number}", notes)
    psi_h = 2.0 * math.pi * hash_uniform(int(seed), key, int(td.slot), 0, 0)
    psi_t = 2.0 * math.pi * hash_uniform(int(seed), key, int(td.slot), 1, 0)
    U_h = u_h * np.array([math.cos(psi_h), math.sin(psi_h), 0.0])
    U_tg = u_tg * np.array([math.cos(psi_t), math.sin(psi_t), 0.0])
    mass = m_t + m_h
    z_U = (m_t * z_t + m_h * td.holder_com_z_m) / mass if mass > 0.0 else 0.0
    out.static_T_kg_m = U_t + U_h + U_tg
    out.z_T_m = float(z_U)
    out.couple_T_kg_m2 = C_t + U_t * (z_t - z_U) + U_h * (td.holder_com_z_m - z_U) + U_tg * (z_t - z_U)
    out.couple_T_kg_m2[2] = 0.0
    out.mass_kg = mass
    out.tool_kg_m = float(np.linalg.norm(U_t))
    out.holder_kg_m = float(u_h)
    out.tool_residual_kg_m = float(u_tg)
    out.worst_kg_m = out.tool_kg_m + out.holder_kg_m + out.tool_residual_kg_m
    published = [g for g in ((td.holder_grade or (None,))[0], (td.tool_grade or (None,))[0]) if g is not None]
    if published:
        out.required_grade_m_s = float(min(published))
        out.required_source = "published grade of the holder or tool"
    else:
        out.required_grade_m_s = P.p(P.BALANCE, "required_grade_m_s")
        out.required_source = "ISO 21940-11 G 2.5 (machine-tool drives)"
    out.max_rad_s_tool = float(vmax_t) if vmax_t is not None else math.inf
    out.max_rad_s_holder = float(vmax_h) if vmax_h is not None else math.inf
    out.max_rad_s = min(out.max_rad_s_tool, out.max_rad_s_holder)
    out.notes = notes
    return out


__all__ = ["Unbalance", "assembly_unbalance", "tool_eccentricity", "permissible_unbalance_kg_m",
           "permissible_unbalance_g_mm", "specific_unbalance_m", "effective_grade_m_s", "STREAM_PHASE",
           "ISO_UNBALANCE_CONSTANT"]
