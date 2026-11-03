"""Chip colour from temperature and time at temperature (docs/models/mechanics.md, "Chip colour").

Oxide (temper) films grow parabolically, x^2 = k_p t with k_p = k_0 exp(-Q / (R T)) (Wagner
1933), so after a thermal history T(t) the film is that of a hold at temperature T_eq for the
reference time t_ref when exp(-Q/(R T_eq)) t_ref = int exp(-Q/(R T(t))) dt (k_0 cancels). The
material's oxide colour table (temperature -> sRGB, read as the colour after t_ref at that
temperature) is evaluated at T_eq. Materials without a table keep their bare metallic colour.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from . import kernels as K
from . import params as P
from .materials import WorkMaterial


def activation_temperature(work: WorkMaterial) -> float:
    """Q / R (K) of the oxide film growth of the material family."""
    p = P.get(P.COLOUR)
    if work.family == "titanium":
        return float(p["oxide_activation_titanium_K"])
    if work.family in ("steel", "tool_steel", "cast_iron"):
        return float(p["oxide_activation_steel_K"])
    return float(p["oxide_activation_other_K"])


def exposure_from_history(temps_K, dt_s: float, activation_K: float) -> float:
    """Arrhenius exposure (s) of a sampled temperature history (Numba kernel)."""
    return float(K.oxide_exposure(np.ascontiguousarray(temps_K, dtype=np.float64), float(dt_s), float(activation_K)))


def exposure_hold(temp_K: float, time_s: float, activation_K: float) -> float:
    return float(time_s) * math.exp(-activation_K / temp_K) if temp_K > 0.0 else 0.0


def exposure_cooling(start_K: float, ambient_K: float, time_constant_s: float, activation_K: float) -> float:
    """Exposure of a chip cooling exponentially from ``start_K`` toward ``ambient_K`` with
    ``time_constant_s`` (lumped body: rho c t / (2 h)), integrated to the end of cooling."""
    if start_K <= ambient_K or time_constant_s <= 0.0:
        return exposure_hold(start_K, time_constant_s, activation_K)
    # s = t / tau on a geometric grid from 1e-9 to 40 (the integrand falls by orders of magnitude
    # within the first fraction of tau; trapezoid on 4000 nodes is accurate to < 1e-6 relative)
    s = np.concatenate([[0.0], np.geomspace(1e-9, 40.0, 4000)])
    temp = ambient_K + (start_K - ambient_K) * np.exp(-s)
    f = np.exp(-activation_K / temp)
    return float(integrate.trapezoid(f, s)) * time_constant_s


def equivalent_temperature(exposure_s: float, activation_K: float) -> float | None:
    """Temperature whose hold of the reference time gives the same oxide film (None: bare)."""
    t_ref = P.number(P.COLOUR, "oxide_reference_time_s")
    floor = P.number(P.COLOUR, "min_oxide_exposure_s")
    if exposure_s <= floor:
        return None
    ratio = t_ref / exposure_s
    if ratio <= 1.0:
        return math.inf
    return activation_K / math.log(ratio)


def bare_colour(work: WorkMaterial) -> tuple[int, int, int]:
    table = P.get(P.COLOUR)["bare_colours"]
    code = table.get(work.family) or table["steel"]
    code = code.lstrip("#")
    return int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)


def colour_at(work: WorkMaterial, temp_eq_K: float | None) -> tuple[tuple[int, int, int], str]:
    """(sRGB, name) of the material's oxide table at the equivalent temperature; the bare
    colour below the first row or without a table; linear sRGB blend between rows."""
    rows = work.oxide_table
    if rows and temp_eq_K is not None and math.isfinite(temp_eq_K):
        for row in rows:                 # snap rounding noise onto a row (round trip of a table hold)
            if math.isclose(temp_eq_K, row.temp_K, rel_tol=1e-9):
                temp_eq_K = row.temp_K
                break
    if not rows or temp_eq_K is None or temp_eq_K < rows[0].temp_K:
        return bare_colour(work), "bare"
    if temp_eq_K >= rows[-1].temp_K:
        return rows[-1].rgb, rows[-1].name
    for lo, hi in zip(rows[:-1], rows[1:]):
        if lo.temp_K <= temp_eq_K < hi.temp_K:
            w = (temp_eq_K - lo.temp_K) / (hi.temp_K - lo.temp_K)
            rgb = tuple(int(round(a + w * (b - a))) for a, b in zip(lo.rgb, hi.rgb))
            return rgb, lo.name if w < 0.5 else hi.name
    return rows[-1].rgb, rows[-1].name


def chip_colour(work: WorkMaterial, exposure_s: float) -> tuple[tuple[int, int, int], str, float | None]:
    """(sRGB, colour name, equivalent temperature) of a chip with Arrhenius exposure ``exposure_s``."""
    q_over_r = activation_temperature(work)
    teq = equivalent_temperature(exposure_s, q_over_r)
    rgb, name = colour_at(work, teq)
    return rgb, name, teq


def emitted_chip_colour(work: WorkMaterial, chip_temp_K: float, contact_time_s: float, chip_thickness_m: float,
                        h_W_m2K: float, ambient_K: float) -> tuple[tuple[int, int, int], str, float]:
    """Colour of a chip that leaves the tool at ``chip_temp_K`` after ``contact_time_s`` at the
    interface and then cools in a fluid of film coefficient ``h_W_m2K`` (both faces, lumped):
    exposure = hold during contact + exponential cooling. Returns (sRGB, name, exposure_s)."""
    q_over_r = activation_temperature(work)
    rho_c = work.rho_kg_m3 * work.cp(chip_temp_K)
    tau = rho_c * chip_thickness_m / (2.0 * max(h_W_m2K, 1e-9))
    exposure = exposure_hold(chip_temp_K, contact_time_s, q_over_r) + exposure_cooling(chip_temp_K, ambient_K, tau,
                                                                                       q_over_r)
    rgb, name, _ = chip_colour(work, exposure)
    return rgb, name, exposure


__all__ = ["activation_temperature", "exposure_from_history", "exposure_hold", "exposure_cooling",
           "equivalent_temperature", "bare_colour", "colour_at", "chip_colour", "emitted_chip_colour"]
