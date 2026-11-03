"""Model coefficients of the chips domain (db/models/chips-*.yaml) and the material/fluid properties the
kernels need, in kernel form (docs/models/chips.md).

:func:`load` reads the seven chips model files once; :func:`chip_model` combines them with the
workpiece material (``ctx.project.material``) and the numerical settings into an
:class:`cnctwin.chips.abi.ChipModel`; :func:`chip_shape` derives the per-chip aerodynamic and thermal
quantities of a chip of given thickness, width and length.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..db.models import params as model_params
from ..db.simap import Table
from ..units import REFERENCE_TEMP_K
from . import abi

MODELS = ("chips-aero", "chips-heat", "chips-contact", "chips-bulk", "chips-flute", "chips-flush", "chips-hazards")

DEG = math.pi / 180.0


@dataclass(frozen=True)
class ChipParams:
    aero: Mapping
    heat: Mapping
    contact: Mapping
    bulk: Mapping
    flute: Mapping
    flush: Mapping
    hazards: Mapping


def load() -> ChipParams:
    """All chips model parameters (SI; cached by cnctwin.db.models)."""
    return ChipParams(*(model_params(name) for name in MODELS))


# ------------------------------------------------------------------ material helpers

def value(x, default: float | None = None) -> float | None:
    """SI float of a database value (Sourced, Table at 20 degC, number), or ``default``."""
    if x is None:
        return default
    try:
        number = float(x)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def family(material) -> str:
    fam = getattr(material, "family", None) if material is not None else None
    return str(fam) if fam else "steel"


def density(material) -> float:
    physical = getattr(material, "physical", None) if material is not None else None
    rho = value(getattr(physical, "density_kg_m3", None) if physical is not None else None)
    if rho is None or rho <= 0.0:
        raise ValueError("the workpiece material has no density (physical.density_kg_m3): chip masses need it")
    return rho


def cp_table(material) -> tuple[np.ndarray, np.ndarray]:
    """(T_K, cp) of the material's specific heat (a table, or one value held constant)."""
    physical = getattr(material, "physical", None) if material is not None else None
    cp = getattr(physical, "specific_heat_J_kgK", None) if physical is not None else None
    if isinstance(cp, Table) and len(cp.points_K) >= 1:
        points, values = cp.points_K, cp.values
        order = np.argsort(np.asarray(points, dtype=float))
        return (np.ascontiguousarray(np.asarray(points, dtype=float)[order]),
                np.ascontiguousarray(np.asarray(values, dtype=float)[order]))
    single = value(cp)
    if single is None or single <= 0.0:
        raise ValueError("the workpiece material has no specific heat (physical.specific_heat_J_kgK)")
    return np.array([REFERENCE_TEMP_K]), np.array([single])


def enthalpy_table(cp_T_K: np.ndarray, cp_J_kgK: np.ndarray) -> np.ndarray:
    """Specific enthalpy at the table points, 0 at 20 degC, for the piecewise-linear cp held constant
    outside the table (exact integral)."""
    t = np.asarray(cp_T_K, dtype=float)
    c = np.asarray(cp_J_kgK, dtype=float)
    h = np.zeros_like(t)
    for i in range(1, len(t)):
        h[i] = h[i - 1] + 0.5 * (c[i] + c[i - 1]) * (t[i] - t[i - 1])
    return h - _h_at(REFERENCE_TEMP_K, t, c, h)


def _h_at(temp: float, t: np.ndarray, c: np.ndarray, h: np.ndarray) -> float:
    if temp <= t[0]:
        return h[0] + c[0] * (temp - t[0])
    if temp >= t[-1]:
        return h[-1] + c[-1] * (temp - t[-1])
    i = int(np.searchsorted(t, temp)) - 1
    dt = temp - t[i]
    slope = (c[i + 1] - c[i]) / (t[i + 1] - t[i])
    return h[i] + c[i] * dt + 0.5 * slope * dt * dt


def enthalpy_array(temp_K, cp_T_K: np.ndarray, cp_J_kgK: np.ndarray, h_J_kg: np.ndarray) -> np.ndarray:
    """Vectorised twin of kernels.enthalpy (piecewise-quadratic enthalpy of the piecewise-linear cp)."""
    t = np.atleast_1d(np.asarray(temp_K, dtype=float))
    ct = np.asarray(cp_T_K, dtype=float)
    cv = np.asarray(cp_J_kgK, dtype=float)
    ht = np.asarray(h_J_kg, dtype=float)
    out = np.empty_like(t)
    low = t <= ct[0]
    high = t >= ct[-1]
    out[low] = ht[0] + cv[0] * (t[low] - ct[0])
    out[high] = ht[-1] + cv[-1] * (t[high] - ct[-1])
    mid = ~(low | high)
    if np.any(mid):
        i = np.clip(np.searchsorted(ct, t[mid], side="left") - 1, 0, len(ct) - 2)
        dt = t[mid] - ct[i]
        slope = (cv[i + 1] - cv[i]) / (ct[i + 1] - ct[i])
        out[mid] = ht[i] + cv[i] * dt + 0.5 * slope * dt * dt
    return out


def melting_K(material) -> float | None:
    physical = getattr(material, "physical", None) if material is not None else None
    if physical is None:
        return None
    for key in ("solidus_K", "liquidus_K", "decomposition_K", "glass_transition_K"):
        temp = value(getattr(physical, key, None))
        if temp is not None and temp > 0.0:
            return temp
    flow = getattr(material, "flow_stress", None)
    melt = value(getattr(flow, "T_melt_K", None)) if flow is not None else None
    return melt


def yield_strength(material) -> float:
    physical = getattr(material, "physical", None) if material is not None else None
    sy = value(getattr(physical, "yield_Pa", None)) if physical is not None else None
    if sy is None or sy <= 0.0:
        flow = getattr(material, "flow_stress", None)
        sy = value(getattr(flow, "A_Pa", None)) if flow is not None else None
    if sy is None or sy <= 0.0:
        raise ValueError("the workpiece material gives neither physical.yield_MPa nor flow_stress.A_MPa "
                         "(the chip compaction stress scales with it)")
    return sy


def ignition_K(material) -> tuple[float | None, float | None]:
    """(chip ignition / hazard temperature, fines/powder ignition temperature) in K from the material's
    ignition section (None when not given)."""
    section = getattr(material, "ignition", None) if material is not None else None
    if section is None:
        return None, None
    chip = [value(getattr(section, key, None)) for key in ("chip_ignition_K", "chip_hazard_K")]
    fines = [value(getattr(section, key, None)) for key in ("fines_ignition_K", "powder_ignition_K")]
    chip = [t for t in chip if t is not None]
    fines = [t for t in fines if t is not None]
    chip_t = min(chip) if chip else None
    fines_t = min(fines) if fines else chip_t
    return chip_t, fines_t


def critical_thickness(material, bulk: Mapping) -> float:
    section = getattr(material, "ignition", None) if material is not None else None
    t = value(getattr(section, "critical_chip_thickness_m", None)) if section is not None else None
    return t if t is not None and t > 0.0 else float(bulk["fines_thickness_m"])


def dust_fraction(material, bulk: Mapping) -> float:
    """Mass fraction of the removed material leaving as airborne dust."""
    composite = getattr(material, "composite", None) if material is not None else None
    fraction = value(getattr(composite, "dust_mass_fraction", None)) if composite is not None else None
    if fraction is not None:
        return min(max(fraction, 0.0), 1.0)
    chip = getattr(material, "chip", None) if material is not None else None
    tendency = getattr(chip, "tendency", None) if chip is not None else None
    if str(tendency) == "powder":
        return float(bulk["powder_dust_fraction"])
    return 0.0


# ------------------------------------------------------------------ form tables

def form_tables(bulk: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(R, tan(repose), long flag, wrap flag) indexed by ISO 3685 code 0..99."""
    ratio = np.full(abi.N_FORM, float(bulk["bulk_ratio_default"]))
    for table in (bulk["bulk_ratio_by_form"], bulk["bulk_ratio_estimated"]):
        for code, r in dict(table).items():
            ratio[int(code)] = float(r)
    repose = np.full(abi.N_FORM, math.tan(float(bulk["repose_default_rad"])))
    for code, deg in dict(bulk["repose_deg_by_form"]).items():
        repose[int(code)] = math.tan(float(deg) * DEG)
    long_flag = np.zeros(abi.N_FORM, np.int64)
    long_flag[[int(c) for c in bulk["long_forms"]]] = 1
    wrap_flag = np.zeros(abi.N_FORM, np.int64)
    wrap_flag[[int(c) for c in bulk["wrap_forms"]]] = 1
    return ratio, repose, long_flag, wrap_flag


def pair_tables(contact: Mapping, material_family: str) -> tuple[np.ndarray, np.ndarray]:
    """(restitution, friction) of the chip material against every surface kind (abi.SURF_*)."""
    e0 = float(dict(contact["restitution_by_family"]).get(material_family, contact["restitution_default"]))
    mu0 = float(dict(contact["friction_by_family"]).get(material_family, contact["friction_default"]))
    fe = dict(contact["surface_restitution_factor"])
    fm = dict(contact["surface_friction_factor"])
    e = np.zeros(abi.N_SURF)
    mu = np.zeros(abi.N_SURF)
    # file keys of the surface kinds ("table" would read as a temperature table in a parameter file)
    keys = {"none": "enclosure", "table": "machine_table"}
    for kind, name in enumerate(abi.SURF_NAMES):
        key = keys.get(name, name)
        e[kind] = min(e0 * float(fe[key]), 1.0)
        mu[kind] = mu0 * float(fm[key])
    return e, mu


def hl_coefficients(aero: Mapping, sphericity) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Haider-Levenspiel A, B, C, D for sphericities (clamped to the fitted range)."""
    psi = np.clip(np.asarray(sphericity, dtype=float), float(aero["hl_min_sphericity"]), 1.0)
    a = np.exp(aero["hl_a0"] + aero["hl_a1"] * psi + aero["hl_a2"] * psi ** 2)
    b = aero["hl_b0"] + aero["hl_b1"] * psi
    c = np.exp(aero["hl_c0"] + aero["hl_c1"] * psi + aero["hl_c2"] * psi ** 2 + aero["hl_c3"] * psi ** 3)
    d = np.exp(aero["hl_d0"] + aero["hl_d1"] * psi + aero["hl_d2"] * psi ** 2 + aero["hl_d3"] * psi ** 3)
    return a, b, c, d


def chip_shape(thickness_m, width_m, length_m, rho_kg_m3: float):
    """Per-chip (mass, volume-equivalent diameter d_v, drag reference area, surface, sphericity) of a
    rectangular strip t x w x L (curled forms keep the strip's volume and surface).

    The drag reference area is the cross-section of the volume-equivalent sphere, pi d_v^2 / 4, the area
    Haider & Levenspiel's C_D is defined with (the shape enters through the sphericity
    psi = pi^(1/3) (6 V)^(2/3) / S).
    """
    t = np.maximum(np.asarray(thickness_m, dtype=float), 1e-9)
    w = np.maximum(np.asarray(width_m, dtype=float), t)
    length = np.maximum(np.asarray(length_m, dtype=float), t)
    volume = t * w * length
    surface = 2.0 * (t * w + t * length + w * length)
    dv = np.cbrt(6.0 * volume / math.pi)
    psi = math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface
    return rho_kg_m3 * volume, dv, 0.25 * math.pi * dv * dv, surface, psi


# ------------------------------------------------------------------ the kernel model

def chip_model(material, settings, params: ChipParams) -> abi.ChipModel:
    """ChipModel of the project's workpiece material (kernel layout)."""
    fam = family(material)
    rho = density(material)
    cp_t, cp_v = cp_table(material)
    h_t = enthalpy_table(cp_t, cp_v)
    melt = melting_K(material)
    weld = float(params.contact["weld_homologous_temperature"]) * melt if melt else math.inf
    emis = float(dict(params.heat["emissivity_by_family"]).get(fam, params.heat["emissivity_default"]))
    aero = params.aero
    hl = np.array([aero["hl_stokes"], aero["hl_a0"], aero["hl_a1"], aero["hl_a2"], aero["hl_b0"], aero["hl_b1"],
                   aero["hl_c0"], aero["hl_c1"], aero["hl_c2"], aero["hl_c3"], aero["hl_d0"], aero["hl_d1"],
                   aero["hl_d2"], aero["hl_d3"]], dtype=float)
    heat = params.heat
    rm = np.array([heat["rm_a"], heat["rm_b"], heat["rm_m"], heat["rm_n"]], dtype=float)
    e_pair, mu_pair = pair_tables(params.contact, fam)
    ratio, repose, long_flag, wrap_flag = form_tables(params.bulk)
    contact = params.contact
    wet = np.array([math.log(float(contact["film_thickness_m"]) / float(contact["asperity_height_m"])),
                    float(contact["contact_angle_cos"]), float(contact["wilhelmy_factor"])])
    numerics = np.array([settings.dt_max_s, settings.drag_dt_factor, settings.contact_offset_m,
                         float(settings.max_bounces), float(settings.max_zero_hits)])
    return abi.ChipModel(np.array([rho]), cp_t, cp_v, np.ascontiguousarray(h_t), np.array([emis]),
                         np.array([weld]), hl, rm, e_pair, mu_pair, np.array([float(contact["paddle_restitution"])]),
                         wet, ratio, repose, long_flag, wrap_flag,
                         np.array([critical_thickness(material, params.bulk)]),
                         np.array([float(contact["settle_bounce_sizes"])]), np.array([float(contact["wrap_turns"])]),
                         np.array([float(contact["hook_radii"])]), numerics)


def fluid_properties(fluid, temp_K: float) -> dict:
    """rho, mu, k, Pr, surface tension of the coolant fluid at its temperature (db/coolants/<id>.yaml);
    water-based properties are not invented: a missing property raises."""
    if fluid is None:
        raise ValueError("no coolant fluid")

    def prop(key):
        item = getattr(fluid, key, None)
        if item is None:
            return None
        if isinstance(item, Table):
            return float(item(temp_K))
        return value(item)

    rho = prop("density_kg_m3")
    mu = prop("viscosity_Pa_s")
    k = prop("conductivity_W_mK")
    cp = prop("specific_heat_J_kgK")
    gamma = prop("surface_tension_N_per_m")
    missing = [name for name, v in (("density_kg_m3", rho), ("viscosity_mPa_s", mu), ("conductivity_W_mK", k),
                                    ("specific_heat_J_kgK", cp)) if v is None]
    if missing:
        raise ValueError(f"coolant fluid {getattr(fluid, 'id', '?')} lacks {', '.join(missing)}")
    return {"rho": rho, "mu": mu, "k": k, "pr": cp * mu / k, "gamma": gamma if gamma is not None else 0.0}
