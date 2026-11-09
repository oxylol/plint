"""Model coefficients (db/models/surface-*.yaml) and the project / material values the surface models read.

Project and database objects arrive as SIMap views (cnctwin.db.simap), plain mappings (tests) or simple
namespaces; :func:`field_of` and :func:`num` read any of them. :func:`material_props` gathers the work
material values the burr, integrity and roughness models need, with the defaults each falls back to
recorded in ``notes`` (listed in surface.json under ``estimates``).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..db.models import params as model_params
from ..units import REFERENCE_TEMP_K, ZERO_DEGC_K

ROUGHNESS = "surface-roughness"
BURR = "surface-burr"
INTEGRITY = "surface-integrity"


@functools.lru_cache(maxsize=None)
def _params(model: str) -> dict:
    return dict(model_params(model))


def roughness() -> dict:
    return _params(ROUGHNESS)


def burr() -> dict:
    return _params(BURR)


def integrity() -> dict:
    return _params(INTEGRITY)


def field_of(obj, key: str):
    """``obj.key`` / ``obj[key]`` of a SIMap, mapping or namespace (None when absent)."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    try:
        return getattr(obj, key)
    except AttributeError:
        return None


def plain(value):
    """The number / text / list inside a Sourced value (unchanged otherwise)."""
    inner = getattr(value, "value", value)
    return inner


def num(obj, *keys, default=None):
    """First of ``keys`` present in obj as a float (Sourced and {value: x} mappings unwrapped)."""
    for key in keys:
        value = field_of(obj, key)
        if value is None:
            continue
        if isinstance(value, Mapping) and "value" in value:
            value = value["value"]
        value = plain(value)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def table_value(obj, key: str, at_K: float = REFERENCE_TEMP_K):
    """A number, a {value: x} mapping, a SIMap Table or a raw {table: [[degC, v], ...]} at ``at_K``."""
    value = field_of(obj, key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "table" in value:
            rows = value["table"]
            xs = [float(r[0]) + ZERO_DEGC_K for r in rows]
            ys = [float(r[1]) for r in rows]
            import numpy as np
            return float(np.interp(at_K, xs, ys))
        value = value.get("value")
    if value is None:
        return None
    inner = value.value if hasattr(value, "src") and hasattr(value, "value") else value   # a Sourced scalar or table
    if callable(inner) and hasattr(inner, "x"):
        return float(inner(at_K))
    try:
        return float(plain(value))
    except (TypeError, ValueError):
        return None


@dataclass
class MaterialProps:
    """Work material values in SI (defaults recorded in ``notes``)."""

    family: str = ""
    density_kg_m3: float = 7850.0
    youngs_Pa: float = 2.1e11
    poisson: float = 0.3
    expansion_1_K: float = 1.2e-5
    conductivity_W_mK: float = 50.0
    specific_heat_J_kgK: float = 480.0
    yield_Pa: float = 3.5e8
    ultimate_Pa: float = 6.0e8
    fracture_strain: float = 0.3
    hardness_HV: float = 0.0
    min_chip_ratio: float = 0.3
    wh_depth_m: float = 0.0
    wh_increase_frac: float = 0.0
    burn_K: float | None = None
    overtemper_K: float | None = None
    white_layer_K: float | None = None
    melt_smear_K: float | None = None
    mu_sliding: float | None = None
    composite: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# fallback values when a material file lacks a property (structural steel, listed as estimates)
_FALLBACK = {"density_kg_m3": 7850.0, "youngs_Pa": 2.1e11, "poisson": 0.3, "expansion_1_K": 1.2e-5,
             "conductivity_W_mK": 50.0, "specific_heat_J_kgK": 480.0, "yield_Pa": 3.5e8, "ultimate_Pa": 6.0e8,
             "fracture_strain": 0.3}


def material_props(material) -> MaterialProps:
    """Values of a resolved material (SI view) or a raw material mapping."""
    out = MaterialProps()
    notes = out.notes
    out.family = str(plain(field_of(material, "family")) or "")
    phys = field_of(material, "physical")
    raw = getattr(material, "raw", None)
    raw_phys = raw.get("physical") if isinstance(raw, Mapping) else (phys if isinstance(phys, Mapping) else None)

    def pick(name, si_keys, raw_key, scale):
        value = None
        for key in si_keys:
            value = table_value(phys, key)
            if value is not None:
                break
        if value is None and raw_phys is not None:
            found = table_value(raw_phys, raw_key)
            value = found * scale if found is not None else None
        if value is None:
            value = _FALLBACK[name]
            notes.append(f"material {name}: not in the material file, {value:g} (structural steel) used")
        setattr(out, name, float(value))

    pick("density_kg_m3", ("density_kg_m3",), "density_kg_m3", 1.0)
    pick("youngs_Pa", ("youngs_modulus_Pa",), "youngs_modulus_GPa", 1e9)
    pick("poisson", ("poisson",), "poisson", 1.0)
    pick("expansion_1_K", ("expansion_1_K",), "expansion_1_K", 1.0)
    pick("conductivity_W_mK", ("conductivity_W_mK",), "conductivity_W_mK", 1.0)
    pick("specific_heat_J_kgK", ("specific_heat_J_kgK",), "specific_heat_J_kgK", 1.0)
    pick("yield_Pa", ("yield_Pa",), "yield_MPa", 1e6)
    pick("ultimate_Pa", ("ultimate_Pa",), "ultimate_MPa", 1e6)
    strain = num(phys, "fracture_strain")
    if strain is None:
        elong = num(phys, "elongation_frac")
        if elong is None and raw_phys is not None:
            e_pct = num(raw_phys, "elongation_pct")
            elong = e_pct / 100.0 if e_pct is not None else None
        if elong is not None:
            strain = math.log(1.0 + max(elong, 0.0))
            notes.append("material fracture strain: true strain of the elongation at break ln(1 + A)")
    if strain is None:
        strain = _FALLBACK["fracture_strain"]
        notes.append(f"material fracture_strain: not in the material file, {strain:g} used")
    out.fracture_strain = float(strain)
    hard = field_of(phys, "hardness") if phys is not None else None
    out.hardness_HV = _hardness_HV(hard)
    cutting = field_of(material, "cutting")
    ratio = num(cutting, "min_chip_thickness_ratio")
    if ratio is None:
        ratio = float(roughness()["min_chip_ratio"])
        notes.append(f"material min_chip_thickness_ratio: not in the material file, {ratio:g} (surface-roughness) used")
    out.min_chip_ratio = float(ratio)
    wh = field_of(cutting, "work_hardening")
    depth = num(wh, "layer_depth_m")
    if depth is None and isinstance(wh, Mapping):
        d_um = num(wh, "layer_depth_um")
        depth = d_um * 1e-6 if d_um is not None else None
    inc = num(wh, "hardness_increase_frac")
    if inc is None and isinstance(wh, Mapping):
        i_pct = num(wh, "hardness_increase_pct")
        inc = i_pct / 100.0 if i_pct is not None else None
    if depth is None or inc is None:
        notes.append("material work_hardening: not in the material file, no hardened layer is predicted")
    out.wh_depth_m = float(depth or 0.0)
    out.wh_increase_frac = float(inc or 0.0)
    damage = field_of(material, "thermal_damage")
    for attr, keys in (("burn_K", ("burn_K", "burn_degC")), ("overtemper_K", ("overtemper_K", "overtemper_degC")),
                       ("white_layer_K", ("white_layer_K", "white_layer_degC")),
                       ("melt_smear_K", ("melt_smear_K", "melt_smear_degC"))):
        value = num(damage, keys[0])
        if value is None:
            raw_value = num(damage, keys[1])
            value = raw_value + ZERO_DEGC_K if raw_value is not None else None
        setattr(out, attr, value)
    friction = field_of(material, "friction")
    if isinstance(friction, (list, tuple)):
        for entry in friction:
            mu = num(entry, "mu_sliding")
            if mu is not None:
                out.mu_sliding = float(mu)
                break
    comp = field_of(material, "composite")
    if comp is not None and (isinstance(comp, Mapping) or hasattr(comp, "raw")):
        out.composite = _composite(comp)
    return out


def _hardness_HV(hard) -> float:
    """Vickers hardness from HV, HB (HV ~ 1.05 HB, ASTM E140 for steels) or HRC (ASTM E140 table fit)."""
    if hard is None:
        return 0.0
    hv = num(hard, "HV")
    if hv is not None:
        return hv
    hb = num(hard, "HB")
    if hb is not None:
        return 1.05 * hb
    hrc = num(hard, "HRC")
    if hrc is not None:
        # ASTM E140 Table 1 (steels), 20-68 HRC: HV ~ 223 + 2.5 HRC + 0.135 HRC^2 within 3 %
        return 223.0 + 2.5 * hrc + 0.135 * hrc * hrc
    return 0.0


def _composite(comp) -> dict:
    out = {}
    raw = getattr(comp, "raw", comp)
    if not isinstance(raw, Mapping):
        return out
    out["fibre"] = str(raw.get("fibre", ""))
    out["layup"] = str(raw.get("layup", ""))
    delam = raw.get("delamination") if isinstance(raw.get("delamination"), Mapping) else {}
    g = num(delam, "G_IC_J_m2")
    if g is not None:
        out["G_IC_J_m2"] = g
    f = num(delam, "critical_thrust_N")
    if f is not None:
        out["critical_thrust_N"] = f
    e = num(delam, "E_GPa")
    if e is not None:
        out["E_Pa"] = e * 1e9
    nu = num(delam, "poisson")
    if nu is not None:
        out["poisson"] = nu
    vf = num(raw, "fibre_volume_fraction")
    if vf is not None:
        out["fibre_volume_fraction"] = vf
    return out


__all__ = ["roughness", "burr", "integrity", "field_of", "plain", "num", "table_value", "MaterialProps",
           "material_props"]
