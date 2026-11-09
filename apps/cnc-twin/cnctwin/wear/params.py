"""Model coefficients (db/models/wear-*.yaml) and the database data the wear models use, as plain SI
values (docs/models/wear.md "Inputs").

Database objects arrive as SIMap views (cnctwin.db / cnctwin.project, architecture 18) or plain
mappings with file keys (wrapped here). Usui constants are converted from the unit system the file
states, explicitly: ``A_per_MPa`` with its ``rate_unit`` (cnctwin.db.schemas.USUI_RATE_UNITS), or
``A_per_Pa`` in SI rates; a constant without its unit system is an input error. Taylor entries keep
their reference values (``T_ref``, feed and depth references, the VB limit) and their life basis
(what T counts, and the share of it the edge is in contact in the reference cut).
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from ..db import models
from ..db.schemas import LIFE_BASES, USUI_RATE_UNITS, default_life_basis
from ..db.simap import SIMap, Sourced, Table, wrap
from ..mechanics.materials import coating_matches, normalise, table_arrays
from ..units import REFERENCE_TEMP_K, STANDARD_GRAVITY_M_S2

FLANK = "wear-flank"
CALIBRATION = "wear-calibration"
FRACTURE = "wear-fracture"
LIFE = "wear-life"

HV_PA = STANDARD_GRAVITY_M_S2 * 1e6     # 1 HV = 1 kgf/mm2 in Pa
SI_RATES = "m/s with sigma_n in Pa, V_s in m/s"
TOOL_CLASSES_BRITTLE = ("carbide", "cermet", "ceramic", "cbn", "pcd")


@functools.lru_cache(maxsize=None)
def get(model: str) -> dict:
    """name -> SI value of one wear model file (validated by cnctwin.db.models)."""
    return dict(models.params(model))


def number(model: str, name: str) -> float:
    return float(get(model)[name])


# ------------------------------------------------------------------------------------------ helpers
def as_map(value):
    if value is None:
        return None
    if isinstance(value, SIMap):
        return value
    if isinstance(value, Mapping):
        return wrap(dict(value))
    raise InputError(f"expected a mapping, got {type(value).__name__}")


def field_of(mapping, key):
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return getattr(mapping, key, None)


def plain(value):
    if isinstance(value, Sourced):
        return value.value
    return value


def num(mapping, key, default=None):
    value = plain(field_of(mapping, key))
    if isinstance(value, Table):
        return float(value) if value.x_unit == "K" else default
    if value is None or isinstance(value, (str, bool, list, tuple, Mapping)):
        return default
    number_ = float(value)
    return number_ if math.isfinite(number_) else default


def text(mapping, key, default: str = "") -> str:
    value = plain(field_of(mapping, key))
    return default if value is None else str(value)


def table(mapping, key, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None:
    """(T_K, values) of a temperature-dependent property (Table or constant), None when absent."""
    value = field_of(mapping, key)
    if value is None or plain(value) is None:
        return None
    temps, values = table_arrays(value)
    return np.ascontiguousarray(temps, dtype=float), np.ascontiguousarray(values * scale, dtype=float)


def _raw(item) -> Mapping:
    raw = getattr(item, "raw", None)
    if isinstance(raw, Mapping):
        return raw
    if isinstance(item, Mapping):
        return item
    raise InputError(f"expected a mapping, got {type(item).__name__}")


def _raw_number(value, what: str) -> float:
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None or isinstance(value, (str, bool)):
        raise InputError(f"{what} needs a number")
    return float(value)


# ------------------------------------------------------------------------------------------ Usui
@dataclass(frozen=True)
class UsuiEntry:
    """One published Usui set dW/dt = A sigma_n V_s exp(-B / T) in SI rates (one temperature branch)."""

    A_per_Pa: float
    B_K: float
    T_lo_K: float              # -inf when open
    T_hi_K: float              # +inf when open
    surface: str = "both"      # rake | flank | both
    work_family: str = ""
    tool_class: str = ""
    coating: str = ""
    src: str = ""
    origin: str = ""           # tool_material:<id> | material:<id>


def usui_entry(item, origin: str) -> UsuiEntry:
    """Read one Usui entry, converting A explicitly from the stated unit system."""
    raw = _raw(item)
    rate = raw.get("rate_unit")
    if "A_per_MPa" in raw:
        factor = USUI_RATE_UNITS.get(rate)
        if factor is None:
            raise InputError(f"{origin}: Usui A_per_MPa needs rate_unit, one of: {'; '.join(USUI_RATE_UNITS)}")
        value = _raw_number(raw["A_per_MPa"], f"{origin}: Usui A_per_MPa") * factor
    elif "A_per_Pa" in raw:
        if rate is not None and USUI_RATE_UNITS.get(rate) != 1.0:
            raise InputError(f"{origin}: Usui A_per_Pa means SI rates (rate_unit {SI_RATES!r}), got {rate!r}")
        value = _raw_number(raw["A_per_Pa"], f"{origin}: Usui A_per_Pa")
    else:
        raise InputError(f"{origin}: a Usui entry needs A_per_MPa (with rate_unit) or A_per_Pa")
    if "B_K" not in raw:
        raise InputError(f"{origin}: a Usui entry needs B_K")
    b_value = _raw_number(raw["B_K"], f"{origin}: Usui B_K")
    if value <= 0.0 or b_value <= 0.0:
        raise InputError(f"{origin}: Usui A and B must be positive")
    rng = raw.get("T_range_K")
    if isinstance(rng, Mapping):
        rng = rng.get("value")
    lo, hi = -math.inf, math.inf
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo = -math.inf if rng[0] is None else float(rng[0])
        hi = math.inf if rng[1] is None else float(rng[1])
    src = raw.get("src") if isinstance(raw.get("src"), str) else (getattr(item, "src", "") or "")
    return UsuiEntry(value, b_value, lo, hi, str(raw.get("surface") or "both"), str(raw.get("work_family") or ""),
                     str(raw.get("tool_class") or ""), str(raw.get("coating") or ""), str(src), origin)


def usui_entries(items, origin: str) -> tuple[UsuiEntry, ...]:
    out = []
    for i, item in enumerate(items or []):
        out.append(usui_entry(item, f"{origin} usui[{i}]"))
    return tuple(out)


def select_usui(entries, surface: str, family: str, tool_class: str, coating: str) -> list[UsuiEntry]:
    """Branches (sorted by the lower end of their range) of the best-matching set for one face."""
    candidates = []
    for order, e in enumerate(entries):
        if e.surface not in (surface, "both", ""):
            continue
        if e.work_family not in ("", "any", family):
            continue
        if e.tool_class not in ("", tool_class):
            continue
        candidates.append((coating_matches(e.coating, coating), order, e))
    if not candidates:
        return []
    best = max(score for score, _, _ in candidates)
    chosen = [e for score, order, e in sorted(candidates, key=lambda c: c[1]) if score == best]
    out: list[UsuiEntry] = []
    for e in sorted(chosen, key=lambda x: x.T_lo_K):
        if out and out[-1].T_lo_K == e.T_lo_K:
            continue
        out.append(e)
    return out


# ------------------------------------------------------------------------------------------ Taylor
CUTTING_TIME = "cutting_time"
CONTACT_TIME = "contact_time"
INTERRUPTED_PROCESSES = ("milling",)     # an edge is in the material only over its engaged arc


@dataclass(frozen=True)
class TaylorEntry:
    """Extended Taylor: vc (T / T_ref)^n (f / f_ref)^a (ap / ap_ref)^b = C at the VB limit.

    ``life_basis`` says what T counts: ``cutting_time`` (the tool feeding in the material) or
    ``contact_time`` (the edge in contact). ``contact_fraction`` is the share of the entry's own life
    the edge is in contact in its reference cut: 1 for a contact-time life and for a continuous cut;
    engaged arc / 2 pi for a milling cutting-time life. The wear law integrates contact time, so the
    calibration targets ``contact_life_s`` (docs/models/wear.md section 3)."""

    tool_class: str
    coating: str
    process: str
    C_m_s: float
    n: float
    a: float
    b: float
    T_ref_s: float
    feed_ref_m: float | None
    ap_ref_m: float | None
    VB_limit_m: float
    vc_range_m_s: tuple[float, float] | None
    src: str
    index: int
    life_basis: str = CUTTING_TIME
    contact_fraction: float = 1.0
    engagement: str = "continuous"     # stated | default | continuous | contact (how contact_fraction was set)
    grade: str = ""                    # the entry's tool grade text (tool-material family within the class)

    def _load(self, feed_m, ap_m) -> float:
        factor = 1.0
        if self.a and feed_m is not None and self.feed_ref_m:
            factor *= (feed_m / self.feed_ref_m) ** self.a
        if self.b and ap_m is not None and self.ap_ref_m:
            factor *= (ap_m / self.ap_ref_m) ** self.b
        return factor

    def life_s(self, vc_m_s: float, feed_m: float | None = None, ap_m: float | None = None) -> float:
        """The Taylor life on the entry's own basis (cutting or contact time)."""
        return self.T_ref_s * (self.C_m_s / (vc_m_s * self._load(feed_m, ap_m))) ** (1.0 / self.n)

    def contact_life_s(self, vc_m_s: float, feed_m: float | None = None, ap_m: float | None = None) -> float:
        """Edge contact time to the VB limit in the entry's reference cut: life x contact fraction."""
        return self.life_s(vc_m_s, feed_m, ap_m) * self.contact_fraction

    def speed_for_life(self, life_s: float, feed_m: float | None = None, ap_m: float | None = None) -> float:
        return self.C_m_s * (self.T_ref_s / life_s) ** self.n / self._load(feed_m, ap_m)

    @property
    def uncoated(self) -> bool:
        return normalise(self.coating or "none") in ("none", "uncoated", "")


def life_basis_of(entry) -> tuple[str, float, str]:
    """(life basis, contact fraction, how the fraction was set) of one Taylor entry (database view):
    the stated ``life_basis``, else the default of its process (cnctwin.db.schemas.default_life_basis);
    for a cutting-time milling life the fraction is the reference cut's engaged arc / 2 pi (the entry's
    ``engaged_arc_ref_deg``, else wear-calibration's ``milling_reference_arc_rad``, ISO 8688-2)."""
    process = text(entry, "process")
    basis = text(entry, "life_basis") or default_life_basis(process or None)
    if basis not in LIFE_BASES:
        raise InputError(f"Taylor entry: life_basis {basis!r} is not one of {', '.join(LIFE_BASES)}")
    if basis == CONTACT_TIME:
        return basis, 1.0, "contact"
    arc = num(entry, "engaged_arc_ref_rad")
    if arc is not None:
        how = "stated"
    elif process in INTERRUPTED_PROCESSES:
        arc, how = number(CALIBRATION, "milling_reference_arc_rad"), "default"
    else:
        return basis, 1.0, "continuous"
    if not 0.0 < arc <= 2.0 * math.pi:
        raise InputError(f"Taylor entry: engaged_arc_ref_deg must lie in (0, 360] deg, got {math.degrees(arc):g}")
    return basis, arc / (2.0 * math.pi), how


def taylor_entries(items) -> tuple[TaylorEntry, ...]:
    out = []
    for i, e in enumerate(items or []):
        c = num(e, "C_m_s")
        n = num(e, "n")
        t_ref = num(e, "T_ref_s")
        vb = num(e, "VB_limit_m")
        if c is None or n is None or vb is None:
            raise InputError(f"tool_life[{i}]: a Taylor entry needs C_m_min, n and VB_limit_mm")
        if t_ref is None:
            raise InputError(f"tool_life[{i}]: a Taylor entry needs its reference life T_ref_min")
        rng = plain(field_of(e, "vc_range_m_s"))
        vc_range = (float(rng[0]), float(rng[1])) if isinstance(rng, (list, tuple)) and len(rng) == 2 else None
        try:
            basis, fraction, how = life_basis_of(e)
        except InputError as error:
            raise InputError(f"tool_life[{i}]: {error}") from None
        out.append(TaylorEntry(text(e, "tool_class"), text(e, "coating"), text(e, "process"), c, n,
                               num(e, "a", 0.0), num(e, "b", 0.0), t_ref, num(e, "feed_ref_m"), num(e, "ap_ref_m"),
                               vb, vc_range, text(e, "src"), i, basis, fraction, how, text(e, "grade")))
    return tuple(out)


def _words(value: str) -> set[str]:
    return {w for w in re.split(r"[^0-9a-z]+", (value or "").lower()) if len(w) > 1}


def grade_score(entry: TaylorEntry, tool_material: "ToolMaterialData | None") -> int:
    """Words of the entry's grade text (``SiAlON``, ``Al2O3-SiC whisker``) found in the tool material's id and
    name: which entry of a tool class describes this tool material (0 without a grade or a tool material)."""
    if tool_material is None or not entry.grade:
        return 0
    known = _words(tool_material.id) | _words(tool_material.name)
    return len(_words(entry.grade) & known)


def select_taylor(entries, tool_class: str, coating: str, process: str,
                  tool_material: "ToolMaterialData | None" = None) -> tuple[TaylorEntry | None, str]:
    """The Taylor entry the calibration uses and how its coating relates to the tool's:
    ``same`` (coating matches), ``substrate`` (an uncoated entry), ``other`` (another coating). Among entries
    of the tool class: coating first, then process, then the entry whose grade names the tool material
    (``grade_score``: a whisker-reinforced alumina tool takes the whisker entry, not a SiAlON one of the same
    class), then file order."""
    same_class = [e for e in entries if e.tool_class == tool_class]
    if not same_class:
        return None, ""

    def rank(e):
        process_score = 2 if e.process == process else (1 if not e.process else 0)
        return process_score, grade_score(e, tool_material), -e.index

    tool_uncoated = normalise(coating or "none") in ("none", "uncoated", "")
    matched = [e for e in same_class if coating_matches(e.coating, coating) == 3 and (tool_uncoated or not e.uncoated)]
    if matched:
        return max(matched, key=rank), ("substrate" if tool_uncoated else "same")
    plain_entries = [e for e in same_class if e.uncoated]
    if plain_entries:
        return max(plain_entries, key=rank), "substrate"
    return max(same_class, key=rank), "other"


# ------------------------------------------------------------------------------------------ tool material
@dataclass(frozen=True)
class ToolMaterialData:
    id: str
    tool_class: str
    density_kg_m3: float
    E_T_K: np.ndarray
    E_Pa: np.ndarray
    poisson: float
    alpha_T_K: np.ndarray
    alpha_1_K: np.ndarray
    k_T_K: np.ndarray
    k_W_mK: np.ndarray
    cp_T_K: np.ndarray
    cp_J_kgK: np.ndarray
    hardness_T_K: np.ndarray
    hardness_Pa: np.ndarray
    trs_Pa: float | None
    kic_Pa_sqrt_m: float | None
    weibull_m: float | None
    grain_m: float | None
    max_service_K: float
    usui: tuple[UsuiEntry, ...] = ()
    name: str = ""                 # the file's descriptive name (grade family; matched against Taylor grades)

    def youngs(self, temp_K):
        return np.interp(temp_K, self.E_T_K, self.E_Pa)

    def expansion(self, temp_K):
        return np.interp(temp_K, self.alpha_T_K, self.alpha_1_K)

    def diffusivity(self, temp_K):
        k = np.interp(temp_K, self.k_T_K, self.k_W_mK)
        cp = np.interp(temp_K, self.cp_T_K, self.cp_J_kgK)
        return k / (self.density_kg_m3 * cp)

    def hardness_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Hardness (K, Pa) as the database gives it: a hot-hardness table, else one value held at every
        temperature (the service limit is a separate check, not an extrapolated softening)."""
        return np.ascontiguousarray(self.hardness_T_K), np.ascontiguousarray(self.hardness_Pa)


def tool_material_data(obj, tool_class: str = "") -> ToolMaterialData:
    m = as_map(obj)
    if m is None:
        raise InputError("wear: the tool has no tool material")
    mid = text(m, "id", "inline")
    cls = text(m, "class", tool_class or "carbide")
    rho = num(m, "density_kg_m3")
    if rho is None:
        raise InputError(f"tool material {mid}: density_kg_m3 missing")
    e_tab = table(m, "youngs_modulus_Pa")
    a_tab = table(m, "expansion_1_K")
    k_tab = table(m, "conductivity_W_mK")
    c_tab = table(m, "specific_heat_J_kgK")
    h_tab = table(m, "hardness_HV", HV_PA)
    for name, value in (("youngs_modulus_GPa", e_tab), ("expansion_1_K", a_tab), ("conductivity_W_mK", k_tab),
                        ("specific_heat_J_kgK", c_tab), ("hardness_HV", h_tab)):
        if value is None:
            raise InputError(f"tool material {mid}: {name} missing")
    nu = num(m, "poisson")
    if nu is None:
        raise InputError(f"tool material {mid}: poisson missing")
    t_max = num(m, "max_service_temp_K")
    if t_max is None:
        raise InputError(f"tool material {mid}: max_service_temp_degC missing")
    return ToolMaterialData(
        id=mid, tool_class=cls, density_kg_m3=rho, E_T_K=e_tab[0], E_Pa=e_tab[1], poisson=nu, alpha_T_K=a_tab[0],
        alpha_1_K=a_tab[1], k_T_K=k_tab[0], k_W_mK=k_tab[1], cp_T_K=c_tab[0], cp_J_kgK=c_tab[1],
        hardness_T_K=h_tab[0], hardness_Pa=h_tab[1], trs_Pa=num(m, "transverse_rupture_Pa"),
        kic_Pa_sqrt_m=num(m, "fracture_toughness_Pa_sqrt_m"), weibull_m=num(m, "weibull_modulus"),
        grain_m=num(m, "grain_size_m"), max_service_K=t_max,
        usui=usui_entries(field_of(m, "usui"), f"tool material {mid}"), name=text(m, "name"))


# ------------------------------------------------------------------------------------------ coating
@dataclass(frozen=True)
class CoatingData:
    id: str
    factor: float                  # Usui-rate factor while the coating covers the contact
    thickness_m: float
    max_service_K: float
    hardness_T_K: np.ndarray
    hardness_Pa: np.ndarray
    process: str = ""
    generic: bool = False          # a coating known only from a Taylor entry's text

    def hardness_table(self) -> tuple[np.ndarray, np.ndarray]:
        return np.ascontiguousarray(self.hardness_T_K), np.ascontiguousarray(self.hardness_Pa)


def coating_data(obj, thickness_m: float | None) -> CoatingData | None:
    """The tool's coating (None for an uncoated tool); thickness from the tool item, else the mean
    of the coating's typical thickness."""
    m = as_map(obj)
    if m is None:
        return None
    cid = text(m, "id", "inline")
    if normalise(cid) in ("none", "uncoated", "") or text(m, "process") == "none":
        return None
    factor = num(m, "wear_rate_factor")
    t_max = num(m, "max_service_temp_K")
    h = table(m, "hardness_HV", HV_PA)
    if factor is None or t_max is None or h is None:
        raise InputError(f"coating {cid}: wear_rate_factor, max_service_temp_degC and hardness_HV are needed")
    if thickness_m is None:
        typical = plain(field_of(m, "typical_thickness_m"))
        if isinstance(typical, (list, tuple)) and len(typical) == 2:
            thickness_m = 0.5 * (float(plain(typical[0])) + float(plain(typical[1])))
        elif typical is not None:
            thickness_m = float(typical)
    if thickness_m is None or thickness_m <= 0.0:
        raise InputError(f"coating {cid}: no thickness (give coating.thickness_um on the tool or typical_thickness_um)")
    return CoatingData(cid, factor, float(thickness_m), t_max, h[0], h[1], text(m, "process"))


def generic_coating(description: str, reference: CoatingData | None, substrate: ToolMaterialData) -> CoatingData:
    """A coating known only from a Taylor entry's text: generic factor, PVD/CVD thickness by keyword;
    hardness and service limit of the tool's own coating when it has one; otherwise the substrate's
    hardness table and service limit (the generic layer then protects only through its Usui factor)."""
    p = get(FLANK)
    words = normalise(description or "")
    thickness = p["generic_cvd_thickness_m"] if "cvd" in words else p["generic_pvd_thickness_m"]
    process = "CVD" if "cvd" in words else "PVD"
    if reference is not None:
        return CoatingData(f"generic:{description}", p["generic_coating_factor"], thickness, reference.max_service_K,
                           reference.hardness_T_K, reference.hardness_Pa, process, True)
    return CoatingData(f"generic:{description}", p["generic_coating_factor"], thickness, substrate.max_service_K,
                       substrate.hardness_T_K, substrate.hardness_Pa, process, True)


# ------------------------------------------------------------------------------------------ work material
@dataclass(frozen=True)
class WorkData:
    id: str
    family: str
    taylor: tuple[TaylorEntry, ...]
    usui: tuple[UsuiEntry, ...]
    skin_type: str
    skin_thickness_m: float
    skin_factor: float
    abrasive_factor: float             # Archard abrasiveness relative to AISI 1045 (composites; 1 otherwise)
    hardening_frac: float              # hardness increase of the work-hardened layer
    layer_m: float
    composite: bool
    notes: list = field(default_factory=list)


def work_data(obj) -> WorkData:
    m = as_map(obj)
    if m is None:
        raise InputError("wear: the project has no workpiece material")
    mid = text(m, "id", "inline")
    family = text(m, "family")
    notes = []
    skin = field_of(m, "skin")
    skin_type = text(skin, "type", "none") if skin is not None else "none"
    thickness = num(skin, "thickness_m", 0.0) if skin is not None else 0.0
    factor = num(skin, "abrasiveness_factor", 1.0) if skin is not None else 1.0
    if skin_type == "none":
        thickness, factor = 0.0, 1.0
    composite = field_of(m, "composite")
    is_composite = family == "composite" or (composite is not None and plain(composite) is not None)
    abrasive = 1.0
    if is_composite:
        value = num(composite, "abrasive_wear_factor") if composite is not None else None
        if value is None:
            notes.append(f"material {mid}: composite without abrasive_wear_factor, the reference steel's (1) used")
        else:
            abrasive = value
    cutting = field_of(m, "cutting")
    hardening = field_of(cutting, "work_hardening") if cutting is not None else None
    increase = num(hardening, "hardness_increase_frac", 0.0) if hardening is not None else 0.0
    layer = num(hardening, "layer_depth_m", 0.0) if hardening is not None else 0.0
    usui = usui_entries(field_of(m, "tool_life_usui"), f"material {mid} tool_life_usui")
    usui = tuple(UsuiEntry(e.A_per_Pa, e.B_K, e.T_lo_K, e.T_hi_K, e.surface, e.work_family or family, e.tool_class,
                           e.coating, e.src, e.origin) for e in usui)
    return WorkData(mid, family, taylor_entries(field_of(m, "tool_life")), usui, skin_type, float(thickness),
                    float(factor), float(abrasive), float(increase), float(layer), bool(is_composite), notes)


__all__ = ["get", "number", "HV_PA", "UsuiEntry", "usui_entry", "usui_entries", "select_usui", "TaylorEntry",
           "taylor_entries", "life_basis_of", "select_taylor", "grade_score", "CUTTING_TIME", "CONTACT_TIME", "INTERRUPTED_PROCESSES", "ToolMaterialData", "tool_material_data",
           "CoatingData", "coating_data", "generic_coating", "WorkData", "work_data", "as_map", "field_of", "num",
           "text", "plain", "table", "FLANK", "CALIBRATION", "FRACTURE", "LIFE", "TOOL_CLASSES_BRITTLE"]
