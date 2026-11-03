"""Work material and tool data in the form the mechanics models use.

:meth:`WorkMaterial.from_db` and :meth:`ToolSide.from_db` read database objects (``SIMap``
views of db/materials, db/tools, db/tool_materials, db/coatings; any mapping with the same SI
keys works, e.g. ``cnctwin.db.simap.wrap(raw_yaml)`` in tests) into plain frozen dataclasses with
NumPy property tables, so the solvers never touch the database layer. Selection rules for the
friction, Kienzle and mechanistic entries of a tool class and coating are here too.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..units import REFERENCE_TEMP_K
from .abi import (MAT_A_PA, MAT_B_PA, MAT_C, MAT_EPS0_1_S, MAT_M, MAT_N, MAT_RHO_KG_M3, MAT_TMELT_K, MAT_TREF_K,
                  N_MAT)

METAL_FAMILIES = ("aluminium", "steel", "tool_steel", "stainless", "titanium", "nickel", "copper", "brass", "bronze",
                  "magnesium", "cast_iron", "zinc", "cobalt")
OXIDE_FAMILIES = ("steel", "tool_steel", "stainless", "titanium", "nickel", "cast_iron", "cobalt")


# ------------------------------------------------------------------------------------------ helpers
def num(value, default=None):
    """SI float of a database leaf (Sourced, Table at 20 degC, number), or ``default``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        inner = getattr(value, "value", value)
        if inner is None and getattr(value, "points_K", None) is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def text(value, default: str = "") -> str:
    if value is None:
        return default
    inner = getattr(value, "value", value)
    return str(inner) if inner is not None else default


def pair(value):
    """(lo, hi) of a range given as [lo, hi], {range: [lo, hi]} or a single value; None if absent."""
    if value is None:
        return None
    inner = getattr(value, "value", value)
    if isinstance(inner, (int, float)) and not isinstance(inner, bool):
        return float(inner), float(inner)
    rng = getattr(value, "range", None) if not isinstance(value, (list, tuple)) else None
    if inner is None and rng is not None:
        inner = rng
    if isinstance(inner, Mapping):
        inner = inner.get("range")
    if isinstance(inner, (list, tuple)) and len(inner) == 2 and all(isinstance(v, (int, float)) for v in inner):
        lo, hi = float(inner[0]), float(inner[1])
        return (lo, hi) if lo <= hi else (hi, lo)
    return None


def table_arrays(value, default: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(T_K, values) arrays of a temperature-dependent property (a Table, or a constant)."""
    points = getattr(value, "points_K", None) if value is not None else None
    values = getattr(value, "values", None) if value is not None else None
    if points is not None and values is not None and not isinstance(values, (int, float)):
        temps = np.ascontiguousarray(points, dtype=np.float64)
        vals = np.ascontiguousarray(values, dtype=np.float64)
        if temps.size and np.all(np.isfinite(temps)) and np.all(np.isfinite(vals)):
            return temps, vals
    number = num(value, default)
    if number is None:
        raise ValueError("missing temperature-dependent property")
    return np.array([REFERENCE_TEMP_K]), np.array([number])


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def coating_matches(entry_coating: str, coating: str) -> int:
    """Match quality of a database entry's coating text for a tool coating id: 3 exact or
    substring match, 2 the entry is for any coating, 1 the entry is uncoated and so is the tool
    or the entry gives no coating, 0 no match."""
    e = normalise(entry_coating or "")
    c = normalise(coating or "none")
    uncoated = c in ("none", "uncoated", "")
    if e in ("any", "all"):
        return 2
    if e in ("", ):
        return 1
    if e in ("none", "uncoated"):
        return 3 if uncoated else 0
    if uncoated:
        return 0
    if e == c or c in e or e in c:
        return 3
    return 0


def _hex_rgb(colour: str) -> tuple[int, int, int]:
    value = colour.strip().lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


# ------------------------------------------------------------------------------------------ entries
@dataclass(frozen=True)
class FrictionEntry:
    tool_class: str
    coating: str
    mu: float
    sticking_ratio: float | None


@dataclass(frozen=True)
class CalibrationConditions:
    """Conditions a measured coefficient set refers to (SI); None where the file is silent."""

    speed_m_s: tuple[float, float] | None = None
    h_m: tuple[float, float] | None = None
    edge_radius_m: float | None = None
    rake_rad: float | None = None
    coolant: str = ""
    work_temp_K: float | None = None
    method: str = ""


@dataclass(frozen=True)
class KienzleEntry:
    tool_class: str
    coating: str
    process: str
    kc11_Pa: float
    mc: float
    kf11_Pa: float | None
    mf: float | None
    kp11_Pa: float | None
    mp: float | None
    h_ref_m: float
    b_ref_m: float
    rake_ref_rad: float | None
    vc_ref_m_s: float | None
    calibration: CalibrationConditions
    src: str

    def cutting_force_per_width(self, h_m: np.ndarray) -> np.ndarray:
        """Kienzle F_c / b = kc11 (b_ref / b) ... per unit width: kc11 h_ref (h / h_ref)^(1 - mc)."""
        return self.kc11_Pa * self.h_ref_m * (np.asarray(h_m) / self.h_ref_m) ** (1.0 - self.mc)

    def feed_force_per_width(self, h_m: np.ndarray) -> np.ndarray | None:
        if self.kf11_Pa is None or self.mf is None:
            return None
        return self.kf11_Pa * self.h_ref_m * (np.asarray(h_m) / self.h_ref_m) ** (1.0 - self.mf)

    def passive_force_per_width(self, h_m: np.ndarray) -> np.ndarray | None:
        if self.kp11_Pa is None or self.mp is None:
            return None
        return self.kp11_Pa * self.h_ref_m * (np.asarray(h_m) / self.h_ref_m) ** (1.0 - self.mp)


@dataclass(frozen=True)
class MechanisticEntry:
    tool_class: str
    coating: str
    process: str
    helix_rad: float | None
    rake_rad: float | None
    Ktc_Pa: float
    Krc_Pa: float
    Kac_Pa: float
    Kte_N_per_m: float
    Kre_N_per_m: float
    Kae_N_per_m: float
    speed_m_s: tuple[float, float] | None
    fz_m: tuple[float, float] | None
    calibration: CalibrationConditions
    src: str


@dataclass(frozen=True)
class Segmentation:
    criterion: str                      # critical_speed | thermal_instability
    speed_onset_m_s: float | None
    temp_onset_K: float | None
    h_onset_m: float | None


@dataclass(frozen=True)
class BueWindow:
    temp_K: tuple[float, float] | None  # interface temperature window (None: speed window only)
    speed_m_s: tuple[float, float]


@dataclass(frozen=True)
class OxideRow:
    temp_K: float
    rgb: tuple[int, int, int]
    name: str


# ------------------------------------------------------------------------------------------ work
@dataclass(frozen=True)
class WorkMaterial:
    """Everything the mechanics models need about a work material (SI)."""

    id: str
    family: str
    iso_group: str
    jc: np.ndarray | None                     # abi.MAT_* vector; None: no flow-stress law (data-only mode)
    rho_kg_m3: float
    cp_T_K: np.ndarray
    cp_J_kgK: np.ndarray
    k_T_K: np.ndarray
    k_W_mK: np.ndarray
    youngs_Pa: float
    poisson: float
    yield_Pa: float | None
    ultimate_Pa: float | None
    fracture_strain: float | None
    solidus_K: float | None
    liquidus_K: float | None
    glass_transition_K: float | None
    melt_smear_K: float | None
    friction: tuple[FrictionEntry, ...]
    kienzle: tuple[KienzleEntry, ...]
    mechanistic: tuple[MechanisticEntry, ...]
    min_chip_thickness_ratio: float | None
    indentation_coeff_N_per_m3: float | None
    indentation_estimated: bool
    tendency: str
    segmentation: Segmentation | None
    bue: BueWindow | None
    breakability: str
    oxide_table: tuple[OxideRow, ...] | None
    dust_mass_fraction: float
    soft_flow_temp_K: float | None = None      # data-only mode: the flow stress falls linearly to 0 here
    notes: dict = field(default_factory=dict)

    @property
    def is_metal(self) -> bool:
        return self.family in METAL_FAMILIES

    @property
    def has_flow_stress(self) -> bool:
        return self.jc is not None

    def cp(self, temp_K: float) -> float:
        return float(np.interp(temp_K, self.cp_T_K, self.cp_J_kgK))

    def conductivity(self, temp_K: float) -> float:
        return float(np.interp(temp_K, self.k_T_K, self.k_W_mK))

    def diffusivity(self, temp_K: float) -> float:
        return self.conductivity(temp_K) / (self.rho_kg_m3 * self.cp(temp_K))

    def friction_for(self, tool_class: str, coating: str) -> FrictionEntry:
        """Best friction entry: same tool class with the best coating match, then any class with
        the best coating match (first in file order on ties)."""
        return _best(self.friction, tool_class, coating, require_class=False)

    @classmethod
    def from_db(cls, m: Mapping) -> "WorkMaterial":
        physical = m["physical"] if "physical" in m else {}
        flow = m.get("flow_stress") if isinstance(m, Mapping) else None
        jc = None
        soft = None
        if isinstance(flow, Mapping) and text(flow.get("model")) == "johnson_cook":
            jc = np.zeros(N_MAT)
            jc[MAT_A_PA] = num(flow.get("A_Pa"))
            jc[MAT_B_PA] = num(flow.get("B_Pa"))
            jc[MAT_N] = num(flow.get("n"))
            jc[MAT_C] = num(flow.get("C"))
            jc[MAT_M] = num(flow.get("m"))
            jc[MAT_EPS0_1_S] = num(flow.get("ref_strain_rate_1_s"))
            jc[MAT_TREF_K] = num(flow.get("T_ref_K"))
            jc[MAT_TMELT_K] = num(flow.get("T_melt_K"))
        elif isinstance(flow, Mapping):
            soft = num(flow.get("T_melt_K")) or num(flow.get("T_soft_K"))
        rho = num(physical.get("density_kg_m3"))
        if jc is not None:
            jc[MAT_RHO_KG_M3] = rho
        cp_T, cp_v = table_arrays(physical.get("specific_heat_J_kgK"))
        k_T, k_v = table_arrays(physical.get("conductivity_W_mK"))
        cutting = m.get("cutting") or {}
        chip = m.get("chip") or {}
        damage = m.get("thermal_damage") or {}
        composite = m.get("composite") or {}
        family = text(m.get("family"))
        melt_smear = num(damage.get("melt_smear_K")) if isinstance(damage, Mapping) else None
        indentation = cutting.get("indentation_coeff_N_per_m3") if isinstance(cutting, Mapping) else None
        src = getattr(indentation, "src", None)
        oxide = None
        rows = chip.get("oxide_colours") if isinstance(chip, Mapping) else None
        rows = getattr(rows, "value", rows)
        if isinstance(rows, Mapping):
            rows = rows.get("table")
        if isinstance(rows, (list, tuple)) and rows:
            oxide = tuple(OxideRow(float(r[0]), _hex_rgb(str(r[1])), str(r[2]) if len(r) > 2 else "")
                          for r in rows)
        dust = num(composite.get("dust_mass_fraction"), None) if isinstance(composite, Mapping) else None
        if dust is None:
            dust = 1.0 if (family == "composite" or text(chip.get("tendency")) == "powder"
                           and family in ("wood", "foam", "composite")) else 0.0
        if soft is None and jc is None:
            soft = (melt_smear or num(physical.get("decomposition_K")) or num(physical.get("melting_K"))
                    or num(physical.get("liquidus_K")))
        return cls(
            id=text(m.get("id")), family=family, iso_group=text(m.get("iso_group")), jc=jc, rho_kg_m3=rho,
            cp_T_K=cp_T, cp_J_kgK=cp_v, k_T_K=k_T, k_W_mK=k_v,
            youngs_Pa=num(physical.get("youngs_modulus_Pa"), 0.0), poisson=num(physical.get("poisson"), 0.3),
            yield_Pa=num(physical.get("yield_Pa")), ultimate_Pa=num(physical.get("ultimate_Pa")),
            fracture_strain=num(physical.get("fracture_strain")),
            solidus_K=num(physical.get("solidus_K")), liquidus_K=num(physical.get("liquidus_K")),
            glass_transition_K=num(physical.get("glass_transition_K")), melt_smear_K=melt_smear,
            friction=tuple(_friction(e) for e in (m.get("friction") or [])),
            kienzle=tuple(_kienzle(e) for e in (cutting.get("kienzle") or [])),
            mechanistic=tuple(_mechanistic(e) for e in (cutting.get("mechanistic") or [])),
            min_chip_thickness_ratio=num(cutting.get("min_chip_thickness_ratio")),
            indentation_coeff_N_per_m3=num(indentation),
            indentation_estimated=(src is None or src == "est"),
            tendency=text(chip.get("tendency"), "continuous"),
            segmentation=_segmentation(chip.get("segmentation")),
            bue=_bue(chip.get("bue")),
            breakability=text(chip.get("breakability"), "medium"),
            oxide_table=oxide, dust_mass_fraction=float(dust), soft_flow_temp_K=soft,
        )


def _best(entries, tool_class: str, coating: str, require_class: bool):
    best = None
    best_score = -1
    for entry in entries:
        same_class = entry.tool_class == tool_class
        if require_class and not same_class:
            continue
        score = (10 if same_class else 0) + coating_matches(entry.coating, coating)
        if score > best_score:
            best = entry
            best_score = score
    if best is None:
        raise LookupError(f"no entry for tool class {tool_class!r}")
    return best


def _friction(e: Mapping) -> FrictionEntry:
    return FrictionEntry(tool_class=text(e.get("tool_class")), coating=text(e.get("coating")),
                         mu=num(e.get("mu_sliding")), sticking_ratio=num(e.get("sticking_ratio")))


def _calibration(c) -> CalibrationConditions:
    if not isinstance(c, Mapping):
        return CalibrationConditions()
    radius = pair(c.get("edge_radius_m"))
    rake = pair(c.get("rake_rad"))
    return CalibrationConditions(
        speed_m_s=pair(c.get("vc_m_s")), h_m=pair(c.get("h_m")) or pair(c.get("f_m")),
        edge_radius_m=None if radius is None else 0.5 * (radius[0] + radius[1]),
        rake_rad=None if rake is None else 0.5 * (rake[0] + rake[1]),
        coolant=text(c.get("coolant")), work_temp_K=num(c.get("work_temp_K")), method=text(c.get("method")))


def _kienzle(e: Mapping) -> KienzleEntry:
    return KienzleEntry(
        tool_class=text(e.get("tool_class")), coating=text(e.get("coating")), process=text(e.get("process"), "any"),
        kc11_Pa=num(e.get("kc11_Pa")), mc=num(e.get("mc")),
        kf11_Pa=num(e.get("kf11_Pa")), mf=num(e.get("mf")), kp11_Pa=num(e.get("kp11_Pa")), mp=num(e.get("mp")),
        h_ref_m=num(e.get("h_ref_m"), 1e-3), b_ref_m=num(e.get("b_ref_m"), 1e-3),
        rake_ref_rad=num(e.get("rake_ref_rad")), vc_ref_m_s=num(e.get("vc_ref_m_s")),
        calibration=_calibration(e.get("calibration")), src=text(e.get("src")))


def _mechanistic(e: Mapping) -> MechanisticEntry:
    return MechanisticEntry(
        tool_class=text(e.get("tool_class")), coating=text(e.get("coating")), process=text(e.get("process"), "any"),
        helix_rad=num(e.get("helix_rad")), rake_rad=num(e.get("rake_rad")),
        Ktc_Pa=num(e.get("Ktc_Pa")), Krc_Pa=num(e.get("Krc_Pa")), Kac_Pa=num(e.get("Kac_Pa")),
        Kte_N_per_m=num(e.get("Kte_N_per_m")), Kre_N_per_m=num(e.get("Kre_N_per_m")),
        Kae_N_per_m=num(e.get("Kae_N_per_m")), speed_m_s=pair(e.get("vc_m_s")), fz_m=pair(e.get("fz_m")),
        calibration=_calibration(e.get("calibration")), src=text(e.get("src")))


def _segmentation(s) -> Segmentation | None:
    if not isinstance(s, Mapping):
        return None
    return Segmentation(criterion=text(s.get("criterion"), "thermal_instability"),
                        speed_onset_m_s=num(s.get("vc_onset_m_s")), temp_onset_K=num(s.get("temp_onset_K")),
                        h_onset_m=num(s.get("h_onset_m")))


def _bue(b) -> BueWindow | None:
    if not isinstance(b, Mapping):
        return None
    speed = pair(b.get("vc_m_s"))
    if speed is None:
        return None
    return BueWindow(temp_K=pair(b.get("interface_temp_K")), speed_m_s=speed)


# ------------------------------------------------------------------------------------------ tool
TOOL_PROCESS = {
    "turning_insert": "turning", "grooving_insert": "turning", "parting_insert": "turning",
    "threading_insert": "turning", "boring_bar": "turning", "form_tool": "turning",
    "twist_drill": "drilling", "indexable_drill": "drilling", "gun_drill": "drilling", "spot_drill": "drilling",
    "centre_drill": "drilling", "reamer": "drilling", "tap_cutting": "tapping", "tap_forming": "tapping",
}


@dataclass(frozen=True)
class ChipBreaker:
    land_m: float
    groove_width_m: float
    groove_depth_m: float
    rake_rad: float | None
    feed_window_m: tuple[float, float] | None
    depth_window_m: tuple[float, float] | None


@dataclass(frozen=True)
class ToolSide:
    """Tool data used by the heat partition, friction selection and chip form models (SI)."""

    tool_class: str
    coating: str
    rho_kg_m3: float
    k_T_K: np.ndarray
    k_W_mK: np.ndarray
    cp_T_K: np.ndarray
    cp_J_kgK: np.ndarray
    coating_resistance_m2K_W: float
    process: str = "milling"            # milling | turning | drilling | tapping
    chip_breaker: ChipBreaker | None = None
    gullet_area_m2: float | None = None
    diameter_m: float | None = None
    corner_radius_m: float = 0.0

    @classmethod
    def from_db(cls, tool: Mapping | None, tool_material: Mapping, coating: Mapping | None = None) -> "ToolSide":
        k_T, k_v = table_arrays(tool_material.get("conductivity_W_mK"))
        c_T, c_v = table_arrays(tool_material.get("specific_heat_J_kgK"))
        coating_id = "none"
        thickness = None
        geometry = {}
        kind = ""
        if isinstance(tool, Mapping):
            coat = tool.get("coating") or {}
            coating_id = text(coat.get("id"), "none") if isinstance(coat, Mapping) else "none"
            thickness = num(coat.get("thickness_m")) if isinstance(coat, Mapping) else None
            geometry = tool.get("geometry") or {}
            kind = text(tool.get("type"))
        resistance = 0.0
        if coating is not None and coating_id not in ("none", ""):
            if thickness is None:
                typical = pair(coating.get("typical_thickness_m"))
                thickness = None if typical is None else 0.5 * (typical[0] + typical[1])
            k_coat = num(coating.get("conductivity_W_mK"))
            if thickness is not None and k_coat:
                resistance = thickness / k_coat
        breaker = None
        insert = geometry.get("insert") if isinstance(geometry, Mapping) else None
        if isinstance(insert, Mapping) and isinstance(insert.get("chip_breaker"), Mapping):
            cb = insert["chip_breaker"]
            window = cb.get("window") or {}
            breaker = ChipBreaker(land_m=num(cb.get("land_m"), 0.0), groove_width_m=num(cb.get("groove_width_m"), 0.0),
                                  groove_depth_m=num(cb.get("groove_depth_m"), 0.0), rake_rad=num(cb.get("rake_rad")),
                                  feed_window_m=pair(window.get("f_m")) if isinstance(window, Mapping) else None,
                                  depth_window_m=pair(window.get("ap_m")) if isinstance(window, Mapping) else None)
        corner = 0.0
        if isinstance(geometry, Mapping):
            corner = num(geometry.get("corner_radius_m"), 0.0) or 0.0
            if isinstance(insert, Mapping):
                corner = num(insert.get("nose_radius_m"), corner) or corner
        return cls(tool_class=text(tool_material.get("class"), "carbide"), coating=coating_id,
                   rho_kg_m3=num(tool_material.get("density_kg_m3")), k_T_K=k_T, k_W_mK=k_v, cp_T_K=c_T,
                   cp_J_kgK=c_v, coating_resistance_m2K_W=resistance, process=TOOL_PROCESS.get(kind, "milling"),
                   chip_breaker=breaker,
                   gullet_area_m2=num(geometry.get("gullet_area_m2")) if isinstance(geometry, Mapping) else None,
                   diameter_m=num(geometry.get("diameter_m")) if isinstance(geometry, Mapping) else None,
                   corner_radius_m=corner)


def flow_stress_fallback(work: WorkMaterial, temp_K: float) -> float:
    """Equivalent flow stress (Pa) of a material without a flow-stress law: the yield strength,
    falling linearly to zero at its softening/melt temperature (data-only mode)."""
    sigma = work.yield_Pa or work.ultimate_Pa or 0.0
    t_soft = work.soft_flow_temp_K
    if t_soft is None or t_soft <= REFERENCE_TEMP_K:
        return sigma
    frac = (t_soft - temp_K) / (t_soft - REFERENCE_TEMP_K)
    return sigma * min(max(frac, 0.0), 1.0)


def homologous(work: WorkMaterial, temp_K: float) -> float:
    if work.jc is None:
        return 0.0
    return (temp_K - work.jc[MAT_TREF_K]) / (work.jc[MAT_TMELT_K] - work.jc[MAT_TREF_K])


__all__ = ["WorkMaterial", "ToolSide", "FrictionEntry", "KienzleEntry", "MechanisticEntry", "CalibrationConditions",
           "Segmentation", "BueWindow", "OxideRow", "ChipBreaker", "num", "pair", "table_arrays", "coating_matches",
           "flow_stress_fallback", "homologous", "METAL_FAMILIES", "OXIDE_FAMILIES"]
