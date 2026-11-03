"""Cutting fluid properties at temperature (docs/models/coolant.md "Fluid").

:class:`Fluid` reads a coolant database object (``ctx.project.coolant.fluid``, a SIMap of
db/coolants/<id>.yaml) and answers the liquid properties at a temperature from its tables, the
vapour properties for film boiling, the saturation temperature under pressure (Clausius-Clapeyron
from the file's boiling point and latent heat), and the friction factor of a tool class / work
family pair. A property the file leaves out is taken from ``db/models/coolant-fluid.yaml`` and
the substitution is listed in :attr:`Fluid.estimated` (reported by the domain).

:func:`air_props` gives dry air at 1 atm (ideal gas, Sutherland viscosity and conductivity): the
carrier of air blasts, MQL and mist whatever the project's fluid is.
"""

from __future__ import annotations

import bisect
import functools
import math
from dataclasses import dataclass, field

import numpy as np

from ..db.models import params as model_params
from ..db.simap import Sourced, Table
from ..units import AIR_CP_20C_J_KGK, REFERENCE_TEMP_K, STANDARD_ATMOSPHERE_PA

WATER_BASED = ("emulsion", "semi_synthetic", "synthetic", "water")
OILS = ("straight_oil", "mql_oil")
GASES = ("air",)


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-fluid")


@dataclass(frozen=True)
class Props:
    """Transport properties of one phase at one temperature (SI)."""

    rho_kg_m3: float
    mu_Pa_s: float
    k_W_mK: float
    cp_J_kgK: float
    sigma_N_per_m: float = 0.0

    @property
    def nu_m2_s(self) -> float:
        return self.mu_Pa_s / self.rho_kg_m3

    @property
    def pr(self) -> float:
        return self.mu_Pa_s * self.cp_J_kgK / self.k_W_mK


def air_props(temp_K: float, pressure_Pa: float = STANDARD_ATMOSPHERE_PA) -> Props:
    """Dry air: ideal-gas density, Sutherland viscosity and conductivity (White 2006)."""
    p = P()
    t = max(float(temp_K), 1.0)
    t0 = p["air_ref_temp_K"]
    mu = p["air_mu_ref_Pa_s"] * (t / t0) ** 1.5 * (t0 + p["air_sutherland_mu_K"]) / (t + p["air_sutherland_mu_K"])
    k = p["air_k_ref_W_mK"] * (t / t0) ** 1.5 * (t0 + p["air_sutherland_k_K"]) / (t + p["air_sutherland_k_K"])
    rho = pressure_Pa / (p["air_gas_constant_J_kgK"] * t)
    return Props(rho, mu, k, AIR_CP_20C_J_KGK, 0.0)


class _Linear:
    """Scalar linear interpolation of a property table, end values held and counted (as
    cnctwin.db.simap.Table, whose ``clamped`` count the report uses)."""

    __slots__ = ("x", "y", "clamped")

    def __init__(self, x, y):
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.clamped = 0

    def __call__(self, t: float) -> float:
        x, y = self.x, self.y
        if t <= x[0]:
            if t < x[0]:
                self.clamped += 1
            return y[0]
        if t >= x[-1]:
            if t > x[-1]:
                self.clamped += 1
            return y[-1]
        k = bisect.bisect_right(x, t) - 1
        return y[k] + (t - x[k]) * (y[k + 1] - y[k]) / (x[k + 1] - x[k])


def _callable(value):
    """A table, a sourced number or a number as a function of temperature (None if absent)."""
    if value is None:
        return None
    table = value.value if isinstance(value, Sourced) else value
    if isinstance(table, Table) and len(table.x) == len(table.values) and len(table.x) >= 1:
        return _Linear(table.x, table.values)
    if callable(value) and not isinstance(value, (int, float)):
        return lambda t, v=value: float(v(float(t)))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return lambda t, n=number: n


def _number(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _attr(obj, name):
    if obj is None:
        return None
    try:
        return getattr(obj, name)
    except AttributeError:
        return None


@dataclass
class Fluid:
    """One cutting fluid (see the module docstring). ``kind`` is the database kind; ``dry`` is the
    absence of a fluid (no project coolant)."""

    id: str
    kind: str
    concentration_frac: float
    density: object
    viscosity: object
    specific_heat: object
    conductivity: object
    sigma_N_per_m: float
    boiling_K: float
    leidenfrost_K: float
    latent_J_kg: float
    friction: list = field(default_factory=list)          # (tool_class, work_family, value)
    lubricity: float = 0.0
    drag_out_m3_per_kg: float = 0.0
    evaporation_m3_s_per_m2: float = 0.0
    mist_loss_fraction: float = 0.0
    dosage_m3_s: float | None = None
    air_pressure_Pa: float | None = None
    supply_pressure_Pa: float | None = None
    typical_flow_m3_s: float | None = None
    estimated: list = field(default_factory=list)

    # ------------------------------------------------------------------ classification
    @property
    def is_gas(self) -> bool:
        return self.kind in GASES

    @property
    def water_based(self) -> bool:
        return self.kind in WATER_BASED

    @property
    def is_mql(self) -> bool:
        return self.kind == "mql_oil"

    @property
    def dry(self) -> bool:
        return self.kind == "dry"

    def clamped_lookups(self) -> dict:
        """Property tables evaluated outside their temperature range (end value held): name -> count."""
        out = {}
        for name in ("density", "viscosity", "specific_heat", "conductivity"):
            count = getattr(getattr(self, name), "clamped", 0)
            if count:
                out[name] = int(count)
        return out

    # ------------------------------------------------------------------ properties
    def props(self, temp_K: float) -> Props:
        """Liquid (or gas, for kind air) properties at ``temp_K`` (memoised: a pure function of T)."""
        t = float(temp_K)
        memo = self.__dict__.setdefault("_memo", {})
        hit = memo.get(t)
        if hit is not None:
            return hit
        if self.dry:
            out = air_props(t)
        else:
            out = Props(self.density(t), self.viscosity(t), self.conductivity(t), self.specific_heat(t),
                        self.sigma_N_per_m)
        if len(memo) > 4096:
            memo.clear()
        memo[t] = out
        return out

    def vapour(self, temp_K: float, pressure_Pa: float = STANDARD_ATMOSPHERE_PA) -> Props:
        """Vapour of the fluid at the film temperature (water vapour for water-based fluids, Incropera
        Table A.4 scaled to the pressure as an ideal gas; an ideal heavy-organic vapour for oils)."""
        p = P()
        t = float(temp_K)
        if self.water_based:
            temps = np.asarray(p["vapour_water_temp_K"], dtype=float)
            rho = float(np.interp(t, temps, p["vapour_water_rho_kg_m3"])) * pressure_Pa / STANDARD_ATMOSPHERE_PA
            return Props(rho, float(np.interp(t, temps, p["vapour_water_mu_Pa_s"])),
                         float(np.interp(t, temps, p["vapour_water_k_W_mK"])),
                         float(np.interp(t, temps, p["vapour_water_cp_J_kgK"])))
        rho = pressure_Pa / (p["oil_vapour_gas_constant_J_kgK"] * max(t, 1.0))
        return Props(rho, p["oil_vapour_mu_Pa_s"], p["oil_vapour_k_W_mK"], p["oil_vapour_cp_J_kgK"])

    def vapour_gas_constant(self) -> float:
        p = P()
        return p["water_vapour_gas_constant_J_kgK"] if self.water_based else p["oil_vapour_gas_constant_J_kgK"]

    def saturation_K(self, pressure_Pa: float = STANDARD_ATMOSPHERE_PA) -> float:
        """Boiling temperature at an absolute pressure: Clausius-Clapeyron through the file's boiling
        point at 1 atm with its latent heat, ln(p/p0) = -(h_fg/R_v)(1/T - 1/T0)."""
        if pressure_Pa <= 0.0 or abs(pressure_Pa - STANDARD_ATMOSPHERE_PA) < 1e-9 * STANDARD_ATMOSPHERE_PA:
            return self.boiling_K
        inv = 1.0 / self.boiling_K - self.vapour_gas_constant() / self.latent_J_kg * math.log(
            pressure_Pa / STANDARD_ATMOSPHERE_PA)
        return 1.0 / inv if inv > 0.0 else self.boiling_K

    def leidenfrost_at_K(self, pressure_Pa: float = STANDARD_ATMOSPHERE_PA) -> float:
        """Film-boiling onset at a pressure: the database value shifted with the saturation temperature
        (the Leidenfrost point rises with pressure, Bernardin & Mudawar 1999)."""
        return self.leidenfrost_K + (self.saturation_K(pressure_Pa) - self.boiling_K)

    def friction_factor(self, tool_class: str, work_family: str) -> tuple[float, str]:
        """(factor, how it was found): the database entry for the pair, else the closest available
        (same work family, same tool class, then the mean of all entries), 1.0 for a fluid without any
        (no lubrication: air)."""
        tool_class = (tool_class or "any").lower()
        work_family = (work_family or "any").lower()
        rows = self.friction
        for want_tool, want_family in ((tool_class, work_family), (tool_class, "any"), ("any", work_family),
                                       ("any", "any")):
            for t, w, v in rows:
                if t == want_tool and w == want_family:
                    return float(v), "database"
        same_family = [v for t, w, v in rows if w == work_family]
        if same_family:
            return float(np.mean(same_family)), f"mean of the {work_family} entries (no {tool_class} entry)"
        same_tool = [v for t, w, v in rows if t == tool_class]
        if same_tool:
            return float(np.mean(same_tool)), f"mean of the {tool_class} entries (no {work_family} entry)"
        if rows:
            return float(np.mean([v for _, _, v in rows])), "mean of all entries (no matching pair)"
        return 1.0, "no friction data (no lubrication)"


def dry_fluid() -> Fluid:
    """No coolant: the 'fluid' is the shop air."""
    return Fluid(id="dry", kind="dry", concentration_frac=0.0, density=None, viscosity=None, specific_heat=None,
                 conductivity=None, sigma_N_per_m=0.0, boiling_K=math.inf, leidenfrost_K=math.inf, latent_J_kg=1.0)


def fluid_from_db(obj) -> Fluid:
    """A :class:`Fluid` from a coolant database object (SIMap or None)."""
    if obj is None:
        return dry_fluid()
    p = P()
    kind = str(_attr(obj, "kind") or "emulsion")
    estimated: list[str] = []
    density = _callable(_attr(obj, "density_kg_m3"))
    viscosity = _callable(_attr(obj, "viscosity_Pa_s"))
    cp = _callable(_attr(obj, "specific_heat_J_kgK"))
    k = _callable(_attr(obj, "conductivity_W_mK"))
    if kind in GASES:
        air = air_props(REFERENCE_TEMP_K)
        density = density or (lambda t: air_props(t).rho_kg_m3)
        viscosity = viscosity or (lambda t: air_props(t).mu_Pa_s)
        cp = cp or (lambda t: air.cp_J_kgK)
        k = k or (lambda t: air_props(t).k_W_mK)
    missing = [n for n, f in (("density", density), ("viscosity", viscosity), ("specific heat", cp),
                              ("conductivity", k)) if f is None]
    if missing:
        from ..core.errors import InputError
        raise InputError(f"coolant {_attr(obj, 'id')}: no {', '.join(missing)} (required by the coolant schema)")
    sigma = _number(_attr(obj, "surface_tension_N_per_m"))
    if sigma is None:
        sigma = p["water_surface_tension_N_per_m"] if kind in WATER_BASED else (
            0.0 if kind in GASES else p["oil_surface_tension_N_per_m"])
        if kind not in GASES:
            estimated.append(f"surface tension {sigma:.4f} N/m (coolant-fluid fallback)")
    boiling = _number(_attr(obj, "boiling_K"))
    if boiling is None and kind not in GASES:
        boiling = p["water_boiling_K"] if kind in WATER_BASED else p["oil_boiling_K"]
        estimated.append(f"boiling point {boiling - 273.15:.0f} degC (coolant-fluid fallback)")
    leiden = _number(_attr(obj, "leidenfrost_K"))
    if leiden is None and kind not in GASES:
        leiden = p["water_leidenfrost_K"] if kind in WATER_BASED else boiling + p["oil_leidenfrost_above_boiling_dK"]
        estimated.append(f"Leidenfrost temperature {leiden - 273.15:.0f} degC (coolant-fluid fallback)")
    heat = _attr(obj, "heat_transfer")
    latent = _number(_attr(heat, "latent_heat_J_kg")) if heat is not None else None
    if latent is None and kind not in GASES:
        latent = p["water_latent_heat_J_kg"] if kind in WATER_BASED else p["oil_latent_heat_J_kg"]
        estimated.append(f"latent heat {latent / 1e3:.0f} kJ/kg (coolant-fluid fallback)")
    rows = []
    for row in _attr(obj, "friction_factor") or []:
        value = _number(_attr(row, "value"))
        if value is None:
            continue
        rows.append((str(_attr(row, "tool_class") or "any").lower(), str(_attr(row, "work_family") or "any").lower(),
                     value))
    conc = _number(_attr(obj, "concentration_frac"), 0.0)
    return Fluid(
        id=str(_attr(obj, "id") or "fluid"), kind=kind, concentration_frac=conc, density=density, viscosity=viscosity,
        specific_heat=cp, conductivity=k, sigma_N_per_m=sigma,
        boiling_K=boiling if boiling is not None else math.inf, leidenfrost_K=leiden if leiden is not None else math.inf,
        latent_J_kg=latent if latent is not None else 1.0, friction=rows,
        lubricity=_number(_attr(obj, "lubricity_index"), 0.0),
        drag_out_m3_per_kg=_number(_attr(obj, "drag_out_m3_per_kg"), 0.0),
        evaporation_m3_s_per_m2=_number(_attr(obj, "evaporation_m3_s_per_m2"), 0.0),
        mist_loss_fraction=_number(_attr(obj, "mist_loss_fraction"), 0.0),
        dosage_m3_s=_number(_attr(obj, "dosage_default_m3_s")),
        air_pressure_Pa=_number(_attr(obj, "min_air_pressure_Pa")),
        supply_pressure_Pa=_number(_attr(obj, "supply_pressure_Pa")),
        typical_flow_m3_s=_number(_attr(obj, "typical_flow_m3_s")),
        estimated=estimated)
