"""Coolant supply: pumps, pipe networks, nozzles, through-spindle path, air blast, MQL and mist
(docs/models/coolant.md "Supply").

* **Pipe friction** (Darcy-Weisbach): dp = (f L/d + K) rho v^2/2 with the Darcy factor f = 64/Re below
  ``laminar_re_hi``, Colebrook-White (Newton on 1/sqrt(f) from Haaland's explicit start) above
  ``turbulent_re_lo``, linear in Re in between; minor losses K per element (Crane TP-410).
* **Nozzles**: converging orifices dp = (1 - beta^4) rho (Q/(C_d A))^2 / 2 (velocity of approach,
  ISO 5167), jet velocity Q/A; coolant holes end in their own velocity head (K = 1 at the exit).
* **Pumps**: radial (flood) pumps p(Q) = p0 [1 - (1 - p_r/p0)(Q/Q_r)^2] through the rated point with
  the shut-off ratio p0/p_r (Guelich 2014) and shaft power linear in Q from the shut-off fraction;
  displacement (through-spindle, high-pressure) pumps deliver their rated flow until the system
  needs more than the rated pressure, then the relief valve holds the rated pressure and bypasses the
  rest (Karassik 2008); a ``pressure_flow_bar_l_min`` curve of the machine file replaces either.
  Shaft power ends up as heat in the fluid (pump losses, pipe friction, jet impact, relief throttling).
* **Network**: a common line (with static head) feeds parallel branches; the manifold pressure is
  found by Brent's method with every branch flow inverted by Brent's method (fixed tolerances: the
  same inputs always give the same iterations).
* **Air** (air blast, MQL carrier, mist): isentropic orifice flow, choked above the critical ratio;
  when the orifices could pass more than the supply delivers the nozzle pressure falls until they
  pass the supply flow; the jet is replaced by its fully expanded notional nozzle (Birch et al. 1984).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

from scipy.optimize import brentq

from ..db.models import params as model_params
from ..units import STANDARD_ATMOSPHERE_PA, STANDARD_GRAVITY_M_S2, AIR_CP_20C_J_KGK
from . import fluid as F

LIQUID = "liquid"
GAS = "gas"
AEROSOL = "aerosol"

THROUGH_KINDS = ("through_spindle", "high_pressure", "through_tool")
DISPLACEMENT_KINDS = THROUGH_KINDS + ("mql",)


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-supply")


# ============================================================================ friction factor

def colebrook(re: float, rel_rough: float, tol: float, max_iter: int) -> float:
    """Darcy factor of turbulent flow, Colebrook (1939) solved by Newton on x = 1/sqrt(f) from
    Haaland's (1983) explicit value. Stops when |dx| <= tol x (or after max_iter)."""
    p = P()
    a, b = p["colebrook_a"], p["colebrook_b"]
    x = -p["haaland_a"] * math.log10((rel_rough / b) ** p["haaland_n"] + p["haaland_b"] / re)
    ln10 = math.log(10.0)
    for _ in range(max_iter):
        arg = rel_rough / b + a * x / re
        g = x + 2.0 * math.log10(arg)
        dg = 1.0 + 2.0 / ln10 * (a / re) / arg
        step = g / dg
        x -= step
        if abs(step) <= tol * abs(x):
            break
    return 1.0 / (x * x)


def darcy(re: float, rel_rough: float, tol: float, max_iter: int) -> float:
    """Darcy friction factor at a Reynolds number (0 without flow)."""
    if re <= 0.0:
        return 0.0
    p = P()
    lo, hi = p["laminar_re_hi"], p["turbulent_re_lo"]
    if re <= lo:
        return p["laminar_friction_constant"] / re
    if re >= hi:
        return colebrook(re, rel_rough, tol, max_iter)
    f_lo = p["laminar_friction_constant"] / lo
    f_hi = colebrook(hi, rel_rough, tol, max_iter)
    return f_lo + (re - lo) / (hi - lo) * (f_hi - f_lo)


# ============================================================================ elements

@dataclass(frozen=True)
class Pipe:
    """A straight passage (``count`` identical ones in parallel share the flow) with a minor loss K
    on its own velocity."""

    name: str
    diameter_m: float
    length_m: float
    roughness_m: float
    k_minor: float = 0.0
    count: int = 1

    @property
    def area_m2(self) -> float:
        return self.count * math.pi * self.diameter_m ** 2 / 4.0

    def dp(self, flow_m3_s: float, props: F.Props, tol: float, max_iter: int) -> float:
        if flow_m3_s <= 0.0:
            return 0.0
        v = flow_m3_s / self.area_m2
        re = props.rho_kg_m3 * v * self.diameter_m / props.mu_Pa_s
        f = darcy(re, self.roughness_m / self.diameter_m, tol, max_iter)
        return (f * self.length_m / self.diameter_m + self.k_minor) * 0.5 * props.rho_kg_m3 * v * v


@dataclass(frozen=True)
class Orifice:
    """A converging nozzle discharging to ambient: includes the jet's kinetic energy."""

    name: str
    diameter_m: float
    cd: float
    approach_diameter_m: float
    count: int = 1

    @property
    def area_m2(self) -> float:
        return self.count * math.pi * self.diameter_m ** 2 / 4.0

    def dp(self, flow_m3_s: float, props: F.Props, tol: float, max_iter: int) -> float:
        if flow_m3_s <= 0.0:
            return 0.0
        beta = min(self.diameter_m / self.approach_diameter_m, 0.99) if self.approach_diameter_m > 0.0 else 0.0
        v = flow_m3_s / (self.cd * self.area_m2)
        return (1.0 - beta ** 4) * 0.5 * props.rho_kg_m3 * v * v

    def exit_speed(self, flow_m3_s: float) -> float:
        return flow_m3_s / self.area_m2


@dataclass(frozen=True)
class OpenEnd:
    """The end of a passage discharging a jet: its velocity head (K = 1)."""

    name: str
    area_m2: float

    def dp(self, flow_m3_s: float, props: F.Props, tol: float, max_iter: int) -> float:
        v = flow_m3_s / self.area_m2
        return 0.5 * props.rho_kg_m3 * v * v

    def exit_speed(self, flow_m3_s: float) -> float:
        return flow_m3_s / self.area_m2


@dataclass(frozen=True)
class Branch:
    """Elements in series from the manifold to the jet(s); the last element is the exit."""

    outlet: str
    elements: tuple

    def dp(self, flow_m3_s: float, props: F.Props, tol: float, max_iter: int) -> float:
        return sum(e.dp(flow_m3_s, props, tol, max_iter) for e in self.elements)

    @property
    def exit(self):
        return self.elements[-1]

    def flow_at(self, dp_Pa: float, props: F.Props, tol: float, max_iter: int) -> float:
        """Flow through the branch at a pressure difference (inverse of dp, monotone)."""
        if dp_Pa <= 0.0:
            return 0.0
        exit_area = getattr(self.exit, "area_m2", 0.0)
        cd = getattr(self.exit, "cd", 1.0)
        upper = 2.0 * max(cd, 1.0) * exit_area * math.sqrt(2.0 * dp_Pa / props.rho_kg_m3) + 1e-12
        while self.dp(upper, props, tol, max_iter) < dp_Pa:
            upper *= 2.0
        return brentq(lambda q: self.dp(q, props, tol, max_iter) - dp_Pa, 0.0, upper, xtol=1e-30, rtol=tol,
                      maxiter=max_iter)


# ============================================================================ pumps

@dataclass(frozen=True)
class PumpModel:
    id: str
    kind: str                       # database kind (flood, through_spindle, ...)
    displacement: bool
    rated_Pa: float
    rated_m3_s: float
    curve: tuple = ()               # ((p_Pa, Q_m3_s), ...) ascending in p
    efficiency: float = 0.35
    shutoff_ratio: float = 1.25
    shutoff_power_ratio: float = 0.5

    @property
    def shutoff_Pa(self) -> float:
        if self.curve:
            return self.curve[-1][0] if self.displacement else max(p for p, _ in self.curve)
        return self.rated_Pa if self.displacement else self.shutoff_ratio * self.rated_Pa

    def pressure(self, flow_m3_s: float) -> float:
        """Outlet gauge pressure at a flow (radial and curve pumps)."""
        if self.curve:
            pts = sorted(((q, p) for p, q in self.curve), key=lambda r: r[0])
            qs = [r[0] for r in pts]
            ps = [r[1] for r in pts]
            if flow_m3_s <= qs[0]:
                return ps[0]
            if flow_m3_s >= qs[-1]:
                if len(qs) < 2 or qs[-1] == qs[-2]:
                    return max(ps[-1], 0.0)
                slope = (ps[-1] - ps[-2]) / (qs[-1] - qs[-2])
                return max(ps[-1] + slope * (flow_m3_s - qs[-1]), 0.0)
            for i in range(1, len(qs)):
                if flow_m3_s <= qs[i]:
                    t = (flow_m3_s - qs[i - 1]) / (qs[i] - qs[i - 1]) if qs[i] > qs[i - 1] else 0.0
                    return ps[i - 1] + t * (ps[i] - ps[i - 1])
        p0 = self.shutoff_Pa
        ratio = (flow_m3_s / self.rated_m3_s) ** 2 if self.rated_m3_s > 0.0 else 0.0
        return max(p0 * (1.0 - (1.0 - self.rated_Pa / p0) * ratio), 0.0)

    def shaft_power(self, pressure_Pa: float, pumped_m3_s: float) -> float:
        """Shaft power (W): displacement pumps p Q / eta of the pumped (incl. bypassed) flow; radial
        pumps linear in Q from the shut-off fraction of the rated power."""
        if self.displacement:
            return pressure_Pa * pumped_m3_s / self.efficiency
        rated = self.rated_Pa * self.rated_m3_s / self.efficiency
        frac = pumped_m3_s / self.rated_m3_s if self.rated_m3_s > 0.0 else 0.0
        return rated * (self.shutoff_power_ratio + (1.0 - self.shutoff_power_ratio) * frac)


@dataclass
class Circuit:
    """Operating point of one pump with its line and open branches."""

    pump_id: str
    pressure_Pa: float = 0.0        # pump outlet (gauge)
    manifold_Pa: float = 0.0
    flow_m3_s: float = 0.0          # delivered to the outlets
    bypass_m3_s: float = 0.0        # through the relief valve back to the tank
    shaft_power_W: float = 0.0
    relief_open: bool = False
    branch_flow_m3_s: dict = field(default_factory=dict)    # outlet -> flow
    branch_speed_m_s: dict = field(default_factory=dict)    # outlet -> exit speed


def solve_circuit(pump: PumpModel, line: tuple, branches: list[Branch], props: F.Props, static_Pa: float,
                  tol: float, max_iter: int) -> Circuit:
    """Operating point of a pump feeding parallel branches through a common line."""
    out = Circuit(pump.id)

    def line_dp(q):
        return sum(e.dp(q, props, tol, max_iter) for e in line) + static_Pa

    def branch_total(pm):
        return sum(b.flow_at(pm, props, tol, max_iter) for b in branches)

    if not branches:
        if pump.displacement:
            out.pressure_Pa = pump.shutoff_Pa
            out.bypass_m3_s = pump.rated_m3_s
            out.relief_open = True
            out.shaft_power_W = pump.shaft_power(out.pressure_Pa, pump.rated_m3_s)
        else:
            out.pressure_Pa = pump.shutoff_Pa
            out.shaft_power_W = pump.shaft_power(out.pressure_Pa, 0.0)
        return out
    if pump.displacement and not pump.curve:
        demand = line_dp(pump.rated_m3_s)
        pm = _manifold_for_flow(branches, pump.rated_m3_s, props, tol, max_iter)
        if pm + demand <= pump.rated_Pa:
            out.pressure_Pa = pm + demand
            out.manifold_Pa = pm
            out.flow_m3_s = pump.rated_m3_s
        else:
            # relief valve holds the rated pressure; the outlets take what that pressure drives
            def resid(q):
                return line_dp(q) + _manifold_for_flow(branches, q, props, tol, max_iter) - pump.rated_Pa
            q = brentq(resid, 0.0, pump.rated_m3_s, xtol=1e-30, rtol=tol, maxiter=max_iter) \
                if resid(0.0) < 0.0 else 0.0
            out.pressure_Pa = pump.rated_Pa
            out.flow_m3_s = q
            out.manifold_Pa = max(pump.rated_Pa - line_dp(q), 0.0)
            out.bypass_m3_s = pump.rated_m3_s - q
            out.relief_open = True
        out.shaft_power_W = pump.shaft_power(out.pressure_Pa, out.flow_m3_s + out.bypass_m3_s)
    else:
        p0 = pump.shutoff_Pa
        if p0 <= static_Pa:
            out.pressure_Pa = p0
            out.shaft_power_W = pump.shaft_power(p0, 0.0)
            return out

        def resid(pm):
            q = branch_total(pm)
            return pump.pressure(q) - line_dp(q) - pm

        pm = brentq(resid, 0.0, p0, xtol=1e-30, rtol=tol, maxiter=max_iter)
        q = branch_total(pm)
        out.manifold_Pa = pm
        out.flow_m3_s = q
        out.pressure_Pa = pump.pressure(q)
        if pump.displacement:
            out.shaft_power_W = pump.shaft_power(out.pressure_Pa, q)
        else:
            out.shaft_power_W = pump.shaft_power(out.pressure_Pa, q)
    for b in branches:
        qb = b.flow_at(out.manifold_Pa, props, tol, max_iter)
        out.branch_flow_m3_s[b.outlet] = qb
        out.branch_speed_m_s[b.outlet] = b.exit.exit_speed(qb) if qb > 0.0 else 0.0
    return out


def _manifold_for_flow(branches, flow_m3_s, props, tol, max_iter) -> float:
    """Manifold pressure at which the parallel branches take a total flow."""
    if flow_m3_s <= 0.0:
        return 0.0
    upper = max(b.dp(flow_m3_s, props, tol, max_iter) for b in branches) + 1.0
    return brentq(lambda pm: sum(b.flow_at(pm, props, tol, max_iter) for b in branches) - flow_m3_s, 0.0, upper,
                  xtol=1e-30, rtol=tol, maxiter=max_iter)


# ============================================================================ air (compressible)

@dataclass(frozen=True)
class GasJet:
    """A compressed-air jet at its fully expanded (notional) state."""

    mass_kg_s: float
    stagnation_Pa: float            # absolute
    speed_m_s: float
    diameter_m: float
    temp_K: float                   # static temperature of the expanded jet
    rho_kg_m3: float
    choked: bool


def _mass_flux_factor(gamma: float, pr: float) -> float:
    """m / (C_d A p0 / sqrt(R T0)) for an isentropic nozzle at back-pressure ratio pr = p_a/p0."""
    crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    if pr <= crit:
        return math.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return math.sqrt(2.0 * gamma / (gamma - 1.0) * (pr ** (2.0 / gamma) - pr ** ((gamma + 1.0) / gamma)))


def air_outlets(area_m2: float, supply_gauge_Pa: float, free_flow_m3_s: float, temp_K: float,
                tol: float, max_iter: int) -> GasJet:
    """Air from orifices of total area ``area_m2`` fed with ``free_flow_m3_s`` (free air at the shop
    temperature) at a supply gauge pressure: the orifices pass the smaller of their capacity at the
    supply pressure and the supply flow (the nozzle pressure then falls until they pass it)."""
    fp = F.P()
    gamma = fp["air_gamma"]
    R = fp["air_gas_constant_J_kgK"]
    cd = P()["air_nozzle_discharge_coefficient"]
    pa = STANDARD_ATMOSPHERE_PA
    p_supply = supply_gauge_Pa + pa
    supply_mass = free_flow_m3_s * pa / (R * temp_K)

    def capacity(p0):
        return cd * area_m2 * p0 / math.sqrt(R * temp_K) * _mass_flux_factor(gamma, pa / p0)

    if area_m2 <= 0.0 or supply_mass <= 0.0 or p_supply <= pa:
        return GasJet(0.0, pa, 0.0, 0.0, temp_K, pa / (R * temp_K), False)
    if capacity(p_supply) <= supply_mass:
        p0 = p_supply
        mass = capacity(p0)
    else:
        p0 = brentq(lambda p: capacity(p) - supply_mass, pa * (1.0 + 1e-12), p_supply, xtol=1e-30, rtol=tol,
                    maxiter=max_iter)
        mass = supply_mass
    crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    ratio = pa / p0
    t_e = temp_K * ratio ** ((gamma - 1.0) / gamma)
    cp = AIR_CP_20C_J_KGK
    u_e = math.sqrt(max(2.0 * cp * (temp_K - t_e), 0.0))
    rho_e = pa / (R * t_e)
    d_e = math.sqrt(4.0 * mass / (math.pi * rho_e * u_e)) if u_e > 0.0 else 0.0
    return GasJet(mass, p0, u_e, d_e, t_e, rho_e, ratio <= crit)


def _free_air(mass_kg_s: float, ambient_K: float) -> float:
    """Free-air volume flow (shop pressure and temperature) of an air mass flow."""
    return mass_kg_s * F.P()["air_gas_constant_J_kgK"] * ambient_K / STANDARD_ATMOSPHERE_PA


# ============================================================================ outlets and the supply

@dataclass
class Outlet:
    """One jet source of the step (a nozzle orifice group or a tool's coolant holes)."""

    jet_id: str                     # bus id: nozzle id, '<nozzle>#<k>' for holes
    nozzle_id: str                  # machine nozzle id ('' for a pump feeding the tool directly)
    pump_id: str
    kind: str                       # bus Jet.kind
    phase: str                      # liquid | gas | aerosol
    flow_m3_s: float                # volumetric flow at the jet state (liquid; expanded gas)
    speed_m_s: float
    diameter_m: float               # jet diameter at the exit (notional for gas)
    mass_kg_s: float
    liquid_m3_s: float = 0.0        # liquid carried (aerosol: oil or water droplets; liquid jets = flow)
    rho_kg_m3: float = 0.0          # density of the jet (liquid, or expanded gas)
    through_tool: bool = False
    hole: int = -1
    temp_K: float = 0.0


@dataclass
class SupplyResult:
    outlets: list = field(default_factory=list)
    circuits: dict = field(default_factory=dict)        # pump id -> Circuit
    air_free_m3_s: float = 0.0                           # free-air consumption (blast, MQL carrier, mist)
    mql_oil_m3_s: float = 0.0
    pumped_m3_s: float = 0.0                             # liquid pumped from the tank (incl. bypass)
    shaft_power_W: float = 0.0
    notes: list = field(default_factory=list)            # (key, text) of conditions worth a trace event

    @property
    def through_flow_m3_s(self) -> float:
        return sum(o.flow_m3_s for o in self.outlets if o.through_tool and o.phase == LIQUID)

    @property
    def through_speed_m_s(self) -> float:
        flows = [(o.flow_m3_s, o.speed_m_s) for o in self.outlets if o.through_tool]
        total = sum(q for q, _ in flows)
        return sum(q * v for q, v in flows) / total if total > 0.0 else 0.0


def _num(obj, name, default=None):
    value = getattr(obj, name, None) if obj is not None else None
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(obj, name, default=""):
    value = getattr(obj, name, None) if obj is not None else None
    return str(value) if value is not None else default


@dataclass(frozen=True)
class NozzleSpec:
    id: str
    kind: str
    pump: str
    link: str
    position_m: tuple
    aim: tuple
    diameter_m: float
    count: int
    spread_rad: float | None
    programmable_positions: int = 0
    programmable_step_rad: float = 0.0
    programmable_code: str = ""


class SupplyModel:
    """The machine's coolant system with the project's fluid (built once; :meth:`solve` per state)."""

    def __init__(self, machine, fluid: F.Fluid, tol: float, max_iter: int):
        self.fluid = fluid
        self.tol = tol
        self.max_iter = max_iter
        coolant = getattr(machine, "coolant", None) if machine is not None else None
        self.pumps: dict[str, PumpModel] = {}
        self.nozzles: list[NozzleSpec] = []
        self.air_flow_m3_s = 0.0
        self.air_pressure_Pa = 0.0
        self.tank_m3 = _num(coolant, "tank_m3", 0.0) or 0.0
        self.chiller = getattr(coolant, "chiller", None) if coolant is not None else None
        p = P()
        for pump in (getattr(coolant, "pumps", None) or []) if coolant is not None else []:
            kind = _text(pump, "kind", "flood")
            curve = getattr(pump, "pressure_flow_bar_l_min", None) or getattr(pump, "pressure_flow_Pa_m3_s", None)
            rows = tuple((float(r[0]), float(r[1])) for r in curve) if curve else ()
            disp = kind in DISPLACEMENT_KINDS
            self.pumps[_text(pump, "id")] = PumpModel(
                id=_text(pump, "id"), kind=kind, displacement=disp, rated_Pa=_num(pump, "pressure_Pa", 0.0),
                rated_m3_s=_num(pump, "flow_m3_s", 0.0), curve=tuple(sorted(rows)),
                efficiency=p["pump_efficiency_displacement"] if disp else p["pump_efficiency_centrifugal"],
                shutoff_ratio=p["centrifugal_shutoff_ratio"], shutoff_power_ratio=p["centrifugal_shutoff_power_ratio"])
        for nozzle in (getattr(coolant, "nozzles", None) or []) if coolant is not None else []:
            prog = getattr(nozzle, "programmable", None)
            aim = getattr(nozzle, "aim", None)
            if aim is None and prog is not None:
                aim = getattr(prog, "home_aim", None)
            position = getattr(nozzle, "position_m", None)
            self.nozzles.append(NozzleSpec(
                id=_text(nozzle, "id"), kind=_text(nozzle, "kind", "flood"), pump=_text(nozzle, "pump"),
                link=_text(nozzle, "attached_to"),
                position_m=tuple(float(v) for v in position) if position is not None else (0.0, 0.0, 0.0),
                aim=tuple(float(v) for v in aim) if aim is not None else (0.0, 0.0, -1.0),
                diameter_m=_num(nozzle, "diameter_m", p["default_nozzle_diameter_m"]),
                count=int(_num(nozzle, "count", 1) or 1), spread_rad=_num(nozzle, "spread_rad"),
                programmable_positions=int(_num(prog, "positions", 0) or 0) if prog is not None else 0,
                programmable_step_rad=_num(prog, "step_rad", 0.0) if prog is not None else 0.0,
                programmable_code=_text(prog, "code") if prog is not None else ""))
        blast = getattr(coolant, "air_blast", None) if coolant is not None else None
        if blast is not None:
            self.air_flow_m3_s = _num(blast, "flow_m3_s", 0.0) or 0.0
            self.air_pressure_Pa = _num(blast, "pressure_Pa", 0.0) or 0.0
        if fluid.is_gas:
            self.air_flow_m3_s = self.air_flow_m3_s or (fluid.typical_flow_m3_s or 0.0)
            self.air_pressure_Pa = self.air_pressure_Pa or (fluid.supply_pressure_Pa or 0.0)

    # ------------------------------------------------------------------ helpers
    def nozzle(self, nozzle_id: str) -> NozzleSpec | None:
        for n in self.nozzles:
            if n.id == nozzle_id:
                return n
        return None

    def programmable(self) -> list[NozzleSpec]:
        return [n for n in self.nozzles if n.programmable_positions > 0]

    def _flood_branch(self, nozzle: NozzleSpec) -> Branch:
        p = P()
        hose = Pipe(f"{nozzle.id}:hose", p["flood_branch_diameter_m"], p["flood_branch_length_m"],
                    p["hose_roughness_m"], p["k_flood_branch"])
        tip = Orifice(f"{nozzle.id}:orifice", nozzle.diameter_m, p["nozzle_discharge_coefficient"],
                      p["flood_branch_diameter_m"], nozzle.count)
        return Branch(nozzle.id, (hose, tip))

    def _through_branch(self, outlet: str, tool) -> Branch:
        """Rotary union, drawbar bore, holder passage, contraction into the holes, holes, exit."""
        p = P()
        union = Pipe("rotary_union", p["rotary_union_diameter_m"], 0.0, p["hose_roughness_m"], p["k_rotary_union"])
        drawbar = Pipe("drawbar_bore", p["drawbar_bore_diameter_m"], p["drawbar_bore_length_m"], p["hose_roughness_m"],
                       p["k_contraction"] * (1.0 - min((p["rotary_union_diameter_m"]
                                                        / p["drawbar_bore_diameter_m"]) ** 2, 1.0)))
        holder_d = p["holder_passage_diameter_m"]
        holder = Pipe("holder_passage", holder_d, max(tool.holder_length_m, 0.0), p["hose_roughness_m"], p["k_holder"])
        n = tool.hole_count
        d = tool.hole_diameter_m
        beta2 = min(n * d * d / (holder_d * holder_d), 1.0)
        holes = Pipe("coolant_holes", d, tool.hole_length_m, p["hole_roughness_m"],
                     p["k_contraction"] * (1.0 - beta2), n)
        end = OpenEnd("hole_exit", holes.area_m2)
        return Branch(outlet, (union, drawbar, holder, holes, end))

    def _line(self, pump: PumpModel) -> tuple:
        p = P()
        if pump.kind in THROUGH_KINDS:
            return (Pipe(f"{pump.id}:line", p["tsc_line_diameter_m"], p["tsc_line_length_m"], p["hose_roughness_m"],
                         p["k_tsc_line"]),)
        return (Pipe(f"{pump.id}:line", p["flood_line_diameter_m"], p["flood_line_length_m"], p["hose_roughness_m"],
                     p["k_flood_line"]),)

    # ------------------------------------------------------------------ solve
    def solve(self, pumps_on: dict, air_on: bool, blocked: tuple, tool, temp_K: float,
              ambient_K: float) -> SupplyResult:
        """Outlets of one coolant state. ``tool`` is the ToolShape in the spindle (None: none);
        ``blocked`` the nozzle ids chips block (sorted tuple)."""
        res = SupplyResult()
        fluid = self.fluid
        if fluid.dry:
            return res
        on = sorted(pid for pid, state in pumps_on.items() if state and pid in self.pumps)
        if fluid.is_gas or fluid.is_mql:
            self._solve_gas_fluid(res, on, air_on, blocked, tool, ambient_K)
            return res
        props = fluid.props(temp_K)
        static = props.rho_kg_m3 * STANDARD_GRAVITY_M_S2 * P()["static_head_m"]
        for pid in on:
            pump = self.pumps[pid]
            if pump.kind in ("mist", "mql"):
                self._mist(res, pump, blocked, ambient_K)
                continue
            branches, outlets = self._branches(pump, blocked, tool, res)
            circuit = solve_circuit(pump, self._line(pump), branches, props, static, self.tol, self.max_iter)
            res.circuits[pid] = circuit
            res.pumped_m3_s += circuit.flow_m3_s + circuit.bypass_m3_s
            res.shaft_power_W += circuit.shaft_power_W
            if circuit.relief_open and branches:
                res.notes.append((f"relief:{pid}", f"pump {pid}: relief valve open, {circuit.flow_m3_s * 6e4:.2f} "
                                                   f"of {pump.rated_m3_s * 6e4:.2f} l/min reach the outlets"))
            for branch in branches:
                q = circuit.branch_flow_m3_s[branch.outlet]
                info = outlets[branch.outlet]
                speed = circuit.branch_speed_m_s[branch.outlet]
                self._liquid_outlets(res, info, pump, q, speed, props, tool, temp_K)
        if air_on and self.air_flow_m3_s > 0.0:
            self._air_blast(res, blocked, ambient_K)
        return res

    def _branches(self, pump: PumpModel, blocked: tuple, tool, res: SupplyResult):
        branches: list[Branch] = []
        outlets: dict[str, tuple] = {}
        fed = [n for n in self.nozzles if n.pump == pump.id]
        through = [n for n in fed if n.kind in THROUGH_KINDS] or (
            [] if fed else ([NozzleSpec("", pump.kind, pump.id, "", (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.0, 1, None)]
                            if pump.kind in THROUGH_KINDS else []))
        for nozzle in fed:
            if nozzle.kind in THROUGH_KINDS:
                continue
            if nozzle.id in blocked:
                res.notes.append((f"blocked:{nozzle.id}", f"nozzle {nozzle.id} is blocked by chips: no flow"))
                continue
            branches.append(self._flood_branch(nozzle))
            outlets[nozzle.id] = ("nozzle", nozzle)
        for nozzle in through:
            key = nozzle.id or pump.id
            if tool is None or tool.hole_count <= 0:
                number = getattr(tool, "number", None) if tool is not None else None
                res.notes.append((f"no_holes:{key}:{number}",
                                  f"through-tool coolant ({pump.id}) on, but the tool in the spindle "
                                  + (f"(T{number}) has no coolant holes" if number is not None else "is missing")
                                  + ": no flow"))
                continue
            branches.append(self._through_branch(key, tool))
            outlets[key] = ("holes", nozzle)
        return branches, outlets

    def _liquid_outlets(self, res, info, pump, flow, speed, props, tool, temp_K) -> None:
        kind_tag, nozzle = info
        if flow <= 0.0:
            return
        if kind_tag == "holes":
            n = tool.hole_count
            jet_kind = pump.kind if pump.kind in THROUGH_KINDS else "through_spindle"
            for k in range(n):
                q = flow / n
                res.outlets.append(Outlet(f"{nozzle.id or pump.id}#{k}", nozzle.id, pump.id, jet_kind, LIQUID, q,
                                          speed, tool.hole_diameter_m, q * props.rho_kg_m3, q, props.rho_kg_m3, True, k,
                                          temp_K))
            return
        count = max(nozzle.count, 1)
        for k in range(count):
            q = flow / count
            jet_id = nozzle.id if count == 1 else f"{nozzle.id}#{k}"
            kind = "high_pressure" if pump.kind == "high_pressure" else (nozzle.kind if nozzle.kind != "through_tool"
                                                                         else "flood")
            res.outlets.append(Outlet(jet_id, nozzle.id, pump.id, kind, LIQUID, q, speed, nozzle.diameter_m,
                                      q * props.rho_kg_m3, q, props.rho_kg_m3, False, -1, temp_K))

    def _mist(self, res, pump, blocked, ambient_K) -> None:
        """A mist (or MQL) pump: an air-atomised spray from its nozzles."""
        p = P()
        nozzles = [n for n in self.nozzles if n.pump == pump.id and n.id not in blocked]
        if not nozzles:
            return
        liquid = pump.rated_m3_s if pump.rated_m3_s > 0.0 else p["mist_liquid_m3_s"]
        air = p["mist_air_m3_s"] if pump.kind == "mist" else p["mql_air_m3_s"]
        pressure = self.air_pressure_Pa if self.air_pressure_Pa > 0.0 else p["mql_air_pressure_Pa"]
        for nozzle in nozzles:
            area = max(nozzle.count, 1) * math.pi * nozzle.diameter_m ** 2 / 4.0
            gas = air_outlets(area, pressure, air, ambient_K, self.tol, self.max_iter)
            res.air_free_m3_s += _free_air(gas.mass_kg_s, ambient_K)
            q_liq = liquid / len(nozzles)
            res.outlets.append(Outlet(nozzle.id, nozzle.id, pump.id, "mist" if pump.kind == "mist" else "mql", AEROSOL,
                                      gas.mass_kg_s / gas.rho_kg_m3 if gas.rho_kg_m3 > 0 else 0.0, gas.speed_m_s,
                                      gas.diameter_m, gas.mass_kg_s, q_liq, gas.rho_kg_m3, False, -1, gas.temp_K))
        res.pumped_m3_s += liquid

    def _air_blast(self, res, blocked, ambient_K) -> None:
        nozzles = [n for n in self.nozzles if n.kind == "air" and n.id not in blocked] or \
                  [n for n in self.nozzles if n.kind == "flood" and n.id not in blocked]
        self._gas_from(res, nozzles, "air", self.air_flow_m3_s, self.air_pressure_Pa, 0.0, ambient_K, None, "air")

    def _gas_from(self, res, nozzles, pump_id, free_m3_s, gauge_Pa, liquid_m3_s, ambient_K, tool, kind) -> None:
        """Gas or aerosol jets from orifices (nozzles, or the tool's holes when ``tool`` is given)."""
        if tool is not None:
            area = tool.hole_count * math.pi * tool.hole_diameter_m ** 2 / 4.0
            gas = air_outlets(area, gauge_Pa, free_m3_s, ambient_K, self.tol, self.max_iter)
            n = tool.hole_count
            for k in range(n):
                res.outlets.append(Outlet(f"{pump_id}#{k}", "", pump_id, kind, AEROSOL if liquid_m3_s > 0 else GAS,
                                          gas.mass_kg_s / n / gas.rho_kg_m3 if gas.rho_kg_m3 > 0 else 0.0, gas.speed_m_s,
                                          gas.diameter_m / math.sqrt(n), gas.mass_kg_s / n, liquid_m3_s / n,
                                          gas.rho_kg_m3, True, k, gas.temp_K))
            res.air_free_m3_s += _free_air(gas.mass_kg_s, ambient_K)
            return
        if not nozzles:
            return
        areas = [max(n.count, 1) * math.pi * n.diameter_m ** 2 / 4.0 for n in nozzles]
        total_area = sum(areas)
        gas = air_outlets(total_area, gauge_Pa, free_m3_s, ambient_K, self.tol, self.max_iter)
        res.air_free_m3_s += _free_air(gas.mass_kg_s, ambient_K)
        for nozzle, area in zip(nozzles, areas):
            share = area / total_area
            count = max(nozzle.count, 1)
            for k in range(count):
                m = gas.mass_kg_s * share / count
                jet_id = nozzle.id if count == 1 else f"{nozzle.id}#{k}"
                res.outlets.append(Outlet(jet_id, nozzle.id, pump_id, kind, AEROSOL if liquid_m3_s > 0 else GAS,
                                          m / gas.rho_kg_m3 if gas.rho_kg_m3 > 0 else 0.0, gas.speed_m_s,
                                          gas.diameter_m * math.sqrt(share / count), m,
                                          liquid_m3_s * share / count, gas.rho_kg_m3, False, -1, gas.temp_K))

    def _solve_gas_fluid(self, res, on, air_on, blocked, tool, ambient_K) -> None:
        """Fluid kind air (dry machining with air) or MQL oil: every active pump's outlets emit air or
        an oil aerosol instead of a liquid."""
        p = P()
        fluid = self.fluid
        channels = []
        for pid in on:
            pump = self.pumps[pid]
            fed = [n for n in self.nozzles if n.pump == pid]
            external = [n for n in fed if n.kind not in THROUGH_KINDS and n.id not in blocked]
            through = (pump.kind in THROUGH_KINDS and not fed) or any(n.kind in THROUGH_KINDS for n in fed)
            if external:
                channels.append((pid, external, None))
            if through:
                if tool is not None and tool.hole_count > 0:
                    channels.append((pid, [], tool))
                else:
                    res.notes.append((f"no_holes:{pid}:{getattr(tool, 'number', None)}",
                                      f"through-tool supply ({pid}) on, but the tool has no coolant holes: no flow"))
        if fluid.is_mql:
            dosage = fluid.dosage_m3_s if fluid.dosage_m3_s else p["mql_dosage_m3_s"]
            pressure = fluid.air_pressure_Pa if fluid.air_pressure_Pa else p["mql_air_pressure_Pa"]
            n = len(channels)
            for pid, nozzles, holes in channels:
                self._gas_from(res, nozzles, pid, p["mql_air_m3_s"], pressure, dosage / n, ambient_K, holes, "mql")
            res.mql_oil_m3_s = dosage if n else 0.0
            if air_on and self.air_flow_m3_s > 0.0:
                self._air_blast(res, blocked, ambient_K)
            return
        flow = self.air_flow_m3_s if self.air_flow_m3_s > 0.0 else (fluid.typical_flow_m3_s or 0.0)
        pressure = self.air_pressure_Pa if self.air_pressure_Pa > 0.0 else (fluid.supply_pressure_Pa or 0.0)
        if air_on and not channels:
            self._air_blast(res, blocked, ambient_K)
            return
        n = len(channels)
        for pid, nozzles, holes in channels:
            self._gas_from(res, nozzles, pid, flow / n, pressure, 0.0, ambient_K, holes, "air")
