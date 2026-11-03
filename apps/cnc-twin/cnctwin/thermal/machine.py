"""Machine thermal models (docs/models/thermal.md "Machine"): spindles, ballscrew axes, structure.

Each is a first-order lumped body integrated exactly over a step (constant input):

    C dT/dt = P - G (T - T_env),  tau = C / G
    T(t + dt) = T_eq + (T - T_eq) exp(-dt / tau),  T_eq = T_env + P / G
    loss = G [(T_eq - T_env) dt + (T - T_eq) tau (1 - exp(-dt / tau))]

* spindle: bearing heat P = P_max (|omega| / omega_max)^(5/3) (Palmgren's viscous moment), time
  constant and steady growth at max speed from the machine file; the growth delta = alpha L_eff
  (T - T_env) along the spindle axis; the control's compensation leaves ``comp_residual_fraction``.
  The tool holder conducts into the spindle nose (from the tool ladder).
* axes: nut and bearing heat per metre of travel heats the screw (C from its steel volume, G = C /
  tau); the error at axis position s is alpha (T - T0) (s - s_fixed) along the axis (zero with linear
  scales; times the residual with electronic compensation).
* structure: relaxes to ambient with an hours-scale time constant; drift = sensitivity x
  (T - T0) along the tool axis (away from the work when warmer).

Growths and drifts are relative to the state at the start of the run (spindle) or of the
operation (axes, structure), when the tool lengths and work offsets were set.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..db.models import params
from . import props
from .inputs import MIN, MM, UM, flag, get, number, plain
from .ledger import SolverLedger


@functools.lru_cache(maxsize=1)
def _p() -> dict:
    return dict(params("thermal-machine"))


def first_order(theta: float, theta_env: float, power_W: float, g_W_K: float, cap_J_K: float,
                dt: float) -> tuple[float, float]:
    """(excess temperature after dt, heat lost over dt) of C d(theta)/dt = P - G (theta - theta_env),
    exact; temperatures as excesses over 20 degC so small changes keep their precision."""
    if g_W_K <= 0.0:
        return theta + power_W * dt / cap_J_K, 0.0
    tau = cap_J_K / g_W_K
    theta_eq = theta_env + power_W / g_W_K
    growth = -math.expm1(-dt / tau)                  # 1 - exp(-dt/tau) without cancellation
    theta_new = theta + (theta_eq - theta) * growth
    loss = g_W_K * ((theta_eq - theta_env) * dt + (theta - theta_eq) * tau * growth)
    return theta_new, loss


REF_K = 293.15


@dataclass
class LumpedBody:
    """A first-order body with its own ledger; state = excess temperature over 20 degC (energy C theta)."""

    name: str
    cap_J_K: float
    g_W_K: float
    theta_K: float
    ledger: SolverLedger = field(default_factory=lambda: SolverLedger("lumped"))

    @classmethod
    def at(cls, name: str, cap_J_K: float, g_W_K: float, temp_K: float, ledger: SolverLedger) -> "LumpedBody":
        return cls(name, cap_J_K, g_W_K, temp_K - REF_K, ledger)

    @property
    def T_K(self) -> float:
        return REF_K + self.theta_K

    def set_temperature(self, temp_K: float) -> None:
        self.theta_K = temp_K - REF_K

    def energy(self) -> float:
        return self.cap_J_K * self.theta_K

    def step(self, dt: float, power_W: float, T_env: float, heat_in_J: float, tol: float, time_s: float) -> float:
        """Advance with a constant power plus an energy input spread over the step; returns the loss."""
        u0 = self.energy()
        if dt <= 0.0:
            if heat_in_J:
                self.theta_K += heat_in_J / self.cap_J_K
                self.ledger.book(u0, self.energy(), heat_in_J, 0.0, 0.0, tol, time_s)
            return 0.0
        total_power = power_W + heat_in_J / dt
        self.theta_K, loss = first_order(self.theta_K, T_env - REF_K, total_power, self.g_W_K, self.cap_J_K, dt)
        self.ledger.book(u0, self.energy(), total_power * dt, loss, 0.0, tol, time_s)
        return loss

    def to_dict(self) -> dict:
        return {"name": self.name, "cap_J_K": self.cap_J_K, "g_W_K": self.g_W_K, "theta_K": self.theta_K,
                "ledger": self.ledger.to_dict()}

    @classmethod
    def from_dict(cls, values: dict) -> "LumpedBody":
        return cls(values["name"], float(values["cap_J_K"]), float(values["g_W_K"]), float(values["theta_K"]),
                   SolverLedger.from_dict(values["ledger"]))


@dataclass
class SpindleThermal:
    sid: str
    kind: str
    omega_max_rad_s: float
    growth_max_m: float
    bearing_heat_max_W: float
    comp: bool
    residual: float
    exponent: float
    alpha_1_K: float
    length_m: float
    body: LumpedBody
    T_env_K: float
    growth0_m: float = 0.0
    estimated: list = field(default_factory=list)
    last_heat_W: float = 0.0

    def growth_abs_m(self) -> float:
        return self.alpha_1_K * self.length_m * (self.body.T_K - self.T_env_K)

    def growth_rel_m(self) -> float:
        """Growth since the start of the run left after the control's compensation."""
        delta = self.growth_abs_m() - self.growth0_m
        return self.residual * delta if self.comp else delta

    def bearing_heat(self, omega_abs: np.ndarray) -> float:
        """Time mean of the bearing heat over the ticks of a step (W)."""
        if self.omega_max_rad_s <= 0.0 or omega_abs.size == 0:
            return 0.0
        ratio = np.clip(np.abs(omega_abs) / self.omega_max_rad_s, 0.0, None)
        return float(self.bearing_heat_max_W * np.mean(ratio ** self.exponent))

    def step(self, dt: float, omega_abs: np.ndarray, heat_in_J: float, tol: float, time_s: float) -> float:
        self.last_heat_W = self.bearing_heat(omega_abs)
        return self.body.step(dt, self.last_heat_W, self.T_env_K, heat_in_J, tol, time_s)

    def to_dict(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "body"}
        out["body"] = self.body.to_dict()
        return out

    @classmethod
    def from_dict(cls, values: dict) -> "SpindleThermal":
        values = dict(values)
        body = LumpedBody.from_dict(values.pop("body"))
        return cls(body=body, **values)


def spindle_models(machine: dict, ambient_K: float, start, tol: float, time_s: float) -> dict[str, SpindleThermal]:
    """Spindle models of a machine (raw machine mapping) at the start state ('cold', 'warm', dict)."""
    p = _p()
    steel = props.material_table_by_id(str(p["spindle_material"]))
    out = {}
    for spindle in machine.get("spindles") or []:
        sid = str(plain(spindle.get("id")) or "main")
        thermal = spindle.get("thermal") or {}
        estimated = []
        growth = number(thermal, (("growth_um_at_max_rpm", UM, 0.0), ("growth_at_max_speed_um", UM, 0.0),
                                  ("growth_at_max_speed_mm", MM, 0.0)))
        if growth is None:
            growth = p["default_spindle_growth_m"]
            estimated.append("growth")
        tau = number(thermal, (("time_constant_min", MIN, 0.0), ("time_constant_s", 1.0, 0.0)))
        if tau is None:
            tau = p["default_spindle_time_constant_s"]
            estimated.append("time_constant")
        heat = number(thermal, (("bearing_heat_W_at_max_rpm", 1.0, 0.0), ("bearing_heat_at_max_speed_W", 1.0, 0.0),
                                ("bearing_heat_at_max_speed_kW", 1e3, 0.0)))
        if heat is None:
            heat = p["default_bearing_heat_W"]
            estimated.append("bearing_heat")
        comp = flag(thermal, "thermal_comp", False)
        residual = number(thermal, (("comp_residual_fraction", 1.0, 0.0),), None)
        if comp and residual is None:
            residual = p["default_comp_residual_fraction"]
            estimated.append("comp_residual_fraction")
        omega_max = number(spindle, (("max_rpm", 2.0 * math.pi / 60.0, 0.0),), 0.0)
        alpha = props.mean_expansion(steel, 293.15)
        length = p["spindle_growth_length_m"]
        rise_max = growth / (alpha * length) if growth > 0.0 else 1.0
        g = heat / rise_max if heat > 0.0 else 1.0
        cap = g * tau
        body = LumpedBody.at(f"spindle.{sid}", cap, g, ambient_K, SolverLedger(f"spindle.{sid}"))
        model = SpindleThermal(sid, str(plain(spindle.get("kind")) or "milling"), omega_max, growth, heat, comp,
                               float(residual if residual is not None else 1.0), p["bearing_heat_speed_exponent"],
                               alpha, length, body, ambient_K, estimated=estimated)
        model.body.set_temperature(spindle_start_temperature(model, machine, ambient_K, start))
        model.body.ledger.u_J = model.body.energy()
        model.body.ledger.u0_J = model.body.ledger.u_J
        model.growth0_m = model.growth_abs_m()
        out[sid] = model
    return out


def warmup_time_s(machine: dict) -> float:
    return number(get(machine, "thermal", "warmup", default={}), (("cold_to_warm_min", MIN, 0.0),
                                                                   ("cold_to_warm_s", 1.0, 0.0)), 0.0)


def spindle_start_temperature(model: SpindleThermal, machine: dict, ambient_K: float, start) -> float:
    p = _p()
    if isinstance(start, dict):
        return start.get("spindle_K") or ambient_K
    if start == "cold":
        return ambient_K
    omega = p["warmup_speed_fraction"] * model.omega_max_rad_s
    power = model.bearing_heat(np.array([omega]))
    t_warm = warmup_time_s(machine)
    theta, _ = first_order(ambient_K - REF_K, ambient_K - REF_K, power, model.body.g_W_K, model.body.cap_J_K, t_warm)
    return REF_K + theta


@dataclass
class AxisThermal:
    name: str
    direction_M: list
    fixed_end_m: float
    alpha_1_K: float
    heat_per_m_J: float
    scale_feedback: bool
    comp: bool
    residual: float
    body: LumpedBody
    T0_K: float
    estimated: list = field(default_factory=list)

    def strain(self) -> float:
        """Screw strain since the reference state that reaches the axis position."""
        if self.scale_feedback:
            return 0.0
        value = self.alpha_1_K * (self.body.T_K - self.T0_K)
        return self.residual * value if self.comp else value

    def error_at(self, position_m: float) -> float:
        return self.strain() * (position_m - self.fixed_end_m)

    def step(self, dt: float, travel_m: float, T_env: float, tol: float, time_s: float) -> float:
        power = self.heat_per_m_J * travel_m / dt if dt > 0.0 else 0.0
        return self.body.step(dt, power, T_env, 0.0, tol, time_s)

    def to_dict(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if k != "body"}
        out["body"] = self.body.to_dict()
        return out

    @classmethod
    def from_dict(cls, values: dict) -> "AxisThermal":
        values = dict(values)
        body = LumpedBody.from_dict(values.pop("body"))
        return cls(body=body, **values)


def axis_directions(machine: dict) -> dict[str, list]:
    """Direction (in the base frame) of each NC axis from the kinematic chains, else its name."""
    out = {}
    kin = machine.get("kinematics") or {}
    for chain in ("workpiece_chain", "tool_chain"):
        for link in kin.get(chain) or []:
            if isinstance(link, dict) and link.get("axis") is not None and link.get("direction") is not None:
                vec = np.asarray(plain(link.get("direction")), dtype=float)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    out[str(link["axis"])] = list(vec / norm)
    return out


def axis_models(machine: dict, ambient_K: float, start, tol: float, time_s: float) -> dict[str, AxisThermal]:
    p = _p()
    steel = props.material_table_by_id(str(p["ballscrew_material"]))
    drift = get(machine, "thermal", "axis_drift", default={})
    estimated = []
    alpha = number(drift, (("expansion_1_K", 1.0, 0.0),))
    if alpha is None:
        alpha = p["default_screw_expansion_1_K"]
        estimated.append("expansion")
    heat = number(drift, (("heat_per_m_travel_J", 1.0, 0.0),))
    if heat is None:
        heat = p["default_axis_heat_per_m_J"]
        estimated.append("heat_per_m_travel")
    tau = number(drift, (("time_constant_min", MIN, 0.0), ("time_constant_s", 1.0, 0.0)))
    if tau is None:
        tau = p["default_axis_time_constant_s"]
        estimated.append("time_constant")
    scales = flag(drift, "scale_feedback", False)
    comp = flag(drift, "electronic_comp", False)
    residual = number(drift, (("comp_residual_fraction", 1.0, 0.0),))
    if comp and residual is None:
        residual = p["default_comp_residual_fraction"]
        estimated.append("comp_residual_fraction")
    directions = axis_directions(machine)
    rho = steel.rho_kg_m3[0]
    cp = float(np.interp(293.15, steel.T_K, steel.cp_J_kgK))
    out = {}
    for name in sorted((machine.get("axes") or {}).keys()):
        axis = machine["axes"][name]
        if str(plain(axis.get("type")) or "linear") != "linear":
            continue
        travel = plain(axis.get("travel_mm"))
        lo, hi = (float(travel[0]) * MM, float(travel[1]) * MM) if travel else (0.0, 0.0)
        screw = axis.get("ballscrew") or {}
        diameter = number(screw, (("diameter_mm", MM, 0.0),), p["default_ballscrew_diameter_m"])
        length = number(screw, (("length_mm", MM, 0.0),), (hi - lo) + p["ballscrew_extra_length_m"])
        fixed = str(plain(screw.get("fixed_end")) or p["ballscrew_fixed_end"])
        if fixed in ("motor",):
            fixed = str(p["ballscrew_fixed_end"])
        fixed_pos = lo if fixed == "min" else hi
        both_fixed = fixed == "both"
        cap = rho * cp * math.pi * diameter * diameter / 4.0 * length
        body = LumpedBody.at(f"axis.{name}", cap, cap / tau, ambient_K, SolverLedger(f"axis.{name}"))
        direction = directions.get(name) or {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0],
                                             "Z": [0.0, 0.0, 1.0]}.get(name[:1], [1.0, 0.0, 0.0])
        model = AxisThermal(name, list(direction), fixed_pos, 0.0 if both_fixed else alpha, heat, scales, comp,
                            float(residual if residual is not None else 1.0), body, ambient_K,
                            estimated=list(estimated) + (["fixed_end"] if not screw.get("fixed_end") else []))
        if isinstance(start, dict) and start.get("axes_K"):
            body.set_temperature(float(start["axes_K"]))
        elif start == "warm":
            power = heat * p["warmup_axis_speed_m_s"]
            theta, _ = first_order(ambient_K - REF_K, ambient_K - REF_K, power, body.g_W_K, cap, warmup_time_s(machine))
            body.set_temperature(REF_K + theta)
        model.T0_K = body.T_K
        body.ledger.u_J = body.energy()
        body.ledger.u0_J = body.ledger.u_J
        out[name] = model
    return out


@dataclass
class StructureThermal:
    sensitivity_m_per_K: float
    body: LumpedBody
    T0_K: float
    estimated: list = field(default_factory=list)

    def drift_m(self) -> float:
        return self.sensitivity_m_per_K * (self.body.T_K - self.T0_K)

    def to_dict(self) -> dict:
        return {"sensitivity_m_per_K": self.sensitivity_m_per_K, "T0_K": self.T0_K, "estimated": self.estimated,
                "body": self.body.to_dict()}

    @classmethod
    def from_dict(cls, values: dict) -> "StructureThermal":
        return cls(float(values["sensitivity_m_per_K"]), LumpedBody.from_dict(values["body"]), float(values["T0_K"]),
                   list(values["estimated"]))


def structure_model(machine: dict, ambient_K: float, start) -> StructureThermal:
    p = _p()
    estimated = []
    sens = number(get(machine, "thermal", default={}), (("structure_sensitivity_um_per_K", UM, 0.0),
                                                          ("structure_sensitivity_m_per_K", 1.0, 0.0)))
    if sens is None:
        sens = 0.0
        estimated.append("structure_sensitivity (none given: no drift)")
    mass = number(get(machine, "power", default={}), (("machine_mass_kg", 1.0, 0.0),))
    if mass is None:
        mass = p["default_machine_mass_kg"]
        estimated.append("machine_mass")
    iron = props.material_table_by_id(str(p["structure_material"]))
    cap = mass * float(np.interp(293.15, iron.T_K, iron.cp_J_kgK))
    body = LumpedBody.at("structure", cap, cap / p["structure_time_constant_s"], ambient_K, SolverLedger("structure"))
    if isinstance(start, dict) and start.get("structure_K"):
        body.set_temperature(float(start["structure_K"]))
    body.ledger.u_J = body.energy()
    body.ledger.u0_J = body.ledger.u_J
    return StructureThermal(sens, body, body.T_K, estimated)
