"""Electrical energy per part: spindle, axes, pumps, chip conveyor, base load, compressed air.

Pure functions over sampled signals (``dt_s`` is the duration each sample stands for; energies are
the rectangle-rule sums ``sum(P_i dt_i)``, exact for piecewise-constant signals such as the per-tick
means of the recorder). Model (docs/models/production.md "Energy"):

* spindle: ``P_el = P_shaft / eta + P_idle(omega) + max(dE_kin/dt, 0) / eta - recovered braking``
  with ``P_idle = P_idle,ref (|omega| / omega_ref)^x`` (x = 5/3 from Palmgren's load-independent
  bearing friction torque ~ (nu n)^(2/3)), ``E_kin = J omega^2 / 2`` (rotor inertia of the machine
  file plus what the caller adds for tool and holder), ``eta`` the machine file's spindle efficiency
  (electrical to shaft at rated load), ``P_idle,ref`` its idle power at the maximum speed.
* axes: ``F = m a + m g_z + sign(v) (mu N + F_seal)`` per axis (N the weight on the guides of a
  horizontal axis, g_z the lift against gravity per unit of +motion), mechanical power ``F v`` plus
  the process feed power, electrical ``P / (eta_screw eta_drive)`` when positive, braking recovered
  with ``axis_regen_recovered_frac`` (default 0), plus the servo standby power of the machine file.
* pumps: ``P_rated load_factor / eta_motor`` while each pump runs; conveyor by type while on.
* base load: ``base_W`` of the machine file for the whole time the machine is on (cycle and waits).
* compressed air: free-air volume times ``p0 ln((p0 + p_g)/p0) / eta_iso`` (isothermal compression
  work over the compressor's isothermal efficiency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..db.models import params as model_params
from ..sanity.si import get, items, num, plain, si_view, text
from ..units import STANDARD_ATMOSPHERE_PA, STANDARD_GRAVITY_M_S2


def _params(params):
    return params if params is not None else model_params("production-energy")


# ----------------------------------------------------------------------------- spindle

@dataclass(frozen=True)
class SpindleModel:
    efficiency: float
    idle_power_W: float
    ref_omega_rad_s: float
    inertia_kgm2: float
    exponent: float
    regen_frac: float

    @classmethod
    def from_machine(cls, machine, spindle_id: str = "main", *, extra_inertia_kgm2: float = 0.0,
                     params=None) -> "SpindleModel":
        p = _params(params)
        spindles = items(get(si_view(machine), "spindles"))
        spindle = next((s for s in spindles if text(get(s, "id")) == spindle_id), spindles[0] if spindles else None)
        if spindle is None:
            raise ValueError(f"the machine has no spindle {spindle_id!r}")
        efficiency = num(get(spindle, "efficiency"))
        idle = num(get(spindle, "idle_power_W"))
        max_speed = num(get(spindle, "max_rad_s"))
        inertia = num(get(spindle, "rotor_inertia_kgm2"), 0.0)
        missing = [k for k, v in (("efficiency", efficiency), ("idle_power_W", idle), ("max_rpm", max_speed)) if v is None]
        if missing:
            raise ValueError(f"spindle {spindle_id}: the machine file gives no {', '.join(missing)}")
        return cls(float(efficiency), float(idle), float(max_speed), float(inertia) + float(extra_inertia_kgm2),
                   float(p["palmgren_speed_exponent"]), float(p["spindle_regen_recovered_frac"]))

    def idle_W(self, omega_rad_s) -> np.ndarray:
        ratio = np.abs(np.asarray(omega_rad_s, dtype=float)) / self.ref_omega_rad_s
        return self.idle_power_W * np.power(ratio, self.exponent)


def hold_within(values, tolerance) -> np.ndarray:
    """``values`` with changes no larger than ``tolerance`` held at the last accepted value (a
    hysteresis that keeps the quantisation noise of recorded signals out of energy changes); with
    tolerance 0 the values are returned unchanged. Works on [n] and [n, k] arrays (per column)."""
    data = np.asarray(values, dtype=float)
    if not np.any(np.asarray(tolerance) > 0.0) or len(data) == 0:
        return data
    held = data.copy()
    tol = np.broadcast_to(np.asarray(tolerance, dtype=float), data.shape[1:]) if data.ndim > 1 else float(tolerance)
    for i in range(1, len(held)):
        keep = np.abs(data[i] - held[i - 1]) <= tol
        held[i] = np.where(keep, held[i - 1], data[i])
    return held


def spindle_energy(dt_s, omega_rad_s, shaft_power_W, model: SpindleModel, omega_tolerance_rad_s: float = 0.0) -> dict:
    """Energy terms of the spindle over the samples (J); speed changes within
    ``omega_tolerance_rad_s`` (the quantisation of a recorded speed) do not count as accelerations."""
    dt = np.asarray(dt_s, dtype=float)
    omega = np.asarray(omega_rad_s, dtype=float)
    shaft = np.asarray(shaft_power_W, dtype=float)
    if not (dt.shape == omega.shape == shaft.shape):
        raise ValueError("dt_s, omega_rad_s and shaft_power_W must have the same length")
    shaft_J = float(np.sum(np.maximum(shaft, 0.0) * dt))
    process_back_J = float(np.sum(np.minimum(shaft, 0.0) * dt))
    idle_J = float(np.sum(model.idle_W(omega) * dt))
    held = hold_within(omega, omega_tolerance_rad_s)
    kinetic = 0.5 * model.inertia_kgm2 * held * held
    change = np.diff(kinetic) if len(kinetic) > 1 else np.zeros(0)
    accel_J = float(np.sum(np.maximum(change, 0.0)))
    brake_J = float(-np.sum(np.minimum(change, 0.0)))
    recovered_J = (brake_J - process_back_J) * model.regen_frac * model.efficiency
    loss_J = (shaft_J + accel_J) * (1.0 / model.efficiency - 1.0)
    electrical_J = shaft_J + accel_J + loss_J + idle_J - recovered_J
    return {"shaft_J": shaft_J, "accel_J": accel_J, "brake_J": brake_J, "motor_loss_J": loss_J, "idle_J": idle_J,
            "recovered_J": recovered_J, "electrical_J": electrical_J}


# ----------------------------------------------------------------------------- axes

@dataclass(frozen=True)
class AxisModel:
    name: str
    mass_kg: float
    lift_m_s2: float                 # g times the upward component of the moving body's +motion
    normal_load_N: float             # weight on the guides (horizontal axes)
    friction_coeff: float
    seal_drag_N: float
    efficiency: float                # ball screw x drive
    regen_frac: float

    @property
    def friction_N(self) -> float:
        return self.friction_coeff * self.normal_load_N + self.seal_drag_N


def axis_models(machine, *, payload_kg: dict[str, float] | None = None, params=None) -> list[AxisModel]:
    """Linear axes of a machine file in file order (payload: workpiece and fixture mass the table
    axes carry, by axis name)."""
    p = _params(params)
    machine = si_view(machine)
    axes = get(machine, "axes") or {}
    directions = _axis_directions(machine)
    payload = payload_kg or {}
    models = []
    for name in axes:
        axis = get(axes, name)
        if not hasattr(axis, "get") and not hasattr(axis, "keys"):
            continue        # the axes block's own entries (src, note): not an axis (haas-ec400: axes.src)
        if text(get(axis, "type"), "linear") != "linear":
            continue
        mass = num(get(axis, "moving_mass_kg"))
        if mass is None:
            raise ValueError(f"axis {name}: the machine file gives no moving_mass_kg")
        mass += float(payload.get(str(name), 0.0))
        up = directions.get(str(name), (0.0, 0.0, 0.0))[2]
        vertical = abs(up) > 0.5
        models.append(AxisModel(str(name), mass, STANDARD_GRAVITY_M_S2 * up,
                                0.0 if vertical else mass * STANDARD_GRAVITY_M_S2,
                                float(p["axis_guide_friction_coeff"]), float(p["axis_seal_drag_N"]),
                                float(p["ballscrew_efficiency"]) * float(p["axis_drive_efficiency"]),
                                float(p["axis_regen_recovered_frac"])))
    return models


def _axis_directions(machine) -> dict[str, tuple[float, float, float]]:
    kinematics = get(machine, "kinematics")
    found: dict[str, tuple[float, float, float]] = {}
    for chain in ("workpiece_chain", "tool_chain"):
        for link in items(get(kinematics, chain)):
            axis = text(get(link, "axis"))
            direction = plain(get(link, "direction"))
            if axis and isinstance(direction, (list, tuple)) and len(direction) == 3:
                found[axis] = tuple(float(v) for v in direction)
    return found


def axis_energy(dt_s, vel_m_s, models: list[AxisModel], process_power_W=None, velocity_tolerance_m_s=0.0) -> dict:
    """Energy terms per axis (J): ``vel_m_s`` is [n, n_axes] in the order of ``models``, each sample
    holding for its ``dt_s``; ``process_power_W`` [n, n_axes] is the cutting load times velocity on
    each axis (optional). Per sample the work is ``(m g_z v + F_fric |v| + P_process) dt`` plus the
    kinetic energy change ``m (v_next^2 - v^2) / 2`` at its end (exact for piecewise-constant
    velocity); positive work is drawn through ``eta``, negative work recovered with the regenerated
    fraction. Velocity changes within ``velocity_tolerance_m_s`` (one value or one per axis: the
    quantisation of recorded positions) do not count in the kinetic energy."""
    dt = np.asarray(dt_s, dtype=float)
    vel = np.asarray(vel_m_s, dtype=float).reshape(len(dt), len(models))
    process = np.zeros_like(vel) if process_power_W is None else np.asarray(process_power_W, float).reshape(vel.shape)
    held = hold_within(vel, velocity_tolerance_m_s)
    result: dict[str, dict] = {}
    total_mech = total_el = 0.0
    for k, model in enumerate(models):
        v = vel[:, k]
        kinetic = 0.5 * model.mass_kg * held[:, k] * held[:, k]
        inertial = np.zeros_like(v)
        if len(v) > 1:
            inertial[:-1] = np.diff(kinetic)
        gravity = model.mass_kg * model.lift_m_s2 * v * dt
        friction = model.friction_N * np.abs(v) * dt
        work = inertial + gravity + friction + process[:, k] * dt
        electrical = np.where(work > 0.0, work / model.efficiency, work * model.regen_frac * model.efficiency)
        terms = {"inertial_J": float(np.sum(inertial)), "gravity_J": float(np.sum(gravity)),
                 "friction_J": float(np.sum(friction)), "process_J": float(np.sum(process[:, k] * dt)),
                 "mechanical_J": float(np.sum(work)), "electrical_J": float(np.sum(electrical))}
        result[model.name] = terms
        total_mech += terms["mechanical_J"]
        total_el += terms["electrical_J"]
    return {"axes": result, "mechanical_J": total_mech, "electrical_J": total_el}


def process_feed_energy(feed_work_J: float, params=None) -> float:
    """Electrical energy of the process feed work when it is known only as a total (the ledger's
    ``feed_work_J``): divided by the ball screw and drive efficiencies."""
    p = _params(params)
    return max(float(feed_work_J), 0.0) / (float(p["ballscrew_efficiency"]) * float(p["axis_drive_efficiency"]))


# ----------------------------------------------------------------------------- pumps, conveyor, base, air

def pump_energy(on_s: dict[str, float], machine, params=None) -> dict:
    """Energy of each coolant pump (J) for its running time; unknown pump ids raise."""
    p = _params(params)
    pumps = {text(get(x, "id")): x for x in items(get(si_view(machine), "coolant", "pumps"))}
    result = {}
    for pump_id, seconds in sorted(on_s.items()):
        if seconds <= 0.0:
            continue
        pump = pumps.get(pump_id)
        if pump is None:
            raise ValueError(f"pump {pump_id!r} is not in the machine file's coolant.pumps")
        rated = num(get(pump, "power_W"))
        if rated is None:
            raise ValueError(f"pump {pump_id!r}: the machine file gives no power_kW")
        power = rated * float(p["pump_load_factor"]) / float(p["pump_motor_efficiency"])
        result[pump_id] = {"on_s": seconds, "power_W": power, "electrical_J": power * seconds}
    return result


def conveyor_power_W(machine, params=None) -> float:
    conveyor = get(si_view(machine), "chip_conveyor")
    if conveyor is None:
        return 0.0
    kind = text(get(conveyor, "type"), "none")
    table = _params(params)["conveyor_power_W"]
    if kind not in table:
        raise ValueError(f"chip conveyor type {kind!r} has no power in db/models/production-energy.yaml")
    return float(table[kind])


def base_energy(on_s: float, machine) -> dict:
    power = get(si_view(machine), "power")
    base = num(get(power, "base_W"), 0.0)
    standby = num(get(power, "servo_standby_W"), 0.0)
    return {"base_W": base, "servo_standby_W": standby, "base_J": base * on_s, "servo_standby_J": standby * on_s}


def air_specific_energy_J_m3(gauge_pressure_Pa: float, params=None) -> float:
    """Electrical energy per m3 of free air compressed to ``gauge_pressure_Pa`` (isothermal work
    over the compressor's isothermal efficiency)."""
    p0 = STANDARD_ATMOSPHERE_PA
    return p0 * math.log((p0 + gauge_pressure_Pa) / p0) / float(_params(params)["compressor_isothermal_efficiency"])


def supply_pressure_Pa(machine, params=None) -> float:
    blast = get(si_view(machine), "coolant", "air_blast")
    pressure = num(get(blast, "pressure_Pa"))
    return pressure if pressure is not None else float(_params(params)["air_supply_gauge_pressure_Pa"])
