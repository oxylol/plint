"""Calibration targets: what is fitted from which measurement (docs/models/calibration.md "Targets").

Each target builds a :class:`Problem`: its parameters (prior = the project's / database's value,
bounds from db/models/mdata-calibration.yaml, the overlay path each value goes to), its observation
sets (measured values with their times in the simulated time base) and ``observe(sim_log, values,
sim_effects)`` that predicts them. ``sim_effects=True`` applies the parameters' effect on the
simulation analytically to the nominal run (the surrogate); ``False`` takes a run already simulated
with those values (cnctwin.core.loop) and applies only the observation model.

=================  ===============================================  =========================================
target             measured                                          parameters (default free first)
=================  ===============================================  =========================================
spindle            spindle.power_W (electrical) or spindle.load_frac  kc_scale, idle power (power) / bearing
                   against the run's shaft power and speed            friction power (load); efficiency,
                                                                     meter time constant (fixed)
servo              axis.<A>.following_error_m (or pos_cmd - pos)     K_v, K_ff per axis
axis_friction      axis.<A>.load_frac (air moves)                    seal drag, guide friction coefficient
block_time         block markers (program.line ...)                  block processing time
thermal_growth     spindle.growth_m with spindle.speed_rad_s         growth at max speed, time constant,
                                                                     initial growth, offset (absolute data)
tool_life          table: vc, f, ap, life                            Taylor C, n (a, b when f / ap vary)
usui               table: cutting time, VB, sigma_n, V_s, theta      Usui A (log), B
deflection         table: part features measured vs simulated        runout, tool stiffness factor
kienzle            cut segments (ae, ap) and spindle power            kc1.1, mc
mechanistic        cut segments and spindle power                     Ktc, Kte
=================  ===============================================  =========================================
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.errors import InputError
from ..db import keyunits
from ..db.models import params as model_params
from ..units import STANDARD_GRAVITY_M_S2
from .align import Alignment
from .fit import ObservationSet, Param
from .log import Channel, MachineLog
from .overlay import Target
from .params import calibration_params
from . import surrogates as S

TARGETS = ("spindle", "servo", "axis_friction", "block_time", "thermal_growth", "tool_life", "usui", "deflection",
           "kienzle", "mechanistic")
LOOP_TARGETS = ("spindle", "servo", "axis_friction", "block_time")


class Skip(Exception):
    """A target cannot be built from the data at hand (the reason is reported, nothing is fitted)."""


@dataclass
class Problem:
    target: str
    params: list[Param]
    observations: list[ObservationSet]
    observe: Callable[[MachineLog | None, dict, bool], dict]
    loop: bool = False
    constants: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)


@dataclass
class Context:
    """What the targets read: the resolved project (or None), the measured log, the nominal run and
    the alignment, the options of the request and the values calibrated by earlier targets."""

    project: object = None
    log: MachineLog | None = None
    sim_log: MachineLog | None = None
    run: object = None                      # cnctwin.mdata.simrun.SimulatedRun
    alignment: Alignment | None = None
    options: dict = field(default_factory=dict)
    base_dir: Path | None = None
    calibrated: dict = field(default_factory=dict)   # parameter name -> value from earlier targets

    # ------------------------------------------------------------------ options
    def target_options(self, target: str) -> dict:
        value = (self.options.get("target_options") or {}).get(target) or {}
        if not isinstance(value, dict):
            raise InputError(f"options.target_options.{target}: expected a mapping")
        return value

    def table(self, target: str, key: str) -> list[dict] | None:
        """Rows of a measurement table: options.target_options.<target>.<key> as a list, or a CSV file
        named there (relative to the project folder; column names carry their units)."""
        value = self.target_options(target).get(key)
        if value is None:
            value = self.options.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            path = Path(value)
            if self.base_dir is not None and not path.is_absolute():
                path = self.base_dir / path
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError as error:
                raise InputError(f"cannot read the {key} table: {error.strerror}", str(path)) from None
            rows = list(csv.DictReader(line for line in text.splitlines() if line.strip() and not line.startswith("#")))
            out = []
            for row in rows:
                out.append({k.strip(): _cell(v) for k, v in row.items() if k})
            return out
        if not isinstance(value, list):
            raise InputError(f"{key}: expected a list of rows or a CSV file name")
        return value

    # ------------------------------------------------------------------ machine facts
    @property
    def machine(self):
        return getattr(self.project, "machine", None) if self.project is not None else None

    @property
    def material(self):
        return getattr(self.project, "material", None) if self.project is not None else None

    def require_machine(self, what: str):
        if self.machine is None:
            raise Skip(f"{what} needs the project (its machine file)")
        return self.machine

    def spindle(self):
        machine = self.require_machine("the spindle model")
        wanted = self.options.get("spindle")
        spindles = list(machine.spindles or [])
        if not spindles:
            raise Skip("the machine has no spindle")
        for s in spindles:
            if wanted is None or str(s.id) == str(wanted):
                return s
        raise InputError(f"options.spindle {wanted!r}: the machine has spindles {', '.join(str(s.id) for s in spindles)}")


def _cell(value):
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _f(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounds(name: str) -> tuple[float, float]:
    lo, hi = calibration_params()[name]
    return float(lo), float(hi)


def _within(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _free(ctx: Context, target: str, params: list[Param]) -> list[Param]:
    """Apply options: target_options.<target>.free (names freed, the others fixed) and .fixed
    ({name: value}: fixed at that value)."""
    opts = ctx.target_options(target)
    names = {p.name for p in params}
    out = []
    free = opts.get("free")
    fixed = opts.get("fixed") or {}
    for name in list(fixed) + list(free or []):
        if name not in names:
            raise InputError(f"options.target_options.{target}: no parameter {name!r} (parameters: "
                             f"{', '.join(sorted(names))})")
    for p in params:
        is_free = (p.name in free) if free is not None else p.free
        value = p.value
        if p.name in fixed:
            value = float(fixed[p.name])
            is_free = False
            if not p.lower <= value <= p.upper:
                raise InputError(f"options.target_options.{target}.fixed.{p.name} = {value:g} is outside "
                                 f"[{p.lower:g}, {p.upper:g}]")
        out.append(Param(p.name, value, p.lower, p.upper, p.unit, is_free, p.log, p.scale, p.targets, p.doc))
    return out


def _measured_in_run(ctx: Context, channel: Channel) -> tuple[np.ndarray, np.ndarray]:
    """(t_sim, values) of the finite measured samples that fall inside the simulated run."""
    data = channel.finite()
    t_sim = ctx.alignment.to_sim(data.t_s)
    t0, t1 = ctx.sim_log.span_s()
    keep = (t_sim >= t0) & (t_sim <= t1)
    return t_sim[keep], np.asarray(data.values[keep], dtype=np.float64)


def _floor(channel: Channel) -> float:
    return channel.resolution() / math.sqrt(12.0)


def _selector(entries: list, index: int, keys=("tool_class", "coating", "process", "surface")) -> str:
    """A key=value selector naming entry ``index`` of a list uniquely, else its index."""
    entry = entries[index]
    if isinstance(entry, dict):
        def plain(v):
            return v.get("value") if isinstance(v, dict) and "value" in v else v
        wanted = {k: plain(entry[k]) for k in keys if k in entry and isinstance(plain(entry[k]), str)}
        if wanted:
            matches = [e for e in entries if isinstance(e, dict) and all(plain(e.get(k)) == v for k, v in wanted.items())]
            if len(matches) == 1:
                return ",".join(f"{k}={v}" for k, v in wanted.items())
    return str(index)


# ============================================================================ spindle

def _power_reference(spindle) -> Callable[[np.ndarray], np.ndarray]:
    ranges = list(spindle.gear_ranges or [])
    if not ranges or ranges[0].power_W is None:
        raise Skip("a spindle load needs the machine's spindle power curve (gear_ranges[].power_kW)")

    def reference(omega):
        w = np.abs(np.asarray(omega, dtype=np.float64))
        out = np.asarray(ranges[0].power_W(w), dtype=np.float64)
        for gear in ranges[1:]:
            span = gear.rad_s
            if span is None or gear.power_W is None:
                continue
            inside = (w >= float(span[0])) & (w <= float(span[1]))
            out = np.where(inside, np.asarray(gear.power_W(w), dtype=np.float64), out)
        return out
    return reference


def _kc_targets(ctx: Context) -> tuple[Target, ...]:
    material = ctx.material
    if material is None:
        return ()
    raw = material.raw
    cutting = raw.get("cutting") or {}
    out = []
    for section, keys in (("kienzle", ("kc11_N_mm2", "kf11_N_mm2", "kp11_N_mm2")),
                          ("mechanistic", ("Ktc_N_mm2", "Krc_N_mm2", "Kac_N_mm2", "Kte_N_mm", "Kre_N_mm", "Kae_N_mm"))):
        entries = cutting.get(section) or []
        for i, entry in enumerate(entries):
            for key in keys:
                value = entry.get(key) if isinstance(entry, dict) else None
                value = value.get("value") if isinstance(value, dict) else value
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out.append(Target(f"material.cutting.{section}[{_selector(entries, i)}].{key}", "scale",
                                      float(value)))
    return tuple(out)


def build_spindle(ctx: Context) -> Problem:
    if ctx.sim_log is None or "spindle.shaft_power_W" not in ctx.sim_log:
        raise Skip("needs a simulated run with cutting.spindle_power_W")
    if ctx.log is None:
        raise Skip("needs machine data")
    measured = ctx.log.first("spindle.power_W", "spindle.load_frac")
    if measured is None:
        raise Skip("the machine data has neither spindle.power_W nor spindle.load_frac")
    spindle = ctx.spindle()
    sid = str(spindle.id)
    omega_max = float(spindle.max_rad_s)
    inertia = _f(spindle.rotor_inertia_kgm2, 0.0)
    mode = "power" if measured.name == "spindle.power_W" else "load"
    p = calibration_params()
    t_obs, y_obs = _measured_in_run(ctx, measured)
    if len(t_obs) < 3:
        raise Skip("fewer than three measured spindle samples fall inside the aligned run")
    tau_lo, tau_hi = _bounds("meter_tau_bounds_s")
    kc_lo, kc_hi = _bounds("kc_scale_bounds")
    params = [Param("kc_scale", 1.0, kc_lo, kc_hi, "-", True, targets=_kc_targets(ctx),
                    doc="multiplier on every specific cutting force of the material")]
    if mode == "power":
        eta = _f(spindle.efficiency)
        idle = _f(spindle.idle_power_W)
        if eta is None or idle is None:
            raise Skip(f"spindle {sid}: the machine file gives no efficiency / idle_power_W")
        e_lo, e_hi = _bounds("efficiency_bounds")
        r_lo, r_hi = _bounds("idle_power_rel_bounds")
        upper = r_hi * idle if idle > 0 else float(p["idle_power_max_W"])
        params += [Param("spindle_efficiency", _within(eta, e_lo, e_hi), e_lo, e_hi, "-", False,
                         targets=(Target(f"machine.spindles[{sid}].efficiency"),),
                         doc="electrical to shaft at load (confounded with kc_scale: fit one of them)"),
                   Param("spindle_idle_power_W", idle, r_lo * idle, upper, "W", True,
                         targets=(Target(f"machine.spindles[{sid}].idle_power_W"),),
                         doc="idle (no-load) electrical power at the maximum speed")]
        exponent = float(model_params("production-energy")["palmgren_speed_exponent"])
    else:
        bearing = _f(spindle.thermal.bearing_heat_at_max_speed_W if spindle.thermal is not None else None)
        if bearing is None:
            raise Skip(f"spindle {sid}: the machine file gives no thermal.bearing_heat_at_max_speed_W")
        b_lo, b_hi = _bounds("bearing_power_rel_bounds")
        upper = b_hi * bearing if bearing > 0 else float(p["bearing_power_max_W"])
        params.append(Param("spindle_bearing_power_W", bearing, b_lo * bearing, upper, "W", True,
                            targets=(Target(f"machine.spindles[{sid}].thermal.bearing_heat_at_max_speed_W"),),
                            doc="bearing friction power at the maximum speed (the load meter's no-load reading)"))
        exponent = float(model_params("thermal-machine")["bearing_heat_speed_exponent"])
        reference_W = ctx.target_options("spindle").get("load_reference_W")
        reference = (lambda w: np.full(np.shape(w), float(reference_W))) if reference_W is not None \
            else _power_reference(spindle)
    params.append(Param("meter_tau_s", 0.0, tau_lo, tau_hi, "s", False,
                        doc="low-pass time constant of the meter (nuisance)"))
    params = _free(ctx, "spindle", params)
    nominal = ctx.sim_log

    def observe(sim_log, values, sim_effects):
        log = sim_log or nominal
        shaft_c = log.channels["spindle.shaft_power_W"].finite()
        grid = shaft_c.t_s
        shaft = shaft_c.values * (values["kc_scale"] if sim_effects else 1.0)
        omega = log.channels["spindle.speed_rad_s"].at(grid)
        omega = np.where(np.isfinite(omega), omega, 0.0)
        if mode == "power":
            model = S.spindle_electrical_power(grid, omega, shaft, efficiency=values["spindle_efficiency"],
                                               idle_power_max_W=values["spindle_idle_power_W"],
                                               omega_max_rad_s=omega_max, exponent=exponent, inertia_kgm2=inertia)
        else:
            out = S.spindle_motor_output(grid, omega, shaft, bearing_power_max_W=values["spindle_bearing_power_W"],
                                         omega_max_rad_s=omega_max, exponent=exponent, inertia_kgm2=inertia)
            ref = reference(omega)
            model = np.where(ref > 0, out / np.where(ref > 0, ref, 1.0), np.nan)
        model = S.first_order_filter(grid, np.nan_to_num(model, nan=0.0), values["meter_tau_s"]) \
            if values["meter_tau_s"] > 0 else model
        return {measured.name: np.interp(t_obs, grid, model)}

    prior = {q.name: q.value for q in params}
    valid = np.isfinite(observe(None, prior, True)[measured.name])
    y = np.where(valid, y_obs, np.nan)
    unit = "W" if mode == "power" else "-"
    obs = ObservationSet(measured.name, y, None, unit, t_obs, _floor(measured),
                         "measured spindle " + ("electrical power" if mode == "power" else "load fraction"))
    notes = []
    if mode == "power":
        notes.append("kc_scale and spindle_efficiency are confounded in power data (only kc_scale / efficiency is "
                     "determined); efficiency is fixed at the machine file's value unless freed")
    return Problem("spindle", params, [obs], observe, loop=True, notes=notes,
                   inputs={"mode": mode, "spindle": sid, "exponent": exponent, "omega_max_rad_s": omega_max,
                           "samples": int(np.isfinite(y).sum())})


def spindle_signal(ctx: Context) -> tuple[Channel, str] | None:
    """The measured spindle quantity predicted with the nominal machine values on the run's tick grid
    (for the cross-correlation alignment), and the measured channel it corresponds to."""
    if ctx.sim_log is None or ctx.log is None or "spindle.shaft_power_W" not in ctx.sim_log or ctx.machine is None:
        return None
    measured = ctx.log.first("spindle.power_W", "spindle.load_frac")
    if measured is None:
        return None
    try:
        spindle = ctx.spindle()
        shaft = ctx.sim_log.channels["spindle.shaft_power_W"].finite()
        omega = np.nan_to_num(ctx.sim_log.channels["spindle.speed_rad_s"].at(shaft.t_s), nan=0.0)
        omega_max = float(spindle.max_rad_s)
        inertia = _f(spindle.rotor_inertia_kgm2, 0.0)
        if measured.name == "spindle.power_W":
            model = S.spindle_electrical_power(shaft.t_s, omega, shaft.values, efficiency=float(spindle.efficiency),
                                               idle_power_max_W=float(spindle.idle_power_W), omega_max_rad_s=omega_max,
                                               exponent=float(model_params("production-energy")["palmgren_speed_exponent"]),
                                               inertia_kgm2=inertia)
        else:
            out = S.spindle_motor_output(shaft.t_s, omega, shaft.values,
                                         bearing_power_max_W=float(spindle.thermal.bearing_heat_at_max_speed_W),
                                         omega_max_rad_s=omega_max,
                                         exponent=float(model_params("thermal-machine")["bearing_heat_speed_exponent"]),
                                         inertia_kgm2=inertia)
            ref = _power_reference(spindle)(omega)
            model = np.where(ref > 0, out / np.where(ref > 0, ref, 1.0), 0.0)
    except (Skip, TypeError):
        return None
    return Channel("spindle.model_" + ("power_W" if measured.name == "spindle.power_W" else "load_frac"),
                   shaft.t_s, model, "sample", "nominal spindle model"), measured.name


# ============================================================================ servo

def _velocity(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(x, t) if len(x) >= 2 else np.zeros(len(x))


def build_servo(ctx: Context) -> Problem:
    if ctx.log is None:
        raise Skip("needs machine data")
    machine = ctx.require_machine("the servo model")
    axes = machine.axes or {}
    kv_lo, kv_hi = _bounds("kv_bounds_1_s")
    ff_lo, ff_hi = _bounds("feedforward_bounds")
    params, observations, drives = [], [], {}
    source = ctx.target_options("servo").get("command", "auto")
    for name in sorted(axes.keys()):
        axis = axes[name]
        if str(axis.type or "linear") != "linear" or axis.servo is None:
            continue
        measured = ctx.log.channels.get(f"axis.{name}.following_error_m")
        derived = False
        if measured is None and f"axis.{name}.pos_cmd_m" in ctx.log and f"axis.{name}.pos_machine_m" in ctx.log:
            act = ctx.log.channels[f"axis.{name}.pos_machine_m"].finite()
            cmd = ctx.log.channels[f"axis.{name}.pos_cmd_m"].at(act.t_s)
            measured = Channel(f"axis.{name}.following_error_m", act.t_s, cmd - act.values, "sample", "pos_cmd - pos")
            derived = True
        if measured is None or len(measured.finite()) < 3:
            continue
        # the commanded velocity: measured commanded positions, else the run's commanded path
        if source in ("auto", "measured") and f"axis.{name}.pos_cmd_m" in ctx.log:
            cmd = ctx.log.channels[f"axis.{name}.pos_cmd_m"].finite()
            t_cmd = ctx.alignment.to_sim(cmd.t_s) if ctx.alignment is not None else cmd.t_s
            v_cmd = _velocity(t_cmd, cmd.values)
            origin = "measured"
        elif ctx.sim_log is not None and f"axis.{name}.pos_stock_m" in ctx.sim_log:
            pos = ctx.sim_log.channels[f"axis.{name}.pos_stock_m"]
            err = ctx.sim_log.channels.get(f"axis.{name}.following_error_m")
            t_cmd = pos.t_s
            command = pos.values + (np.nan_to_num(err.at(t_cmd)) if err is not None else 0.0)
            v_cmd = _velocity(t_cmd, command)
            origin = "simulated"
        else:
            continue
        if ctx.alignment is not None and ctx.sim_log is not None:
            data = measured.finite()
            t_obs = ctx.alignment.to_sim(data.t_s)
            keep = (t_obs >= t_cmd[0]) & (t_obs <= t_cmd[-1])
            t_obs, y = t_obs[keep], np.asarray(data.values[keep], dtype=np.float64)
        else:
            data = measured.finite()
            t_obs, y = data.t_s, np.asarray(data.values, dtype=np.float64)
            keep = (t_obs >= t_cmd[0]) & (t_obs <= t_cmd[-1])
            t_obs, y = t_obs[keep], y[keep]
        if len(t_obs) < 3:
            continue
        kv = _f(axis.servo.kv_1_s)
        ff = _f(axis.servo.feedforward, 0.0)
        if kv is None:
            raise Skip(f"axis {name}: the machine file gives no servo.kv_1_s")
        params += [Param(f"axis_{name}_kv_1_s", _within(kv, kv_lo, kv_hi), kv_lo, kv_hi, "1/s", True,
                         targets=(Target(f"machine.axes.{name}.servo.kv_1_s"),), doc=f"position loop gain of {name}"),
                   Param(f"axis_{name}_feedforward", _within(ff, ff_lo, ff_hi), ff_lo, ff_hi, "-", True,
                         targets=(Target(f"machine.axes.{name}.servo.feedforward"),),
                         doc=f"velocity feed-forward of {name}")]
        observations.append(ObservationSet(measured.name, y, None, "m", t_obs, _floor(measured),
                                           f"following error of {name}" + (" (pos_cmd - pos)" if derived else "")))
        drives[name] = (t_cmd, v_cmd, origin)
    if not observations:
        raise Skip("no axis with a measured following error (or commanded and actual positions) and a command")
    params = _free(ctx, "servo", params)
    times = {o.name: o.t_s for o in observations}

    def observe(sim_log, values, sim_effects):
        out = {}
        for name, (t_cmd, v_cmd, origin) in drives.items():
            key = f"axis.{name}.following_error_m"
            if not sim_effects and sim_log is not None and origin == "simulated":
                out[key] = sim_log.channels[key].at(times[key])
                continue
            kv, ff = values[f"axis_{name}_kv_1_s"], values[f"axis_{name}_feedforward"]
            e0 = (1.0 - ff) * v_cmd[0] / kv
            e = S.following_error(t_cmd, v_cmd, kv_1_s=kv, feedforward=ff, e0_m=e0)
            out[key] = np.interp(times[key], t_cmd, e)
        return out

    notes = [f"axis {n}: commanded velocity from the {d[2]} positions" for n, d in sorted(drives.items())]
    notes.append("K_v and K_ff separate only through transients (acceleration); at constant feed only "
                 "(1 - K_ff) / K_v is determined (reported as confounded)")
    from_run = any(d[2] == "simulated" for d in drives.values())
    return Problem("servo", params, observations, observe, loop=from_run, notes=notes,
                   inputs={"axes": sorted(drives)})


# ============================================================================ axis friction

def _directions(machine) -> dict[str, float]:
    up = {}
    kinematics = machine.kinematics
    for chain in ("workpiece_chain", "tool_chain"):
        for link in (getattr(kinematics, chain, None) or []) if kinematics is not None else []:
            axis = link.axis if "axis" in link else None
            direction = link.direction if "direction" in link else None
            if axis and direction is not None and len(direction) == 3:
                up[str(axis)] = float(direction[2])
    return up


def build_axis_friction(ctx: Context) -> Problem:
    if ctx.log is None:
        raise Skip("needs machine data")
    machine = ctx.require_machine("the axis friction model")
    energy = model_params("production-energy")
    seal0, mu0 = float(energy["axis_seal_drag_N"]), float(energy["axis_guide_friction_coeff"])
    up = _directions(machine)
    axes = machine.axes or {}
    observations, drives = [], {}
    opts = ctx.target_options("axis_friction")
    reference = opts.get("load_reference_N") or {}
    for name in sorted(axes.keys()):
        axis = axes[name]
        measured = ctx.log.channels.get(f"axis.{name}.load_frac")
        if measured is None or len(measured.finite()) < 3 or str(axis.type or "linear") != "linear":
            continue
        mass = _f(axis.moving_mass_kg)
        force_ref = _f(reference.get(name)) if isinstance(reference, dict) else None
        force_ref = force_ref if force_ref is not None else _f(axis.motor_force_N)
        if mass is None or force_ref is None:
            raise Skip(f"axis {name}: needs moving_mass_kg and motor_force_N (or load_reference_N)")
        if ctx.sim_log is not None and ctx.alignment is not None and f"axis.{name}.pos_stock_m" in ctx.sim_log:
            pos = ctx.sim_log.channels[f"axis.{name}.pos_stock_m"]
            t_grid, x = pos.t_s, pos.values
            t_obs, y = _measured_in_run(ctx, measured)
            shaft = ctx.sim_log.channels.get("spindle.shaft_power_W")
            cutting = np.zeros(len(t_obs), bool)
            if shaft is not None:
                cutting = np.abs(np.nan_to_num(shaft.at(t_obs))) > 0.0
            y = np.where(cutting, np.nan, y)
        elif f"axis.{name}.pos_machine_m" in ctx.log:
            pos = ctx.log.channels[f"axis.{name}.pos_machine_m"].finite()
            t_grid, x = pos.t_s, pos.values
            data = measured.finite()
            t_obs, y = data.t_s, np.asarray(data.values, dtype=np.float64)
        else:
            continue
        v = _velocity(t_grid, x)
        a = _velocity(t_grid, v)
        lift = up.get(name, 0.0)
        normal = 0.0 if abs(lift) > 0.5 else mass * STANDARD_GRAVITY_M_S2
        drives[name] = (t_grid, v, a, mass, lift, normal, force_ref)
        observations.append(ObservationSet(measured.name, y, None, "-", t_obs, _floor(measured),
                                           f"servo load of {name} (non-cutting samples)"))
    if not observations:
        raise Skip("no axis with a measured load meter and a motion history")
    s_lo, s_hi = _bounds("seal_drag_bounds_N")
    m_lo, m_hi = _bounds("friction_coeff_bounds")
    params = _free(ctx, "axis_friction", [
        Param("axis_seal_drag_N", _within(seal0, s_lo, s_hi), s_lo, s_hi, "N", True,
              targets=(Target("models.production-energy.axis_seal_drag_N"),), doc="load-independent drag per axis"),
        Param("axis_friction_coeff", _within(mu0, m_lo, m_hi), m_lo, m_hi, "-", True,
              targets=(Target("models.production-energy.axis_guide_friction_coeff"),),
              doc="guide friction on the weight of horizontal axes")])
    times = {o.name: o.t_s for o in observations}

    def observe(sim_log, values, sim_effects):
        out = {}
        for name, (t_grid, v, a, mass, lift, normal, force_ref) in drives.items():
            force = S.axis_force(v, a, mass_kg=mass, lift_frac=lift, friction_coeff=values["axis_friction_coeff"],
                                 normal_load_N=normal, seal_drag_N=values["axis_seal_drag_N"])
            key = f"axis.{name}.load_frac"
            out[key] = np.interp(times[key], t_grid, force / force_ref)
        return out

    return Problem("axis_friction", params, observations, observe, loop=False,
                   notes=["axis loads are referred to motor_force_N of the machine file unless load_reference_N "
                          "is given; samples while cutting are left out (the run records no axis cutting load)"],
                   inputs={"axes": sorted(drives)})


# ============================================================================ block time

def build_block_time(ctx: Context) -> Problem:
    if ctx.run is None or ctx.alignment is None or ctx.alignment.method != "markers":
        raise Skip("needs a run aligned by block markers")
    machine = ctx.require_machine("the block-time model")
    controller = machine.controller
    t_b0 = _f(controller.block_processing_s if controller is not None else None)
    lookahead = int(_f(controller.lookahead_blocks if controller is not None else None, 1) or 1)
    if t_b0 is None:
        raise Skip("the machine file gives no controller.block_processing_ms")
    blocks = ctx.run.timeline()
    tick_s = ctx.run.tick_s
    motion = np.array([(int(b["tick1"]) - int(b["tick0"]) - int((b.get("ticks") or {}).get("data_starvation", 0)))
                       * tick_s for b in blocks])
    starts_sim = np.array([int(b["tick0"]) * tick_s for b in blocks])
    pairs = ctx.alignment.quality.get("pairs") or []
    index_of = {round(t, 9): i for i, t in enumerate(starts_sim)}
    matched = [(tm, index_of.get(round(ts, 9))) for tm, ts in pairs]
    matched = [(tm, j) for tm, j in matched if j is not None]
    if len(matched) < 3:
        raise Skip("fewer than three matched block markers")
    measured = np.array([ctx.alignment.scale * (b[0] - a[0]) for a, b in zip(matched[:-1], matched[1:])])
    first = np.array([a[1] for a in matched[:-1]])
    last = np.array([b[1] for b in matched[1:]])
    poll = float(ctx.alignment.quality.get("poll_interval_s") or 0.0)
    lo, hi = _bounds("block_time_bounds_s")
    params = _free(ctx, "block_time", [
        Param("block_processing_s", _within(t_b0, lo, hi), lo, hi, "s", True,
              targets=(Target("machine.controller.block_processing_ms"),),
              doc="minimum time per block (interpreter): data starvation of short blocks")])
    t0 = starts_sim[0]

    def observe(sim_log, values, sim_effects):
        if sim_effects or ctx.run is None:
            start, _ = S.block_queue(motion, block_s=values["block_processing_s"], lookahead=lookahead, start_s=t0)
        else:
            start = sim_log.channels["program.line"].t_s
        return {"block_spans_s": start[last] - start[first]}

    obs = ObservationSet("block_spans_s", measured, None, "s", starts_sim[first], math.sqrt(2.0) * poll / math.sqrt(12.0),
                         "time between consecutive measured block markers")
    notes = [f"look-ahead {lookahead} blocks (machine file); motion times from the timeline"
             + (" (minus its data_starvation ticks)" if any("data_starvation" in (b.get("ticks") or {}) for b in blocks)
                else ": the run's block durations, so only blocks the machine runs longer than simulated inform a "
                     "block time above the simulated one (simulate with a small block_processing_ms for the "
                     "reference run)")]
    return Problem("block_time", params, [obs], observe, loop=True, notes=notes,
                   inputs={"matched_markers": len(matched), "lookahead": lookahead})


# ============================================================================ thermal growth

def build_thermal_growth(ctx: Context) -> Problem:
    if ctx.log is None or "spindle.growth_m" not in ctx.log:
        raise Skip("needs measured spindle.growth_m")
    if "spindle.speed_rad_s" not in ctx.log:
        raise Skip("needs the measured spindle speed history (spindle.speed_rad_s) with the growth")
    spindle = ctx.spindle()
    thermal = spindle.thermal
    sid = str(spindle.id)
    growth0 = _f(thermal.growth_at_max_speed_m if thermal is not None else None)
    tau0 = _f(thermal.time_constant_s if thermal is not None else None)
    if growth0 is None or tau0 is None:
        raise Skip(f"spindle {sid}: the machine file gives no thermal.growth_at_max_speed_um / time_constant_min")
    opts = ctx.target_options("thermal_growth")
    compensated = str(opts.get("growth_measured", "physical")) == "compensated"
    residual = _f(thermal.comp_residual_fraction, 1.0) if compensated else 1.0
    absolute = bool(opts.get("absolute", False))
    exponent = float(model_params("thermal-machine")["bearing_heat_speed_exponent"])
    omega_max = float(spindle.max_rad_s)
    growth_c = ctx.log.channels["spindle.growth_m"].finite()
    speed_c = ctx.log.channels["spindle.speed_rad_s"].finite()
    grid = np.unique(np.concatenate([speed_c.t_s, growth_c.t_s]))
    speed = Channel("speed", speed_c.t_s, speed_c.values, "event").at(grid)
    speed = np.nan_to_num(speed, nan=0.0)
    g_lo, g_hi = _bounds("growth_bounds_m")
    t_lo, t_hi = _bounds("thermal_tau_bounds_s")
    o_lo, o_hi = _bounds("growth_offset_bounds_m")
    params = _free(ctx, "thermal_growth", [
        Param("spindle_growth_max_m", _within(growth0, g_lo, g_hi), g_lo, g_hi, "m", True,
              targets=(Target(f"machine.spindles[{sid}].thermal.growth_at_max_speed_um"),),
              doc="steady axial growth at the maximum speed"),
        Param("spindle_thermal_tau_s", _within(tau0, t_lo, t_hi), t_lo, t_hi, "s", True,
              targets=(Target(f"machine.spindles[{sid}].thermal.time_constant_min"),), doc="thermal time constant"),
        Param("spindle_growth_initial_m", 0.0, g_lo, g_hi, "m", True, doc="growth at the first sample (nuisance)"),
        Param("growth_offset_m", 0.0, o_lo, o_hi, "m", absolute,
              doc="zero of an absolute length measurement (nuisance; free with absolute: true)")])

    def observe(sim_log, values, sim_effects):
        delta = S.spindle_growth(grid, speed, growth_max_m=values["spindle_growth_max_m"],
                                 tau_s=values["spindle_thermal_tau_s"], omega_max_rad_s=omega_max, exponent=exponent,
                                 initial_m=values["spindle_growth_initial_m"])
        return {"spindle.growth_m": residual * np.interp(growth_c.t_s, grid, delta) + values["growth_offset_m"]}

    obs = ObservationSet("spindle.growth_m", growth_c.values, None, "m", growth_c.t_s, _floor(growth_c),
                         "measured axial growth" + (" (compensated)" if compensated else ""))
    return Problem("thermal_growth", params, [obs], observe, loop=False,
                   notes=[f"speed law exponent {exponent:g} (thermal-machine bearing_heat_speed_exponent); "
                          + ("measured growth is the compensated residual" if compensated else "physical growth")],
                   inputs={"samples": len(growth_c), "time_base": "machine log"})


# ============================================================================ tool life

def _si_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if value is None:
            continue
        si_key, si_value = keyunits.to_si(str(key), value) if isinstance(value, (int, float)) else (key, value)
        out[si_key] = si_value
    return out


def _tool(ctx: Context, target: str):
    number = ctx.target_options(target).get("tool")
    tools = list(getattr(getattr(ctx.project, "tools", None), "list", None) or []) if ctx.project is not None else []
    if not tools:
        return None
    for entry in tools:
        if number is None or int(entry.number) == int(number):
            return entry
    raise InputError(f"options.target_options.{target}.tool {number}: the project has no such tool")


def _material_entry(ctx: Context, section: str, target: str):
    material = ctx.material
    if material is None:
        raise Skip(f"{target} needs the project's material")
    raw_entries = (material.raw.get(section) if section in ("tool_life", "tool_life_usui")
                   else (material.raw.get("cutting") or {}).get(section)) or []
    entries = list((material[section] if section in ("tool_life", "tool_life_usui") else material.cutting[section])
                   or [])
    if not entries:
        return None, None, raw_entries
    index = ctx.target_options(target).get("entry")
    if index is None:
        tool = _tool(ctx, target)
        klass = None
        if tool is not None and tool.tool is not None and tool.tool.tool_material is not None:
            klass = tool.tool.tool_material["class"] if "class" in tool.tool.tool_material else None
        index = next((i for i, e in enumerate(entries) if klass is not None and str(e.tool_class) == str(klass)), 0)
    index = int(index)
    if not 0 <= index < len(entries):
        raise InputError(f"options.target_options.{target}.entry {index}: the material has {len(entries)} entries")
    return index, entries[index], raw_entries


def build_tool_life(ctx: Context) -> Problem:
    rows = ctx.table("tool_life", "tool_life")
    if not rows:
        raise Skip("needs a tool_life table (vc_m_min, life_min, optional feed_mm, ap_mm)")
    index, entry, raw_entries = _material_entry(ctx, "tool_life", "tool_life")
    if entry is None:
        raise Skip("the material has no tool_life entry to calibrate")
    data = [_si_row(r) for r in rows]
    for i, r in enumerate(data):
        if "vc_m_s" not in r or not ("life_s" in r):
            raise InputError(f"tool_life row {i + 1}: needs vc_m_min (or vc_m_s) and life_min (or life_s)")
    vc = np.array([r["vc_m_s"] for r in data])
    life = np.array([r["life_s"] for r in data])
    feed = np.array([r.get("feed_m", np.nan) for r in data])
    ap = np.array([r.get("ap_m", np.nan) for r in data])
    C0, n0, T_ref = float(entry.C_m_s), float(entry.n), float(entry.T_ref_s)
    a0, b0 = _f(entry.a, 0.0), _f(entry.b, 0.0)
    f_ref, ap_ref = _f(entry.feed_ref_m, 1.0), _f(entry.ap_ref_m, 1.0)
    sel = _selector(raw_entries, index)
    c_lo, c_hi = _bounds("taylor_C_rel_bounds")
    n_lo, n_hi = _bounds("taylor_n_bounds")
    x_lo, x_hi = _bounds("taylor_exponent_bounds")
    vary_f = np.isfinite(feed).all() and np.ptp(feed) > 0
    vary_ap = np.isfinite(ap).all() and np.ptp(ap) > 0
    params = _free(ctx, "tool_life", [
        Param("taylor_C_m_s", C0, c_lo * C0, c_hi * C0, "m/s", True, targets=(Target(f"material.tool_life[{sel}].C_m_min"),),
              doc="Taylor constant (cutting speed for T = T_ref)"),
        Param("taylor_n", _within(n0, n_lo, n_hi), n_lo, n_hi, "-", True, targets=(Target(f"material.tool_life[{sel}].n"),),
              doc="Taylor exponent"),
        Param("taylor_a", _within(a0, x_lo, x_hi), x_lo, x_hi, "-", bool(vary_f),
              targets=(Target(f"material.tool_life[{sel}].a"),), doc="feed exponent"),
        Param("taylor_b", _within(b0, x_lo, x_hi), x_lo, x_hi, "-", bool(vary_ap),
              targets=(Target(f"material.tool_life[{sel}].b"),), doc="depth exponent")])
    feed_in = np.where(np.isfinite(feed), feed, f_ref)
    ap_in = np.where(np.isfinite(ap), ap, ap_ref)

    def observe(sim_log, values, sim_effects):
        T = S.taylor_life(vc, C_m_s=values["taylor_C_m_s"], n=values["taylor_n"], T_ref_s=T_ref, feed_m=feed_in,
                          ap_m=ap_in, a=values["taylor_a"], b=values["taylor_b"], feed_ref_m=f_ref, ap_ref_m=ap_ref)
        return {"ln_tool_life": np.log(T)}

    obs = ObservationSet("ln_tool_life", np.log(life), None, "ln(s)", None, 0.0, "natural log of the measured tool life")
    return Problem("tool_life", params, [obs], observe, loop=False,
                   notes=["residuals in ln T: tool life scatters multiplicatively"],
                   inputs={"rows": len(rows), "entry": sel})


# ============================================================================ Usui wear

def build_usui(ctx: Context) -> Problem:
    rows = ctx.table("usui", "wear")
    if not rows:
        raise Skip("needs a wear table (test, t_min, VB_mm, sigma_n_MPa, sliding_m_min, temp_degC)")
    data = [_si_row(r) for r in rows]
    opts = ctx.target_options("usui")
    tool = _tool(ctx, "usui")
    clearance = _f(opts.get("clearance_deg"))
    rake = _f(opts.get("rake_deg"))
    clearance = clearance * math.pi / 180.0 if clearance is not None else None
    rake = rake * math.pi / 180.0 if rake is not None else None
    if tool is not None and tool.tool is not None and tool.tool.geometry is not None:
        geometry = tool.tool.geometry
        flank = geometry.clearance_rad
        if flank is not None and hasattr(flank, "keys"):
            flank = flank["primary"] if "primary" in flank else None
        clearance = clearance if clearance is not None else _f(flank)
        rake = rake if rake is not None else _f(geometry.rake_rad)
    if clearance is None or rake is None:
        raise InputError("usui: give the tool's clearance_deg and rake_deg (options.target_options.usui)")
    tests: dict = {}
    for i, r in enumerate(data):
        for key in ("t_s", "VB_m", "sigma_n_Pa", "sliding_m_s", "temp_K"):
            if key not in r:
                raise InputError(f"wear row {i + 1}: needs t_min, VB_mm, sigma_n_MPa, sliding_m_min, temp_degC")
        tests.setdefault(str(r.get("test", "1")), []).append(r)
    # per test: times [0, t_1 .. t_n]; the conditions of row k hold over (t_k-1, t_k], so the condition
    # arrays (indexed by interval start) are the rows' values, the last one repeated for the unused end
    series = []
    for name in sorted(tests):
        rows_t = sorted(tests[name], key=lambda r: r["t_s"])
        t = np.array([0.0] + [r["t_s"] for r in rows_t])

        def conditions(key, rows=rows_t):
            values = [r[key] for r in rows]
            return np.array(values + values[-1:], dtype=np.float64)

        series.append((name, t, conditions("sigma_n_Pa"), conditions("sliding_m_s"), conditions("temp_K"),
                       np.array([r["VB_m"] for r in rows_t]), float(rows_t[0].get("VB0_m", 0.0))))
    a_lo, a_hi = _bounds("usui_A_bounds_per_Pa")
    b_lo, b_hi = _bounds("usui_B_bounds_K")
    klass = "carbide"
    if tool is not None and tool.tool is not None and tool.tool.tool_material is not None:
        tm = tool.tool.tool_material
        klass = str(tm["class"]) if "class" in tm else klass
    existing = None
    if ctx.material is not None:
        for e in (ctx.material.tool_life_usui or []):
            if str(e.tool_class) == klass and str(e.surface or "flank") in ("flank", "both"):
                existing = e
                break
    A0 = _f(existing.A_per_Pa, None) if existing is not None else None
    B0 = _f(existing.B_K, None) if existing is not None else None
    A0 = A0 if A0 is not None else math.sqrt(a_lo * a_hi)
    B0 = B0 if B0 is not None else math.sqrt(b_lo * b_hi)
    sel = f"tool_class={klass},surface=flank"
    params = _free(ctx, "usui", [
        Param("usui_A_per_Pa", _within(A0, a_lo, a_hi), a_lo, a_hi, "1/Pa", True, log=True,
              targets=(Target(f"material.tool_life_usui[{sel}].A_per_Pa"),), doc="Usui A (SI rates)"),
        Param("usui_B_K", _within(B0, b_lo, b_hi), b_lo, b_hi, "K", True,
              targets=(Target(f"material.tool_life_usui[{sel}].B_K"),), doc="Usui B")])

    def observe(sim_log, values, sim_effects):
        out = []
        for name, t, sigma, vs, temp, vb, vb0 in series:
            wear = S.usui_flank_wear(t, sigma_n_Pa=sigma, sliding_m_s=vs, temp_K=temp, A_per_Pa=values["usui_A_per_Pa"],
                                     B_K=values["usui_B_K"], clearance_rad=clearance, rake_rad=rake, VB0_m=vb0)
            out.append(wear[1:])
        return {"VB_m": np.concatenate(out)}

    measured = np.concatenate([s[5] for s in series])
    obs = ObservationSet("VB_m", measured, None, "m", None, 0.0, "measured flank wear land width")
    constants = {f"material.tool_life_usui[{sel}].rate_unit": "m/s with sigma_n in Pa, V_s in m/s"}
    return Problem("usui", params, [obs], observe, loop=False, constants=constants,
                   notes=["A and B are determined separately only with tests at two or more temperatures"],
                   inputs={"tests": len(series), "clearance_rad": clearance, "rake_rad": rake, "tool_class": klass})


# ============================================================================ deflection and runout

def build_deflection(ctx: Context) -> Problem:
    rows = ctx.table("deflection", "part_measurements")
    if not rows:
        raise Skip("needs a part_measurements table (feature, measured_mm, simulated_mm, runout_sensitivity, "
                   "deflection_mm)")
    tool = _tool(ctx, "deflection")
    number = int(tool.number) if tool is not None else int(ctx.target_options("deflection").get("tool", 1))
    runout0 = _f(tool.runout_m, 0.0) if tool is not None else 0.0
    simulated_by_id = {}
    if ctx.run is not None and (Path(ctx.run.dir) / "inspection.json").exists():
        import json
        document = json.loads((Path(ctx.run.dir) / "inspection.json").read_text(encoding="utf-8"))
        for entry in document.get("tolerances", []):
            if entry.get("measured_m") is not None:
                simulated_by_id[str(entry.get("id"))] = float(entry["measured_m"])
    data = [_si_row(r) for r in rows]
    sim, meas, sens, defl = [], [], [], []
    for i, r in enumerate(data):
        feature = str(r.get("feature", i + 1))
        simulated = r.get("simulated_m", simulated_by_id.get(feature))
        if simulated is None or "measured_m" not in r:
            raise InputError(f"part_measurements row {i + 1} ({feature}): needs measured_mm and simulated_mm "
                             f"(or an inspection.json tolerance of that id)")
        sim.append(simulated)
        meas.append(r["measured_m"])
        sens.append(float(r.get("runout_sensitivity", 0.0)))
        defl.append(float(r.get("deflection_m", 0.0)))
    sim, meas, sens, defl = (np.array(v, dtype=np.float64) for v in (sim, meas, sens, defl))
    r_lo, r_hi = _bounds("runout_bounds_m")
    k_lo, k_hi = _bounds("stiffness_factor_bounds")
    params = _free(ctx, "deflection", [
        Param("tool_runout_m", _within(runout0, r_lo, r_hi), r_lo, r_hi, "m", bool(np.any(sens != 0)),
              targets=(Target(f"tools[{number}].runout_um"),), doc="radial runout of the cutting edge"),
        Param("tool_stiffness_factor", 1.0, k_lo, k_hi, "-", bool(np.any(defl != 0)),
              targets=(Target(f"factors.tool_stiffness[{number}]"),),
              doc="multiplier on the static stiffness of the tool chain at the tip")])

    def observe(sim_log, values, sim_effects):
        return {"feature_m": S.part_dimension(sim, runout_sensitivity=sens, deflection_m=defl,
                                              runout_m=values["tool_runout_m"], runout_sim_m=runout0,
                                              stiffness_factor=values["tool_stiffness_factor"])}

    obs = ObservationSet("feature_m", meas, None, "m", None, 0.0, "measured part features")
    return Problem("deflection", params, [obs], observe, loop=False,
                   notes=["the tool stiffness factor has no database key: it is written as factors.tool_stiffness "
                          "for the structure domain"], inputs={"features": len(meas), "tool": number})


# ============================================================================ Kienzle / mechanistic

def _cut_segments(ctx: Context, target: str) -> list[dict]:
    rows = ctx.table(target, "cuts")
    if not rows:
        raise Skip("needs cuts: [{line or t0_s/t1_s, ae_mm, ap_mm}] (with the run, or rpm, feed_mm_min, "
                   "diameter_mm, teeth)")
    if ctx.log is None:
        raise Skip("needs machine data")
    measured = ctx.log.first("spindle.power_W", "spindle.load_frac")
    if measured is None:
        raise Skip("needs measured spindle.power_W or spindle.load_frac")
    spindle = ctx.spindle()
    omega_max = float(spindle.max_rad_s)
    trim = float(calibration_params()["segment_trim_frac"])
    tool = _tool(ctx, target)
    geometry = tool.tool.geometry if tool is not None and tool.tool is not None else None
    segments = []
    blocks = ctx.run.timeline() if ctx.run is not None else []
    for i, raw in enumerate(rows):
        r = _si_row(raw)
        if "line" in r and blocks:
            spans = [(int(b["tick0"]) * ctx.run.tick_s, int(b["tick1"]) * ctx.run.tick_s) for b in blocks
                     if int(b.get("line") or -1) == int(r["line"])]
            if not spans:
                raise InputError(f"cuts row {i + 1}: the run has no block on line {int(r['line'])}")
            t0, t1 = spans[0]
        elif "t0_s" in r and "t1_s" in r:
            t0, t1 = float(r["t0_s"]), float(r["t1_s"])
        else:
            raise InputError(f"cuts row {i + 1}: give line (with a run) or t0_s and t1_s (simulated time)")
        span = t1 - t0
        t0, t1 = t0 + trim * span, t1 - trim * span
        if "ae_m" not in r or "ap_m" not in r:
            raise InputError(f"cuts row {i + 1}: needs ae_mm and ap_mm")
        times = np.linspace(t0, t1, 64)
        omega = r.get("rad_s")
        if omega is None and ctx.sim_log is not None and "spindle.speed_rad_s" in ctx.sim_log:
            omega = float(np.nanmean(ctx.sim_log.channels["spindle.speed_rad_s"].at(times)))
        feed = r.get("feed_m_s")
        if feed is None and ctx.sim_log is not None and "path.feed_m_s" in ctx.sim_log:
            feed = float(np.nanmean(ctx.sim_log.channels["path.feed_m_s"].at(times)))
        diameter = r.get("diameter_m", _f(geometry.diameter_m) if geometry is not None else None)
        teeth = r.get("teeth", _f(geometry.flutes) if geometry is not None else None)
        if None in (omega, feed, diameter, teeth) or not omega:
            raise InputError(f"cuts row {i + 1}: spindle speed, feed, tool diameter and teeth are needed "
                             f"(rpm, feed_mm_min, diameter_mm, teeth, or the run and the project's tool)")
        data = measured.finite()
        t_meas = ctx.alignment.to_meas(np.array([t0, t1])) if ctx.alignment is not None else np.array([t0, t1])
        inside = (data.t_s >= t_meas[0]) & (data.t_s <= t_meas[1])
        if inside.sum() < 2:
            raise InputError(f"cuts row {i + 1}: fewer than two measured samples in the segment")
        values = np.asarray(data.values[inside], dtype=np.float64)
        n_rev_s = abs(omega) / (2.0 * math.pi)
        segments.append({"t0_s": t0, "t1_s": t1, "omega": abs(omega), "n_rev_s": n_rev_s, "feed": feed,
                         "fz": feed / (int(round(teeth)) * n_rev_s), "teeth": int(round(teeth)), "R": 0.5 * diameter,
                         "ae": r["ae_m"], "ap": r["ap_m"], "mean": float(values.mean()),
                         "sem": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0})
    return segments, measured.name, spindle, omega_max


def _shaft_from_measured(ctx: Context, segments, channel: str, spindle, omega_max) -> np.ndarray:
    """Mean shaft power of each segment from the measured mean (power: eta (P - P_idle); load:
    lambda P_ref - P_bearing), with the spindle values calibrated before (else the machine file's)."""
    omega = np.array([s["omega"] for s in segments])
    mean = np.array([s["mean"] for s in segments])
    if channel == "spindle.power_W":
        eta = ctx.calibrated.get("spindle_efficiency", float(spindle.efficiency))
        idle = ctx.calibrated.get("spindle_idle_power_W", float(spindle.idle_power_W))
        x = float(model_params("production-energy")["palmgren_speed_exponent"])
        return eta * (mean - idle * (omega / omega_max) ** x)
    bearing = ctx.calibrated.get("spindle_bearing_power_W", float(spindle.thermal.bearing_heat_at_max_speed_W))
    x = float(model_params("thermal-machine")["bearing_heat_speed_exponent"])
    reference = _power_reference(spindle)(omega)
    return mean * reference - bearing * (omega / omega_max) ** x


def build_kienzle(ctx: Context) -> Problem:
    segments, channel, spindle, omega_max = _cut_segments(ctx, "kienzle")
    index, entry, raw_entries = _material_entry(ctx, "kienzle", "kienzle")
    if entry is None:
        raise Skip("the material has no kienzle entry")
    kc0, mc0, h_ref = float(entry.kc11_Pa), float(entry.mc), float(entry.h_ref_m)
    sel = _selector(raw_entries, index)
    k_lo, k_hi = _bounds("kc11_rel_bounds")
    m_lo, m_hi = _bounds("mc_bounds")
    params = _free(ctx, "kienzle", [
        Param("kc11_Pa", kc0, k_lo * kc0, k_hi * kc0, "Pa", True,
              targets=(Target(f"material.cutting.kienzle[{sel}].kc11_N_mm2"),), doc="Kienzle kc1.1"),
        Param("kc_mc", _within(mc0, m_lo, m_hi), m_lo, m_hi, "-", True,
              targets=(Target(f"material.cutting.kienzle[{sel}].mc"),), doc="Kienzle exponent mc")])
    shaft = _shaft_from_measured(ctx, segments, channel, spindle, omega_max)

    def observe(sim_log, values, sim_effects):
        return {"segment_shaft_power_W": np.array([
            S.milling_power_kienzle(spindle_rev_s=s["n_rev_s"], teeth=s["teeth"], radius_m=s["R"], ap_m=s["ap"],
                                    fz_m=s["fz"], ae_m=s["ae"], kc11_Pa=values["kc11_Pa"], mc=values["kc_mc"],
                                    h_ref_m=h_ref) for s in segments])}

    obs = ObservationSet("segment_shaft_power_W", shaft, None, "W", np.array([0.5 * (s["t0_s"] + s["t1_s"])
                                                                              for s in segments]), 0.0,
                         "mean shaft power of each cut (from the measured spindle signal)")
    return Problem("kienzle", params, [obs], observe, loop=False,
                   notes=["mc is determined only by cuts at different mean chip thicknesses (fz, ae)"],
                   inputs={"segments": segments, "entry": sel})


def build_mechanistic(ctx: Context) -> Problem:
    segments, channel, spindle, omega_max = _cut_segments(ctx, "mechanistic")
    index, entry, raw_entries = _material_entry(ctx, "mechanistic", "mechanistic")
    if entry is None:
        raise Skip("the material has no mechanistic entry")
    ktc0, kte0 = float(entry.Ktc_Pa), float(entry.Kte_N_per_m)
    sel = _selector(raw_entries, index)
    c_lo, c_hi = _bounds("cut_coeff_rel_bounds")
    e_hi = float(calibration_params()["edge_coeff_max_N_per_m"])
    params = _free(ctx, "mechanistic", [
        Param("Ktc_Pa", ktc0, c_lo * ktc0, c_hi * ktc0, "Pa", True,
              targets=(Target(f"material.cutting.mechanistic[{sel}].Ktc_N_mm2"),), doc="tangential cutting coefficient"),
        Param("Kte_N_per_m", _within(kte0, 0.0, e_hi), 0.0, e_hi, "N/m", True,
              targets=(Target(f"material.cutting.mechanistic[{sel}].Kte_N_mm"),), doc="tangential edge coefficient")])
    shaft = _shaft_from_measured(ctx, segments, channel, spindle, omega_max)

    def observe(sim_log, values, sim_effects):
        return {"segment_shaft_power_W": np.array([
            S.milling_power_mechanistic(spindle_rev_s=s["n_rev_s"], teeth=s["teeth"], radius_m=s["R"], ap_m=s["ap"],
                                        fz_m=s["fz"], ae_m=s["ae"], Ktc_Pa=values["Ktc_Pa"],
                                        Kte_N_per_m=values["Kte_N_per_m"]) for s in segments])}

    obs = ObservationSet("segment_shaft_power_W", shaft, None, "W", np.array([0.5 * (s["t0_s"] + s["t1_s"])
                                                                              for s in segments]), 0.0,
                         "mean shaft power of each cut (from the measured spindle signal)")
    return Problem("mechanistic", params, [obs], observe, loop=False,
                   notes=["Ktc and Kte separate only with cuts at different feeds per tooth"],
                   inputs={"segments": segments, "entry": sel})


BUILDERS = {"spindle": build_spindle, "servo": build_servo, "axis_friction": build_axis_friction,
            "block_time": build_block_time, "thermal_growth": build_thermal_growth, "tool_life": build_tool_life,
            "usui": build_usui, "deflection": build_deflection, "kienzle": build_kienzle,
            "mechanistic": build_mechanistic}


__all__ = ["TARGETS", "LOOP_TARGETS", "BUILDERS", "Problem", "Context", "Skip", "spindle_signal"]
