"""What the sound synthesis reads from a run directory, gathered into :class:`SoundInputs` (docs/models/sound.md
"Inputs"). A test or a study can build :class:`SoundInputs` directly (synthetic signals) instead.

* frames.bin: the unwrapped spindle angle (path 0) and the axis values per L1 tick.
* timeline.json + scene.json tools: the active tool's tooth count and radius per tick (no tool: no tooth pass).
* series, tick tier: the per-tick ``min/mean/max`` of the cutting force on the tool (``cutting.force_N[0..2]``).
* series, fast tier: the full-rate force where windows were recorded.
* series, macro tier: ``dynamics.chatter_freq_Hz``, ``dynamics.chatter_index``, ``dynamics.vibration_pp_m``,
  ``chips.packing_torque_Nm``, ``chips.emitted_kg`` (else ``removal.removed_volume_m3`` x density) and
  ``coolant.reach_fraction`` (coolant on when > 0).
* keyframes: coolant jets (kind, speed, flow) at each keyframe; chips (velocities, masses) for impact speeds.
* run.json project: spindle power and maximum speed, spindle modes (``frf_modes``), axis ballscrew pitch and motor
  power, coolant pumps, the material's modulus, Poisson ratio and density.
Channels that are absent leave their source silent (``SoundInputs.notes`` lists them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..render.rundir import RunData, db_item
from ..units import to_si
from . import params as P

FORCE_CHANNELS = ("cutting.force_N", "cutting.force_T0_N", "cutting.tool_force_N")
CHATTER_FREQ = "dynamics.chatter_freq_Hz"
CHATTER_INDEX = "dynamics.chatter_index"
VIBRATION_PP = "dynamics.vibration_pp_m"
PACKING_TORQUE = "chips.packing_torque_Nm"
CHIP_MASS = "chips.emitted_kg"
REMOVED_VOLUME = "removal.removed_volume_m3"
COOLANT_REACH = "coolant.reach_fraction"
AIR_JET_KINDS = ("air", "mql", "air_blast")
LIQUID_JET_KINDS = ("flood", "through_spindle", "high_pressure", "through_tool", "mist")


@dataclass
class FastWindow:
    t_first_s: float
    dt_s: float
    force_N: np.ndarray          # [n, 3]

    @property
    def t_last_s(self) -> float:
        return self.t_first_s + (len(self.force_N) - 1) * self.dt_s


@dataclass
class Jet:
    kind: str
    speed_m_s: float
    flow_m3_s: float
    diameter_m: float


@dataclass
class SoundInputs:
    end_s: float
    seed: int = 0
    # per L1 tick (frames)
    frame_t_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spindle_angle_rad: np.ndarray = field(default_factory=lambda: np.zeros(0))
    axis_names: list = field(default_factory=list)
    axis_linear: list = field(default_factory=list)
    axis_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    teeth: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    tool_radius_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # tick tier force envelopes (row time = start of the tick, span = tick_span_s)
    env_t_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    env_span_s: float = 1e-3
    env_min_N: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    env_mean_N: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    env_max_N: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    fast: list = field(default_factory=list)
    # macro tier (row time = end of the macro step, span = macro_span_s)
    macro_t_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    macro_span_s: float = 1e-2
    chatter_freq_Hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chatter_index: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vibration_pp_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    packing_torque_Nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chip_rate_kg_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    coolant_on: np.ndarray | None = None        # [n_macro] bool, None: from the jets alone
    jets_t_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    jets: list = field(default_factory=list)    # per keyframe: list[Jet]
    chip_speed_m_s: float | None = None
    # machine (None: sound-sources spindle_power_W / spindle_max_speed_rad_s)
    spindle_power_W: float | None = None
    spindle_max_rad_s: float | None = None
    spindle_modes: list = field(default_factory=list)   # [(fn_Hz, zeta, k_N_per_m, direction)]
    axis_lead_m: dict = field(default_factory=dict)
    axis_motor_power_W: dict = field(default_factory=dict)
    pumps: dict = field(default_factory=dict)            # kind -> power_W
    enclosure_area_m2: float | None = None
    # chip material
    chip_youngs_Pa: float | None = None
    chip_poisson: float | None = None
    chip_density_kg_m3: float | None = None
    liquid_density_kg_m3: float | None = None
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------- helpers
def _sourced(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    if isinstance(value, dict) and "table" in value and value["table"]:
        return value["table"][0][1]
    if isinstance(value, dict) and "x" in value and value.get("values"):
        # a property table as run.json's resolved project writes it ({x, x_unit, values}): its first point (the
        # lowest temperature tabulated, room temperature for the elastic constants)
        return value["values"][0]
    return value


def si_value(mapping: dict | None, si_key: str):
    """The value of ``si_key`` in a mapping written with SI keys or with unit-suffixed file keys
    (``max_rpm`` read as ``max_rad_s``); sourced values ``{value, src}`` and one-row tables unwrapped."""
    if not isinstance(mapping, dict):
        return None
    if si_key in mapping:
        return _sourced(mapping[si_key])
    for key, raw in mapping.items():
        try:
            k_si, v_si = to_si(key, _sourced(raw))
        except (TypeError, ValueError):
            continue
        if k_si == si_key:
            return v_si
    return None


def _modes(entries, default_direction: str = "x") -> list:
    out = []
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        fn = si_value(item, "fn_Hz")
        zeta = si_value(item, "zeta")
        k = si_value(item, "k_N_per_m")
        if fn is None or zeta is None or k is None:
            continue
        out.append((float(fn), float(zeta), float(k), str(item.get("direction", default_direction))))
    return out


def _fill(run_value, db_value):
    """The run's (resolved) value with the database item's keys filled in where the run has none; lists of
    mappings (spindles, pumps) merge element by element on their ``id``."""
    if isinstance(run_value, dict) and isinstance(db_value, dict):
        out = dict(db_value)
        out.update({k: _fill(v, db_value.get(k)) for k, v in run_value.items()})
        return out
    if isinstance(run_value, list) and isinstance(db_value, list) and all(isinstance(x, dict) for x in run_value):
        by_id = {x.get("id"): x for x in db_value if isinstance(x, dict)}
        return [_fill(x, by_id.get(x.get("id"))) for x in run_value]
    return run_value if run_value is not None else db_value


def _machine(project: dict) -> dict:
    """The resolved machine of run.json, completed from its database item (the run's values win)."""
    machine = project.get("machine") or {}
    item = db_item("machines", machine.get("id")) if machine.get("id") else None
    return _fill(machine, item) if item else dict(machine)


def _spindle(machine: dict) -> dict:
    spindles = machine.get("spindles") or []
    return spindles[0] if spindles and isinstance(spindles[0], dict) else {}


def _spindle_power(spindle: dict) -> float | None:
    """Largest power of the spindle's gear-range power curves ([[speed, power], ...]), or its power entry."""
    best = None
    for gear in spindle.get("gear_ranges") or []:
        if not isinstance(gear, dict):
            continue
        curve = si_value(gear, "power_W")
        if curve:
            peak = max(float(row[1]) for row in curve)
            best = peak if best is None else max(best, peak)
    if best is None:
        value = si_value(spindle, "power_W")
        best = float(value) if value is not None else None
    return best


def _material(project: dict) -> dict:
    """Physical data of the work material: the run's values over the database item's ``physical`` section."""
    material = project.get("material") or {}
    item = db_item("materials", material.get("id")) or {}
    physical = dict(item.get("physical") or {})
    physical.update(material.get("physical") or {})
    for key in ("density_kg_m3", "youngs_modulus_Pa", "youngs_modulus_GPa", "poisson"):
        if key in material:
            physical[key] = material[key]
    return physical


def _mean_dt(t: np.ndarray) -> float:
    return float(np.median(np.diff(t))) if len(t) > 1 else 0.0


# ----------------------------------------------------------------------------- from a run directory
def from_run(run_dir, seed: int | None = None) -> SoundInputs:
    run = run_dir if isinstance(run_dir, RunData) else RunData(run_dir)
    tick_s = run.tick_s
    inputs = SoundInputs(end_s=run.end_tick * tick_s,
                         seed=int(seed if seed is not None else run.run.get("seed", run.project.get("seed", 0)) or 0))
    _frames(run, inputs)
    _forces(run, inputs)
    _macro(run, inputs)
    _coolant(run, inputs)
    _machine_data(run, inputs)
    return inputs


def _frames(run: RunData, inputs: SoundInputs) -> None:
    ticks = run.frame_ticks
    inputs.frame_t_s = ticks * run.tick_s
    inputs.spindle_angle_rad = run.spindle_angle(0)
    names, values = run.axis_values()
    header = run.scene.get("frames") or {}
    kinds = {a.get("name"): a.get("kind", "linear") for a in header.get("axes", [])}
    inputs.axis_names = list(names)
    inputs.axis_linear = [kinds.get(n, "linear") == "linear" for n in names]
    inputs.axis_pos = values
    teeth = np.zeros(len(ticks), np.int64)
    radius = np.zeros(len(ticks))
    if len(ticks) and run.timeline:
        starts = np.array([int(b.get("tick0", 0)) for b in run.timeline], np.int64)
        numbers, current = [], None
        for block in run.timeline:
            if block.get("tool") is not None:
                current = int(block["tool"])
            numbers.append(current)
        index = np.clip(np.searchsorted(starts, ticks, side="right") - 1, 0, len(starts) - 1)
        lookup = {}
        for number in set(n for n in numbers if n is not None):
            entry = run.tool_entry(number) or {}
            lookup[number] = (int(entry.get("teeth", 0) or 0), 0.5 * float(entry.get("diameter_m", 0.0) or 0.0))
        for i, block_index in enumerate(index):
            number = numbers[int(block_index)]
            if number is not None and number in lookup:
                teeth[i], radius[i] = lookup[number]
    inputs.teeth = teeth
    inputs.tool_radius_m = radius
    if not len(ticks):
        inputs.notes.append("frames.bin is empty: no spindle, axis or tooth-pass sound")


def _force_channel(run: RunData) -> str | None:
    for name in FORCE_CHANNELS:
        if name in run.channels:
            return name
    return None


def _forces(run: RunData, inputs: SoundInputs) -> None:
    name = _force_channel(run)
    if name is None:
        inputs.notes.append("no cutting force channel: no tooth-pass sound")
        return
    columns = run.channel_columns(name)[:3]
    tick_cols = [f"{c}.{s}" for c in columns for s in ("min", "mean", "max")]
    data = run.series("tick", tick_cols)
    if len(data["tick"]) and all(c in data for c in tick_cols):
        inputs.env_t_s = data["tick"] * run.tick_s
        inputs.env_span_s = run.ipo_ticks * run.tick_s
        stack = lambda s: np.nan_to_num(np.stack([data[f"{c}.{s}"] for c in columns], axis=1))  # noqa: E731
        inputs.env_min_N, inputs.env_mean_N, inputs.env_max_N = stack("min"), stack("mean"), stack("max")
    else:
        inputs.notes.append(f"no tick-tier envelopes of {name}: tooth passes only inside fast windows")
    fast = run.series("fast", columns)
    if len(fast["tick"]) > 1 and all(c in fast for c in columns):
        t = fast["tick"] * run.tick_s
        force = np.nan_to_num(np.stack([fast[c] for c in columns], axis=1))
        dt = _mean_dt(t)
        breaks = np.flatnonzero(np.diff(t) > 1.5 * dt) + 1
        for chunk in np.split(np.arange(len(t)), breaks):
            if len(chunk) >= 4:
                inputs.fast.append(FastWindow(float(t[chunk[0]]), dt, np.ascontiguousarray(force[chunk])))


def _macro(run: RunData, inputs: SoundInputs) -> None:
    wanted = {CHATTER_FREQ: "chatter_freq_Hz", CHATTER_INDEX: "chatter_index", VIBRATION_PP: "vibration_pp_m",
              PACKING_TORQUE: "packing_torque_Nm"}
    columns = [c for c in list(wanted) + [CHIP_MASS, REMOVED_VOLUME, COOLANT_REACH] if c in run.channels]
    data = run.series("macro", columns)
    t = data["tick"] * run.tick_s
    inputs.macro_t_s = t
    clock = run.run.get("clock") or {}
    macro_ticks = int(clock.get("macro_ipo", 10)) * run.ipo_ticks
    inputs.macro_span_s = macro_ticks * run.tick_s
    for channel, attr in wanted.items():
        if channel in data:
            setattr(inputs, attr, np.nan_to_num(data[channel]))
        else:
            setattr(inputs, attr, np.zeros(len(t)))
            if channel != PACKING_TORQUE:
                inputs.notes.append(f"no {channel}: no chatter sound outside fast windows")
    mass = None
    if CHIP_MASS in data:
        mass = np.nan_to_num(data[CHIP_MASS])
    elif REMOVED_VOLUME in data:
        density = si_value(_material(run.project), "density_kg_m3")
        if density is not None:
            mass = np.nan_to_num(data[REMOVED_VOLUME]) * float(density)
    if mass is not None and len(t) > 1:
        rate = np.zeros(len(t))
        rate[1:] = np.maximum(np.diff(mass), 0.0) / np.maximum(np.diff(t), 1e-12)
        rate[0] = mass[0] / max(inputs.macro_span_s, 1e-12)
        inputs.chip_rate_kg_s = rate
    else:
        inputs.chip_rate_kg_s = np.zeros(len(t))
        inputs.notes.append("no chip mass or removed volume channel: chip impacts follow the tooth passes")
    if COOLANT_REACH in data:
        inputs.coolant_on = np.nan_to_num(data[COOLANT_REACH]) > 0.0


def _coolant(run: RunData, inputs: SoundInputs) -> None:
    times, jets_per_kf, speeds = [], [], []
    nozzles = {n.get("id"): n for n in run.scene.get("nozzles", []) or []}
    for k, entry in enumerate(run.keyframes):
        fields = run.fields(k)
        kinds = [str(x) for x in np.asarray(fields.get("jet_kind", []), dtype=str).reshape(-1)]
        ids = [str(x) for x in np.asarray(fields.get("jet_nozzle_id", []), dtype=str).reshape(-1)]
        speed = np.asarray(fields.get("jet_speed_m_s", np.zeros(len(kinds))), dtype=np.float64).reshape(-1)
        flow = np.asarray(fields.get("jet_flow_m3_s", np.zeros(len(kinds))), dtype=np.float64).reshape(-1)
        jets = []
        for i, kind in enumerate(kinds):
            nozzle = nozzles.get(ids[i] if i < len(ids) else None) or {}
            diameter = si_value(nozzle, "diameter_m")
            if diameter is None and speed[i] > 0.0:
                diameter = 2.0 * math.sqrt(flow[i] / (math.pi * speed[i])) if flow[i] > 0.0 else None
            if diameter is None:
                diameter = float(P.number(P.SOURCES, "air_nozzle_diameter_m"))
            jets.append(Jet(kind, float(speed[i]), float(flow[i]), float(diameter)))
        times.append(float(entry["tick"]) * run.tick_s)
        jets_per_kf.append(jets)
        chips = run.chips(k)
        vel = np.asarray(chips.get("vel_M_m_s", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        if len(vel):
            speeds.append(np.linalg.norm(vel, axis=1))
    inputs.jets_t_s = np.asarray(times)
    inputs.jets = jets_per_kf
    if speeds:
        inputs.chip_speed_m_s = float(np.median(np.concatenate(speeds)))


def _machine_data(run: RunData, inputs: SoundInputs) -> None:
    project = run.project
    machine = _machine(project)
    spindle = _spindle(machine)
    power = _spindle_power(spindle)
    if power is not None:
        inputs.spindle_power_W = power
    else:
        inputs.notes.append("spindle power unknown: the spindle motor level uses sound-sources spindle_power_W")
    max_speed = si_value(spindle, "max_rad_s")
    if max_speed is not None:
        inputs.spindle_max_rad_s = float(max_speed)
    inputs.spindle_modes = _modes(spindle.get("frf_modes"))
    structure = machine.get("structure") or {}
    inputs.spindle_modes += [m for m in _modes(structure.get("modes")) if m[3] == "z"]
    for name, axis in (machine.get("axes") or {}).items():
        if not isinstance(axis, dict):
            continue
        screw = axis.get("ballscrew") or {}
        lead = si_value(screw, "pitch_m") or si_value(screw, "lead_m")
        if lead is not None:
            inputs.axis_lead_m[str(name)] = float(lead)
        motor = si_value(axis, "motor_power_W")
        if motor is not None:
            inputs.axis_motor_power_W[str(name)] = float(motor)
    for pump in ((machine.get("coolant") or {}).get("pumps") or []):
        if isinstance(pump, dict):
            power_W = si_value(pump, "power_W")
            if power_W is not None:
                inputs.pumps[str(pump.get("kind", pump.get("id", "flood")))] = float(power_W)
    box = si_value(machine.get("enclosure") or {}, "box_m")
    if box is not None:
        lo, hi = np.asarray(box, dtype=np.float64).reshape(2, 3)
        size = np.abs(hi - lo)
        inputs.enclosure_area_m2 = float(2.0 * (size[0] * size[1] + size[1] * size[2] + size[0] * size[2]))
    physical = _material(project)
    youngs = si_value(physical, "youngs_modulus_Pa")
    inputs.chip_youngs_Pa = float(youngs) if youngs is not None else None
    poisson = si_value(physical, "poisson")
    inputs.chip_poisson = float(poisson) if poisson is not None else None
    density = si_value(physical, "density_kg_m3")
    inputs.chip_density_kg_m3 = float(density) if density is not None else None
    fluid = (project.get("coolant") or {}).get("fluid") or {}
    item = db_item("coolants", fluid.get("id") if isinstance(fluid, dict) else fluid) or {}
    density = _sourced(item.get("density_kg_m3")) if item else None
    inputs.liquid_density_kg_m3 = float(density) if density is not None else None
