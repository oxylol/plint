"""The machine as motion sees it: axes and their drives, paths, kinematic chains, spindles, error map.

Built from a machine definition (``db/machines/<id>.yaml`` as a :class:`cnctwin.db.SIMap`, or a plain
mapping wrapped with :func:`cnctwin.db.simap.wrap`). Machine-file values always win; quantities a file
does not give come from ``db/models/motion.yaml`` and are listed in :attr:`MachineModel.estimated`.

* **Paths** follow the controller's documented rule (docs/dialects.md): ``controller.two_path`` or
  type ``swiss`` groups the axes by the trailing digit of their names; otherwise one path holds every
  axis; ``physics.controller.paths`` (a list of ``{letter: axis}``) overrides. Axis order within a
  path: X, (Y), Z first (lathes X, Z), then the file order. The global axis order is the paths in
  order, each in its own order.
* **Chains**: a path uses the workpiece chain and the tool chain that contain its axes (the first of
  ``workpiece_chain`` + ``workpiece_chains.*`` / ``tool_chain`` + ``tool_chains.*``, names sorted),
  else the main ones. Link i: ``T_i = T_{i-1} J_i(q) E_i(q) X_i`` (cnctwin.motion.abi.Chain).
* **Work spindles** (kind turning/sub) turn the last rotary link of their workpiece chain (the i-th
  turning spindle the i-th workpiece chain with a rotary link); a chain without one gets a synthetic
  rotary link about the chuck-face z axis. Angles follow ISO 841: the spindle reference axis points
  from the work toward the spindle (-z of the chuck face), so ``q_C = kappa * theta`` with
  ``kappa = (-z_F) . d_C``.
* **Error map** (ISO 230-1): squareness tilts the second axis of each pair toward the first (relative
  tool-to-work sense), straightness is a bow 4 D xi (1 - xi) in both directions normal to the axis
  (D = value x travel length), angular errors (pitch, yaw, roll) vary linearly from -e to +e over the
  travel, pivot errors move and tilt the rotary axis line, ``y_axis_centre_height_um`` offsets the Y
  link. They are per-link error transforms of the chain, never axis offsets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from ..db.models import params as model_params
from ..db.simap import PoseMap, SIMap, Sourced, wrap
from .abi import CHAIN_TOOL, CHAIN_WORK, LINK_FIXED, LINK_LINEAR, LINK_ROTARY, Chain, eye_stack

LATHE_TYPES = ("lathe", "mill_turn", "swiss")
TOOL_SPINDLE_KINDS = ("milling", "live_tool")
WORK_SPINDLE_KINDS = ("turning", "sub")
# Keys a sourced database mapping may carry beside its entries (docs/formats.md "Sources and estimates"): they
# describe the mapping, they are not axes (the controller's machine model skips the same keys, docs/dialects.md).
SOURCE_KEYS = frozenset(("src", "note", "notes", "page", "pages", "est", "estimated", "accessed", "digitised",
                         "uncertainty_pct"))


def axis_names(axes) -> list[str]:
    """Names of the axes of a machine's ``axes`` mapping in file order: the entries that are axis definitions
    (mappings), without the source keys a sourced mapping carries beside them (``src``, ``note``, ``page`` ...)."""
    if axes is None:
        return []
    names = []
    for name in axes.keys():
        if str(name) in SOURCE_KEYS:
            continue
        if isinstance(axes[name], Mapping):
            names.append(str(name))
    return names


def plain(value):
    if isinstance(value, Sourced):
        return value.value
    return value


def num(value, default=None):
    value = plain(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vec(value, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    value = plain(value)
    if value is None:
        return np.array(default, dtype=float)
    return np.array([float(plain(v)) for v in value], dtype=float)


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n == 0.0:
        raise InputError(f"{what}: direction {v.tolist()} is not a non-zero vector")
    return v / n


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name)
        return default if value is None else value
    value = getattr(obj, name, None)
    return default if value is None else value


@dataclass
class LinkDef:
    name: str
    kind: int
    axis: str | None
    direction: np.ndarray                     # actual (with squareness / pivot errors)
    point: np.ndarray
    fixed: np.ndarray
    synthetic_spindle: str | None = None      # synthetic rotary link turned by this spindle
    direction_nom: np.ndarray | None = None   # nominal (the control's kinematic model)
    point_nom: np.ndarray | None = None

    def __post_init__(self):
        if self.direction_nom is None:
            self.direction_nom = self.direction.copy()
        if self.point_nom is None:
            self.point_nom = self.point.copy()


@dataclass
class ChainDef:
    name: str
    side: int
    links: list[LinkDef]

    def axis_names(self) -> list[str]:
        return [l.axis for l in self.links if l.axis is not None]

    def rotary_links(self, axes: dict) -> list[int]:
        return [i for i, l in enumerate(self.links) if l.kind == LINK_ROTARY]


@dataclass
class AxisDef:
    name: str
    index: int
    path: int
    letter: str
    rotary: bool
    continuous: bool
    lo: float
    hi: float
    home: float
    rapid: float              # SI per second
    max_feed: float
    accel: float
    jerk: float
    kv: float
    kff: float
    bw_Hz: float
    mass: float               # kg or kg m^2
    force_max: float          # N or N m
    backlash: float
    spike: float
    scale_fb: bool
    synthetic: bool = False   # a spindle angle given as an axis (no NC axis in the file)
    spindle: str | None = None
    kappa: float = 1.0
    chain: str = ""
    side: int = -1
    link: int = -1
    direction_M: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity_N: float = 0.0

    @property
    def scale(self) -> float:
        """Path-metric scale: 1 for linear axes, m per rad (1 deg = 1 mm) for rotary axes."""
        return 1.0


@dataclass
class GearDef:
    lo: float
    hi: float
    curve_w: np.ndarray
    curve_t: np.ndarray


@dataclass
class SpindleDef:
    id: str
    kind: str
    index: int
    max_omega: float
    gears: list[GearDef]
    accel_time: float | None
    decel_time: float | None
    inertia: float
    drag_max: float
    bw_Hz: float
    pos_kv: float
    pos_kff: float
    pos_max_accel: list[float]
    orient_s: float
    gear_change_s: float
    reversible: bool
    c_engage_s: float = 0.0     # entering C-axis mode at rest (machine c_axis_engage_time_s, else the orientation)
    c_axis: str | None = None
    kappa: float = 1.0
    chain: str = ""

    @property
    def accel_rate(self) -> float:
        return self.max_omega / self.accel_time if self.accel_time else math.inf

    @property
    def decel_rate(self) -> float:
        return self.max_omega / self.decel_time if self.decel_time else math.inf

    def gear_for(self, omega: float) -> int:
        w = abs(omega)
        for i, g in enumerate(self.gears):
            if g.lo - 1e-9 <= w <= g.hi + 1e-9:
                return i
        return len(self.gears) - 1 if self.gears else 0

    def torque_at(self, omega: float, gear: int) -> float:
        if not self.gears:
            return 0.0
        g = self.gears[min(max(gear, 0), len(self.gears) - 1)]
        return float(np.interp(abs(omega), g.curve_w, g.curve_t))


@dataclass
class PathDef:
    index: int
    axes: list[str]
    letters: dict[str, str]
    work_chain: str
    tool_chain: str
    workpiece: int
    work_spindle: str | None
    tool_spindle: str | None


class MachineModel:
    """Read-only motion view of one machine definition."""

    def __init__(self, machine, path_letters: list[dict] | None = None):
        m = machine if isinstance(machine, SIMap) else wrap(machine)
        self.raw = m
        self.p = model_params("motion")
        self.estimated: list[str] = []
        self.id = str(plain(m.id) or "")
        self.type = str(plain(m.type) or "vertical_mill")
        self.lathe = self.type in LATHE_TYPES
        ctrl = m.controller
        self.dialect = str(plain(_get(ctrl, "dialect", "fanuc")))
        self.lin_increment = num(_get(ctrl, "least_input_increment_m"), 1e-6) or 1e-6
        self.rot_increment = num(_get(ctrl, "least_input_increment_rad"), math.radians(0.001)) or math.radians(0.001)
        self.ipo_s = num(_get(ctrl, "ipo_period_s"), 1e-3)
        self.block_processing_s = num(_get(ctrl, "block_processing_s"), 0.0)
        self.lookahead = int(num(_get(ctrl, "lookahead_blocks"), 1) or 1)
        self.max_feed = num(_get(ctrl, "max_feed_m_s"), math.inf)
        self.in_position = num(_get(ctrl, "in_position_m"), None)
        if self.in_position is None:
            self.in_position = self.p["default_in_position_m"]
            self.estimated.append("controller.in_position_mm")
        self.corner_default = str(plain(_get(ctrl, "corner_default", "blend")))
        # length of the control's axis setpoint filter (moving average after interpolation: FANUC parameter 1769
        # in AI contour mode, the TNC nominal-position / HSC filter, SINUMERIK MD32410 axial jerk filter); 0 = the
        # control limits jerk before interpolation only, which leaves the contour unfiltered
        self.setpoint_filter_s = num(_get(ctrl, "setpoint_filter_s"), 0.0) or 0.0
        self.smoothing = [{"code": str(plain(e.code)), "name": str(plain(e.name) or ""),
                           "tolerance_m": num(e.tolerance_m, 0.0)} for e in (_get(ctrl, "smoothing") or [])]
        self.rapid_override_steps = [float(plain(v)) for v in (plain(_get(ctrl, "rapid_override_steps_frac")) or [])]
        self._ctrl_rapid = _get(ctrl, "rapid_m_s")
        self.kin = m.kinematics
        if self.kin is None:
            raise InputError(f"machine {self.id}: no kinematics section")
        self.tool_axis_N = _unit(_vec(_get(self.kin, "tool_axis"), (0.0, 0.0, 1.0)), "kinematics.tool_axis")
        self.rot_len = self.p["rotary_length_per_rad_m"]
        self._read_chains()
        self._read_paths(path_letters)
        self._read_axes()
        self._read_spindles()
        self._assign_work_spindles()
        self._read_errors()
        self.singularities = [{"axis": str(plain(s.axis)), "when": str(plain(s.when)),
                               "behaviour": str(plain(s.behaviour))} for s in (_get(self.kin, "singularities") or [])]
        self.tool_changer = m.tool_changer
        self.turrets = list(m.turrets or [])
        self.pallet_changer = m.pallet_changer
        self.doors = m.doors
        self.probes = m.probes
        self.coolant = m.coolant

    # ------------------------------------------------------------------ chains
    def _parse_chain(self, name: str, side: int, links) -> ChainDef:
        out = []
        axes_section = self.raw.axes or {}
        for i, link in enumerate(links or []):
            lname = str(plain(_get(link, "link", f"{name}_{i}")))
            axis = plain(_get(link, "axis"))
            if axis is not None:
                axis = str(axis)
                adef = axes_section.get(axis) if hasattr(axes_section, "get") else None
                if adef is None:
                    raise InputError(f"machine {self.id}: chain {name} link {lname} names axis {axis} that is "
                                     f"not under axes")
                kind = LINK_ROTARY if str(plain(adef.type)) == "rotary" else LINK_LINEAR
                direction = _unit(_vec(_get(link, "direction")), f"{name}.{lname}")
                point = _vec(_get(link, "axis_point_m"))
                out.append(LinkDef(lname, kind, axis, direction, point, np.eye(4)))
            else:
                fixed = _get(link, "fixed")
                pose = fixed.matrix() if isinstance(fixed, PoseMap) else np.eye(4)
                out.append(LinkDef(lname, LINK_FIXED, None, np.zeros(3), np.zeros(3), pose))
        return ChainDef(name, side, out)

    def _read_chains(self) -> None:
        kin = self.kin
        self.chains: dict[str, ChainDef] = {}
        self.work_chains: list[str] = []
        self.tool_chains: list[str] = []
        main_w = _get(kin, "workpiece_chain")
        main_t = _get(kin, "tool_chain")
        if not main_w or not main_t:
            raise InputError(f"machine {self.id}: kinematics needs workpiece_chain and tool_chain")
        self.chains["workpiece_chain"] = self._parse_chain("workpiece_chain", CHAIN_WORK, main_w)
        self.work_chains.append("workpiece_chain")
        for key in sorted((_get(kin, "workpiece_chains") or {}).keys()):
            name = f"workpiece_chains.{key}"
            self.chains[name] = self._parse_chain(name, CHAIN_WORK, kin.workpiece_chains[key])
            self.work_chains.append(name)
        self.chains["tool_chain"] = self._parse_chain("tool_chain", CHAIN_TOOL, main_t)
        self.tool_chains.append("tool_chain")
        for key in sorted((_get(kin, "tool_chains") or {}).keys()):
            name = f"tool_chains.{key}"
            self.chains[name] = self._parse_chain(name, CHAIN_TOOL, kin.tool_chains[key])
            self.tool_chains.append(name)

    # ------------------------------------------------------------------ paths
    def _read_paths(self, override) -> None:
        names = axis_names(self.raw.axes)
        ctrl = self.raw.controller
        two_path = bool(plain(_get(ctrl, "two_path", False))) or self.type == "swiss"
        if override:
            letters = [dict(p) for p in override]
        elif two_path:
            groups: dict[int, dict[str, str]] = {}
            for name in names:
                digit = name[-1] if name[-1].isdigit() else "1"
                letter = name[0] if name[-1].isdigit() else name
                groups.setdefault(int(digit), {})[letter] = name
            letters = [groups[k] for k in sorted(groups)]
        else:
            single: dict[str, str] = {}
            for name in names:
                letter = name[0] if (len(name) > 1 and name[-1].isdigit()) else name
                if letter not in single:
                    single[letter] = name
                else:
                    single[name] = name
            letters = [single]
        self.paths: list[PathDef] = []
        for p, mapping in enumerate(letters):
            first = ["X", "Z"] if self.lathe else ["X", "Y", "Z"]
            order = [l for l in first if l in mapping] + [l for l in mapping if l not in first]
            axes = [mapping[l] for l in order]
            axset = set(axes)
            work = next((c for c in self.work_chains if axset & set(self.chains[c].axis_names())), "workpiece_chain")
            tool = next((c for c in self.tool_chains if axset & set(self.chains[c].axis_names())), "tool_chain")
            self.paths.append(PathDef(p, axes, {l: mapping[l] for l in order}, work, tool,
                                      self.work_chains.index(work), None, None))

    # ------------------------------------------------------------------ axes and drives
    def _axis_value(self, axis, key, default_key, what, scale=1.0):
        value = num(_get(axis, key))
        if value is None:
            self.estimated.append(what)
            return self.p[default_key] * scale
        return value

    def _read_axes(self) -> None:
        home_lin = _get(self.kin, "home_m") or {}
        home_rot = _get(self.kin, "home_rad") or {}
        self.axes: list[AxisDef] = []
        self.axis_index: dict[str, int] = {}
        p = self.p
        for path in self.paths:
            for letter, name in path.letters.items():
                a = self.raw.axes[name]
                rotary = str(plain(a.type)) == "rotary"
                continuous = bool(plain(_get(a, "continuous", False)))
                travel = plain(a.travel_rad if rotary else a.travel_m)
                if travel is not None and not continuous:
                    lo, hi = float(plain(travel[0])), float(plain(travel[1]))
                else:
                    lo, hi = -math.inf, math.inf
                home = num((home_rot if rotary else home_lin).get(name) if hasattr(home_rot if rotary else home_lin,
                                                                                    "get") else None, 0.0)
                if rotary:
                    rapid = num(_get(a, "rapid_rad_s")) or num(_get(a, "max_rad_s"))
                    max_feed = num(_get(a, "max_rad_s"))
                    if rapid is None:
                        rapid = self._ctrl_rapid_of(name)
                    if max_feed is None:
                        max_feed = rapid
                    accel = self._axis_value(a, "accel_rad_s2", "default_accel_rad_s2", f"axes.{name}.accel_rad_s2")
                    jerk = self._axis_value(a, "jerk_rad_s3", "default_jerk_rad_s3", f"axes.{name}.jerk_rad_s3")
                    inertia = num(_get(a, "inertia_kgm2"))
                    torque = num(_get(a, "motor_torque_Nm"))
                    if inertia is None:
                        if torque is not None and accel > 0:
                            inertia = torque / (p["force_margin"] * accel)
                        else:
                            inertia = p["default_rotary_inertia_kgm2"]
                        self.estimated.append(f"axes.{name}.inertia_kgm2")
                    if torque is None:
                        torque = p["force_margin"] * inertia * accel
                        self.estimated.append(f"axes.{name}.motor_torque_Nm")
                    mass, fmax = inertia, torque
                    backlash = num(_get(a, "backlash_rad"))
                    if backlash is None:
                        backlash = p["default_backlash_m"] / self.rot_len
                        self.estimated.append(f"axes.{name}.backlash_arcsec")
                    spike = num(_get(a, "reversal_spike_rad"), p["default_reversal_spike_m"] / self.rot_len)
                else:
                    rapid = num(_get(a, "rapid_m_s")) or self._ctrl_rapid_of(name)
                    max_feed = num(_get(a, "max_feed_m_s"), rapid)
                    accel = self._axis_value(a, "accel_m_s2", "default_accel_m_s2", f"axes.{name}.accel_m_s2")
                    jerk = self._axis_value(a, "jerk_m_s3", "default_jerk_m_s3", f"axes.{name}.jerk_m_s3")
                    mass = self._axis_value(a, "moving_mass_kg", "default_moving_mass_kg", f"axes.{name}.moving_mass_kg")
                    fmax = num(_get(a, "motor_force_N"))
                    if fmax is None:
                        fmax = p["force_margin"] * mass * accel
                        self.estimated.append(f"axes.{name}.motor_force_N")
                    backlash = self._axis_value(a, "backlash_m", "default_backlash_m", f"axes.{name}.backlash_um")
                    spike = self._axis_value(a, "reversal_spike_m", "default_reversal_spike_m",
                                             f"axes.{name}.reversal_spike_um")
                if not rapid:
                    raise InputError(f"machine {self.id}: axis {name} has no rapid rate")
                servo = _get(a, "servo")
                kv = num(_get(servo, "kv_1_s"))
                if kv is None:
                    kv = p["default_kv_1_s"]
                    self.estimated.append(f"axes.{name}.servo.kv_1_s")
                kff = num(_get(servo, "feedforward"))
                if kff is None:
                    kff = p["default_feedforward"]
                    self.estimated.append(f"axes.{name}.servo.feedforward")
                bw = num(_get(servo, "velocity_bw_Hz"))
                if bw is None:
                    bw = p["default_velocity_bw_Hz"]
                    self.estimated.append(f"axes.{name}.servo.velocity_bw_Hz")
                scale_fb = str(plain(_get(a, "feedback", ""))) == "linear_scale"
                idx = len(self.axes)
                self.axis_index[name] = idx
                self.axes.append(AxisDef(name, idx, path.index, letter, rotary, continuous, lo, hi, home, float(rapid),
                                         float(max_feed), float(accel), float(jerk), float(kv), float(kff), float(bw),
                                         float(mass), float(fmax), float(backlash), float(spike), scale_fb))
        # where each axis sits in the chains, its direction in M at zero, gravity along it
        gravity = _vec(_get(self.kin, "gravity_direction"), (-1.0, 0.0, 0.0) if self.lathe else (0.0, 0.0, -1.0))
        gravity = _unit(gravity, "kinematics.gravity_direction") * 9.80665
        counter = self.p["gravity_counterbalanced_frac"]
        for name, chain in self.chains.items():
            R = np.eye(3)
            for i, link in enumerate(chain.links):
                if link.axis is not None and link.axis in self.axis_index:
                    ax = self.axes[self.axis_index[link.axis]]
                    if ax.chain == "":
                        ax.chain, ax.side, ax.link = name, chain.side, i
                        ax.direction_M = R @ link.direction
                        if not ax.rotary:
                            ax.gravity_N = float(ax.mass * (gravity @ ax.direction_M)) * (1.0 - counter)
                if link.kind == LINK_FIXED:
                    R = R @ link.fixed[:3, :3]

    def _ctrl_rapid_of(self, name: str) -> float:
        rapid = self._ctrl_rapid
        if isinstance(rapid, Mapping):
            value = rapid.get(name)
            return num(value, 0.0) or 0.0
        return num(rapid, 0.0) or 0.0

    # ------------------------------------------------------------------ spindles
    def _read_spindles(self) -> None:
        p = self.p
        self.spindles: list[SpindleDef] = []
        for index, s in enumerate(self.raw.spindles or []):
            sid = str(plain(s.id))
            max_w = num(s.max_rad_s, 0.0) or 0.0
            gears = []
            for g in s.gear_ranges or []:
                span = plain(_get(g, "rad_s")) or [0.0, max_w]
                curve = _get(g, "torque_Nm")
                if curve is not None and hasattr(curve, "x") and len(curve):
                    w, t = np.asarray(curve.x, float), np.asarray(curve.y, float)
                else:
                    w, t = np.array([0.0, max_w]), np.array([0.0, 0.0])
                gears.append(GearDef(float(plain(span[0])), float(plain(span[1])), w, t))
            inertia = num(_get(s, "rotor_inertia_kgm2"))
            if inertia is None:
                inertia = p["default_spindle_inertia_kgm2"]
                self.estimated.append(f"spindles.{sid}.rotor_inertia_kgm2")
            idle = num(_get(s, "idle_power_W"), 0.0) or 0.0
            loop = _get(s, "position_loop")
            bw = num(_get(loop, "velocity_bw_Hz"))
            if bw is None:
                bw = p["default_spindle_bw_Hz"]
            pos_kv = num(_get(loop, "kv_1_s"))
            if pos_kv is None:
                pos_kv = p["default_position_loop_kv_1_s"]
                self.estimated.append(f"spindles.{sid}.position_loop.kv_1_s")
            pos_kff = num(_get(loop, "feedforward"), 0.0)
            acc = [float(num(v, 0.0)) for v in (plain(_get(loop, "max_accel_rad_s2_per_gear")) or [])]
            orient = num(_get(s, "orient_time_s"))
            if orient is None:
                orient = p["default_orient_s"]
                self.estimated.append(f"spindles.{sid}.orient_time_s")
            gear_change = num(_get(s, "gear_change_s"))
            if gear_change is None:
                gear_change = p["default_gear_change_s"] if len(gears) > 1 else 0.0
            # C-axis mode references the spindle position like an orientation unless the machine says otherwise
            engage = num(_get(s, "c_axis_engage_time_s"))
            self.spindles.append(SpindleDef(
                sid, str(plain(s.kind)), index, max_w, gears, num(_get(s, "accel_time_s")),
                num(_get(s, "decel_time_s")), inertia, idle / max_w if max_w > 0 else 0.0, bw, pos_kv, pos_kff, acc,
                orient, gear_change, bool(plain(_get(s, "direction_reversible", True))),
                float(engage) if engage is not None else float(orient)))
        self.spindle_index = {s.id: s.index for s in self.spindles}

    def _assign_work_spindles(self) -> None:
        work = [s for s in self.spindles if s.kind in WORK_SPINDLE_KINDS]
        chains_with_rotary = [c for c in self.work_chains]
        for i, sp in enumerate(work):
            if i >= len(chains_with_rotary):
                break
            cname = chains_with_rotary[i]
            chain = self.chains[cname]
            sp.chain = cname
            rot = [j for j, l in enumerate(chain.links) if l.kind == LINK_ROTARY]
            R_F = self._chain_rotation_at_zero(chain, len(chain.links))
            z_ref = -(R_F[:, 2])
            if rot:
                j = rot[-1]
                link = chain.links[j]
                R_parent = self._chain_rotation_at_zero(chain, j)
                d_M = R_parent @ link.direction
                sp.c_axis = link.axis
                sp.kappa = 1.0 if float(z_ref @ d_M) >= 0.0 else -1.0
            else:
                # synthetic rotary link about the chuck-face z axis, turned by the spindle
                name = f"_{sp.id}_angle"
                chain.links.append(LinkDef(f"{sp.id}_rotor", LINK_ROTARY, name, np.array([0.0, 0.0, 1.0]),
                                           np.zeros(3), np.eye(4), synthetic_spindle=sp.id))
                sp.c_axis = name
                sp.kappa = -1.0
                path = next((pd.index for pd in self.paths if pd.work_chain == cname), 0)
                idx = len(self.axes)
                self.axis_index[name] = idx
                self.axes.append(AxisDef(name, idx, path, name, True, True, -math.inf, math.inf, 0.0, sp.max_omega,
                                         sp.max_omega, 1.0, 1.0, 30.0, 0.0, 50.0, 1.0, 1.0, 0.0, 0.0, False,
                                         synthetic=True, chain=cname, side=CHAIN_WORK, link=len(chain.links) - 1,
                                         direction_M=R_F[:, 2]))
                for pd in self.paths:
                    if pd.index == path and name not in pd.axes:
                        pd.axes.append(name)
            ax = self.axes[self.axis_index[sp.c_axis]]
            ax.spindle = sp.id
            ax.kappa = sp.kappa
        # default spindles of each path: main work spindle for path 0, sub for the others; tool spindle
        work_ids = [s.id for s in work]
        tool_ids = [s.id for s in self.spindles if s.kind in TOOL_SPINDLE_KINDS]
        for pd in self.paths:
            by_chain = next((s.id for s in work if s.chain == pd.work_chain), None)
            if by_chain is not None:
                pd.work_spindle = by_chain
            elif work_ids:
                pd.work_spindle = work_ids[0] if pd.index == 0 else work_ids[min(pd.index, len(work_ids) - 1)]
            milling = [s.id for s in self.spindles if s.kind == "milling"]
            live = [s.id for s in self.spindles if s.kind == "live_tool"]
            pd.tool_spindle = (milling[0] if milling else (live[0] if live else (tool_ids[0] if tool_ids else None)))

    def _chain_rotation_at_zero(self, chain: ChainDef, upto: int) -> np.ndarray:
        R = np.eye(3)
        for link in chain.links[:upto]:
            if link.kind == LINK_FIXED:
                R = R @ link.fixed[:3, :3]
        return R

    # ------------------------------------------------------------------ geometric errors (ISO 230-1)
    def _read_errors(self) -> None:
        n_links = {name: len(c.links) for name, c in self.chains.items()}
        self.err_poly = {name: np.zeros((n, 6, 3)) for name, n in n_links.items()}
        self.err_static = {name: np.zeros((n, 6)) for name, n in n_links.items()}
        em = self.raw.error_map
        if em is None:
            return
        # pivot errors: move/tilt the rotary axis lines (in place)
        for name, item in (_get(em, "pivot_errors") or {}).items():
            if name not in self.axis_index or not isinstance(item, Mapping):
                continue
            ax = self.axes[self.axis_index[name]]
            link = self.chains[ax.chain].links[ax.link]
            link.point = link.point + _vec(_get(item, "position_m"))
            tilt = _vec(_get(item, "tilt_rad"))
            angle = float(np.linalg.norm(tilt))
            if angle > 0:
                from .kernels import axis_rotation
                R = np.zeros((3, 3))
                axis_rotation(np.ascontiguousarray(tilt / angle), angle, R)
                link.direction = _unit(R @ link.direction, f"pivot error of {name}")
        # squareness: tilt the second axis of each pair toward the first (relative tool-to-work sense)
        for pair, value in (_get(em, "squareness_rad") or {}).items():
            e = num(value)
            if e is None or not isinstance(pair, str):
                continue
            first, second = self._split_pair(pair)
            if first is None:
                continue
            a1, a2 = self.axes[self.axis_index[first]], self.axes[self.axis_index[second]]
            s1 = 1.0 if a1.side == CHAIN_TOOL else -1.0
            s2 = 1.0 if a2.side == CHAIN_TOOL else -1.0
            link2 = self.chains[a2.chain].links[a2.link]
            r1 = s1 * a1.direction_M
            r2 = s2 * a2.direction_M
            r2n = _unit(math.cos(e) * r2 + math.sin(e) * r1, f"squareness {pair}")
            R_parent = self._chain_rotation_at_zero(self.chains[a2.chain], a2.link)
            link2.direction = _unit(R_parent.T @ (s2 * r2n), f"squareness {pair}")
        # straightness (bow) and angular errors of linear axes
        straight = _get(em, "straightness_m_per_m") or {}
        angular = _get(em, "angular_rad") or {}
        for ax in self.axes:
            if ax.rotary or ax.synthetic or ax.chain == "":
                continue
            link = self.chains[ax.chain].links[ax.link]
            d = link.direction
            up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = _unit(np.cross(up, d), "error basis")
            e2 = np.cross(d, e1)
            span = (ax.hi - ax.lo) if math.isfinite(ax.hi - ax.lo) else 1.0
            poly = self.err_poly[ax.chain][ax.link]
            value = num(straight.get(ax.name)) if hasattr(straight, "get") else None
            if value:
                bow = value * span
                for basis in (e1, e2):
                    for c in range(3):
                        poly[c, 1] += 4.0 * bow * basis[c]
                        poly[c, 2] += -4.0 * bow * basis[c]
            item = angular.get(ax.name) if hasattr(angular, "get") else None
            if isinstance(item, Mapping):
                for key, vec in (("pitch", e1), ("yaw", e2), ("roll", d)):
                    e = num(item.get(key))
                    if not e:
                        continue
                    for c in range(3):
                        poly[3 + c, 0] += -e * vec[c]
                        poly[3 + c, 1] += 2.0 * e * vec[c]
        height = num(_get(em, "y_axis_centre_height_m"))
        if height and "Y" in self.axis_index:
            ax = self.axes[self.axis_index["Y"]]
            if ax.chain:
                self.err_static[ax.chain][ax.link, 0:3] += height * self.chains[ax.chain].links[ax.link].direction

    def _split_pair(self, pair: str):
        names = sorted(self.axis_index, key=len, reverse=True)
        for first in names:
            if pair.startswith(first) and pair[len(first):] in self.axis_index:
                return first, pair[len(first):]
        return None, None

    # ------------------------------------------------------------------ helpers
    def axis(self, name: str) -> AxisDef:
        return self.axes[self.axis_index[name]]

    def path_axes(self, p: int) -> list[AxisDef]:
        return [self.axes[self.axis_index[n]] for n in self.paths[p].axes]

    def spindle(self, sid: str | None) -> SpindleDef | None:
        if sid is None or sid not in self.spindle_index:
            return None
        return self.spindles[self.spindle_index[sid]]

    def rapid_mode(self) -> str:
        key = "default_rapid_mode_" + ("heidenhain" if self.dialect.startswith("heidenhain") else self.dialect)
        return str(self.p.get(key, "dogleg"))

    def max_links(self) -> int:
        return max(len(c.links) for c in self.chains.values())

    def build_chain(self, tips_N: list[np.ndarray], poses_F_W: list[np.ndarray], poses_W_S: list[np.ndarray],
                    tool_spindles: list[int], work_chains: list[str] | None = None, nominal: bool = False) -> Chain:
        """Chain arrays of every path. ``nominal``: the control's kinematic model (no geometric errors), used
        for inverse kinematics; otherwise the physical machine with its ISO 230-1 errors (the loop's FK).
        Thermal and holding terms start at zero (MotionDomain applies them per macro step)."""
        P = len(self.paths)
        L = self.max_links()
        n_links = np.zeros((P, 2), np.int32)
        kind = np.zeros((P, 2, L), np.int32)
        axis = np.full((P, 2, L), -1, np.int32)
        direction = np.zeros((P, 2, L, 3))
        point = np.zeros((P, 2, L, 3))
        fixed = eye_stack(P, 2, L)
        err_poly = np.zeros((P, 2, L, 6, 3))
        err_lo = np.zeros((P, 2, L))
        err_span = np.ones((P, 2, L))
        err_const = np.zeros((P, 2, L, 6))
        work_link = np.full(P, -1, np.int32)
        pose_N_T0 = eye_stack(P)
        for p, pd in enumerate(self.paths):
            wname = (work_chains[p] if work_chains else None) or pd.work_chain
            for c, cname in ((CHAIN_WORK, wname), (CHAIN_TOOL, pd.tool_chain)):
                chain = self.chains[cname]
                n_links[p, c] = len(chain.links)
                for i, link in enumerate(chain.links):
                    kind[p, c, i] = link.kind
                    direction[p, c, i] = link.direction_nom if nominal else link.direction
                    point[p, c, i] = link.point_nom if nominal else link.point
                    fixed[p, c, i] = link.fixed
                    if not nominal:
                        err_poly[p, c, i] = self.err_poly[cname][i] if i < len(self.err_poly[cname]) else 0.0
                        err_const[p, c, i] = self.err_static[cname][i] if i < len(self.err_static[cname]) else 0.0
                    if link.axis is not None:
                        ax = self.axes[self.axis_index[link.axis]]
                        axis[p, c, i] = ax.index
                        if math.isfinite(ax.lo) and math.isfinite(ax.hi) and ax.hi > ax.lo:
                            err_lo[p, c, i] = ax.lo
                            err_span[p, c, i] = ax.hi - ax.lo
                        if c == CHAIN_WORK and ax.spindle is not None:
                            work_link[p] = i
            R = tool_frame_rotation(self.tool_axis_N)
            pose_N_T0[p, :3, :3] = R
            pose_N_T0[p, :3, 3] = tips_N[p]
        return Chain(n_links, kind, axis, direction, point, fixed, err_poly, err_lo, err_span, err_const,
                     np.zeros((P, 2, L)), np.zeros((P, 2, L)), work_link, pose_N_T0,
                     np.ascontiguousarray(np.array(poses_F_W, dtype=float).reshape(P, 4, 4)),
                     np.ascontiguousarray(np.array(poses_W_S, dtype=float).reshape(P, 4, 4)),
                     np.array(tool_spindles, np.int32), np.zeros(P))


def tool_frame_rotation(tool_axis_N: np.ndarray) -> np.ndarray:
    """Rotation of T0 in the nose frame: z_T = tool axis; x_T, y_T the nose x, y turned with it (identity
    for the usual tool axis +z_N; the minimal rotation otherwise)."""
    z = np.asarray(tool_axis_N, float)
    z = z / np.linalg.norm(z)
    ez = np.array([0.0, 0.0, 1.0])
    c = float(ez @ z)
    if c > 1.0 - 1e-15:
        return np.eye(3)
    if c < -1.0 + 1e-15:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(ez, z)
    s = float(np.linalg.norm(v))
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
