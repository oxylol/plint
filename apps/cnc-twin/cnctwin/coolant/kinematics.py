"""Link poses of the machine's kinematic chains, for nozzle mounts and machine meshes
(docs/models/coolant.md "Frames").

The machine file (db/_templates/machine.yaml ``kinematics``) lists chains from the machine base M:
``workpiece_chain*`` (to the fixture face) and ``*tool_chain`` (to the spindle nose). A link is an NC
axis (linear: translation by value x direction; rotary: right-handed rotation by the value about
``direction`` through ``axis_point``) or fixed (a pose). :func:`chain_poses` evaluates every link's
pose in M from the axis values of the motion frame.

**Mount frames of nozzles** (:class:`ChainMounts`): a nozzle attached to a link of a tool chain is
given in the spindle-nose frame N of that chain (the machine files say "head frame at spindle nose";
their geometry primitives use the literal link frames). N's origin is placed at the tool's gauge
line, ``tip_M + z_gauge * axis_T_M`` from the motion record, so jets stay registered to the tool the
hot loop moved; its axes are those of the chain's last link (so positions and aims keep the axes the
machine file was written in). A nozzle on a workpiece-chain link is in that link's frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..db.simap import pose_matrix


def _plain(value):
    return getattr(value, "value", value)


def _vec(value, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(_plain(v)) for v in _plain(value)], dtype=float)


def axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Right-handed rotation about a unit axis (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0.0:
        return np.eye(3)
    a = a / n
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


@dataclass(frozen=True)
class Link:
    name: str
    axis: str | None
    direction: tuple
    axis_point_m: tuple
    fixed: tuple | None             # 4x4 as nested tuples


def _fixed_matrix(fixed) -> np.ndarray:
    if fixed is None:
        return np.eye(4)
    if hasattr(fixed, "matrix") and callable(fixed.matrix):
        return np.asarray(fixed.matrix(), dtype=float)
    translate = getattr(fixed, "translate_m", None)
    rotate = getattr(fixed, "rotate_rad", None)
    order = getattr(fixed, "order", None) or "xyz"
    if translate is None and isinstance(fixed, dict):
        translate = [v * 1e-3 for v in fixed.get("translate_mm", [0, 0, 0])]
        rotate = [math.radians(v) for v in fixed.get("rotate_deg", [0, 0, 0])]
        order = fixed.get("order", "xyz")
    return pose_matrix(_vec(translate), _vec(rotate), str(order))


class MachineChains:
    """The kinematic chains of one machine file (SIMap or None)."""

    def __init__(self, machine):
        kin = getattr(machine, "kinematics", None) if machine is not None else None
        self.chains: dict[str, list[Link]] = {}
        self.axis_type: dict[str, str] = {}
        axes = getattr(machine, "axes", None) if machine is not None else None
        if axes is not None:
            for name in axes:
                self.axis_type[str(name)] = str(getattr(axes[name], "type", "linear") or "linear")
        if kin is not None:
            for key in sorted(kin):
                if "chain" not in str(key):
                    continue
                value = kin[key]
                if value is None:
                    continue
                if hasattr(value, "keys"):
                    # migrated form: tool_chains.<name> / workpiece_chains.<name> (machine file
                    # <name>_tool_chain / workpiece_chain_<name>)
                    for name in sorted(value.keys()):
                        label = f"{name}_tool_chain" if str(key) == "tool_chains" else f"workpiece_chain_{name}"
                        self.chains[label] = self._links(value[name])
                else:
                    self.chains[str(key)] = self._links(value)
        tool_axis = getattr(kin, "tool_axis", None) if kin is not None else None
        self.tool_axis_N = _vec(tool_axis, (0.0, 0.0, 1.0))

    @staticmethod
    def _links(entries) -> list[Link]:
        links = []
        for entry in entries or []:
            fixed = getattr(entry, "fixed", None)
            axis = getattr(entry, "axis", None)
            links.append(Link(str(getattr(entry, "link", "")), str(axis) if axis is not None else None,
                              tuple(_vec(getattr(entry, "direction", None), (0.0, 0.0, 1.0))),
                              tuple(_vec(getattr(entry, "axis_point_m", None))),
                              tuple(map(tuple, _fixed_matrix(fixed))) if fixed is not None else None))
        return links

    @property
    def tool_chain_names(self) -> list[str]:
        names = [n for n in self.chains if "tool_chain" in n]
        return sorted(names, key=lambda n: (n != "tool_chain", n))

    def chain_of(self, link: str) -> str | None:
        for name, links in sorted(self.chains.items()):
            if any(l.name == link for l in links):
                return name
        return None

    def is_tool_link(self, link: str) -> bool:
        chain = self.chain_of(link)
        return chain is not None and "tool_chain" in chain

    def path_of_chain(self, chain: str) -> int:
        names = self.tool_chain_names
        return names.index(chain) if chain in names else 0

    def poses(self, axis_values: dict[str, float]) -> dict[str, np.ndarray]:
        """Pose in M of every link of every chain at the given axis values (missing axes at 0)."""
        out: dict[str, np.ndarray] = {}
        for name, links in sorted(self.chains.items()):
            T = np.eye(4)
            for link in links:
                if link.axis is not None:
                    value = float(axis_values.get(link.axis, 0.0))
                    direction = np.asarray(link.direction, dtype=float)
                    if self.axis_type.get(link.axis, "linear") == "rotary":
                        R = axis_rotation(direction, value)
                        p = np.asarray(link.axis_point_m, dtype=float)
                        M = np.eye(4)
                        M[:3, :3] = R
                        M[:3, 3] = p - R @ p
                    else:
                        M = np.eye(4)
                        norm = np.linalg.norm(direction)
                        M[:3, 3] = value * (direction / norm if norm > 0 else direction)
                    T = T @ M
                if link.fixed is not None:
                    T = T @ np.asarray(link.fixed, dtype=float)
                out[link.name] = T.copy()
        return out

    def nose_rotation(self, chain: str, poses: dict[str, np.ndarray]) -> np.ndarray:
        links = self.chains.get(chain) or []
        if not links:
            return np.eye(3)
        return poses[links[-1].name][:3, :3]


def axis_values(paths) -> dict[str, float]:
    """Axis values at the end of the step from every path of a MotionFrame."""
    values: dict[str, float] = {}
    for path in paths:
        names = list(getattr(path, "axis_names", ()) or ())
        actual = np.asarray(getattr(path, "axis_actual", np.zeros((0, 0))))
        if actual.ndim == 2 and actual.shape[0] and actual.shape[1] == len(names):
            for i, name in enumerate(names):
                values[str(name)] = float(actual[-1, i])
    return values


def frame_from_z(z_axis: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    """Rotation whose third column is z_axis and whose first lies closest to x_hint."""
    z = np.asarray(z_axis, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, dtype=float) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-9:
        e = np.eye(3)[int(np.argmin(np.abs(z)))]
        x = e - np.dot(e, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class ChainMounts:
    """Default MountPoses: see the module docstring."""

    def __init__(self, chains: MachineChains):
        self.chains = chains

    def pose_M(self, link: str, path, gauge_m: float | None, poses: dict[str, np.ndarray]) -> np.ndarray | None:
        chains = self.chains
        chain = chains.chain_of(link)
        if chain is None:
            return None
        if "tool_chain" not in chain:
            return poses.get(link)
        links = chains.chains.get(chain) or []
        T = np.eye(4)
        if links:
            T[:3, :3] = poses[links[-1].name][:3, :3]
            T[:3, 3] = poses[links[-1].name][:3, 3]
        tip = np.asarray(path.tip_M_m[-1], dtype=float) if path is not None and len(getattr(path, "tip_M_m", [])) \
            else None
        axis = np.asarray(path.axis_T_M[-1], dtype=float) if path is not None and len(getattr(path, "axis_T_M", [])) \
            else None
        if tip is not None and axis is not None and gauge_m is not None and np.linalg.norm(axis) > 0.0:
            T[:3, 3] = tip + gauge_m * axis / np.linalg.norm(axis)
        return T
