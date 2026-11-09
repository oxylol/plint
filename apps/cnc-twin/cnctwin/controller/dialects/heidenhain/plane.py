"""PLANE functions of the TNC 640 (tilted working plane) on machines with two rotary table axes.

The plane's rotation R (columns = the tilted X', Y', Z' in the untilted work frame):

* SPATIAL SPA SPB SPC: R = Rz(SPC) Ry(SPB) Rx(SPA) (rotations about the fixed machine axes A, then B, then C);
* PROJECTED PROPR PROMIN ROT: X' along the trace of the plane in ZX at angle PROPR (about +Y), the trace in YZ
  at PROMIN (about +X) also in the plane: Z' = trace_ZX x trace_YZ, then ROT about Z' (equal to SPATIAL when
  one projection angle is 0);
* EULER EULPR EULNU EULROT: R = Rz(EULPR) Rx(EULNU) Rz(EULROT);
* VECTOR BX BY BZ NX NY NZ: Z' = N / |N|, X' = the base vector made normal to Z', Y' = Z' x X';
* POINTS P1 P2 P3: X' = P1 -> P2, Z' = (P2 - P1) x (P3 - P1);
* RELATIV SPA | SPB | SPC: the active R rotated about its own (tilted) X, Y or Z;
* AXIAL A B C: the plane the rotary axes produce at these positions (R = T(axes)^-1 of the table);
* RESET: no tilt.

Positioning: MOVE (rotary axes with compensating linear motion, the point DIST above the tool tip fixed
relative to the workpiece: TCPM interpolation), TURN (rotary axes only, linear axes stay), STAY (no
motion; Q120-Q122 hold the axis angles for a later positioning block). With MOVE and TURN, MB (move back)
first retracts the tool along the current tool axis: MB MAX to the end of the traverse range, MB d by d
(incremental, MB 0 none), at the positioning feed (the M140 MB retraction). SEQ+ / SEQ- choose the solution
with the first rotary axis of the workpiece chain positive / negative; without SEQ the one with the
least rotary motion. The tilt turns about the active datum; afterwards the program frame is the tilted
one (core stage ``feature``) and the linear axes follow the table kinematics (stage heidenhain:PLANE).
"""

from __future__ import annotations

import math

import numpy as np

from ..siemens840d.kit import Alarm, clean_matrix, rot_x, rot_y, rot_z, rpy_matrix

IDENTITY = np.eye(3)


def _vector(values: dict, names: tuple, error: str) -> np.ndarray:
    v = np.array([values.get(n, 0.0) for n in names], dtype=float)
    if np.linalg.norm(v) < 1e-9:
        raise Alarm(error, " ".join(names))
    return v


def rotation(h, variant: str, values: dict) -> np.ndarray:
    current = np.asarray(h.s.sys["hh"]["R"], dtype=float).reshape(3, 3)
    if variant == "RESET":
        return IDENTITY.copy()
    if variant == "SPATIAL":
        return rpy_matrix(values.get("SPA", 0.0), values.get("SPB", 0.0), values.get("SPC", 0.0))
    if variant == "PROJECTED":
        b, a = math.radians(values.get("PROPR", 0.0)), math.radians(values.get("PROMIN", 0.0))
        u = np.array([math.cos(b), 0.0, -math.sin(b)])
        w = np.array([0.0, math.cos(a), math.sin(a)])
        z = np.cross(u, w)
        if np.linalg.norm(z) < 1e-9:
            raise Alarm("tilt_no_angle", "PROPR / PROMIN")
        z /= np.linalg.norm(z)
        x = u / np.linalg.norm(u)
        y = np.cross(z, x)
        return np.column_stack([x, y, z]) @ rot_z(math.radians(values.get("ROT", 0.0)))
    if variant == "EULER":
        return (rot_z(math.radians(values.get("EULPR", 0.0))) @ rot_x(math.radians(values.get("EULNU", 0.0)))
                @ rot_z(math.radians(values.get("EULROT", 0.0))))
    if variant == "VECTOR":
        base = _vector(values, ("BX", "BY", "BZ"), "tilt_vectors_short")
        normal = _vector(values, ("NX", "NY", "NZ"), "tilt_vectors_short")
        z = normal / np.linalg.norm(normal)
        x = base - (base @ z) * z
        if np.linalg.norm(x) < 1e-6 * np.linalg.norm(base):
            raise Alarm("tilt_vectors", "base vector parallel to the normal")
        x /= np.linalg.norm(x)
        return np.column_stack([x, np.cross(z, x), z])
    if variant == "POINTS":
        p1 = np.array([values.get(f"P1{c}", 0.0) for c in "XYZ"], dtype=float)
        p2 = np.array([values.get(f"P2{c}", 0.0) for c in "XYZ"], dtype=float)
        p3 = np.array([values.get(f"P3{c}", 0.0) for c in "XYZ"], dtype=float)
        e1 = p2 - p1
        if np.linalg.norm(e1) < 1e-6 or np.linalg.norm(p3 - p1) < 1e-6:
            raise Alarm("tilt_points_close", "P1 P2 P3")
        z = np.cross(e1, p3 - p1)
        if np.linalg.norm(z) < 1e-9 * np.linalg.norm(e1) * np.linalg.norm(p3 - p1):
            raise Alarm("tilt_points_line", "P1 P2 P3")
        x = e1 / np.linalg.norm(e1)
        z /= np.linalg.norm(z)
        return np.column_stack([x, np.cross(z, x), z])
    if variant == "RELATIV":
        given = [(k, v) for k, v in values.items() if k in ("SPA", "SPB", "SPC")]
        if len(given) != 1:
            raise Alarm("tilt_angle", "PLANE RELATIV takes one angle")
        name, angle = given[0]
        axis = {"SPA": rot_x, "SPB": rot_y, "SPC": rot_z}[name](math.radians(angle))
        return current @ axis
    if variant == "AXIAL":
        rotary = axial_values(h, values)
        transform = h.machine.table_transform(rotary)[:3, :3]
        return transform.T
    raise Alarm("tilt_not_allowed", variant)


def axial_values(h, values: dict) -> dict[str, float]:
    out = h.rotary_values()
    for letter in ("A", "B", "C"):
        key = letter if letter in values else ("I" + letter if "I" + letter in values else None)
        if key is None:
            continue
        axis = h.ex.by_letter.get(letter)
        if axis is None or axis.kind != "rotary":
            raise Alarm("tilt_no_rotary", f"axis {letter}")
        angle = math.radians(values[key])
        out[axis.name] = out.get(axis.name, 0.0) + angle if key.startswith("I") else angle
    return out


def execute(h, f) -> None:
    ex, s = h.ex, h.s
    machine = h.machine
    if not any(a.kind == "rotary" for a in ex.axes):
        raise Alarm("tilt_no_rotary", "PLANE")
    if not machine.table_kinematics_supported():
        raise Alarm("transform_unavailable", "PLANE needs two rotary table axes")
    if s.modal.get("7", "G40") != "G40":
        raise Alarm("tilt_not_allowed", "PLANE with radius compensation active")
    variant = f["variant"]
    values = {k: h.number(v) for k, v in f["values"].items()}
    new = rotation(h, variant, values)
    if variant == "AXIAL":
        solution = axial_values(h, values)
    elif variant == "RESET":
        solution = {a.name: 0.0 for a in ex.axes if a.kind == "rotary" and a.name in
                    [link.axis for link in machine.work_rotaries]}
        solution = {name: h.place_rotary(name, value) for name, value in solution.items()}
    else:
        normal = new @ np.array([0.0, 0.0, 1.0])
        solution = h.choose_table_solution(normal, master_sign=int(f.get("seq") or 0))
    position = f.get("position") or "STAY"
    feed_word = f.get("feed")
    rapid = feed_word is None or feed_word[0] == "FMAX"
    feed = ex.feed_spec(True) if rapid else ex.feed_spec(False, feed_si=h.number(feed_word[1]) * h.feed_unit_m() / 60.0,
                                                           mode="per_min")
    if f.get("mb") is not None and position in ("MOVE", "TURN"):
        # MB (move back): before the rotary axes tilt, the tool retracts along the current tool axis by the
        # incremental length, or with MB MAX to just below the end of the traverse range, at the positioning feed
        # (TNC 640 Klartext manual, PLANE "Automatic positioning", retraction MB); on table kinematics the tool
        # axis stays the machine axis of the TOOL CALL, so this is the M140 MB retraction, in the frame active
        # before this PLANE
        h.retract_tool_axis({"m140": f["mb"], **({} if rapid else {"m140_feed": feed_word[1]})})
    angles = {}
    for axis in ex.axes:
        if axis.name in solution:
            angles[axis.letter] = math.degrees(solution[axis.name])
    s.sys["plane_angles"] = angles
    hh = s.sys["hh"]
    hh["R"] = clean_matrix(new)
    active = variant != "RESET"
    if active:
        s.sys["plane"] = variant
    else:
        s.sys.pop("plane", None)
    s.tcp = "heidenhain:PLANE" if (active or s.tcp == "heidenhain:PLANE") else s.tcp
    if position == "MOVE":
        h.update_transform()
        dist = h.length_si(h.number(f["dist"])) if f.get("dist") is not None else 0.0
        target = {a.letter: ex.counts(a.letter, solution[a.name]) for a in ex.axes if a.name in solution}
        if dist:
            target.update(_pivot_target(h, solution, dist))
        s.tcp = "heidenhain:TCPM"
        ex.move("rapid" if rapid else "linear", target, feed)
        s.tcp = "heidenhain:PLANE" if active else ""
        ex.resync_program_position()
    elif position == "TURN":
        h.update_transform()
        target = {a.letter: ex.counts(a.letter, solution[a.name]) for a in ex.axes if a.name in solution}
        ex.machine_move("rapid" if rapid else "linear", target, feed, modal={"plane_positioning": "TURN"})
        if not active:
            s.tcp = ""
            ex.resync_program_position()
    else:
        h.update_transform()
        if not active:
            s.tcp = ""
            ex.resync_program_position()


def _pivot_target(h, solution: dict, dist: float) -> dict[str, int]:
    """MOVE DIST: the point ``dist`` above the tip on the tool axis stays fixed relative to the part; the new
    tip lies ``dist`` below it along the new tool axis (program frame)."""
    ex, s = h.ex, h.s
    hh = s.sys["hh"]
    rotation_now = np.asarray(hh["R"], dtype=float).reshape(3, 3)
    current = h.rotary_values()
    table_now = h.machine.table_transform(current)[:3, :3]
    table_new = h.machine.table_transform({**current, **solution})[:3, :3]
    tool = np.array(h.machine.tool_axis, dtype=float)
    axis_now = rotation_now.T @ table_now.T @ tool
    axis_new = rotation_now.T @ table_new.T @ tool
    tip = np.array([ex.si(l, s.pos.get(l, 0)) if l in ex.by_letter else 0.0 for l in "XYZ"])
    pivot = tip + dist * axis_now
    new_tip = pivot - dist * axis_new
    return {l: ex.counts(l, float(new_tip[i])) for i, l in enumerate("XYZ") if l in ex.by_letter}


__all__ = ["execute", "rotation"]
