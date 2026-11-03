"""Five-axis functions of the FANUC dialect (fanuc-30ibplus-om-2019 22.1, 22.6): tool centre point control
G43.4 (rotary axes programmed) and G43.5 (tool vector I J K programmed), the tilted working plane
G68.2 (Euler angles Z X' Z'', roll-pitch-yaw P1, three points P2), G68.3 (by the tool axis direction),
G68.4 (incremental), G53.1 (tool axis normal to the plane), G69 (cancel).

Kinematics: the machine file's chains are multiplied link by link (linear axis: translation along its
direction, rotary axis: rotation about its line, fixed links: their pose) giving the pose of the
spindle nose in the fixture-face frame, ``Rel(q) = Wp(q)^-1 Tl(q)``. The work frame W is fixed to the
table (FANUC's table coordinate system): its origin is the gauge point at the work offset with rotary
axes at zero, its axes those of the fixture frame. A tool tip p_W with length L along the tool axis
(nose frame ``machine.tool_axis``, from the tip to the spindle) is reached at the linear axes that solve
the affine system ``Rel(q_l, q_r) (-L a) = Z_T + p_W`` (exact, 3 x 3 linear solve). The rotary values for
a tool direction d (G43.5, G53.1) come from Paden-Kahan subproblem 2 on the product of exponentials of
the two rotary joints (murray-li-sastry-1994 ch. 3): two solutions, the one nearest the current rotary
positions within travel is taken (continuous axes unwrapped to the nearest turn); a direction along the
first joint axis is the singular case, where that joint keeps its value; no solution: TW0022, outside
travel: TW0023.

Commands are Space.TIP: ``tip_target_m`` (tip in W), ``tip_axis`` (tool axis in W), ``target`` the
machine axes of the end point (this inverse kinematics), ``modal tip_interpolation`` "tip" (G43.4/G43.5,
motion keeps the tip on the path; ``tcp_type`` "vector" for G43.5) or "joint" (moves under G68.2 and
the start-up block of G43.4, FANUC: "only tool length compensation is performed" there), and
``modal work_origin_M0_m``. Under G68.2 the program frame is the feature frame (core pipeline stage
``feature``: w = o + M p) and every move is positioned with the current rotary values.
"""

from __future__ import annotations

import math

import numpy as np

from . import hooks
from .alarms import ControllerAlarm
from .cycles import common
from .dialects.base import GSpec
from .expr import round_half_away
from .hooks import CycleCall, Step
from .ir import ArcSpec, Command, Kind, Space
from .machine import plain

TCP_KEYS = ("fanuc:G43.4", "fanuc:G43.5")
TILT_KEY = "fanuc:G68.2"


# ====================================================================== kinematics

def _rot(direction, angle: float) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    x, y, z = d
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def _vec(value, n=3) -> np.ndarray:
    value = plain(value)
    if value is None:
        return np.zeros(n)
    return np.array([float(plain(v)) for v in value], dtype=float)


class Link:
    def __init__(self, kind: str, axis: str = "", direction=None, point=None, matrix=None):
        self.kind = kind                    # "linear" | "rotary" | "fixed"
        self.axis = axis
        self.direction = np.asarray(direction if direction is not None else (0, 0, 1), dtype=float)
        if np.linalg.norm(self.direction) > 0:
            self.direction = self.direction / np.linalg.norm(self.direction)
        self.point = np.asarray(point if point is not None else (0, 0, 0), dtype=float)
        self.matrix = matrix if matrix is not None else np.eye(4)

    def transform(self, value: float) -> np.ndarray:
        out = np.eye(4)
        if self.kind == "linear":
            out[:3, 3] = self.direction * value
        elif self.kind == "rotary":
            r = _rot(self.direction, value)
            out[:3, :3] = r
            out[:3, 3] = self.point - r @ self.point
        else:
            out = self.matrix
        return out


class Kinematics:
    """The machine's two chains as the control's kinematic model (machine file drawing values)."""

    def __init__(self, ex):
        machine = ex.machine
        raw = machine.raw
        kin = raw.kinematics if raw is not None else None
        if kin is None:
            raise ControllerAlarm("transform_unavailable", "the machine file has no kinematics")
        axes = raw.axes
        self.tool_axis = np.array(machine.tool_axis, dtype=float)
        self.tool_axis /= np.linalg.norm(self.tool_axis)
        self.work = self._chain(kin.workpiece_chain or [], axes)
        self.tool = self._chain(kin.tool_chain or [], axes)
        names = {a.name: a for a in ex.axes}
        self.rotaries = [l for l in self.work + self.tool if l.kind == "rotary" and l.axis in names]
        self.linears = [a.name for a in ex.axes if a.kind == "linear" and a.letter in "XYZ"]
        self.limits = {a.name: a.travel for a in ex.axes}
        self.continuous = {a.name: a.continuous for a in ex.axes}

    @staticmethod
    def _chain(links, axes) -> list[Link]:
        out = []
        for link in links:
            if link.axis is not None:
                name = str(plain(link.axis))
                axis_def = axes.get(name) if axes is not None else None
                kind = str(plain(axis_def.type)) if axis_def is not None else "linear"
                out.append(Link(kind, name, _vec(link.direction), _vec(link.axis_point_m)))
            elif link.fixed is not None:
                out.append(Link("fixed", matrix=np.asarray(link.fixed.matrix(), dtype=float)))
        return out

    @staticmethod
    def _product(links: list[Link], q: dict) -> np.ndarray:
        total = np.eye(4)
        for link in links:
            total = total @ link.transform(float(q.get(link.axis, 0.0)) if link.kind != "fixed" else 0.0)
        return total

    def rel(self, q: dict) -> np.ndarray:
        """Pose of the spindle nose in the fixture-face frame for axis values q (name -> SI)."""
        return np.linalg.inv(self._product(self.work, q)) @ self._product(self.tool, q)

    def tip(self, q: dict, length: float) -> np.ndarray:
        pose = self.rel(q)
        return pose[:3, :3] @ (-length * self.tool_axis) + pose[:3, 3]

    def axis(self, q: dict) -> np.ndarray:
        return self.rel(q)[:3, :3] @ self.tool_axis

    def solve_linear(self, q: dict, target: np.ndarray, length: float) -> dict:
        """Linear axis values putting the tip at ``target`` (fixture frame) with the rotary values in q."""
        base = dict(q)
        for name in self.linears:
            base[name] = 0.0
        f0 = self.tip(base, length)
        columns = []
        for name in self.linears:
            probe = dict(base)
            probe[name] = 1.0
            columns.append(self.tip(probe, length) - f0)
        jac = np.array(columns).T
        if jac.shape != (3, 3) or abs(np.linalg.det(jac)) < 1e-12:
            raise ControllerAlarm("kinematics_no_solution", "the linear axes do not span the tool tip space")
        values = np.linalg.solve(jac, target - f0)
        return {name: float(values[i]) for i, name in enumerate(self.linears)}

    # ------------------------------------------------------------------ orientation (Paden-Kahan 2)
    def _exponentials(self) -> tuple[list, np.ndarray, np.ndarray]:
        """[(spatial axis at zero, sign, joint name), ...] in the order d_W = R_w0^T E1 E2 v0, and v0, R_w0."""
        seq = []
        frame = np.eye(4)
        work_axes = []
        for link in self.work:
            if link.kind == "rotary":
                work_axes.append((frame[:3, :3] @ link.direction, link.axis))
            frame = frame @ link.transform(0.0)
        r_w0 = frame[:3, :3]
        frame = np.eye(4)
        tool_axes = []
        for link in self.tool:
            if link.kind == "rotary":
                tool_axes.append((frame[:3, :3] @ link.direction, link.axis))
            frame = frame @ link.transform(0.0)
        v0 = frame[:3, :3] @ self.tool_axis
        for direction, name in reversed(work_axes):
            seq.append((direction, -1.0, name))
        for direction, name in tool_axes:
            seq.append((direction, 1.0, name))
        return seq, v0, r_w0

    def orient(self, d_w: np.ndarray, current: dict) -> dict:
        """Rotary values (name -> rad) turning the tool axis to d_w (fixture frame)."""
        seq, v0, r_w0 = self._exponentials()
        d = r_w0 @ (np.asarray(d_w, dtype=float) / np.linalg.norm(d_w))
        if len(seq) == 1:
            (w1, s1, n1), = seq
            x1 = _subproblem1(w1, v0, d)
            if x1 is None:
                raise ControllerAlarm("kinematics_no_solution", "tool direction not reachable with one rotary axis")
            return self._pick([{n1: s1 * x1}], current)
        if len(seq) != 2:
            raise ControllerAlarm("transform_unavailable", f"{len(seq)} rotary axes: two are needed for a tool vector")
        (w1, s1, n1), (w2, s2, n2) = seq
        candidates = []
        if np.linalg.norm(np.cross(w1, d)) < 1e-9:
            x2 = _subproblem1(w2, v0, d)
            if x2 is not None:
                keep = float(current.get(n1, 0.0)) * s1
                candidates.append({n1: s1 * keep, n2: s2 * x2})
        else:
            for c in _cones(w1, w2, v0, d):
                x2 = _subproblem1(w2, v0, c)
                x1 = _subproblem1(w1, c, d)
                if x1 is not None and x2 is not None:
                    candidates.append({n1: s1 * x1, n2: s2 * x2})
        if not candidates:
            raise ControllerAlarm("kinematics_no_solution", f"tool direction {tuple(round(v, 6) for v in d_w)}")
        return self._pick(candidates, current)

    def _pick(self, candidates: list[dict], current: dict) -> dict:
        best, best_cost = None, None
        for cand in candidates:
            chosen = {}
            ok = True
            for name, value in cand.items():
                cur = float(current.get(name, 0.0))
                v = cur + math.remainder(value - cur, 2.0 * math.pi)
                limits = self.limits.get(name)
                if not self.continuous.get(name):
                    options = [v, v - 2 * math.pi, v + 2 * math.pi]
                    inside = [o for o in options if limits is None or limits[0] - 1e-9 <= o <= limits[1] + 1e-9]
                    if not inside:
                        ok = False
                        break
                    v = min(inside, key=lambda o: abs(o - cur))
                chosen[name] = v
            if not ok:
                continue
            cost = sum(abs(chosen[n] - float(current.get(n, 0.0))) for n in chosen)
            if best_cost is None or cost < best_cost - 1e-12:
                best, best_cost = chosen, cost
        if best is None:
            raise ControllerAlarm("rotary_out_of_range", "every solution of the tool direction is outside travel")
        return best


def _subproblem1(w, p, q) -> float | None:
    """Angle x with exp(w x) p = q (components along w must match)."""
    w = np.asarray(w, dtype=float)
    u = p - w * (w @ p)
    v = q - w * (w @ q)
    if np.linalg.norm(u) < 1e-12 or np.linalg.norm(v) < 1e-12:
        return 0.0 if abs((w @ p) - (w @ q)) < 1e-9 else None
    if abs((w @ p) - (w @ q)) > 1e-7 or abs(np.linalg.norm(u) - np.linalg.norm(v)) > 1e-7:
        return None
    return math.atan2(w @ np.cross(u, v), u @ v)


def _cones(w1, w2, p, q) -> list[np.ndarray]:
    """Points c with exp(w2 x2) p = c = exp(-w1 x1) q (Paden-Kahan subproblem 2, axes through the origin)."""
    w12 = float(w1 @ w2)
    den = w12 * w12 - 1.0
    if abs(den) < 1e-12:
        return []
    alpha = (w12 * (w2 @ p) - (w1 @ q)) / den
    beta = (w12 * (w1 @ q) - (w2 @ p)) / den
    cross = np.cross(w1, w2)
    g2 = ((p @ p) - alpha * alpha - beta * beta - 2.0 * alpha * beta * w12) / (cross @ cross)
    if g2 < -1e-12:
        return []
    gamma = math.sqrt(max(g2, 0.0))
    base = alpha * w1 + beta * w2
    if gamma == 0.0:
        return [base]
    return [base + gamma * cross, base - gamma * cross]


def kinematics(ex) -> Kinematics:
    cached = getattr(ex, "_tcp_kinematics", None)
    if cached is None:
        cached = Kinematics(ex)
        ex._tcp_kinematics = cached
    return cached


# ====================================================================== final stage (TIP commands)

class KinematicStage:
    """Final stage of G43.4 ("rotary"), G43.5 ("vector") and G68.2 ("tilt")."""

    def __init__(self, mode: str):
        self.mode = mode

    def _parts(self, ex, item):
        kin = kinematics(ex)
        length = float(item.modal.get("length_offset_m", 0.0))
        axis_letter = str(item.modal.get("tool_axis_letter", "Z"))
        origin_q = {}
        rotary = {}
        for i, a in enumerate(ex.axes):
            if a.kind == "rotary":
                rotary[a.name] = (int(round_half_away(item.end[i] / a.quantum)) + item.offset[i]) * a.quantum
                origin_q[a.name] = 0.0
            else:
                origin_q[a.name] = item.offset[i] * a.quantum - (length if a.letter == axis_letter else 0.0)
        z_t = kin.tip(origin_q, 0.0)
        tip_w = np.array([item.end[ex.letters.index(l)] if l in ex.letters else 0.0 for l in "XYZ"])
        return kin, length, rotary, origin_q, z_t, tip_w

    def _rotary(self, ex, item, kin, rotary):
        """G43.5: the rotary values of the block's tool vector (solved once, kept in the item; a block
        without I J K keeps the previous direction, FANUC 22.1)."""
        if self.mode != "vector":
            return rotary
        if "tcp_rotary" in item.modal:
            return {k: float(v) for k, v in item.modal["tcp_rotary"].items()}
        vector = item.modal.get("tool_vector")
        out = dict(rotary)
        if vector is None:
            out.update(ex.s.sys.get("tcp_rotary") or {})
        else:
            current = {a.name: ex.s.mach.get(a.letter, 0) * a.quantum for a in ex.axes if a.kind == "rotary"}
            out.update(kin.orient(np.array(vector, dtype=float), current))
        item.modal["tcp_rotary"] = {k: float(v) for k, v in out.items()}
        ex.s.sys["tcp_rotary"] = dict(item.modal["tcp_rotary"])
        return out

    def machine_counts(self, ex, item) -> list[int]:
        kin, length, rotary, _, z_t, tip_w = self._parts(ex, item)
        rotary = self._rotary(ex, item, kin, rotary)
        quanta = {a.name: a.quantum for a in ex.axes}
        rotary = {n: int(round_half_away(v / quanta[n])) * quanta[n] for n, v in rotary.items()}
        q = dict(rotary)
        linear = kin.solve_linear(q, z_t + tip_w, length)
        counts = []
        for i, a in enumerate(ex.axes):
            if a.kind == "rotary":
                counts.append(int(round_half_away(rotary[a.name] / a.quantum)))
            elif a.name in linear:
                counts.append(int(round_half_away(linear[a.name] / a.quantum)))
            else:
                counts.append(int(round_half_away(item.end[i] / a.quantum)) + item.offset[i])
        return counts

    def to_command(self, ex, item) -> Command:
        kin, _, _, _, z_t, tip_w = self._parts(ex, item)
        counts = self.machine_counts(ex, item)
        target = {a.name: counts[i] * a.quantum for i, a in enumerate(ex.axes)}
        q = {a.name: target[a.name] for a in ex.axes}
        axis_w = kin.axis(q)
        kind = {"rapid": Kind.RAPID, "linear": Kind.LINEAR, "arc": Kind.ARC, "probe": Kind.PROBE}[item.kind]
        arc = None
        if item.kind == "arc":
            arc = ArcSpec(item.plane, tuple(item.centre), tuple(item.normal), item.sweep_rad, item.r_start, item.r_end)
        modal = dict(item.modal)
        startup = bool(modal.get("tcp_startup"))
        modal["tip_interpolation"] = "tip" if self.mode in ("rotary", "vector") and not startup else "joint"
        if self.mode == "vector":
            modal["tcp_type"] = "vector"
        modal["work_origin_M0_m"] = [float(v) for v in z_t]
        cmd = Command(kind, item.ref, Space.TIP, target, tuple(float(v) for v in tip_w),
                      tuple(float(v) for v in axis_w), arc, item.feed, item.smoothing, modal=modal)
        if item.kind == "probe":
            cmd.probe = dict(item.probe or {})
        return cmd

    def work_from_machine(self, ex, machine_counts: dict, applied: dict) -> dict:
        kin = kinematics(ex)
        length = sum(applied.get(l, 0) for l in ("Z",)) * ex.machine.lin_increment_m
        base = ex.offset_counts(letters_tool=False)
        q = {a.name: machine_counts.get(a.letter, 0) * a.quantum for a in ex.axes}
        origin_q = {a.name: (base[i] * a.quantum if a.kind == "linear" else 0.0) for i, a in enumerate(ex.axes)}
        tip = kin.tip(q, length) - kin.tip(origin_q, 0.0)
        work = {}
        for index, a in enumerate(ex.axes):
            if a.letter in "XYZ" and a.kind == "linear":
                work[a.letter] = float(tip["XYZ".index(a.letter)])
            else:
                work[a.letter] = (machine_counts.get(a.letter, 0) - base[index]) * a.quantum
        return work


class TcpWords:
    """G43.5: I J K of a motion block are the tool vector at its end (FANUC 22.1); the first motion
    block after G43.4 / G43.5 is the start-up block (tool length compensation only)."""

    def motion_words(self, ex, code: str, bw) -> dict:
        s = ex.s
        out = {}
        if s.tcp in TCP_KEYS:
            if s.sys.get("tcp_seen") != s.tcp:
                s.sys["tcp_seen"] = s.tcp
                out["tcp_startup"] = True
        else:
            s.sys.pop("tcp_seen", None)
            s.sys.pop("tcp_rotary", None)
        if s.tcp == "fanuc:G43.5" and code in ("G00", "G01"):
            vec = {}
            for letter in ("I", "J", "K"):
                word = bw.take(letter)
                if word is not None:
                    vec[letter] = word.value
            if vec:
                vector = [float(vec.get(l, 0.0)) for l in "IJK"]
                if math.sqrt(sum(v * v for v in vector)) == 0.0:
                    raise ControllerAlarm("tcp_illegal", "tool vector I J K of length 0")
                s.sys["tcp_vector"] = vector
                out["tool_vector"] = vector
        return out


# ====================================================================== tilted working plane

def _euler_zxz(a: float, b: float, c: float) -> np.ndarray:
    return _rot((0, 0, 1), a) @ _rot((1, 0, 0), b) @ _rot((0, 0, 1), c)


def _rpy(angles: tuple[float, float, float], order: str) -> np.ndarray:
    total = np.eye(3)
    axes = {"1": (1, 0, 0), "2": (0, 1, 0), "3": (0, 0, 1)}
    for digit in order:
        index = int(digit) - 1
        total = total @ _rot(axes[digit], angles[index])
    return total


class TiltedPlane:
    """G68.2 / G68.3 / G68.4 (modal group 16), G69 and G53.1 (``ext`` G codes)."""

    def execute(self, ex, code: str, spec, bw) -> bool:
        if code == "G69":
            self._cancel(ex, bw)
            return False
        if code == "G53.1":
            return self._g53_1(ex, bw)
        s = ex.s
        if s.comp_state is not None:
            raise ControllerAlarm("tilted_illegal", f"{code} in cutter compensation")
        origin, matrix = self._frame(ex, code, bw)
        if code == "G68.4" and s.sys.get("tilted"):
            o0 = np.array(s.sys["tilted"]["origin"])
            m0 = np.array(s.sys["tilted"]["matrix"]).reshape(3, 3)
            origin, matrix = o0 + m0 @ origin, m0 @ matrix
        if s.rotation is not None:
            s.rotation = None
        s.sys["tilted"] = {"origin": [float(v) for v in origin], "matrix": [float(v) for v in matrix.ravel()],
                           "code": code}
        s.feature = {"origin": [ex.counts(l, float(origin[i])) if l in ex.by_letter else 0 for i, l in enumerate("XYZ")],
                     "matrix": [float(v) for v in matrix.ravel()]}
        s.modal["16"] = code
        if s.tcp not in TCP_KEYS and any(a.kind == "rotary" for a in ex.axes):
            s.tcp = TILT_KEY                  # positioning with the rotary axes' kinematics (3-axis: default stage)
            s.sys["tcp_group"] = 16
        ex.invalidate_transform()
        ex.resync_program_position()
        return False

    def _angles(self, ex, bw, letters="IJK") -> tuple[float, float, float]:
        out = []
        for letter in letters:
            word = bw.take(letter)
            out.append(math.radians(word.value if (word is None or word.point or word.from_variable)
                                    else word.value * 0.001) if word is not None else 0.0)
        return tuple(out)

    def _origin(self, ex, bw, default) -> np.ndarray:
        words = {l: bw.take(l) for l in "XYZ"}
        given = [l for l, w in words.items() if w is not None]
        if not given:
            return np.array(default, dtype=float)
        out = np.array(default, dtype=float)
        for i, l in enumerate("XYZ"):
            if words[l] is not None:
                out[i] = ex.handler.lin(words[l])
        return out

    def _frame(self, ex, code: str, bw):
        h = ex.handler
        p_word = bw.take("P")
        kind = h.integer(p_word) if p_word is not None else 0
        if code in ("G68.2", "G68.4"):
            if kind == 0:
                origin = self._origin(ex, bw, (0.0, 0.0, 0.0))
                a, b, c = self._angles(ex, bw)
                if a == 0.0 and b == 0.0 and c == 0.0 and code == "G68.2":
                    raise ControllerAlarm("tilted_format", "G68.2 with I0 J0 K0")
                return origin, _euler_zxz(a, b, c)
            if kind == 1:
                q_word = bw.take("Q")
                order = str(h.integer(q_word)) if q_word is not None else "123"
                if sorted(order) != ["1", "2", "3"]:
                    raise ControllerAlarm("tilted_format", f"{code} P1 Q{order}")
                origin = self._origin(ex, bw, (0.0, 0.0, 0.0))
                return origin, _rpy(self._angles(ex, bw), order)
            if kind == 2:
                return self._three_points(ex, code, bw)
            raise ControllerAlarm("unsupported_function", f"{code} P{kind} (two vectors / projection angles)")
        if code == "G68.3":
            return self._by_tool_axis(ex, bw)
        raise ControllerAlarm("tilted_format", code)

    def _three_points(self, ex, code: str, bw):
        """G68.2 P2 Q1/Q2/Q3 X Y Z (Q0: shift, R: rotation about Zc), one point per block (FANUC 22.6.1.4)."""
        s = ex.s
        q = ex.handler.integer(bw.take("Q")) if bw.has("Q") else None
        if q is None or q not in (0, 1, 2, 3):
            raise ControllerAlarm("tilted_format", f"{code} P2 Q{q}")
        points = s.sys.setdefault("tilted_points", {})
        r_word = bw.take("R")
        if r_word is not None:
            points["R"] = math.radians(r_word.value if (r_word.point or r_word.from_variable) else r_word.value * 0.001)
        points[str(q)] = [float(v) for v in self._origin(ex, bw, (0.0, 0.0, 0.0))]
        if not all(str(k) in points for k in (1, 2, 3)):
            raise _Pending()
        p1, p2, p3 = (np.array(points[str(k)]) for k in (1, 2, 3))
        xc = p2 - p1
        if np.linalg.norm(xc) < 1e-12:
            raise ControllerAlarm("tilted_format", "G68.2 P2: two points coincide")
        xc /= np.linalg.norm(xc)
        v = p3 - p1
        yc = v - xc * (xc @ v)
        if np.linalg.norm(yc) < 1e-9:
            raise ControllerAlarm("tilted_format", "G68.2 P2: three points on a line")
        yc /= np.linalg.norm(yc)
        zc = np.cross(xc, yc)
        matrix = np.column_stack([xc, yc, zc]) @ _rot((0, 0, 1), float(points.get("R", 0.0)))
        shift = np.array(points.get("0", [0.0, 0.0, 0.0]))
        s.sys.pop("tilted_points", None)
        return p1 + matrix @ shift, matrix

    def _by_tool_axis(self, ex, bw):
        """G68.3: Zc the current tool axis (the reference tool axis on table-type machines), Xc = P x Zc."""
        kin = kinematics(ex)
        q = {a.name: ex.s.mach.get(a.letter, 0) * a.quantum for a in ex.axes}
        if not any(l.kind == "rotary" for l in kin.tool):
            q = {a.name: (q[a.name] if a.kind == "linear" else 0.0) for a in ex.axes}   # table type: reference axis
        zc = kin.axis(q)
        zc = zc / np.linalg.norm(zc)
        vertical = int(common.params(ex)["tilted_vertical_axis"])
        p = np.eye(3)[vertical - 1] if vertical in (1, 2, 3) else kin.tool_axis
        parallel = math.acos(max(-1.0, min(1.0, abs(float(p @ zc))))) <= float(common.params(ex)["tilted_parallel_rad"])
        if parallel:
            index = vertical if vertical in (1, 2, 3) else 3
            xc = np.eye(3)[index % 3]
        else:
            xc = np.cross(p, zc)
            xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        r_word = bw.take("R")
        angle = math.radians(r_word.value if (r_word is not None and (r_word.point or r_word.from_variable))
                             else (r_word.value * 0.001 if r_word is not None else 0.0))
        matrix = np.column_stack([xc, yc, zc]) @ _rot((0, 0, 1), angle)
        given = [l for l in "XYZ" if bw.has(l)]
        if given and len(given) != 3:
            raise ControllerAlarm("tilted_format", "G68.3 needs all of X, Y, Z or none")
        here = ex.work_from_program(ex.s.pos)
        current = [here[ex.letters.index(l)] if l in ex.letters else 0.0 for l in "XYZ"]
        return self._origin(ex, bw, current), matrix

    def _cancel(self, ex, bw) -> None:
        s = ex.s
        s.modal["16"] = "G69"
        s.rotation = None
        if s.sys.pop("tilted", None) is not None or s.feature is not None:
            s.feature = None
        if s.tcp == TILT_KEY:
            s.tcp = ""
            s.sys.pop("tcp_group", None)
        s.sys.pop("tilted_points", None)
        ex.invalidate_transform()
        ex.resync_program_position()

    def _g53_1(self, ex, bw) -> bool:
        s = ex.s
        tilted = s.sys.get("tilted")
        if tilted is None or tilted.get("code") == "G68.3":
            raise ControllerAlarm("tilted_g53_1", "G53.1 without G68.2")
        if bw.addr or len(bw.g) > 1:
            raise ControllerAlarm("tilted_g53_1", "G53.1 must be alone in its block")
        kin = kinematics(ex)
        matrix = np.array(tilted["matrix"]).reshape(3, 3)
        direction = matrix @ np.array([0.0, 0.0, 1.0])
        current = {a.name: s.mach.get(a.letter, 0) * a.quantum for a in ex.axes if a.kind == "rotary"}
        solved = kin.orient(direction, current)
        target = {}
        for a in ex.axes:
            if a.kind == "rotary" and a.name in solved:
                target[a.letter] = int(round_half_away(solved[a.name] / a.quantum))
        rapid = s.modal.get("1", "G00") == "G00"
        feed = ex.feed_spec(rapid)
        if not rapid and (not s.feed_set or s.feed_si <= 0.0):
            raise ControllerAlarm("feed_zero", "G53.1 in G01 without F")
        ex.machine_move("rapid" if rapid else "linear", target, feed, modal={"tool_axis_control": "G53.1"})
        return True


class _Pending(Exception):
    """G68.2 P2: more points to come (the plane is set by the Q3 block)."""


class TiltedHandler(TiltedPlane):
    def execute(self, ex, code: str, spec, bw) -> bool:
        try:
            return super().execute(ex, code, spec, bw)
        except _Pending:
            return False


def register(replace: bool = True) -> None:
    hooks.register_transform("fanuc:G43.4", KinematicStage("rotary"), replace=replace)
    hooks.register_transform("fanuc:G43.5", KinematicStage("vector"), replace=replace)
    hooks.register_transform(TILT_KEY, KinematicStage("tilt"), replace=replace)
    hooks.register_motion_words("tcp", TcpWords(), replace=replace)
    hooks.register_gcode_handler("fanuc:tilted", TiltedHandler(), replace=replace)
    for code in ("G68.2", "G68.3", "G68.4"):
        hooks.register_gcode(code, GSpec(16, "ext", "fanuc:tilted"), variant="M", dialect="fanuc", replace=replace)
    hooks.register_gcode("G69", GSpec(16, "ext", "fanuc:tilted"), variant="M", dialect="fanuc", replace=replace)
    hooks.register_gcode("G53.1", GSpec(0, "ext", "fanuc:tilted"), variant="M", dialect="fanuc", replace=replace)
    common.register_alarms(FANUC_ALARMS, HAAS_ALARMS)


FANUC_ALARMS = {
    "tcp_illegal": ("PS5421", "ILLEGAL COMMAND IN G43.4/G43.5"),
    "tilted_format": ("PS5457", "G68.2 FORMAT ERROR"),
    "tilted_g53_1": ("PS5458", "ILLEGAL USE OF G53.1/G53.6"),
    "tilted_illegal": ("PS5462", "ILLEGAL COMMAND (G68.2/G69)"),
    "kinematics_no_solution": ("TW0022", "TOOL ORIENTATION NOT REACHABLE"),
    "rotary_out_of_range": ("TW0023", "ROTARY AXIS LIMIT FOR THE TOOL ORIENTATION"),
}
HAAS_ALARMS = {
    "tcp_illegal": ("2062", "TCPC OR DWO INCOMPATIBLE CODE"),
    "tilted_format": ("TW0032", "TILTED PLANE COMMAND ERROR"),
    "tilted_g53_1": ("TW0032", "TILTED PLANE COMMAND ERROR"),
    "tilted_illegal": ("TW0032", "TILTED PLANE COMMAND ERROR"),
    "kinematics_no_solution": ("TW0022", "TOOL ORIENTATION NOT REACHABLE"),
    "rotary_out_of_range": ("TW0023", "ROTARY AXIS LIMIT FOR THE TOOL ORIENTATION"),
}

register()

__all__ = ["Kinematics", "KinematicStage", "TiltedPlane", "TcpWords", "kinematics", "register", "CycleCall", "Step"]
