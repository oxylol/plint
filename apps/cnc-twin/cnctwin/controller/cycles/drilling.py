"""Canned cycles for drilling: FANUC M series G73 G74 G76 G81-G89 and G84.2/G84.3 (FS15 rigid tapping),
T series G83-G89 (face cycles drill along Z, side cycles along X), and the Haas NGC variants.

Every hole is the six operations of FANUC (fanuc-30ibplus-om-mc-2019 p. 45): 1 positioning of the plane
axes (rapid), 2 rapid to the R level, 3 hole machining, 4 the operation at the bottom, 5 retraction to R,
6 rapid to the initial level (G98). Levels along the drilling axis d (program frame, SI):

* R level: G90 the R word; G91 (and T series system A always) initial level + R (FANUC 5.1, lathe 4.3).
  Whether R was absolute is remembered with the modal data (R given in G90 stays a level when the block
  that repeats the hole is in G91, as in FANUC's example N023/N024 p. 76).
* Z level: the drilling-axis word, absolute in G90, R level + Z in G91 (core: ``data["absolute"]``).
* Return level: the initial level (G98, T series system A, G87 back boring) or R (G99).

Cycle by cycle (q = |Q|, d_73 = parameter 5114, d_83 = 5115, P = dwell):

* G81 feed to Z, rapid out; G82 + dwell; G85 feed out to R; G89 dwell then feed out; G86 M05 at Z, rapid
  out, spindle restarted at the return level; G88 dwell, M05, STOP (manual retract), then rapid to the
  return level and restart.
* G73 pecks of q, each followed by a rapid retract of d_73; G83 pecks of q, rapid to R after each, rapid
  back to d_83 above the last depth. Q missing or 0: PS0045.
* G76: dwell, M05 (5101#7 M5B = 0), M19, shift q (parameter 5148 direction, rapid), rapid out, shift back,
  spindle restarted. G87 (back boring, return always to the initial level): M19 and shift at the initial
  level, rapid to R (below the part), shift back, spindle on, feed up to Z, dwell, M19, shift, rapid to
  the initial level, shift back, spindle on.
* G84 / G74 conventional (floating) tapping: feed to Z (override off), dwell, spindle reversed (M04/M03),
  feed out to R, dwell, spindle restored. Rigid tapping (M29 before or in the block, parameter 5200#0,
  G84.2/G84.3, Haas G84/G74 always): the spindle is position-controlled and synchronised with the drilling
  axis: lead = F (G95) or F / S (G94); the tap moves are LINEAR with ``Feed(PER_REV, lead, override off)``,
  ``modal rigid_tap``, ``modal tap {lead_m, direction, phase}``; motion reverses the spindle for the
  outward move. P dwells at Z and at R; Q gives peck tapping (high-speed: retract by 5213, or standard:
  out to R and in at feed to 5213 before the last depth). Extraction override: 5211 (DOV) or Haas J.
  PS0200 (S above the spindle maximum, CSS active, S0), PS0201 (no F), PS0203 (axis motion or S between
  M29 and the cycle), PS0204 (M29 inside the tapping mode), PS0206 (plane change in rigid mode).
* Haas: G73/G83 with I (first peck), J (reduction per peck), K (minimum peck) instead of Q; Setting 22
  (G73 retract / G83 clearance), Setting 52 (G83 retract above R); G76 shift by Q (Setting 27 direction)
  or I/J; G87 is bore in and manual retract (stop at Z, then rapid out).

Plane change while the drilling mode is active: PS0549 (PS0206 in rigid mode). The cancel hook
(G80, group 01, reset) ends rigid mode (MODE rigid_tap false) and clears S as FANUC does.
"""

from __future__ import annotations

import math

from ..alarms import ControllerAlarm
from ..hooks import Step, register_resumer
from ..ir import Command, Kind
from . import common

TAP_CODES = ("G84", "G74", "G84.2", "G84.3")
PLANE_CODE = {"XY": 17.0, "ZX": 18.0, "YZ": 19.0}


def _rapid(axis: str, value: float) -> Step:
    return Step("rapid", {axis: value})


def _feed(axis: str, value: float, feed: float, mode: str, *, override: bool = True, modal: dict | None = None,
          exact: bool = False) -> Step:
    return Step("linear", {axis: value}, feed=feed, feed_mode=mode, override=override, exact_stop=exact,
                modal=dict(modal or {}))


def _dwell(seconds: float) -> list[Step]:
    return [Step("dwell", dwell_s=float(seconds))] if seconds > 0.0 else []


def _m(codes: list[int], words: dict | None = None) -> Step:
    return Step("mcodes", data={"m": list(codes), "words": dict(words or {})})


class DrillCycle:
    """One provider object per G code; ``haas`` selects the Haas NGC variant of the code."""

    def __init__(self, code: str, haas: bool = False):
        self.code = code
        self.haas = haas

    # ------------------------------------------------------------------ provider interface
    def expand(self, ex, call) -> list[Step]:
        lathe = bool(getattr(ex.handler, "T", False))
        d = call.extra.get("drilling_axis") or common.plane_axes(call.plane)[2]
        if d not in ex.by_letter:
            raise ControllerAlarm("illegal_plane", f"{call.code}: no drilling axis {d} on this path")
        self._check_mode(ex, call, lathe)
        levels = self._levels(ex, call, lathe, d)
        steps = self._position(call, d)
        code = self._effective_code(call.code, lathe)
        builder = getattr(self, "_" + code.replace(".", "_").lower())
        steps.extend(builder(ex, call, d, levels))
        return common.drop_null_moves(steps, call.current)

    def cancel(self, ex, code: str) -> None:
        """End of the drilling mode: rigid tapping released, its S cleared (FANUC 5.4.4)."""
        s = ex.s
        if s.sys.pop("rigid_tap", None):
            s.sys.pop("rigid_tap_at", None)
            s.sys.pop("rigid_haas", None)
            s.spindle.s_value = 0.0
            s.spindle.direction = 0
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"rigid_tap": False}))

    # ------------------------------------------------------------------ modal checks and levels
    def _check_mode(self, ex, call, lathe: bool) -> None:
        cycle = ex.s.cycle
        if cycle is None:
            return
        data = cycle.setdefault("data", {})
        plane = PLANE_CODE[call.plane]
        if "plane_code" not in data:
            data["plane_code"] = plane
            data["first_seq"] = float(ex.s.block_seq)
        elif data["plane_code"] != plane:
            rigid = bool(ex.s.sys.get("rigid_tap"))
            raise ControllerAlarm("rigid_plane_change" if rigid else "drill_plane_change",
                                  f"{call.code}: plane changed in the drilling mode")
        if "R" in call.words and not lathe:
            data["R_absolute"] = 1.0 if call.absolute else 0.0

    def _levels(self, ex, call, lathe: bool, d: str) -> dict:
        data = call.data
        initial = float(call.initial_level if call.initial_level is not None else call.current.get(d, 0.0))
        if lathe and self.haas:
            # Haas lathes: R is the position of the R plane (haas-ngc-lathe-om-2020 G81-G89; the radial cycles
            # G241-G249 give it as a diameter); without R the cycle starts from the starting plane
            axis = ex.by_letter.get(d)
            r_level = float(data["R"]) / (2.0 if axis is not None and axis.diameter else 1.0) if "R" in data \
                else initial
        elif lathe:
            r_level = initial + float(data.get("R", 0.0))
        else:
            r_abs = (call.absolute if "R" in call.words else bool(data.get("R_absolute", 1.0 if call.absolute else 0.0)))
            if "R" in data:
                r_level = float(data["R"]) if r_abs else initial + float(data["R"])
            else:
                r_level = 0.0 if call.absolute else initial
        if d in data:
            z_abs = bool(data.get("absolute", 1.0))
            z_level = float(data[d]) if z_abs else r_level + float(data[d])
        else:
            z_level = 0.0 if (call.absolute or lathe) else r_level
        back_to_initial = call.return_to_initial or lathe or self._effective_code(call.code, lathe) in ("G87", "G77")
        return {"initial": initial, "r": r_level, "z": z_level,
                "return": initial if back_to_initial else r_level,
                "sign": -1.0 if z_level < r_level else 1.0,
                "p": float(data.get("P", 0.0)), "q": abs(float(data.get("Q", 0.0)))}

    def _position(self, call, d: str) -> list[Step]:
        target = {l: v for l, v in call.position.items() if l != d and abs(v - call.current.get(l, v)) > 0.0}
        return [Step("rapid", target)] if target else []

    def _effective_code(self, code: str, lathe: bool) -> str:
        if lathe and self.haas:
            # Haas lathes run the mill sequences: face cycles along Z, radial cycles G241-G249 along X
            return HAAS_RADIAL.get(code, code)
        if lathe:
            return {"G83": "G83T", "G87": "G83T", "G84": "G84", "G88": "G84", "G85": "G85T", "G89": "G89T"}.get(code, code)
        return code

    # ------------------------------------------------------------------ simple cycles
    def _g81(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        return [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), _rapid(d, lv["return"])]

    def _g82(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        return [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), *_dwell(lv["p"]), _rapid(d, lv["return"])]

    def _g85(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        steps = [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), _feed(d, lv["r"], f, mode, exact=True)]
        if lv["return"] != lv["r"]:
            steps.append(_rapid(d, lv["return"]))
        return steps

    def _g89(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        steps = [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), *_dwell(lv["p"]),
                 _feed(d, lv["r"], f, mode, exact=True)]
        if lv["return"] != lv["r"]:
            steps.append(_rapid(d, lv["return"]))
        return steps

    def _g85t(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        return [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), *_dwell(lv["p"]),
                _feed(d, lv["r"], f, mode, exact=True), _rapid(d, lv["return"])]

    _g89t = _g85t

    def _restart_spindle(self, ex) -> list[Step]:
        direction = ex.s.spindle.direction
        if direction == 0:
            return []
        return [_m([3 if direction < 0 else 4])]

    def _g86(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        return [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), _m([5]), _rapid(d, lv["return"]),
                *self._restart_spindle(ex)]

    def _g88(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        dwell = [] if self.haas and call.code == "G87" else _dwell(lv["p"])
        resume = Step("resume", provider="cycles:manual_retract",
                      data={"axis": d, "level": lv["return"], "code": call.code,
                            "restart": ex.s.spindle.direction})
        return [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), *dwell, _m([5]), resume]

    # ------------------------------------------------------------------ peck drilling
    def _haas_choice(self, ex, call, key: str, own: tuple[str, ...], other: str) -> bool:
        """Haas: True when the cycle's ``own`` words (I J K pecks, I J shift) apply rather than ``other`` (Q).
        The cycle data are modal until G80 (haas-ngc-mill-om-2020 canned cycles), so a later hole of the same
        cycle keeps the choice of the block that gave either; the choice is kept with the modal data."""
        if any(letter in call.words for letter in own):
            choice = True
        elif other in call.words:
            choice = False
        elif key in call.data:
            choice = bool(call.data[key])
        else:
            choice = own[0] in call.data and other not in call.data
        cycle = ex.s.cycle
        if cycle is not None:
            cycle.setdefault("data", {})[key] = 1.0 if choice else 0.0
        return choice

    def _pecks(self, ex, call, lv) -> list[float]:
        """Depth levels of a peck cycle (the last is Z); Haas I/J/K (modal cycle data) when I is given."""
        start, end, sign = lv["r"], lv["z"], lv["sign"]
        total = abs(end - start)
        first = float(call.data["I"]) if self.haas and self._haas_choice(ex, call, "haas_peck_ijk", ("I",), "Q") \
            and "I" in call.data else None
        if first is not None:
            reduce = abs(float(call.data.get("J", 0.0)))
            minimum = abs(float(call.data.get("K", 0.0)))
            depths, done, step = [], 0.0, abs(first)
            if step <= 0.0:
                raise ControllerAlarm("cycle_q_missing", f"{call.code}: I must be positive")
            while done + step < total - 1e-12:
                done += step
                depths.append(start + sign * done)
                step = max(step - reduce, minimum) if minimum > 0.0 else max(step - reduce, 1e-6)
            depths.append(end)
            return depths
        q = lv["q"]
        if q <= 0.0:
            raise ControllerAlarm("cycle_q_missing", f"{call.code} needs Q > 0")
        count = max(1, int(math.ceil(total / q - 1e-9)))
        return [start + sign * min(q * k, total) for k in range(1, count)] + [end]

    def _g73(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        p = common.params(ex)
        back = p["haas_g73_retract_m"] if self.haas else p["g73_retract_m"]
        steps = [_rapid(d, lv["r"])]
        depths = self._pecks(ex, call, lv)
        for k, depth in enumerate(depths):
            steps.append(_feed(d, depth, f, mode, exact=k == len(depths) - 1))
            if k < len(depths) - 1:
                steps.append(_rapid(d, depth - lv["sign"] * back))
        steps.extend(_dwell(lv["p"]))
        steps.append(_rapid(d, lv["return"]))
        return steps

    def _g83(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        p = common.params(ex)
        clear = p["haas_g73_retract_m"] if self.haas else p["g83_clearance_m"]
        above_r = p["haas_g83_retract_above_r_m"] if self.haas else 0.0
        retract = lv["r"] - lv["sign"] * above_r
        steps = [_rapid(d, lv["r"])]
        depths = self._pecks(ex, call, lv)
        last = lv["r"]
        for k, depth in enumerate(depths):
            if k > 0:
                steps.append(_rapid(d, last - lv["sign"] * clear))
            steps.append(_feed(d, depth, f, mode, exact=k == len(depths) - 1))
            if k < len(depths) - 1:
                steps.append(_rapid(d, retract))
            last = depth
        steps.extend(_dwell(lv["p"]))
        steps.append(_rapid(d, lv["return"]))
        return steps

    def _g83t(self, ex, call, d, lv) -> list[Step]:
        """T series G83/G87: plain drilling without Q, else high-speed (5101#2 RTR = 0) or normal pecks."""
        if lv["q"] <= 0.0:
            return self._g82(ex, call, d, lv)
        if common.params(ex)["lathe_g83_high_speed"]:
            return self._g73(ex, call, d, lv)
        return self._g83(ex, call, d, lv)

    # ------------------------------------------------------------------ boring with orientation and shift
    def _shift(self, ex, call, d: str, amount: float, back: bool) -> dict[str, float]:
        """Program-frame position after the oriented-spindle shift (parameter 5148 / Haas Setting 27 or I, J)."""
        p = common.params(ex)
        work = {}
        if self.haas and self._haas_choice(ex, call, "haas_shift_ij", ("I", "J"), "Q") and \
                ("I" in call.data or "J" in call.data):
            work = {"X": float(call.data.get("I", 0.0)), "Y": float(call.data.get("J", 0.0))}
        else:
            axis, sign = common.signed_axis(p["haas_g76_shift_direction"] if self.haas else p["g76_shift_direction"])
            if axis == d:
                raise ControllerAlarm("unsupported_function", f"{call.code}: shift direction along the drilling axis")
            work = {axis: sign * amount}
        delta = common.program_delta(ex, work)
        out = {}
        for letter, value in delta.items():
            if letter == d or abs(value) == 0.0:
                continue
            base = call.position.get(letter, 0.0)
            out[letter] = base if back else base + value
        return out

    def _orient(self, ex) -> list[Step]:
        p = common.params(ex)
        steps = [_m([5])] if p["g76_m05_before_orient"] else []
        angle = math.degrees(p["orient_angle_rad"])
        steps.append(_m([19], {"R": [angle, True, False]}))
        return steps

    def _g76(self, ex, call, d, lv) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        q = lv["q"]
        steps = [_rapid(d, lv["r"]), _feed(d, lv["z"], f, mode, exact=True), *_dwell(lv["p"]), *self._orient(ex)]
        shifted = self._shift(ex, call, d, q, back=False)
        if shifted:
            steps.append(Step("rapid", shifted))
        steps.append(_rapid(d, lv["return"]))
        if shifted:
            steps.append(Step("rapid", self._shift(ex, call, d, q, back=True)))
        steps.extend(self._restart_spindle(ex))
        return steps

    def _g77(self, ex, call, d, lv) -> list[Step]:
        """Haas G77 back bore (haas-ngc-mill-om-2020 p. 353-355): "works similar to G76 except that before
        beginning to bore the hole, it moves the tool to clear the hole, moves down into the hole, and bores
        to the specified depth"; R lies past the bottom of the part, the return is always to the initial
        level. The sequence is FANUC's G87 back boring with the Haas shift (Q along Setting 27, else I/J).
        Spindle: the last commanded direction, M03 when the program has commanded none, no spin after M05."""
        return self._back_bore(ex, call, d, lv, self._haas_restart(ex))

    def _haas_restart(self, ex) -> list[Step]:
        direction = ex.s.spindle.direction
        if direction != 0:
            return [_m([3 if direction < 0 else 4])]
        return [] if ex.s.sys.get("spindle_commanded") else [_m([3])]

    def _g87(self, ex, call, d, lv) -> list[Step]:
        if self.haas:
            return self._g88(ex, call, d, lv)
        return self._back_bore(ex, call, d, lv, self._restart_spindle(ex) or [_m([3])])

    def _back_bore(self, ex, call, d, lv, restart: list[Step]) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        q = lv["q"]
        shifted = self._shift(ex, call, d, q, back=False)
        home = self._shift(ex, call, d, q, back=True)
        steps = list(self._orient(ex))
        if shifted:
            steps.append(Step("rapid", shifted))
        steps.append(_rapid(d, lv["r"]))
        if shifted:
            steps.append(Step("rapid", home))
        steps.extend(restart)
        steps.append(_feed(d, lv["z"], f, mode, exact=True))
        steps.extend(_dwell(lv["p"]))
        steps.extend(self._orient(ex))
        if shifted:
            steps.append(Step("rapid", shifted))
        steps.append(_rapid(d, lv["initial"]))
        if shifted:
            steps.append(Step("rapid", home))
        steps.extend(restart)
        return steps

    # ------------------------------------------------------------------ tapping
    def _g84(self, ex, call, d, lv) -> list[Step]:
        return self._tap(ex, call, d, lv, -1 if call.code in ("G84", "G88", "G84.2") else 1)

    _g74 = _g84
    _g84_2 = _g84

    def _g84_3(self, ex, call, d, lv) -> list[Step]:
        return self._tap(ex, call, d, lv, 1)

    def _g184(self, ex, call, d, lv) -> list[Step]:
        """Haas lathe G184 reverse tapping for left-hand threads (haas-ngc-lathe-om-2020 p. 371): rigid, the
        control turns the spindle counter-clockwise itself."""
        return self._tap(ex, call, d, lv, 1)

    def _rigid(self, ex, call) -> bool:
        if call.code in ("G84.2", "G84.3") or self.haas:
            return True
        return bool(ex.s.sys.get("rigid_tap")) or bool(common.params(ex)["rigid_tap_by_g84"])

    def _tap(self, ex, call, d, lv, direction: int) -> list[Step]:
        if not self._rigid(ex, call):
            return self._floating_tap(ex, call, d, lv, direction)
        return self._rigid_tap(ex, call, d, lv, direction)

    def _floating_tap(self, ex, call, d, lv, direction: int) -> list[Step]:
        f, mode = common.feed_of(ex, call), call.feed_mode
        tap = {"tap": {"rigid": False, "direction": "cw" if direction < 0 else "ccw", "cycle": call.code}}
        reverse = 4 if direction < 0 else 3
        restore = 3 if direction < 0 else 4
        return [_rapid(d, lv["r"]),
                _feed(d, lv["z"], f, mode, override=False, modal=dict(tap, phase="in"), exact=True),
                *_dwell(lv["p"]), _m([reverse]),
                _feed(d, lv["r"], f, mode, override=False, modal=dict(tap, phase="out"), exact=True),
                *_dwell(lv["p"]), _m([restore])] + ([_rapid(d, lv["return"])] if lv["return"] != lv["r"] else [])

    def _rigid_tap(self, ex, call, d, lv, direction: int) -> list[Step]:
        s = ex.s
        p = common.params(ex)
        at = s.sys.get("rigid_tap_at")
        cycle = s.cycle or {}
        if at is not None:
            if int(at.get("seq", -1)) == s.block_seq and cycle.get("data", {}).get("first_seq", s.block_seq) < s.block_seq:
                raise ControllerAlarm("rigid_axis_operation", "M29 inside the tapping mode")
            moved = any(at["pos"].get(l) != c for l, c in s.pos.items())
            if moved or at.get("s") != s.spindle.s_value:
                raise ControllerAlarm("rigid_program_miss", "axis motion or S between M29 and the tapping cycle")
            s.sys.pop("rigid_tap_at", None)
        if s.spindle.css:
            raise ControllerAlarm("rigid_s_illegal", "rigid tapping during constant surface speed")
        rpm = float(s.spindle.s_value)
        spindle = ex.machine.spindles.get(ex.machine.path_spindle[ex.ctx.path])
        top = spindle.max_rad_s * 60.0 / (2.0 * math.pi) if spindle is not None and spindle.max_rad_s > 0 else 0.0
        if rpm <= 0.0 or (top > 0.0 and rpm > top + 1e-9):
            raise ControllerAlarm("rigid_s_illegal", f"S{rpm:g} for rigid tapping")
        feed = call.data.get("F")
        if feed is None or feed <= 0.0:
            raise ControllerAlarm("rigid_feed_missing", call.code)
        lead = float(feed) if call.feed_mode == "per_rev" else float(feed) / (rpm / 60.0)
        override = self._extraction_override(ex, call, rpm, top, p)
        steps: list[Step] = []
        if not s.sys.get("rigid_tap"):
            steps.append(Step("mode", data={"rigid_tap": True, "rigid_haas": self.haas}))
        tap = {"lead_m": lead, "direction": "cw" if direction < 0 else "ccw", "cycle": call.code, "rigid": True}

        def move(target: float, phase: str) -> Step:
            return _feed(d, target, lead, "per_rev", override=False, exact=True,
                         modal={"rigid_tap": True, "tap": dict(tap, phase=phase)})

        def spindle(speed: float) -> Step:
            return Step("command", command=Command(Kind.SPINDLE, None, spindle=common.spindle_payload(ex, direction, speed)))

        def spin_in() -> Step:
            return spindle(rpm)

        def spin_out() -> Step:
            return spindle(rpm * override)

        steps += [_rapid(d, lv["r"]), spin_in()]
        q = lv["q"]
        if q > 0.0:
            depths = self._pecks(ex, call, lv)
            back = p["peck_tap_return_m"]
            last = lv["r"]
            for k, depth in enumerate(depths):
                if k > 0 and not p["peck_tap_high_speed"]:
                    steps += [spin_in(), move(last - lv["sign"] * back, "in")]
                steps.append(move(depth, "in"))
                if k < len(depths) - 1:
                    steps.append(spin_out())
                    steps.append(move(depth - lv["sign"] * back if p["peck_tap_high_speed"] else lv["r"], "out"))
                    if p["peck_tap_high_speed"]:
                        steps.append(spin_in())
                last = depth
        else:
            steps.append(move(lv["z"], "in"))
        steps += _dwell(lv["p"])
        steps += [spin_out(), move(lv["r"], "out"), *_dwell(lv["p"])]
        if override != 1.0:
            steps.append(spin_in())
        if lv["return"] != lv["r"]:
            steps.append(_rapid(d, lv["return"]))
        return steps

    def _extraction_override(self, ex, call, rpm: float, top: float, p: dict) -> float:
        """Spindle speed factor of the extraction: Haas J (retract multiple), else parameter 5211 (DOV),
        limited so the extraction speed stays within the spindle maximum (FANUC 5.4.5.1 note 3)."""
        factor = float(p["rigid_extraction_override"])
        if self.haas and common.word(call, "J") is not None:
            factor = max(1.0, float(common.word(call, "J").value))
        if top > 0.0 and rpm * factor > top:
            factor = top / rpm
        return factor


class ManualRetract:
    """G88 (and Haas G87): the program stops at the bottom (feed hold state, manual retract); CYCLE START
    returns the tool at rapid to the return level and restarts the spindle."""

    def resume(self, ex, data: dict, feedback) -> list[Step]:
        ex.emit(Command(Kind.STOP, ex.line_ref(), stop=data.get("code", "G88"),
                        message=f"{data.get('code', 'G88')}: retract manually, then CYCLE START"))
        ex.wait("stop", reason=data.get("code", "G88"))
        steps = [_rapid(data["axis"], float(data["level"]))]
        direction = int(data.get("restart", 0))
        if direction != 0:
            steps.append(_m([3 if direction < 0 else 4]))
        return steps


register_resumer("cycles:manual_retract", ManualRetract(), replace=True)

MILL_CODES = ("G73", "G74", "G76", "G81", "G82", "G83", "G84", "G85", "G86", "G87", "G88", "G89")
FS15_CODES = ("G84.2", "G84.3")
LATHE_CODES = ("G83", "G84", "G85", "G87", "G88", "G89")
# Haas NGC (haas-ngc-mill-om-2020, haas-ngc-lathe-om-2020): mill G77 back bore; lathe face cycles along Z
# (R absolute) and the radial cycles along X (R and X in diameter), each the mill sequence of its partner
HAAS_MILL_CODES = ("G77",)
HAAS_LATHE_FACE = ("G81", "G82", "G83", "G84", "G85", "G86", "G89", "G184")
HAAS_RADIAL = {"G241": "G81", "G242": "G82", "G243": "G83", "G245": "G85", "G246": "G86", "G249": "G89"}
HAAS_LATHE_CODES = HAAS_LATHE_FACE + tuple(HAAS_RADIAL)

__all__ = ["DrillCycle", "ManualRetract", "MILL_CODES", "FS15_CODES", "LATHE_CODES", "TAP_CODES", "HAAS_MILL_CODES",
           "HAAS_LATHE_FACE", "HAAS_RADIAL", "HAAS_LATHE_CODES"]
