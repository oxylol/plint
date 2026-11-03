"""Execution of TNC 640 ISO blocks (``iso`` kind): modal G codes, then the motion of the active group.

Mapping to the Klartext semantics (TNC 640 ISO manual, function overview): G00 / G01 straight line at
rapid / feed (both modal, unlike FMAX), G02 / G03 circle about the centre I J K given in the block or before
(G90 absolute, G91 incremental from the last position) or with the radius R (CR), G05 circle with the last
direction, G06 tangential circle, G10 / G11 polar line (R radius, H angle about the pole), G12 / G13 / G15
polar circle, G16 tangential polar circle, G24 / G25 chamfer / rounding R between the neighbouring elements
(G26 / G27 tangential approach / departure are not modelled: TW0012), G28 mirroring, G29 the last position as pole, G30 / G31 blank, G36 S spindle orientation, G38 stop,
G40-G42 radius compensation (modal), G51 T tool preselection, G54 datum shift, G62 T tolerance, G70 / G71
units, G72 F scaling, G73 H rotation, G79 cycle call, G90 / G91, G99 T L R tool definition, G200-G253 cycle
definitions with their Q parameters, T tool call (axis by G17 / G18 / G19).
"""

from __future__ import annotations

import math

from ...pipeline import PLANES
from ..siemens840d.kit import Alarm, arc_end_tangent
from . import contour, cycles
from .handler import ISO_MOTION

KNOWN = {"G17", "G18", "G19", "G40", "G41", "G42", "G90", "G91", "G70", "G71", "G04", "G24", "G25", "G26", "G27",
         "G28", "G29", "G30", "G31", "G36", "G38", "G51", "G53", "G54", "G62", "G72", "G73", "G79", "G98", "G99"}


def execute(h, f) -> None:
    ex, s = h.ex, h.s
    codes = f["g"]
    motion_codes = [g for g in codes if g in ISO_MOTION]
    if len(motion_codes) > 1:
        raise Alarm("multiple_codes", " ".join(motion_codes))
    cycle_codes = [g for g in codes if g.startswith("G2") and len(g) == 4]
    unknown = [g for g in codes if g not in ISO_MOTION and g not in KNOWN and g not in cycle_codes]
    if unknown:
        raise Alarm("syntax", " ".join(unknown))
    keys = f["keys"]
    absolute = s.modal.get("distance", "G90") == "G90"
    for code in codes:
        if code in ("G90", "G91"):
            s.modal["distance"] = code
            absolute = code == "G90"
        elif code in ("G17", "G18", "G19"):
            if s.modal.get("7", "G40") != "G40" and s.modal.get("plane") != code:
                raise Alarm("comp_switch", "plane change with radius compensation")
            s.modal["plane"] = code
        elif code in ("G70", "G71"):
            s.units = "inch" if code == "G70" else "mm"
    if cycle_codes:
        _cycle_definition(h, f, cycle_codes)
        return
    if "G99" in codes:
        h.do_tool_def({"tool": keys.get("T"), "keys": {k: v for k, v in keys.items() if k in ("L", "R")}})
        return
    if "G51" in codes:
        h.do_tool_def({"tool": keys.get("T"), "keys": {}})
        return
    if "G30" in codes or "G31" in codes:
        h.do_blk({"form": "0.1" if "G30" in codes else "0.2", "axis": None, "coords": f["coords"]})
        return
    if "G28" in codes:
        hh = s.sys["hh"]
        hh["mirror"] = [1 if any(c[0] == l for c in f["coords"]) else 0 for l in "XYZ"]
        h.update_transform()
        return
    if "G53" in codes:
        raise Alarm("unsupported_function", "G53 datum shift from the datum table (only G54 values in the program "
                    "and the presets of work_offsets_m are modelled)")
    if "G54" in codes:
        values = {letter: (not absolute, h.number(node)) for letter, inc, node in f["coords"]}
        h.datum_shift(values)
        return
    if "G72" in codes:
        factor = h.number(keys["F"]) if "F" in keys else 1.0
        if factor <= 0.0:
            raise Alarm("illegal_scale", f"F{factor}")
        s.sys["hh"]["scale"] = [factor] * 3
        s.sys["hh"]["centre"] = [0.0, 0.0, 0.0]
        h.update_transform()
        return
    if "G73" in codes:
        angle = h.number(keys["H"]) if "H" in keys else 0.0
        hh = s.sys["hh"]
        hh["rot"] = float(hh["rot"] + angle if not absolute else angle)
        h.update_transform()
        return
    if "G04" in codes:
        if "F" not in keys:
            raise Alarm("syntax", "G04 needs F")
        ex.dwell(h.number(keys["F"]))
        return
    if "G36" in codes:
        angle = h.number(keys["S"]) if "S" in keys else 0.0
        s.spindle.direction = 0
        s.spindle.oriented = True
        h.emit_spindle(orient=math.radians(angle))
        return
    if "G62" in codes:
        if "T" in keys:
            tol = h.length_si(h.number(keys["T"]))
            s.smoothing.update({"corner": "blend", "tolerance_m": tol, "mode_code": "G62"})
        return
    if "G29" in codes:
        s.sys["pole"] = list(h.plane_point())
        return
    if f.get("pgm"):
        h.do_call_pgm({"name": f["pgm"]})
        return
    if f.get("label_call"):
        from .formula import value_node

        label, rep = f["label_call"]
        h.do_call_lbl({"label": label, "rep": value_node(str(rep)) if rep else None})
        return
    if "T" in keys and "G99" not in codes:
        axis = {"G17": "Z", "G18": "Y", "G19": "X"}.get(s.modal.get("plane", "G17"), "Z")
        call = {"tool": keys["T"], "axis": axis, "keys": {k: v for k, v in keys.items() if k in ("S", "F")}}
        h.do_tool_call(call)
        keys = {k: v for k, v in keys.items() if k not in ("T", "S", "F")}
    elif "S" in keys:
        h.do_tool_call({"tool": None, "axis": None, "keys": {"S": keys["S"]}})
        keys = {k: v for k, v in keys.items() if k != "S"}
    if "G38" in codes:
        h.do_stop({"m": f["m"]})
        return
    if "G79" in codes:
        before, after = h.m_split(f["m"])
        for m in before:
            h.m_function(m, None)
        cycles.call_defined(h, None)
        for m in after:
            h.m_function(m, None)
        return
    _motion(h, f, codes, keys, motion_codes, absolute)


def _comp(codes) -> str | None:
    for code, name in (("G40", "R0"), ("G41", "RL"), ("G42", "RR")):
        if code in codes:
            return name
    return None


def _coords(f, absolute: bool):
    return [(letter, not absolute, node) for letter, inc, node in f["coords"]]


def _set_pole(h, f, absolute: bool) -> bool:
    a, b, _ = PLANES[h.plane()]
    names = {"X": "I", "Y": "J", "Z": "K"}
    given = {letter: node for letter, node in f["ijk"]}
    if not given:
        return False
    here = h.plane_point()
    pole = list(h.s.sys.get("pole") or here)
    for k, axis in enumerate((a, b)):
        node = given.get(names[axis])
        if node is not None:
            value = h.axis_si(axis, h.number(node))
            pole[k] = here[k] + value if not absolute else value
    h.s.sys["pole"] = [float(pole[0]), float(pole[1])]
    return True


def _motion(h, f, codes, keys, motion_codes, absolute: bool) -> None:
    s = h.s
    if motion_codes:
        s.modal["iso_motion"] = motion_codes[0]
    motion = s.modal.get("iso_motion", "G00")
    kind = ISO_MOTION[motion]
    feed = ("F", keys["F"]) if "F" in keys else None
    common = {"comp": _comp(codes), "m": list(f["m"]), "coords": _coords(f, absolute), "polar": {}, "keys": {},
              "dr": None, "feed": feed}
    pole_set = _set_pole(h, f, absolute)
    if "G24" in codes or "G25" in codes:
        if f["coords"]:
            raise Alarm("syntax", "G24 / G25 take only R")
        h.transition_block({"feed": feed}, "CHF" if "G24" in codes else "RND")
        return
    if "G26" in codes or "G27" in codes:
        raise Alarm("unsupported_function", "G26 / G27 tangential approach and departure (use APPR / DEP)")
    if not f["coords"] and not keys.get("R") and not keys.get("H") and not pole_set:
        # no end point: the modal motion does not repeat (a circle or tangent arc would end where it
        # starts); compensation, feed and M functions of the block still act
        if not f["m"] and feed is None:
            if common["comp"] is not None:
                h.compensation(common["comp"], "iso_line", False)
            return
        h.positioning(common, "iso_line", lambda rapid, machine: False)
        return
    if pole_set and not f["coords"] and motion not in ("G02", "G03", "G05") and "R" not in keys:
        return
    if kind in ("rapid", "linear"):
        if kind == "rapid":
            common["feed"] = ("FMAX",)
        h.positioning(common, "iso_line", lambda rapid, machine: _line(h, common, rapid, machine))
        return
    if kind in ("cw", "ccw", "circle"):
        direction = {"cw": -1, "ccw": 1}.get(kind, s.sys.get("dr_last"))
        if direction is None:
            raise Alarm("arc_direction", "G05 without a previous direction")
        if not f["coords"] and pole_set:
            return
        if "R" in keys and not pole_set:
            h.do_CR(dict(common, dr=direction, keys={"R": keys["R"]}))
        else:
            h.do_C(dict(common, dr=direction))
        return
    if kind == "ct":
        h.do_CT(common)
        return
    polar = {}
    if "R" in keys:
        polar["PR"] = (not absolute, keys["R"])
    if "H" in keys:
        polar["PA"] = (not absolute, keys["H"])
    common["polar"] = polar
    if kind in ("polar_rapid", "polar_linear"):
        if kind == "polar_rapid":
            common["feed"] = ("FMAX",)
        common["coords"] = [c for c in common["coords"] if c[0] not in PLANES[h.plane()][:2]]
        h.do_LP(common)
    elif kind in ("polar_cw", "polar_ccw", "polar_circle"):
        direction = {"polar_cw": -1, "polar_ccw": 1}.get(kind, s.sys.get("dr_last"))
        if direction is None:
            raise Alarm("arc_direction", "G15 without a previous direction")
        common["dr"] = direction
        common["polar"] = {k: v for k, v in polar.items() if k == "PA"}
        h.do_CP(common)
    else:
        h.do_CTP(common)


def _line(h, f, rapid: bool, machine: bool) -> bool:
    target = h.targets(f["coords"])
    if 92 in f.get("m", []):
        target = {k: v + h.ex.by_letter[k].home for k, v in target.items()}
    h.move_line(target, rapid, None if (rapid or machine) else h.corner_for(), machine)
    return bool(target)


def _cycle_definition(h, f, codes) -> None:
    number = int(codes[0][1:])
    if number not in cycles.CALL_ACTIVE and number != 247:
        from .tables import TNC640_CYCLES

        if number in TNC640_CYCLES:
            raise Alarm("unsupported_function", f"{codes[0]} (a TNC 640 cycle this build does not model)")
        raise Alarm("cycle_undefined", codes[0])
    cycles.define(h, {"number": number, "sub": None, "params": f["params"], "words": []})


def iso_next_tangent(h, f, start):
    s = h.s
    codes = f["g"]
    motion = next((g for g in codes if g in ISO_MOTION), s.modal.get("iso_motion", "G00"))
    absolute = not ("G91" in codes or (s.modal.get("distance") == "G91" and "G90" not in codes))
    coords = [(letter, not absolute, node) for letter, inc, node in f["coords"]]
    end, _ = contour.eval_point(h, {"coords": coords}, start)
    if motion in ("G00", "G01") and end is not None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 0.0 else None
    if motion in ("G02", "G03") and not f["keys"].get("R"):
        return arc_end_tangent(h.pole(), start, -1 if motion == "G02" else 1)
    return None


__all__ = ["execute", "iso_next_tangent"]
