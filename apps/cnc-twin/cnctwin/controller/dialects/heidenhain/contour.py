"""APPR / DEP approach and departure and FK free contour programming (HEIDENHAIN TNC 640 Klartext).

All geometry is in the working plane of the program frame (SI); the radius compensation itself is the
core's cutter compensation provider (FANUC type A start-up and cancel, docs/dialects.md "Hooks"), so
the approach elements are given as programmed-contour elements the provider offsets:

* APPR LT (X Y = first contour point PA, LEN): PH = PA - LEN t (t: start tangent of the next element);
  start-up PS -> PH, then PH -> PA (the tool arrives on the offset tangent line).
* APPR LN: PH = PA + (LEN + r) n (n the normal toward the tool, RL left / RR right); PS -> PH without
  compensation, then the start-up block PH -> PA ends at the offset point of PA (tool path along the
  normal, length LEN).
* APPR CT (CCA, R): arc tangent to the contour at PA, centre M = PA + |R| n on the tool side for R > 0
  (the opposite side for R < 0), centre angle CCA; start-up PS -> arc start PH, then the arc.
* APPR LCT (R): the same arc, entered by the tangent from PS (the tangent point on the arc's side).
* DEP LT (LEN): PN = PE + LEN t_end; DEP LN: PN = PE + (LEN + r) n; both are the cancel block.
* DEP CT (CCA, R): arc tangent at PE, then the compensation is cancelled without motion (the provider
  moves the tool from the offset point onto PN); DEP LCT (X Y, R): arc to the tangent point, then the
  cancel block to PN.
The polar variants (APPR PLT ... DEP PLCT) give PA / PN by PR / PA about CC.

FK elements are solved one at a time from the known start point and start tangent, with the element's
own data: FL / FLT (end point X Y / IX IY / PR PA about FPOL, direction AN / IAN / tangential, length
LEN), FC / FCT (end point, centre CCX CCY / ICCX ICCY / CCPR CCPA, radius R, direction DR, tangential).
CLSD+ marks the first element of a closed contour (its start point is kept), CLSD- the last one: its
missing end point coordinates are those of the CLSD+ start point.
An element with one solution is executed; with several (a line of given length to one coordinate, an
arc of given radius ...) the following FSELECT n picks one (solutions ordered by the end point's first,
then second plane coordinate); an element whose data do not fix it, contradictory data, or several
solutions without FSELECT stop with the FK error (TW0040 / TW0041; the TNC's own numbers were not in
the error list read for the database: a database entry with kind fk_undetermined / fk_contradiction
replaces them). Data relative to other blocks (RX RY RAN PAR DP PDX PDY RPR RPA) are not modelled.
"""

from __future__ import annotations

import math

from ...pipeline import PLANES
from ..siemens840d.kit import Alarm, arc_end_tangent, tangent_arc_centre

FK_UNMODELLED = ("RX", "RY", "RAN", "PAR", "DP", "PDX", "PDY", "RPR", "RPA")


def _unit(v):
    n = math.hypot(v[0], v[1])
    if n == 0.0:
        raise Alarm("approach_failed", "zero direction")
    return (v[0] / n, v[1] / n)


def _left(t):
    return (-t[1], t[0])


def _add(p, v, k=1.0):
    return (p[0] + k * v[0], p[1] + k * v[1])


def _rotate(point, centre, angle):
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    return (centre[0] + c * dx - s * dy, centre[1] + s * dx + c * dy)


# ====================================================================== shared helpers

def plane_counts(h, point) -> dict[str, int]:
    a, b, _ = PLANES[h.plane()]
    return {a: h.ex.counts(a, point[0]), b: h.ex.counts(b, point[1])}


def eval_point(h, f, base) -> tuple[tuple[float, float] | None, dict[str, int]]:
    """Plane point given by a block's coordinates (relative to ``base`` for I-coordinates) or polar values;
    also the other axes' targets."""
    ex = h.ex
    a, b, _ = PLANES[h.plane()]
    point = list(base)
    given = False
    others = {}
    for letter, inc, node in f.get("coords", []):
        value = h.value(node, lone=True)
        if value is None:
            continue
        if letter in (a, b):
            k = 0 if letter == a else 1
            si = h.axis_si(letter, value)
            point[k] = base[k] + si if inc else si
            given = True
        else:
            counts = ex.counts(letter, h.axis_si(letter, value))
            others[letter] = ex.s.pos.get(letter, 0) + counts if inc else counts
    polar = f.get("polar") or {}
    if polar:
        pole = h.pole()
        radius = math.hypot(base[0] - pole[0], base[1] - pole[1])
        angle = math.atan2(base[1] - pole[1], base[0] - pole[0])
        if "PR" in polar:
            inc, node = polar["PR"]
            value = h.length_si(h.number(node))
            radius = radius + value if inc else value
        if "PA" in polar:
            inc, node = polar["PA"]
            value = math.radians(h.number(node))
            angle = angle + value if inc else value
        point = [pole[0] + radius * math.cos(angle), pole[1] + radius * math.sin(angle)]
        given = True
    return (tuple(point) if given else None), others


def next_start_tangent(h, start) -> tuple[float, float]:
    """Start tangent at ``start`` of the contour element in the block after the current one."""
    ex, s = h.ex, h.s
    frame = s.frames[-1]
    program = ex.store.programs[frame.program]
    k = frame.pc + 1
    while k < len(program.blocks) and program.blocks[k].extra.kind in ("comment", "empty"):
        k += 1
    if k >= len(program.blocks):
        raise Alarm("approach_failed", "no contour element after APPR")
    block = program.blocks[k]
    nxt = block.extra
    if block.error is not None:
        raise Alarm(block.error[0], block.error[1])
    kind = nxt.kind
    f = nxt.f
    if kind in ("L", "LP"):
        end, _ = eval_point(h, f, start)
        if end is None:
            raise Alarm("approach_failed", "the first contour element has no end point in the plane")
        return _unit((end[0] - start[0], end[1] - start[1]))
    if kind in ("C", "CP"):
        if f.get("dr") is None:
            raise Alarm("arc_direction", "first contour element C without DR")
        return arc_end_tangent(h.pole(), start, f["dr"])
    if kind == "CR":
        end, _ = eval_point(h, f, start)
        if end is None or f.get("dr") is None or "R" not in f["keys"]:
            raise Alarm("approach_failed", "first contour element CR incomplete")
        centre = h.radius_centre(start, end, h.length_si(h.number(f["keys"]["R"])), f["dr"])
        return arc_end_tangent(centre, start, f["dr"])
    if kind == "FK" and f["element"] == "FL":
        if "AN" in f["keys"]:
            angle = math.radians(h.number(f["keys"]["AN"]))
            return (math.cos(angle), math.sin(angle))
        end, _ = eval_point(h, f, start)
        if end is not None:
            return _unit((end[0] - start[0], end[1] - start[1]))
    if kind == "iso":
        from .isoexec import iso_next_tangent

        tangent = iso_next_tangent(h, f, start)
        if tangent is not None:
            return tangent
    raise Alarm("approach_failed", f"cannot take the start direction of a {kind} block")


def _side(code: str | None) -> int:
    return {"RL": 1, "RR": -1}.get(code or "", 0)


def _feed(h, f):
    rapid, _ = h.feed_for(f)
    if rapid:
        raise Alarm("approach_failed", "FMAX in an approach or departure")


def _tangent_point(external, centre, radius: float, direction: int):
    """Tangent point on the circle (centre, radius) from ``external`` where the line continues into an arc
    turning in ``direction``."""
    dx, dy = external[0] - centre[0], external[1] - centre[1]
    d = math.hypot(dx, dy)
    if d <= radius + 1e-12:
        return None
    theta = math.acos(radius / d)
    base = math.atan2(dy, dx)
    for sign in (1.0, -1.0):
        angle = base + sign * theta
        point = (centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle))
        tangent = arc_end_tangent(centre, point, direction)
        line = (point[0] - external[0], point[1] - external[1])
        if line[0] * tangent[0] + line[1] * tangent[1] > 0.0:
            return point
    return None


# ====================================================================== APPR

def approach(h, f) -> None:
    ex, s = h.ex, h.s
    variant = f["variant"]
    polar = variant.startswith("P")
    base = variant[1:] if polar else variant
    code = f.get("comp") or "R0"
    side = _side(code)
    if code == "R0" and base in ("LN", "CT"):
        raise Alarm("appr_without_comp", f"APPR {variant} with R0")
    if s.modal.get("7", "G40") != "G40":
        raise Alarm("comp_start", "APPR with radius compensation already active")
    _feed(h, f)
    start = h.plane_point()
    pa, others = eval_point(h, f, start)
    if pa is None:
        raise Alarm("approach_failed", "APPR without the first contour point")
    t = next_start_tangent(h, pa)
    normal = _left(t)
    tool_side = (normal[0] * side, normal[1] * side) if side else normal
    before, after = h.m_split(f.get("m", []))
    for m in before:
        h.m_function(m, None)
    radius_tool = h.comp_radius()
    keys = f["keys"]
    if base == "LT":
        length = h.length_si(abs(h.number(_need(keys, "LEN"))))
        ph = _add(pa, t, -length)
        h.compensation(code, "APPR", True)
        h.move_line(dict(plane_counts(h, ph), **others), False)
        h.move_line(plane_counts(h, pa), False)
    elif base == "LN":
        length = h.length_si(abs(h.number(_need(keys, "LEN"))))
        ph = _add(pa, tool_side, length + radius_tool)
        h.move_line(dict(plane_counts(h, ph), **others), False)
        h.compensation(code, "APPR", True)
        h.move_line(plane_counts(h, pa), False)
    elif base in ("CT", "LCT"):
        radius = h.length_si(h.number(_need(keys, "R")))
        if radius == 0.0:
            raise Alarm("approach_failed", "R = 0")
        centre_side = tool_side if radius > 0.0 else (-tool_side[0], -tool_side[1])
        centre = _add(pa, centre_side, abs(radius))
        direction = 1 if (centre_side[0] * normal[0] + centre_side[1] * normal[1]) > 0.0 else -1
        if base == "CT":
            cca = math.radians(abs(h.number(_need(keys, "CCA"))))
            if not 0.0 < cca < 2.0 * math.pi:
                raise Alarm("approach_failed", "CCA")
            ph = _rotate(pa, centre, -direction * cca)
        else:
            ph = _tangent_point(start, centre, abs(radius), direction)
            if ph is None:
                raise Alarm("approach_failed", "the start point lies inside the approach arc")
        h.compensation(code, "APPR", True)
        h.move_line(dict(plane_counts(h, ph), **others), False)
        h.move_arc(plane_counts(h, pa), centre, direction)
    else:
        raise Alarm("syntax", f"APPR {variant}")
    s.sys["tangent"] = [t[0], t[1]]
    for m in after:
        h.m_function(m, None)


def _need(keys: dict, name: str):
    if name not in keys or keys[name] is None:
        raise Alarm("approach_failed", f"{name} missing")
    return keys[name]


# ====================================================================== DEP

def departure(h, f) -> None:
    ex, s = h.ex, h.s
    variant = f["variant"]
    polar = variant.startswith("P")
    base = variant[1:] if polar else variant
    current = s.modal.get("7", "G40")
    side = {"G41": 1, "G42": -1}.get(current, 0)
    if side == 0 and base in ("LN", "CT"):
        raise Alarm("appr_without_comp", f"DEP {variant} with inactive radius compensation")
    tangent = s.sys.get("tangent")
    if tangent is None:
        raise Alarm("departure_failed", "DEP without a preceding contour element")
    _feed(h, f)
    t = _unit(tangent)
    normal = _left(t)
    tool_side = (normal[0] * side, normal[1] * side) if side else normal
    pe = h.plane_point()
    keys = f["keys"]
    before, after = h.m_split(f.get("m", []))
    for m in before:
        h.m_function(m, None)
    radius_tool = h.comp_radius()
    if base == "LT":
        length = h.length_si(abs(h.number(_need_dep(keys, "LEN"))))
        h.compensation("R0", "DEP", True)
        h.move_line(plane_counts(h, _add(pe, t, length)), False)
    elif base == "LN":
        length = h.length_si(abs(h.number(_need_dep(keys, "LEN"))))
        h.compensation("R0", "DEP", True)
        h.move_line(plane_counts(h, _add(pe, tool_side, length + radius_tool)), False)
    elif base in ("CT", "LCT"):
        radius = h.length_si(h.number(_need_dep(keys, "R")))
        if radius == 0.0:
            raise Alarm("departure_failed", "R = 0")
        centre_side = tool_side if radius > 0.0 else (-tool_side[0], -tool_side[1])
        centre = _add(pe, centre_side, abs(radius))
        direction = 1 if (centre_side[0] * normal[0] + centre_side[1] * normal[1]) > 0.0 else -1
        if base == "CT":
            cca = math.radians(abs(h.number(_need_dep(keys, "CCA"))))
            if not 0.0 < cca < 2.0 * math.pi:
                raise Alarm("departure_failed", "CCA")
            pn = _rotate(pe, centre, direction * cca)
            h.move_arc(plane_counts(h, pn), centre, direction)
            h.compensation("R0", "DEP", True)
            h.move_line(plane_counts(h, pn), False)
        else:
            end, _ = eval_point(h, f, pe)
            if end is None:
                raise Alarm("departure_failed", "DEP LCT without the end point")
            reverse = _tangent_point(end, centre, abs(radius), -direction)
            if reverse is None:
                raise Alarm("departure_failed", "the end point lies inside the departure arc")
            h.move_arc(plane_counts(h, reverse), centre, direction)
            h.compensation("R0", "DEP", True)
            h.move_line(plane_counts(h, end), False)
    else:
        raise Alarm("syntax", f"DEP {variant}")
    if s.modal.get("7", "G40") != "G40":
        h.compensation("R0", "DEP", True)
    for m in after:
        h.m_function(m, None)


def _need_dep(keys: dict, name: str):
    if name not in keys or keys[name] is None:
        raise Alarm("departure_failed", f"{name} missing")
    return keys[name]


# ====================================================================== FK

def fk_pole(h, f) -> None:
    a, b, _ = PLANES[h.plane()]
    point = list(h.s.sys.get("fk_pole") or h.plane_point())
    for letter, inc, node in f["coords"]:
        if letter not in (a, b):
            raise Alarm("pole_illegal", f"FPOL with axis {letter}")
        if inc:
            raise Alarm("pole_illegal", "FPOL takes absolute coordinates")
        point[0 if letter == a else 1] = h.axis_si(letter, h.number(node))
    h.s.sys["fk_pole"] = [float(point[0]), float(point[1])]


def _fk_select(h) -> int | None:
    ex, s = h.ex, h.s
    frame = s.frames[-1]
    program = ex.store.programs[frame.program]
    k = frame.pc + 1
    while k < len(program.blocks) and program.blocks[k].extra.kind in ("comment", "empty"):
        k += 1
    if k < len(program.blocks) and program.blocks[k].extra.kind == "FSELECT":
        return int(program.blocks[k].extra.f["n"])
    return None


def _value(h, keys, name):
    node = keys.get(name)
    return None if node is None else h.value(node, lone=True)


def fk_element(h, f) -> None:
    s = h.s
    keys = f["keys"]
    unmodelled = [k for k in FK_UNMODELLED if k in keys]
    if unmodelled:
        raise Alarm("unsupported_function", "FK data relative to other blocks: " + " ".join(unmodelled))
    element = f["element"]
    start = h.plane_point()
    tangent = s.sys.get("tangent")
    closing = 0
    if "CLSD" in keys:
        if keys["CLSD"] is None:
            raise Alarm("syntax", "CLSD without + or -")
        closing = 1 if h.number(keys["CLSD"]) > 0 else -1
        if closing > 0:
            s.sys["fk_closed"] = [float(start[0]), float(start[1])]
    end, others = _fk_end(h, f, start)
    if closing < 0:
        # the last element of a closed contour ends at the start point of the CLSD+ element
        point = s.sys.pop("fk_closed", None)
        if point is None:
            raise Alarm("fk_contradiction", "CLSD- without CLSD+")
        for k in (0, 1):
            end.setdefault(k, point[k])
    if element in ("FL", "FLT"):
        solutions = _fk_line(h, f, element, start, tangent, end)
        kind = "line"
    else:
        solutions = _fk_arc(h, f, element, start, tangent, end)
        kind = "arc"
    if not solutions:
        raise Alarm("fk_contradiction", f"{element}: no solution with the given data")
    if solutions == "undetermined":
        raise Alarm("fk_undetermined", f"{element}: the element is not determined by its data")
    solutions = _distinct(sorted(solutions, key=lambda sol: (round(sol[0][0], 12), round(sol[0][1], 12))))
    if len(solutions) > 1:
        choice = _fk_select(h)
        if choice is None:
            raise Alarm("fk_undetermined", f"{element}: {len(solutions)} solutions, FSELECT needed")
        if not 1 <= choice <= len(solutions):
            raise Alarm("fk_contradiction", f"FSELECT {choice} of {len(solutions)}")
        chosen = solutions[choice - 1]
        s.sys["fk_selected"] = True
    else:
        chosen = solutions[0]

    def body(rapid, machine):
        if rapid:
            raise Alarm("circle_illegal", "FMAX in an FK element")
        target = dict(plane_counts(h, chosen[0]), **others)
        if kind == "line":
            h.move_line(target, False, h.corner_for())
        else:
            h.move_arc(target, chosen[1], chosen[2], corner=h.corner_for())
        return True

    h.positioning(f, "FL" if kind == "line" else "FC", body)


def _distinct(solutions: list) -> list:
    """Solutions that coincide (end point, and for arcs centre and direction, within 1 nm) count once: a
    semicircle of given radius has one centre, not two."""
    out = []
    for sol in solutions:
        for kept in out:
            if all(abs(x - y) <= 1e-9 for x, y in zip(sol[0], kept[0])) and (
                    len(sol) < 3 or (all(abs(x - y) <= 1e-9 for x, y in zip(sol[1], kept[1])) and sol[2] == kept[2])):
                break
        else:
            out.append(sol)
    return out


def _fk_end(h, f, start):
    """(end point if both plane coordinates are known else a dict of known coordinates, other axes)."""
    a, b, _ = PLANES[h.plane()]
    known = {}
    others = {}
    ex = h.ex
    for letter, inc, node in f.get("coords", []):
        value = h.value(node, lone=True)
        if value is None:
            continue
        if letter in (a, b):
            k = 0 if letter == a else 1
            si = h.axis_si(letter, value)
            known[k] = start[k] + si if inc else si
        else:
            counts = ex.counts(letter, h.axis_si(letter, value))
            others[letter] = ex.s.pos.get(letter, 0) + counts if inc else counts
    polar = f.get("polar") or {}
    if polar:
        pole = h.s.sys.get("fk_pole")
        if pole is None:
            raise Alarm("pole_missing", "FPOL not defined")
        if "PR" in polar and "PA" in polar:
            r = h.length_si(h.number(polar["PR"][1]))
            angle = math.radians(h.number(polar["PA"][1]))
            known = {0: pole[0] + r * math.cos(angle), 1: pole[1] + r * math.sin(angle)}
        else:
            known["polar"] = {k: (v[0], h.number(v[1])) for k, v in polar.items()}
    return known, others


def _fk_line(h, f, element, start, tangent, end) -> list | str:
    keys = f["keys"]
    direction = None
    if element == "FLT":
        if tangent is None:
            raise Alarm("fk_undetermined", "FLT without a preceding element")
        direction = _unit(tangent)
    if "AN" in keys:
        angle = math.radians(h.number(keys["AN"]))
        d2 = (math.cos(angle), math.sin(angle))
        if direction is not None and abs(direction[0] * d2[1] - direction[1] * d2[0]) > 1e-9:
            return []
        direction = d2
    if "IAN" in keys:
        if tangent is None:
            raise Alarm("fk_undetermined", "IAN without a preceding element")
        base = math.atan2(tangent[1], tangent[0])
        angle = base + math.radians(h.number(keys["IAN"]))
        direction = (math.cos(angle), math.sin(angle))
    length = h.length_si(abs(_value(h, keys, "LEN"))) if _value(h, keys, "LEN") is not None else None
    if "polar" in end:
        return "undetermined"
    if 0 in end and 1 in end:
        point = (end[0], end[1])
        if direction is not None:
            dx, dy = point[0] - start[0], point[1] - start[1]
            if abs(dx * direction[1] - dy * direction[0]) > 1e-9 or dx * direction[0] + dy * direction[1] < 0.0:
                return []
        if length is not None and abs(math.hypot(point[0] - start[0], point[1] - start[1]) - length) > 1e-9:
            return []
        return [(point,)]
    if direction is not None and length is not None:
        if 0 in end or 1 in end:
            return []
        return [(_add(start, direction, length),)]
    if direction is not None and (0 in end or 1 in end):
        k = 0 if 0 in end else 1
        if abs(direction[k]) < 1e-15:
            return []
        t = (end[k] - start[k]) / direction[k]
        if t <= 0.0:
            return []
        return [(_add(start, direction, t),)]
    if length is not None and (0 in end or 1 in end):
        k = 0 if 0 in end else 1
        other = 1 - k
        delta = end[k] - start[k]
        if abs(delta) > length + 1e-12:
            return []
        rest = math.sqrt(max(length * length - delta * delta, 0.0))
        out = []
        for sign in ((1.0,) if rest == 0.0 else (1.0, -1.0)):
            point = [0.0, 0.0]
            point[k] = end[k]
            point[other] = start[other] + sign * rest
            out.append((tuple(point),))
        return out
    return "undetermined"


def _fk_arc(h, f, element, start, tangent, end) -> list | str:
    keys = f["keys"]
    direction = f.get("dr")
    radius = _value(h, keys, "R")
    radius = h.length_si(abs(radius)) if radius is not None else None
    centre = None
    a, b, _ = PLANES[h.plane()]
    if "CCX" in keys or "CCY" in keys or "ICCX" in keys or "ICCY" in keys:
        cx = _centre_coord(h, keys, "CCX", "ICCX", start[0], a)
        cy = _centre_coord(h, keys, "CCY", "ICCY", start[1], b)
        if cx is None or cy is None:
            return "undetermined"
        centre = (cx, cy)
    if "CCPR" in keys and "CCPA" in keys:
        pole = h.s.sys.get("fk_pole")
        if pole is None:
            raise Alarm("pole_missing", "FPOL not defined")
        r = h.length_si(h.number(keys["CCPR"]))
        angle = math.radians(h.number(keys["CCPA"]))
        centre = (pole[0] + r * math.cos(angle), pole[1] + r * math.sin(angle))
    tangential = element == "FCT"
    if tangential and tangent is None:
        raise Alarm("fk_undetermined", "FCT without a preceding element")
    if "polar" in end:
        return "undetermined"
    full = 0 in end and 1 in end
    if centre is None and tangential and radius is not None and direction is not None:
        centre = _add(start, _left(_unit(tangent)), radius * direction)
    if centre is None and tangential and full:
        found = tangent_arc_centre(start, (end[0], end[1]), tangent)
        if found is None:
            return []
        centre, implied = found
        if direction is not None and implied != direction:
            return []
        direction = implied
        if radius is not None and abs(math.hypot(start[0] - centre[0], start[1] - centre[1]) - radius) > 1e-9:
            return []
    if direction is None:
        return "undetermined"
    if centre is not None:
        r = math.hypot(start[0] - centre[0], start[1] - centre[1])
        if radius is not None and abs(r - radius) > 1e-9:
            return []
        if tangential:
            t_here = arc_end_tangent(centre, start, direction)
            tu = _unit(tangent)
            if abs(t_here[0] * tu[1] - t_here[1] * tu[0]) > 1e-9 or t_here[0] * tu[0] + t_here[1] * tu[1] < 0.0:
                return []
        if full:
            point = (end[0], end[1])
            if abs(math.hypot(point[0] - centre[0], point[1] - centre[1]) - r) > float(
                    h.p["arc_end_point_tolerance_m"]):
                return []
            return [(point, centre, direction)]
        if 0 in end or 1 in end:
            k = 0 if 0 in end else 1
            other = 1 - k
            delta = end[k] - centre[k]
            if abs(delta) > r + 1e-12:
                return []
            rest = math.sqrt(max(r * r - delta * delta, 0.0))
            out = []
            for sign in ((1.0,) if rest == 0.0 else (1.0, -1.0)):
                point = [0.0, 0.0]
                point[k] = end[k]
                point[other] = centre[other] + sign * rest
                out.append((tuple(point), centre, direction))
            return out
        return "undetermined"
    if full and radius is not None:
        out = []
        for sign in (1.0, -1.0):
            try:
                c = h.radius_centre(start, (end[0], end[1]), sign * radius, direction)
            except Alarm:
                return []
            out.append(((end[0], end[1]), c, direction))
        return out
    return "undetermined"


def _centre_coord(h, keys, absolute: str, incremental: str, base: float, letter: str):
    if absolute in keys:
        return h.axis_si(letter, h.number(keys[absolute]))
    if incremental in keys:
        return base + h.axis_si(letter, h.number(keys[incremental]))
    return None


__all__ = ["approach", "departure", "fk_pole", "fk_element", "next_start_tangent", "eval_point", "plane_counts"]
