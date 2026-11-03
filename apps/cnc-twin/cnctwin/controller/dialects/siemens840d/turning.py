"""SINUMERIK 840D sl turning cycles: CYCLE95 stock removal, CYCLE97 thread cutting, CYCLE93 grooving.

    CYCLE95(NPP, MID, FALZ, FALX, FAL, FF1, FF2, FF3, VARI, DT, DAM, _VRT)
    CYCLE97(PIT, MPIT, SPL, FPL, DM1, DM2, APP, ROP, TDEP, FAL, IANG, NSP, NRC, NID, VARI, NUMT, _VRT)
    CYCLE93(SPD, SPL, WIDG, DIAG, STA1, ANG1, ANG2, RCO1, RCI1, RCO2, RCI2, FAL1, FAL2, IDEP, DTB, VARI, _VRT)

Geometry is done in the turning plane (Z, X) with X as a radius; diameters of the parameters (DM1,
DM2, SPD) and of the contour (DIAMON) are halved on entry.

CYCLE95: the contour is the subprogram NPP (G0/G1/G2/G3 with X/Z, I/K or CR=, RND=/CHR=/CHF= between
its elements), its first block being the start point. VARI (units digit) 1-4 roughing, 5-8 finishing,
9-12 complete; within each, longitudinal outside, face outside, longitudinal inside, face inside. The
problem is mapped to coordinates (u, v): u along the cut, in the contour's direction, v along the infeed
toward the material (outside: +X; inside: -X; face: +Z). The blank is the contour's envelope in v (the
region between the contour and its highest point v_top), as on the control, which takes no blank
definition. The machining contour is the contour shifted by FALZ/FALX away from the part plus its
equidistant offset by FAL (exact offsets of lines and arcs joined by arcs about convex corners and at the
intersection of concave ones). Roughing: levels from v_top down to the lowest point in equal infeeds
<= MID; on each level every interval of u where the machining contour lies below the level is cut at FF1
(FF2 for intervals that begin inside the contour, i.e. undercuts), then the contour is followed up to
the previous level, the tool lifts by _VRT in both axes and returns at rapid. DAM > 0 interrupts a cut
every DAM for the dwell DT (chip breaking). Finishing at FF3 along the contour, with the tool nose
radius compensation (G42 outside / G41 inside, toward the material) through the core's nose provider.

CYCLE97: per thread start (NUMT, phase NSP + k 360/NUMT) the roughing cuts NRC (VARI 1/2 constant infeed
(TDEP - FAL)/NRC; 3/4 constant chip cross-section, depth_k = (TDEP - FAL) sqrt(k / NRC)), one finishing cut
removing FAL and NID idle passes; each pass: rapid to the run-in start (SPL - APP), infeed, thread move
(feed per revolution = pitch, override off) to FPL + ROP with the taper DM1 -> DM2, retract by _VRT, rapid
back. IANG > 0 feeds in along one flank (the start shifted by depth tan(IANG)), IANG < 0 alternates the
flanks. VARI 1/3 outside, 2/4 inside.

CYCLE93: grooves with straight or slanted flanks (ANG1, ANG2) and radii (RCO/RCI > 0) or chamfers (< 0)
at the four corners; STA1 = 0 (grooves on a cone are TW0012). VARI 1-8: odd longitudinal (groove in a
cylinder, plunging along X), even transverse (in a face, plunging along Z); 1, 2, 5, 6 outside; 1-4
starting point at the left flank (smaller coordinate), 5-8 at the right. Roughing plunges across the
width in steps of at most the tool width (the control takes it from the second cutting edge D2; the
twin's tools have one edge, so the width is twice the tool's radius_offset), pecking by IDEP with a
retraction of _VRT, dwell DTB at the base; the allowances FAL1 (base) and FAL2 (flanks) are left; then
the finishing contour flank - base - flank.
"""

from __future__ import annotations

import math

from ...comp.geometry import Seg, chamfer, intersections, nearest, offset, round_corner, turn
from ...expr import round_half_away
from ...hooks import Step
from . import values as V
from .cycles import ACTION, ISO_COARSE_PITCH_MM, arg, num
from .kit import Alarm

ZX = "ZX"


# ====================================================================== contour of a subprogram

def read_contour(h, name: str) -> tuple[tuple[float, float], list[Seg]]:
    """(start point, elements) of the contour subprogram ``name`` in (Z, X radius) SI."""
    from .tables import G_FUNCTIONS

    index = h.find_program(str(name).upper())
    program = h.ex.store.programs[index]
    env = V.Env(h)
    motion = "G1"
    absolute = h.absolute()
    diameter = h.s.modal.get("diameter", "DIAMOF")
    point = None
    elements: list[Seg] = []
    corners: dict[int, dict] = {}
    for block in program.blocks:
        sb = block.extra
        if block.error is not None:
            raise Alarm(block.error[0], block.error[1])
        if sb.kind == "eof":
            break
        if sb.kind in ("empty",) or (sb.kind == "stmt" and sb.stmt[0] in ("proc", "def", "extern")):
            continue
        if sb.kind == "stmt":
            raise Alarm("cycle_contour", f"{name}: control structure in a contour")
        words = {}
        stop = False
        for item in sb.items:
            kind = item[0]
            if kind == "g":
                spec = G_FUNCTIONS.get(item[1])
                if spec is None:
                    raise Alarm("cycle_contour", f"{name}: {item[1]}")
                if spec[0] == 1:
                    motion = item[1]
                elif item[1] in ("G90", "G91"):
                    absolute = item[1] == "G90"
                elif spec[1] == "diameter":
                    diameter = item[1]
                elif spec[0] not in (10, 11, 6):
                    raise Alarm("cycle_contour", f"{name}: {item[1]} in a contour")
            elif kind in ("axis", "addr"):
                words[item[1]] = (item[2], env.value(item[3]))
            elif kind == "m":
                code = V._number(env.value(item[1]))
                if code in (17.0, 2.0, 30.0):
                    stop = True
                else:
                    raise Alarm("cycle_contour", f"{name}: M{code:g} in a contour")
            elif kind == "kw" and item[1] == "RET":
                stop = True
            elif kind == "assign":
                continue
            else:
                raise Alarm("cycle_contour", f"{name}: {kind} in a contour")
        target = _contour_point(h, words, point, absolute, diameter)
        if point is None:
            if target is None or "X" not in words or "Z" not in words:
                raise Alarm("cycle_start_point", f"{name}: the first block must give X and Z")
            point = target
        elif target is not None and (target[0] != point[0] or target[1] != point[1]):
            element = _contour_element(h, motion, point, target, words)
            elements.append(element)
            corner = {k: V._number(words[k][1]) for k in ("RND", "CHR", "CHF") if k in words}
            if corner:
                corners[len(elements) - 1] = corner
            point = target
        if stop:
            break
    if point is None or not elements:
        raise Alarm("cycle_contour", f"{name}: no contour elements")
    return elements[0].p0, _apply_corners(h, elements, corners)


def _contour_point(h, words, point, absolute: bool, diameter: str):
    if "X" not in words and "Z" not in words:
        return None
    z0, x0 = point if point is not None else (0.0, 0.0)
    out = []
    for letter, base in (("Z", z0), ("X", x0)):
        if letter not in words:
            out.append(base)
            continue
        mod, value = words[letter]
        is_abs = {"AC": True, "IC": False}.get(mod, absolute)
        si = h.length_si(V._number(value, letter))
        if letter == "X" and (diameter == "DIAMON" or (diameter == "DIAM90" and is_abs)):
            si /= 2.0
        out.append(si if is_abs else base + si)
    return out[0], out[1]


def _contour_element(h, motion: str, start, end, words) -> Seg:
    if motion in ("G0", "G1"):
        return Seg("line", start, end)
    if motion not in ("G2", "G3"):
        raise Alarm("cycle_contour", f"{motion} in a contour")
    direction = -1 if motion == "G2" else 1
    if "CR" in words:
        radius = h.length_si(V._number(words["CR"][1], "CR"))
        centre = h.radius_centre(start, end, radius, direction)
    elif "I" in words or "K" in words:
        cz = start[0] + h.length_si(V._number(words["K"][1], "K")) if "K" in words else start[0]
        cx = start[1] + h.length_si(V._number(words["I"][1], "I")) if "I" in words else start[1]
        centre = (cz, cx)
    else:
        raise Alarm("cycle_contour", "arc without centre or CR")
    sweep = _sweep(centre, start, end, direction)
    return Seg("arc", start, end, centre, sweep)


def _sweep(centre, start, end, direction: int) -> float:
    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(end[1] - centre[1], end[0] - centre[0])
    delta = (a1 - a0) % (2.0 * math.pi)
    if direction > 0:
        return delta if delta > 0.0 else 2.0 * math.pi
    delta = (a0 - a1) % (2.0 * math.pi)
    return -(delta if delta > 0.0 else 2.0 * math.pi)


def _apply_corners(h, elements: list[Seg], corners: dict) -> list[Seg]:
    out = list(elements)
    shift = 0
    for k in sorted(corners):
        i = k + shift
        if i + 1 >= len(out):
            raise Alarm("cycle_contour", "chamfer/rounding at the last contour element")
        request = corners[k]
        a, b = out[i], out[i + 1]
        if "RND" in request:
            a2, extra, b2 = round_corner(a, b, h.length_si(request["RND"]))
        elif "CHR" in request:
            a2, extra, b2 = chamfer(a, b, h.length_si(request["CHR"]))
        else:
            half = abs(turn(a, b)) / 2.0
            a2, extra, b2 = chamfer(a, b, h.length_si(request["CHF"]) / (2.0 * math.cos(half)))
        out[i:i + 2] = [a2, extra, b2]
        shift += 1
    return [seg for seg in out if seg.length() > 1e-12]


# ====================================================================== (u, v) mapping and offsets

class Frame:
    """(z, x) <-> (u, v): u along the cut (in the contour's direction), v toward the material."""

    def __init__(self, longitudinal: bool, su: float, sv: float):
        self.longitudinal = longitudinal
        self.su = su
        self.sv = sv

    def to_uv(self, p):
        z, x = p
        return (self.su * z, self.sv * x) if self.longitudinal else (self.su * x, self.sv * z)

    def to_zx(self, p):
        u, v = p
        return (u / self.su, v / self.sv) if self.longitudinal else (v / self.sv, u / self.su)

    def seg(self, seg: Seg) -> Seg:
        flip = self.su * self.sv if self.longitudinal else -self.su * self.sv
        return Seg(seg.kind, self.to_uv(seg.p0), self.to_uv(seg.p1), self.to_uv(seg.c) if seg.kind == "arc" else
                   (0.0, 0.0), seg.sweep * flip)


def offset_chain(chain: list[Seg], d: float) -> list[Seg]:
    """Equidistant of a chain of elements by d to their left: offset elements, convex joins by arcs about
    the corner point, concave joins at the intersection."""
    if d <= 0.0:
        return list(chain)
    parts: list[Seg] = []
    for seg in chain:
        moved = offset(seg, d)
        if moved is None:
            raise Alarm("cycle_finish_allowance", "allowance larger than a contour radius")
        parts.append(moved)
    out: list[Seg] = [parts[0]]
    for k in range(1, len(parts)):
        prev, cur = out[-1], parts[k]
        corner = chain[k].p0
        gap = math.hypot(prev.p1[0] - cur.p0[0], prev.p1[1] - cur.p0[1])
        if gap <= 1e-12:
            out.append(cur)
            continue
        phi = turn(chain[k - 1], chain[k])
        if phi < 0.0:
            sweep = phi
            out.append(Seg("arc", prev.p1, cur.p0, corner, sweep))
            out.append(cur)
        else:
            point = nearest(intersections(prev, cur), corner)
            if point is None:
                raise Alarm("cycle_contour_prep", "no intersection of the offset elements")
            out[-1] = _trim(prev, point, end=True)
            out.append(_trim(cur, point, end=False))
    return out


def _trim(seg: Seg, point, end: bool) -> Seg:
    if seg.kind == "line":
        return Seg("line", seg.p0, point) if end else Seg("line", point, seg.p1)
    if end:
        return Seg("arc", seg.p0, point, seg.c, _arc_sweep(seg.c, seg.p0, point, seg.sweep))
    return Seg("arc", point, seg.p1, seg.c, _arc_sweep(seg.c, point, seg.p1, seg.sweep))


def _arc_sweep(c, p0, p1, sign) -> float:
    a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    if sign > 0:
        return (a1 - a0) % (2.0 * math.pi) or 2.0 * math.pi
    return -((a0 - a1) % (2.0 * math.pi) or 2.0 * math.pi)


def shift_chain(chain: list[Seg], du: float, dv: float) -> list[Seg]:
    def mv(p):
        return (p[0] + du, p[1] + dv)
    return [Seg(s.kind, mv(s.p0), mv(s.p1), mv(s.c) if s.kind == "arc" else s.c, s.sweep) for s in chain]


def v_at(seg: Seg, u: float) -> list[float]:
    """v values of the element at abscissa u."""
    if seg.kind == "line":
        (u0, v0), (u1, v1) = seg.p0, seg.p1
        if abs(u1 - u0) <= 1e-15:
            return []
        t = (u - u0) / (u1 - u0)
        return [v0 + t * (v1 - v0)] if -1e-12 <= t <= 1.0 + 1e-12 else []
    r = seg.radius
    du = u - seg.c[0]
    if abs(du) > r:
        return []
    out = []
    for sign in (1.0, -1.0):
        v = seg.c[1] + sign * math.sqrt(max(r * r - du * du, 0.0))
        if _on_arc(seg, (u, v)):
            out.append(v)
    return out


def _on_arc(seg: Seg, p) -> bool:
    part = _arc_sweep(seg.c, seg.p0, p, seg.sweep)
    return abs(part) <= abs(seg.sweep) + 1e-9 or math.hypot(p[0] - seg.p0[0], p[1] - seg.p0[1]) < 1e-12


def level_crossings(chain: list[Seg], v: float) -> list[tuple[float, int, tuple]]:
    """(u, element index, point) where the chain crosses the level v."""
    out = []
    probe_line = Seg("line", (-1e3, v), (1e3, v))
    for k, seg in enumerate(chain):
        for p in intersections(probe_line, seg) if seg.kind == "arc" else _line_level(seg, v):
            if seg.kind == "arc" and not _on_arc(seg, p):
                continue
            out.append((p[0], k, p))
    out.sort(key=lambda t: t[0])
    return out


def _line_level(seg: Seg, v: float) -> list:
    (u0, v0), (u1, v1) = seg.p0, seg.p1
    if (v0 - v) * (v1 - v) > 0.0 or abs(v1 - v0) <= 1e-15:
        return []
    t = (v - v0) / (v1 - v0)
    return [(u0 + t * (u1 - u0), v)]


def chain_v(chain: list[Seg], u: float) -> float | None:
    """Lowest v of the chain at abscissa u (None outside)."""
    values = [v for seg in chain for v in v_at(seg, u)]
    return min(values) if values else None


# ====================================================================== step helpers (Z, X program SI)

def _rapid(frame: Frame, p) -> Step:
    z, x = frame.to_zx(p)
    return Step("rapid", {"Z": z, "X": x})


def _line(frame: Frame, p, feed: float) -> Step:
    z, x = frame.to_zx(p)
    return Step("linear", {"Z": z, "X": x}, feed=feed, feed_mode="per_rev")


def _element(frame: Frame, seg: Seg, feed: float, comp: bool = False) -> Step:
    z, x = frame.to_zx(seg.p1)
    if seg.kind == "line":
        return Step("linear", {"Z": z, "X": x}, feed=feed, feed_mode="per_rev", comp=comp)
    cz, cx = frame.to_zx(seg.c)
    flip = frame.su * frame.sv if frame.longitudinal else -frame.su * frame.sv
    direction = 1 if seg.sweep * flip > 0.0 else -1
    return Step("arc", {"Z": z, "X": x}, centre={"Z": cz, "X": cx}, direction=direction, plane=ZX, feed=feed,
                feed_mode="per_rev", comp=comp)


def _follow(chain: list[Seg], k: int, start, v_stop: float) -> list[Seg]:
    """The chain from point ``start`` on element k forward until it reaches the level v_stop."""
    out = []
    seg = chain[k]
    piece = _trim(seg, start, end=False) if seg.kind == "arc" else Seg("line", start, seg.p1)
    for current in [piece] + chain[k + 1:]:
        crossing = _first_level_point(current, v_stop)
        if crossing is not None:
            out.append(_trim(current, crossing, end=True) if current.kind == "arc" else Seg("line", current.p0, crossing))
            return [s for s in out if s.length() > 1e-12]
        out.append(current)
    return [s for s in out if s.length() > 1e-12]


def _first_level_point(seg: Seg, v: float):
    if seg.kind == "line":
        hits = _line_level(seg, v)
        return hits[0] if hits and math.hypot(hits[0][0] - seg.p0[0], hits[0][1] - seg.p0[1]) > 1e-12 else None
    hits = [p for p in intersections(Seg("line", (-1e3, v), (1e3, v)), seg) if _on_arc(seg, p)]
    hits = [p for p in hits if math.hypot(p[0] - seg.p0[0], p[1] - seg.p0[1]) > 1e-12]
    if not hits:
        return None
    return min(hits, key=lambda p: abs(_arc_sweep(seg.c, seg.p0, p, seg.sweep)))


# ====================================================================== CYCLE95

def cycle95(h, values) -> list:
    name = arg(values, 0, "")
    if not isinstance(name, str) or not name:
        raise Alarm("cycle_contour", "NPP: contour name missing")
    mid = h.length_si(abs(num(values, 1, "MID")))
    falz, falx, fal = (h.length_si(abs(num(values, k, n))) for k, n in ((2, "FALZ"), (3, "FALX"), (4, "FAL")))
    ff1, ff2, ff3 = (num(values, k, n) for k, n in ((5, "FF1"), (6, "FF2"), (7, "FF3")))
    vari = int(num(values, 8, "VARI"))
    dt = num(values, 9, "DT")
    dam = h.length_si(abs(num(values, 10, "DAM")))
    vrt = h.length_si(abs(num(values, 11, "_VRT"))) or float(h.p["cycle95_lift_m"])
    if not 1 <= vari <= 12:
        raise Alarm("cycle_machining_type", f"VARI={vari}")
    kind = (vari - 1) % 4
    stage = (vari - 1) // 4                 # 0 roughing, 1 finishing, 2 complete
    if stage in (0, 2) and mid <= 0.0:
        raise Alarm("cycle_infeed_depth", "MID")
    unit = h.feed_unit_m()
    feed1, feed2, feed3 = ff1 * unit, (ff2 or ff1) * unit, (ff3 or ff1) * unit
    if (stage in (0, 2) and feed1 <= 0.0) or (stage in (1, 2) and feed3 <= 0.0):
        raise Alarm("cycle_no_feed", "FF1/FF3")
    start, contour = read_contour(h, name)
    longitudinal = kind in (0, 2)
    inside = kind in (2, 3)
    end = contour[-1].p1
    if longitudinal:
        su = 1.0 if end[0] >= start[0] else -1.0
        sv = -1.0 if inside else 1.0
    else:
        su = 1.0 if end[1] >= start[1] else -1.0
        sv = 1.0
    frame = Frame(longitudinal, su, sv)
    chain = [frame.seg(s) for s in contour]
    if any(seg.p1[0] < seg.p0[0] - 1e-9 and seg.kind == "line" and abs(seg.p1[1] - seg.p0[1]) < 1e-12
           for seg in chain):
        raise Alarm("cycle_contour", "contour runs against the machining direction")
    du = -(falz if longitudinal else falx)
    dv = falx if longitudinal else falz
    material_left = _material_left(chain)
    offset_side = fal if material_left else -fal
    machining = shift_chain(offset_chain(chain, offset_side) if offset_side > 0 else
                            _offset_right(chain, -offset_side), du, dv)
    out: list[Step] = []
    ex = h.ex
    here = (ex.si("Z", ex.s.pos["Z"]), ex.si("X", ex.s.pos["X"]))
    here_uv = frame.to_uv(here)
    v_top = max(max(s.p0[1], s.p1[1]) for s in machining)
    u_start = machining[0].p0[0]
    entry_u = u_start - vrt
    if stage in (0, 2):
        out += _roughing(frame, machining, v_top, mid, vrt, entry_u, feed1, feed2, dt, dam)
    if stage in (1, 2):
        out += _finishing(h, frame, chain, inside, entry_u, v_top + vrt, feed3)
    final = frame.to_zx((entry_u, max(v_top + vrt, here_uv[1])))
    out.append(Step("rapid", {"Z": final[0], "X": final[1]}))
    return out


def _material_left(chain: list[Seg]) -> bool:
    """True when the material (+v side) is on the left of the contour direction."""
    seg = max(chain, key=lambda s: s.length())
    t = seg.tangent(False)
    return t[0] >= 0.0


def _offset_right(chain: list[Seg], d: float) -> list[Seg]:
    reversed_chain = [Seg(s.kind, s.p1, s.p0, s.c, -s.sweep) for s in reversed(chain)]
    moved = offset_chain(reversed_chain, d)
    return [Seg(s.kind, s.p1, s.p0, s.c, -s.sweep) for s in reversed(moved)]


def _roughing(frame: Frame, chain: list[Seg], v_top: float, mid: float, vrt: float, entry_u: float, feed1: float,
              feed2: float, dt: float, dam: float) -> list[Step]:
    v_bottom = min(min(s.p0[1], s.p1[1]) for s in chain)
    for seg in chain:
        if seg.kind == "arc":
            for angle in (-math.pi / 2.0,):
                p = (seg.c[0] + seg.radius * math.cos(angle), seg.c[1] + seg.radius * math.sin(angle))
                if _on_arc(seg, p):
                    v_bottom = min(v_bottom, p[1])
    total = v_top - v_bottom
    if total <= 1e-12:
        return []
    count = int(math.ceil(total / mid - 1e-9))
    out = [_rapid(frame, (entry_u, v_top + vrt))]
    previous = v_top
    u_end = chain[-1].p1[0]
    for k in range(1, count + 1):
        level = v_top - total * k / count
        crossings = level_crossings(chain, level)
        intervals = _intervals(chain, crossings, level, chain[0].p0[0], u_end)
        for ua, ub, k_end, point_end, undercut in intervals:
            start_u = entry_u if not undercut else ua
            out.append(_rapid(frame, (start_u, previous + vrt)))
            out.append(_rapid(frame, (start_u, level)) if not undercut else _line(frame, (ua, level), feed2))
            out += _cut(frame, start_u, ub, level, feed1, dt, dam)
            follow = _follow(chain, k_end, point_end, previous) if k_end is not None else []
            for seg in follow:
                out.append(_element(frame, seg, feed1))
            last = follow[-1].p1 if follow else (ub, level)
            out.append(_rapid(frame, (last[0] - vrt, last[1] + vrt)))
            out.append(_rapid(frame, (entry_u, last[1] + vrt)))
        previous = level
    return out


def _cut(frame: Frame, ua: float, ub: float, level: float, feed: float, dt: float, dam: float) -> list[Step]:
    out = []
    if dam > 0.0 and dt > 0.0:
        u = ua
        while ub - u > dam + 1e-12:
            u += dam
            out.append(_line(frame, (u, level), feed))
            out.append(Step("dwell", dwell_s=dt))
    out.append(_line(frame, (ub, level), feed))
    return out


def _intervals(chain, crossings, level, u_first, u_end):
    """Cutting intervals on one level: (u from, u to, element index at the end, end point, undercut)."""
    out = []
    tol = 1e-9
    first_v = chain_v(chain, u_first + 1e-9)
    inside = first_v is not None and first_v <= level + tol
    begin = u_first if inside else None
    for u, k, point in crossings:
        before = chain_v(chain, u - 1e-7)
        after = chain_v(chain, u + 1e-7)
        if after is not None and after <= level + tol and (before is None or before > level + tol):
            begin = u
        elif before is not None and before <= level + tol and (after is None or after > level + tol) and \
                begin is not None:
            out.append((begin, u, k, point, begin > u_first + 1e-9))
            begin = None
    if begin is not None:
        out.append((begin, u_end, None, None, begin > u_first + 1e-9))
    return out


def _finishing(h, frame: Frame, chain: list[Seg], inside: bool, entry_u: float, v_safe: float,
               feed: float) -> list[Step]:
    out = [_rapid(frame, (entry_u, v_safe)), _rapid(frame, (entry_u, chain[0].p0[1]))]
    comp = h.comp_radius() > 0.0
    if comp:
        out.append(Step("resume", provider=ACTION, data={"action": "nose", "code": "G41" if inside else "G42"}))
    out.append(Step("linear", dict(zip(("Z", "X"), frame.to_zx(chain[0].p0))), feed=feed, feed_mode="per_rev",
                    comp=comp))
    for seg in chain:
        out.append(_element(frame, seg, feed, comp=comp))
    if comp:
        out.append(Step("resume", provider=ACTION, data={"action": "nose", "code": "G40"}))
        last = chain[-1].p1
        out.append(Step("linear", dict(zip(("Z", "X"), frame.to_zx((last[0], last[1] + 1e-3)))), feed=feed,
                        feed_mode="per_rev", comp=True))
    return out


# ====================================================================== CYCLE97

def cycle97(h, values) -> list:
    pit = num(values, 0, "PIT")
    mpit = num(values, 1, "MPIT")
    if pit and mpit:
        raise Alarm("cycle_pitch", "PIT and MPIT")
    if pit:
        lead = h.length_si(abs(pit))
    elif mpit:
        size = int(round_half_away(abs(mpit)))
        if size not in ISO_COARSE_PITCH_MM:
            raise Alarm("cycle_pitch", f"MPIT={mpit:g}")
        lead = ISO_COARSE_PITCH_MM[size] * 1e-3
    else:
        raise Alarm("cycle_pitch", "no pitch")
    spl, fpl = h.length_si(num(values, 2, "SPL")), h.length_si(num(values, 3, "FPL"))
    dm1, dm2 = h.length_si(num(values, 4, "DM1")) / 2.0, h.length_si(num(values, 5, "DM2")) / 2.0
    app, rop = h.length_si(abs(num(values, 6, "APP"))), h.length_si(abs(num(values, 7, "ROP")))
    tdep, fal = h.length_si(abs(num(values, 8, "TDEP"))), h.length_si(abs(num(values, 9, "FAL")))
    iang = num(values, 10, "IANG")
    nsp = num(values, 11, "NSP")
    nrc = int(round_half_away(num(values, 12, "NRC")))
    nid = int(round_half_away(num(values, 13, "NID")))
    vari = int(num(values, 14, "VARI"))
    numt = int(round_half_away(num(values, 15, "NUMT"))) or 1
    vrt = h.length_si(abs(num(values, 16, "_VRT"))) or float(h.p["cycle95_lift_m"])
    if vari not in (1, 2, 3, 4):
        raise Alarm("cycle_machining_type", f"VARI={vari}")
    if nrc < 1 or tdep <= 0.0 or fal > tdep:
        raise Alarm("cycle_parameter", "NRC / TDEP / FAL", "NRC")
    if spl == fpl:
        raise Alarm("cycle_parameter", "SPL = FPL", "FPL")
    outside = vari in (1, 3)
    sign = -1.0 if outside else 1.0         # radius change into the material
    direction = 1.0 if fpl > spl else -1.0
    z_start, z_end = spl - direction * app, fpl + direction * rop
    slope = (dm2 - dm1) / (fpl - spl)
    depths = []
    rough = tdep - fal
    for k in range(1, nrc + 1):
        depths.append(rough * (k / nrc if vari in (1, 2) else math.sqrt(k / nrc)))
    if fal > 0.0:
        depths.append(tdep)
    depths += [tdep] * max(nid, 0)
    out = []
    for start_k in range(numt):
        angle = math.radians(nsp + start_k * 360.0 / numt)
        for k, depth in enumerate(depths):
            shift = 0.0
            if iang:
                flank = depth * math.tan(math.radians(abs(iang)))
                shift = flank if (iang > 0.0 or k % 2 == 0) else -flank
            z0 = z_start - direction * shift
            x_at = lambda z: dm1 + slope * (z - spl) + sign * depth
            safe = max(dm1, dm2) + vrt if outside else min(dm1, dm2) - vrt
            out.append(Step("rapid", {"Z": z0, "X": safe}))
            out.append(Step("rapid", {"Z": z0, "X": x_at(z0)}))
            out.append(Step("linear", {"Z": z_end - direction * shift, "X": x_at(z_end - direction * shift)},
                            feed=lead, feed_mode="per_rev", override=False,
                            modal={"thread": {"lead_m": lead, "start_angle_rad": angle}}))
            out.append(Step("rapid", {"X": safe}))
            out.append(Step("rapid", {"Z": z0}))
    return out


# ====================================================================== CYCLE93

def cycle93(h, values) -> list:
    spd = h.length_si(num(values, 0, "SPD")) / 2.0
    spl = h.length_si(num(values, 1, "SPL"))
    widg = h.length_si(abs(num(values, 2, "WIDG")))
    diag = h.length_si(abs(num(values, 3, "DIAG")))
    sta1 = num(values, 4, "STA1")
    ang1, ang2 = num(values, 5, "ANG1"), num(values, 6, "ANG2")
    rco1, rci1, rco2, rci2 = (h.length_si(num(values, k, n)) for k, n in ((7, "RCO1"), (8, "RCI1"), (9, "RCO2"),
                                                                           (10, "RCI2")))
    fal1, fal2 = h.length_si(abs(num(values, 11, "FAL1"))), h.length_si(abs(num(values, 12, "FAL2")))
    idep = h.length_si(abs(num(values, 13, "IDEP")))
    dtb = num(values, 14, "DTB")
    vari = int(num(values, 15, "VARI")) % 10
    vrt = h.length_si(abs(num(values, 16, "_VRT"))) or float(h.p["cycle95_lift_m"])
    if sta1:
        raise Alarm("unsupported_function", "CYCLE93 on a cone (STA1 not 0)")
    if not 1 <= vari <= 8:
        raise Alarm("cycle_recess_type", f"VARI={vari}")
    if widg <= 0.0 or diag <= 0.0:
        raise Alarm("cycle_parameter", "WIDG / DIAG", "WIDG")
    if not (0.0 <= ang1 < 90.0 and 0.0 <= ang2 < 90.0):
        raise Alarm("cycle_parameter", "ANG1 / ANG2", "ANG1")
    width_counts = h.s.sys.get("tool_width", {}).get(str(h.s.tool))
    if not width_counts:
        raise Alarm("cycle_tool_width", f"T{h.s.tool}: no width (radius_offset of the grooving tool)")
    tool_w = width_counts * h.machine.lin_increment_m
    if tool_w > widg + 1e-12:
        raise Alarm("cycle_tool_width", "tool wider than the groove")
    feed = h.s.feed_si if h.s.feed_set else 0.0
    if feed <= 0.0:
        raise Alarm("feed_zero", "CYCLE93 without F")
    longitudinal = vari % 2 == 1
    outside = vari in (1, 2, 5, 6)
    left = vari <= 4
    # local groove frame: w across the groove (from the starting flank toward the other), d into the material
    s_w = 1.0 if left else -1.0
    s_d = -1.0 if outside else 1.0

    def to_zx(w, d):
        if longitudinal:
            return (spl + s_w * w, spd + s_d * d)
        return (spl + s_d * d, spd + s_w * w)
    t1, t2 = math.tan(math.radians(ang1)), math.tan(math.radians(ang2))
    bottom1, bottom2 = diag * t1, widg - diag * t2
    if bottom2 - bottom1 < tool_w - 1e-12:
        raise Alarm("cycle_tool_width", "groove base narrower than the tool")
    out = []
    safe_d = -vrt

    def point(w, d):
        z, x = to_zx(w, d)
        return {"Z": z, "X": x}

    rough_lo = bottom1 + fal2
    rough_hi = bottom2 - fal2 - tool_w
    positions = []
    span = rough_hi - rough_lo
    if span < -1e-12:
        raise Alarm("cycle_finish_allowance", "FAL2 leaves no room for the tool")
    count = max(1, int(math.ceil(max(span, 0.0) / tool_w - 1e-9)) + 1) if span > 0 else 1
    for k in range(count):
        positions.append(rough_lo + (span * k / (count - 1) if count > 1 else 0.0))
    depth_rough = diag - fal1
    for w in positions:
        out.append(Step("rapid", point(w, safe_d)))
        level = 0.0
        while level < depth_rough - 1e-12:
            level = min(depth_rough, level + (idep if idep > 0.0 else depth_rough))
            out.append(Step("linear", point(w, level), feed=feed))
            if level < depth_rough - 1e-12:
                out.append(Step("rapid", point(w, level - vrt)))
        if dtb > 0.0:
            out.append(Step("dwell", dwell_s=dtb))
        elif dtb < 0.0:
            out.append(Step("dwell", dwell_rev=-dtb))
        out.append(Step("rapid", point(w, safe_d)))
    # finishing contour (tool reference at its first corner): flank 1 down, base, flank 2 up
    flank1 = [Seg("line", (0.0, 0.0), (bottom1, diag))]
    base = Seg("line", (bottom1, diag), (bottom2 - tool_w, diag))
    flank2 = Seg("line", (bottom2 - tool_w, diag), (widg - tool_w, 0.0))
    chain = flank1 + [base, flank2]
    chain = _groove_corners(chain, rco1, rci1, rci2, rco2)
    out.append(Step("rapid", point(-vrt, safe_d)))
    out.append(Step("rapid", point(chain[0].p0[0], safe_d)))
    for seg in chain:
        if seg.kind == "line":
            out.append(Step("linear", point(*seg.p1), feed=feed))
        else:
            cz, cx = to_zx(*seg.c)
            direction = 1 if seg.sweep > 0 else -1
            mirror = s_w * s_d if longitudinal else -s_w * s_d
            out.append(Step("arc", point(*seg.p1), centre={"Z": cz, "X": cx}, direction=direction * int(mirror),
                            plane=ZX, feed=feed))
    out.append(Step("rapid", point(chain[-1].p1[0], safe_d)))
    return out


def _groove_corners(chain: list[Seg], rco1: float, rci1: float, rci2: float, rco2: float) -> list[Seg]:
    """Radii (> 0) or chamfers (< 0) at the inner (base) corners; the outer ones round the entry edges."""
    out = list(chain)
    for k, value in ((1, rci2), (0, rci1)):
        if value == 0.0:
            continue
        a, b = out[k], out[k + 1]
        if value > 0.0:
            a2, extra, b2 = round_corner(a, b, value)
        else:
            a2, extra, b2 = chamfer(a, b, -value)
        out[k:k + 2] = [a2, extra, b2]
    if rco1 != 0.0:
        lead = Seg("line", (-abs(rco1) * 4.0, 0.0), out[0].p0)
        a2, extra, b2 = round_corner(lead, out[0], rco1) if rco1 > 0.0 else chamfer(lead, out[0], -rco1)
        out = [Seg("line", a2.p1, a2.p1), extra, b2] + out[1:]
    if rco2 != 0.0:
        tail_end = out[-1].p1
        trail = Seg("line", tail_end, (tail_end[0] + abs(rco2) * 4.0, tail_end[1]))
        a2, extra, b2 = round_corner(out[-1], trail, rco2) if rco2 > 0.0 else chamfer(out[-1], trail, -rco2)
        out = out[:-1] + [a2, extra]
    return [seg for seg in out if seg.length() > 1e-12]


def run(h, name: str, values: list) -> list:
    if not h.T:
        raise Alarm("cycle_axes", f"{name} needs a lathe (turning plane G18)")
    if h.plane() != "ZX":
        raise Alarm("cycle_axes", f"{name} needs the turning plane G18")
    return {"CYCLE95": cycle95, "CYCLE97": cycle97, "CYCLE93": cycle93}[name](h, values)


__all__ = ["run", "read_contour", "offset_chain", "cycle95", "cycle97", "cycle93"]
