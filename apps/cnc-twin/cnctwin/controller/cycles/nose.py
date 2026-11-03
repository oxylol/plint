"""Tool nose radius compensation of cycle paths computed by the cycle itself (FANUC T series: G90/G94
with start-up from the cycle start point and cancel on the return, 4.1.5; G70-G73: the target figure
compensated with the vector 0 at the start point A and start-up in the block A -> A', 4.2.1).

The same corner rules as the block-stream filter (cnctwin.controller.comp.geometry): start-up type A
or B (parameter 5003), inner corners at the intersection, outer corners by the connection type
(19607#2), the commanded point being the imaginary tool nose (nose centre + r x tip direction). A
compensated element running against its programmed direction (or an arc whose angle changes by 180
deg or more) is an interference: PS0041.
"""

from __future__ import annotations

from ..alarms import ControllerAlarm
from ..comp import geometry as G
from . import common


def active(ex) -> bool:
    return ex.s.comp_state is not None and ex.s.comp_kind == "nose" and ex.s.modal.get("7", "G40") in ("G41", "G42")


def radius_tip(ex) -> tuple[float, int]:
    r = float(ex.handler.comp_radius())
    row = ex.s.tool_offsets.get(str(ex.s.offset_number))
    return r, int(row[3]) if row else 0


def _types(ex) -> tuple[str, str]:
    p = common.params(ex)
    t = str(p["crc_startup_type"]).upper()
    return ("B" if t in ("B", "CB") else "A"), str(p["crc_outer_corner"])


def nose_side(ex, a: str, b: str, points: list[dict]) -> float:
    """Signed offset (+ left, - right) putting the nose on the side of the start point (away from the
    material) of the first cutting element points[1] -> points[2]."""
    r, _ = radius_tip(ex)
    p1, p2, start = points[1], points[2], points[0]
    t = (p2[a] - p1[a], p2[b] - p1[b])
    to_start = (start[a] - p1[a], start[b] - p1[b])
    side = 1.0 if (-t[1] * to_start[0] + t[0] * to_start[1]) >= 0.0 else -1.0
    return side * abs(r)


def modal_side(ex) -> float:
    r, _ = radius_tip(ex)
    return (1.0 if ex.s.modal.get("7") == "G41" else -1.0) * r


def compensate(ex, elements: list[G.Seg], d: float, tip: int, cancel: bool) -> list[tuple[int, G.Seg]]:
    """Compensated elements (index of the programmed element each belongs to, element). elements[0] is the
    start-up element from the uncompensated start; with ``cancel`` the last element returns to its
    programmed end."""
    moving, outer = _types(ex)
    shift = G.tip_vector(tip, d)
    n = len(elements)
    out: list[tuple[int, G.Seg]] = []
    cur = elements[0].p0
    for i, el in enumerate(elements):
        if cancel and i == n - 1:
            out.append((i, G.Seg("line", cur, el.p1)))
            cur = el.p1
            continue
        nxt = elements[i + 1] if i + 1 < n else None
        if nxt is None:
            pts = [G.offset_point(el, d, True)]
        elif cancel and i + 1 == n - 1:
            pts = _cancel(el, nxt, d, moving, outer)
        elif i == 0:
            pts = _startup(el, nxt, d, moving, outer)
        else:
            pts = G.corner(el, nxt, d, outer).points
        pts = [G.add(q, shift) for q in pts]
        if i == 0:
            seg = G.Seg("line", cur, pts[0])
        else:
            moved = G.Seg(el.kind, G.add(el.p0, shift), G.add(el.p1, shift), G.add(el.c, shift), el.sweep)
            seg = G.compensated(moved, cur, pts[0], d)
            prog0, prog1 = G.add(el.p0, shift), G.add(el.p1, shift)
            if G.direction_interferes(prog0, prog1, seg.p0, seg.p1) or G.arc_interferes(moved, seg.p0, seg.p1):
                raise ControllerAlarm("crc_interference", "the tool nose cannot follow the cycle contour")
        out.append((i, seg))
        cur = pts[0]
        for q in pts[1:]:
            out.append((min(i + 1, n - 1), G.Seg("line", cur, q)))
            cur = q
    return out


def _startup(a: G.Seg, b: G.Seg, d: float, moving: str, outer: str) -> list:
    phi = G.turn(a, b) * (1.0 if d > 0.0 else -1.0)
    if phi > 0.0 or abs(phi) < G.TANGENT_RAD:
        return G.corner(a, b, d, outer).points
    if moving == "A":
        return [G.offset_point(b, d, at_end=False)]
    return [G.offset_point(a, d, at_end=True)] + G.corner(a, b, d, outer).points


def _cancel(a: G.Seg, b: G.Seg, d: float, moving: str, outer: str) -> list:
    phi = G.turn(a, b) * (1.0 if d > 0.0 else -1.0)
    if phi > 0.0:
        return G.corner(a, b, d, outer).points
    if moving == "B":
        return G.corner(a, b, d, outer).points + [G.offset_point(b, d, at_end=False)]
    return [G.offset_point(a, d, at_end=True)]


def compensate_polyline(ex, a: str, b: str, points: list[dict], d: float, cancel: bool) -> list[tuple[int, dict]]:
    """Compensated moves of a polyline of plane points (dicts a, b): (index of the programmed move, end)."""
    _, tip = radius_tip(ex)
    elements = []
    for p0, p1 in zip(points, points[1:]):
        elements.append(G.Seg("line", (p0[a], p0[b]), (p1[a], p1[b])))
    out = []
    for index, seg in compensate(ex, elements, d, tip, cancel):
        out.append((index, {a: seg.p1[0], b: seg.p1[1]}))
    return out


__all__ = ["active", "radius_tip", "nose_side", "modal_side", "compensate", "compensate_polyline"]
