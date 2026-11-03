"""Revolved envelopes of rotating tools and thread tooth profiles (docs/models/tooling.md
"Envelope").

The envelope continues the cutting head's outline (cutting pieces flagged) with the fluted body
beyond the cutting length, the neck, the shank up to the holder nose, the holder outline from its
profile (z from the nose toward the spindle, diameter) and, when the machine gives it, the
spindle nose housing; it closes on the axis at the top. Convention and layout: cnctwin.tooling.abi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.errors import InputError
from .abi import (ENV_ARC, ENV_FRUSTUM, PART_BODY, PART_EDGE, PART_HOLDER, PART_NECK, PART_SHANK, PART_SPINDLE,
                  Envelope, ThreadProfile)
from .built import Built
from .profile import ARC, CURVE, Seg
from .spec import HolderSpec, ToolSpec, num, plain, sub

CURVE_CHORDS = 16       # frusta per parametric piece (drill lips): sagitta error < 1e-4 of the lip length


@dataclass
class SpindleNose:
    """The rotating spindle nose / housing above the gauge line as a cylinder (from the machine)."""

    radius_m: float
    length_m: float


def _piece_rows(seg: Seg, part: int, cutting: int) -> list[tuple]:
    if seg.shape == ARC:
        z0, r0 = seg.start
        z1, r1 = seg.end
        zs, rs = seg.point(np.linspace(0.0, 1.0, 65))
        return [(ENV_ARC, part, cutting, z0, r0, z1, r1, seg.centre[0], seg.centre[1], seg.rho, seg.a0, seg.a1,
                 float(zs.min()), float(zs.max()), float(rs.max()))]
    if seg.shape == CURVE:
        u = np.linspace(0.0, 1.0, CURVE_CHORDS + 1)
        z, r = seg.point(u)
        rows = []
        for k in range(CURVE_CHORDS):
            rows.append(_frustum_row((float(z[k]), float(r[k])), (float(z[k + 1]), float(r[k + 1])), part, cutting))
        return rows
    return [_frustum_row(seg.start, seg.end, part, cutting)]


def _frustum_row(p0, p1, part, cutting) -> tuple:
    return (ENV_FRUSTUM, part, cutting, p0[0], p0[1], p1[0], p1[1], 0.0, 0.0, 0.0, 0.0, 0.0,
            min(p0[0], p1[0]), max(p0[0], p1[0]), max(p0[1], p1[1]))


def _segment(rows, p0, p1, part):
    if abs(p0[0] - p1[0]) < 1e-15 and abs(p0[1] - p1[1]) < 1e-15:
        return
    rows.append(_frustum_row(p0, p1, part, 0))


def build_envelope(spec: ToolSpec, built: Built, holder: HolderSpec | None, stickout_m: float | None,
                   spindle: SpindleNose | None = None) -> tuple[Envelope, dict]:
    """Envelope of a rotating tool and the axial layout (z of flute end, neck end, holder nose,
    gauge line) the property models use.

    Chain 0 is the tool (cutting head, fluted body, neck, shank to the overall length), chain 1 the
    holder (nose face at the stickout, profile to the gauge line), chain 2 the spindle nose."""
    rows: list[tuple] = []
    chains: list[int] = []

    def add(row_list, chain):
        rows.extend(row_list)
        chains.extend([chain] * len(row_list))

    head = []
    for seg in built.outline:
        cutting = 1 if seg.kind is not None else 0
        part = PART_EDGE if cutting else (seg.part or PART_BODY)
        head.extend(_piece_rows(seg, part, cutting))
    if not head:
        raise InputError(f"tool {spec.tool_id}: no outline")
    first = (head[0][3], head[0][4])
    tool: list[tuple] = []
    if first[1] > 1e-15:
        _segment(tool, (first[0], 0.0), first, PART_BODY)
    tool.extend(head)
    info = built.info
    R = info.get("R", head[-1][6])
    geometry = spec.geometry
    point = (head[-1][5], head[-1][6])
    z_head = max(r[13] for r in head)
    flute_len = num(geometry, "flute_length_m")
    overall = num(geometry, "overall_length_m")
    shank = sub(geometry, "shank")
    neck = sub(geometry, "neck")
    d_shank = num(shank, "diameter_m") if shank is not None else None
    if d_shank is None:
        d_shank = 2.0 * R if not info.get("disc") else info.get("neck_d", R)
    d_neck = num(neck, "diameter_m") if neck is not None else None
    neck_end = num(neck, "length_m") if neck is not None else None
    if overall is None:
        overall = max(z_head, neck_end or 0.0) + (num(shank, "length_m", 0.0) if shank is not None else 0.0)
        if overall <= z_head:
            overall = z_head + 3.0 * d_shank
            spec.notes.append(f"overall length not given: {overall * 1e3:.1f} mm assumed for the envelope")
    layout = {"z_head": z_head, "overall": overall}
    if flute_len is not None and flute_len > point[0] + 1e-12 and not info.get("disc") and not info.get("indexable"):
        _segment(tool, point, (point[0], R), PART_BODY)
        _segment(tool, (point[0], R), (flute_len, R), PART_BODY)
        point = (flute_len, R)
    layout["z_flute_end"] = point[0]
    if d_neck is not None:
        z_end = neck_end if neck_end is not None and neck_end > point[0] else point[0]
        if neck_end is not None and neck_end <= point[0]:
            spec.notes.append("neck length (reach from the tip) does not exceed the head: no exposed neck")
        _segment(tool, point, (point[0], 0.5 * d_neck), PART_NECK)
        point = (point[0], 0.5 * d_neck)
        if z_end > point[0]:
            _segment(tool, point, (z_end, 0.5 * d_neck), PART_NECK)
            point = (z_end, 0.5 * d_neck)
        layout["z_neck_end"] = z_end
    _segment(tool, point, (point[0], 0.5 * d_shank), PART_SHANK)
    point = (point[0], 0.5 * d_shank)
    z_top = max(overall, point[0])
    _segment(tool, point, (z_top, point[1]), PART_SHANK)
    _segment(tool, (z_top, point[1]), (z_top, 0.0), PART_SHANK)
    add(tool, 0)
    layout["z_gauge"] = None
    if stickout_m is not None:
        layout["stickout"] = stickout_m
    if holder is not None:
        stick = stickout_m if stickout_m is not None else _default_stickout(spec, holder, overall, shank)
        layout["stickout"] = stick
        if stick < layout.get("z_neck_end", layout["z_flute_end"]) - 1e-12:
            spec.notes.append(f"stickout {stick * 1e3:.2f} mm is shorter than the fluted part / neck: the holder "
                              f"covers part of it")
        chain = []
        nose_r = 0.5 * holder.profile_m[0][1]
        p = (stick + holder.profile_m[0][0], 0.0)
        _segment(chain, p, (p[0], nose_r), PART_HOLDER)
        p = (p[0], nose_r)
        for z_row, d_row in holder.profile_m[1:]:
            target = (stick + z_row, 0.5 * d_row)
            if target[0] < p[0] - 1e-12:
                raise InputError(f"holder {holder.holder_id}: profile_mm z must not decrease")
            _segment(chain, p, target, PART_HOLDER)
            p = target
        z_gauge = stick + holder.gauge_length_m
        if p[0] < z_gauge - 1e-12:
            _segment(chain, p, (z_gauge, p[1]), PART_HOLDER)
            p = (z_gauge, p[1])
        _segment(chain, p, (p[0], 0.0), PART_HOLDER)
        add(chain, 1)
        layout["z_holder_nose"] = stick
        layout["z_gauge"] = z_gauge
        if spindle is not None and spindle.radius_m > 0.0 and spindle.length_m > 0.0:
            sp = []
            _segment(sp, (z_gauge, 0.0), (z_gauge, spindle.radius_m), PART_SPINDLE)
            _segment(sp, (z_gauge, spindle.radius_m), (z_gauge + spindle.length_m, spindle.radius_m), PART_SPINDLE)
            _segment(sp, (z_gauge + spindle.length_m, spindle.radius_m), (z_gauge + spindle.length_m, 0.0),
                     PART_SPINDLE)
            add(sp, 2)
    columns = list(zip(*rows))
    env = Envelope(
        kind=np.ascontiguousarray(np.array(columns[0], np.int32)),
        part=np.ascontiguousarray(np.array(columns[1], np.int32)),
        cutting=np.ascontiguousarray(np.array(columns[2], np.int8)),
        **{name: np.ascontiguousarray(np.array(columns[i], dtype=float)) for i, name in
           enumerate(("z0_m", "r0_m", "z1_m", "r1_m", "zc_m", "rc_m", "rho_m", "a0_rad", "a1_rad", "z_lo_m",
                      "z_hi_m", "r_max_m"), start=3)},
        chain=np.ascontiguousarray(np.array(chains, np.int32)))
    layout["d_shank"] = d_shank
    layout["d_neck"] = d_neck
    return env, layout


def _default_stickout(spec, holder: HolderSpec, overall: float, shank) -> float:
    if holder.clamp_length_m is not None:
        value = overall - holder.clamp_length_m
        spec.notes.append(f"stickout not given: overall length - holder clamp length = {value * 1e3:.2f} mm")
        return value
    shank_len = num(shank, "length_m") if shank is not None else None
    if shank_len is not None:
        value = overall - shank_len
        spec.notes.append(f"stickout not given: overall length - shank length = {value * 1e3:.2f} mm")
        return value
    raise InputError(f"tool {spec.tool_id}: stickout_mm is needed (no holder clamp length or shank length)")


# --------------------------------------------------------------------------- thread profiles
def _clip(poly: list[tuple[float, float]], inside, cross) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clipping of a polygon against one half-plane."""
    out: list[tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = inside(a), inside(b)
        if ina:
            out.append(a)
        if ina != inb:
            out.append(cross(a, b))
    return out


def _half_plane(z0, r0, z1, r1):
    """Region below the line through (z0, r0)-(z1, r1) in (z, r): r <= line(z)."""
    slope = (r1 - r0) / (z1 - z0) if z1 != z0 else 0.0

    def line_r(z):
        return r0 + slope * (z - z0)

    def inside(p):
        return p[1] <= line_r(p[0]) + 1e-15

    def cross(a, b):
        fa = a[1] - line_r(a[0])
        fb = b[1] - line_r(b[0])
        t = fa / (fa - fb)
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    return inside, cross


def _z_limit(z_lim, keep_above: bool):
    def inside(p):
        return p[0] >= z_lim - 1e-15 if keep_above else p[0] <= z_lim + 1e-15

    def cross(a, b):
        t = (z_lim - a[0]) / (b[0] - a[0])
        return (z_lim, a[1] + t * (b[1] - a[1]))
    return inside, cross


def thread_profile(built: Built, phases: np.ndarray) -> ThreadProfile | None:
    """Tooth polygons of taps (lead shift per land, chamfer truncation) and thread mills (annular
    rows); None for other tools."""
    info = built.info
    form = info.get("thread")
    if form is None or not built.rotating:
        return None
    R = info["R"]
    pitch, c, h = form["pitch"], form["crest"], form["depth"]
    run = h * math.tan(0.5 * form["form"])
    tap = bool(info.get("tap"))
    nt = len(built.teeth)
    if tap:
        z_first, n_rows = info["z_first"], info["rows"]
        lead, L_ch, L, r_tip = info["lead"], info["chamfer_length"], info["cutting_length"], info["r_tip"]
        tan_b = math.tan(info["helix"][0]) if info.get("helix") else 0.0
    else:
        n_rows = info["rows"]
        z_first = 0.5 * c + run
        lead = 0.0
    n_max = 10
    pts = np.zeros((nt, n_rows, n_max, 2))
    count = np.zeros((nt, n_rows), np.int32)
    row_z = np.zeros((nt, n_rows))
    for j in range(nt):
        trailing = -_unwrapped(phases[j])
        for m in range(n_rows):
            zc = z_first + m * pitch
            if tap:
                zc = (zc + lead * trailing / (2.0 * math.pi)) / (1.0 - lead * tan_b / (2.0 * math.pi * R))
            poly = [(zc - 0.5 * c - run, R - h), (zc + 0.5 * c + run, R - h), (zc + 0.5 * c, R), (zc - 0.5 * c, R)]
            if tap:
                poly = _clip(poly, *_half_plane(0.0, r_tip, L_ch, R))
                poly = _clip(poly, *_z_limit(0.0, True)) if poly else poly
                poly = _clip(poly, *_z_limit(L, False)) if poly else poly
            row_z[j, m] = zc
            if len(poly) >= 3:
                arr = np.array(poly)
                area = _area(arr)
                if abs(area) <= 1e-6 * (pitch * h):      # clipped away (tooth beyond the tip or the thread length)
                    continue
                if area < 0.0:
                    arr = arr[::-1]
                count[j, m] = len(arr)
                pts[j, m, :len(arr)] = arr
    return ThreadProfile(
        pts_zr_m=np.ascontiguousarray(pts), n_pts=count, pitch_m=np.array([pitch]), lead_m=np.array([lead]),
        starts=np.array([form["starts"]], np.int32), thread_hand=np.array([form["thread_hand"]]),
        form_angle_rad=np.array([form["form"]]), major_r_m=np.array([R]), depth_m=np.array([h]),
        row_z_m=np.ascontiguousarray(row_z), tooth_phase_rad=np.ascontiguousarray(np.asarray(phases, dtype=float)))


def _unwrapped(phase: float) -> float:
    """Phase in (-2 pi, 0]: the trailing angle of a land behind land 0."""
    value = math.fmod(phase, 2.0 * math.pi)
    if value > 0.0:
        value -= 2.0 * math.pi
    return value


def _area(arr: np.ndarray) -> float:
    return 0.5 * float(np.sum(arr[:, 0] * np.roll(arr[:, 1], -1) - np.roll(arr[:, 0], -1) * arr[:, 1]))


def spindle_nose_from_machine(machine, spindle_id: str = "main") -> SpindleNose | None:
    """The spindle housing around the nose from the machine's geometry: a cylinder primitive
    coaxial with the spindle-nose link of the tool chain that starts at the nose plane (machine
    file convention: geometry in the frame of the link it is attached to)."""
    if machine is None:
        return None
    kin = sub(machine, "kinematics")
    chain = kin.get("tool_chain") if kin is not None else None
    if not chain:
        return None
    nose_link, parent, offset = None, None, np.zeros(3)
    for i, link in enumerate(chain):
        name = plain(link.get("link"))
        if name == "spindle_nose" or i == len(chain) - 1:
            nose_link = name
            parent = plain(chain[i - 1].get("link")) if i > 0 else None
            fixed = link.get("fixed")
            translate = plain(fixed.get("translate_m")) if fixed is not None else None
            if translate is not None:
                offset = np.array([float(plain(v)) for v in translate])
            break
    axis = plain(kin.get("tool_axis")) or [0.0, 0.0, 1.0]
    axis = np.array([float(plain(v)) for v in axis])
    axis = axis / np.linalg.norm(axis)
    best = None
    for item in machine.get("geometry") or []:
        attached = plain(item.get("attached_to"))
        if attached not in (nose_link, parent):
            continue
        base = offset if attached == parent else np.zeros(3)
        for prim in item.get("primitives") or []:
            cyl = prim.get("cylinder")
            if cyl is None:
                continue
            p0 = np.array([float(plain(v)) for v in plain(cyl.get("p0_m"))]) - base
            p1 = np.array([float(plain(v)) for v in plain(cyl.get("p1_m"))]) - base
            radius = float(plain(cyl.get("radius_m")))
            d = p1 - p0
            length = float(np.linalg.norm(d))
            if length <= 0.0 or abs(abs(float(np.dot(d / length, axis))) - 1.0) > 1e-6:
                continue
            lateral = p0 - float(np.dot(p0, axis)) * axis
            if float(np.linalg.norm(lateral)) > 1e-6:
                continue
            start = min(float(np.dot(p0, axis)), float(np.dot(p1, axis)))
            if abs(start) > 1e-3:
                continue
            if best is None or radius > best.radius_m:
                best = SpindleNose(radius, length)
    return best
