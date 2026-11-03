"""Non-rotating tools: turning, grooving, parting and threading inserts in their holders, boring
bars and form tools (docs/models/tooling.md "Turning tools").

Frame T of a non-rotating tool: origin at the imaginary tool tip P (the tangents of the nose
circle along x_T and z_T); x_T points into the work material (toward the work axis for external
tools, away from it for internal ones); z_T points from the tip into the tool body along the work
axis, so the major cutting edge faces -z_T and leads when the tool feeds along -z_T (grooving
tools feed along +x_T); y_T = z_T x x_T. Elements sit in the reference plane (psi = 0): r is the
x_T coordinate (<= 0), z the z_T coordinate, n = sin(kappa) x_T - cos(kappa) z_T points into the
material, and the intended cutting velocity of the tool relative to the work is h * y_T
(h = hand sign; the rake face faces it). On the major edge kappa = 90 deg - kappa_r (kappa_r: ISO
entering angle from the feed direction); the nose spans kappa_major .. 90 deg + kappa_r'; the
minor edge has kappa = 90 deg + kappa_r', kappa_r' = 180 deg - kappa_r - epsilon_r.
The seat (holder rake and inclination) and the chip-breaker rake give every element its normal
rake, clearance and inclination from the rake plane (cnctwin.tooling.inserts).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.abi import EDGE_INSERT, EDGE_NOSE, EDGE_THREAD_FLANK, EDGE_WIPER
from ..core.errors import InputError
from . import params as P
from .abi import InsertProfile, ThreadProfile
from .built import Built
from .common import edge_prep
from .edges import Face, ToothDef
from .indexable import corner_profile
from .inserts import chip_breaker_rake, edge_length, face_planes, included_angle, insert_clearance, wiper_length
from .mills import thread_form
from .profile import arc, line
from .spec import ToolSpec, clearance_primary, flag, num, numlist, plain, sub, text

HALF_PI = 0.5 * math.pi
NOSE_STEP = math.radians(5.0)     # polygonisation of nose arcs in the 2-D insert outline


def _holder_seat(spec: ToolSpec, holder, alpha_i: float) -> tuple[float, float]:
    rake = num(holder, "rake_rad") if holder is not None else None
    incl = num(holder, "inclination_rad") if holder is not None else None
    negative = alpha_i <= 1e-9
    if rake is None:
        rake = P.pf("negative_holder_rake_rad") if negative else 0.0
        spec.notes.append(f"holder rake not given: {math.degrees(rake):.0f} deg")
    if incl is None:
        incl = P.pf("negative_holder_incl_rad") if negative else 0.0
        spec.notes.append(f"holder inclination not given: {math.degrees(incl):.0f} deg")
    return rake, incl


def _place_tip(segs, nose_tag: str = "nose") -> None:
    """Translate so the nose circle's tangents along x_T and z_T pass through the origin."""
    nose = next((s for s in segs if s.tag == nose_tag and s.shape == "arc"), None)
    if nose is not None:
        dz = -(nose.centre[0] - nose.rho)
        dr = -(nose.centre[1] + nose.rho)
    else:
        corner = segs[0].end
        dz, dr = -corner[0], -corner[1]
    for seg in segs:
        if seg.shape == "line":
            seg.p0 = (seg.p0[0] + dz, seg.p0[1] + dr)
            seg.p1 = (seg.p1[0] + dz, seg.p1[1] + dr)
        else:
            seg.centre = (seg.centre[0] + dz, seg.centre[1] + dr)
            seg.p0, seg.p1 = seg.start, seg.end


def _face(spec, N_r, cb, alpha_i):
    radius, width, angle = edge_prep(spec, insert=True)
    return Face(mode="insert", N_rake=tuple(N_r), rake=cb, clear=alpha_i, edge_radius=radius, chamfer_width=width,
                chamfer_angle=angle, lift=True)


def _one_tooth() -> list[ToothDef]:
    return [ToothDef(phase=0.0, params={"psi": 0.0})]


def _polyline(segs) -> list[tuple[float, float, int]]:
    """(z, x, cutting) points along the pieces, arcs polygonised by NOSE_STEP."""
    pts = []
    for seg in segs:
        if seg.shape == "arc":
            n = max(2, int(math.ceil(abs(seg.a1 - seg.a0) / NOSE_STEP)))
        else:
            n = 1
        u = np.linspace(0.0, 1.0, n + 1)
        z, r = seg.point(u)
        for k in range(n):
            pts.append((float(z[k]), float(r[k]), 1 if seg.kind is not None else 0))
    last = segs[-1].end
    pts.append((last[0], last[1], 0))
    return pts


def _insert_profile(edge_pts, back_pts, nose_seg) -> InsertProfile:
    """Closed outline: the cutting chain followed by the non-cutting back of the insert."""
    pts = [(z, x) for z, x, _ in edge_pts] + [(z, x) for z, x in back_pts]
    flags = [c for _, _, c in edge_pts[:-1]] + [0] * (len(back_pts) + 1)
    arr = np.ascontiguousarray(np.array(pts, dtype=float))
    # counter-clockwise in (z, x): signed area > 0
    area = 0.5 * float(np.sum(arr[:, 0] * np.roll(arr[:, 1], -1) - np.roll(arr[:, 0], -1) * arr[:, 1]))
    cut = np.array(flags[:len(arr)], dtype=np.int8)
    if area < 0.0:
        arr = np.ascontiguousarray(arr[::-1])
        cut = np.ascontiguousarray(np.roll(cut[::-1], -1))
    if nose_seg is not None:
        centre = np.array([nose_seg.centre[0], nose_seg.centre[1]])
        rho = nose_seg.rho
    else:
        centre = np.zeros(2)
        rho = 0.0
    return InsertProfile(arr, cut, centre, np.array([rho]))


# --------------------------------------------------------------------------- turning inserts and boring bars
def build_turning(spec: ToolSpec, settings) -> Built:
    insert = sub(spec.geometry, "insert")
    if insert is None:
        raise InputError(f"tool {spec.tool_id}: a {spec.type} needs geometry.insert")
    holder = sub(spec.geometry, "holder")
    nose = num(insert, "nose_radius_m", 0.0) or 0.0
    corner = included_angle(spec, insert)
    kappa_r = num(holder, "approach_angle_rad") if holder is not None else None
    if kappa_r is None:
        if spec.type == "boring_bar":
            kappa_r = P.pf("boring_bar_approach_rad")
            spec.notes.append(f"approach angle not given: {math.degrees(kappa_r):.0f} deg")
        else:
            raise InputError(f"tool {spec.tool_id}: holder.approach_angle_deg is required")
    round_insert = corner is None
    if round_insert:
        ic = num(insert, "ic_m")
        if ic is None:
            raise InputError(f"tool {spec.tool_id}: a round insert needs ic_mm")
        nose = 0.5 * ic
        # the active arc runs from the front (kappa 0, depth of cut up to the radius) to the back
        # (kappa pi, back-profiling): the half circle facing the work
        kappa_r = HALF_PI
        corner = 0.0
        spec.notes.append("round insert: active edge is the half circle facing the work (ap <= ic/2)")
    minor_angle = math.pi - kappa_r - corner
    if not round_insert and minor_angle < -1e-9:
        raise InputError(f"tool {spec.tool_id}: approach angle {math.degrees(kappa_r):.1f} deg and corner angle "
                         f"{math.degrees(corner):.1f} deg leave no minor-edge clearance")
    ell = edge_length(spec, insert, None if round_insert else corner)
    alpha_i = insert_clearance(spec, insert)
    cb = chip_breaker_rake(insert)
    seat_rake, incl = _holder_seat(spec, holder, alpha_i)
    kappa_major = 0.0 if round_insert else HALF_PI - kappa_r
    kappa_minor = math.pi if round_insert else HALF_PI + minor_angle
    N_r, _ = face_planes(kappa_major, seat_rake, incl, alpha_i - seat_rake)
    face = _face(spec, N_r, cb, alpha_i)
    wiper = flag(insert, "wiper", False)
    if round_insert:
        segs = [arc((nose, -nose), nose, kappa_major, kappa_minor, EDGE_NOSE, rule="const", face=face, tag="nose",
                    extra={"psi": 0.0})]
    elif wiper:
        flat = wiper_length(spec, insert, ell, nose, turning=True)
        segs = corner_profile((0.0, 0.0), kappa_major, HALF_PI, nose, ell, 0.0,
                              kinds=(EDGE_INSERT, EDGE_NOSE, EDGE_INSERT), tags=("major", "nose", "minor"), face=face)
        end = segs[-1].end
        flat_end = (end[0] + flat, end[1])
        segs.append(line(end, flat_end, EDGE_WIPER, rule="const", face=face, tag="wiper", extra={"psi": 0.0}))
        d = (math.sin(kappa_minor), math.cos(kappa_minor))
        segs.append(line(flat_end, (flat_end[0] + d[0] * ell, flat_end[1] + d[1] * ell), EDGE_INSERT, rule="const",
                         face=face, tag="minor", extra={"psi": 0.0}))
    else:
        # the insert side length runs between theoretical corners; the nose takes r tan(turn / 2) of it
        reach = nose * math.tan(0.5 * (kappa_minor - kappa_major))
        side = max(ell - reach, 0.0)
        segs = corner_profile((0.0, 0.0), kappa_major, kappa_minor, nose, side, side,
                              kinds=(EDGE_INSERT, EDGE_NOSE, EDGE_INSERT), tags=("major", "nose", "minor"), face=face)
    _place_tip(segs)
    nose_seg = next((s for s in segs if s.tag == "nose"), None)
    edge_pts = _polyline(segs)
    back = _insert_back(spec, segs, corner, round_insert, insert)
    profile = _insert_profile(edge_pts, back, nose_seg)
    info = _turning_info(spec, segs, holder, kappa_r, corner, nose, seat_rake, incl, alpha_i, ell)
    return Built(segs, _one_tooth(), [], info, insert=profile, rotating=False)


def _insert_back(spec, segs, corner, round_insert, insert) -> list[tuple[float, float]]:
    """The rest of the insert polygon behind the active corner (parallelogram for rhombic and
    rectangular shapes, triangle for T/W, regular polygon for H/O/P, circle for R)."""
    start = np.array(segs[0].start)
    end = np.array(segs[-1].end)
    if round_insert:
        # the back half of the circle (a from pi to 2 pi), not cutting
        nose = segs[0]
        angles = np.linspace(nose.a1, nose.a0 + 2.0 * math.pi, max(2, int(math.ceil(math.pi / NOSE_STEP))) + 1)[1:-1]
        return [(nose.centre[0] - nose.rho * math.cos(a), nose.centre[1] + nose.rho * math.sin(a)) for a in angles]
    shape = (text(insert, "shape") or (text(insert, "iso_code") or " ")[0]).upper()
    if shape in ("T", "W"):
        return []
    if shape in ("H", "O", "P"):
        sides = {"H": 6, "O": 8, "P": 5}[shape]
        pts, a = [], end.copy()
        d = end - np.array(segs[-1].start)
        d /= np.linalg.norm(d)
        length = float(np.linalg.norm(end - np.array(segs[-1].start)))
        turn = 2.0 * math.pi / sides
        c, s = math.cos(turn), math.sin(turn)
        for _ in range(sides - 3):
            d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
            a = a + d * length
            pts.append((float(a[0]), float(a[1])))
        return pts
    corner_pt = _line_intersection(segs[0], segs[-1])
    fourth = start + (end - corner_pt)
    return [(float(fourth[0]), float(fourth[1]))]


def _line_intersection(a, b) -> np.ndarray:
    """Theoretical corner: intersection of the lines carrying pieces a and b (z, x)."""
    p, d = np.array(a.start), np.array(a.end) - np.array(a.start)
    q, e = np.array(b.start), np.array(b.end) - np.array(b.start)
    mat = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    if abs(np.linalg.det(mat)) < 1e-18:
        return np.array(a.end)
    t = np.linalg.solve(mat, q - p)
    return p + t[0] * d


def _turning_info(spec, segs, holder, kappa_r, corner, nose, seat_rake, incl, alpha_i, ell) -> dict:
    bar = sub(spec.geometry, "bar")
    shank = numlist(holder, "shank_m") if holder is not None else None
    info = {"kappa_r": kappa_r, "corner": corner, "nose": nose, "seat_rake": seat_rake, "incl": incl,
            "insert_clear": alpha_i, "edge_length": ell, "flutes": 1, "turning": True,
            "overhang": num(holder, "overhang_m") if holder is not None else None,
            "shank_hb": shank, "shank_dir_T": (-1.0, 0.0, 0.0)}
    if spec.type == "boring_bar":
        if bar is None:
            raise InputError(f"tool {spec.tool_id}: a boring bar needs geometry.bar")
        d = num(bar, "diameter_m")
        overhang = num(bar, "overhang_m")
        min_bore = num(spec.geometry, "min_bore_m")
        if min_bore is None:
            raise InputError(f"tool {spec.tool_id}: a boring bar needs min_bore_mm")
        if d >= min_bore:
            raise InputError(f"tool {spec.tool_id}: bar diameter must be smaller than the minimum bore")
        info.update({"bar_d": d, "overhang": overhang, "min_bore": min_bore, "bar_material": text(bar, "material"),
                     "bar_axis_x": -0.5 * min_bore, "shank_dir_T": (0.0, 0.0, 1.0)})
    return info


# --------------------------------------------------------------------------- grooving and parting
def build_grooving(spec: ToolSpec, settings) -> Built:
    insert = sub(spec.geometry, "insert")
    if insert is None:
        raise InputError(f"tool {spec.tool_id}: a {spec.type} needs geometry.insert")
    holder = sub(spec.geometry, "holder")
    w = num(insert, "width_m")
    if w is None or w <= 0.0:
        raise InputError(f"tool {spec.tool_id}: grooving and parting inserts need insert.width_mm")
    nose = num(insert, "nose_radius_m", 0.0) or 0.0
    if 2.0 * nose > w:
        raise InputError(f"tool {spec.tool_id}: corner radius larger than half the insert width")
    side = P.pf("groove_side_clearance_rad")
    depth = num(insert, "max_depth_m") or (num(holder, "max_depth_m") if holder is not None else None)
    if depth is None:
        depth = w
        spec.notes.append(f"groove depth not given: insert outline drawn to {depth * 1e3:.2f} mm")
    kappa_r = num(holder, "approach_angle_rad") if holder is not None else None
    front_k = math.pi - kappa_r if kappa_r is not None else HALF_PI
    alpha_i = insert_clearance(spec, insert)
    cb = chip_breaker_rake(insert)
    seat_rake, incl = _holder_seat(spec, holder, alpha_i)
    N_r, _ = face_planes(front_k, seat_rake, incl, alpha_i - seat_rake)
    face = _face(spec, N_r, cb, alpha_i)
    k_side_in = -side
    k_side_out = math.pi + side
    # left corner, front edge, right corner as two corner profiles sharing the front edge
    d_front = (math.sin(front_k), math.cos(front_k))
    left = corner_profile((0.0, 0.0), k_side_in, front_k, nose, depth, 0.0,
                          kinds=(EDGE_INSERT, EDGE_NOSE, EDGE_INSERT), tags=("side_a", "nose", "front"), face=face)
    p_right = (w, -w * math.tan(front_k - HALF_PI))      # the front line through (0, 0) at z = w
    right = corner_profile(p_right, front_k, k_side_out, nose, 0.0, depth,
                           kinds=(EDGE_INSERT, EDGE_NOSE, EDGE_INSERT), tags=("front", "nose_b", "side_b"), face=face)
    join0 = left[-1].end
    join1 = right[0].start
    segs = left + [line(join0, join1, EDGE_INSERT, rule="const", face=face, tag="front", extra={"psi": 0.0})] + right
    _place_tip(segs)
    edge_pts = _polyline(segs)
    profile = _insert_profile(edge_pts, [], next((s for s in segs if s.tag == "nose"), None))
    info = {"width": w, "nose": nose, "depth": depth, "seat_rake": seat_rake, "incl": incl, "insert_clear": alpha_i,
            "flutes": 1, "turning": True, "kappa_r": kappa_r if kappa_r is not None else HALF_PI,
            "overhang": num(holder, "overhang_m") if holder is not None else None,
            "shank_hb": numlist(holder, "shank_m") if holder is not None else None, "shank_dir_T": (-1.0, 0.0, 0.0)}
    return Built(segs, _one_tooth(), [], info, insert=profile, rotating=False)


# --------------------------------------------------------------------------- threading inserts
def build_threading(spec: ToolSpec, settings) -> Built:
    insert = sub(spec.geometry, "insert")
    holder = sub(spec.geometry, "holder")
    form = thread_form(spec)
    tip = num(insert, "nose_radius_m", 0.0) if insert is not None else 0.0
    tip = tip or 0.0
    half = 0.5 * form["form"]
    depth = form["depth"]
    alpha_i = insert_clearance(spec, insert) if insert is not None else clearance_primary(spec.geometry, 0.0)
    cb = chip_breaker_rake(insert) if insert is not None else 0.0
    seat_rake, incl = _holder_seat(spec, holder, alpha_i)
    k_left = half
    k_right = math.pi - half
    N_r, _ = face_planes(HALF_PI, seat_rake, incl, alpha_i - seat_rake)
    face = _face(spec, N_r, cb, alpha_i)
    flank = (depth + tip) / math.cos(half)
    segs = corner_profile((0.0, 0.0), k_left, k_right, tip, flank, flank,
                          kinds=(EDGE_THREAD_FLANK, EDGE_NOSE, EDGE_THREAD_FLANK),
                          tags=("flank_a", "nose", "flank_b"), face=face)
    # origin: apex of the tip arc (x) on the profile's centre line (z)
    apex_x = max(seg.point(np.linspace(0, 1, 65))[1].max() for seg in segs)
    for seg in segs:
        if seg.shape == "line":
            seg.p0 = (seg.p0[0], seg.p0[1] - apex_x)
            seg.p1 = (seg.p1[0], seg.p1[1] - apex_x)
        else:
            seg.centre = (seg.centre[0], seg.centre[1] - apex_x)
            seg.p0, seg.p1 = seg.start, seg.end
    edge_pts = _polyline(segs)
    profile = _insert_profile(edge_pts, [], next((s for s in segs if s.tag == "nose"), None))
    pts = profile.pts_zx_m
    thread = ThreadProfile(
        pts_zr_m=np.ascontiguousarray(pts[None, None, :, :]), n_pts=np.array([[len(pts)]], np.int32),
        pitch_m=np.array([form["pitch"]]), lead_m=np.array([form["pitch"] * form["starts"]]),
        starts=np.array([form["starts"]], np.int32), thread_hand=np.array([form["thread_hand"]]),
        form_angle_rad=np.array([form["form"]]), major_r_m=np.array([0.0]), depth_m=np.array([depth]),
        row_z_m=np.zeros((1, 1)), tooth_phase_rad=np.zeros(1))
    info = {"thread": form, "nose": tip, "seat_rake": seat_rake, "incl": incl, "insert_clear": alpha_i, "flutes": 1,
            "turning": True, "kappa_r": HALF_PI, "overhang": num(holder, "overhang_m") if holder is not None else None,
            "shank_hb": numlist(holder, "shank_m") if holder is not None else None, "shank_dir_T": (-1.0, 0.0, 0.0)}
    return Built(segs, _one_tooth(), [], info, thread=thread, insert=profile, rotating=False)


# --------------------------------------------------------------------------- form tools
def build_form_tool(spec: ToolSpec, settings) -> Built:
    rows = plain(spec.geometry.get("profile_m"))
    if not isinstance(rows, (list, tuple)) or len(rows) < 2:
        raise InputError(f"tool {spec.tool_id}: a form tool needs profile_mm with at least two [z, x] rows")
    pts = [(float(plain(r)[0]), float(plain(r)[1])) for r in rows]
    if pts[-1][0] < pts[0][0]:
        pts = pts[::-1]
    # rows are [z_T, x_T] with x_T into the material; the control point is the deepest profile point
    deepest = max(p[1] for p in pts)
    if deepest != 0.0:
        spec.notes.append(f"form profile shifted by {-deepest * 1e3:.3f} mm in x_T: the deepest point is the "
                          f"control point")
        pts = [(z, x - deepest) for z, x in pts]
    insert = sub(spec.geometry, "insert")
    holder = sub(spec.geometry, "holder")
    alpha_i = insert_clearance(spec, insert) if insert is not None else (clearance_primary(spec.geometry) or
                                                                         insert_clearance(spec, {}))
    cb = chip_breaker_rake(insert) if insert is not None else 0.0
    seat_rake, incl = _holder_seat(spec, holder, alpha_i)
    rake = num(spec.geometry, "rake_rad")
    if rake is not None:
        seat_rake = rake
    first_dir = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    N_r, _ = face_planes(math.atan2(first_dir[0], first_dir[1]), seat_rake, incl, alpha_i - seat_rake)
    face = _face(spec, N_r, cb, alpha_i)
    segs = []
    for a, b in zip(pts, pts[1:]):
        if math.hypot(b[0] - a[0], b[1] - a[1]) <= 0.0:
            raise InputError(f"tool {spec.tool_id}: profile_mm has two equal consecutive points")
        segs.append(line(a, b, EDGE_INSERT, rule="const", face=face, tag="form", extra={"psi": 0.0}))
    depth = max(p[1] for p in pts) - min(p[1] for p in pts)
    back_x = min(p[1] for p in pts) - max(depth, 1e-3)
    edge_pts = _polyline(segs)
    back = [(pts[-1][0], back_x), (pts[0][0], back_x)]
    profile = _insert_profile(edge_pts, back, None)
    info = {"seat_rake": seat_rake, "incl": incl, "insert_clear": alpha_i, "flutes": 1, "turning": True,
            "kappa_r": HALF_PI, "form_width": pts[-1][0] - pts[0][0], "form_depth": depth,
            "overhang": num(holder, "overhang_m") if holder is not None else None,
            "shank_hb": numlist(holder, "shank_m") if holder is not None else None, "shank_dir_T": (-1.0, 0.0, 0.0)}
    return Built(segs, _one_tooth(), [], info, insert=profile, rotating=False)
