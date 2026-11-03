"""Indexable milling cutters (face and shoulder mills) and boring heads (docs/models/tooling.md
"Indexable cutters").

Every insert is a tooth at its position (cutting sense, trailing by positions_deg); its edges
(minor edge or wiper flat, nose radius, major edge) are laid out in the meridian plane from the
theoretical corner at the cutting diameter and lifted onto the seat's rake plane, so that each
element carries the true angular position, inclination, normal rake and clearance
(cnctwin.tooling.inserts). Milling entering angles kappa_r are measured from the face plane
(kappa = kappa_r in the edge convention); boring heads feed axially, kappa = 90 deg - kappa_r.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.abi import EDGE_CORNER, EDGE_INSERT, EDGE_WIPER
from ..core.errors import InputError
from . import params as P
from .abi import PART_INSERT
from .built import Built
from .common import cutting_length, edge_prep
from .edges import Face, ToothDef
from .inserts import (chip_breaker_rake, edge_length, face_planes, included_angle, insert_clearance, psi_on_plane,
                      seat_angles, wiper_length)
from .profile import arc, fillet_between, line
from .spec import ToolSpec, flag, num, numlist, sub

HALF_PI = 0.5 * math.pi


def _dir(kappa: float) -> tuple[float, float]:
    """Unit (dz, dr) of a profile direction kappa."""
    return math.sin(kappa), math.cos(kappa)


def corner_profile(corner_zr, k_in: float, k_out: float, nose: float, len_in: float, len_out: float,
                   kinds=(EDGE_INSERT, EDGE_CORNER, EDGE_INSERT), tags=("minor", "nose", "major"), face=None):
    """Minor edge (arriving along k_in) - nose - major edge (leaving along k_out) around a corner."""
    d_in, d_out = _dir(k_in), _dir(k_out)
    segs = []
    if nose > 0.0:
        t_in, t_out, centre, a0, a1 = fillet_between(corner_zr, d_in, d_out, nose)
    else:
        t_in = t_out = corner_zr
    start = (t_in[0] - d_in[0] * len_in, t_in[1] - d_in[1] * len_in)
    if len_in > 0.0:
        segs.append(line(start, t_in, kinds[0], rule="const", face=face, tag=tags[0], extra={"psi": 0.0}))
    if nose > 0.0:
        segs.append(arc(centre, nose, a0, a1, kinds[1], rule="const", face=face, tag=tags[1], extra={"psi": 0.0}))
    end = (t_out[0] + d_out[0] * len_out, t_out[1] + d_out[1] * len_out)
    if len_out > 0.0:
        segs.append(line(t_out, end, kinds[2], rule="const", face=face, tag=tags[2], extra={"psi": 0.0}))
    return segs


def _shift_z(segs, dz: float):
    for seg in segs:
        if seg.shape == "line":
            seg.p0 = (seg.p0[0] + dz, seg.p0[1])
            seg.p1 = (seg.p1[0] + dz, seg.p1[1])
        else:
            seg.centre = (seg.centre[0] + dz, seg.centre[1])
            seg.p0, seg.p1 = seg.start, seg.end


def _min_z(segs) -> float:
    lowest = math.inf
    for seg in segs:
        z, _ = seg.point(np.linspace(0.0, 1.0, 129))
        lowest = min(lowest, float(np.min(z)))
    return lowest


def _lifting_curve(segs, N_r: np.ndarray, Q: np.ndarray):
    ids = {id(s) for s in segs}

    def curve(seg, u):
        if id(seg) not in ids:
            return None
        z, r = seg.point(u)
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        return z, r, psi_on_plane(z, r, N_r, Q)
    return curve


# --------------------------------------------------------------------------- face and shoulder mills
def build_indexable_mill(spec: ToolSpec, settings) -> Built:
    D = spec.need("diameter_m")
    R = 0.5 * D
    inserts = sub(spec.geometry, "inserts")
    if inserts is None:
        raise InputError(f"tool {spec.tool_id}: an indexable mill needs geometry.inserts")
    count = int(num(inserts, "count"))
    kappa_r = num(inserts, "lead_angle_rad")
    if kappa_r is None:
        kappa_r = num(spec.geometry, "lead_angle_rad")
    if kappa_r is None:
        raise InputError(f"tool {spec.tool_id}: inserts.lead_angle_deg is required")
    nose = num(inserts, "nose_radius_m")
    if nose is None:
        nose = num(spec.geometry, "corner_radius_m", 0.0) or 0.0
    gamma_p = num(inserts, "axial_rake_rad", 0.0) or 0.0
    gamma_f = num(inserts, "radial_rake_rad", 0.0) or 0.0
    seat_rake, incl = seat_angles(gamma_p, gamma_f, kappa_r)
    corner = included_angle(spec, inserts)
    ell = edge_length(spec, inserts, corner)
    alpha_i = insert_clearance(spec, inserts)
    cb = chip_breaker_rake(inserts)
    L = cutting_length(spec)
    wiper = flag(inserts, "wiper", False)
    ic = num(inserts, "ic_m") or ell
    k_minor = 0.0 if wiper else -P.pf("insert_minor_dish_rad")
    minor_len = wiper_length(spec, inserts, ic) if wiper else P.pf("wiper_flat_per_ic") * ic
    # major edge length: up to ap_max (cutting_length) and the usable insert edge
    rise = math.sin(kappa_r)
    major_len = min(ell, L / rise if rise > 1e-9 else ell)
    radius, width, angle = edge_prep(spec, insert=True)
    N_r, _ = face_planes(kappa_r, seat_rake, incl, alpha_i - seat_rake)
    face = Face(mode="insert", N_rake=tuple(N_r), rake=cb, clear=alpha_i, edge_radius=radius, chamfer_width=width,
                chamfer_angle=angle)
    kinds = (EDGE_WIPER if wiper else EDGE_INSERT, EDGE_CORNER, EDGE_INSERT)
    segs = corner_profile((0.0, R), k_minor, kappa_r, nose, minor_len, major_len, kinds=kinds, face=face)
    lowest = _min_z(segs)
    _shift_z(segs, -lowest)
    Q = np.array([R, 0.0, -lowest])
    positions = numlist(inserts, "positions_rad")
    if positions is None:
        positions = [2.0 * math.pi * j / count for j in range(count)]
    if len(positions) != count:
        raise InputError(f"tool {spec.tool_id}: inserts.positions_deg has {len(positions)} entries for {count}")
    teeth = [ToothDef(phase=-pos, curve=_lifting_curve(segs, N_r, Q)) for pos in positions]
    outline = _insert_outline(segs)
    top = segs[-1].end
    r_body = _body_radius(spec, D, R, top)
    info = {"D": D, "R": R, "flutes": count, "core_d": 2.0 * r_body, "cutting_length": top[0],
            "head_length": max(top[0], num(spec.geometry, "overall_length_m", top[0]) or top[0]),
            "kappa_r": kappa_r, "nose": nose, "seat_rake": seat_rake, "incl": incl, "insert_clear": alpha_i,
            "wiper": wiper, "edge_length": ell, "centre_cutting": False, "indexable": True,
            "max_diameter": 2.0 * max(s.end[1] for s in segs), "bore_d": num(spec.geometry, "bore_m")}
    return Built(segs, teeth, outline, info)


def _insert_outline(segs):
    """Revolved outline of the cutting corner: the cutter body closes it to the axis at the inner
    end of the minor edge (face) and at the top of the major edge."""
    outline = []
    first = segs[0].start
    outline.append(line((first[0], 0.0), first, None, part=PART_INSERT))
    for seg in segs:
        if seg.shape == "line":
            outline.append(line(seg.start, seg.end, seg.kind))
        else:
            outline.append(arc(seg.centre, seg.rho, seg.a0, seg.a1, seg.kind))
    return outline


def _body_radius(spec, D, R, top) -> float:
    hub = num(spec.geometry, "hub_diameter_m")
    if hub is not None:
        return 0.5 * hub
    ratio = P.pf("indexable_body_ratio")
    spec.notes.append(f"cutter body diameter not given: {ratio:.2f} D (chip pockets)")
    return ratio * min(top[1], R)


# --------------------------------------------------------------------------- boring heads
def build_boring_head(spec: ToolSpec, settings) -> Built:
    """Rotating single-point boring tool (N = inserts.count, usually 1; twin cutters 2) at the set
    diameter: diameter_mm of the item, else 2 x radius_offset of the project instance."""
    D = spec.set_diameter_m if getattr(spec, "set_diameter_m", None) else num(spec.geometry, "diameter_m")
    if D is None:
        raise InputError(f"tool {spec.tool_id}: a boring head needs its set diameter (geometry diameter_mm or the "
                         f"project's radius_offset_mm)")
    rng = numlist(spec.geometry, "diameter_range_m")
    if rng is not None and len(rng) == 2 and not (rng[0] - 1e-9 <= D <= rng[1] + 1e-9):
        raise InputError(f"tool {spec.tool_id}: set diameter {D * 1e3:.3f} mm outside the head's range "
                         f"{rng[0] * 1e3:.1f}-{rng[1] * 1e3:.1f} mm")
    R = 0.5 * D
    insert = sub(spec.geometry, "insert")
    if insert is None:
        raise InputError(f"tool {spec.tool_id}: a boring head needs geometry.insert")
    holder = sub(spec.geometry, "holder")
    nose = num(insert, "nose_radius_m", 0.0) or 0.0
    kappa_r = num(holder, "approach_angle_rad") if holder is not None else None
    if kappa_r is None:
        kappa_r = P.pf("boring_head_approach_rad")
        spec.notes.append(f"approach angle not given: {math.degrees(kappa_r):.0f} deg")
    corner = included_angle(spec, insert)
    if corner is None:
        corner = math.pi - kappa_r
    ell = edge_length(spec, insert, corner)
    alpha_i = insert_clearance(spec, insert)
    cb = chip_breaker_rake(insert)
    seat_rake = num(holder, "rake_rad", 0.0) if holder is not None else 0.0
    incl = num(holder, "inclination_rad", 0.0) if holder is not None else 0.0
    count = int(num(sub(spec.geometry, "inserts"), "count", 1) or 1) if sub(spec.geometry, "inserts") else 1
    kappa_major = HALF_PI - kappa_r
    kappa_minor = HALF_PI + (math.pi - kappa_r - corner)
    radius, width, angle = edge_prep(spec, insert=True)
    N_r, _ = face_planes(kappa_major, seat_rake or 0.0, incl or 0.0, alpha_i - (seat_rake or 0.0))
    face = Face(mode="insert", N_rake=tuple(N_r), rake=cb, clear=alpha_i, edge_radius=radius, chamfer_width=width,
                chamfer_angle=angle)
    # corner where the major and minor edge lines meet; the nose circle touches r = R and z = 0
    side = max(ell - nose * math.tan(0.5 * (kappa_minor - kappa_major)), 0.0)    # between theoretical corners
    segs = corner_profile((0.0, 0.0), kappa_major, kappa_minor, nose, side, side,
                          kinds=(EDGE_INSERT, EDGE_CORNER, EDGE_INSERT), tags=("major", "nose", "minor"), face=face)
    z_min, r_max = _nose_extents(segs, nose)
    for seg in segs:
        _translate(seg, -z_min, R - r_max)
    Q = np.array([R - r_max, 0.0, -z_min])
    teeth = [ToothDef(phase=-2.0 * math.pi * j / count, curve=_lifting_curve(segs, N_r, Q)) for j in range(count)]
    outline = _insert_outline(segs)
    top = max(seg.end[0] for seg in segs)
    info = {"D": D, "R": R, "flutes": count, "core_d": 0.5 * D, "cutting_length": top, "head_length": top,
            "kappa_r": kappa_r, "nose": nose, "seat_rake": seat_rake, "incl": incl, "insert_clear": alpha_i,
            "edge_length": ell, "centre_cutting": False, "indexable": True, "single_point": True}
    return Built(segs, teeth, outline, info)


def _nose_extents(segs, nose: float) -> tuple[float, float]:
    """Bottom and outer tangent of the nose circle (the imaginary tip), or of the corner point."""
    for seg in segs:
        if seg.tag == "nose":
            return seg.centre[0] - seg.rho, seg.centre[1] + seg.rho
    corner = segs[0].end
    return corner[0], corner[1]


def _translate(seg, dz: float, dr: float) -> None:
    if seg.shape == "line":
        seg.p0 = (seg.p0[0] + dz, seg.p0[1] + dr)
        seg.p1 = (seg.p1[0] + dz, seg.p1[1] + dr)
    else:
        seg.centre = (seg.centre[0] + dz, seg.centre[1] + dr)
        seg.p0, seg.p1 = seg.start, seg.end
