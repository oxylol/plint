"""Solid milling tools: end mills (square, ball, toroidal, roughing, serrated), chamfer mills and
engravers, keyseat / T-slot cutters, dovetails, undercutting (lollipop) cutters, slitting saws and
thread mills (docs/models/tooling.md "Milling tools").

Edges follow the constant-lead helix of each flute (lag psi = -z tan(beta_j) / R, Altintas 2012
ch. 2; on balls and corner radii the same lead gives the local helix of Lee & Altintas 1996), end
teeth are straight radial edges (kappa = 0 up to the dish) whose chip width is their radial
extent, and knuckle / serrated roughers carry a wave per flute shifted by wavelength / N
(Merdol & Altintas 2004).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.abi import EDGE_CORNER, EDGE_END, EDGE_PERIPHERAL, EDGE_THREAD_FLANK
from ..core.errors import InputError
from . import params as P
from .abi import PART_BODY
from .built import Built
from .common import (core_diameter, cutting_length, extra, flute_count, helical_teeth, helix_angles,
                     phases_from_pitch, radial_face, rake_clear)
from .profile import arc, line
from .spec import ToolSpec, flag, num, sub, text

HALF_PI = 0.5 * math.pi


def _dish(spec: ToolSpec) -> float:
    """End-tooth dish (concavity) of solid tools: the item does not state it (template), the model default."""
    return P.pf("end_dish_rad")


def _end_teeth_mask(count: int, centre_cutting: bool, r_short: float):
    """Long end teeth reach the axis (two opposite teeth for an even count, one for odd counts)
    when the tool is centre cutting; the other teeth, and all teeth of non-centre-cutting tools,
    start at ``r_short`` (the core radius)."""
    if count == 1:
        long_teeth = {0}
    elif count % 2 == 0:
        long_teeth = {0, count // 2}
    else:
        long_teeth = {0}
    masks = []
    for j in range(count):
        inner = 0.0 if (centre_cutting and j in long_teeth) else r_short

        def mask(index, seg, z, r, inner=inner):
            if seg.kind == EDGE_END:
                return r >= inner
            return True
        masks.append(mask)
    return masks


def _base(spec: ToolSpec):
    D = spec.need("diameter_m")
    R = 0.5 * D
    N = flute_count(spec)
    phases = phases_from_pitch(spec, N)
    return D, R, N, phases


# --------------------------------------------------------------------------- end mills
def build_endmill(spec: ToolSpec, settings) -> Built:
    D, R, N, phases = _base(spec)
    helices = helix_angles(spec, N)
    rake, clear = rake_clear(spec)
    face = radial_face(spec, rake, clear)
    L = cutting_length(spec)
    kind = spec.type
    rc = num(spec.geometry, "corner_radius_m", 0.0) or 0.0
    chamfer = num(spec.geometry, "corner_chamfer_m", 0.0) or 0.0
    if kind == "ball_endmill":
        rc = R
    if rc > R * (1.0 + 1e-9):
        raise InputError(f"tool {spec.tool_id}: corner radius exceeds D/2")
    rc = min(rc, R)
    core = core_diameter(spec, D, N)
    centre = flag(spec.geometry, "centre_cutting", True)
    dish = 0.0 if rc >= R else _dish(spec)
    segs, outline = [], []
    r_flat = R - max(rc, chamfer)
    if r_flat > 0.0:
        z_axis = r_flat * math.tan(dish)
        end = line((z_axis, 0.0), (0.0, r_flat), EDGE_END, rule="radial", face=face, tag="end")
        segs.append(end)
        outline.append(line((z_axis, 0.0), (0.0, r_flat), EDGE_END))
    z_side = 0.0
    if rc > 0.0:
        a_top = HALF_PI
        if L < rc:
            a_top = math.acos(max(-1.0, 1.0 - L / rc))
        corner = arc((rc, R - rc), rc, 0.0, a_top, EDGE_CORNER, rule="helix", face=face, tag="corner")
        segs.append(corner)
        outline.append(arc((rc, R - rc), rc, 0.0, a_top, EDGE_CORNER))
        z_side = rc if a_top >= HALF_PI else corner.end[0]
    elif chamfer > 0.0:
        cham = line((0.0, R - chamfer), (chamfer, R), EDGE_CORNER, rule="helix", face=face, tag="corner")
        segs.append(cham)
        outline.append(line((0.0, R - chamfer), (chamfer, R), EDGE_CORNER))
        z_side = chamfer
    if L > z_side + 1e-12:
        side = line((z_side, R), (L, R), EDGE_PERIPHERAL, rule="helix", face=face, tag="periphery")
        segs.append(side)
        outline.append(line((z_side, R), (L, R), EDGE_PERIPHERAL))
    teeth = helical_teeth(phases, helices, R)
    masks = _end_teeth_mask(N, centre, 0.5 * core)
    for tooth, mask in zip(teeth, masks):
        tooth.mask = mask
    info = {"D": D, "R": R, "flutes": N, "core_d": core, "cutting_length": L, "head_length": L,
            "centre_cutting": centre, "corner_radius": rc, "helix": helices}
    if kind in ("roughing_endmill", "serrated_endmill"):
        _add_wave(spec, settings, segs, teeth, R, D, N, z_side, L, kind, info)
    return Built(segs, teeth, outline, info)


def _add_wave(spec, settings, segs, teeth, R, D, N, z_start, L, kind, info) -> None:
    """Knuckle (sinusoidal) or serrated (trapezoidal) roughing profile on the periphery, shifted by
    one wavelength / N from flute to flute (Merdol & Altintas 2004); the amplitude ramps in over
    the first wavelength so the wave runs out into the corner."""
    serration = sub(spec.geometry, "serration")
    if kind == "roughing_endmill":
        lam = P.pf("knuckle_wavelength_per_D") * D
        depth = P.pf("knuckle_depth_per_D") * D
        spec.notes.append(f"knuckle profile not given: wavelength {lam * 1e3:.2f} mm, depth {depth * 1e3:.3f} mm")
        notch = 0.0
    else:
        lam = num(serration, "pitch_m") if serration is not None else None
        depth = num(serration, "depth_m") if serration is not None else None
        if lam is None:
            lam = P.pf("serration_wavelength_per_D") * D
            spec.notes.append(f"serration pitch not given: {lam * 1e3:.2f} mm")
        if depth is None:
            depth = P.pf("serration_depth_per_D") * D
            spec.notes.append(f"serration depth not given: {depth * 1e3:.3f} mm")
        notch = P.pf("serration_notch_fraction")
    info.update({"wave_wavelength": lam, "wave_depth": depth, "wave_form": "sine" if notch == 0.0 else "trapezoid"})
    scale = min(1.0, lam / (settings.wave_elements * settings.edge_element_m))
    for seg in segs:
        if seg.kind == EDGE_PERIPHERAL:
            seg.element_scale = scale
    for j, tooth in enumerate(teeth):
        offset = j / N

        def shift(seg, z, r, offset=offset):
            if seg.kind != EDGE_PERIPHERAL:
                return z, r
            ramp = np.clip((z - z_start) / lam, 0.0, 1.0)
            phase = (z - z_start) / lam + offset
            return z, r - ramp * depth * wave(phase, notch)
        tooth.shift = shift


def wave(phase, notch: float):
    """Profile depth fraction (0 crest .. 1 valley) at wave phase ``phase`` (periodic in 1)."""
    frac = np.mod(phase, 1.0)
    if notch <= 0.0:
        return 0.5 * (1.0 - np.cos(2.0 * math.pi * frac))
    # trapezoidal notch centred in the period: flanks of notch/4 each, flat bottom notch/2
    start = 0.5 - 0.5 * notch
    t = (frac - start) / notch
    out = np.zeros_like(frac)
    flank = 0.25
    out = np.where((t > 0.0) & (t < flank), t / flank, out)
    out = np.where((t >= flank) & (t <= 1.0 - flank), 1.0, out)
    out = np.where((t > 1.0 - flank) & (t < 1.0), (1.0 - t) / flank, out)
    return out


# --------------------------------------------------------------------------- chamfer mills and engravers
def build_chamfer(spec: ToolSpec, settings) -> Built:
    D = spec.need("diameter_m")
    R = 0.5 * D
    N = flute_count(spec, 1 if spec.type == "engraver" else None)
    phases = phases_from_pitch(spec, N)
    helices = helix_angles(spec, N, 0.0)
    rake, clear = rake_clear(spec)
    face = radial_face(spec, rake, clear)
    tip_angle = spec.need("tip_angle_rad")
    half = 0.5 * tip_angle
    if spec.type == "engraver":
        tip_d = extra(spec, "tip_width_m", "tip_diameter_m") or 0.0
    else:
        tip_d = extra(spec, "tip_diameter_m", "tip_width_m") or 0.0
    r_tip = 0.5 * tip_d
    if r_tip >= R:
        raise InputError(f"tool {spec.tool_id}: tip diameter must be smaller than D")
    centre = flag(spec.geometry, "centre_cutting", spec.type == "engraver")
    core = core_diameter(spec, D, N)
    z_cone = (R - r_tip) / math.tan(half)
    L = num(spec.geometry, "cutting_length_m")
    segs, outline = [], []
    if r_tip > 0.0:
        segs.append(line((0.0, 0.0), (0.0, r_tip), EDGE_END, rule="radial", face=face, tag="end"))
        outline.append(line((0.0, 0.0), (0.0, r_tip), EDGE_END))
    cone = line((0.0, r_tip), (z_cone, R), EDGE_PERIPHERAL, rule="helix", face=face, tag="cone")
    segs.append(cone)
    outline.append(line((0.0, r_tip), (z_cone, R), EDGE_PERIPHERAL))
    head = z_cone
    if L is not None and L > z_cone + 1e-9:
        segs.append(line((z_cone, R), (L, R), EDGE_PERIPHERAL, rule="helix", face=face, tag="periphery"))
        outline.append(line((z_cone, R), (L, R), EDGE_PERIPHERAL))
        head = L
    teeth = helical_teeth(phases, helices, R)
    masks = _end_teeth_mask(N, centre, 0.5 * core)
    for tooth, mask in zip(teeth, masks):
        tooth.mask = mask
    info = {"D": D, "R": R, "flutes": N, "core_d": core, "cutting_length": head, "head_length": head,
            "centre_cutting": centre, "tip_angle": tip_angle}
    return Built(segs, teeth, outline, info)


# --------------------------------------------------------------------------- disc cutters
def _neck(spec: ToolSpec, D: float) -> float:
    neck = sub(spec.geometry, "neck")
    value = num(neck, "diameter_m") if neck is not None else None
    if value is None:
        hub = sub(spec.geometry, "hub")
        value = num(hub, "diameter_m") if hub is not None else None
    if value is None:
        value = P.pf("neck_ratio_default") * D
        spec.notes.append(f"neck diameter not given: {value * 1e3:.2f} mm (neck_ratio_default)")
    return value


def build_disc(spec: ToolSpec, settings) -> Built:
    """Keyseat (straight teeth, dished side edges on both faces), T-slot (staggered side edges on
    alternate teeth for an even tooth count) and slitting saws (peripheral teeth only)."""
    D, R, N, phases = _base(spec)
    helices = helix_angles(spec, N, 0.0)
    rake, clear = rake_clear(spec)
    face = radial_face(spec, rake, clear)
    w = spec.g("width_m") or cutting_length(spec)
    rc = num(spec.geometry, "corner_radius_m", 0.0) or 0.0
    if 2.0 * rc > w:
        raise InputError(f"tool {spec.tool_id}: corner radius larger than half the width")
    saw = spec.type == "slitting_saw"
    if saw:
        hub = sub(spec.geometry, "hub")
        r_in = 0.5 * (num(hub, "diameter_m") if hub is not None and num(hub, "diameter_m") else
                      P.pf("slitting_hub_ratio") * D)
    else:
        r_in = 0.5 * _neck(spec, D)
    if r_in >= R - rc:
        raise InputError(f"tool {spec.tool_id}: neck or hub is as large as the head")
    dish = _dish(spec) if not saw else (num(spec.geometry, "side_clearance_rad") or _dish(spec))
    segs, outline = [], []
    z_in_bottom = (R - rc - r_in) * math.tan(dish)
    side_kind = None if saw else EDGE_END
    outline.append(line((z_in_bottom, 0.0), (z_in_bottom, r_in), None))
    bottom = line((z_in_bottom, r_in), (0.0, R - rc), side_kind, rule="radial", face=face, tag="bottom")
    outline.append(line((z_in_bottom, r_in), (0.0, R - rc), side_kind))
    if side_kind is not None:
        segs.append(bottom)
    if rc > 0.0:
        segs.append(arc((rc, R - rc), rc, 0.0, HALF_PI, EDGE_CORNER, rule="helix", face=face, tag="corner_low"))
        outline.append(arc((rc, R - rc), rc, 0.0, HALF_PI, EDGE_CORNER))
    segs.append(line((rc, R), (w - rc, R), EDGE_PERIPHERAL, rule="helix", face=face, tag="periphery"))
    outline.append(line((rc, R), (w - rc, R), EDGE_PERIPHERAL))
    if rc > 0.0:
        segs.append(arc((w - rc, R - rc), rc, HALF_PI, math.pi, EDGE_CORNER, rule="helix", face=face,
                        tag="corner_high"))
        outline.append(arc((w - rc, R - rc), rc, HALF_PI, math.pi, EDGE_CORNER))
    z_in_top = w - (R - rc - r_in) * math.tan(dish)
    top = line((w, R - rc), (z_in_top, r_in), side_kind, rule="radial", face=face, tag="top")
    outline.append(line((w, R - rc), (z_in_top, r_in), side_kind))
    if side_kind is not None:
        segs.append(top)
    teeth = helical_teeth(phases, helices, R)
    if spec.type == "tslot" and N % 2 == 0 and N >= 4:
        for j, tooth in enumerate(teeth):
            keep = "bottom" if j % 2 == 0 else "top"
            drop = {"bottom", "corner_low"} if keep == "top" else {"top", "corner_high"}

            def mask(index, seg, z, r, drop=drop):
                return seg.tag not in drop
            tooth.mask = mask
    info = {"D": D, "R": R, "flutes": N, "core_d": 2.0 * r_in, "cutting_length": w, "head_length": w,
            "width": w, "neck_d": 2.0 * r_in, "centre_cutting": False, "disc": True}
    return Built(segs, teeth, outline, info)


def build_dovetail(spec: ToolSpec, settings) -> Built:
    D, R, N, phases = _base(spec)
    helices = helix_angles(spec, N, 0.0)
    rake, clear = rake_clear(spec)
    face = radial_face(spec, rake, clear)
    included = extra(spec, "tip_angle_rad", "included_angle_rad")
    if included is None:
        raise InputError(f"tool {spec.tool_id}: a dovetail needs tip_angle_deg (included angle)")
    half = 0.5 * included
    w = spec.g("width_m") or cutting_length(spec)
    rc = num(spec.geometry, "corner_radius_m", 0.0) or 0.0
    r_top = R - w * math.tan(half)
    if r_top <= 0.0:
        raise InputError(f"tool {spec.tool_id}: the dovetail cone closes before the width")
    neck = sub(spec.geometry, "neck")
    neck_d = num(neck, "diameter_m") if neck is not None else None
    if neck_d is None:
        neck_d = min(P.pf("neck_ratio_default") * D, 2.0 * r_top)
        spec.notes.append(f"neck diameter not given: {neck_d * 1e3:.2f} mm")
    if 0.5 * neck_d > r_top + 1e-12:
        # the cone ends where it meets the neck: the cutting width is shorter than stated
        w_eff = (R - 0.5 * neck_d) / math.tan(half)
        spec.notes.append(f"dovetail: the neck ({neck_d * 1e3:.2f} mm) is wider than the cone top at the stated width; "
                          f"the cone ends at the neck, cutting width {w_eff * 1e3:.3f} mm instead of {w * 1e3:.3f} mm")
        w, r_top = w_eff, 0.5 * neck_d
    dish = _dish(spec)
    segs, outline = [], []
    side_dir_k = HALF_PI + half                        # kappa of the cone going up and inward
    if rc > 0.0:
        # corner arc tangent to the bottom plane and to the cone r = R - z tan(half): centre at
        # height rc, at distance rc inside the cone line
        rcen = R - rc * math.tan(half) - rc / math.cos(half)
        if rcen <= 0.0:
            raise InputError(f"tool {spec.tool_id}: dovetail corner radius too large for the head")
        corner_c = (rc, rcen)
        r_bottom = rcen
        z_in = r_bottom * math.tan(dish)
        outline.append(line((z_in, 0.0), (0.0, r_bottom), EDGE_END))
        segs.append(line((z_in, 0.0), (0.0, r_bottom), EDGE_END, rule="radial", face=face, tag="bottom"))
        segs.append(arc(corner_c, rc, 0.0, side_dir_k, EDGE_CORNER, rule="helix", face=face, tag="corner"))
        outline.append(arc(corner_c, rc, 0.0, side_dir_k, EDGE_CORNER))
        start = segs[-1].end
    else:
        z_in = R * math.tan(dish)
        outline.append(line((z_in, 0.0), (0.0, R), EDGE_END))
        segs.append(line((z_in, 0.0), (0.0, R), EDGE_END, rule="radial", face=face, tag="bottom"))
        start = (0.0, R)
    cone = line(start, (w, r_top), EDGE_PERIPHERAL, rule="helix", face=face, tag="cone")
    segs.append(cone)
    outline.append(line(start, (w, r_top), EDGE_PERIPHERAL))
    outline.append(line((w, r_top), (w, 0.5 * neck_d), None, part=PART_BODY))
    teeth = helical_teeth(phases, helices, R)
    centre = flag(spec.geometry, "centre_cutting", False)
    masks = _end_teeth_mask(N, centre, 0.5 * neck_d)
    for tooth, mask in zip(teeth, masks):
        tooth.mask = mask
    info = {"D": D, "R": R, "flutes": N, "core_d": neck_d, "cutting_length": w, "head_length": w,
            "neck_d": neck_d, "centre_cutting": centre, "included_angle": included}
    return Built(segs, teeth, outline, info)


def build_lollipop(spec: ToolSpec, settings) -> Built:
    D, R, N, phases = _base(spec)
    helices = helix_angles(spec, N, 0.0)
    rake, clear = rake_clear(spec)
    face = radial_face(spec, rake, clear)
    wrap = extra(spec, "wrap_angle_rad")
    neck = sub(spec.geometry, "neck")
    neck_d = num(neck, "diameter_m") if neck is not None else None
    if wrap is not None:
        a_max = 0.5 * wrap
    elif neck_d is not None:
        a_max = math.pi - math.asin(min(1.0, 0.5 * neck_d / R))
    else:
        a_max = 0.5 * P.pf("lollipop_wrap_rad")
        spec.notes.append(f"wrap angle not given: {math.degrees(2 * a_max):.0f} deg")
    a_max = min(a_max, math.pi - 1e-6)
    r_join = R * math.sin(a_max)
    if neck_d is None:
        neck_d = 2.0 * r_join
    if 0.5 * neck_d > r_join + 1e-9 and a_max > HALF_PI:
        raise InputError(f"tool {spec.tool_id}: the neck ({neck_d * 1e3:.2f} mm) is wider than the ball at the "
                         f"end of its cutting edge")
    ball = arc((R, 0.0), R, 0.0, a_max, EDGE_CORNER, rule="helix", face=face, tag="ball")
    outline = [arc((R, 0.0), R, 0.0, a_max, EDGE_CORNER)]
    z_join = R * (1.0 - math.cos(a_max))
    if r_join > 0.5 * neck_d + 1e-9:
        outline.append(line((z_join, r_join), (z_join, 0.5 * neck_d), None, part=PART_BODY))
    teeth = helical_teeth(phases, helices, R)
    info = {"D": D, "R": R, "flutes": N, "core_d": neck_d, "cutting_length": z_join, "head_length": z_join,
            "neck_d": neck_d, "centre_cutting": True, "wrap": 2.0 * a_max}
    return Built([ball], teeth, outline, info)


# --------------------------------------------------------------------------- thread mills
def thread_form(spec: ToolSpec) -> dict:
    """Pitch, form angle, crest flat and tooth depth of the tool's thread (ISO 68-1 proportions:
    crest P/8, depth 17H/24 with H = P / (2 tan(form/2)))."""
    thread = sub(spec.geometry, "thread")
    if thread is None:
        raise InputError(f"tool {spec.tool_id}: geometry needs thread.pitch_mm")
    pitch = num(thread, "pitch_m")
    if pitch is None or pitch <= 0.0:
        raise InputError(f"tool {spec.tool_id}: thread.pitch_mm missing")
    form = num(thread, "form_angle_rad")
    if form is None:
        form = P.pf("thread_form_angle_rad")
        spec.notes.append(f"thread form angle not given: {math.degrees(form):.0f} deg")
    starts = int(num(thread, "starts", 1) or 1)
    hand_text = text(thread, "hand", "right")
    H = pitch / (2.0 * math.tan(0.5 * form))
    return {"pitch": pitch, "form": form, "crest": P.pf("tap_crest_flat_per_pitch") * pitch,
            "depth": P.pf("tap_depth_per_H") * H, "H": H, "starts": starts,
            "thread_hand": 1.0 if hand_text == "right" else -1.0,
            "major_d": num(thread, "diameter_m")}


def tooth_row_segs(z_centre: float, R: float, form: dict, face, tags=("flank_low", "crest", "flank_high")):
    """Flank - crest - flank of one thread tooth centred at ``z_centre`` with its crest at R."""
    half = 0.5 * form["form"]
    c = form["crest"]
    h = form["depth"]
    run = h * math.tan(half)
    p_low = (z_centre - 0.5 * c - run, R - h)
    p_c0 = (z_centre - 0.5 * c, R)
    p_c1 = (z_centre + 0.5 * c, R)
    p_high = (z_centre + 0.5 * c + run, R - h)
    return [line(p_low, p_c0, EDGE_THREAD_FLANK, rule="helix", face=face, tag=tags[0]),
            line(p_c0, p_c1, EDGE_PERIPHERAL, rule="helix", face=face, tag=tags[1]),
            line(p_c1, p_high, EDGE_THREAD_FLANK, rule="helix", face=face, tag=tags[2])]


def build_thread_mill(spec: ToolSpec, settings) -> Built:
    D, R, N, phases = _base(spec)
    helices = helix_angles(spec, N, 0.0)
    rake, clear = rake_clear(spec)
    face = radial_face(spec, rake, clear)
    form = thread_form(spec)
    L = cutting_length(spec)
    pitch = form["pitch"]
    single = flag(spec.geometry, "single_profile", False)
    rows = num(spec.geometry, "rows")
    run = form["depth"] * math.tan(0.5 * form["form"])
    width = form["crest"] + 2.0 * run
    if single:
        n_rows = 1
    elif rows is not None:
        n_rows = int(rows)
    else:
        n_rows = max(1, int(math.floor((L - width) / pitch + 1e-9)) + 1)
        spec.notes.append(f"thread mill rows not given: {n_rows} from the cutting length")
    z0 = 0.5 * form["crest"] + run
    segs, outline = [], []
    root = R - form["depth"]
    outline.append(line((0.0, 0.0), (0.0, root), None, part=PART_BODY))
    for m in range(n_rows):
        zc = z0 + m * pitch
        row = tooth_row_segs(zc, R, form, face)
        for seg in row:
            seg.extra["row"] = m
        segs.extend(row)
        outline.extend(tooth_row_segs(zc, R, form, None))
        if m < n_rows - 1:
            gap0 = row[-1].end
            gap1 = (z0 + (m + 1) * pitch - 0.5 * form["crest"] - run, root)
            if gap1[0] > gap0[0] + 1e-12:
                outline.append(line(gap0, gap1, None, part=PART_BODY))
    top = z0 + (n_rows - 1) * pitch + 0.5 * form["crest"] + run
    teeth = helical_teeth(phases, helices, R)
    for seg in outline:
        if seg.kind is None:
            seg.part = PART_BODY
    info = {"D": D, "R": R, "flutes": N, "core_d": core_diameter(spec, D, N), "cutting_length": top,
            "head_length": top, "rows": n_rows, "thread": form, "centre_cutting": False}
    return Built(segs, teeth, outline, info)


__all__ = ["build_endmill", "build_chamfer", "build_disc", "build_dovetail", "build_lollipop", "build_thread_mill",
           "thread_form", "tooth_row_segs", "wave"]
