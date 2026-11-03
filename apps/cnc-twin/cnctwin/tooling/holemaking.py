"""Hole-making tools: twist, spot, centre, gun and indexable drills, reamers, cutting and forming
taps (docs/models/tooling.md "Drills", "Reamers", "Taps").

Twist-drill lips are straight edges offset from the axis by half the web thickness w; a lip point
at radius r sits at the web angle asin(w/r) (it leads the outer corner by asin(w/r) - asin(w/R)),
its inclination follows sin(i) = (w/r) sin(p) and its normal rake is that of the helical flute
surface (Armarego & Cheng 1972; Wiriyacosol & Armarego 1979; Stephenson & Agapiou 2016 ch. 3).
The chisel edge is orthogonal (it passes through the axis) with the rake of the planar flank of
the other lip. Margins rub (EDGE_MARGIN, chip width = land width). Taps share the thread depth
among their chamfer teeth along the helical path (Dogra, Kapoor & DeVor 2002): every land carries
the thread profile shifted by lead x (trailing angle / 2 pi), truncated by the chamfer cone.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.abi import EDGE_CHISEL, EDGE_CORNER, EDGE_INSERT, EDGE_LIP, EDGE_MARGIN, EDGE_PERIPHERAL, \
    EDGE_THREAD_FLANK
from ..core.errors import InputError
from . import params as P
from .abi import PART_BODY
from .built import Built
from .common import cutting_length, edge_prep, extra, flute_count, helix_angles, phases_from_pitch, radial_face
from .edges import Face, ToothDef
from .mills import thread_form, tooth_row_segs
from .profile import arc, curve, line
from .spec import ToolSpec, clearance_primary, num, sub, text

HALF_PI = 0.5 * math.pi


# --------------------------------------------------------------------------- twist drill geometry
def _web(spec: ToolSpec, D: float, chisel_angle: float, spot: bool) -> float:
    """Half web thickness w from web_thickness_mm, chisel_length_mm (L = 2 w / sin(psi)) or the
    default web ratio."""
    web = num(spec.geometry, "web_thickness_m")
    if web is not None and web > 0.0:
        return 0.5 * web
    chisel = num(spec.geometry, "chisel_length_m")
    if chisel is not None and chisel > 0.0:
        return 0.5 * chisel * math.sin(chisel_angle)
    ratio = P.pf("spot_drill_web_ratio" if spot else "drill_web_ratio")
    spec.notes.append(f"web thickness not given: {ratio:.2f} D")
    return 0.5 * ratio * D


def chisel_angles(point_half: float, chisel_angle: float) -> tuple[float, float, float]:
    """(chisel rake, chisel clearance, flank relief at the chisel) of the planar-flank model.

    Each lip's flank is a plane through the lip that meets the other flank along the chisel edge;
    for chisel edge angle psi this needs the flank relief tan(a_c) = 1 / (tan(pi - psi) tan p), and
    the leading face of a chisel half then rises at tan(phi) = sqrt(cot^2 p + tan^2 a_c) in the
    chisel's normal plane: rake = phi - pi/2, clearance = phi (docs/models/tooling.md)."""
    tan_ac = 1.0 / (math.tan(math.pi - chisel_angle) * math.tan(point_half))
    phi = math.atan(math.sqrt(1.0 / math.tan(point_half) ** 2 + tan_ac ** 2))
    return phi - HALF_PI, phi, math.atan(tan_ac)


def lip_geometry(R: float, w: float, point_half: float, chisel_angle: float) -> dict:
    """Chisel corner and lip of a conventional point (lip in the plane y = w of the tooth frame).

    The chisel half runs from the axis to (x_c, w) with x_c = -w / tan(psi); the lip from there to
    the outer corner (x_R, w), x_R = sqrt(R^2 - w^2), rising at cot(p) per unit x."""
    if not 0.0 <= w < R:
        raise InputError("web thickness must be smaller than the drill diameter")
    x_c = -w / math.tan(chisel_angle) if w > 0.0 else 0.0
    r_c = math.hypot(x_c, w)
    x_R = math.sqrt(R * R - w * w)
    cot_p = 1.0 / math.tan(point_half)
    z_R = (x_R - x_c) * cot_p
    return {"x_c": x_c, "r_c": r_c, "x_R": x_R, "cot_p": cot_p, "z_R": z_R, "w": w}


def _lip_fn(geo: dict):
    def fn(u):
        u = np.asarray(u, dtype=float)
        x = geo["x_c"] + u * (geo["x_R"] - geo["x_c"])
        return (x - geo["x_c"]) * geo["cot_p"], np.sqrt(x * x + geo["w"] ** 2)
    return fn


def build_twist_drill(spec: ToolSpec, settings, spot: bool = False) -> Built:
    D = spec.need("diameter_m")
    R = 0.5 * D
    lips = num(spec.geometry, "lip_count")
    flutes = num(spec.geometry, "flutes")
    N = int(lips) if lips is not None else flute_count(spec, 2)
    if flutes is None or int(flutes) == N:
        phases = phases_from_pitch(spec, N)
    else:
        # lips differ from flutes (e.g. a step drill): lips evenly spaced
        phases = [-2.0 * math.pi * j / N for j in range(N)]
    point = spec.need("tip_angle_rad") if spot and num(spec.geometry, "point_angle_rad") is None \
        else spec.need("point_angle_rad")
    p = 0.5 * point
    helix = helix_angles(spec, N, P.pf("spot_drill_helix_rad" if spot else "drill_helix_rad"))[0]
    chisel_angle = num(spec.geometry, "chisel_edge_angle_rad")
    if chisel_angle is None:
        chisel_angle = P.pf("drill_chisel_angle_rad")
        spec.notes.append(f"chisel edge angle not given: {math.degrees(chisel_angle):.0f} deg")
    if not HALF_PI < chisel_angle < math.pi:
        raise InputError(f"tool {spec.tool_id}: chisel edge angle must lie between 90 and 180 deg")
    w = _web(spec, D, chisel_angle, spot)
    geo = lip_geometry(R, w, p, chisel_angle)
    clear = clearance_primary(spec.geometry)
    if clear is None:
        clear = P.pf("drill_lip_clearance_rad")
        spec.notes.append(f"lip clearance not given: {math.degrees(clear):.0f} deg")
    radius, width, angle = edge_prep(spec)
    if num(spec.geometry, "rake_rad") is not None:
        spec.notes.append("rake_deg of a drill is not used: the lip rake follows from the flute helix")
    lip_face = Face(mode="helicoid", tan_beta0=math.tan(helix), R0=R, clear=clear, edge_radius=radius,
                    chamfer_width=width, chamfer_angle=angle)
    rake_ch, clear_ch, _ = chisel_angles(p, chisel_angle)
    chisel_face = Face(mode="normal", rake=rake_ch, clear=clear_ch, edge_radius=radius)
    segs, outline = [], []
    if geo["r_c"] > 0.0:
        segs.append(line((0.0, 0.0), (0.0, geo["r_c"]), EDGE_CHISEL, rule="chisel", face=chisel_face, tag="chisel"))
        outline.append(line((0.0, 0.0), (0.0, geo["r_c"]), EDGE_CHISEL))
    lip = curve(_lip_fn(geo), EDGE_LIP, rule="lip", face=lip_face, tag="lip")
    segs.append(lip)
    outline.append(curve(_lip_fn(geo), EDGE_LIP))
    head = geo["z_R"]
    if not spot:
        L = cutting_length(spec)
        taper = P.pf("drill_back_taper")
        margin_w = num(spec.geometry, "margin_width_m")
        if margin_w is None:
            margin_w = P.pf("drill_margin_width_ratio") * D
            spec.notes.append(f"margin width not given: {margin_w * 1e3:.2f} mm")
        if L > geo["z_R"]:
            r_top = R - taper * (L - geo["z_R"])
            margin_face = Face(mode="normal", rake=0.0, clear=0.0, edge_radius=radius, db_override=margin_w)
            segs.append(line((geo["z_R"], R), (L, r_top), EDGE_MARGIN, rule="margin", face=margin_face,
                             tag="margin", element_scale=settings.margin_element_factor))
            outline.append(line((geo["z_R"], R), (L, r_top), EDGE_MARGIN))
            head = L
    teeth = [ToothDef(phase=ph, params={"w": w, "R_lip": R, "r_chisel": geo["r_c"], "tan_beta": math.tan(helix),
                                        "R_helix": R, "z_margin": geo["z_R"]}) for ph in phases]
    info = {"D": D, "R": R, "flutes": N, "core_d": 2.0 * w, "cutting_length": head, "head_length": head,
            "point_angle": point, "web": 2.0 * w, "chisel_length": 2.0 * geo["r_c"], "lip_length": lip.length,
            "helix": [helix] * N, "centre_cutting": True, "drill": True, "chisel_rake": rake_ch,
            "chisel_clear": clear_ch}
    return Built(segs, teeth, outline, info)


def build_spot_drill(spec: ToolSpec, settings) -> Built:
    return build_twist_drill(spec, settings, spot=True)


def build_centre_drill(spec: ToolSpec, settings) -> Built:
    """DIN 333 form A: pilot drill (lips, chisel, short margin) and a 60 deg countersink."""
    d1 = spec.need("diameter_m")
    R1 = 0.5 * d1
    N = flute_count(spec, 2)
    point = spec.need("point_angle_rad")
    p = 0.5 * point
    sink = extra(spec, "tip_angle_rad")
    if sink is None:
        sink = P.pf("centre_drill_countersink_rad")
    shank = sub(spec.geometry, "shank")
    d2 = num(shank, "diameter_m") if shank is not None else None
    if d2 is None:
        d2 = P.pf("centre_drill_body_ratio") * d1
        spec.notes.append(f"centre drill body diameter not given: {d2 * 1e3:.2f} mm")
    R2 = 0.5 * d2
    l1 = num(spec.geometry, "cutting_length_m")
    if l1 is None:
        l1 = P.pf("centre_drill_pilot_length_ratio") * d1
        spec.notes.append(f"pilot length not given: {l1 * 1e3:.2f} mm")
    chisel_angle = num(spec.geometry, "chisel_edge_angle_rad") or P.pf("drill_chisel_angle_rad")
    w = _web(spec, d1, chisel_angle, False)
    geo = lip_geometry(R1, w, p, chisel_angle)
    clear = clearance_primary(spec.geometry, P.pf("drill_lip_clearance_rad"))
    radius, width, angle = edge_prep(spec)
    rake = num(spec.geometry, "rake_rad", 0.0) or 0.0
    lip_face = Face(mode="radial", rake=rake, clear=clear, edge_radius=radius, chamfer_width=width,
                    chamfer_angle=angle)
    rake_ch, clear_ch, _ = chisel_angles(p, chisel_angle)
    segs = [line((0.0, 0.0), (0.0, geo["r_c"]), EDGE_CHISEL, rule="chisel",
                 face=Face(mode="normal", rake=rake_ch, clear=clear_ch, edge_radius=radius), tag="chisel"),
            curve(_lip_fn(geo), EDGE_LIP, rule="lip", face=lip_face, tag="lip")]
    z_pilot_top = max(l1, geo["z_R"])
    margin_w = P.pf("drill_margin_width_ratio") * d1
    if z_pilot_top > geo["z_R"]:
        segs.append(line((geo["z_R"], R1), (z_pilot_top, R1), EDGE_MARGIN, rule="margin",
                         face=Face(mode="normal", rake=0.0, clear=0.0, edge_radius=radius, db_override=margin_w),
                         tag="margin"))
    half = 0.5 * sink
    z_sink = z_pilot_top + (R2 - R1) / math.tan(half)
    segs.append(line((z_pilot_top, R1), (z_sink, R2), EDGE_LIP, rule="const",
                     face=Face(mode="radial", rake=rake, clear=clear, edge_radius=radius), tag="countersink",
                     extra={"psi": 0.0}))
    outline = [line(s.start, s.end, s.kind) if s.shape == "line" else curve(_lip_fn(geo), s.kind) for s in segs]
    teeth = [ToothDef(phase=-2.0 * math.pi * j / N,
                      params={"w": w, "R_lip": R1, "r_chisel": geo["r_c"], "tan_beta": 0.0, "R_helix": R1,
                              "z_margin": geo["z_R"]}) for j in range(N)]
    info = {"D": d2, "R": R2, "flutes": N, "core_d": 2.0 * w, "cutting_length": z_sink, "head_length": z_sink,
            "pilot_d": d1, "pilot_length": z_pilot_top, "countersink": sink, "centre_cutting": True, "drill": True}
    return Built(segs, teeth, outline, info)


def build_gun_drill(spec: ToolSpec, settings) -> Built:
    """Single-lip gundrill: inner and outer edge meeting at the offset point, two bearing pads."""
    D = spec.need("diameter_m")
    R = 0.5 * D
    point = num(spec.geometry, "point_angle_rad")
    outer = P.pf("gun_outer_angle_rad")
    inner = P.pf("gun_inner_angle_rad")
    if point is not None:
        # the point angle splits between the outer and inner edge in the standard proportion
        total = math.pi - point
        share = outer / (outer + inner)
        outer, inner = total * share, total * (1.0 - share)
    r_p = P.pf("gun_point_offset_ratio") * D
    L = cutting_length(spec)
    rake = num(spec.geometry, "rake_rad", 0.0) or 0.0
    clear = clearance_primary(spec.geometry, P.pf("drill_lip_clearance_rad"))
    face = radial_face(spec, rake, clear)
    z_i = r_p * math.tan(inner)
    z_o = (R - r_p) * math.tan(outer)
    pad_w = P.pf("gun_pad_width_ratio") * D
    radius = face.edge_radius
    segs = [line((z_i, 0.0), (0.0, r_p), EDGE_LIP, rule="const", face=face, tag="inner", extra={"psi": 0.0}),
            line((0.0, r_p), (z_o, R), EDGE_LIP, rule="const", face=face, tag="outer", extra={"psi": 0.0})]
    if L > z_o:
        segs.append(line((z_o, R), (L, R), EDGE_MARGIN, rule="const", tag="pad", extra={"psi": 0.0},
                         face=Face(mode="normal", rake=0.0, clear=0.0, edge_radius=radius, db_override=pad_w),
                         element_scale=settings.margin_element_factor))
    outline = [line((z_i, 0.0), (0.0, r_p), EDGE_LIP), line((0.0, r_p), (z_o, R), EDGE_LIP)]
    if L > z_o:
        outline.append(line((z_o, R), (L, R), EDGE_MARGIN))
    cutting = ToothDef(phase=0.0, mask=lambda i, seg, z, r: seg.tag != "pad")
    pads = [ToothDef(phase=-a, mask=lambda i, seg, z, r: seg.tag == "pad") for a in P.p("gun_pad_angles_rad")]
    info = {"D": D, "R": R, "flutes": 1, "core_d": 0.5 * D, "cutting_length": L, "head_length": L,
            "point_offset": r_p, "outer_angle": outer, "inner_angle": inner, "centre_cutting": True, "drill": True,
            "pads": len(pads)}
    return Built(segs, [cutting] + pads, outline, info)


def build_indexable_drill(spec: ToolSpec, settings) -> Built:
    """Central and peripheral insert on one straight edge line (entering angle kappa_r from the
    feed direction: kappa = 90 deg - kappa_r), overlapping radially; nose radius at the periphery."""
    D = spec.need("diameter_m")
    R = 0.5 * D
    inserts = sub(spec.geometry, "inserts")
    count = int(num(inserts, "count", 2) or 2)
    kappa_r = num(inserts, "lead_angle_rad")
    if kappa_r is None:
        raise InputError(f"tool {spec.tool_id}: inserts.lead_angle_deg is required")
    kappa = HALF_PI - kappa_r
    nose = num(inserts, "nose_radius_m", 0.0) or 0.0
    L = cutting_length(spec)
    rake = num(inserts, "axial_rake_rad")
    if rake is None:
        rake = num(inserts, "radial_rake_rad", 0.0) or 0.0
    clear = num(inserts, "clearance_rad")
    if clear is None:
        clear = _iso_clearance(text(inserts, "iso_code"), spec)
    radius, width, angle = edge_prep(spec, insert=True)
    face = Face(mode="radial", rake=rake, clear=clear, edge_radius=radius, chamfer_width=width,
                chamfer_angle=angle)
    split = P.pf("idrill_split_ratio") * R
    overlap = P.pf("idrill_overlap_ratio") * R
    tan_k = math.tan(kappa)
    z_low = min(0.0, R * tan_k)
    p0 = (0.0 - z_low, 0.0)
    if nose > 0.0:
        # nose arc tangent to the edge line (direction kappa) and to the hole wall r = R
        centre_r = R - nose
        z_line_at = (centre_r + nose * math.sin(kappa)) * tan_k - z_low
        centre = (z_line_at + nose * math.cos(kappa), centre_r)
        start = (centre[0] - nose * math.cos(kappa), centre[1] + nose * math.sin(kappa))
        segs = [line(p0, start, EDGE_INSERT, rule="const", face=face, tag="edge", extra={"psi": 0.0}),
                arc(centre, nose, kappa, HALF_PI, EDGE_CORNER, rule="const", face=face, tag="nose",
                    extra={"psi": 0.0})]
        top = (centre[0], R)
    else:
        top = (R * tan_k - z_low, R)
        segs = [line(p0, top, EDGE_INSERT, rule="const", face=face, tag="edge", extra={"psi": 0.0})]
    outline = [line(s.start, s.end, s.kind) if s.shape == "line" else arc(s.centre, s.rho, s.a0, s.a1, s.kind)
               for s in segs]
    if L > top[0]:
        outline.append(line(top, (L, R), None, part=PART_BODY))
    teeth = []
    for j in range(count):
        if j == 0:
            mask = (lambda i, seg, z, r: r <= split + overlap)
        elif j == 1:
            mask = (lambda i, seg, z, r: r >= split - overlap)
        else:
            mask = None
        teeth.append(ToothDef(phase=-math.pi * j if count == 2 else -2.0 * math.pi * j / count, mask=mask))
    info = {"D": D, "R": R, "flutes": count, "core_d": 0.4 * D, "cutting_length": L, "head_length": L,
            "split_radius": split, "overlap": overlap, "centre_cutting": True, "drill": True, "indexable": True}
    return Built(segs, teeth, outline, info)


def _iso_clearance(iso_code: str | None, spec: ToolSpec) -> float:
    symbols = P.p("insert_clearance_symbols")
    values = P.p("insert_clearance_rad")
    if iso_code and len(iso_code) >= 2 and iso_code[1].upper() in symbols:
        return float(values[symbols.index(iso_code[1].upper())])
    value = float(values[symbols.index("C")])
    spec.notes.append(f"insert clearance not given and not in the designation: {math.degrees(value):.0f} deg")
    return value


# --------------------------------------------------------------------------- reamers
def build_reamer(spec: ToolSpec, settings) -> Built:
    D = spec.need("diameter_m")
    R = 0.5 * D
    N = flute_count(spec)
    phases = phases_from_pitch(spec, N)
    helices = helix_angles(spec, N, 0.0)
    rake = num(spec.geometry, "rake_rad", 0.0) or 0.0
    clear = clearance_primary(spec.geometry, P.pf("drill_lip_clearance_rad"))
    face = radial_face(spec, rake, clear)
    lead_len = num(spec.geometry, "corner_chamfer_m")
    if lead_len is None or lead_len <= 0.0:
        lead_len = P.pf("reamer_lead_length_ratio") * D
        spec.notes.append(f"reamer lead length not given: {lead_len * 1e3:.2f} mm")
    tip = extra(spec, "tip_angle_rad")
    half = 0.5 * tip if tip is not None else P.pf("reamer_lead_half_angle_rad")
    depth = lead_len * math.tan(half)
    if depth >= R:
        raise InputError(f"tool {spec.tool_id}: reamer lead is deeper than the radius")
    L = cutting_length(spec)
    margin_w = num(spec.geometry, "margin_width_m")
    if margin_w is None:
        margin_w = P.pf("reamer_margin_width_ratio") * D
        spec.notes.append(f"reamer land width not given: {margin_w * 1e3:.2f} mm")
    taper = P.pf("drill_back_taper")
    segs = [line((0.0, R - depth), (lead_len, R), EDGE_PERIPHERAL, rule="helix", face=face, tag="lead")]
    if L > lead_len:
        segs.append(line((lead_len, R), (L, R - taper * (L - lead_len)), EDGE_MARGIN, rule="helix", tag="land",
                         face=Face(mode="normal", rake=0.0, clear=0.0, edge_radius=face.edge_radius,
                                   db_override=margin_w), element_scale=settings.margin_element_factor))
    outline = [line((0.0, 0.0), (0.0, R - depth), None, part=PART_BODY)]
    outline += [line(s.start, s.end, s.kind) for s in segs]
    teeth = [ToothDef(phase=ph, params={"tan_beta": math.tan(b), "R_helix": R, "helix_z0": 0.0})
             for ph, b in zip(phases, helices)]
    info = {"D": D, "R": R, "flutes": N, "core_d": P.core_ratio(N) * D, "cutting_length": L, "head_length": L,
            "lead_length": lead_len, "lead_half_angle": half, "centre_cutting": False}
    return Built(segs, teeth, outline, info)


# --------------------------------------------------------------------------- taps
def build_tap(spec: ToolSpec, settings) -> Built:
    D = spec.need("diameter_m")
    R = 0.5 * D
    form = thread_form(spec)
    forming = spec.type == "tap_forming" or text(spec.geometry, "tap_kind") == "forming"
    pitch, lead = form["pitch"], form["pitch"] * form["starts"]
    if (form["thread_hand"] > 0) != (spec.hand_sign < 0):
        raise InputError(f"tool {spec.tool_id}: a {'right' if form['thread_hand'] > 0 else 'left'}-hand thread "
                         f"needs a {'right' if form['thread_hand'] > 0 else 'left'}-hand tap")
    if forming:
        N = int(num(spec.geometry, "flutes") or P.pf("forming_tap_lobes"))
    else:
        default_lands = int(P.pf("tap_lands_large") if D > P.pf("tap_lands_split_m") else P.pf("tap_lands_small"))
        N = flute_count(spec, default_lands)
    phases = phases_from_pitch(spec, N)
    spiral = text(spec.geometry, "spiral", "straight")
    helix = 0.0
    if spiral == "flute":
        helix = helix_angles(spec, N, P.pf("tap_spiral_flute_helix_rad"))[0]
    elif num(spec.geometry, "helix_rad") is not None:
        helix = helix_angles(spec, N)[0]
    n_ch = num(spec.geometry, "chamfer_threads")
    if n_ch is None:
        n_ch = P.pf("tap_chamfer_threads")
        spec.notes.append(f"chamfer length not given: {n_ch:g} threads")
    L_ch = n_ch * pitch
    L = cutting_length(spec)
    h = form["depth"]
    r_tip = R - h
    if r_tip <= 0.0:
        raise InputError(f"tool {spec.tool_id}: thread depth exceeds the tap radius")
    radius, width, angle = edge_prep(spec)
    if forming:
        # thread-forming lobes displace the material: no rake face (pure indentation), no edge relief
        face = Face(mode="normal", rake=-HALF_PI, clear=0.0, edge_radius=radius, forming=True)
        point_face = face
    else:
        rake = num(spec.geometry, "rake_rad")
        if rake is None:
            rake = P.pf("tap_rake_rad")
            spec.notes.append(f"tap rake not given: {math.degrees(rake):.0f} deg")
        clear = clearance_primary(spec.geometry)
        if clear is None:
            clear = P.pf("tap_chamfer_clearance_rad")
            spec.notes.append(f"chamfer relief not given: {math.degrees(clear):.0f} deg")
        # full-thread lands are concentric (no relief); the chamfer teeth carry the relief and, for
        # spiral-point taps, the gun-nose inclination that drives the chips ahead
        face = Face(mode="radial", rake=rake, clear=0.0, edge_radius=radius, chamfer_width=width,
                    chamfer_angle=angle)
        point_face = Face(mode="radial", rake=rake, clear=clear, edge_radius=radius, chamfer_width=width,
                          chamfer_angle=angle,
                          extra_incl=-P.pf("tap_spiral_point_angle_rad") if spiral == "point" else 0.0)
    run = h * math.tan(0.5 * form["form"])
    z_first = 0.5 * form["crest"] + run - pitch
    n_rows = int(math.ceil((L - z_first) / pitch)) + 1
    segs = []
    for m in range(n_rows):
        row = tooth_row_segs(z_first + m * pitch, R, form, face)
        for seg in row:
            seg.extra["row"] = m
            seg.rule = "helix"
        segs.extend(row)
    tan_b = math.tan(helix)

    def chamfer_r(z):
        return np.where(z < L_ch, r_tip + (R - r_tip) * np.clip(z, 0.0, None) / L_ch, R)

    teeth = []
    for ph in phases:
        trailing = -ph

        def shift(seg, z, r, trailing=trailing):
            z_own = (z + lead * trailing / (2.0 * math.pi)) / (1.0 - lead * tan_b / (2.0 * math.pi * R))
            return z_own, np.minimum(r, chamfer_r(z_own))

        def mask(index, seg, z, r):
            return 0.0 <= z <= L

        def kind_fn(seg, z, r):
            # flank parts ground down to the chamfer cone become chamfer (crest) edges
            if seg.kind == EDGE_THREAD_FLANK and z < L_ch and r >= float(chamfer_r(np.array([z]))[0]) - 1e-12:
                return EDGE_PERIPHERAL
            return None

        def face_fn(seg, z, r):
            return point_face if z < L_ch else None
        teeth.append(ToothDef(phase=ph, params={"tan_beta": tan_b, "R_helix": R, "helix_z0": 0.0}, shift=shift,
                              mask=mask, kind_fn=kind_fn, face_fn=face_fn))
    outline = [line((0.0, 0.0), (0.0, r_tip), None, part=PART_BODY),
               line((0.0, r_tip), (L_ch, R), EDGE_PERIPHERAL),
               line((L_ch, R), (L, R), EDGE_PERIPHERAL)]
    info = {"D": D, "R": R, "flutes": N, "core_d": 2.0 * r_tip if forming else P.core_ratio(N) * D,
            "cutting_length": L, "head_length": L, "thread": form, "chamfer_length": L_ch, "chamfer_threads": n_ch,
            "rows": n_rows, "z_first": z_first, "lead": lead, "spiral": spiral, "helix": [helix] * N,
            "forming": forming, "centre_cutting": False, "tap": True, "r_tip": r_tip}
    return Built(segs, teeth, outline, info)
