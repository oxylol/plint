"""SINUMERIK 840D sl technology cycles, called like subprograms with positional parameters.

Parameter lists as in the 840D sl Job Planning manual (siemens-840dsl-jobplan-ru-2018 chapter 20.1,
db/controllers/siemens-840d-sl.yaml words.cycle_parameters) and, for the cycles that manual lists only
by name, the 840D cycle manual's classic lists:

    CYCLE81(RTP, RFP, SDIS, DP, DPR, DTB, _GMODE, _DMODE, _AMODE)
    CYCLE82(RTP, RFP, SDIS, DP, DPR, DTB, _GMODE, _DMODE, _AMODE)
    CYCLE83(RTP, RFP, SDIS, DP, DPR, FDEP, FDPR, _DAM, DTB, DTS, FRF, VARI, _AXN, _MDEP, _VRT, _DTD, _DIS1)
    CYCLE84(RTP, RFP, SDIS, DP, DPR, DTB, SDAC, MPIT, PIT, POSS, SST, SST1, _AXN, _PITA, _TECHNO, _VARI,
            _DAM, _VRT)
    CYCLE840(RTP, RFP, SDIS, DP, DPR, DTB, SDR, SDAC, ENC, MPIT, PIT, _AXN, _PITA, _TECHNO)
    CYCLE85(RTP, RFP, SDIS, DP, DPR, DTB, FFR, RFF)
    CYCLE86(RTP, RFP, SDIS, DP, DPR, DTB, SDIR, RPA, RPO, RPAP, POSS)
    CYCLE87(RTP, RFP, SDIS, DP, DPR, SDIR)
    CYCLE88(RTP, RFP, SDIS, DP, DPR, DTB, SDIR)
    CYCLE89(RTP, RFP, SDIS, DP, DPR, DTB)
    HOLES1(SPCA, SPCO, STA1, FDIS, DBH, NUM)          HOLES2(CPA, CPO, RAD, STA1, INDA, NUM)
    POCKET3(_RTP, _RFP, _SDIS, _DP, _LENG, _WID, _CRAD, _PA, _PO, _STA, _MID, _FAL, _FALD, _FFP1, _FFD,
            _CDIR, _VARI, _MIDA, _AP1, _AP2, _AD, _RAD1, _DP1)
    POCKET4(_RTP, _RFP, _SDIS, _DP, _PRAD, _PA, _PO, _MID, _FAL, _FALD, _FFP1, _FFD, _CDIR, _VARI, _MIDA,
            _AP1, _AD, _RAD1, _DP1)
    SLOT1(RTP, RFP, SDIS, DP, DPR, NUM, LENG, WID, CPA, CPO, RAD, STA1, INDA, FFD, FFP1, MID, CDIR, FAL,
          VARI, MIDF, FFP2, SSF, _FALD, _STA2, _DP1)
    SLOT2(RTP, RFP, SDIS, DP, DPR, NUM, AFSL, WID, CPA, CPO, RAD, STA1, INDA, FFD, FFP1, MID, CDIR, FAL,
          VARI, MIDF, FFP2, SSF, _FFCP)
    CYCLE800(_FR, _TC, _ST, _MODE, _X0, _Y0, _Z0, _A, _B, _C, _X1, _Y1, _Z1, _DIR, _FR_I)
    CYCLE832(_TOL, _TOLM, _OTOL)
    CYCLE95 / CYCLE97 / CYCLE93: turning.py

Omitted trailing parameters are 0 (as the control passes them). Lengths are in program units, angles
in degrees. The drilling axis is the plane normal (_DMODE / _AXN select another); the retraction side
is that of RTP relative to RFP; SDIS is unsigned. DPR (relative depth) wins over DP when both are
given (the control's rule). DTB > 0 dwells seconds, DTB < 0 spindle revolutions.
"""

from __future__ import annotations

import math

from ... import hooks
from ...expr import round_half_away
from ...ir import Command, Kind
from ...pipeline import PLANES
from . import milling as M
from .kit import Alarm, clean_matrix, rot_x

ACTION = "siemens840d:action"

# ISO 261 coarse pitch series (mm) for MPIT (thread size)
ISO_COARSE_PITCH_MM = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0, 16: 2.0, 18: 2.5,
                       20: 2.5, 22: 2.5, 24: 3.0, 27: 3.0, 30: 3.5, 33: 3.5, 36: 4.0, 39: 4.0, 42: 4.5, 45: 4.5,
                       48: 5.0, 52: 5.0, 56: 5.5, 60: 5.5}


def arg(values: list, k: int, default=0.0):
    if k < len(values) and values[k] is not None:
        return values[k]
    return default


def num(values: list, k: int, name: str, default: float = 0.0) -> float:
    value = arg(values, k, default)
    if isinstance(value, str):
        raise Alarm("cycle_parameter", f"{name} must be a number", name)
    return float(value)


class Action:
    """Resumer of the cycle steps that act on the spindle or stop the program, in sequence with the moves."""

    def resume(self, ex, data: dict, feedback) -> list:
        h = ex.handler
        s = ex.s
        kind = data["action"]
        if kind == "spindle":
            s.spindle.direction = int(data["direction"])
            if data.get("rpm") is not None:
                s.spindle.s_value = float(data["rpm"])
            h.emit_spindle()
        elif kind == "orient":
            s.spindle.direction = 0
            s.spindle.oriented = True
            h.emit_spindle(orient=math.radians(float(data["angle_deg"])))
        elif kind == "stop":
            h.program_stop("M0", data.get("message", ""))
        elif kind == "mode":
            ex.emit(Command(Kind.MODE, ex.line_ref(), modal=dict(data.get("modal", {}))))
        elif kind == "frames":
            if "tcp" in data:
                s.tcp = str(data["tcp"])
            h.update_frames()
        elif kind == "nose":
            code = str(data["code"])
            if code == "G40":
                if s.comp_state is not None and s.modal.get("7") in ("G41", "G42"):
                    ex.emit(Command(Kind.MODE, ex.line_ref(), modal={"comp_marker": {"cancel": "G40",
                                                                                      "seq": s.block_seq}}))
                s.modal["7"] = "G40"
            else:
                ex.start_comp("nose")
                s.modal["7"] = code
        elif kind == "modal_call":
            mcall = s.sys.get("mcall")
            if mcall is not None:
                values = [a[1] if a is not None else None for a in mcall["args"]]
                return expansion(h, mcall["name"], values)
        return []


hooks.register_resumer(ACTION, Action(), replace=True)


def spindle_step(direction: int, rpm: float | None = None):
    return M.action(ACTION, action="spindle", direction=int(direction), rpm=rpm)


# ====================================================================== drilling

def drill_plane(h, values: list, dmode_index: int | None = None, axn_index: int | None = None) -> M.Plane:
    plane = h.plane()
    if dmode_index is not None:
        dmode = int(num(values, dmode_index, "_DMODE")) % 10
        if dmode:
            plane = {1: "XY", 2: "ZX", 3: "YZ"}.get(dmode)
            if plane is None:
                raise Alarm("cycle_parameter", f"_DMODE {dmode}", "_DMODE")
    a, b, n = PLANES[plane]
    if axn_index is not None:
        axn = int(num(values, axn_index, "_AXN"))
        if axn == 1:
            a, b, n = b, n, a
        elif axn == 2:
            a, b, n = n, a, b
    for letter in (a, b, n):
        if letter not in h.ex.by_letter:
            raise Alarm("cycle_axes", f"axis {letter} is not on this path")
    rtp = num(values, 0, "RTP")
    rfp = num(values, 1, "RFP")
    return M.Plane(a, b, n, 1.0 if rtp >= rfp else -1.0)


def drill_data(h, values: list, plane: M.Plane):
    """(retract, approach, depth) in SI from RTP, RFP, SDIS, DP, DPR."""
    rtp = h.length_si(num(values, 0, "RTP"))
    rfp = h.length_si(num(values, 1, "RFP"))
    sdis = h.length_si(abs(num(values, 2, "SDIS")))
    dp = arg(values, 3, None)
    dpr = num(values, 4, "DPR")
    if dpr:
        depth = rfp - plane.sign * h.length_si(abs(dpr))
    elif dp is not None:
        depth = h.length_si(float(dp))
    else:
        raise Alarm("cycle_parameter", "no drilling depth (DP or DPR)", "DP")
    approach = M.drill_levels(plane, rtp, rfp, sdis, depth)
    return rtp, approach, depth


def dwell_steps(dtb: float) -> list:
    if dtb > 0.0:
        return M.dwell(seconds=dtb)
    if dtb < 0.0:
        return M.dwell(revolutions=-dtb)
    return []


def drilling_feed(h) -> float:
    s = h.s
    if not s.feed_set or s.feed_si <= 0.0:
        raise Alarm("feed_zero", "no feed for the drilling cycle")
    return s.feed_si


def spindle_direction(h, value: float, name: str) -> int:
    code = int(round_half_away(value))
    if code == 3:
        return -1
    if code == 4:
        return 1
    if code == 5:
        return 0
    raise Alarm("cycle_spindle_direction", f"{name}={code}", name)


def cycle81(h, values, extra: list | None = None) -> list:
    plane = drill_plane(h, values, dmode_index=7)
    retract, approach, depth = drill_data(h, values, plane)
    f = drilling_feed(h)
    return [M.rapid(plane, n=approach), M.feed(plane, f, n=depth)] + (extra or []) + [M.rapid(plane, n=retract)]


def cycle82(h, values) -> list:
    return cycle81(h, values, dwell_steps(num(values, 5, "DTB")))


def cycle83(h, values) -> list:
    plane = drill_plane(h, values, axn_index=12)
    retract, approach, depth = drill_data(h, values, plane)
    rfp = h.length_si(num(values, 1, "RFP"))
    f = drilling_feed(h)
    fdep, fdpr = arg(values, 5, None), num(values, 6, "FDPR")
    if fdpr:
        first = h.length_si(abs(fdpr))
    elif fdep is not None and float(fdep) != 0.0:
        first = abs(rfp - h.length_si(float(fdep)))
    else:
        raise Alarm("cycle_first_depth", "FDEP/FDPR")
    dam = num(values, 7, "DAM")
    dtb, dts = num(values, 8, "DTB"), num(values, 9, "DTS")
    frf = num(values, 10, "FRF") or 1.0
    if not 0.0 < frf <= 1.0:
        raise Alarm("cycle_parameter", f"FRF={frf:g}", "FRF")
    vari = int(num(values, 11, "VARI")) % 10
    mdep = h.length_si(num(values, 13, "_MDEP"))
    vrt = h.length_si(num(values, 14, "_VRT")) or float(h.p["cycle83_chip_break_retract_m"])
    dtd = num(values, 15, "_DTD")
    dis1 = h.length_si(num(values, 16, "_DIS1"))
    if dam < 0.0:
        degression = first * min(abs(dam), 100.0) / 100.0
    else:
        degression = h.length_si(dam)
    depths = M.peck_depths(rfp, depth, first, degression, mdep if mdep > 0.0 else (degression or 0.0), plane.sign)
    total = abs(rfp - depth)
    if dis1 <= 0.0:
        dis1 = min(total / 50.0, float(h.p["cycle83_max_anticipation_m"])) if total > float(
            h.p["cycle83_anticipation_limit_depth_m"]) else float(h.p["cycle83_anticipation_m"])
    out = [M.rapid(plane, n=approach)]
    previous = approach
    for k, level in enumerate(depths):
        last = k == len(depths) - 1
        if vari == 1 and k > 0:
            out.append(M.rapid(plane, n=previous + plane.sign * dis1))
        out.append(M.feed(plane, f * (frf if k == 0 else 1.0), n=level))
        if last:
            out += dwell_steps(dtd if dtd else dtb)
            break
        out += dwell_steps(dtb)
        if vari == 1:
            out.append(M.rapid(plane, n=approach))
            out += dwell_steps(dts)
        else:
            out.append(M.feed(plane, f, n=level + plane.sign * vrt))
        previous = level
    out.append(M.rapid(plane, n=retract))
    return out


def thread_pitch(h, values, mpit_index: int, pit_index: int, pita_index: int | None) -> float:
    """Signed pitch (SI, > 0 right-hand) from PIT (unit by _PITA) or MPIT (ISO coarse thread size)."""
    pit = num(values, pit_index, "PIT")
    mpit = num(values, mpit_index, "MPIT")
    if pit and mpit:
        raise Alarm("cycle_pitch", "MPIT and PIT both given")
    if pit:
        pita = int(num(values, pita_index, "_PITA")) if pita_index is not None else 0
        if pita in (0, 1):
            return h.length_si(pit) if h.s.units == "mm" else pit * 1e-3
        if pita == 2:
            return math.copysign(0.0254 / abs(pit), pit)
        if pita == 3:
            return pit * 0.0254
        raise Alarm("cycle_pitch", f"_PITA={pita}")
    if mpit:
        size = int(round_half_away(abs(mpit)))
        if size not in ISO_COARSE_PITCH_MM:
            raise Alarm("cycle_pitch", f"MPIT={mpit:g}")
        return math.copysign(ISO_COARSE_PITCH_MM[size] * 1e-3, mpit)
    raise Alarm("cycle_pitch", "no pitch (PIT or MPIT)")


def cycle84(h, values) -> list:
    plane = drill_plane(h, values, axn_index=12)
    retract, approach, depth = drill_data(h, values, plane)
    dtb = num(values, 5, "DTB")
    sdac = spindle_direction(h, num(values, 6, "SDAC") or 3.0, "SDAC")
    pitch = thread_pitch(h, values, 7, 8, 13)
    poss = num(values, 9, "POSS")
    sst = num(values, 10, "SST") or h.s.spindle.s_value
    sst1 = num(values, 11, "SST1") or sst
    if sst <= 0.0:
        raise Alarm("cycle_parameter", "no tapping speed (SST or S)", "SST")
    vari = int(num(values, 15, "_VARI")) % 10
    dam = h.length_si(abs(num(values, 16, "_DAM")))
    vrt = h.length_si(num(values, 17, "_VRT")) or float(h.p["cycle83_chip_break_retract_m"])
    lead = abs(pitch)
    hand = 1 if pitch > 0.0 else -1
    tap = {"lead_m": lead, "direction": hand}
    previous_rpm = h.s.spindle.s_value
    out = [M.rapid(plane, n=approach), M.action(ACTION, action="orient", angle_deg=poss),
           spindle_step(-hand, sst)]
    depths = M.peck_depths(approach, depth, dam, 0.0, 0.0, plane.sign) if vari in (1, 2) and dam > 0.0 else [depth]
    for k, level in enumerate(depths):
        out.append(M.feed(plane, lead, "per_rev", override=False, n=level,
                          modal={"rigid_tap": True, "tap": dict(tap, phase="in", rpm=sst)}))
        if k == len(depths) - 1:
            break
        back = approach if vari == 2 else level + plane.sign * vrt
        out.append(M.feed(plane, lead, "per_rev", override=False, n=back,
                          modal={"rigid_tap": True, "tap": dict(tap, phase="out", rpm=sst1)}))
        if vari == 2:
            out.append(M.feed(plane, lead, "per_rev", override=False, n=level,
                              modal={"rigid_tap": True, "tap": dict(tap, phase="in", rpm=sst)}))
    out += dwell_steps(dtb)
    out.append(M.feed(plane, lead, "per_rev", override=False, n=approach,
                      modal={"rigid_tap": True, "tap": dict(tap, phase="out", rpm=sst1)}))
    out.append(M.rapid(plane, n=retract))
    out.append(spindle_step(sdac, previous_rpm))
    return out


def cycle840(h, values) -> list:
    plane = drill_plane(h, values, axn_index=11)
    retract, approach, depth = drill_data(h, values, plane)
    dtb = num(values, 5, "DTB")
    sdr = num(values, 6, "SDR")
    sdac = spindle_direction(h, num(values, 7, "SDAC") or 3.0, "SDAC")
    enc = int(num(values, 8, "ENC")) % 10
    direction = h.s.spindle.direction
    if direction == 0:
        raise Alarm("cycle_spindle_direction", "spindle stopped")
    reverse = -direction if not sdr else spindle_direction(h, sdr, "SDR")
    out = [M.rapid(plane, n=approach)]
    if enc == 0:
        pitch = thread_pitch(h, values, 9, 10, 12)
        lead = abs(pitch)
        modal = {"tap": {"lead_m": lead, "direction": 1 if pitch > 0.0 else -1, "floating": True}}
        out.append(M.feed(plane, lead, "per_rev", override=False, n=depth, modal=dict(modal, phase="in")))
        out += dwell_steps(dtb)
        out.append(spindle_step(reverse))
        out.append(M.feed(plane, lead, "per_rev", override=False, n=approach, modal=dict(modal, phase="out")))
    else:
        f = drilling_feed(h)
        out.append(M.feed(plane, f, override=False, n=depth, modal={"tapping": True}))
        out += dwell_steps(dtb)
        out.append(spindle_step(reverse))
        out.append(M.feed(plane, f, override=False, n=approach, modal={"tapping": True}))
    out.append(spindle_step(sdac))
    out.append(M.rapid(plane, n=retract))
    return out


def cycle85(h, values) -> list:
    plane = drill_plane(h, values, dmode_index=9)
    retract, approach, depth = drill_data(h, values, plane)
    unit = h.feed_unit_m() / 60.0
    ffr = num(values, 6, "FFR") * unit
    rff = num(values, 7, "RFF") * unit
    if ffr <= 0.0 or rff <= 0.0:
        raise Alarm("cycle_no_feed", "FFR / RFF")
    return ([M.rapid(plane, n=approach), M.feed(plane, ffr, "per_min", n=depth)] + dwell_steps(num(values, 5, "DTB"))
            + [M.feed(plane, rff, "per_min", n=approach), M.rapid(plane, n=retract)])


def cycle86(h, values) -> list:
    plane = drill_plane(h, values, dmode_index=12)
    retract, approach, depth = drill_data(h, values, plane)
    f = drilling_feed(h)
    sdir = spindle_direction(h, num(values, 6, "SDIR"), "SDIR")
    rpa, rpo, rpap = (h.length_si(num(values, k, name)) for k, name in ((7, "RPA"), (8, "RPO"), (9, "RPAP")))
    poss = num(values, 10, "POSS")
    ex = h.ex
    centre = (ex.si(plane.a, ex.s.pos[plane.a]), ex.si(plane.b, ex.s.pos[plane.b]))
    out = [M.rapid(plane, n=approach), M.feed(plane, f, n=depth)] + dwell_steps(num(values, 5, "DTB"))
    out += [M.action(ACTION, action="orient", angle_deg=poss),
            M.rapid(plane, a=centre[0] + rpa, b=centre[1] + rpo, n=depth + plane.sign * rpap),
            M.rapid(plane, n=retract), M.rapid(plane, a=centre[0], b=centre[1]),
            spindle_step(sdir)]
    return out


def cycle87(h, values) -> list:
    plane = drill_plane(h, values)
    retract, approach, depth = drill_data(h, values, plane)
    sdir = spindle_direction(h, num(values, 5, "SDIR"), "SDIR")
    f = drilling_feed(h)
    return [M.rapid(plane, n=approach), M.feed(plane, f, n=depth), spindle_step(0),
            M.action(ACTION, action="stop", message="CYCLE87: retract the tool manually"),
            M.rapid(plane, n=retract), spindle_step(sdir)]


def cycle88(h, values) -> list:
    plane = drill_plane(h, values)
    retract, approach, depth = drill_data(h, values, plane)
    sdir = spindle_direction(h, num(values, 6, "SDIR"), "SDIR")
    f = drilling_feed(h)
    return ([M.rapid(plane, n=approach), M.feed(plane, f, n=depth)] + dwell_steps(num(values, 5, "DTB")) +
            [spindle_step(0), M.action(ACTION, action="stop", message="CYCLE88: retract the tool manually"),
             M.rapid(plane, n=retract), spindle_step(sdir)])


def cycle89(h, values) -> list:
    plane = drill_plane(h, values)
    retract, approach, depth = drill_data(h, values, plane)
    f = drilling_feed(h)
    return ([M.rapid(plane, n=approach), M.feed(plane, f, n=depth)] + dwell_steps(num(values, 5, "DTB")) +
            [M.feed(plane, f, n=approach), M.rapid(plane, n=retract)])


# ====================================================================== hole patterns

def pattern_positions(h, name: str, values: list) -> list[tuple[float, float]]:
    if name == "HOLES1":
        spca, spco = h.length_si(num(values, 0, "SPCA")), h.length_si(num(values, 1, "SPCO"))
        sta1 = math.radians(num(values, 2, "STA1"))
        fdis, dbh = h.length_si(num(values, 3, "FDIS")), h.length_si(num(values, 4, "DBH"))
        count = int(round_half_away(num(values, 5, "NUM")))
        if count <= 0:
            raise Alarm("cycle_holes_zero", "NUM")
        points = [(spca + (fdis + k * dbh) * math.cos(sta1), spco + (fdis + k * dbh) * math.sin(sta1))
                  for k in range(count)]
        ex = h.ex
        a, b, _ = PLANES[h.plane()]
        here = (ex.si(a, ex.s.pos[a]), ex.si(b, ex.s.pos[b]))
        if math.dist(here, points[-1]) < math.dist(here, points[0]):
            points.reverse()
        return points
    cpa, cpo = h.length_si(num(values, 0, "CPA")), h.length_si(num(values, 1, "CPO"))
    rad = h.length_si(num(values, 2, "RAD"))
    sta1 = num(values, 3, "STA1")
    count = int(round_half_away(num(values, 5, "NUM")))
    if count <= 0:
        raise Alarm("cycle_holes_zero", "NUM")
    inda = num(values, 4, "INDA") or 360.0 / count
    return [(cpa + rad * math.cos(math.radians(sta1 + k * inda)), cpo + rad * math.sin(math.radians(sta1 + k * inda)))
            for k in range(count)]


def holes(h, name: str, values: list) -> list:
    mcall = h.s.sys.get("mcall")
    if mcall is not None and mcall["name"] not in CYCLES:
        raise Alarm("unsupported_function", f"{name} with a modally called user subprogram (MCALL a drilling cycle)")
    a, b, n = PLANES[h.plane()]
    plane = M.Plane(a, b, n)
    out = []
    for point in pattern_positions(h, name, values):
        out.append(M.rapid(plane, a=point[0], b=point[1]))
        if mcall is not None:
            out.append(M.action(ACTION, action="modal_call"))
    return out


# ====================================================================== pockets and slots

def milling_direction(h, cdir: int) -> int:
    """+1 counter-clockwise, -1 clockwise loops for _CDIR 0 climb, 1 conventional, 2 G2, 3 G3."""
    if cdir == 2:
        return -1
    if cdir == 3:
        return 1
    spindle = h.s.spindle.direction
    if spindle == 0:
        raise Alarm("cycle_spindle_direction", "climb/conventional milling needs a running spindle")
    climb = 1 if spindle < 0 else -1
    if cdir == 0:
        return climb
    if cdir == 1:
        return -climb
    raise Alarm("cycle_cdir", f"_CDIR={cdir}")


def tool_radius(h) -> float:
    if h.s.d_number == 0 or not h.active_offset():
        raise Alarm("cycle_no_tool_offset", "no D active")
    radius = h.comp_radius()
    if radius <= 0.0:
        raise Alarm("cycle_radius_negative" if radius < 0.0 else "cycle_no_tool_offset", f"tool radius {radius:g}")
    return radius


def feed_per_min(h, value: float, name: str) -> float:
    if value <= 0.0:
        raise Alarm("cycle_no_feed", name)
    return value * h.feed_unit_m() / 60.0


def pocket_plane(h, values) -> tuple[M.Plane, float, float, float, float]:
    a, b, n = PLANES[h.plane()]
    rtp = h.length_si(num(values, 0, "_RTP"))
    rfp = h.length_si(num(values, 1, "_RFP"))
    sign = 1.0 if rtp >= rfp else -1.0
    return M.Plane(a, b, n, sign), rtp, rfp, h.length_si(abs(num(values, 2, "_SDIS"))), sign


def plunge_mode(code: int) -> str:
    return {0: "vertical_rapid", 1: "vertical", 2: "helix", 3: "oscillate"}.get(code, "")


def pocket3(h, values) -> list:
    plane, rtp, rfp, sdis, sign = pocket_plane(h, values)
    depth = h.length_si(num(values, 3, "_DP"))
    length, width = h.length_si(num(values, 4, "_LENG")), h.length_si(num(values, 5, "_WID"))
    crad = h.length_si(num(values, 6, "_CRAD"))
    pa, po = h.length_si(num(values, 7, "_PA")), h.length_si(num(values, 8, "_PO"))
    angle = math.radians(num(values, 9, "_STA"))
    if length == 0.0 or width == 0.0:
        raise Alarm("cycle_length_width", "_LENG / _WID")
    centre = (pa, po)
    if length < 0.0 or width < 0.0:
        cu, su = math.cos(angle), math.sin(angle)
        du, dv = length / 2.0, width / 2.0
        centre = (pa + cu * du - su * dv, po + su * du + cu * dv)
    return pocket_common(h, values, plane, rtp, rfp, sdis, sign, depth, abs(length), abs(width), crad, centre, angle,
                         10, ap=(11, 12, 13, 14, 15, 16, 17), blank=(18, 19, 20), helix=(21, 22))


def pocket4(h, values) -> list:
    plane, rtp, rfp, sdis, sign = pocket_plane(h, values)
    depth = h.length_si(num(values, 3, "_DP"))
    radius = h.length_si(num(values, 4, "_PRAD"))
    if radius <= 0.0:
        raise Alarm("cycle_length_width", "_PRAD")
    centre = (h.length_si(num(values, 5, "_PA")), h.length_si(num(values, 6, "_PO")))
    return pocket_common(h, values, plane, rtp, rfp, sdis, sign, depth, 2.0 * radius, 2.0 * radius, radius, centre,
                         0.0, 7, ap=(8, 9, 10, 11, 12, 13, 14), blank=(15, None, 16), helix=(17, 18))


def pocket_common(h, values, plane, rtp, rfp, sdis, sign, depth, length, width, crad, centre, angle, mid_k, ap, blank,
                  helix) -> list:
    k_fal, k_fald, k_ffp1, k_ffd, k_cdir, k_vari, k_mida = ap
    mid = h.length_si(abs(num(values, mid_k, "_MID")))
    fal = h.length_si(num(values, k_fal, "_FAL"))
    fald = h.length_si(num(values, k_fald, "_FALD"))
    ffp1 = feed_per_min(h, num(values, k_ffp1, "_FFP1"), "_FFP1")
    ffd = feed_per_min(h, num(values, k_ffd, "_FFD"), "_FFD")
    direction = milling_direction(h, int(num(values, k_cdir, "_CDIR")))
    vari = int(num(values, k_vari, "_VARI"))
    machining, insertion = vari % 10, (vari // 10) % 10
    if machining not in (1, 2):
        raise Alarm("cycle_machining_type", f"_VARI={vari}")
    mode = plunge_mode(insertion)
    if not mode:
        raise Alarm("cycle_parameter", f"_VARI={vari} (insertion)", "_VARI")
    radius = tool_radius(h)
    mida = h.length_si(num(values, k_mida, "_MIDA")) or float(h.p["pocket_stepover_fraction"]) * 2.0 * radius
    if mida > 2.0 * radius:
        raise Alarm("cycle_infeed_width", "_MIDA larger than the tool diameter")
    if (rfp - depth) * sign <= 0.0:
        raise Alarm("cycle_reference_plane", "pocket depth above the reference plane")
    if fald > abs(rfp - depth):
        raise Alarm("cycle_bottom_allowance", "_FALD")
    if crad and crad < 0.0:
        raise Alarm("cycle_crad", "_CRAD")
    swap = width > length
    shape = M.RoundedRect(max(length, width), min(length, width), crad)
    place = M.Placement(centre, angle + (math.pi / 2.0 if swap else 0.0))
    ap1_k, ap2_k, ad_k = blank
    blank_offset = None
    ap1 = h.length_si(num(values, ap1_k, "_AP1")) if ap1_k is not None else 0.0
    ap2 = h.length_si(num(values, ap2_k, "_AP2")) if ap2_k is not None else ap1
    ad = h.length_si(num(values, ad_k, "_AD")) if ad_k is not None else 0.0
    if ap1 > 0.0 and ap2 > 0.0:
        blank_offset = (min(length, width) - min(ap1, ap2)) / 2.0 + radius
    hr = h.length_si(num(values, helix[0], "_RAD1"))
    dp1 = h.length_si(num(values, helix[1], "_DP1"))
    job = M.PocketJob(plane=plane, place=place, shape=shape, tool_radius=radius, top=rfp, bottom=depth, safety=sdis,
                      retract=rtp, max_depth_step=mid, stepover=mida, side_allowance=fal, bottom_allowance=fald,
                      feed_plane=ffp1, feed_depth=ffd, feed_finish=ffp1, direction=direction, plunge=mode,
                      helix_radius=hr, helix_pitch=dp1, ramp_angle=math.radians(num(values, helix[0], "_RAD1")),
                      rough=machining == 1, finish_side=machining == 2, finish_bottom=machining == 2 and fald > 0.0,
                      finish_depth_step=mid, blank_offset=blank_offset,
                      blank_depth=(rfp - sign * ad) if ad > 0.0 else None, plunge_point=place.map((0.0, 0.0)))
    if mode == "helix" and (hr <= 0.0 or dp1 <= 0.0):
        raise Alarm("cycle_helix", "helical insertion needs _RAD1 > 0 and _DP1 > 0")
    if mode == "oscillate":
        job.plunge_point = place.map(shape.start(shape.inner()))
        job.helix_radius = 0.0
    return M.pocket_steps(job)


def slots(h, name: str, values) -> list:
    a, b, n = PLANES[h.plane()]
    rtp, rfp = h.length_si(num(values, 0, "RTP")), h.length_si(num(values, 1, "RFP"))
    sign = 1.0 if rtp >= rfp else -1.0
    plane = M.Plane(a, b, n, sign)
    sdis = h.length_si(abs(num(values, 2, "SDIS")))
    dpr = num(values, 4, "DPR")
    depth = rfp - sign * h.length_si(abs(dpr)) if dpr else h.length_si(num(values, 3, "DP"))
    count = int(round_half_away(num(values, 5, "NUM")))
    if count <= 0:
        raise Alarm("cycle_holes_zero", "NUM")
    size = h.length_si(num(values, 6, "LENG" if name == "SLOT1" else "AFSL")) if name == "SLOT1" else \
        num(values, 6, "AFSL")
    width = h.length_si(num(values, 7, "WID"))
    centre = (h.length_si(num(values, 8, "CPA")), h.length_si(num(values, 9, "CPO")))
    rad = h.length_si(num(values, 10, "RAD"))
    sta1 = num(values, 11, "STA1")
    inda = num(values, 12, "INDA") or 360.0 / count
    ffd = feed_per_min(h, num(values, 13, "FFD"), "FFD")
    ffp1 = feed_per_min(h, num(values, 14, "FFP1"), "FFP1")
    mid = h.length_si(abs(num(values, 15, "MID")))
    direction = milling_direction(h, int(num(values, 16, "CDIR")))
    fal = h.length_si(num(values, 17, "FAL"))
    vari = int(num(values, 18, "VARI")) % 10
    midf = h.length_si(abs(num(values, 19, "MIDF"))) or mid
    ffp2 = feed_per_min(h, num(values, 20, "FFP2") or num(values, 14, "FFP1"), "FFP2")
    ssf = num(values, 21, "SSF")
    fald = h.length_si(num(values, 22, "_FALD")) if name == "SLOT1" else 0.0
    if vari not in (0, 1, 2):
        raise Alarm("cycle_machining_type", f"VARI={vari}")
    radius = tool_radius(h)
    if 2.0 * radius > width + 1e-12:
        raise Alarm("cycle_tool_radius", "tool wider than the slot")
    if width <= 0.0 or size <= 0.0:
        raise Alarm("cycle_length_width", "LENG/AFSL, WID")
    out = []
    for k in range(count):
        angle = math.radians(sta1 + k * inda)
        if name == "SLOT1":
            if size < width:
                raise Alarm("cycle_slot_violation", "slot shorter than its width")
            mid_point = (centre[0] + rad * math.cos(angle), centre[1] + rad * math.sin(angle))
            shape = M.RoundedRect(size, width, width / 2.0)
            place = M.Placement(mid_point, angle)
        else:
            shape = M.ArcSlot(rad, angle, math.radians(size), width)
            place = M.Placement(centre, 0.0)
        stepover = float(h.p["pocket_stepover_fraction"]) * 2.0 * radius
        if vari in (0, 1):
            job = M.PocketJob(plane=plane, place=place, shape=shape, tool_radius=radius, top=rfp, bottom=depth,
                              safety=sdis, retract=rtp, max_depth_step=mid, stepover=stepover,
                              side_allowance=fal if vari in (0, 1) else 0.0, bottom_allowance=fald,
                              feed_plane=ffp1, feed_depth=ffd, feed_finish=ffp2, direction=direction,
                              plunge="vertical", rough=True, finish_side=False, finish_bottom=False)
            out += M.pocket_steps(job)
        if vari in (0, 2):
            if ssf > 0.0:
                out.append(spindle_step(h.s.spindle.direction or -1, ssf))
            job = M.PocketJob(plane=plane, place=place, shape=shape, tool_radius=radius, top=rfp, bottom=depth,
                              safety=sdis, retract=rtp, max_depth_step=midf, stepover=stepover, side_allowance=0.0,
                              bottom_allowance=0.0, feed_plane=ffp2, feed_depth=ffd, feed_finish=ffp2,
                              direction=direction, plunge="vertical", rough=False, finish_side=True,
                              finish_bottom=False, finish_depth_step=midf)
            out += M.pocket_steps(job)
    return out


# ====================================================================== swivel plane and high-speed settings

def cycle800(h, values) -> list:
    """Swivel plane on a machine with two rotary table axes: the swivel frame x_parent = X0 + R (X1 + x)
    (rotation by _MODE: axis by axis in the order of the mode bits, solid angles, projection angles, or
    rotary axes direct), the rotary axes positioned to the solution that makes the new Z the tool axis
    (_DIR: -1 the smaller, +1 the larger value of the first rotary axis, 0 frame only; in direct mode the
    programmed positions), retraction _FR before swivelling. _ST units digit 1 adds to the active swivel
    frame (not in direct mode, whose positions fix the orientation). _TC "0" deselects."""
    ex, s = h.ex, h.s
    fr = int(num(values, 0, "_FR"))
    tc = arg(values, 1, "")
    st = int(num(values, 2, "_ST"))
    mode = int(num(values, 3, "_MODE"))
    x0 = [h.length_si(num(values, k, n)) for k, n in ((4, "_X0"), (5, "_Y0"), (6, "_Z0"))]
    angles = [num(values, k, n) for k, n in ((7, "_A"), (8, "_B"), (9, "_C"))]
    x1 = [h.length_si(num(values, k, n)) for k, n in ((10, "_X1"), (11, "_Y1"), (12, "_Z1"))]
    direction = int(num(values, 13, "_DIR", -1.0)) if len(values) > 13 and values[13] is not None else -1
    fr_i = h.length_si(num(values, 14, "_FR_I"))
    if fr not in (0, 1, 2, 4, 5):
        raise Alarm("tilt_mode", f"_FR={fr}")
    out = retract_steps(h, fr, fr_i)
    if str(tc).strip() == "0":
        s.sys.pop("swivel", None)
        out.append(M.action(ACTION, action="frames", tcp="" if s.tcp == "siemens840d:CYCLE800" else s.tcp))
        return out
    direct = (mode >> 6) & 3 == 3
    if direct:
        rotation, solution = direct_rotation(h, angles)
    else:
        rotation = swivel_rotation(mode, angles)
    previous = s.sys.get("swivel")
    if st % 10 == 1 and previous is not None and not direct:
        import numpy as np

        r_prev = np.asarray(previous["R"], dtype=float).reshape(3, 3)
        rotation = r_prev @ rotation
        x0 = list(np.asarray(previous["X0"]) + r_prev @ (np.asarray(previous["X1"]) + np.asarray(x0)))
    import numpy as np

    normal = rotation @ np.array([0.0, 0.0, 1.0])
    s.sys["swivel"] = {"R": clean_matrix(rotation), "X0": [float(v) for v in x0], "X1": [float(v) for v in x1]}
    tcp = s.tcp if s.tcp == "siemens840d:TRAORI" else "siemens840d:CYCLE800"
    out.append(M.action(ACTION, action="frames", tcp=tcp))
    if direction != 0:
        if not direct:
            solution = h.choose_table_solution(normal, master_sign=1 if direction > 0 else -1)
        out.append(swivel_step(h, solution))
    return out


def swivel_step(h, solution: dict):
    """The rotary axes to the swivel solution in machine coordinates (the linear axes stay)."""
    from ...hooks import Step

    target = {a.letter: float(solution[a.name]) for a in h.ex.axes if a.name in solution}
    return Step("rapid", target, machine=True)


def retract_steps(h, fr: int, fr_i: float) -> list:
    from ...hooks import Step

    ex = h.ex
    axis = ex.by_letter.get("Z")
    if fr == 0 or axis is None:
        return []
    if fr == 5:
        if fr_i <= 0.0:
            raise Alarm("tilt_retract", "_FR_I", "incremental retraction 0")
        return [Step("rapid", {"Z": ex.si("Z", ex.s.mach["Z"]) + fr_i}, machine=True)]
    top = axis.travel[1] if axis.travel is not None else axis.home * axis.quantum
    out = [Step("rapid", {"Z": top}, machine=True)]
    if fr == 2:
        home = {a.letter: a.home * a.quantum for a in ex.axes if a.letter in ("X", "Y")}
        out.append(Step("rapid", home, machine=True))
    return out


def swivel_rotation(mode: int, angles: list[float]):
    """The swivel frame's rotation for _MODE bits 7-6 = 00 axis by axis (bits 1-0, 3-2, 5-4: first, second,
    third rotation axis, 01 X 10 Y 11 Z, each about the already rotated axes; 57 = X, Y, Z), 01 solid angle
    (R_z(_A) R_y(_B) R_z(_C)) and 10 projection angles (bits 1-0 / 3-2: X and Y in either order, then _C
    about the new Z): ``cnctwin.geom.transform.siemens_cycle800`` (docs/models/geom.md 2.1). Mode 11 (rotary
    axes direct) depends on the machine: :func:`direct_rotation`. _MODE 0 (no axis sequence, the default of
    CYCLE800()) defines only the zero plane."""
    import numpy as np

    from ....geom import transform as T

    if (mode >> 6) & 3 == 0 and mode & 0x3F == 0:
        # no axis sequence: only the zero plane (CYCLE800() with its defaults) is defined
        if any(angles):
            raise Alarm("cycle_parameter", f"_MODE={mode}: axis by axis without the axis sequence", "_MODE")
        return np.eye(3)
    try:
        pose = T.siemens_cycle800(int(mode), *(math.radians(a) for a in angles))
    except ValueError as exc:
        raise Alarm("cycle_parameter", str(exc), "_MODE") from None
    return pose[:3, :3]


def direct_rotation(h, angles: list[float]):
    """_MODE 11 (rotary axes direct): _A and _B are the positions of rotary axes 1 and 2 (the base-side and
    the part-side table axis of the machine's chain). The table turns the part by R_t = R_1(_A) R_2(_B); the
    swivel frame is the one whose axes stay on the machine axes, R = R_t^T B (B turns Z onto the machine's
    tool axis, the identity on a vertical spindle). Returns the frame rotation and the axis positions."""
    import numpy as np

    machine = h.machine
    links = machine.work_rotaries
    if len(links) != 2 or machine.tool_rotaries:
        raise Alarm("tilt_direct", "_MODE rotary axes direct", "no two rotary table axes")
    raw = {links[0].axis: math.radians(angles[0]), links[1].axis: math.radians(angles[1])}
    solution = {}
    for name, value in raw.items():
        placed = h.place_rotary(name, value)
        if placed is None:
            raise Alarm("tilt_rotary_positions", f"rotary axis {name} {math.degrees(value):g} deg outside its travel")
        solution[name] = placed
    table = machine.table_transform(solution)[:3, :3]
    tool = np.asarray(machine.tool_axis, dtype=float)
    tool = tool / np.linalg.norm(tool)
    z = np.array([0.0, 0.0, 1.0])
    cos_angle = float(z @ tool)
    if cos_angle > 1.0 - 1e-15:
        base = np.eye(3)
    elif cos_angle < -1.0 + 1e-15:
        base = rot_x(math.pi)
    else:
        from ....geom import transform as T

        base = T.axis_angle(np.cross(z, tool), math.acos(max(-1.0, min(1.0, cos_angle))))
    return table.T @ base, solution


def cycle832(h, values) -> list:
    ex, s = h.ex, h.s
    tol = h.length_si(num(values, 0, "_TOL")) if values else 0.0
    tolm = int(num(values, 1, "_TOLM")) if len(values) > 1 else 0
    level = tolm % 10
    names = {0: "_OFF", 1: "_FINISH", 2: "_SEMIFIN", 3: "_ROUGH"}
    if level not in names:
        raise Alarm("cycle_parameter", f"_TOLM={tolm}", "_TOLM")
    code = f"CYCLE832({num(values, 0, '_TOL') if values else 0:g},{names[level]})"
    if level == 0:
        s.modal["exact"] = "G64"
        s.modal["compressor"] = "COMPOF"
        s.smoothing.update({"corner": "blend", "tolerance_m": 0.0, "mode_code": ""})
    else:
        if tol <= 0.0:
            table = {entry["name"]: entry["tolerance_m"] for entry in h.machine.smoothing}
            tol = float(table.get({1: "finish", 2: "medium", 3: "rough"}[level], 0.0))
        s.modal.update({"exact": "G642", "compressor": "COMPCAD", "accel": "SOFT", "feedforward": "FFWON"})
        s.smoothing.update({"corner": "blend", "tolerance_m": tol, "mode_code": code})
    ex.emit(Command(Kind.MODE, ex.line_ref(), smoothing=ex.smoothing_spec(),
                    modal={"cycle832": {"tolerance_m": s.smoothing["tolerance_m"], "mode": names[level]},
                           "accel": s.modal.get("accel"), "feedforward": s.modal.get("feedforward"),
                           "compressor": s.modal.get("compressor")}))
    return []


# ====================================================================== dispatch

def expansion(h, name: str, values: list) -> list:
    """The steps of one cycle call."""
    if name in TURNING:
        from . import turning

        steps = turning.run(h, name, values)
    elif name in ("HOLES1", "HOLES2"):
        steps = holes(h, name, values)
    elif name in ("SLOT1", "SLOT2"):
        steps = slots(h, name, values)
    else:
        steps = CYCLES[name](h, values)
    h.check_steps_scale(steps, "cycle_scaling")
    h.s.sys["cycle_name"] = name
    return steps


def run(h, name: str, values: list) -> None:
    """Execute one cycle call: the expansion is queued as steps after the current block's commands."""
    steps = expansion(h, name, values)
    if steps:
        h.ex.queue_steps(steps)


CYCLES = {"CYCLE81": cycle81, "CYCLE82": cycle82, "CYCLE83": cycle83, "CYCLE84": cycle84, "CYCLE840": cycle840,
          "CYCLE85": cycle85, "CYCLE86": cycle86, "CYCLE87": cycle87, "CYCLE88": cycle88, "CYCLE89": cycle89,
          "HOLES1": None, "HOLES2": None, "POCKET3": pocket3, "POCKET4": pocket4, "SLOT1": None, "SLOT2": None,
          "CYCLE800": cycle800, "CYCLE832": cycle832, "CYCLE95": None, "CYCLE97": None, "CYCLE93": None}
TURNING = ("CYCLE95", "CYCLE97", "CYCLE93")

__all__ = ["CYCLES", "run", "expansion", "Action", "ACTION", "ISO_COARSE_PITCH_MM", "swivel_rotation", "direct_rotation"]
