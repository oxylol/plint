"""5-axis flank-milling program for the ruled blade (blade.nc), computed from part.step.

Run: cd apps/cnc-twin/examples/blade-5axis-ti && ../../bin/tsafe python make_program.py

1. The blade side is the one B-spline face of part.step; it is ruled (degree 1 across the span), so its
   isolines across the span are the rulings. For N parameter values around the section the ruling end
   points at the hub (z 0) and the tip (z 20) and the surface normals there are read from the surface.
2. Tool axis (two-point offset, Bedi et al.): through A = P_hub + (R + s) n_hub and B = P_tip + (R + s) n_tip,
   s the stock to leave, then moved outwards along the mid-span normal by half of the mid-span overcut
   (the twist error of a non-developable ruled surface is split: +e/2 at the ends, -e/2 at mid-span). Tool tip on that axis L0 = 1.2 mm below A, so the cylindrical part of the flute
   (above the R1 corner) covers the ruling from the hub up; the lowest point of the tool stays above the
   platform (z -7).
3. Points are resampled along the tip path (<= 0.3 mm and <= 0.5 deg of axis change per block) and ordered
   clockwise seen from the tool (climb with M03).
4. B and C for the Haas UMC-750 trunnion (machine file kinematics: B about -y through the pivot, C about
   -z): R_B(B) R_C(C) a = +z gives C = atan2(a_y, a_x), B = atan2(sqrt(a_x^2 + a_y^2), a_z).
5. Checks: machine travel with the G234 transform of docs/dialects.md (work offset at rotary zero, axis
   points from the kinematic chains), rotary rates at the programmed feed, the contact band on the flute,
   and the deviation of the swept cylinder from the ruled surface (envelope over neighbouring positions).

Writes blade.nc and toolpath_stats.json (the numbers EXPECTED.md quotes).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_BSplineSurface
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.gp import gp_Pnt, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

HERE = Path(__file__).resolve().parent
R_TOOL, R_CORNER, LOC = 6.0, 1.0, 24.0            # helical-59487-mhsv-020-41200-r100
L0 = 1.2                                          # tool tip below the hub offset point, along the axis
Z_HUB, Z_TIP, Z_PLATFORM = 0.0, 20.0, -7.0
TOOL_LENGTH = 115.0                               # gauge 70 (haimer-40-720-32) + stickout 45
SPINDLE = 1455                                    # vc 54.9 m/min (Helical start value, D12)
FEED = {"rough": 414, "finish": 325, "spring": 325}   # fz 0.0711 (Rgh) and 0.0559 (Fin) x 4 x 1455
PASSES = [("T1", 1, "rough", 1.0), ("T1", 1, "rough", 0.3), ("T2", 2, "finish", 0.0), ("T2", 2, "spring", 0.0)]
G54 = np.array([-330.0, -254.0, -494.0])          # part zero in machine coordinates at B0 C0
PIVOT_B, PIVOT_C = np.array([-330.0, -254.0, -661.0]), np.array([-330.0, -254.0, -610.0])
TRAVEL = {"X": (-762.0, 0.0), "Y": (-508.0, 0.0), "Z": (-508.0, 0.0), "B": (-35.0, 120.0)}
ROTARY_MAX_DEG_S = 50.0
MAX_STEP_MM, MAX_ANGLE_DEG = 0.3, 0.5
APPROACH_MM, LEAD_MM = 35.0, 3.0


def read_blade_face(path: Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise SystemExit(f"cannot read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_BSplineSurface:
            faces.append(face)
        explorer.Next()
    if len(faces) != 1:
        raise SystemExit(f"expected one B-spline (ruled) face, found {len(faces)}")
    return faces[0]


class RuledFace:
    """The blade face: rulings along the degree-1 direction, outward normals."""

    def __init__(self, face):
        self.face = face
        self.surface = BRep_Tool.Surface_s(face)
        self.reversed = face.Orientation() == TopAbs_REVERSED
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
        bs = BRepAdaptor_Surface(face).BSpline()
        self.span_is_v = bs.VDegree() == 1
        if not self.span_is_v and bs.UDegree() != 1:
            raise SystemExit("the blade face is not ruled (no degree-1 direction)")
        self.around = (u0, u1) if self.span_is_v else (v0, v1)
        self.span = (v0, v1) if self.span_is_v else (u0, u1)

    def _uv(self, t, w):
        return (t, w) if self.span_is_v else (w, t)

    def point(self, t, w):
        p = self.surface.Value(*self._uv(t, w))
        return np.array([p.X(), p.Y(), p.Z()])

    def normal(self, t, w):
        p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        self.surface.D1(*self._uv(t, w), p, du, dv)
        n = np.cross([du.X(), du.Y(), du.Z()], [dv.X(), dv.Y(), dv.Z()])
        n /= np.linalg.norm(n)
        return -n if self.reversed else n

    def span_param(self, t, z):
        """Span parameter of height z on the ruling at t (linear along the ruling)."""
        w0, w1 = self.span
        z0, z1 = self.point(t, w0)[2], self.point(t, w1)[2]
        return w0 + (z - z0) / (z1 - z0) * (w1 - w0)


def b_c_of_axis(a):
    c = math.degrees(math.atan2(a[1], a[0]))
    b = math.degrees(math.atan2(math.hypot(a[0], a[1]), a[2]))
    return b, c


def rot(axis, angle_deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def machine_axes(tip, b, c):
    """X Y Z of the UMC-750 for a program tip point (part frame) at B/C under G234 (docs/dialects.md)."""
    p0 = G54 + tip
    p = PIVOT_C + rot([0, 0, -1], c) @ (p0 - PIVOT_C)
    p = PIVOT_B + rot([0, -1, 0], b) @ (p - PIVOT_B)
    return p + np.array([0.0, 0.0, TOOL_LENGTH])


def sample(face: RuledFace, n=6000):
    t0, t1 = face.around
    ts = np.linspace(t0, t1, n, endpoint=False)
    rows = []
    for t in ts:
        wh, wt = face.span_param(t, Z_HUB), face.span_param(t, Z_TIP)
        rows.append((t, face.point(t, wh), face.point(t, wt), face.normal(t, wh), face.normal(t, wt)))
    hub = np.array([r[1] for r in rows])
    area = 0.5 * np.sum(hub[:, 0] * np.roll(hub[:, 1], -1) - np.roll(hub[:, 0], -1) * hub[:, 1])
    # outward normals: the in-plane part must point away from the section interior
    centre = hub[:, :2].mean(axis=0)
    flip = np.sign(np.mean([np.dot(r[3][:2], r[1][:2] - centre) for r in rows]))
    rows = [(t, ph, pt, flip * nh, flip * nt) for t, ph, pt, nh, nt in rows]
    if area > 0:                                    # counter-clockwise seen from +z: reverse for climb (CW)
        rows = rows[::-1]
    return rows


def tool_positions(rows, stock, balance=True):
    """Two-point offset; with ``balance`` the axis is moved outwards along the mid-span normal by half of
    the mid-span overcut, so the twist error is split between the ends (undercut) and the middle."""
    out = []
    for t, ph, pt, nh, nt in rows:
        a_pt = ph + (R_TOOL + stock) * nh
        b_pt = pt + (R_TOOL + stock) * nt
        axis = (b_pt - a_pt) / np.linalg.norm(b_pt - a_pt)
        if balance:
            mid = 0.5 * (ph + pt)
            d = mid - a_pt
            error = np.linalg.norm(d - np.dot(d, axis) * axis) - (R_TOOL + stock)    # < 0: overcut
            if error < 0.0:
                n_mid = 0.5 * (nh + nt)
                n_mid /= np.linalg.norm(n_mid)
                a_pt = a_pt - 0.5 * error * n_mid
                b_pt = b_pt - 0.5 * error * n_mid
        out.append((a_pt - L0 * axis, axis, a_pt, b_pt))
    return out


def resample(positions):
    """Keep positions so that consecutive blocks differ by <= MAX_STEP_MM at the tip and <= MAX_ANGLE_DEG."""
    keep = [positions[0]]
    for p in positions[1:]:
        last = keep[-1]
        step = np.linalg.norm(p[0] - last[0])
        ang = math.degrees(math.acos(min(1.0, float(np.dot(p[1], last[1])))))
        if step >= MAX_STEP_MM or ang >= MAX_ANGLE_DEG:
            keep.append(p)
    keep.append(positions[0])                       # close the loop
    return keep


def deviation(rows, positions, window=60, heights=np.linspace(Z_HUB, Z_TIP, 21)):
    """Signed distance of the ruled surface from the swept cylinder: min over nearby tool positions of
    (distance of the surface point from the tool axis - R). > 0 material left, < 0 overcut."""
    axes = np.array([p[1] for p in positions])
    starts = np.array([p[2] for p in positions])
    n = len(rows)
    worst_over, worst_under = 0.0, 0.0
    per_height = []
    for z in heights:
        devs = []
        for i in range(0, n, 10):
            _, ph, pt, _, _ = rows[i]
            q = ph + (z - Z_HUB) / (Z_TIP - Z_HUB) * (pt - ph)
            idx = np.arange(i - window, i + window + 1) % n
            d = q - starts[idx]
            along = np.einsum("ij,ij->i", d, axes[idx])
            perp = np.linalg.norm(d - along[:, None] * axes[idx], axis=1)
            devs.append(float(np.min(perp)) - R_TOOL)
        devs = np.array(devs)
        per_height.append((float(z), float(devs.min()), float(devs.max())))
        worst_over = min(worst_over, float(devs.min()))
        worst_under = max(worst_under, float(devs.max()))
    return worst_over, worst_under, per_height


def contact_band(positions, rows):
    """Distance along the tool axis from the tip to the contact at the hub and at the tip of the ruling."""
    lo, hi = [], []
    for (tip, axis, a_pt, b_pt), (_, ph, pt, _, _) in zip(positions, rows):
        lo.append(float(np.dot(ph - tip, axis)))
        hi.append(float(np.dot(pt - tip, axis)))
    return min(lo), max(hi)


def lowest_point(tip, axis):
    """Lowest z of the tool end (flat end and R1 corner) for an axis tilted from +z."""
    tilt = math.acos(max(-1.0, min(1.0, axis[2])))
    return tip[2] - (R_TOOL - R_CORNER) * math.sin(tilt) - R_CORNER * (1 - math.cos(tilt)) + 0.0


def fmt(v, nd=3):
    s = f"{v:.{nd}f}"
    return "0." if float(s) == 0 else s.rstrip("0") if "." in s else s


def block(tip, b, c):
    return f"X{fmt(tip[0])} Y{fmt(tip[1])} Z{fmt(tip[2])} B{fmt(b)} C{fmt(c)}"


def main():
    face = RuledFace(read_blade_face(HERE / "part.step"))
    rows = sample(face)
    lines = ["%", "O01008 (BLADE-5AXIS-TI)",
             "(CNC TWIN EXAMPLE BLADE-5AXIS-TI - WRITTEN BY MAKE_PROGRAM.PY FROM PART.STEP)",
             "(HAAS UMC-750 NGC - G234 TCPC - B/C TRUNNION)",
             "(STOCK TI-6AL-4V BLANK - FIN 2.0 MM OVER THE BLADE - PLATFORM 60 X 32 X 20)",
             "(SETUP LANG MAKRO-GRIP 48085-46 ON QUICK-POINT 45600 ON THE PLATTER - GRIP 3 MM)",
             "(G54 PART ZERO ON THE STACKING AXIS AT THE HUB SECTION - MACHINE COORDINATES AT B0 C0)",
             "(T1 D12 R1 4FL CARBIDE HELICAL 59487 MHSV-020-41200-R1.00 - FLANK ROUGH, 2 PASSES)",
             "(T2 D12 R1 4FL CARBIDE HELICAL 59487 MHSV-020-41200-R1.00 - FLANK FINISH + SPRING PASS)",
             f"(VC 55 S{SPINDLE} - ROUGH FZ 0.0711 F{FEED['rough']} - FINISH FZ 0.0559 F{FEED['finish']})",
             "(TOOL AXIS ALONG THE RULINGS - TWO-POINT OFFSET - CLIMB - TIP FEED UNDER G234)",
             "G90 G17 G21 G40 G49 G80 G94 G54",
             "G53 G00 Z0.",
             "G00 B0. C0."]
    stats = {"passes": [], "travel": {}, "rotary": {}}
    all_axes = {"X": [], "Y": [], "Z": [], "B": [], "C": []}
    current_tool = None
    total_len = {"rough": 0.0, "finish": 0.0, "spring": 0.0}
    max_rate = 0.0
    for name, number, kind, stock in PASSES:
        positions = resample(tool_positions(rows, stock))
        bc = [b_c_of_axis(p[1]) for p in positions]
        cs = np.unwrap(np.radians([c for _, c in bc]))
        bc = [(b, math.degrees(c)) for (b, _), c in zip(bc, cs)]
        if number != current_tool:
            if current_tool is not None:
                lines += ["G49 (TCPC OFF)", "M09", "G53 G00 Z0.", "G00 B0. C0."]
            label = "FLANK ROUGH" if kind == "rough" else "FLANK FINISH"
            lines += [f"N{number * 10} ({name} D12 R1 4FL - {label})",
                      f"{name} M06 (D12 R1 4FL CARBIDE HELICAL 59487)",
                      "G187 P1 (ROUGH SMOOTHING)" if kind == "rough" else "G187 P3 (FINISH SMOOTHING)",
                      f"G90 G43 H{number:02d} Z60. (LENGTH COMPENSATION AT THE CHANGE POSITION, G234 TAKES OVER BELOW)",
                      f"S{SPINDLE} M03"]
            current_tool = number
        tip0, axis0 = positions[0][0], positions[0][1]
        b0, c0 = bc[0]
        normal0 = rows[0][3]
        lead = tip0 + LEAD_MM * np.array([normal0[0], normal0[1], 0.0])
        above = lead + APPROACH_MM * axis0
        title = "SPRING PASS - SAME PATH, TAKES THE DEFLECTION LEFT BY THE FINISH PASS" if kind == "spring" \
            else f"PASS {kind.upper()} - STOCK {stock:.1f}"
        lines += [f"({title})",
                  f"G00 G90 G54 B{fmt(b0)} C{fmt(c0)}",
                  f"G234 H{number:02d} {block(above, b0, c0)}",
                  "M08",
                  f"G01 {block(lead, b0, c0)} F2000.",
                  f"{block(tip0, b0, c0)} F{FEED[kind]}."]
        length = 0.0
        prev_tip, prev_bc = tip0, (b0, c0)
        for (tip, axis, _, _), (b, c) in zip(positions[1:], bc[1:]):
            lines.append(block(tip, b, c))
            seg = float(np.linalg.norm(tip - prev_tip))
            length += seg
            if seg > 1e-9:
                dt = seg / (FEED[kind] / 60.0)
                max_rate = max(max_rate, abs(b - prev_bc[0]) / dt, abs(c - prev_bc[1]) / dt)
            prev_tip, prev_bc = tip, (b, c)
        lines += [f"{block(lead, bc[-1][0], bc[-1][1])}",
                  f"G01 {block(lead + APPROACH_MM * positions[-1][1], bc[-1][0], bc[-1][1])} F2000."]
        total_len[kind] += length + LEAD_MM
        for (tip, axis, _, _), (b, c) in zip(positions, bc):
            m = machine_axes(tip, b, c)
            for k, v in zip("XYZ", m):
                all_axes[k].append(float(v))
            all_axes["B"].append(b)
            all_axes["C"].append(c)
        over, under, per_height = deviation(rows, [p for p in tool_positions(rows, stock)])
        band = contact_band(tool_positions(rows, stock), rows)
        low = min(lowest_point(p[0], p[1]) for p in tool_positions(rows, stock))
        stats["passes"].append({"tool": name, "kind": kind, "stock_mm": stock, "blocks": len(positions),
                                "tip_path_mm": round(length, 2), "B_deg": [round(min(b for b, _ in bc), 2),
                                                                           round(max(b for b, _ in bc), 2)],
                                "C_deg": [round(min(c for _, c in bc), 2), round(max(c for _, c in bc), 2)],
                                "deviation_mm": {"most_overcut": round(over, 4), "most_undercut": round(under, 4),
                                                 "per_height": [(z, round(a, 4), round(b, 4)) for z, a, b in per_height]},
                                "contact_band_on_flute_mm": [round(band[0], 3), round(band[1], 3)],
                                "lowest_tool_point_z_mm": round(low, 3)})
    lines += ["G49 (TCPC OFF)", "M09", "G53 G00 Z0.", "G00 B0. C0.", "M05", "M30", "%"]
    (HERE / "blade.nc").write_text("\n".join(lines) + "\n")
    for k in "XYZB":
        lo, hi = min(all_axes[k]), max(all_axes[k])
        stats["travel"][k] = [round(lo, 3), round(hi, 3)]
        if k in TRAVEL and not (TRAVEL[k][0] <= lo and hi <= TRAVEL[k][1]):
            raise SystemExit(f"axis {k} leaves its travel: {lo:.1f} .. {hi:.1f}")
    stats["travel"]["C"] = [round(min(all_axes["C"]), 3), round(max(all_axes["C"]), 3)]
    stats["rotary"]["max_deg_s_at_programmed_feed"] = round(max_rate, 2)
    if max_rate > ROTARY_MAX_DEG_S:
        raise SystemExit(f"rotary rate {max_rate:.1f} deg/s exceeds {ROTARY_MAX_DEG_S}")
    stats["tip_path_mm"] = {k: round(v, 2) for k, v in total_len.items()}
    stats["feed_time_s"] = {k: round(total_len[k] / FEED[k] * 60.0, 2) for k in total_len}
    (HERE / "toolpath_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "passes"}, indent=1))
    for p in stats["passes"]:
        print({k: v for k, v in p.items() if k != "deviation_mm"}, "deviation", p["deviation_mm"]["most_overcut"],
              p["deviation_mm"]["most_undercut"])


if __name__ == "__main__":
    main()
