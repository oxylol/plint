"""Writes op10.nc (Haas NGC) for the pocket-with-islands part and prints the engagement statistics of
the trochoidal roughing (2-D material map, 0.05 mm pixels).

Run: cd apps/cnc-twin && bin/tsafe python examples/pocket-islands-al6061/make_program.py

Program coordinates: G54 X0 Y0 = lower-left corner of the stock, Z0 = stock top (set by the
probe, N10). Stock frame z = program Z + 30.

Roughing (T1, Helical 59766 D10 3FL): every channel between the pocket wall and the islands is
16 mm wide, so the tool centre runs along the channel centre lines (nodes A(16,16) B(50,16)
C(84,16) D(84,64) E(50,64) F(16,64), Euler path B-C-D-E-F-A-B-E) on counter-clockwise trochoidal
loops (climb with M03) of radius 2.7 mm advancing 1.0 mm per loop: the swept width
10 + 2 x 2.7 = 15.4 mm leaves 0.3 mm on every wall. Each loop is two exact semicircles (G03 with
I/J). Corners are 90 deg counter-clockwise transition arcs about the node. Full depth in one
step: Z-9.8 (0.2 mm left on the floor), helical entry at B (pitch 0.8 mm, ramp angle 2.7 deg).
Island tops: climb raster at Z-3.8, ae <= 3.5 mm (0.35 D; the maker's side-milling limit is 0.4 D).

Finishing (T2, Helical 59744 D12 3FL): floor along the same centre lines at Z-10 (slotting width
at ap 0.2), island tops at Z-4 (ae <= 4.0 mm), then the walls as tool-centre paths with wear
compensation G41 D02 (D02 = 0; climb: pocket counter-clockwise, islands clockwise), lead-in/out
arcs R2 (tool centre), ae 0.3, ap 10.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOP_Z_S = 30.0                       # stock top in the stock frame (program Z0)

# ----------------------------------------------------------------------------- cutting data
RPM = 12000                          # VF-2SS maximum; vc = 377 m/min (D10), 452 m/min (D12)
T1_FZ = 0.0965                       # Helical SF_H35AL-3, wrought Al, 3/8 in column, 'Rgh' (db side_milling start)
T2_FZ = 0.0686                       # Helical SF_H35AL-3, 1/2 in column, 'Slot' / 'Fin' start (D12 item)
T1_F = round(RPM * 3 * T1_FZ)        # 3474 mm/min
T2_F = round(RPM * 3 * T2_FZ)        # 2470 mm/min
HELIX_F = 1700                       # ~50 % of the roughing feed while ramping
T1_R = 5.0
T2_R = 6.0

# ----------------------------------------------------------------------------- geometry (program XY)
NODES = {"A": (16.0, 16.0), "B": (50.0, 16.0), "C": (84.0, 16.0), "D": (84.0, 64.0), "E": (50.0, 64.0),
         "F": (16.0, 64.0)}
EULER = ["B", "C", "D", "E", "F", "A", "B", "E"]
TROCHOID_R = 2.7
TROCHOID_STEP = 1.0
ROUGH_Z = -9.8
HELIX_PITCH = 0.8
ISLAND_ROUGH_Z = -3.8


def f3(v: float) -> str:
    """Haas metric word value: 3 decimals, trailing zeros removed, always a decimal point."""
    s = f"{v:.3f}".rstrip("0")
    if s in ("-0.", "-0"):
        s = "0."
    return s if not s.endswith(".") else s


class Path2:
    """Tool-centre path recorder for the engagement check (XY only, at the cutting depth)."""

    def __init__(self):
        self.moves = []                  # (kind, start, end, centre, ccw)
        self.pos = None

    def line(self, end):
        if self.pos is not None:
            self.moves.append(("line", self.pos, end, None, None))
        self.pos = end

    def arc(self, end, centre, ccw):
        self.moves.append(("arc", self.pos, end, centre, ccw))
        self.pos = end


def trochoid_blocks(path: Path2, lines: list[str]):
    """The Euler path on counter-clockwise loops; returns nothing, appends NC blocks."""
    r, s = TROCHOID_R, TROCHOID_STEP
    for k in range(len(EULER) - 1):
        p = np.array(NODES[EULER[k]])
        q = np.array(NODES[EULER[k + 1]])
        length = float(np.linalg.norm(q - p))
        u = (q - p) / length
        n = np.array([-u[1], u[0]])
        loops = int(round(length / s))
        if k > 0:
            # transition from the previous edge's last point (p - r n_prev) to this edge's start (p - r n)
            start = p - r * n
            lines.append(f"G03 X{f3(start[0])} Y{f3(start[1])} I{f3(p[0] - path.pos[0])} J{f3(p[1] - path.pos[1])}")
            path.arc(tuple(start), tuple(p), True)
        for i in range(loops):
            a = p + (i * s) * u - r * n
            t = p + ((i + 0.5) * s) * u + r * n
            b = p + ((i + 1) * s) * u - r * n
            c1 = 0.5 * (a + t)
            c2 = 0.5 * (t + b)
            lines.append(f"G03 X{f3(t[0])} Y{f3(t[1])} I{f3(c1[0] - a[0])} J{f3(c1[1] - a[1])}")
            path.arc(tuple(t), tuple(c1), True)
            lines.append(f"G03 X{f3(b[0])} Y{f3(b[1])} I{f3(c2[0] - t[0])} J{f3(c2[1] - t[1])}")
            path.arc(tuple(b), tuple(c2), True)


def rough_section(path: Path2) -> list[str]:
    bx, by = NODES["B"]
    start = (bx, by - TROCHOID_R)
    out = [
        "N20 (T1 D10 3FL CARBIDE ENDMILL - TROCHOIDAL ROUGHING, FULL DEPTH)",
        "T1 M06 (D10 3FL CARBIDE ENDMILL HELICAL 59766)",
        "G187 P1 (ROUGH SMOOTHING)",
        f"G90 G54 G00 X{f3(start[0])} Y{f3(start[1])} S{RPM} M03",
        "G43 H01 Z25. M08",
        "Z2.",
        "G01 Z0.5 F1000.",
        f"(HELICAL ENTRY AT B: RADIUS {TROCHOID_R:g}, PITCH {HELIX_PITCH:g}, DOWN TO Z{ROUGH_Z:g})",
    ]
    z = 0.5
    turns = math.ceil((z - ROUGH_Z) / HELIX_PITCH)
    dz = (z - ROUGH_Z) / turns
    for k in range(turns):
        z -= dz
        word = f" F{HELIX_F}." if k == 0 else ""
        out.append(f"G03 X{f3(start[0])} Y{f3(start[1])} Z{f3(z)} I0. J{f3(TROCHOID_R)}{word}")
    out.append(f"G03 X{f3(start[0])} Y{f3(start[1])} I0. J{f3(TROCHOID_R)} (CLEAN THE HELIX BOTTOM)")
    out.append(f"(TROCHOIDAL LOOPS R{TROCHOID_R:g} STEP {TROCHOID_STEP:g}, PATH B-C-D-E-F-A-B-E, FZ {T1_FZ:g})")
    path.pos = start
    body: list[str] = []
    trochoid_blocks(path, body)
    body[0] += f" F{T1_F}."
    out.extend(body)
    ex, ey = NODES[EULER[-1]]
    out += [f"G01 X{f3(ex)} Y{f3(ey)} (TO THE CHANNEL CENTRE AT E)", "G00 Z2."]
    path.line((ex, ey))
    out += island_rest_contours(path)
    out += island_rough(path)
    out += ["G00 Z25.", "M09", "M05", "G53 G49 Z0."]
    return out


def island_rest_contours(path: Path2) -> list[str]:
    """Full-depth contour pass around each island at the 0.3 mm allowance (tool centre 5.3 mm off the
    island), clockwise (climb): removes the wedges the channel trochoids leave at the island corners
    and ends (ae up to 3.85 mm at the obround ends, 0.39 D)."""
    o = T1_R + 0.3
    f = f"F{T1_F}."
    out = ["(ISLAND REST CONTOURS AT THE 0.3 ALLOWANCE, FULL DEPTH, CLOCKWISE = CLIMB)", "G00 X16. Y40.",
           f"G01 Z{f3(ROUGH_Z)} F2000."]
    x0, x1, y0, y1, r = 24.0 - o, 42.0 + o, 24.0 - o, 56.0 + o, 4.0 + o
    out += [f"G01 X{f3(x0)} Y40. {f}", f"Y{f3(52.0)}", f"G02 X28. Y{f3(y1)} R{f3(r)}", "G01 X38.",
            f"G02 X{f3(x1)} Y52. R{f3(r)}", "G01 Y28.", f"G02 X38. Y{f3(y0)} R{f3(r)}", "G01 X28.",
            f"G02 X{f3(x0)} Y28. R{f3(r)}", "G01 Y40.", "X16.", "G00 Z2."]
    path.pos = (16.0, 40.0)
    for kind, *args in (("l", (x0, 40.0)), ("l", (x0, 52.0)), ("a", (28.0, y1), (28.0, 52.0)), ("l", (38.0, y1)),
                        ("a", (x1, 52.0), (38.0, 52.0)), ("l", (x1, 28.0)), ("a", (38.0, y0), (38.0, 28.0)),
                        ("l", (28.0, y0)), ("a", (x0, 28.0), (28.0, 28.0)), ("l", (x0, 40.0)), ("l", (16.0, 40.0))):
        path.line(args[0]) if kind == "l" else path.arc(args[0], args[1], False)
    xa, xb, ro = 58.0 - o, 76.0 + o, 9.0 + o
    out += ["G00 X84. Y40.", f"G01 Z{f3(ROUGH_Z)} F2000.", f"G01 X{f3(xb)} {f}", "Y33.",
            f"G02 X{f3(xa)} Y33. I{f3(-ro)} J0.", "G01 Y47.", f"G02 X{f3(xb)} Y47. I{f3(ro)} J0.", "G01 Y40.", "X84.",
            "G00 Z2."]
    path.pos = (84.0, 40.0)
    for kind, *args in (("l", (xb, 40.0)), ("l", (xb, 33.0)), ("a", (xa, 33.0), (67.0, 33.0)), ("l", (xa, 47.0)),
                        ("a", (xb, 47.0), (67.0, 47.0)), ("l", (xb, 40.0)), ("l", (84.0, 40.0))):
        path.line(args[0]) if kind == "l" else path.arc(args[0], args[1], False)
    return out


def island_raster_x(x0: float, x1: float, radius: float, ae: float) -> list[float]:
    """Tool-centre x of climb raster passes (moving +Y, material on the +X side, stepping +X) that
    remove the block x0..x1 with at most ae per pass; the last pass runs 1 mm past x1."""
    first = x0 - radius + ae
    last = x1 - radius + 1.0
    n = math.ceil((last - first) / ae - 1e-9) + 1
    step = (last - first) / (n - 1)
    return [first + i * step for i in range(n)]


def island_rough(path: Path2) -> list[str]:
    out = [f"(ISLAND TOPS: RASTER AT Z{ISLAND_ROUGH_Z:g}, AE <= 3.5 = 0.35 D, CLIMB)"]
    for name, (x0, x1) in (("ISLAND 1", (23.7, 42.3)), ("ISLAND 2", (57.7, 76.3))):
        xs = island_raster_x(x0, x1, T1_R, 3.5)
        y0, y1 = 23.7 - T1_R - 1.0, 56.3 + T1_R + 1.0
        out.append(f"({name})")
        for x in xs:
            out += [f"G00 X{f3(x)} Y{f3(y0)}", f"G01 Z{f3(ISLAND_ROUGH_Z)} F1000.", f"Y{f3(y1)} F{T1_F}.", "G00 Z2."]
    return out


def floor_finish() -> list[str]:
    """Floor at Z-10 along the channel centre lines (a 12 mm slot at ap 0.2) with an excursion of 1.7 mm
    along the bisector into each pocket corner, so the corner floor is finished out to about the wall
    allowance; the 2 mm floor strips along the walls are taken by the wall passes."""
    f = f"F{T2_F}."
    bx, by = NODES["B"]
    out = ["(FLOOR Z-10.: CHANNEL CENTRE LINES, AP 0.2, D12 SLOT, INTO EACH CORNER ALONG THE BISECTOR)",
           f"G00 X{f3(bx)} Y{f3(by)}", "Z2.", "G01 Z-9.5 F2000.", "Z-10. F600."]
    corner = 8.0 - (T2_R + 0.3)                          # 1.7: tool edge on the 0.3 mm allowance at the R8 corners
    for name in EULER[1:]:
        x, y = NODES[name]
        out.append(f"G01 X{f3(x)} Y{f3(y)} {f}" if name == EULER[1] else f"G01 X{f3(x)} Y{f3(y)}")
        if name in ("A", "C", "D", "F"):
            dx = 1.0 if x > 50 else -1.0
            dy = 1.0 if y > 40 else -1.0
            k = corner / math.sqrt(2.0)
            px, py = x + dx * k, y + dy * k
            out += [f"G01 X{f3(px)} Y{f3(py)}", f"G01 X{f3(x)} Y{f3(y)}"]
    out.append("G00 Z2.")
    return out


def island_top_finish() -> list[str]:
    out = ["(ISLAND TOPS Z-4., CLIMB RASTER, AE <= 4.0 = 0.33 D)"]
    for name, (x0, x1) in (("ISLAND 1", (23.7, 42.3)), ("ISLAND 2", (57.7, 76.3))):
        xs = island_raster_x(x0, x1, T2_R, 4.0)
        y0, y1 = 23.7 - T2_R - 1.0, 56.3 + T2_R + 1.0
        out.append(f"({name})")
        for x in xs:
            out += [f"G00 X{f3(x)} Y{f3(y0)}", "G01 Z-4. F1000.", f"Y{f3(y1)} F{T2_F}.", "G00 Z2."]
    return out


def wall_finish() -> list[str]:
    """Walls with the tool-centre path written out (CAM 'wear' compensation): G41 D02 with D02 = 0 in
    the offset table, so the operator can trim a wall with a radius wear value. Tool centre 6 mm off
    the wall: pocket at x 14 / 86, y 14 / 66 (corners R2), island 1 at 6 mm (corners R10), island 2
    straights at x 52 / 82 and ends R15. Lead-in/out arcs R2 (tool centre) from the channel centre."""
    f = f"F{T2_F}."
    return [
        "(WALLS: TOOL-CENTRE PATH, WEAR COMPENSATION G41 D02 WITH D02 = 0, AE 0.3, AP 10, CLIMB)",
        "(POCKET WALL COUNTER-CLOCKWISE)",
        "G00 X50. Y16.", "Z2.", "G01 Z-10. F2000.",
        f"G41 D02 G01 X48. Y16. {f}",
        "G03 X50. Y14. R2.",
        "G01 X84.", "G03 X86. Y16. R2.", "G01 Y64.", "G03 X84. Y66. R2.", "G01 X16.", "G03 X14. Y64. R2.",
        "G01 Y16.", "G03 X16. Y14. R2.", "G01 X52.",
        "G03 X54. Y16. R2.",
        "G40 G01 X52. Y16.",
        "G00 Z2.",
        "(ISLAND 1 CLOCKWISE)",
        "G00 X16. Y36.", "G01 Z-10. F2000.",
        f"G41 D02 G01 X16. Y38. {f}",
        "G03 X18. Y40. R2.",
        "G01 Y52.", "G02 X28. Y62. R10.", "G01 X38.", "G02 X48. Y52. R10.", "G01 Y28.", "G02 X38. Y18. R10.",
        "G01 X28.", "G02 X18. Y28. R10.", "G01 Y42.",
        "G03 X16. Y44. R2.",
        "G40 G01 X16. Y42.",
        "G00 Z2.",
        "(ISLAND 2 OBROUND CLOCKWISE)",
        "G00 X84. Y44.", "G01 Z-10. F2000.",
        f"G41 D02 G01 X84. Y42. {f}",
        "G03 X82. Y40. R2.",
        "G01 Y33.", "G02 X52. Y33. I-15. J0.", "G01 Y47.", "G02 X82. Y47. I15. J0.", "G01 Y38.",
        "G03 X84. Y36. R2.",
        "G40 G01 X84. Y38.",
        "G00 Z25.",
    ]


def finish_section() -> list[str]:
    out = [
        "N30 (T2 D12 3FL CARBIDE ENDMILL - FLOOR, ISLAND TOPS, WALLS)",
        "T2 M06 (D12 3FL CARBIDE ENDMILL HELICAL 59744)",
        "G187 P3 (FINISH SMOOTHING)",
        f"G90 G54 G00 X50. Y16. S{RPM} M03",
        "G43 H02 Z25. M08",
    ]
    out += floor_finish() + island_top_finish() + wall_finish()
    out += ["M09", "M05", "G187 (SMOOTHING BACK TO THE SETTING)", "G53 G49 Z0.", "G53 Y0.", "M30"]
    return out


def probe_section() -> list[str]:
    return [
        "N10 (T20 SPINDLE PROBE - MEASURE THE STOCK TOP, SET G54 Z)",
        "T20 M06 (RENISHAW OMP40-2 SPINDLE PROBE, 6 MM RUBY BALL)",
        "G10 L10 P20 R125.000 (T20 LENGTH FROM THE PROBE CALIBRATION, ELECTRICAL TRIGGER POINT)",
        "M19 (ORIENT THE SPINDLE FOR THE PROBE)",
        "G90 G54 G00 X50. Y40.",
        "G43 H20 Z50.",
        "G65 P9810 Z5. F2000. (PROTECTED POSITIONING TO 5 MM ABOVE THE TOP)",
        "G65 P9811 Z0. S54. (SINGLE SURFACE Z: CORRECT G54 Z BY THE MEASURED ERROR)",
        "G65 P9810 Z50. F2000.",
        "G53 G49 Z0.",
    ]


MACROS = """O09810 (PROTECTED POSITIONING - RENISHAW-STYLE)
(X Y Z = TARGET IN THE ACTIVE WORK COORDINATES, F = FEED; ALARM IF THE STYLUS TOUCHES ON THE WAY)
IF [#9 EQ #0] THEN #9 = 2000.
IF [#24 EQ #0] THEN #24 = #5041
IF [#25 EQ #0] THEN #25 = #5042
IF [#26 EQ #0] THEN #26 = #5043
G90 G31 X#24 Y#25 Z#26 F#9
M78 (ALARM 378 WHEN THE PROBE TRIGGERED: OBSTRUCTION)
M99

O09811 (SINGLE SURFACE Z MEASUREMENT - RENISHAW-STYLE)
(Z = NOMINAL SURFACE Z IN THE ACTIVE WORK COORDINATES, S = WORK OFFSET 54-59 TO CORRECT)
(Q = OVERTRAVEL BEYOND Z, DEFAULT 4. CALL ABOVE THE SURFACE WITH THE PROBE IN THE SPINDLE AND G43 ON)
IF [#26 EQ #0] GOTO 91
IF [#19 EQ #0] GOTO 92
IF [#17 EQ #0] THEN #17 = 4.
#101 = #5043 (START HEIGHT)
G90 G31 Z[#26 - #17] F600. (FIRST TOUCH)
M79 (ALARM 379 WHEN NO SURFACE WAS FOUND)
G91 G01 Z1.5 F2000. (BACK OFF)
G31 Z-3.5 F60. (MEASURING TOUCH AT THE CALIBRATED FEED)
M79
#102 = #5063 (SKIP POSITION Z, WORK COORDINATES)
G90 G01 Z#101 F2000. (BACK TO THE START HEIGHT)
#103 = #102 - #26 (SURFACE ERROR)
#104 = 5203 + [#19 - 53] * 20 (Z ENTRY OF THE WORK OFFSET: #5223 = G54 Z)
#[#104] = #[#104] + #103
#135 = #103 (LAST MEASURED ERROR)
M99
N91 #3000 = 91 (Z INPUT MISSING)
N92 #3000 = 92 (S INPUT MISSING)
M99
%
"""


def header() -> list[str]:
    return [
        "%",
        "O01001 (POCKET ISLANDS AL6061-T6)",
        "(CNC TWIN EXAMPLE POCKET-ISLANDS-AL6061, WRITTEN BY MAKE_PROGRAM.PY)",
        "(HAAS VF-2SS NGC, STOCK 100 X 80 X 30 6061-T651 PLATE IN A LANG MAKRO-GRIP 48120-46)",
        "(G54: X0 Y0 LOWER LEFT CORNER OF THE STOCK, Z0 STOCK TOP SET BY THE PROBE IN N10)",
        "(T20 RENISHAW OMP40-2 SPINDLE PROBE)",
        "(T1 D10 3FL CARBIDE ENDMILL HELICAL 59766 MH35AL-025-31000 - ROUGH)",
        "(T2 D12 3FL CARBIDE ENDMILL HELICAL 59744 MH35AL-020-31200 - FINISH)",
        "G90 G17 G21 G40 G49 G80 G94 G54",
        "G53 G00 Z0.",
    ]


# ----------------------------------------------------------------------------- engagement check
PX = 0.05


def engagement(path: Path2, radius: float, allowance: float = 0.3):
    """Engagement angle along the roughing path on a 2-D material map of the pocket (outside the
    pocket and inside the islands is 'no material to cut': the walls and islands are offset by the
    allowance, so a sample that touches them means the path gouges)."""
    nx, ny = int(100 / PX), int(80 / PX)
    xs = (np.arange(nx) + 0.5) * PX
    ys = (np.arange(ny) + 0.5) * PX
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    inside = _rounded_rect(X, Y, 8 + allowance, 92 - allowance, 8 + allowance, 72 - allowance, 8 - allowance)
    island1 = _rounded_rect(X, Y, 24 - allowance, 42 + allowance, 24 - allowance, 56 + allowance, 4 + allowance)
    island2 = _obround(X, Y, 67.0, 24 - allowance, 56 + allowance, 9 + allowance)
    material = inside & ~island1 & ~island2           # what roughing must remove
    keep = ~inside | island1 | island2                 # what it must not touch
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    rpx = int(math.ceil(radius / PX)) + 1
    disk_i, disk_j = np.meshgrid(np.arange(-rpx, rpx + 1), np.arange(-rpx, rpx + 1), indexing="ij")
    disk = (disk_i * PX) ** 2 + (disk_j * PX) ** 2 <= radius ** 2
    angles, gouge = [], 0
    bx, by = NODES["B"]
    material &= (X - bx) ** 2 + (Y - by) ** 2 > (TROCHOID_R + radius) ** 2      # the helical entry
    for kind, a, b, c, ccw in path.moves:
        for p in _samples(kind, a, b, c, ccw, 0.1):
            pts = p + radius * ring
            i = np.clip((pts[:, 0] / PX).astype(int), 0, nx - 1)
            j = np.clip((pts[:, 1] / PX).astype(int), 0, ny - 1)
            angles.append(float(np.count_nonzero(material[i, j])) * 0.5)
            ci, cj = int(p[0] / PX), int(p[1] / PX)
            sl = (slice(max(ci - rpx, 0), min(ci + rpx + 1, nx)), slice(max(cj - rpx, 0), min(cj + rpx + 1, ny)))
            d = disk[sl[0].start - ci + rpx: sl[0].stop - ci + rpx, sl[1].start - cj + rpx: sl[1].stop - cj + rpx]
            gouge += int(np.count_nonzero(keep[sl] & d))
            material[sl] &= ~d
    angles = np.array(angles)
    return angles, int(np.count_nonzero(material)), gouge


def _rounded_rect(X, Y, x0, x1, y0, y1, r):
    cx = np.clip(X, x0 + r, x1 - r)
    cy = np.clip(Y, y0 + r, y1 - r)
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r) & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def _obround(X, Y, xc, y0, y1, r):
    cy = np.clip(Y, y0 + r, y1 - r)
    return (X - xc) ** 2 + (Y - cy) ** 2 <= r * r


def _samples(kind, a, b, c, ccw, step):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if kind == "line":
        n = max(int(np.linalg.norm(b - a) / step), 1)
        return [a + (b - a) * (k / n) for k in range(1, n + 1)]
    c = np.asarray(c, float)
    r = float(np.linalg.norm(a - c))
    t0 = math.atan2(a[1] - c[1], a[0] - c[0])
    t1 = math.atan2(b[1] - c[1], b[0] - c[0])
    sweep = (t1 - t0) % (2 * math.pi) if ccw else -((t0 - t1) % (2 * math.pi))
    if abs(sweep) < 1e-12:
        sweep = 2 * math.pi if ccw else -2 * math.pi
    n = max(int(abs(sweep) * r / step), 1)
    return [c + r * np.array([math.cos(t0 + sweep * k / n), math.sin(t0 + sweep * k / n)]) for k in range(1, n + 1)]


def main():
    rough_path = Path2()
    lines = header() + probe_section() + rough_section(rough_path) + finish_section()
    text = "\n".join(lines) + "\n\n" + MACROS
    (HERE / "op10.nc").write_text(text, encoding="ascii")
    print(f"wrote {HERE / 'op10.nc'}: {len(text.splitlines())} lines")
    # engagement of the trochoidal loops (the island raster has its own fixed ae)
    angles, left, gouge = engagement(rough_path, T1_R)
    cutting = angles[angles > 0]
    ae = T1_R * (1 - np.cos(np.radians(cutting)))
    print(f"roughing samples {len(angles)}, cutting {len(cutting)}; engagement angle mean {cutting.mean():.1f} deg, "
          f"p95 {np.percentile(cutting, 95):.1f}, max {cutting.max():.1f}; equivalent ae mean {ae.mean():.2f} mm, "
          f"p95 {np.percentile(ae, 95):.2f}, max {ae.max():.2f}")
    print(f"material left after the loops (outside the 0.3 mm allowance): {left * PX * PX:.2f} mm2; "
          f"gouge pixels: {gouge}")
    length = sum(np.linalg.norm(np.array(m[2]) - np.array(m[1])) if m[0] == "line" else 0.0 for m in rough_path.moves)
    arcs = 0.0
    for kind, a, b, c, ccw in rough_path.moves:
        if kind == "arc":
            pts = [np.asarray(a, float)] + _samples(kind, a, b, c, ccw, 0.05)
            arcs += sum(float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(len(pts) - 1))
    print(f"trochoid path length {arcs:.0f} mm (+ {length:.0f} mm lines) at F{T1_F}: {60 * arcs / T1_F:.1f} s")


if __name__ == "__main__":
    main()
