"""Writes op10.nc (Haas NGC) for the thin-wall part and prints the numbers EXPECTED.md uses: the
engagement of the trochoid between the fins (2-D material map), and the stiffness and first
natural frequency of a fin in each state of the finishing ladder (Euler-Bernoulli strip, finite
elements), and the long-reach finishing tool's stiffness.

Run: cd apps/cnc-twin && bin/tsafe python examples/thin-wall-al7075/make_program.py

Program coordinates: G54 X0 Y0 = lower-left corner of the stock, Z0 = stock top (= fin tops);
stock frame z = program Z + 32. Fins: x 15..75, y 19.5..20.5 and 39.5..40.5, Z0..Z-25.

Roughing (T1, Helical 59766 D10 3FL LOC 25, ER32): two levels of ap 12.5 (Z-12.5, Z-25.0; the
maker's roughing ADOC limit is 2 D): constant-engagement peel passes (ae 2.375 = 0.24 D) from the
front and back stock faces towards the fins, a trochoid (r 4.0, 1.0 mm per loop) along the 18 mm
band between the fins, then the 2 mm strips at the fin ends. 0.5 mm is left on every fin face and
end; the floor (base top, Z-25) is final.

Finishing (T2, a second D10 LOC 25 in the slim ER16 at 42 mm stickout, L/D 4.2): the 'ladder':
five levels of 5 mm (Z-5 ... Z-25), at each level one clockwise (climb) contour around each fin,
so both faces of a band are finished while the 2 mm rough wall below still supports it. ae 0.5,
ap 5, tool-centre path with wear compensation (G41 D02, D02 = 0).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOP_Z_S = 32.0
RPM = 12000
T1_FZ = 0.0965                 # Helical SF_H35AL-3 wrought Al, 3/8 in column, 'Rgh' (db side_milling start)
T2_FZ = 0.0584                 # same chart, 'Fin' (db finishing start)
T1_F = round(RPM * 3 * T1_FZ)  # 3474
T2_F = round(RPM * 3 * T2_FZ)  # 2102
LINK_F = 8000                  # feed for links at depth in cleared stock (no rapid below the stock top)
R = 5.0                        # both tools D10
WALL_STOCK = 0.5
FINS = [(19.5, 20.5), (39.5, 40.5)]
FIN_X = (15.0, 75.0)
ROUGH_LEVELS = [-12.5, -25.0]
LADDER = [-5.0, -10.0, -15.0, -20.0, -25.0]
PEEL_AE = 2.375
TROCH_R = 4.0
TROCH_STEP = 1.0


def f3(v: float) -> str:
    s = f"{v:.3f}".rstrip("0")
    return "0." if s in ("-0.", "-0", "0") else s


def peel_front(z: float) -> list[str]:
    """Front band y 0..19.0: passes moving -X (climb: the uncut stock is on the +Y side, on the right
    of the motion), stepping +Y by ae; each link runs back 1 mm off the new wall, the step-over is
    made beyond the stock end (x 96)."""
    out = [f"G00 X96. Y-4.", "Z2.", f"G01 Z{f3(z)} F2000."]
    edge_final = FINS[0][0] - WALL_STOCK              # 19.0
    n = round(edge_final / PEEL_AE)
    for k in range(1, n + 1):
        yc = k * edge_final / n - R
        out += [f"G01 Y{f3(yc)} F{LINK_F}.", f"X-6. F{T1_F}.", f"Y{f3(yc - 1.0)} F{LINK_F}.", "X96."]
    return out + ["G00 Z2."]


def peel_back(z: float) -> list[str]:
    """Back band y 41.0..60: passes moving +X (stock on the -Y side, right of the motion), stepping -Y."""
    out = [f"G00 X-6. Y64.", "Z2.", f"G01 Z{f3(z)} F2000."]
    edge_final = FINS[1][1] + WALL_STOCK              # 41.0
    width = 60.0 - edge_final
    n = round(width / PEEL_AE)
    for k in range(1, n + 1):
        yc = 60.0 - k * width / n + R
        out += [f"G01 Y{f3(yc)} F{LINK_F}.", f"X96. F{T1_F}.", f"Y{f3(yc + 1.0)} F{LINK_F}.", "X-6."]
    return out + ["G00 Z2."]


def trochoid_between(z: float) -> list[str]:
    """Band y 21.0..39.0 (18 mm = D + 2 r): counter-clockwise loops (climb) advancing +X along y 30."""
    yc = 0.5 * (FINS[0][1] + FINS[1][0])              # 30
    r, s = TROCH_R, TROCH_STEP
    x0, x1 = -6.0 - r - R, 96.0 + r + R
    loops = math.ceil((x1 - x0) / s)
    out = [f"G00 X{f3(x0)} Y{f3(yc - r)}", "Z2.", f"G01 Z{f3(z)} F2000."]
    for i in range(loops):
        a = (x0 + i * s, yc - r)
        t = (x0 + (i + 0.5) * s, yc + r)
        b = (x0 + (i + 1) * s, yc - r)
        c1 = (0.5 * (a[0] + t[0]), yc)
        c2 = (0.5 * (t[0] + b[0]), yc)
        feed = f" F{T1_F}." if i == 0 else ""
        out.append(f"G03 X{f3(t[0])} Y{f3(t[1])} I{f3(c1[0] - a[0])} J{f3(c1[1] - a[1])}{feed}")
        out.append(f"G03 X{f3(b[0])} Y{f3(b[1])} I{f3(c2[0] - t[0])} J{f3(c2[1] - t[1])}")
    return out + ["G00 Z2."]


def end_strips(z: float) -> list[str]:
    """The 2 mm strips beyond the fin ends (y 19..21 and 39..41, x < 14.5 and x > 75.5), centred
    passes from outside the stock and back along the same cleared line."""
    out = []
    for y0, y1 in FINS:
        yc = 0.5 * (y0 + y1)
        out += [f"G00 X-6. Y{f3(yc)}", "Z2.", f"G01 Z{f3(z)} F2000.", f"X{f3(FIN_X[0] - WALL_STOCK - R)} F{T1_F}.",
                f"X-6. F{LINK_F}.", "G00 Z2.",
                f"G00 X96. Y{f3(yc)}", f"G01 Z{f3(z)} F2000.", f"X{f3(FIN_X[1] + WALL_STOCK + R)} F{T1_F}.",
                f"X96. F{LINK_F}.", "G00 Z2."]
    return out


def rough_section() -> list[str]:
    out = ["N10 (T1 D10 3FL CARBIDE ENDMILL - ROUGHING IN TWO LEVELS OF 12.5)",
           "T1 M06 (D10 3FL CARBIDE ENDMILL HELICAL 59766)",
           "G187 P1 (ROUGH SMOOTHING)",
           f"G90 G54 G00 X96. Y-4. S{RPM} M03",
           "G43 H01 Z25. M08"]
    for z in ROUGH_LEVELS:
        out.append(f"(LEVEL Z{f3(z)}: FRONT PEEL, BACK PEEL, TROCHOID BETWEEN THE FINS, END STRIPS)")
        out += peel_front(z) + peel_back(z) + trochoid_between(z) + end_strips(z)
    out += ["G00 Z25.", "M09", "M05", "G53 G49 Z0."]
    return out


def fin_contour(y0: float, y1: float, z: float) -> list[str]:
    """Clockwise tool-centre contour 5 mm off a fin (climb with M03), entered along the front face
    from beyond the right end and left with an arc into the front band."""
    x0, x1 = FIN_X
    yf, yb = y0 - R, y1 + R
    f = f"F{T2_F}."
    return [
        f"G00 X85. Y{f3(yf)}", f"G01 Z{f3(z)} F2000.",
        f"G41 D02 G01 X80. Y{f3(yf)} {f}",
        f"X{f3(x0)}",
        f"G02 X{f3(x0 - R)} Y{f3(y0)} R{f3(R)}",
        f"G01 Y{f3(y1)}",
        f"G02 X{f3(x0)} Y{f3(yb)} R{f3(R)}",
        f"G01 X{f3(x1)}",
        f"G02 X{f3(x1 + R)} Y{f3(y1)} R{f3(R)}",
        f"G01 Y{f3(y0)}",
        f"G02 X{f3(x1)} Y{f3(yf)} R{f3(R)}",
        f"G01 X{f3(x1 - 5.0)}",
        f"G03 X{f3(x1 - 10.0)} Y{f3(yf - 5.0)} R5.",
        f"G40 G01 X{f3(x1 - 10.0)} Y{f3(yf - 6.0)}",
        "G00 Z2.",
    ]


def finish_section() -> list[str]:
    out = ["N20 (T2 D10 3FL CARBIDE ENDMILL, 42 MM STICKOUT IN ER16 - LADDER FINISHING OF THE FINS)",
           "T2 M06 (D10 3FL CARBIDE ENDMILL HELICAL 59766 LONG STICKOUT)",
           "G187 P3 (FINISH SMOOTHING)",
           f"G90 G54 G00 X85. Y14.5 S{RPM} M03",
           "G43 H02 Z25. M08",
           "Z2."]
    for z in LADDER:
        out.append(f"(LADDER LEVEL Z{f3(z)}: AP 5, AE 0.5, FIN 1 THEN FIN 2)")
        for y0, y1 in FINS:
            out += fin_contour(y0, y1, z)
    out += ["G00 Z25.", "M09", "M05", "G187 (SMOOTHING BACK TO THE SETTING)", "G53 G49 Z0.", "G53 Y0.", "M30"]
    return out


def header() -> list[str]:
    return ["%",
            "O01003 (THIN WALL AL7075-T6)",
            "(CNC TWIN EXAMPLE THIN-WALL-AL7075, WRITTEN BY MAKE_PROGRAM.PY)",
            "(HAAS VF-2SS NGC, STOCK 90 X 60 X 32 7075-T651 PLATE IN A LANG MAKRO-GRIP 48120-46)",
            "(G54: X0 Y0 LOWER LEFT CORNER OF THE STOCK, Z0 STOCK TOP)",
            "(TWO FINS 1.0 THICK, 25 TALL, 60 LONG ON A 7 MM BASE)",
            "(T1 D10 3FL CARBIDE ENDMILL HELICAL 59766 MH35AL-025-31000 - ROUGH)",
            "(T2 D10 3FL CARBIDE ENDMILL HELICAL 59766 MH35AL-025-31000 - FINISH, ER16, 42 STICKOUT)",
            "G90 G17 G21 G40 G49 G80 G94 G54",
            "G53 G00 Z0."]


# ----------------------------------------------------------------------------- checks and numbers
def beam_frequency(segments, E=71.7e9, rho=2800.0, n_el=100):
    """First natural frequency (Hz) of a clamped-free strip of unit width with thickness segments
    [(length_m, thickness_m), ...] from the root; Euler-Bernoulli finite elements."""
    L = sum(s[0] for s in segments)
    xs = np.linspace(0.0, L, n_el + 1)
    nd = 2 * (n_el + 1)
    K = np.zeros((nd, nd))
    M = np.zeros((nd, nd))
    for e in range(n_el):
        xm = 0.5 * (xs[e] + xs[e + 1])
        acc = 0.0
        t = segments[-1][1]
        for length, thick in segments:
            acc += length
            if xm <= acc + 1e-12:
                t = thick
                break
        le = xs[e + 1] - xs[e]
        EI = E * t ** 3 / 12.0
        m = rho * t
        k = EI / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                     [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        mm = m * le / 420.0 * np.array([[156, 22 * le, 54, -13 * le], [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                                        [54, 13 * le, 156, -22 * le], [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += k
        M[np.ix_(idx, idx)] += mm
    free = np.arange(2, nd)
    from scipy.linalg import eigh
    w2 = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True, subset_by_index=[0, 0])[0]
    return math.sqrt(w2) / (2 * math.pi)


def strip_compliance(segments, a_m, E=71.7e9):
    """Deflection per unit force per unit width (m^2/N) at height a of a stepped clamped-free strip:
    delta = int_0^a (a - x)^2 / (E t(x)^3 / 12) dx."""
    n = 4000
    xs = np.linspace(0.0, a_m, n + 1)
    acc_edges = np.cumsum([s[0] for s in segments])
    thick = np.array([segments[min(np.searchsorted(acc_edges, x, side="right"), len(segments) - 1)][1] for x in xs])
    integrand = (a_m - xs) ** 2 * 12.0 / (E * thick ** 3)
    return float(np.trapezoid(integrand, xs))


def ladder_numbers():
    print("fin states during the ladder (strip model, unit width; load at mid-band; b_eff = 2 x load height):")
    H = 25e-3
    for k, z in enumerate(LADDER, start=1):
        band_top = H - (k - 1) * 5e-3
        band_bot = H - k * 5e-3
        a = 0.5 * (band_top + band_bot)
        for side, t_band in (("A", 2.0e-3), ("B", 1.5e-3)):
            segs = [(band_bot, 2.0e-3), (5e-3, t_band)]
            if k > 1:
                segs.append(((k - 1) * 5e-3, 1.0e-3))
            c = strip_compliance(segs, a)
            b_eff = min(2.0 * a, 60e-3)
            k_wall = b_eff / c
            f1 = beam_frequency(segs)
            print(f"  level {k} (Z{z:g}) side {side}: load at {a * 1e3:.1f} mm above the base, "
                  f"k_wall = {k_wall * 1e-6:.3f} N/um (b_eff {b_eff * 1e3:.0f} mm), f1 = {f1:.0f} Hz")
    f_fin = beam_frequency([(25e-3, 1.0e-3)])
    print(f"  finished fin 1.0 x 25: f1 = {f_fin:.0f} Hz (closed form 0.5596 sqrt(E t^2/12 rho)/L^2 = "
          f"{0.5596 * math.sqrt(71.7e9 * 1e-6 / 12 / 2800) / 25e-3 ** 2:.0f} Hz)")
    for label, i_flute in (("0.8 D (Kops-Vo)", math.pi * 8.0 ** 4 / 64), ("0.55 D core", math.pi * 5.5 ** 4 / 64)):
        c = (25.0 ** 3 / (3 * i_flute) + (42.0 ** 3 - 25.0 ** 3) / (3 * math.pi * 10.0 ** 4 / 64)) / 530000.0
        k_tool = 1.0 / c / 1000.0
        k_tot = 1.0 / (1.0 / k_tool + 1.0 / 70.0)
        print(f"  T2 tool (D10, 42 stickout, flutes {label}): k_tool {k_tool:.1f} N/um, with the spindle {k_tot:.1f} N/um")


PX = 0.05


def trochoid_engagement():
    nx, ny = int(110 / PX), int(30 / PX)                # x -10..100, y 15..45 window
    xs = -10.0 + (np.arange(nx) + 0.5) * PX
    ys = 15.0 + (np.arange(ny) + 0.5) * PX
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    material = (X >= 0) & (X <= 90) & (Y > 21.0) & (Y < 39.0)
    keep = (X >= 14.5) & (X <= 75.5) & ((Y <= 21.0) | (Y >= 39.0))
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], 1)
    yc = 30.0
    x0 = -6.0 - TROCH_R - R
    angles, gouge = [], 0.0
    t = 0.0
    while True:
        cx = x0 + TROCH_STEP * t / (2 * np.pi)
        if cx > 96.0 + TROCH_R + R:
            break
        p = np.array([cx + TROCH_R * math.sin(t), yc - TROCH_R * math.cos(t)])
        pts = p + R * ring
        i = np.clip(((pts[:, 0] + 10.0) / PX).astype(int), 0, nx - 1)
        j = np.clip(((pts[:, 1] - 15.0) / PX).astype(int), 0, ny - 1)
        angles.append(float(np.count_nonzero(material[i, j])) * 0.5)
        ci, cj = int((p[0] + 10.0) / PX), int((p[1] - 15.0) / PX)
        w = int(R / PX) + 2
        sl = (slice(max(ci - w, 0), min(ci + w + 1, nx)), slice(max(cj - w, 0), min(cj + w + 1, ny)))
        d2 = (X[sl] - p[0]) ** 2 + (Y[sl] - p[1]) ** 2 <= R * R
        gouge = max(gouge, float(np.count_nonzero(d2 & keep[sl])))
        material[sl] &= ~d2
        t += 0.1 / max(TROCH_R, 1.0)
    a = np.array(angles)
    cut = a[a > 0]
    ae = R * (1 - np.cos(np.radians(cut)))
    print(f"trochoid between the fins: engagement angle mean {cut.mean():.1f}, p95 {np.percentile(cut, 95):.1f}, "
          f"max {cut.max():.1f} deg; ae equivalent mean {ae.mean():.2f}, max {ae.max():.2f} mm; "
          f"left {np.count_nonzero(material) * PX * PX:.2f} mm2, gouge pixels {gouge:.0f}")


def main():
    lines = header() + rough_section() + finish_section()
    text = "\n".join(lines) + "\n%\n"
    (HERE / "op10.nc").write_text(text, encoding="ascii")
    print(f"wrote {HERE / 'op10.nc'}: {len(lines) + 1} lines")
    trochoid_engagement()
    ladder_numbers()


if __name__ == "__main__":
    main()
