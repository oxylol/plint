"""Hand calculations behind EXPECTED.md (4140 shaft, Haas ST-20Y).

Run: cd apps/cnc-twin && bin/tsafe python examples/lathe-shaft-4140/expected_calc.py

Closed-form laws with the database values: Kienzle kc11 2000 N/mm2, mc 0.25 (db/materials/aisi4140-qt30,
Walter), Taylor (same file, est.), spindle curve (db/machines/haas-st20y), insert geometry (db/tools).
The G71 passes are those of a type I cycle with D 2.5 (radius), U 0.6 / W 0.1 from the start point X46.
"""

from __future__ import annotations

import math

import numpy as np

KC11, MC = 2000.0, 0.25
E = 212000.0                                     # N/mm2 at 20 degC
RHO = 7850e-9                                    # kg/mm3
CP = 461.0                                       # J/kgK
ALPHA = 12.1e-6                                  # 1/K (20-100 degC)
CSS_CLAMP = 2500.0
PEAK = [[1400, 15.3], [1450, 15.7], [1500, 15.7], [1600, 15.2], [1800, 13.8], [2000, 13.0], [2400, 12.2],
        [2800, 11.7]]                            # kW, 200 % = peak curve; S1 = 50 %
TAYLOR = dict(C=453.0, n=0.25, a=0.3, b=0.12, f_ref=0.3, ap_ref=2.0)

# offset contour of the rough boundary (radius + U/2, shoulders + W): (radius, z_from, z_to)
BOUNDARY = [(12.0 + 0.3, 0.0, -23.0 + 0.1), (14.9915 + 0.3, -22.9, -60.0 + 0.1), (17.494 + 0.3, -59.9, -140.0 + 0.1),
            (21.0 + 0.3, -139.9, -183.5)]
STOCK_R = 22.5


def rpm(vc, r):
    return min(vc * 1000.0 / (2.0 * math.pi * r), CSS_CLAMP)


def feed_time(length, r, f, vc):
    return length / (f * rpm(vc, r) / 60.0)


def kienzle_fc(b, h):
    return KC11 * b * h ** (1.0 - MC)


def g71_passes(depth=2.5, start_r=23.0):
    passes = []
    r = start_r - depth
    while r > BOUNDARY[0][0]:
        z_end = next((zf for rb, zf, zt in BOUNDARY if rb > r), BOUNDARY[-1][2])
        stop = 2.0 - z_end if z_end != BOUNDARY[-1][2] else 2.0 - BOUNDARY[-1][2]
        ap = min(depth, STOCK_R - r) if r + depth > STOCK_R else depth
        passes.append((r, 2.0 - z_end if z_end < 0 else 0.0, ap))
        r -= depth
    return passes


def main():
    print("== spindle speeds (CSS, clamp 2500)")
    for d in (45.0, 41.0, 36.0, 31.0, 26.0, 24.6):
        print(f"T1 vc 200 at d {d}: {rpm(200, d / 2):.0f} rpm")
    print("== roughing passes (G71 type I, D 2.5)")
    t_total = 0.0
    for r, length, ap in g71_passes():
        n = rpm(200, r)
        h = 0.3 * math.sin(math.radians(95.0))
        fc = kienzle_fc(ap / math.sin(math.radians(95.0)), h)
        pc = fc * 200.0 / 60.0
        peak = float(np.interp(n, [p[0] for p in PEAK], [p[1] for p in PEAK]))
        t = feed_time(length, r, 0.3, 200)
        t_total += t
        print(f"pass r {r:.1f}: length {length:.1f} mm, ap {ap:.2f}, n {n:.0f}, Fc {fc:.0f} N, Pc {pc / 1000:.2f} kW "
              f"= {pc / 1000 / (0.5 * peak) * 100:.0f} % of S1 ({0.5 * peak:.2f} kW), time {t:.1f} s")
    final = [(12.3, 23.1, None), (15.29, 37.0, None), (17.79, 80.0, None), (21.3, 43.6, None)]
    t_final = sum(feed_time(length, r, 0.3, 200) for r, length, _ in final) + 4 * 1.0 / (0.3 * 2000 / 60)
    t_total += t_final
    print(f"final profile pass: {t_final:.1f} s -> roughing feed time {t_total:.1f} s")
    print("== finishing (G70, vc 220)")
    segs = [(12.0, 23.0, 0.12), (15.0, 36.3, 0.08), (17.5, 79.3, 0.12), (21.0, 42.3, 0.12)]
    t_fin = sum(feed_time(length, r, f, 220) for r, length, f in segs) + 4 * 3.0 / (0.12 * 2000 / 60)
    print(f"finishing feed time {t_fin:.1f} s")
    for name, f, rn in (("d30 seat", 0.08, 0.4), ("d35 / collar / thread OD", 0.12, 0.4)):
        rt = f ** 2 / (8 * rn) * 1000
        print(f"{name}: Rt = f^2/(8 r) = {rt:.2f} um, Ra = f^2/(31.2 r) = {f ** 2 / (31.2 * rn) * 1000:.2f} um")
    # finishing forces: ap 0.3 < r 0.4: effective lead from the nose arc
    kappa = 0.5 * math.acos(1.0 - 0.3 / 0.4)
    h = 0.12 * math.sin(kappa)
    b = 0.3 / math.sin(kappa)
    fc = kienzle_fc(b, h)
    print(f"finishing: effective kappa {math.degrees(kappa):.1f} deg, h {h:.3f}, b {b:.2f} -> Fc {fc:.0f} N, "
          f"passive force 0.6-1.0 Fc = {0.6 * fc:.0f}-{fc:.0f} N")
    print("== shaft as a beam: fixed at the jaw face (S z 30), pinned at the centre (S z 230)")
    seg = [(30.0, 50.0, 22.5), (50.0, 89.0, 21.0), (89.0, 170.0, 17.494), (170.0, 207.0, 14.9915), (207.0, 230.0, 11.3)]
    x = np.linspace(30.0, 230.0, 2001)
    radius = np.array([next(r for a, b_, r in seg if a <= xi <= b_) for xi in x])
    EI = E * math.pi * (2 * radius) ** 4 / 64.0
    for k_tail in (30.0, 60.0):
        for z_load in (188.0, 130.0, 70.0):
            delta = beam_compliance(x, EI, z_load, k_tail * 1000.0)
            print(f"tailstock {k_tail:.0f} N/um, load at S z {z_load:.0f}: compliance {delta * 1e3:.4f} um/N -> "
                  f"100 N gives {delta * 1e3 * 100:.2f} um radial ({2 * delta * 1e3 * 100:.2f} um on diameter)")
    print("== thread M24x1.5 (G76 P1, D 0.3, K 0.971)")
    n_pass = math.ceil((0.971 / 0.3) ** 2)
    depths = [0.3 * math.sqrt(k) for k in range(1, n_pass)] + [0.971]
    n_thr = 1326.0
    t_pass = 27.5 / (1.5 * n_thr / 60.0)
    print(f"passes {n_pass}: cumulative depths {[round(d, 3) for d in depths]}, {t_pass:.2f} s per pass in feed, "
          f"~{n_pass * (t_pass + 0.35):.1f} s with returns")
    print("== groove and part-off")
    t_groove = 2.2 / (0.08 * rpm(100, 11.4) / 60.0) + 0.1
    print(f"groove plunge 2.2 mm at 0.08 mm/rev: {t_groove:.2f} s; Fc = {kienzle_fc(3.0, 0.08):.0f} N")
    d_clamp = 90000.0 / (math.pi * 2000.0)
    r_c = d_clamp / 2
    t_css = 2 * math.pi * 60.0 / (0.08 * 90000.0) * (22.0 ** 2 - r_c ** 2) / 2.0
    t_rpm = (r_c + 0.2) / (0.08 * 2000.0 / 60.0)
    print(f"part-off: CSS down to d {d_clamp:.1f} then 2000 rpm: {t_css:.1f} + {t_rpm:.1f} s; Fc {kienzle_fc(3.0, 0.08):.0f} N, "
          f"power {kienzle_fc(3.0, 0.08) * 90 / 60 / 1000:.2f} kW, torque at d45 {kienzle_fc(3.0, 0.08) * 0.0225:.1f} Nm")
    print("== mass")
    v_bar, v_part = 365715.0, 167680.0             # make_cad.py (exact B-rep volumes)
    v_stub = math.pi * STOCK_R ** 2 * 47.0
    v_chips = v_bar - v_part - v_stub
    print(f"part {v_part * RHO:.3f} kg, stub in the chuck {v_stub * RHO:.3f} kg, chips {v_chips:.0f} mm3 = {v_chips * RHO:.3f} kg")
    print("== heat (flood)")
    e_rough = 6.8e3 * 45.0
    for frac in (0.1, 0.2):
        m = (v_bar - 0.5 * v_chips) * RHO
        print(f"work fraction {frac}: {frac * e_rough / 1000:.0f} kJ into the bar -> adiabatic {frac * e_rough / (m * CP):.1f} K")
    h_flood, area = 6000.0, math.pi * 0.040 * 0.200
    tau = (v_part * RHO) * CP / (h_flood * area)
    print(f"flood cooling h {h_flood:.0f} W/m2K on {area:.3f} m2: time constant {tau:.1f} s; steady rise at 1 kW "
          f"{1000.0 / (h_flood * area):.1f} K; d30 change per K {30 * ALPHA * 1000:.2f} um/K")
    print("== tool life (Taylor, est.)")
    t_life = (TAYLOR["C"] / (200 * (0.3 / TAYLOR["f_ref"]) ** TAYLOR["a"] * (2.5 / TAYLOR["ap_ref"]) ** TAYLOR["b"])) \
        ** (1.0 / TAYLOR["n"])
    print(f"T1 roughing: T = {t_life:.1f} min; {t_total / 60:.2f} min in cut -> {t_total / 60 / t_life * 100:.1f} % of life")


def beam_compliance(x, EI, z_load, k_tail):
    """Deflection at z_load per unit load of a beam fixed at x[0], elastically supported (k_tail N/mm) at
    x[-1]: superposition of the cantilever under the load and the support reaction R (compatibility)."""
    def cantilever(load_at, at):
        # deflection at 'at' of a cantilever (fixed at x[0]) under a unit load at 'load_at', unit-load method
        m1 = np.where(x <= load_at, load_at - x, 0.0)
        m2 = np.where(x <= at, at - x, 0.0)
        return np.trapezoid(m1 * m2 / EI, x)
    a_ll = cantilever(z_load, z_load)
    a_lt = cantilever(z_load, x[-1])
    a_tt = cantilever(x[-1], x[-1])
    reaction = a_lt / (a_tt + 1.0 / k_tail)
    return a_ll - reaction * a_lt


if __name__ == "__main__":
    main()
