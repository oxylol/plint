"""Hand calculations behind EXPECTED.md (ruled Ti-6Al-4V blade, Haas UMC-750, G234).

Run: cd apps/cnc-twin/examples/blade-5axis-ti && ../../bin/tsafe python expected_calc.py

Kienzle kc11 2110 N/mm2, mc 0.22 (db/materials/ti6al4v, ISCAR group 37, the entry the twin selects for a
coated carbide tool); E 114.5 GPa (same file); blade sections from blade_geom.py; toolpath numbers from
toolpath_stats.json (make_program.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import blade_geom as G

HERE = Path(__file__).resolve().parent
KC11, MC = 2110.0, 0.22
KC11_MEAS = 1253.0
D, Z, R = 12.0, 4, 6.0
N = 1455.0
AP = 21.8                                         # contact band on the flute (toolpath_stats contact_band)
PASSES = [("rough 1", 0.0711, 0.95), ("rough 2", 0.0711, 0.70), ("finish", 0.0559, 0.30)]
E_TI, E_C = 114500.0, 600000.0
STICKOUT, GAUGE, NUT_D = 45.0, 70.0, 50.0
SPINDLE_K = 70.0                                  # N/um, haas-umc750 frf modes (est.)


def forces(fz, ae, kr, kc11=KC11, samples=7200):
    phi = np.linspace(0, 2 * math.pi, samples, endpoint=False)
    phi_e = math.acos(1 - 2 * ae / D)
    lo, hi = math.pi - phi_e, math.pi
    ft_sum = np.zeros_like(phi)
    fn = np.zeros_like(phi)
    for k in range(Z):
        a = (phi + k * 2 * math.pi / Z) % (2 * math.pi)
        on = (a >= lo) & (a <= hi)
        h = np.maximum(fz * np.sin(a), 0.0)
        ft = np.where(on, kc11 * AP * h ** (1 - MC), 0.0)
        ft_sum += ft
        fn += ft * np.sin(a) + kr * ft * (-np.cos(a))   # component normal to the surface, pushing the blade
    torque = float(np.mean(ft_sum)) * R * 1e-3
    return torque, torque * N * 2 * math.pi / 60, float(np.mean(fn)), fz * math.sin(phi_e)


def section_inertia(points):
    """Area, centroid and principal second moments of a closed polygon (Green's theorem)."""
    x, y = points[:, 0], points[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    area = 0.5 * cr.sum()
    cx = ((x + xn) * cr).sum() / (6 * area)
    cy = ((y + yn) * cr).sum() / (6 * area)
    ixx = ((y ** 2 + y * yn + yn ** 2) * cr).sum() / 12 - area * cy ** 2
    iyy = ((x ** 2 + x * xn + xn ** 2) * cr).sum() / 12 - area * cx ** 2
    ixy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cr).sum() / 24 - area * cx * cy
    mean, diff = 0.5 * (ixx + iyy), math.hypot(0.5 * (ixx - iyy), ixy)
    return abs(area), mean - diff, mean + diff


def blade_compliance(z_eval, area_load=(0.0, 20.0)):
    """Deflection at z_eval of the blade (cantilever from the platform z -7) per N of a load spread
    uniformly over z 0..20, bending about the weak principal axis (similar sections: I ~ scale^4)."""
    _, i_min_hub, _ = section_inertia(G.section_points(G.HUB))
    z = np.linspace(G.ROOT_Z, G.TIP["z"], 2001)
    scale = 1.0 + (G.TIP["chord"] / G.HUB["chord"] - 1.0) * (z - G.HUB["z"]) / (G.TIP["z"] - G.HUB["z"])
    EI = E_TI * i_min_hub * scale ** 4
    q = np.where((z >= area_load[0]) & (z <= area_load[1]), 1.0 / (area_load[1] - area_load[0]), 0.0)
    # bending moment at z from the load above it: M(z) = int_z^L q(s) (s - z) ds; unit load at z_eval: m(z) = z_eval - z
    moment = np.array([np.trapezoid(q[i:] * (z[i:] - z[i]), z[i:]) if i < len(z) - 1 else 0.0 for i in range(len(z))])
    m_unit = np.where(z <= z_eval, z_eval - z, 0.0)
    return float(np.trapezoid(moment * m_unit / EI, z)), i_min_hub


def tool_compliance():
    lt = STICKOUT - AP / 2
    it = math.pi * (0.8 * D) ** 4 / 64
    ih = math.pi * NUT_D ** 4 / 64
    c = lt ** 3 / (3 * E_C * it) + GAUGE ** 3 / (3 * 210000 * ih) + GAUGE ** 2 * lt / (210000 * ih) \
        + GAUGE * lt ** 2 / (210000 * ih)
    return c * 1e3 + 1.0 / SPINDLE_K


def main():
    stats = json.loads((HERE / "toolpath_stats.json").read_text())
    area, i_min, i_max = section_inertia(G.section_points(G.HUB))
    print(f"hub section: area {area:.1f} mm2, I_min {i_min:.1f} mm4, I_max {i_max:.0f} mm4; edge radii {G.edge_radii(G.HUB)}")
    c_tool = tool_compliance()
    print(f"tool-holder-spindle compliance {c_tool:.4f} um/N (k {1 / c_tool:.1f} N/um)")
    for name, fz, ae in PASSES:
        for kr in (0.4, 0.9):
            torque, power, fn, hmax = forces(fz, ae, kr)
            line = f"{name} kr {kr}: h_max {hmax:.4f} mm, torque {torque:.2f} Nm, power {power:.0f} W, normal force {fn:.0f} N"
            if name == "finish":
                parts = []
                for z in (0.0, 10.0, 20.0):
                    cb, _ = blade_compliance(z)
                    parts.append(f"z {z:.0f}: blade {fn * cb * 1e3:.1f} um")
                line += f"; tool {fn * c_tool:.1f} um; " + ", ".join(parts)
            print(line)
    for z in (0.0, 10.0, 20.0):
        cb, _ = blade_compliance(z)
        print(f"blade compliance at z {z:.0f}: {cb * 1e3:.4f} um/N")
    fin = stats["passes"][2]
    print("twist error (balanced two-point offset):", fin["deviation_mm"]["most_overcut"], fin["deviation_mm"]["most_undercut"])
    print("B, C ranges (finish):", fin["B_deg"], fin["C_deg"], "rotary max deg/s:", stats["rotary"])
    print("tip path:", stats["tip_path_mm"], "feed time at the tip:", stats["feed_time_s"])
    frev = Z * 0.0559
    print(f"flank Ra: per-tooth cusp {0.0559 ** 2 / (8 * R) * 1000:.3f} um, per-rev mark {frev ** 2 / (8 * R) * 1000:.2f} um "
          f"-> Ra ~ {frev ** 2 / (8 * R) * 1000 / 3.9:.2f} um")
    removed = 45267 - 41126
    print(f"removed {removed} mm3 = {removed * 4430e-9 * 1000:.1f} g")


if __name__ == "__main__":
    main()
