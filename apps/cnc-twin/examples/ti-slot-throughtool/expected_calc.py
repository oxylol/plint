"""Hand calculations behind EXPECTED.md (Ti-6Al-4V slot, DMU 50 / 840D sl).

Run: cd apps/cnc-twin && bin/tsafe python examples/ti-slot-throughtool/expected_calc.py

Everything here is closed-form or a 1-D quadrature of a closed-form law with the database values:
Kienzle (db/materials/ti6al4v.yaml), tool and holder geometry (db/tools, db/holders), spindle modes and
the ICS pump (db/machines/dmg-dmu50-3rd-siemens.yaml), Taylor (ti6al4v tool_life), coolant density
(db/coolants/emulsion-8pct.yaml). No simulator code is used.
"""

from __future__ import annotations

import math

import numpy as np

# --- data (units as in the database files) -------------------------------------------------------
D, Z = 25.0, 4                       # mm, teeth (tungaloy-epav12m025c250r04)
R = D / 2.0
RE = 0.8                             # insert nose radius
KC11, MC = 2110.0, 0.22              # N/mm2, ISCAR group 37 'carbide, any coating' (the entry the twin selects
                                     # for a TiAlN-coated carbide tool: coating 'any' scores above 'uncoated')
KC11_MEAS, MC_MEAS = 1253.0, 0.22    # Klippel 2024, uncoated, 40 m/min (measured, lower bound)
KF_RATIO = (0.4, 0.9)                # radial / tangential force ratio band (Klippel kf/kc at h 0.05 mm: 0.88-0.92
                                     # for a 0 deg rake; positive-rake inserts lower it)
E_STEEL = 210000.0                   # N/mm2, tool body and holder (steel)
STICKOUT = 60.0                      # holder nose to tool tip
GAUGE = 75.0                         # holder gauge length (haimer-40-445-25-3)
HOLDER_D = 45.5                      # holder nose diameter
SPINDLE_K = (80.0, 220.0)            # N/um, the two x/y flange modes of the machine file (static compliance adds)
RHO_TI = 4430e-9                     # kg/mm3
RHO_COOLANT = 992.0                  # kg/m3 at 20 degC (emulsion-8pct)
CP_COOLANT = 4022.0                  # J/kgK
P_ICS = 40e5                         # Pa (machine file: ICS up to 40 bar)
HOLES, HOLE_D = 4, 1.0               # coolant holes (tool file, est.)
TAYLOR = dict(C=170.0, n=0.3, a=0.3, b=0.1, f_ref=0.2, ap_ref=2.0, VB=0.3)   # carbide TiAlN in ti6al4v (est.)

ROUGH = dict(n=573.0, fz=0.08, ap=3.9, ae=25.0, feed=183.0, length=132.0, passes=2)
FINISH = dict(n=700.0, fz=0.12, ap=7.8, ae=0.512, feed=336.0, length=132.0, passes=2)


def kienzle_ft(h, b, kc11=KC11, mc=MC):
    """Tangential force per tooth: Fc = kc11 b h^(1-mc) (h_ref = b_ref = 1 mm)."""
    return kc11 * b * np.power(np.maximum(h, 0.0), 1.0 - mc)


def milling_forces(n, fz, ap, ae, kc11=KC11, mc=MC, kr=0.6, climb_side=-1, samples=7200):
    """Mean and peak forces of a 90 deg lead cutter: angle phi of a tooth measured from the +y side
    (tool moving +x, M3), h = fz sin(phi). Returns mean torque, power, mean |F| components."""
    phi = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    if ae >= D:
        lo, hi = 0.0, math.pi                          # full slot
    else:
        # side cut on the -y side (climb with M3, tool moving +x): engagement from pi - phi_e to pi
        phi_e = math.acos(1.0 - 2.0 * ae / D)
        lo, hi = math.pi - phi_e, math.pi
    total_t = np.zeros_like(phi)
    fx = np.zeros_like(phi)
    fy = np.zeros_like(phi)
    for k in range(Z):
        a = (phi + k * 2.0 * math.pi / Z) % (2.0 * math.pi)
        engaged = (a >= lo) & (a <= hi)
        h = fz * np.sin(a)
        ft = np.where(engaged, kienzle_ft(h, ap, kc11, mc), 0.0)
        fr = kr * ft
        # tooth position angle a measured from +y toward +x ... unit vectors in the xy plane
        # radial outward r = (sin a, cos a); tangential (cutting velocity, CW from above) t = (cos a, -sin a)
        rx, ry = np.sin(a), np.cos(a)
        tx, ty = np.cos(a), -np.sin(a)
        # force on the tool: tangential opposes the edge velocity, radial pushes the tool inward
        fx += -ft * tx - fr * rx
        fy += -ft * ty - fr * ry
        total_t += ft
    torque = np.mean(total_t) * R * 1e-3             # N m
    omega = n * 2.0 * math.pi / 60.0
    return {"torque_Nm": torque, "power_W": torque * omega, "Ft_sum_mean_N": float(np.mean(total_t)),
            "Fx_mean_N": float(np.mean(fx)), "Fy_mean_N": float(np.mean(fy)),
            "F_peak_N": float(np.max(np.hypot(fx, fy))), "h_max_mm": fz * (1.0 if ae >= D else
                                                                          math.sin(math.acos(1 - 2 * ae / D)))}


def tool_compliance_um_per_N(load_from_tip):
    """Static compliance at the load point: steel body (d 25) as a cantilever from the holder nose, the
    holder (d 45.5) as a cantilever from the spindle face, plus the spindle flange modes."""
    lt = STICKOUT - load_from_tip
    it = math.pi * D ** 4 / 64.0
    ih = math.pi * HOLDER_D ** 4 / 64.0
    c_tool = lt ** 3 / (3 * E_STEEL * it)
    # holder end: deflection + rotation times the tool lever
    c_holder = GAUGE ** 3 / (3 * E_STEEL * ih) + GAUGE ** 2 * lt / (2 * E_STEEL * ih) * 2 \
        + GAUGE * lt ** 2 / (E_STEEL * ih)
    c_spindle = sum(1.0 / k for k in SPINDLE_K) * 1e-3      # mm/N
    return (c_tool + c_holder + c_spindle) * 1e3           # um/N


def coolant():
    v = math.sqrt(2.0 * P_ICS / RHO_COOLANT)
    area = HOLES * math.pi * (HOLE_D * 1e-3) ** 2 / 4.0
    q = [cd * area * v * 60000.0 for cd in (0.6, 0.8)]       # L/min
    return v, q


def taylor_life_min(vc, f, ap):
    t = TAYLOR
    return (t["C"] / (vc * (f / t["f_ref"]) ** t["a"] * (ap / t["ap_ref"]) ** t["b"])) ** (1.0 / t["n"])


def main():
    print("== cutting data")
    for name, c in (("rough", ROUGH), ("finish", FINISH)):
        vc = math.pi * D * c["n"] / 1000.0
        print(f"{name}: vc {vc:.1f} m/min, vf {c['n'] * Z * c['fz']:.1f} mm/min (programmed F{c['feed']:.0f})")
    print("== forces (Kienzle, ISCAR kc11 2110 / mc 0.22; measured 1253 in brackets)")
    for name, c in (("rough", ROUGH), ("finish", FINISH)):
        for kr in KF_RATIO:
            f = milling_forces(c["n"], c["fz"], c["ap"], c["ae"], kr=kr)
            fm = milling_forces(c["n"], c["fz"], c["ap"], c["ae"], KC11_MEAS, MC_MEAS, kr=kr)
            print(f"{name} kr {kr}: torque {f['torque_Nm']:.2f} Nm [{fm['torque_Nm']:.2f}], power {f['power_W']:.0f} W "
                  f"[{fm['power_W']:.0f}], Fx {f['Fx_mean_N']:.0f} N, Fy {f['Fy_mean_N']:.0f} N, |F|max {f['F_peak_N']:.0f} N, "
                  f"hmax {f['h_max_mm']:.4f} mm")
    print("== static compliance and deflection")
    c_r = tool_compliance_um_per_N(ROUGH["ap"] / 2)
    c_f = tool_compliance_um_per_N(FINISH["ap"] / 2)
    print(f"compliance at ap/2: rough {c_r:.4f} um/N (k {1 / c_r:.1f} N/um), finish {c_f:.4f} um/N (k {1 / c_f:.1f} N/um)")
    for kr in KF_RATIO:
        f = milling_forces(FINISH["n"], FINISH["fz"], FINISH["ap"], FINISH["ae"], kr=kr)
        # pass A also takes the 0.2 mm floor layer over the whole width (a full-slot cut of ap 0.2, load at the tip)
        floor = milling_forces(FINISH["n"], FINISH["fz"], 0.2, D, kr=kr)
        d_wall = abs(f["Fy_mean_N"]) * c_f
        d_a = d_wall + abs(floor["Fy_mean_N"]) * tool_compliance_um_per_N(0.1)
        width = 26.024 - (d_a + d_wall) / 1000.0
        print(f"finish kr {kr}: wall force {f['Fy_mean_N']:.0f} N, floor layer (pass A) {floor['Fy_mean_N']:.0f} N, "
              f"floor torque {floor['torque_Nm']:.2f} Nm -> deflection wall A {d_a:.1f} um, wall B {d_wall:.1f} um, "
              f"slot width {width:.4f} mm")
    f = milling_forces(ROUGH["n"], ROUGH["fz"], ROUGH["ap"], ROUGH["ae"], kr=0.6)
    for kr in KF_RATIO:
        fr = milling_forces(ROUGH["n"], ROUGH["fz"], ROUGH["ap"], ROUGH["ae"], kr=kr)
        print(f"rough kr {kr}: mean side force {fr['Fy_mean_N']:.0f} N -> tool deflection {abs(fr['Fy_mean_N']) * c_r:.0f} um "
              f"(+y on the +X pass, -y on the -X pass)")
    print(f"spindle: torque {f['torque_Nm']:.1f} Nm of 85 Nm S1 (below 1800 rpm), power {f['power_W'] / 1000:.2f} kW of 16 kW S1")
    print("== coolant")
    v, q = coolant()
    print(f"jet velocity (ideal) {v:.1f} m/s, flow through 4 x 1.0 mm at Cd 0.6-0.8: {q[0]:.1f}-{q[1]:.1f} L/min")
    f = milling_forces(ROUGH["n"], ROUGH["fz"], ROUGH["ap"], ROUGH["ae"], kr=0.6)
    dT = f["power_W"] / (RHO_COOLANT * CP_COOLANT * (q[0] / 60000.0))
    print(f"coolant temperature rise if all slotting heat went into it: {dT:.2f} K")
    print("== removal and mass")
    volume = 100.0 * 26.024 * 8.0 - 2 * 100.0 * (RE ** 2 - math.pi * RE ** 2 / 4.0)
    print(f"removed volume {volume:.0f} mm3, mass {volume * RHO_TI * 1e3:.1f} g")
    mrr = ROUGH["ap"] * ROUGH["ae"] * ROUGH["feed"]
    print(f"MRR slotting {mrr:.0f} mm3/min; specific energy {f['power_W'] / (mrr / 60.0):.2f} J/mm3")
    print("== surface")
    frev = Z * FINISH["fz"]
    print(f"wall: per-tooth cusp {FINISH['fz'] ** 2 / (8 * R) * 1000:.3f} um, once-per-rev mark "
          f"{frev ** 2 / (8 * R) * 1000:.2f} um (runout > per-tooth cusp) -> Ra ~ {frev ** 2 / (8 * R) * 1000 / 3.9:.2f} um")
    print("== time")
    t_rough = ROUGH["passes"] * ROUGH["length"] / ROUGH["feed"] * 60.0
    t_fin = FINISH["passes"] * FINISH["length"] / FINISH["feed"] * 60.0
    air = (5.9 + 3.9 + 0.512 + 0.2 + 1.024) / 2000.0 * 60.0
    print(f"feed time rough {t_rough:.1f} s, finish {t_fin:.1f} s, air feeds {air:.2f} s, dwell 2.0 s, "
          f"lower bound {t_rough + t_fin + air + 2.0:.1f} s")
    contact = (ROUGH["passes"] * (ROUGH["length"] - D) / ROUGH["feed"] + FINISH["passes"] * (FINISH["length"] - D)
               / FINISH["feed"]) * 60.0
    print(f"time with the cutter over the part (in cut): ~{contact:.1f} s")
    print("== tool life (Taylor, est. constants)")
    for name, c in (("rough", ROUGH), ("finish", FINISH)):
        vc = math.pi * D * c["n"] / 1000.0
        life = taylor_life_min(vc, c["fz"], c["ap"])
        print(f"{name}: T = {life:.0f} min of edge contact")


if __name__ == "__main__":
    main()
