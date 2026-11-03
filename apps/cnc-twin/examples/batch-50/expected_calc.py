"""Hand calculations behind EXPECTED.md (batch of 50 brackets, Haas VF-2SS).

Run: cd apps/cnc-twin && bin/tsafe python examples/batch-50/expected_calc.py

Kienzle kc11 800 N/mm2, mc 0.25 (db/materials/al6061-t6, ISCAR, carbide any coating); Taylor (same file,
uncoated carbide, est.); tool geometry from db/tools; the ideal dry run of bracket.nc gave 29.33 s with
the probe moves run to their end (cnctwin.controller.dry_run, G81 expanded by hand).
"""

from __future__ import annotations

import math

import numpy as np

KC11, MC = 800.0, 0.25
KC11_HSS = 920.0                      # hss entry of al6061-t6 (est.): the drill is HSS-Co
D, Z, R = 20.0, 3, 10.0
N1 = 10186.0
ROUGH = dict(fz=0.183, ae=5.94, ap=17.0, feed=5592.0, passes=5)
FINISH = dict(fz=0.0838, ae=0.3, ap=17.0, feed=2561.0, passes=1)
CONTACT_LENGTH = 80.0                 # tool centre x -10 .. 70 with the cutter over the stock
E_CARBIDE = 600000.0                  # N/mm2
STICKOUT, GAUGE, NUT_D = 55.0, 70.0, 50.0
SPINDLE_K = 60.0                      # N/um, haas-vf2ss frf modes (x and y, est.)
CLEARANCE = math.radians(12.0)        # primary clearance of the MH35AL (tool file, est.)
TAYLOR = dict(C=1800.0, n=0.3, a=0.3, b=0.1, f_ref=0.2, ap_ref=2.0, VB=0.3)
REMAINING_S = (60.0 - 57.5) * 60.0  # life left on T1 at the start of the batch
DRY_RUN_S = 29.33


def forces(fz, ae, ap, samples=7200):
    phi = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    phi_e = math.acos(1.0 - 2.0 * ae / D)
    lo, hi = math.pi - phi_e, math.pi           # climb on the -y side (tool moving +x, M3)
    ft_sum = np.zeros_like(phi)
    fy = np.zeros_like(phi)
    for k in range(Z):
        a = (phi + k * 2 * math.pi / Z) % (2 * math.pi)
        on = (a >= lo) & (a <= hi)
        h = np.maximum(fz * np.sin(a), 0.0)
        ft = np.where(on, KC11 * ap * h ** (1 - MC), 0.0)
        fr = 0.5 * ft                            # Kr/Kt ~ 0.5 for aluminium (band 0.3-0.7)
        ft_sum += ft
        fy += -ft * (-np.sin(a)) - fr * np.cos(a)
    torque = float(np.mean(ft_sum)) * R * 1e-3
    return {"torque_Nm": torque, "power_W": torque * N1 * 2 * math.pi / 60, "Fy_mean_N": float(np.mean(fy)),
            "Ft_peak_N": float(np.max(ft_sum)), "h_max": fz * math.sin(phi_e)}


def compliance_um_per_N(load_from_tip):
    """Tool (fluted part as d 0.8 D over the stickout) in the ER32 holder (nut d50 over the gauge length)
    in the spindle (machine file modes)."""
    lt = STICKOUT - load_from_tip
    it = math.pi * (0.8 * D) ** 4 / 64.0
    ih = math.pi * NUT_D ** 4 / 64.0
    e_holder = 210000.0
    c = lt ** 3 / (3 * E_CARBIDE * it) + GAUGE ** 3 / (3 * e_holder * ih) + GAUGE ** 2 * lt / (e_holder * ih) \
        + GAUGE * lt ** 2 / (e_holder * ih)
    return c * 1e3 + 1.0 / SPINDLE_K


def main():
    vc = math.pi * D * N1 / 1000
    print(f"vc {vc:.0f} m/min; feeds: rough {N1 * Z * ROUGH['fz']:.0f} mm/min, finish {N1 * Z * FINISH['fz']:.0f}")
    c = compliance_um_per_N(ROUGH["ap"] / 2)
    print(f"tool-holder-spindle compliance at ap/2: {c:.4f} um/N (k {1 / c:.1f} N/um)")
    for name, p in (("rough", ROUGH), ("finish", FINISH)):
        f = forces(p["fz"], p["ae"], p["ap"])
        mrr = p["ae"] * p["ap"] * p["feed"]
        print(f"{name}: h_max {f['h_max']:.3f} mm, torque {f['torque_Nm']:.2f} Nm, power {f['power_W'] / 1000:.2f} kW, "
              f"MRR {mrr / 1000:.0f} cm3/min, wall-normal force {f['Fy_mean_N']:.0f} N -> deflection "
              f"{abs(f['Fy_mean_N']) * c:.1f} um, peak summed Ft {f['Ft_peak_N']:.0f} N")
    print(f"radial wear of T1 at VB 0.15: VB tan(alpha) = {0.15 * math.tan(CLEARANCE) * 1000:.1f} um")
    t_rough = ROUGH["passes"] * CONTACT_LENGTH / ROUGH["feed"] * 60
    t_fin = CONTACT_LENGTH / FINISH["feed"] * 60
    t_contact = t_rough + t_fin
    t_feed = ROUGH["passes"] * (84 / ROUGH["feed"] + 22 / 3000) * 60 + (84 / FINISH["feed"] + 22 / 3000) * 60
    print(f"T1 per part: cutter over the stock {t_contact:.2f} s, all G01 feed {t_feed:.2f} s")
    print(f"T1 life left {REMAINING_S:.0f} s: expires in part {math.ceil(REMAINING_S / t_contact)} by in-cut time "
          f"(sister T11 from part {math.ceil(REMAINING_S / t_contact) + 1}), in part {math.ceil(REMAINING_S / t_feed)} "
          f"by feed time (T11 from part {math.ceil(REMAINING_S / t_feed) + 1})")
    phi_e = math.acos(1.0 - 2.0 * ROUGH["ae"] / D)
    edge_contact = t_rough * phi_e / (2 * math.pi) + t_fin * math.acos(1 - 2 * FINISH["ae"] / D) / (2 * math.pi)
    t_life = (TAYLOR["C"] / (vc * (ROUGH["fz"] / TAYLOR["f_ref"]) ** TAYLOR["a"]
                             * (ROUGH["ap"] / TAYLOR["ap_ref"]) ** TAYLOR["b"])) ** (1 / TAYLOR["n"])
    print(f"Taylor life at the roughing conditions {t_life:.1f} min of edge contact; edge contact per part "
          f"{edge_contact:.2f} s -> {edge_contact / 60 / t_life * 100:.3f} % of life, dVB ~ "
          f"{TAYLOR['VB'] * edge_contact / 60 / t_life * 1000:.2f} um per part (linear)")
    # drilling
    n2, f2 = 4213.0, 0.4
    fc = 0.5 * KC11_HSS * (6.8 / 2) * (f2 / 2 * math.sin(math.radians(65))) ** (1 - MC) * 2
    print(f"drill: vc {math.pi * 6.8 * n2 / 1000:.0f} m/min, vf {n2 * f2:.0f} mm/min, torque ~ "
          f"{fc * 6.8 / 4 / 1000:.2f} Nm, thrust ~ {0.5 * fc * 2:.0f} N (0.5-1.0 x Fc)")
    # takt
    operator = 12 + 6 + 6 + 2 * 3
    cycle = DRY_RUN_S - 2.4 + 2.0
    print(f"cycle per part ~ {cycle:.1f} s (dry run {DRY_RUN_S} - probe overtravel 2.4 + ramps 2.0); operator "
          f"{operator} s; takt ~ {cycle + operator:.1f} s; inspection every 10th part 45 s")
    batch = 50 * (cycle + operator) + 6 * 45
    print(f"batch ~ {batch / 60:.1f} min ({50 * 3600 / batch:.1f} parts per hour)")
    v_removed = 60 * 30 * 17 + 2 * math.pi * 3.4 ** 2 * 8
    print(f"chips per part {v_removed:.0f} mm3 = {v_removed * 2700e-9 * 1000:.1f} g; part {(60 * 50 * 25 - v_removed) * 2.7e-6:.3f} kg")


if __name__ == "__main__":
    main()
