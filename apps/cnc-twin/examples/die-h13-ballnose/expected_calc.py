"""Hand calculations behind EXPECTED.md (H13 die cavity, DMU 50 / TNC 640).

Run: cd apps/cnc-twin/examples/die-h13-ballnose && ../../bin/tsafe python expected_calc.py

Kienzle kc11 3000 N/mm2, mc 0.25 (db/materials/h13-50hrc, Walter H1); extended Taylor for TiAlN carbide in
the same file (est.: T = 15 min at vc 100, fz 0.1, ap 0.5); tool, holder and machine data from the
project and the database files; cavity area from part.step; toolpath numbers from toolpath_stats.json
(make_program.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

HERE = Path(__file__).resolve().parent
KC11, MC = 3000.0, 0.25
R = 5.0
A_HALF, B_HALF, DEPTH, R_RIM = 24.0, 16.0, 9.6, 2.0
E_C, E_S = 600000.0, 210000.0                     # carbide (submicron 10 % Co), steel holder
# TEB100E4L-20/30C10CFH72 at stickout 40 (db/tools): flutes 0-20 (core 6.0, taken as d 8 equivalent),
# relieved neck d 9.8 to 30, shank d 10 to 40; HAIMER 40.445.10.3 nose d 26.5 over A 65 (project, inline)
TOOL_SEGMENTS = [(0.0, 20.0, 8.0), (20.0, 30.0, 9.8), (30.0, 40.0, 10.0)]
HOLDER = {"d": 26.5, "length": 65.0}
SPINDLE_K = {"x": 80.0, "y": 220.0}               # N/um, dmg-dmu50-3rd flange modes (est.)
TAYLOR = dict(C=197.0, n=0.25, a=0.3, b=0.1, f_ref=0.1, ap_ref=0.5, VB=0.2)
CLEARANCE_DEG = 8.0                               # TEB primary clearance (db, est.)
V_REMOVED = 427.8                                 # mm3, make_cad.py (blank - part)
RHO = 7800e-9                                     # kg/mm3
CP = 460.0
INSERT_KG = 2.45
RUNOUT_UM = 3.0
PIVOT_B_UM = 10.0                                 # machine file pivot_errors B position z
TOOL_CHANGE_S = 5.5                               # chip-to-chip (machine file, est.)
SWIVEL_S = 4 * 2.0                                # PLANE ... TURN MB MAX: retract + B 15 deg at 180 deg/s + C, 4 times (est.)


def cavity_area():
    """Area of the non-planar faces of part.step (paraboloid + rim fillet)."""
    reader = STEPControl_Reader()
    reader.ReadFile(str(HERE / "part.step"))
    reader.TransferRoots()
    shape = reader.OneShape()
    area = 0.0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face(exp.Current())
        if BRepAdaptor_Surface(face).GetType() != GeomAbs_Plane:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area += props.Mass()
        exp.Next()
    return area


def compliance_um_per_n(lever_from_tip=1.0):
    """Static compliance at the ball (force lever_from_tip above the tip): stepped carbide cantilever in a
    rigid-ended shrink chuck, the chuck as a steel cantilever from the spindle nose, the spindle's softer
    flange mode in series."""
    stick = TOOL_SEGMENTS[-1][1]
    c = 0.0
    for z0, z1, d in TOOL_SEGMENTS:                # x measured from the chuck face towards the tip
        i = math.pi * d ** 4 / 64.0
        a, b = stick - z1, stick - z0               # segment from a to b (from the chuck face)
        lever = stick - lever_from_tip
        a, b = max(a, 0.0), min(b, lever)
        if b > a:
            c += ((lever - a) ** 3 - (lever - b) ** 3) / (3.0 * E_C * i)
    ih = math.pi * HOLDER["d"] ** 4 / 64.0
    g, lt = HOLDER["length"], stick - lever_from_tip
    c += (g ** 3 / 3.0 + g ** 2 * lt + g * lt ** 2) / (E_S * ih)
    return c * 1e3 + 1.0 / min(SPINDLE_K.values())


def ball_chip(fz, ap, d_eff):
    """Ball nose as a local cylinder of diameter d_eff in an engagement ap: h_max and h_mean."""
    ratio = min(ap / d_eff, 0.5)
    h_max = 2.0 * fz * math.sqrt(ratio * (1.0 - ratio))
    return h_max, 0.5 * h_max


def taylor_life_min(vc, fz, ap):
    t = TAYLOR
    return (t["C"] / (vc * (fz / t["f_ref"]) ** t["a"] * (ap / t["ap_ref"]) ** t["b"])) ** (1.0 / t["n"])


def main():
    s = json.loads((HERE / "toolpath_stats.json").read_text())
    area = cavity_area()
    rough, fin = s["tools"]["rough"], s["tools"]["finish"]
    print(f"cavity + rim area {area:.0f} mm2; removed {V_REMOVED} mm3 -> mean allowance {V_REMOVED / area:.3f} mm")
    fin_stock = rough["stock_mm"]
    v_fin = area * (fin_stock + rough["scallop_um"]["p50"] / 1000.0 / 3.0)
    v_rough = V_REMOVED - v_fin
    print(f"volumes: roughing {v_rough:.0f} mm3, finishing {v_fin:.0f} mm3 (stock {fin_stock} + mean of the "
          f"roughing cusps h/3, h p50 {rough['scallop_um']['p50']} um)")
    k_tool = 1.0 / compliance_um_per_n()
    print(f"static stiffness at the ball (stepped carbide in the chuck + chuck + spindle {min(SPINDLE_K.values())} N/um): "
          f"{k_tool:.1f} N/um ({compliance_um_per_n():.3f} um/N)")
    t_total = 0.0
    for kind, t, ap_list, vol in (("rough", rough, (0.24,), v_rough), ("finish", fin, (0.06, 0.08, 0.10), v_fin)):
        n, f = t["S"], t["F"]
        fz = f / (n * 4)
        vc50 = t["vc_m_min"]["p50"]
        d_eff = vc50 * 1000 / (math.pi * n)
        cut_s = t["cut_path_mm"] / f * 60.0
        mrr = vol / cut_s                                   # mm3/s
        print(f"{kind}: S{n} F{f} fz {fz:.4f}; vc p5/p50/p80/max {t['vc_m_min']}; contact angle {t['contact_angle_deg']}")
        print(f"  {t['lines']} lines, stepover {t['stepover_mm']}, predicted scallop {t['scallop_um']} um; "
              f"cut {t['cut_path_mm']} mm = {cut_s:.0f} s, feed time with links {t['feed_time_s']} s")
        for ap in ap_list:
            h_max, h_mean = ball_chip(fz, ap, d_eff)
            kc = KC11 * h_mean ** (-MC)
            power = kc * mrr * 1e-3 * (ap / sum(ap_list) * len(ap_list))
            ft = power / (vc50 / 60.0)
            print(f"  ap {ap}: d_eff {d_eff:.2f}, h_max {h_max * 1000:.1f} um, h_mean {h_mean * 1000:.1f} um -> kc "
                  f"{kc:.0f} N/mm2; MRR {mrr * ap / sum(ap_list) * len(ap_list):.2f} mm3/s; cutting power {power:.0f} W; "
                  f"mean tangential force {ft:.1f} N -> deflection {ft / k_tool:.2f} um (x1.0-2.0 for the resultant)")
        ap_mid = ap_list[len(ap_list) // 2]
        life = taylor_life_min(vc50, fz, ap_mid)
        used = cut_s / 60.0
        # the twin calibrates its wear law to the Taylor life as continuous edge contact (docs/models/wear.md
        # section 17): an edge wears only while it is in the material, a fraction of the revolution
        r_depth = ap_mid / math.sin(math.radians(t["contact_angle_deg"]["p50"]))
        d_loc = 2 * R * math.sin(math.radians(t["contact_angle_deg"]["p50"]))
        engaged = math.acos(1.0 - 2.0 * min(r_depth / d_loc, 0.5)) / (2 * math.pi)
        for basis, minutes in (("edge contact (the twin's basis)", used * engaged), ("in-cut time", used)):
            dvb = TAYLOR["VB"] * minutes / life
            print(f"  Taylor life {life:.0f} min at p50 vc; {basis}: {minutes:.2f} min = {minutes / life * 100:.2f} % "
                  f"-> dVB ~ {dvb * 1000:.1f} um (linear); radial VB tan {CLEARANCE_DEG:.0f} deg = "
                  f"{dvb * math.tan(math.radians(CLEARANCE_DEG)) * 1000:.2f} um")
        print(f"  edge engaged {engaged * 100:.1f} % of a revolution (local circle d {d_loc:.2f} at psi p50, "
              f"radial depth ap / sin psi = {r_depth:.3f})")
        t_total += t["feed_time_s"]
    # surface
    h = fin["scallop_um"]
    fz = fin["F"] / (fin["S"] * 4)
    f_rev = fin["F"] / fin["S"]
    print(f"finishing scallop p50/p95/max {h['p50']}/{h['p95']}/{h['max']} um -> Ra (parabolic cusps, Ra = h/3.9) "
          f"{h['p50'] / 3.9:.2f}/{h['p95'] / 3.9:.2f}/{h['max'] / 3.9:.2f} um")
    print(f"feed direction: per-tooth cusp fz^2/8R = {fz ** 2 / (8 * R) * 1000:.3f} um; with runout {RUNOUT_UM} um "
          f"one tooth makes the surface, marks every f_rev = {f_rev:.3f} mm, cusp f_rev^2/8R = "
          f"{f_rev ** 2 / (8 * R) * 1000:.2f} um")
    # analytic checks of the curvature correction
    for name, rho in (("bottom, y-section", B_HALF ** 2 / (2 * DEPTH)), ("bottom, x-section", A_HALF ** 2 / (2 * DEPTH)),
                      ("rim fillet (convex)", -R_RIM)):
        fac = rho / (rho - R) if rho > 0 else -rho / (-rho + R)
        print(f"  {name}: surface radius {abs(rho):.1f} mm -> scallop factor (centre spacing) {fac:.2f}")
    # machine
    nx = 0.8 / math.sqrt(1 + 0.8 ** 2)
    dx = PIVOT_B_UM * math.sin(math.radians(15))
    print(f"B pivot error {PIVOT_B_UM} um (z): each half shifted {dx:.1f} um in x (opposite signs), "
          f"normal error <= {dx * nx:.1f} um at the x-ends (n_x {nx:.2f}); 0 at the joint x = 0 (dz/dx = 0)")
    print(f"feed time total {t_total:.0f} s + 2 tool changes {2 * TOOL_CHANGE_S:.0f} s + 4 swivels ~{SWIVEL_S:.0f} s "
          f"+ approaches -> about {t_total + 2 * TOOL_CHANGE_S + SWIVEL_S + 15:.0f} s")
    print(f"hard-milled mass {V_REMOVED * RHO * 1000:.2f} g; energy at kc 10-13 kN/mm2: {V_REMOVED * 10 / 1000:.1f}-"
          f"{V_REMOVED * 13 / 1000:.1f} kJ -> insert ({INSERT_KG} kg) <= {V_REMOVED * 13 / (INSERT_KG * CP):.1f} K if all "
          f"of it stayed in the part")
    print(f"travel used: {s['travel']}")


if __name__ == "__main__":
    main()
