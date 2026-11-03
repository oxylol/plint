"""Turning on the polar field: a sharp straight edge against Kienzle's force (Kienzle 1952), the uncut
chip h = f sin(kappa_r) of the steady revolution and the chip volume of a revolution."""

import math

import numpy as np
import pytest

from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting.drive import Driver
from tests.unit.cutting.cutfix import linear_layout

KC = dict(kc11=1.8e9, mc=0.25, ratio_r=0.45, ratio_a=0.3, h_ref_m=1e-3)
R_W = 20e-3              # work radius
AP = 1.0e-3              # depth of cut
F = 0.2e-3               # feed per revolution
OMEGA_C = -200.0         # work rotation (the tool passes the work along its cutting direction)
DT = 40e-6


def _pose(t):
    """Tool in the rotating work frame S: tip at radius R_W - AP, feeding along -z (x_T into the work)."""
    phi = -OMEGA_C * t
    c, s = math.cos(phi), math.sin(phi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    vf = F * abs(OMEGA_C) / (2 * math.pi)
    p0 = np.array([R_W - AP, 0.0, 2e-3 - vf * t])
    tip = Rz @ p0
    w = np.array([0.0, 0.0, -OMEGA_C])
    v = np.cross(w, tip) + Rz @ np.array([0.0, 0.0, -vf])
    return Rz @ np.diag([-1.0, -1.0, 1.0]), tip, v, w


def _run(kappa_r, revs=2.5):
    tool = AN.straight_turning_tool(kappa_r, 1.6 * AP / math.sin(kappa_r), 48)
    ct, table, _ = linear_layout(tool, {"default": KC}, OMEGA_C)
    polar = A.empty_polar(720, 600)
    polar.r_out_m[:] = R_W
    polar.geom[:] = (2 * math.pi / 720, -4e-3, 10e-3 / 600, 0.0)
    drv = Driver(ct, table, 1, 48, rotating=False, polar=polar)
    drv.minor_side = 1                       # feeding toward -z: the surface behind the tip is at its radius
    n_rev = int(round(2 * math.pi / abs(OMEGA_C) / DT))
    drv.run_turning(int(revs * n_rev), OMEGA_C, DT, _pose)
    return tool, ct, drv, n_rev


def _strip_force(kr, n=20001):
    """Kienzle per unit edge length on the element strips of the parallelogram chip: the depth along -n from
    the edge to the previous revolution's edge, truncated by the uncut surface (R_W) near the outer end."""
    r = np.linspace(R_W - AP, R_W, n)
    h = np.full(n, F * math.sin(kr))
    if math.cos(kr) > 1e-12:
        h = np.minimum(h, (R_W - r) / math.cos(kr))
    q = KC["kc11"] * h * (np.maximum(h, 1e-30) / KC["h_ref_m"]) ** (-KC["mc"])
    return float(np.trapezoid(q, r)) / math.sin(kr)


@pytest.mark.parametrize("kappa_deg", [90.0, 75.0, 45.0])
def test_straight_edge_against_kienzle(kappa_deg):
    kr = math.radians(kappa_deg)
    tool, ct, drv, n_rev = _run(kr)
    t = np.array(drv.record.t_s)[-n_rev // 2:]
    # cutting force: the component of the force on the tool against its motion relative to the work
    Fc = []
    for k in range(len(t)):
        R, tip, v, w = _pose(t[k])
        F_S = np.array(drv.record.force_S[-n_rev // 2 + k])
        Fc.append(-F_S @ v / np.linalg.norm(v))
    strips = _strip_force(kr)
    assert np.mean(Fc) == pytest.approx(strips, rel=5e-3)
    # Kienzle's uniform h = f sin(kappa_r) over b = ap / sin(kappa_r) covers the whole parallelogram chip; the
    # element strips miss the corner triangle at the tip (relative area f sin cos / (2 ap)) and are cut short
    # by the uncut surface at the outer end
    kienzle = AN.kienzle_turning(KC["kc11"], KC["mc"], F, AP, kr)
    assert abs(np.mean(Fc) / kienzle - 1.0) < F * math.sin(kr) * math.cos(kr) / AP + 5e-3
    # steady uncut chip of the engaged elements away from the uncut surface (a radial edge sees the polar
    # field's z resolution: its previous wall is a step in z)
    h = np.array(drv.record.h[-1])
    r_e = R_W - AP + np.abs(ct.p_T_m[:, 0])
    inner = (h > 0.0) & (R_W - r_e > F * math.sin(kr) * math.cos(kr) + ct.dS_m)
    tol = 5e-3 if kappa_deg < 89.0 else drv.polar.geom[2] / F
    assert inner.sum() > 20
    assert np.allclose(h[inner], F * math.sin(kr), rtol=tol)
    # engaged edge length = AP / sin(kappa_r) (the element straddling the uncut surface is cut partly)
    assert ct.dS_m[h > 0.0].sum() == pytest.approx(AP / math.sin(kr), abs=ct.dS_m[0])


def test_chip_volume_of_a_revolution():
    kr = math.radians(75.0)
    tool, ct, drv, n_rev = _run(kr, revs=2.0)
    v0 = drv.acc.scalars[A.ACC_CHIP_VOLUME_M3]
    drv.run_turning(n_rev, OMEGA_C, DT, _pose, t0_s=2.0 * n_rev * DT, record=False)
    vol = drv.acc.scalars[A.ACC_CHIP_VOLUME_M3] - v0
    # a ring of width F and depth AP at the mean radius, less the corner triangle at the tip that no element
    # strip (normal to the edge) reaches: area f^2 sin(kappa_r) cos(kappa_r) / 2
    ring = F * AP * 2 * math.pi * (R_W - 0.5 * AP)
    assert vol == pytest.approx(ring * (1.0 - 0.5 * F * math.sin(kr) * math.cos(kr) / AP), rel=2e-3)
    a = drv.acc.scalars
    heat = a[A.ACC_HEAT_SHEAR_J] + a[A.ACC_HEAT_RAKE_J] + a[A.ACC_HEAT_FLANK_J] + a[A.ACC_PLOUGHING_J] + \
        a[A.ACC_PROCESS_DAMPING_J]
    assert a[A.ACC_SHAFT_WORK_J] + a[A.ACC_FEED_WORK_J] == pytest.approx(heat, rel=1e-9)
