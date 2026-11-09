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


def _nose_pose(f, ap):
    def pose(t):
        phi = -OMEGA_C * t
        c, s = math.cos(phi), math.sin(phi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        vf = f * abs(OMEGA_C) / (2 * math.pi)
        tip = Rz @ np.array([R_W - ap, 0.0, 2e-3 - vf * t])
        w = np.array([0.0, 0.0, -OMEGA_C])
        return Rz @ np.diag([-1.0, -1.0, 1.0]), tip, np.cross(w, tip) + Rz @ np.array([0.0, 0.0, -vf]), w
    return pose


def test_edge_curvature_of_a_nose_and_of_straight_edges():
    """abi.edge_curvature: 1 / r_eps on the nose arc (exact for equal elements on a circle), 0 on the straight major
    and minor edges, in between at the two junctions."""
    r_eps = 0.8e-3
    tool = AN.nose_turning_tool(math.radians(95.0), r_eps, 1.0e-3, 0.05e-3)
    ct, table, _ = linear_layout(tool, {"default": KC}, OMEGA_C)
    curv = A.edge_curvature(ct)
    on_arc = np.abs(np.linalg.norm(ct.p_T_m - np.array([-r_eps, 0.0, 0.0]), axis=1) - r_eps) < 1e-12
    straight = ~on_arc
    inner_arc = on_arc & np.roll(on_arc, 1) & np.roll(on_arc, -1)
    inner_straight = straight & np.roll(straight, 1) & np.roll(straight, -1)
    inner_straight[[0, -1]] = straight[[0, -1]]
    assert inner_arc.sum() > 20 and inner_straight.sum() > 10
    assert np.allclose(curv[inner_arc], 1.0 / r_eps, rtol=1e-9)
    assert np.allclose(curv[inner_straight], 0.0, atol=1e-9)
    assert np.all((curv >= -1e-9) & (curv <= 1.0 / r_eps + 1e-6))
    # the packed row the kernels read
    k = np.asarray(ct.st_elem)
    assert np.allclose(ct.pk[:, A.PK_CURV], curv[k])


def test_nose_radius_chip_volume_of_a_revolution():
    """A nose-radius insert at a feed comparable to its radius (r_eps 0.8 mm, f 0.3 mm/rev, ap 1 mm): the chip of a
    revolution is the material between the profile and the previous revolution's (analytic.nose_profile_depth).
    The element strips narrow behind the curved nose, db (1 - h curv / 2) (abi.edge_curvature: h db alone
    overstated the chip by up to h / 2 r_eps = 19 % on the nose, +5 % on the whole chip), and sweep about the axis at
    their centroid (Pappus)."""
    kr, km, r_eps, f, ap = math.radians(75.0), math.radians(5.0), 0.8e-3, 0.3e-3, 1.0e-3
    depth_major = ap - r_eps + r_eps * math.cos(kr)
    tool = AN.nose_turning_tool(kr, r_eps, 1.5 * depth_major / math.sin(kr) + 0.2e-3, 0.04e-3, km, minor_len_m=0.8e-3)
    ct, table, _ = linear_layout(tool, {"default": KC}, OMEGA_C)
    polar = A.empty_polar(720, 900)
    polar.r_out_m[:] = R_W
    polar.geom[:] = (2 * math.pi / 720, -4e-3, 9e-3 / 900, 0.0)
    drv = Driver(ct, table, 1, ct.p_T_m.shape[0], rotating=False, polar=polar)
    drv.minor_side = 0
    pose = _nose_pose(f, ap)
    n_rev = int(round(2 * math.pi / abs(OMEGA_C) / DT))
    drv.run_turning(2 * n_rev, OMEGA_C, DT, pose, record=False)
    v0 = drv.acc.scalars[A.ACC_CHIP_VOLUME_M3]
    drv.run_turning(n_rev, OMEGA_C, DT, pose, t0_s=2.0 * n_rev * DT, record=False)
    vol = drv.acc.scalars[A.ACC_CHIP_VOLUME_M3] - v0
    z = np.linspace(-4e-3, 3e-3, 700001)
    cur = np.clip(AN.nose_profile_depth(z, ap, r_eps, kr, km), 0.0, ap)
    prev = np.clip(AN.nose_profile_depth(z - f, ap, r_eps, kr, km), 0.0, ap)
    gap = np.clip(cur - prev, 0.0, None)
    exact = 2 * math.pi * float(np.trapezoid((R_W - 0.5 * (cur + prev)) * gap, z))
    area, _ = AN.nose_chip_area(f, ap, r_eps, kr, km)
    print("chip of a revolution", vol, "exact", exact, "ratio", vol / exact, "area", area)
    assert vol == pytest.approx(exact, rel=8e-3)
