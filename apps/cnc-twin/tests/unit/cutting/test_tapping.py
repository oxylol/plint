"""Tapping: chamfer teeth share the thread depth (cutting taps), a synchronisation error through a floating
holder's axial float and stiffness, and the forming-tap torque from the displaced volume."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_PERIPHERAL, EDGE_THREAD_FLANK
from cnctwin.cutting import abi as A
from cnctwin.cutting.drive import Driver
from cnctwin.cutting.memory import BoreStock
from tests.unit.cutting.cutfix import built, linear_layout, rev_steps

OMEGA = -209.4          # 2000 rpm, M03
DT = 20e-6
# linear edge-force coefficients with a ploughing ramp below h_min (no force jump where the full-thread lands
# follow the thread they cut: h = 0 there)
TAP = dict(Ktc=2.0e9, Krc=0.9e9, Kac=0.2e9, Kte=30e3, Kre=40e3, Kae=2e3, h_min_m=2e-6)


def _tap_driver(kind, coeffs, r_hole, feed_ratio=1.0, float_holder=None):
    g = built(kind)
    nt, ne = g.edge.elem_r_m.shape
    ct, table, rows = linear_layout(g, coeffs, OMEGA, float_holder=float_holder)
    drv = Driver(ct, table, nt, ne)
    pitch = g.info["lead"]
    vz = -feed_ratio * pitch * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([0.0, 0.0, vz * t])
    # the tap is inside its pre-drilled hole: every station's circle meets the bore wall or the thread
    drv.set_full_circle()
    # memory as left by the synchronised tap (the thread the chamfer has cut so far); where the tap has not cut
    # (inside the bore) the surface is the bore wall, from the field planes
    sync_tip = lambda t: np.array([0.0, 0.0, -pitch * abs(OMEGA) / (2 * math.pi) * t])
    drv.prefill_memory(sync_tip, 0.0, OMEGA)
    radius = np.hypot(drv.mem.point_S_m[..., 0], drv.mem.point_S_m[..., 1])
    drv.mem.valid[radius < r_hole] = 0
    drv.set_planes(BoreStock(r_hole))
    used = [A.row_station(ct, r) for r in range(drv.n_rows)]
    assert all(drv.refill.valid[r].all() for r in range(drv.n_rows) if ct.st_ptr[used[r] + 1] > ct.st_ptr[used[r]])
    return g, ct, drv, tip


def test_chamfer_teeth_share_the_thread_depth():
    """Each land of the chamfer cuts the radial rise of the cone over one land pitch (Dogra, Kapoor & DeVor
    2002): h = rise sin(kappa) on the crest; the first engaged teeth cut from the bore wall; the full-thread
    lands follow the helix the chamfer cut (no chip beyond the memory's chord sagitta); the lands share the
    load; the chip volume of a revolution is the thread groove between the bore and the major diameter."""
    r_hole = 4.25e-3
    g, ct, drv, tip = _tap_driver("tap_cutting", {"default": TAP}, r_hole=r_hole)
    n = rev_steps(OMEGA, DT)
    drv.run_rotating(n, OMEGA, DT, tip, record=False)
    v0 = drv.acc.scalars[A.ACC_CHIP_VOLUME_M3]
    ea0 = drv.state.elem_acc[:, A.ELEM_INT_H_DT].copy()
    rec = drv.run_rotating(n, OMEGA, DT, tip, t0_s=n * DT)
    h_max = np.array(rec.h).max(axis=0)
    L_ch, R_, r_tip, N, lead = (g.info[k] for k in ("chamfer_length", "R", "r_tip", "flutes", "lead"))
    rise = (R_ - r_tip) * (lead / N) / L_ch          # radial rise of the cone from one land to the next
    kap = math.atan2(L_ch, R_ - r_tip)               # the cone normal is (sin kappa, -cos kappa) in (r, z)
    nb = drv.mem.point_S_m.shape[1]
    sag = R_ * (1.0 - math.cos(math.pi / nb))        # chord of the memory stencil between bin centres
    n_r = np.hypot(ct.n_T[:, 0], ct.n_T[:, 1])
    cone = (ct.kind == EDGE_PERIPHERAL) & (np.abs(ct.n_T[:, 2] + math.cos(kap)) < 0.02) & (ct.valid == 1) \
        & (ct.p_T_m[:, 2] < L_ch) & (ct.r_xy_m < R_ - 2e-5)
    full_engaged = cone & (ct.r_xy_m > r_hole + rise)
    assert full_engaged.sum() > 30
    assert np.allclose(h_max[full_engaged], rise * math.sin(kap), rtol=0.01, atol=sag)
    # the first engaged crest elements: their predecessor passed inside the bore and handed the bore wall on, at
    # its own height: the tilted cone normal sees the height mismatch dz of the lands' discretisations
    # (|n_z| dz, dz up to one crest element)
    first = cone & (ct.r_xy_m > r_hole + 0.1 * rise) & (ct.r_xy_m < r_hole + 0.9 * rise)
    assert first.sum() >= 3
    assert np.allclose(h_max[first], (ct.r_xy_m[first] - r_hole) * n_r[first], atol=0.1 * rise)
    inside = cone & (ct.r_xy_m < r_hole - 1e-6)            # in the bore: no chip (same bound)
    assert np.all(h_max[inside] < (ct.r_xy_m[inside] - r_hole) * n_r[inside] + 0.1 * rise)
    full = (ct.p_T_m[:, 2] > L_ch + lead) & (ct.valid == 1)
    # (the stencil's chord is corrected to second order; what is left are nanometre residues of the partial
    # renewal between h_min and 2 h_min at the end of the chamfer)
    assert np.all(h_max[full] < 1e-4 * rise)
    W = np.array(rec.wrench)
    assert W[:, 5].mean() > 0.0                 # resists the rotation
    # the lands carry about the same crest load (their discretisations differ where the cone clips them)
    hdt = drv.state.elem_acc[:, A.ELEM_INT_H_DT] - ea0
    per_tooth = [np.sum((hdt * ct.db_m)[cone & (ct.tooth == j)]) for j in range(N)]
    assert max(per_tooth) / min(per_tooth) < 1.10
    # chip volume of a revolution vs the groove of one pitch between the bore and the major diameter (60 deg
    # thread, crest flat P / 8): the convex tooth corners are counted twice by the element normals (error bound
    # in docs/models/cutting.md)
    vol = drv.acc.scalars[A.ACC_CHIP_VOLUME_M3] - v0
    w = lambda r: lead / 8 + 2 * (R_ - r) * math.tan(math.radians(30))
    rr = np.linspace(r_hole, R_, 2001)
    groove = np.trapezoid(w(rr) * 2 * math.pi * rr, rr)
    assert 1.0 < vol / groove < 1.2


def test_synchronisation_error_through_the_floating_holder():
    """A 2 % feed / lead mismatch: a rigid holder forces it into the thread flanks (they are recut, the axial
    load grows); a floating holder lets the tap follow its thread, and the spring then carries the axial
    cutting load (k u = F_z, the implicit static equilibrium of the float)."""
    n = rev_steps(OMEGA, DT)
    out = {}
    for label, ratio, holder in (("sync", 1.0, None), ("rigid", 1.02, None), ("float", 1.02, (2e6, 2e-3))):
        g, ct, drv, tip = _tap_driver("tap_cutting", {"default": TAP}, r_hole=4.25e-3, feed_ratio=ratio,
                                      float_holder=holder)
        rec = drv.run_rotating(2 * n, OMEGA, DT, tip)
        W = np.array(rec.wrench)
        out[label] = (W[-n // 4:, 2].mean(), W[-n // 4:, 5].mean(), float(drv.state.misc[A.MS_FLOAT_U]))
    fz_sync, tq_sync, _ = out["sync"]
    fz_rigid, tq_rigid, _ = out["rigid"]
    fz_float, tq_float, u_float = out["float"]
    assert u_float > 0.0                                   # the tap lags its holder (pushed back by the thread)
    assert fz_float == pytest.approx(2e6 * u_float, rel=0.1)
    # a rigid tap is forced into one flank of its thread: it recuts it, axial load and torque rise
    assert fz_rigid > fz_sync and tq_rigid > tq_sync
    assert fz_float < fz_rigid


def test_forming_tap_work_is_pressure_times_displaced_volume():
    sigma = 400e6
    coeffs = {"forming": dict(flow_stress=sigma, mu=0.0), "default": TAP}
    g, ct, drv, tip = _tap_driver("tap_forming", coeffs, r_hole=3.69e-3)
    assert np.all(ct.forming[ct.valid == 1] == 1)
    n = rev_steps(OMEGA, DT)
    drv.run_rotating(n // 2, OMEGA, DT, tip, record=False)
    w0 = drv.acc.scalars[A.ACC_SHAFT_WORK_J]
    v0 = drv.cacc.scalars[A.CA_DISPLACED_M3]
    drv.run_rotating(n, OMEGA, DT, tip, t0_s=(n // 2) * DT, record=False)
    work = drv.acc.scalars[A.ACC_SHAFT_WORK_J] - w0
    vol = drv.cacc.scalars[A.CA_DISPLACED_M3] - v0
    p = (2.0 + math.pi) * sigma / math.sqrt(3.0)
    assert vol > 0.0 and drv.acc.scalars[A.ACC_CHIP_VOLUME_M3] == 0.0
    # frictionless lobes: the forming work of a revolution is the pressure times the displaced volume
    assert work == pytest.approx(p * vol, rel=0.02)
    # all the forming work is booked as ploughing heat (no chip, no shear zone)
    assert drv.acc.scalars[A.ACC_HEAT_SHEAR_J] == 0.0
    assert drv.cacc.scalars[A.CA_FORMING_J] > 0.0
    # with friction the torque rises above the displacement term
    coeffs_mu = {"forming": dict(flow_stress=sigma, mu=0.15), "default": TAP}
    g2, ct2, drv2, tip2 = _tap_driver("tap_forming", coeffs_mu, r_hole=3.69e-3)
    drv2.run_rotating(n // 2 + n, OMEGA, DT, tip2, record=False)
    assert drv2.acc.scalars[A.ACC_SHAFT_WORK_J] > drv.acc.scalars[A.ACC_SHAFT_WORK_J] * 1.05
