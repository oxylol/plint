"""Coefficient tables (linearisation, continuity, forming rows), the spindle model (curves, S1 / S6
derating, winding temperature, stall), and the surface memory's field planes and refill."""

import math

import numpy as np
import pytest

from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting import coefficients as CO
from cnctwin.cutting import memory as MEM
from cnctwin.cutting import spindle as SP
from cnctwin.cutting.drive import Driver, straight_immersion
from cnctwin.db.models import params
from tests.unit.cutting.cutfix import ALTINTAS, linear_layout, rev_steps, steady_tip

KIENZLE = dict(kc11=1.7e9, mc=0.26, ratio_r=0.5, ratio_a=0.2, h_ref_m=1e-3, h_min_m=4e-6, Kte=20e3, Kre=30e3)


def test_linearised_table_reproduces_the_rows_at_the_nodes_and_between():
    tool = AN.cylinder_mill(5e-3, 2, 2e-3, 2)
    ct, table, rows = linear_layout(tool, {"default": KIENZLE}, -1047.0, n_lo=16, n_hi=384)
    r = rows[0]
    h = np.concatenate([np.linspace(1e-7, 3.9e-6, 50), np.geomspace(4.01e-6, 5.9e-4, 400)])
    exact = CO.node_values(r, h)
    lin = np.array([CO.table_forces(table, 0, x) for x in h])
    # tangential force per chip width: piecewise-linear interpolation of a smooth Kienzle curve (h^0.74):
    # second-order error, < 0.2 % with 384 segments above h_min
    big = h > 2e-5
    assert np.allclose(lin[big, A.COL_FT], exact[big, A.COL_FT], rtol=2e-3)
    # continuity of every column across the segment boundaries of the high grid
    hm, ih = table.h_min_m[0], table.inv_dhi[0]
    nodes = hm + np.arange(1, 300) / ih
    for x in nodes[::17]:
        a = CO.table_forces(table, 0, x * (1 - 1e-12))
        b = CO.table_forces(table, 0, x * (1 + 1e-12))
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6 * (abs(a).max() + 1.0))
    # below h_min the ploughing branch: no shear-plane force, the edge force ramps from 0 to Ke at h_min
    low = CO.table_forces(table, 0, 0.5 * hm)
    assert low[A.COL_FT] == pytest.approx(0.0, abs=1e-6)
    assert low[A.COL_ET] == pytest.approx(0.5 * KIENZLE["Kte"], rel=1e-6)


def test_forming_rows_pressure_and_contact_length():
    p_ = params("cutting-contact")
    sigma, r, lobes, D = 400e6, 3.0e-3, 4, 6e-3
    h = CO.log_grid(1e-7, 1e-4, 30)
    rows = CO.forming_rows(h, sigma, 0.0, r, lobes, D, p_)
    k = sigma / math.sqrt(3.0)
    p = float(p_["forming_pressure_factor"]) * k
    assert float(p_["forming_pressure_factor"]) == pytest.approx(2.0 + math.pi)
    relief = float(p_["forming_lobe_relief_per_D"]) * D
    rho = 2 * r * r / (relief * lobes * lobes)
    l = (1 + float(p_["forming_recovery_contact_factor"])) * np.sqrt(2 * rho * h)
    assert np.allclose(rows.Kr_Pa * h, p * l, rtol=1e-12)
    assert np.allclose(rows.Kt_Pa * h, p * h, rtol=1e-12)              # frictionless: frontal term only
    sticky = CO.forming_rows(h, sigma, -1.0, r, lobes, D, p_)           # unknown friction: tau = k
    assert np.allclose(sticky.Kt_Pa * h, p * h + k * l, rtol=1e-12)


def test_spindle_curves_derating_temperature_and_stall():
    g = SP.Gear(0.0, 1000.0, np.array([0.0, 200.0, 1000.0]), np.array([100.0, 100.0, 100.0]),
                np.array([0.0, 1000.0]), np.array([20e3, 20e3]), "S6 40 %", math.sqrt(0.4))
    sp = SP.Spindle("S1", [g], 0.01, 1000.0)
    assert sp.available(100.0) == pytest.approx(100.0)                    # torque-limited
    assert sp.available(400.0) == pytest.approx(50.0)                     # power-limited: P / omega
    assert sp.s1_torque(100.0) == pytest.approx(100.0 * math.sqrt(0.4))
    f, how = SP.s1_factor("S6 40%", 600.0, 1800.0, 0.6)
    assert f == pytest.approx(math.sqrt(0.4)) and how.startswith("S6")
    f2, _ = SP.s1_factor("30 min", 600.0, 1800.0, 0.6)
    assert f2 == pytest.approx(math.sqrt(1 - math.exp(-1800.0 / 600.0)))
    # winding temperature: exact first-order response to a constant torque ratio
    th = SP.thermal_step(0.0, 120.0, 100.0, 300.0, 600.0)
    assert th == pytest.approx(1.44 * (1 - math.exp(-0.5)))
    # stall: T (w0 / w)^(1 - m) meets P / w at the speed solving T0 w0^(1-m) w^m = P
    w0, T0, m = 400.0, 52.0, 0.25
    w, stalled = SP.stall_speed(sp, w0, T0, m, 0.2)
    assert not stalled and w == pytest.approx((20e3 / (T0 * w0 ** (1 - m))) ** (1 / m), rel=1e-9)
    _, stalled = SP.stall_speed(sp, w0, 5000.0, m, 0.2)
    assert stalled


def test_bore_and_box_stocks_give_the_boundary_and_its_outward_normal():
    bore = MEM.BoreStock(4e-3, z_top_m=0.0)
    q, m, ok = bore.surface_planes(0, np.array([[3e-3, 0.0, -1e-3], [0.0, 3.5e-3, -2e-3]]),
                                   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 2e-3)
    assert ok.all()
    assert np.allclose(q, [[4e-3, 0.0, -1e-3], [0.0, 4e-3, -2e-3]])
    assert np.allclose(m, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    box = MEM.BoxStock([-1.0, -1.0, -1.0], [1.0, 1.0, 0.0])
    q, m, ok = box.surface_planes(0, np.array([[0.2, 0.1, 0.3]]), np.array([[0.0, 0.0, -1.0]]), 1.0)
    assert ok[0] and np.allclose(q[0], [0.2, 0.1, 0.0]) and np.allclose(m[0], [0.0, 0.0, 1.0])


def _slot_driver():
    R, ap, c, omega, dt = 5e-3, 1e-3, 0.05e-3, -1047.0, 8e-6
    tool = AN.cylinder_mill(R, 2, ap, 2)
    ct, table, _ = linear_layout(tool, {"default": ALTINTAS}, omega)
    drv = Driver(ct, table, 2, 2)
    vf = c * 2 * abs(omega) / (2 * math.pi)
    tip = steady_tip(vf)
    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    return drv, tip, omega, dt, R, c


def test_stale_memory_is_refilled_from_the_field_planes():
    """A slot whose memory is empty (new setup): the first pass of every bin reads the field plane, and the
    chip of the following teeth is the feed again (h = c sin(phi))."""
    drv, tip, omega, dt, R, c = _slot_driver()
    ref, _ = _slot_driver()[:2]
    # the reference memory: the steady slot
    ref.prefill_memory(tip, 0.0, omega)
    # the field: the slot side walls are far, the surface ahead is the cylinder the previous tooth left,
    # given here as planes from the reference memory (point, outward normal of the cut surface)
    nb = drv.mem.point_S_m.shape[1]
    ang = (np.arange(nb) + 0.5) * 2 * math.pi / nb
    for s in range(drv.n_rows):
        drv.refill.q_S_m[s] = ref.mem.point_S_m[s]
        drv.refill.m_S[s] = -np.stack([np.cos(ang), np.sin(ang), np.zeros(nb)], axis=-1)
        drv.refill.valid[s] = 1
    drv.mem.valid[:] = 0                                  # nothing known
    n = rev_steps(omega, dt)
    rec = drv.run_rotating(n, omega, dt, tip)
    assert drv.cacc.scalars[A.CA_STALE_READS] > 0
    assert drv.cacc.scalars[A.CA_REFILL_MISSING] == 0
    ref.run_rotating(n, omega, dt, tip)
    h, h_ref = np.array(rec.h), np.array(ref.record.h)
    # a plane is tangent to the swept surface at its bin centre: second-order error r dphi^2 / 4 at most
    # (the plane itself and the chord correction applied to the point refilled off the bin centre)
    width = 2 * math.pi / drv.mem.point_S_m.shape[1]
    assert np.max(np.abs(h - h_ref)) < 0.5 * R * width ** 2


def test_missing_planes_become_refill_requests():
    drv, tip, omega, dt, R, c = _slot_driver()
    drv.mem.valid[:] = 0
    drv.run_rotating(20, omega, dt, tip, record=False)
    rows = drv.ev.refill.rec[:int(drv.ev.refill.n[0])]
    assert rows.shape[0] > 0 and drv.cacc.scalars[A.CA_REFILL_MISSING] == rows.shape[0]
    # the integration answers them from the field (here a half space x <= R ahead of the slot) and replays
    stock = MEM.HalfSpaceStock([1.0, 0.0, 0.0], R)
    found = MEM.planes_for_requests(stock, 0, drv.refill, rows, 4 * R)
    assert found > 0
    assert np.all(drv.refill.valid[rows[:, 1].astype(int), rows[:, 2].astype(int)] == 1)


def test_candidates_cover_the_engaged_bins_that_will_be_stale():
    drv, tip, omega, dt, R, c = _slot_driver()
    drv.prefill_memory(tip, 0.0, omega)
    drv.mem.tick[:, :360] = -10**9                         # half the bins are old
    pairs = MEM.candidates(drv.tool, drv.mem, drv.iv.lo_rad, drv.iv.hi_rad, drv.iv.count, 0, 1000, -10**8,
                           0.0, 0)
    bins = {b for _, b in pairs}
    lo = drv.iv.lo_rad[0, 0]
    width = 2 * math.pi / 720
    inside_old = [b for b in range(360) if (((b + 0.5) * width - lo) % (2 * math.pi)) <= math.pi]
    assert bins == set(inside_old)
    capped = MEM.candidates(drv.tool, drv.mem, drv.iv.lo_rad, drv.iv.hi_rad, drv.iv.count, 0, 1000, -10**8, 0.0, 7)
    assert len(capped) == 7
