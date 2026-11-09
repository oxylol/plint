"""The compiled kernels against the plain-Python reference and their py_func, one signature each."""

import copy
import math

import numpy as np
import pytest

from cnctwin.core.abi import ModalModel, ModalState, N_ELEM_ACC
from cnctwin.cutting import abi as A
from cnctwin.cutting import kernels as K
from cnctwin.cutting import reference as REF
from cnctwin.cutting.drive import Driver, straight_immersion
from tests.unit.cutting.cutfix import built, linear_layout

COEFFS = {"default": dict(Ktc=900e6, Krc=350e6, Kac=150e6, Kte=30e3, Kre=40e3, Kae=2e3, h_min_m=2.4e-6, K_sp=4e13,
                          sigma0=1.2e9, tau0=0.5e9, mu=0.4, sigma_rake=8e8, chip_ratio=0.35, ps=0.7, pr=0.22)}


def _clone(nt):
    return type(nt)(*(copy.deepcopy(getattr(nt, f)) for f in nt._fields))


@pytest.fixture(scope="module")
def setup():
    g = built("square_endmill", runout_um=6, runout_phase_deg=30)
    nt, ne = g.edge.elem_r_m.shape
    vb = np.full((nt, ne), 40e-6)
    ct, table, rows = linear_layout(g, COEFFS, VB=vb)
    return g, ct, table


def test_one_signature_each():
    for kern in (K.cut_step, K.turn_step, K.deflection_from_modal, K.copy_element_accumulators, K.gate,
                 K.memory_commit, K.flank_interference_area, K.polar_inside, K.polar_depth, K.recut_scan,
                 K.finish_step, K.close_teeth, K.emit_pass, K.element_exit, K.section_peaks, K.refill_bin):
        assert len(kern.signatures) == 1, kern


def test_cut_step_matches_reference(setup):
    g, ct, table = setup
    nt, ne = g.edge.elem_r_m.shape
    drv = Driver(ct, table, nt, ne)
    omega = -1047.0
    dt = 8e-6
    vf = 0.06e-3 * 4 * abs(omega) / (2 * math.pi)
    tip = lambda t: np.array([vf * t, 0.0, 0.0])
    lo, hi = straight_immersion(math.radians(25.0), math.radians(160.0))
    drv.set_interval(lo, hi)
    drv.prefill_memory(tip, 0.0, omega)
    # warm-up so that memory, slopes and pending writes are populated
    drv.run_rotating(400, omega, dt, tip, record=False)
    rng = np.random.default_rng(3)
    for k in range(60):
        t_end = (400 + k + 1) * dt
        t_mid = t_end - 0.5 * dt
        theta = (omega * t_mid) % (2 * math.pi)
        drv.step[A.ST_THETA_MID_RAD] = theta
        drv.step[A.ST_COS_MID] = math.cos(theta)
        drv.step[A.ST_SIN_MID] = math.sin(theta)
        drv.istep[A.SI_TICK_END] = int(round(t_end / drv.tick_s))
        drv.pose.tip_S_m[:] = tip(t_mid)
        drv.pose.v_tip_S_m_s[:] = [vf, 0.0, 1e-3]
        drv.pose.w_T0_S_rad_s[:] = [0.02, -0.01, 0.0]
        R = REF.rot_z(0.3) @ np.array([[1.0, 0.0, 0.0], [0.0, math.cos(0.01), -math.sin(0.01)],
                                       [0.0, math.sin(0.01), math.cos(0.01)]])
        drv.pose.R_S_T0[:] = R
        drv.defl.u_st_T0_m[:] = rng.normal(0.0, 1e-6, drv.defl.u_st_T0_m.shape) * np.array([1, 1, 1, 1e-3, 1e-3, 1e-3])
        drv.defl.v_st_T0_m_s[:] = rng.normal(0.0, 1e-3, drv.defl.v_st_T0_m_s.shape) * np.array([1, 1, 1, 0.1, 0.1, 0.1])
        drv.defl.u_work_S_m[:] = [2e-7, -1e-7, 0.0]
        drv.defl.v_work_S_m_s[:] = [1e-4, 0.0, -5e-5]
        mem0 = _clone(drv.mem)
        st0 = _clone(drv.state)
        acc0 = drv.acc.scalars.copy()
        ea0 = drv.state.elem_acc.copy()
        ref = REF.cut_step_ref(ct, table, mem0, drv.iv, drv.step, drv.istep, drv.pose, drv.defl, st0)
        K.cut_step(ct, table, drv.mem, drv.iv, drv.refill, drv.step, drv.istep, drv.pose, drv.defl, drv.occ, drv.pack,
                   drv.axes, drv.state, drv.acc, drv.cacc, drv.out, drv.ev)
        fscale = max(np.abs(ref.wrench_T0[:3]).max(), 1.0)
        assert np.allclose(drv.out.wrench_T0[:3], ref.wrench_T0[:3], rtol=0, atol=1e-9 * fscale)
        assert np.allclose(drv.out.wrench_T0[3:], ref.wrench_T0[3:], rtol=0, atol=1e-9 * fscale * 0.01)
        assert np.allclose(drv.out.station_T0, ref.station_T0, rtol=0, atol=1e-9 * fscale)
        d = drv.acc.scalars - acc0
        escale = max(abs(ref.heat_shear_J), 1e-12)
        assert d[A.ACC_HEAT_SHEAR_J] == pytest.approx(ref.heat_shear_J, rel=1e-9, abs=1e-9 * escale)
        assert d[A.ACC_HEAT_RAKE_J] == pytest.approx(ref.heat_rake_J, rel=1e-9, abs=1e-9 * escale)
        assert d[A.ACC_HEAT_FLANK_J] == pytest.approx(ref.heat_flank_J, rel=1e-8, abs=1e-9 * escale)
        assert d[A.ACC_PLOUGHING_J] == pytest.approx(ref.ploughing_J, rel=1e-8, abs=1e-9 * escale)
        assert d[A.ACC_PROCESS_DAMPING_J] == pytest.approx(ref.process_damping_J, rel=1e-7, abs=1e-9 * escale)
        assert d[A.ACC_CHIP_VOLUME_M3] == pytest.approx(ref.chip_volume_m3, rel=1e-9)
        assert d[A.ACC_SHAFT_WORK_J] == pytest.approx(ref.shaft_J, rel=1e-9)
        assert d[A.ACC_FEED_WORK_J] == pytest.approx(ref.feed_J, rel=1e-7, abs=1e-9 * abs(ref.shaft_J))
        inc = [c for c in range(N_ELEM_ACC) if c != A.ELEM_H_MAX]
        assert np.allclose((drv.state.elem_acc - ea0)[:, inc], ref.elem_acc[:, inc], rtol=1e-9, atol=1e-18)
        assert np.allclose(drv.state.elem_acc[:, A.ELEM_H_MAX],
                           np.maximum(ea0[:, A.ELEM_H_MAX], ref.elem_acc[:, A.ELEM_H_MAX]), rtol=1e-9, atol=1e-15)
        for e, h in ref.h.items():
            # clipped arcs: the kernel's rotation series (dl^4 / 24, dl <= 0.25 deg) against exact cos / sin
            assert drv.state.h_last_m[e] == pytest.approx(h, abs=1e-12)
        assert np.allclose(drv.mem.point_S_m, ref.memory, rtol=0, atol=1e-12)
        assert np.allclose(drv.state.pend, ref.pend, rtol=0, atol=1e-12)
    # energy closure of the kernel over the whole run (shaft + feed = heat + deflection work)
    a = drv.acc.scalars
    heat = a[A.ACC_HEAT_SHEAR_J] + a[A.ACC_HEAT_RAKE_J] + a[A.ACC_HEAT_FLANK_J] + a[A.ACC_PLOUGHING_J] + \
        a[A.ACC_PROCESS_DAMPING_J]
    total_in = a[A.ACC_SHAFT_WORK_J] + a[A.ACC_FEED_WORK_J]
    assert total_in == pytest.approx(heat + drv.cacc.scalars[A.CA_DEFLECTION_J], rel=1e-12)


def test_gate_matches_reference():
    iv = A.empty_intervals(2, 4)
    iv.lo_rad[0, :2] = [0.3, 5.9]
    iv.hi_rad[0, :2] = [1.2, 5.9 + 0.7]
    iv.count[0] = 2
    rng = np.random.default_rng(1)
    for _ in range(400):
        a = rng.uniform(0.0, 2 * math.pi)
        dal = rng.choice([1, -1]) * rng.uniform(0.0, 0.02)
        got = K.gate(iv, 0, a, 0.5 * abs(dal), 1.0 / dal if dal else 0.0)
        exp = REF.engaged_part([(0.3, 1.2), (5.9, 6.6)], a, dal)
        assert np.allclose(got, exp, atol=1e-12)
        assert np.allclose(K.gate.py_func(iv, 0, a, 0.5 * abs(dal), 1.0 / dal if dal else 0.0), got, atol=0)


def test_memory_commit_elastic_recovery_rule():
    mem = A.empty_memory(1, 8)
    mem.valid[0, :] = 1
    h_min = 2e-6
    for h, w in ((1.9e-6, 0.0), (2.0e-6, 0.0), (3.0e-6, 0.5), (4.0e-6, 1.0), (9e-6, 1.0)):
        mem.point_S_m[0, 1] = [0.0, 0.0, 0.0]
        K.memory_commit(mem, 0, 1, 1.0, 2.0, 3.0, h, h_min, 1.0, 7)
        assert np.allclose(mem.point_S_m[0, 1], [w * 1.0, w * 2.0, w * 3.0], atol=1e-15)
        pts = np.zeros((1, 8, 3))
        tk = np.zeros((1, 8), np.int64)
        vl = np.ones((1, 8), np.int8)
        REF.commit_ref(pts, tk, vl, 0, 1, [1.0, 2.0, 3.0], h, h_min, 1.0, 7)
        assert np.allclose(pts[0, 1], mem.point_S_m[0, 1], atol=1e-15)
    mem.valid[0, 2] = 0
    K.memory_commit(mem, 0, 2, 1.0, 1.0, 1.0, 2.5e-6, h_min, 1.0, 9)       # an empty bin is set fully
    assert np.allclose(mem.point_S_m[0, 2], 1.0) and mem.valid[0, 2] == 1 and mem.tick[0, 2] == 9


def test_flank_interference_area_against_quadrature():
    rng = np.random.default_rng(2)
    for _ in range(200):
        slope = rng.uniform(-0.05, 0.4)
        L = rng.uniform(0.0, 3e-4)
        tg = math.tan(rng.uniform(0.02, 0.3))
        lmax = 5e-4
        x = np.linspace(0.0, lmax, 200001)
        delta = slope * x - np.maximum(x - min(L, lmax), 0.0) * tg
        exp = float(np.trapezoid(np.maximum(delta, 0.0), x)) if slope > 0 else 0.0
        got = K.flank_interference_area(slope, L, tg, lmax)
        assert got == pytest.approx(exp, rel=1e-6, abs=1e-14)


def test_polar_queries_against_brute_force():
    nth, nz = 72, 40
    polar = A.empty_polar(nth, nz)
    polar.geom[:] = [2 * math.pi / nth, 0.0, 0.5e-3, 0.0]
    zc = (np.arange(nz) + 0.5) * 0.5e-3
    polar.r_out_m[:] = (0.02 + 0.5e-3 * np.sin(zc / 2e-3))[None, :]
    polar.r_in_m[:] = 0.005
    for th, r, z in ((0.3, 0.0195, 0.006), (2.0, 0.0201, 0.0101), (4.5, 0.006, 0.012)):
        g = K.polar_inside(polar, th, r, z)
        ro = np.interp(z, zc, polar.r_out_m[0])
        assert g == pytest.approx(min(ro - r, r - 0.005), abs=1e-12)
        assert K.polar_inside.py_func(polar, th, r, z) == pytest.approx(g, abs=0)
    # depth along a ray from inside: brute force
    th, r, z, ur, uz = 1.0, 0.0198, 0.008, 0.6, 0.8
    h, capped = K.polar_depth(polar, th, r, z, ur, uz, 5e-3)
    s = np.linspace(0, 5e-3, 200001)
    rr = r + s * ur
    zz = z + s * uz
    ro = np.interp(zz, zc, polar.r_out_m[0])
    inside = (ro - rr) > 0
    s_exit = s[np.argmin(inside)]
    assert capped == 0.0 and h == pytest.approx(s_exit, abs=2e-6)


def test_deflection_from_modal_against_numpy():
    nst = 5
    ms = np.random.default_rng(4).normal(size=(2, nst, 6))
    model = ModalModel(np.zeros((3, 2, 2)), np.zeros((3, 2)), np.zeros((3, 2)), np.ascontiguousarray(ms),
                       np.array([[0.1, 0.2, 0.3]]), np.array([0, 0, 1], np.int8), np.array([-1, -1, 0], np.int32),
                       np.ascontiguousarray(np.eye(3 * nst) * 1e-9), np.array([1e-5]))
    x = np.array([[1e-6, 2e-3], [-3e-6, 1e-3], [5e-7, -4e-4]])
    state = ModalState(x, np.zeros(3))
    out = A.empty_deflection(nst)
    force = np.ascontiguousarray(np.random.default_rng(5).normal(size=(nst, 3)))
    K.deflection_from_modal(model, state, force, out)
    exp_u = ms[0] * x[0, 0] + ms[1] * x[1, 0]
    exp_u[:, :3] += 1e-9 * force
    assert np.allclose(out.u_st_T0_m, exp_u, atol=1e-20)
    assert np.allclose(out.v_st_T0_m_s, ms[0] * x[0, 1] + ms[1] * x[1, 1], atol=1e-20)
    assert np.allclose(out.u_work_S_m, np.array([0.1, 0.2, 0.3]) * x[2, 0], atol=1e-20)
    out2 = A.empty_deflection(nst)
    K.deflection_from_modal.py_func(model, state, force, out2)
    assert np.allclose(out2.u_st_T0_m, out.u_st_T0_m, atol=0)


def test_copy_element_accumulators():
    st = A.empty_state(6, 2, 3)
    st.elem_acc[:, :N_ELEM_ACC] = np.arange(6 * N_ELEM_ACC).reshape(6, N_ELEM_ACC)
    acc = A.core_accum(1, 1, 2, 3)
    K.copy_element_accumulators(st, acc, 3)
    for e in range(6):
        assert np.array_equal(acc.elem[:, e // 3, e % 3], st.elem_acc[e, :N_ELEM_ACC])


def test_layouts_are_kernel_layouts(setup):
    g, ct, table = setup
    for layout in (ct, table, A.empty_state(4, 2, 2), A.empty_events(4), A.identity_pose(), A.empty_occupancy()):
        assert A.is_kernel_layout(layout)


def test_polar_depth_ends_on_the_gap_of_the_next_row():
    """A depth ray entering the air gap of the next row ends on that row's boundary (the nearest-row gap of
    polar_inside), not on the linear root of a radial distance that jumps there (which sat near the outside
    sample: +dz / 4 on average on a turning chip)."""
    nth, nz, dz = 8, 60, 36e-6
    polar = A.empty_polar(nth, nz)
    polar.geom[:] = [2 * math.pi / nth, 0.0, dz, 0.0]
    polar.r_out_m[:] = 0.0225
    polar.r_in_m[:] = 0.0
    ig = 30
    polar.gap_lo_m[:, ig:] = 0.020
    polar.gap_hi_m[:, ig:] = 0.022
    for frac in (0.1, 0.3, 0.55, 0.8):
        z = (ig - 5 - frac) * dz
        h, capped = K.polar_depth(polar, 0.4, 0.0215, z, 0.0, 1.0, 5e-3)
        assert capped == 0.0 and h == pytest.approx(ig * dz - z, abs=dz / 2 ** 12)
        # an oblique ray: the boundary where the nearest row turns to the gap
        ur, uz = 0.1, math.sqrt(1 - 0.01)
        h2, _ = K.polar_depth(polar, 0.4, 0.0211, z, ur, uz, 5e-3)
        assert h2 == pytest.approx((ig * dz - z) / uz, abs=dz / 2 ** 12)
