"""Balance (ISO 21940-11): permissible unbalance numbers, the unbalance and couple of an eccentric, tilted tool
against a direct point-mass integration, the assembly with deterministic residual phases, and the unbalance
and overspeed findings at the programmed speed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.holding import holdfix as fx
from cnctwin.core.findings import Category, Severity
from cnctwin.holding import balance as B
from cnctwin.holding.domain import HoldingDomain


def test_iso_21940_permissible_unbalance_numbers():
    # G 2.5, 1.6 kg holder balanced at 25 000 rpm: U_per = 9549 x 2.5 x 1.6 / 25000 = 1.528 g mm
    assert B.permissible_unbalance_g_mm(2.5, 1.6, 25000.0) == pytest.approx(1.5279, rel=1e-4)
    omega = 25000.0 * math.pi / 30.0
    U = B.permissible_unbalance_kg_m(0.0025, 1.6, omega)
    assert U == pytest.approx(1.5279e-6, rel=1e-4)                 # kg m (1 g mm = 1e-6 kg m)
    assert B.specific_unbalance_m(0.0025, omega) == pytest.approx(0.955e-6, rel=1e-3)   # e_per = 0.955 um
    # G 6.3, 10 kg rotor at 3000 rpm: 9549 x 6.3 x 10 / 3000 = 200.5 g mm
    assert B.permissible_unbalance_g_mm(6.3, 10.0, 3000.0) == pytest.approx(200.53, rel=1e-4)
    # the effective grade is the inverse relation
    assert B.effective_grade_m_s(U, 1.6, omega) == pytest.approx(0.0025, rel=1e-12)
    assert B.ISO_UNBALANCE_CONSTANT == pytest.approx(9549.3, rel=1e-5)


def _cylinder_points(radius, z0, z1, mass, n_r=8, n_t=24, n_z=40):
    """Point masses of a solid cylinder (midpoint rule in r, theta, z)."""
    r = (np.arange(n_r) + 0.5) / n_r * radius
    t = (np.arange(n_t) + 0.5) / n_t * 2 * math.pi
    z = z0 + (np.arange(n_z) + 0.5) / n_z * (z1 - z0)
    R, T, Z = np.meshgrid(r, t, z, indexing="ij")
    w = R.ravel()                                            # area element r dr dtheta
    pts = np.stack([R.ravel() * np.cos(T.ravel()), R.ravel() * np.sin(T.ravel()), Z.ravel()], axis=1)
    return pts, mass * w / w.sum()


def test_eccentric_tilted_tool_against_point_masses():
    radius, z0, z1, mass = 0.005, 0.0, 0.06, 0.09
    pts, m = _cylinder_points(radius, z0, z1, mass)
    com = (m[:, None] * pts).sum(axis=0) / m.sum()
    I_axial = float((m * (pts[:, 0] ** 2 + pts[:, 1] ** 2)).sum())
    I_trans = float((m * (pts[:, 1] ** 2 + (pts[:, 2] - com[2]) ** 2)).sum())
    offset = np.array([3e-6, -2e-6, 0.0])
    tilt = np.array([2e-5, -4e-5])                      # about x_T and y_T
    omega = np.array([tilt[0], tilt[1], 0.0])
    moved = pts + np.cross(omega, pts) + offset         # the displacement field the EdgeModel applies
    U_points = (m[:, None] * moved[:, :2]).sum(axis=0)
    C_points = (m[:, None] * moved[:, 2:3] * moved[:, :2]).sum(axis=0)
    U, z_c, C = B.tool_eccentricity(mass, float(com[2]), I_axial, I_trans, offset, tilt)
    np.testing.assert_allclose(U[:2], U_points, rtol=1e-6)
    # couple about the centre-of-mass plane: first moments minus z_c U
    np.testing.assert_allclose(C[:2], C_points - z_c * U_points, rtol=1e-4, atol=1e-16)
    assert z_c == pytest.approx(0.03)


def test_assembly_unbalance_is_deterministic_and_worst_case_adds():
    td = fx.tool_data(runout_T_m=np.array([2.5e-6, 0.0, 0.0]))
    a = B.assembly_unbalance(td, seed=11)
    b = B.assembly_unbalance(td, seed=11)
    c = B.assembly_unbalance(td, seed=12)
    np.testing.assert_array_equal(a.static_T_kg_m, b.static_T_kg_m)
    assert not np.allclose(a.static_T_kg_m, c.static_T_kg_m)
    # holder: G 2.5 at 25 000 rpm on 1.4 kg; tool: 0.05 kg x 2.5 um
    u_h = 0.0025 * 1.4 / (25000 * math.pi / 30)
    assert a.holder_kg_m == pytest.approx(u_h, rel=1e-12)
    assert a.tool_kg_m == pytest.approx(0.05 * 2.5e-6, rel=1e-12)
    assert a.worst_kg_m == pytest.approx(u_h + 0.05 * 2.5e-6, rel=1e-12)
    assert np.linalg.norm(a.static_T_kg_m) <= a.worst_kg_m * (1 + 1e-12)
    assert a.z_T_m == pytest.approx((0.05 * 0.03 + 1.4 * 0.12) / 1.45)
    assert a.required_grade_m_s == 0.0025 and "published" in a.required_source
    assert a.max_rad_s == pytest.approx(25000 * math.pi / 30)
    # no published grade anywhere: the holder is taken as G 6.3 at its limit, G 2.5 is required
    td2 = fx.tool_data(holder_grade=None, tool_grade=None)
    d = B.assembly_unbalance(td2, seed=11)
    assert d.required_grade_m_s == 0.0025 and d.notes
    assert d.holder_kg_m == pytest.approx(0.0063 * 1.4 / (25000 * math.pi / 30), rel=1e-12)


def _run_speed(rpm, td=None):
    proj = fx.project([fx.entry(fx.VISE, translate_mm=(50, 0, 0), input_torque_Nm=80, parallels_mm=10)])
    ctx = fx.context(proj)
    state = fx.sim_state()
    dom = HoldingDomain(tools={0: td or fx.tool_data(runout_T_m=np.array([10e-6, 0.0, 0.0]))})
    dom.init(ctx, state)
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), omega=-rpm * math.pi / 30))
    return rec.tools[0], state


def test_unbalance_force_and_finding_above_the_grade_at_the_programmed_speed():
    t, state = _run_speed(12000.0)
    omega = 12000 * math.pi / 30
    ub = B.assembly_unbalance(fx.tool_data(runout_T_m=np.array([10e-6, 0.0, 0.0])), 11)
    assert t.unbalance_force_N == pytest.approx(np.linalg.norm(ub.static_T_kg_m) * omega ** 2, rel=1e-12)
    assert t.balance_quality_m_s == pytest.approx(ub.worst_kg_m * omega / ub.mass_kg, rel=1e-12)
    assert t.balance_quality_m_s < 0.0025
    assert not [f for f in state.findings if f.category == Category.UNBALANCE]
    t, state = _run_speed(24000.0)
    assert t.balance_quality_m_s > 0.0025
    found = [f for f in state.findings if f.category == Category.UNBALANCE]
    assert found and found[0].severity == Severity.WARNING
    assert not [f for f in state.findings if f.category == Category.OVERSPEED]


def test_overspeed_of_holder_and_tool():
    t, state = _run_speed(26000.0)
    over = [f for f in state.findings if f.category == Category.OVERSPEED]
    assert over and over[0].severity == Severity.ERROR and "holder" in over[0].message
    assert t.max_rad_s == pytest.approx(25000 * math.pi / 30)
    td = fx.tool_data(max_rad_s_tool=20000 * math.pi / 30)
    t, state = _run_speed(21000.0, td)
    over = [f for f in state.findings if f.category == Category.OVERSPEED]
    assert over and "tool" in over[0].message
