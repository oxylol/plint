"""Mode assembly (station mapping, unbalance coefficients, machine and rotary modes), the system arrays, the L0
stride rule and the basis-change projection with its energy bookkeeping."""

from __future__ import annotations

import cmath
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.dynamics import discretize as DZ
from cnctwin.dynamics import modal as MD
from cnctwin.dynamics.abi import KIND_MACHINE, KIND_TOOL, KIND_WORK, MI_RELAX
from cnctwin.dynamics.domain import l0_ticks_rule
from tests.unit.dynamics.dynfix import cantilever_mode, machine_doc, system, tool_mode, work_grid

TWO_PI = 2.0 * math.pi


def test_station_mapping():
    z_s = np.array([0.0, 0.01, 0.02, 0.04])
    shape = np.zeros((4, 6))
    shape[:, 0] = [0.0, 1.0, 2.0, 4.0]
    shape[:, 4] = 100.0                         # r_y = d u_x / dz
    W, dz = MD.station_map(z_s, np.array([0.005, 0.03, 0.05]))
    out = MD.map_shape(shape, W, dz)
    np.testing.assert_allclose(out[:, 0], [0.5, 3.0, 4.0 + 0.01 * 100.0])     # rigid extrapolation above
    W, dz = MD.station_map(z_s, z_s)
    np.testing.assert_allclose(MD.map_shape(shape, W, dz), shape)
    R = np.eye(12) * 1e-9
    np.testing.assert_allclose(MD.map_residual(R, np.eye(4)), R)


def test_structure_tool_modes():
    z_s = np.array([0.0, 0.02, 0.04, 0.08])
    tm = SimpleNamespace(freq_Hz=np.array([1500.0]), zeta=np.array([0.02]),
                         shape=np.array([[[0, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 10, 0], [0.5, 0, 0, 0, 15, 0],
                                          [1.0, 0, 0, 0, 12, 0]]], float),
                         station_z_m=z_s, static_residual_m_per_N=np.eye(12) * 2e-9)
    modes, res = MD.tool_modes_from_structure(tm, np.array([0.0, 0.01, 0.02]), 0.1, 7)
    assert len(modes) == 1 and modes[0].kind == KIND_TOOL and modes[0].key == ("structure", 7, 0)
    np.testing.assert_allclose(modes[0].stations[:, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(modes[0].at_unbalance[0], 1.0 + 0.02 * 12.0)
    assert res.shape == (9, 9)
    bad = SimpleNamespace(freq_Hz=np.array([1500.0]), zeta=np.array([1.5]), shape=tm.shape, station_z_m=z_s,
                          static_residual_m_per_N=np.zeros((0, 0)))
    with pytest.raises(InputError):
        MD.tool_modes_from_structure(bad, np.zeros(1), 0.0, 0)


def test_unbalance_coefficient_matches_direct_projection():
    """Re[P (W^2 - i alpha) e^(i theta)] = phi . F + psi . M of the rotating static and couple unbalance."""
    m = cantilever_mode(1200.0, 0.02, np.array([0.0, 0.01]), 0.05, 0.1)
    m.at_unbalance = np.array([0.3, -0.2, 0.0, 0.7, 0.4, 0.0])
    U = np.array([2e-6, 1e-6, 0.0])
    C = np.array([-3e-8, 5e-8, 0.0])
    P = complex(*MD.unbalance_coefficients([m], U, C)[0])
    for theta, W, alpha in ((0.3, -900.0, 0.0), (2.1, 500.0, -300.0)):
        Rz = np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0, 0, 1.0]])
        # d'Alembert force of the unbalance mass points: F = (W^2 U_rot - alpha e_z x U_rot), M = e_z x (W^2 C - ...)
        U_rot = Rz @ U
        C_rot = Rz @ C
        F = W * W * U_rot - alpha * np.cross([0, 0, 1.0], U_rot)
        M = np.cross([0, 0, 1.0], W * W * C_rot - alpha * np.cross([0, 0, 1.0], C_rot))
        direct = float(m.at_unbalance[:3] @ F + m.at_unbalance[3:] @ M)
        model = (P * (W * W - 1j * alpha) * cmath.exp(1j * theta)).real
        np.testing.assert_allclose(model, direct, rtol=1e-12, atol=1e-18)


def test_machine_modes():
    mach = machine_doc(frame_modes=[{"fn_Hz": 50.0, "zeta": 0.05, "k_N_per_m": 3e7, "direction": "x"}],
                       rotary_modes=[{"axis": "C", "fn_Hz": 140.0, "zeta": 0.05, "k_Nm_per_rad": 2e6}])
    mach["kinematics"]["workpiece_chain"].append({"link": "c", "axis": "C", "axis_point_m": [0.1, 0.0, 0.0],
                                                  "direction": [0, 0, 1]})
    mach["axes"]["C"] = {"type": "rotary"}
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # T0 rotated 90 deg about z in M
    st_M = np.array([[0.3, 0.2, 0.0], [0.3, 0.2, 0.01]])
    modes = MD.machine_modes(mach, st_M, R, {"X": 0.05, "Y": 0.0, "Z": 0.0, "C": 0.0}, {"C": 0})
    frame, rot = modes
    m = 3e7 / (TWO_PI * 50.0) ** 2
    np.testing.assert_allclose(frame.stations[0, :3], R.T @ np.array([1.0, 0, 0]) / math.sqrt(m))
    np.testing.assert_allclose(frame.base[:3], -math.sqrt(m) * np.array([1.0, 0, 0]))
    I = 2e6 / (TWO_PI * 140.0) ** 2
    # X moves the table -x by 0.05 (POE), so the C axis passes through (0.1 - 0.05, 0, 0)
    p = np.array([0.05, 0.0, 0.0])
    expected = R.T @ (-np.cross([0, 0, 1.0], st_M[0] - p)) / math.sqrt(I)
    np.testing.assert_allclose(rot.stations[0, :3], expected, rtol=1e-12)
    assert rot.base[3] == pytest.approx(-math.sqrt(I))


def test_fallback_spindle_modes():
    mach = machine_doc(spindle_modes=[{"fn_Hz": 900.0, "zeta": 0.04, "k_N_per_m": 7e7, "direction": "y"}])
    modes = MD.fallback_tool_modes(mach, "main", True, 3, np.eye(3))
    assert len(modes) == 1 and modes[0].source == "spindle_frf"
    m = 7e7 / (TWO_PI * 900.0) ** 2
    np.testing.assert_allclose(modes[0].stations[:, 1], 1.0 / math.sqrt(m))


def test_system_arrays_and_relaxation():
    modes = [tool_mode(900.0, 0.03, 2.0, (1.0, 0, 0)), tool_mode(60.0, 0.05, 300.0, (0, 1.0, 0), kind=KIND_MACHINE)]
    wm, grid = work_grid(1, (700.0,))
    sys, layout = system(modes + wm, grid=grid, point_S=[0.015, 0.01, 0.01])
    assert [m.kind for m in layout.modes] == [KIND_TOOL, KIND_MACHINE, KIND_WORK]
    np.testing.assert_array_equal(sys.dm.row, [0, 1, 0])
    np.testing.assert_array_equal(sys.model.is_work, [0, 2, 1])
    assert sys.dm.iscal[MI_RELAX] == 0
    Kc = np.diag([-2e6, -1e6, 0.0])
    sys2, _ = system(modes + wm, grid=grid, point_S=[0.015, 0.01, 0.01], K_c=Kc)
    Psi = sys2.dm.contact_shape.T
    Km = -Psi.T @ Kc @ Psi
    np.testing.assert_allclose(np.linalg.inv(sys2.dm.relax), np.eye(3) + Km @ np.diag(sys2.model.G1[:, 0]), rtol=1e-12)
    Phi, G0, G1 = DZ.foh(modes[0].omega_rad_s, 0.03, sys.model.dt_s[0])
    np.testing.assert_allclose(sys.model.Phi[0], Phi)


def test_l0_rule():
    # 10000 rpm, 4 teeth, 1 kHz mode, 0.5 deg per step, 96 steps per tooth, 2 pi f dt <= 0.3
    w = 10000.0 * TWO_PI / 60.0
    l0 = l0_ticks_rule(1000, 1e-6, w, 4, 1000.0, math.radians(0.5), 96, 0.3)
    dt = l0 * 1e-6
    assert 1000 % l0 == 0 and w * dt <= math.radians(0.5) + 1e-12 and TWO_PI * 1000.0 * dt <= 0.3
    assert TWO_PI / (4 * w) / dt >= 96
    assert l0_ticks_rule(1000, 1e-6, w, 4, 1000.0, math.radians(0.5), 96, 0.3) == 8
    f = MD.f_max_resolved_Hz([tool_mode(900.0, 0.03, 1.0, (1, 0, 0)), tool_mode(9000.0, 0.03, 1000.0, (1, 0, 0))],
                             0, 0.01)
    assert f == 900.0                         # the stiff 9 kHz mode carries far less than 1 % of the compliance


def test_basis_projection_identical_basis_is_exact():
    modes = [cantilever_mode(1500.0, 0.02, np.array([0.0, 0.01, 0.02]), 0.06, 0.1, key=("s", 1, 0)),
             cantilever_mode(1600.0, 0.02, np.array([0.0, 0.01, 0.02]), 0.06, 0.1, axis=1, key=("s", 1, 1))]
    x = np.array([[1e-6, 2e-3], [-3e-7, 5e-4]])
    new = [MD.Mode(m.kind, m.omega_rad_s, m.zeta, "fixture", m.label, stations=-m.stations.copy(),
                   at_unbalance=m.at_unbalance, key=("s", 2, k)) for k, m in enumerate(modes)]
    x_new, e_old, e_new, tracked = MD.project_basis(modes, x, new, ref_station=0, old_work_shape=None,
                                                    new_work_shape=None, length_m=0.01, mac_min=0.999, freq_tol=1e-6)
    assert tracked == 2
    np.testing.assert_allclose(x_new, -x, rtol=1e-14)        # sign-flipped shapes: coordinates flip, energy kept
    assert e_new == pytest.approx(e_old, rel=1e-14)


def test_basis_projection_energy_check():
    """A changed basis: the physical displacement and velocity at the contact points are kept (least squares) and
    the energy difference E_new - E_old is what the domain books as basis_change_J."""
    z = np.array([0.0, 0.01, 0.02, 0.03])
    old = [cantilever_mode(1500.0, 0.02, z, 0.06, 0.1, key=("s", 1, 0)),
           cantilever_mode(4200.0, 0.02, z, 0.06, 0.4, key=("s", 1, 1))]
    x = np.array([[2e-6, 3e-3], [4e-7, -1e-3]])
    new = [cantilever_mode(1450.0, 0.02, z, 0.058, 0.11, key=("s", 2, 0)),
           cantilever_mode(4000.0, 0.02, z, 0.061, 0.38, key=("s", 2, 1))]
    # make the second new shape differ in form (not only scale) so the projection is a true least-squares fit
    new[1].stations[:, 0] = (z / 0.061) ** 3 / math.sqrt(0.38)
    new[1].stations[:, 4] = 3 * z ** 2 / 0.061 ** 3 / math.sqrt(0.38)
    x_new, e_old, e_new, tracked = MD.project_basis(old, x, new, ref_station=0, old_work_shape=None,
                                                    new_work_shape=None, length_m=0.01, mac_min=0.999, freq_tol=1e-6)
    assert tracked == 0
    A_old = MD.physical_dofs(old, 0, None, 0.01)
    A_new = MD.physical_dofs(new, 0, None, 0.01)
    u_old, u_new = A_old @ x[:, 0], A_new @ x_new[:, 0]
    best = np.linalg.lstsq(A_new, u_old, rcond=None)[0]
    np.testing.assert_allclose(x_new[:, 0], best, rtol=1e-8, atol=1e-10 * float(np.max(np.abs(best))))
    assert np.linalg.norm(u_new - u_old) <= 0.05 * np.linalg.norm(u_old)
    assert e_old == pytest.approx(DZ.modal_energy(x, np.array([m.omega_rad_s for m in old])))
    assert e_new == pytest.approx(DZ.modal_energy(x_new, np.array([m.omega_rad_s for m in new])))


def test_basis_projection_work_side_minimum_energy():
    """One contact point (3 DOFs) and four workpiece modes: the new coordinates reproduce the displacement exactly
    with the least strain energy (and the velocity with the least kinetic energy)."""
    _, grid = work_grid(1, (800.0,))
    old = [MD.Mode(KIND_WORK, TWO_PI * 800.0, 0.02, "work", "w0", work_row=0, key=("work", 1, 0))]
    new = [MD.Mode(KIND_WORK, TWO_PI * f, 0.02, "work", f"w{k}", work_row=k, key=("work", 2, k))
           for k, f in enumerate((780.0, 1500.0, 2600.0, 3300.0))]
    ws_old = np.array([[0.5, 0.0, 0.0]])
    ws_new = np.array([[0.4, 0.1, 0.0], [0.3, -0.2, 0.1], [0.1, 0.1, 0.4], [0.2, 0.3, -0.1]])
    x = np.array([[3e-6, 4e-3]])
    x_new, e_old, e_new, _ = MD.project_basis(old, x, new, ref_station=0, old_work_shape=ws_old,
                                              new_work_shape=ws_new, length_m=0.01, mac_min=0.999, freq_tol=1e-6)
    np.testing.assert_allclose(ws_new.T @ x_new[:, 0], ws_old.T @ x[:, 0], rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(ws_new.T @ x_new[:, 1], ws_old.T @ x[:, 1], rtol=1e-10, atol=1e-15)
    w = np.array([m.omega_rad_s for m in new])
    # any other solution of the same displacement (along the null space) has more strain energy
    null = np.linalg.svd(ws_new.T)[2][-1]
    assert np.linalg.norm(ws_new.T @ null) < 1e-12
    for eps in (1e-7, -1e-7):
        q = x_new[:, 0] + eps * null
        assert np.sum((w * q) ** 2) >= np.sum((w * x_new[:, 0]) ** 2)
