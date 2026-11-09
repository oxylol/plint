"""(d) Out-of-contact motion with unbalance and base excitation, dynamics live (the machine file's spindle FRF modes
carrying the rigid tool and its frame and table modes): the hot loop propagates the modes exactly over every L1
tick out of contact (dynamics' modal_free with the spindle angle and speed of the tick and the axis accelerations
of the path). Checked against the closed forms of the same modal model:

* a static unbalance U at height z_U (published as holding's ToolHolding record) at 10 000 rpm: the steady
  rotating response at the reference station is Re[sum_k s_k P_k W^2 / (w_k^2 - W^2 + 2 i z_k w_k W) e^(i theta)];
* a long X rapid: on the acceleration plateau the relative deflection is quasi-static, sum_k s_k (b_k . a) / w_k^2;
* the modes' energy identity of every step (dynamics' absorb raises otherwise) and the work of both excitations."""

import math

import numpy as np
import pytest

from cnctwin.core.bus import ToolHolding
from cnctwin.core.drive import OperationDriver
from cnctwin.dynamics.kernels import relative_axis_acceleration
from tests.integration import hotfix as HF

U_KG_M = 20e-6          # 20 g mm (about G 6.3 for a 1.5 kg assembly at 10 000 rpm)
Z_U_M = 0.08            # height of the unbalance above the tool tip (holder body)


@pytest.fixture(scope="module")
def air_run(tmp_path_factory):
    drv = OperationDriver(HF.air_project(tmp_path_factory.mktemp("air"), dwell_s=3.0), use_dynamics=True,
                          dynamics_l2=False)

    def publish(d):
        d.state.bus_next.holding.tools[0] = ToolHolding(unbalance_static_kg_m=np.array([U_KG_M, 0.0, 0.0]),
                                                        unbalance_z_T_m=Z_U_M)
    rows = []

    def observe(d, rec):
        di = d.hot.last["di"]
        frame = d.state.bus_next.motion.paths[0]
        sysm = di.systems.get(0) if di is not None else None
        if sysm is None:
            return
        n = int(sysm.da.tick_n[0])
        theta = np.asarray(frame.spindle_angle_rad) + 2 * np.pi * np.asarray(frame.spindle_rev)
        rows.append({"tick": np.asarray(frame.tick)[:n], "u": np.array(sysm.da.tick_u[:n, 6:9]),
                     "theta": theta[:n], "omega": np.asarray(frame.spindle_omega_rad_s)[:n],
                     "x": np.asarray(frame.axis_actual)[:n].copy(), "names": tuple(frame.axis_names),
                     "dm": sysm.dm, "base": di.axes[0], "frame_dyn": d.state.bus_next.dynamics})
    drv.publishers.append(publish)
    drv.observers.append(observe)
    res = HF.run_until_done(drv)
    return drv, res, rows


def test_unbalance_steady_response(air_run):
    drv, res, rows = air_run
    assert res.finished
    # the dwell at full speed, after the transients (the slowest mode, 55 Hz at 5 %, decays by e in 58 ms)
    names = rows[0]["names"]
    ix = names.index("X")
    at_speed = [r for r in rows if np.all(np.abs(np.abs(r["omega"]) - abs(HF.OMEGA)) < 1e-3 * abs(HF.OMEGA))
                and np.ptp(r["x"][:, ix]) == 0.0]           # the dwell: no axis moves (no base excitation)
    assert len(at_speed) >= 150
    use = at_speed[-100:]                       # the last second of the dwell (transients decayed by e^-10)
    th = np.concatenate([r["theta"] for r in use])
    u = np.concatenate([r["u"] for r in use])
    W = float(np.mean(np.concatenate([r["omega"] for r in use])))
    dm = use[-1]["dm"]
    P = dm.unb[:, 0] + 1j * dm.unb[:, 1]
    D = dm.omega_rad_s ** 2 - W ** 2 + 2j * dm.zeta * dm.omega_rad_s * W
    Z = (dm.contact_shape * (P * W ** 2 / D)[:, None]).sum(axis=0)          # complex amplitude, T0 x y z
    M = np.stack([np.cos(th), np.sin(th), np.ones_like(th)], axis=1)
    for c in range(2):
        (A, B, C), *_ = np.linalg.lstsq(M, u[:, c], rcond=None)
        print("unbalance response", "xy"[c], "fit", A, B, "closed form", Z[c].real, -Z[c].imag)
        amp = abs(Z[c])
        assert amp > 1e-9
        assert A == pytest.approx(Z[c].real, abs=3e-3 * amp)
        assert B == pytest.approx(-Z[c].imag, abs=3e-3 * amp)
    w_unb = sum(float(r["frame_dyn"].unbalance_work_J) for r in rows)
    assert np.isfinite(w_unb)


def test_base_excitation_quasi_static_on_the_acceleration_plateau(air_run):
    drv, res, rows = air_run
    names = rows[0]["names"]
    ix = names.index("X")
    x = np.concatenate([r["x"][:, ix] for r in rows])
    u = np.concatenate([r["u"] for r in rows])
    dt = drv.ctx.ipo_ticks * drv.ctx.tick_s
    acc = np.zeros_like(x)
    acc[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / dt ** 2
    a_max = float(np.max(np.abs(acc)))
    assert a_max > 1.0                                                   # the rapid accelerates (m/s2)
    plateau = np.nonzero(np.abs(np.abs(acc) - a_max) < 0.005 * a_max)[0]
    plateau = plateau[(plateau > 2) & (plateau < len(acc) - 2)]
    assert len(plateau) >= 10
    dm = rows[-1]["dm"]
    base = rows[-1]["base"]
    for k in plateau[len(plateau) // 2:len(plateau) // 2 + 1]:
        a_axis = np.zeros(len(names))
        a_axis[ix] = acc[k]
        b = np.zeros(dm.base_proj.shape[1])
        relative_axis_acceleration(base, np.ascontiguousarray(a_axis), b)
        q_qs = (dm.base_proj @ b) / dm.omega_rad_s ** 2
        u_qs = (dm.contact_shape * q_qs[:, None]).sum(axis=0)
        print("quasi-static base response", u[k, :2], "closed form", u_qs[:2], "a", acc[k])
        assert np.linalg.norm(u_qs[:2]) > 1e-7
        assert np.allclose(u[k, :2], u_qs[:2], atol=0.15 * np.linalg.norm(u_qs[:2]))
    w_base = sum(float(r["frame_dyn"].base_work_J) for r in rows)
    assert w_base != 0.0
    # the energy of every step balanced: dynamics' absorb raises otherwise (dE = W_unbalance + W_base - D)
