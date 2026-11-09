"""(b) A G83 peck-drilling cycle (D6.8 carbide drill, Al 6061-T6, R2 Q3 pecks to 10 mm, 7000 rpm, 0.15 mm/rev) through
the whole chain: the steady thrust and torque of a peck against cutting's stand-alone kernel on the same tool and
table (full-circle engagement, prefilled memory on the axial feed path), the hole volume against the drill's
envelope on the lattice and in the continuum (exact ledger), the retracts and re-entries of the cycle."""

import math

import numpy as np
import pytest

from cnctwin.core.drive import OperationDriver
from cnctwin.cutting.drive import Driver
from tests.integration import hotfix as HF

RPM = 7000.0
F_REV = 0.15e-3
DEPTH = 10e-3


@pytest.fixture(scope="module")
def drill_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("drill")
    drv = OperationDriver(HF.drill_project(tmp, DEPTH * 1e3, 3.0, F_REV * RPM * 1e3, RPM), use_dynamics=False,
                          cutting_model_factory=HF.kienzle_model)
    res = HF.Recorder(drv).run()
    return drv, res


def test_cycle_runs_with_four_pecks(drill_run):
    drv, res = drill_run
    assert res.finished
    pecks = 0
    cutting = False
    for st in drv.steps:
        now = st.in_cut_s > 0.0 and st.wrench_T0 and float(np.mean(st.wrench_T0[0][:, 2])) > 100.0
        if now and not cutting:
            pecks += 1
        cutting = bool(now)
    assert pecks == 4              # from R2 by Q3: to Z-1, -4, -7 and -10
    assert res.ledger["time"]["ok"]
    # removal flags any material taken while the cycle rapids back into the hole (wall rubbing of the runout
    # tooth, docs/models/hotloop.md "Open problems"): never deeper than a quantum
    for f in res.findings:
        if f.category.value == "rapid_into_material":
            assert f.quantities.get("depth_m", 0.0) <= 1e-6


def test_thrust_and_torque_against_the_stand_alone_kernel(drill_run):
    drv, res = drill_run
    vz = F_REV * RPM / 60.0
    top = float(res.field.lv_box_hi_m[0, 2])
    # full engagement (the point cone of 1.1 mm inside the stock) at the steady feed of a peck
    steady = [st for st in drv.steps if st.in_cut_s >= 0.0099 and st.z_range[1] < top - 2e-3 and
              abs((st.z_first_last[0] - st.z_first_last[1]) - vz * 0.009) < 0.02 * vz * 0.009]
    assert len(steady) >= 15
    W = np.concatenate([st.wrench_T0[0] for st in steady])
    thrust, torque = float(W[:, 2].mean()), float(W[:, 5].mean())
    tc = drv.state.cutting.tools[0]
    nt, ne = np.asarray(drv.state.tooling.effective[0].elem_r_m).shape
    ref = Driver(tc.tool, tc.table, nt, ne, memory_bins=int(drv.state.cutting.settings.memory_bins), rho_kg_m3=drv.rho)
    omega = float(np.mean([st.omega for st in steady]))
    tip = lambda t: np.array([0.0, 0.0, -F_REV * abs(omega) / (2 * math.pi) * t])   # noqa: E731
    ref.set_full_circle()
    ref.prefill_memory(tip, 0.0, omega)
    dt = max(st.l0_ticks.get(0, 0) for st in steady) * 1e-6
    n = int(round(2 * math.pi / abs(omega) / dt))
    ref.run_rotating(2 * n, omega, dt, tip, record=False)
    Wr = np.array(ref.run_rotating(4 * n, omega, dt, tip, t0_s=2 * n * dt).wrench)
    print("drill thrust", thrust, "reference", Wr[:, 2].mean(), "torque", torque, "reference", Wr[:, 5].mean())
    assert thrust == pytest.approx(float(Wr[:, 2].mean()), rel=5e-3)
    assert torque == pytest.approx(float(Wr[:, 5].mean()), rel=5e-3)


def test_hole_volume(drill_run):
    drv, res = drill_run
    assert res.removed_volume_m3 == pytest.approx(res.field_volume_change_m3, rel=1e-12)
    field = res.field
    top = float(field.lv_box_hi_m[0, 2])
    z_tip = min(st.z_range[0] for st in drv.steps)
    assert top - z_tip == pytest.approx(DEPTH, abs=20e-6)
    shape = drv.state.removal.tools[0]
    dr = float(np.max(shape.slabs.dr))
    xy = [st.tip_yz for st in drv.steps if st.in_cut_s > 0][0]
    centre = (20e-3, xy[0])
    lattice = HF.lattice_hole_volume(field, shape.cut, centre, dr, z_tip, top)
    R = float(shape.cut.c0[-1] + shape.cut.c1[-1] * (top - z_tip)) + dr
    print("hole volume", res.removed_volume_m3, "lattice", lattice)
    assert res.removed_volume_m3 == pytest.approx(lattice, rel=3e-3)
    a = float(field.lv_cell_m[0])
    assert abs(res.removed_volume_m3 - lattice) <= 0.5 * a * 2 * math.pi * R * (top - z_tip)
    assert res.chip_mass_kg == pytest.approx(drv.rho * res.removed_volume_m3, rel=1e-12)
