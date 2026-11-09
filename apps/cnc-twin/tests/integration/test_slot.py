"""(a) Straight full slot in Al 6061-T6 with a D10 4-flute end mill through the whole chain (controller -> motion ->
hot loop -> absorb): mean and peak forces against cutting's stand-alone kernel driver for the same engagement,
the removed volume against the lattice and continuous slot volumes (exact ledger), chip mass = removed volume x
rho, cutting's analytic chip volume against the field's."""

import math

import numpy as np
import pytest

from cnctwin.core.drive import OperationDriver
from tests.integration import hotfix as HF


@pytest.fixture(scope="module")
def slot_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("slot")
    # ap from the tool's stations: the stock top midway between two peripheral stations (field resolution)
    probe = OperationDriver(HF.slot_project(tmp / "probe", 3.0), use_dynamics=False)
    ap_mm = HF.station_ap_mm(probe.state.cutting.tools[0].tool if probe.state.cutting.tools[0].tool is not None
                             else _tool(probe), 14)
    drv = OperationDriver(HF.slot_project(tmp / "run", ap_mm), use_dynamics=False)
    res = HF.Recorder(drv).run()
    return drv, res, ap_mm * 1e-3


def _tool(drv):
    drv.cutting.update_tool(drv.ctx, drv.state, 0, drv.state.bus_prev, force=True)
    return drv.state.cutting.tools[0].tool


def test_program_ends_and_ledgers_close(slot_run):
    drv, res, ap = slot_run
    assert res.finished
    assert not [f for f in res.findings if f.severity.value == "fatal"]
    assert res.ledger["time"]["ok"] and res.ledger["time"]["residual_ticks"] == 0
    en = res.ledger["energy_run"]
    assert abs(en["in_J"] - en["out_J"]) <= 1e-9 * en["abs_J"]        # cutting's own identity, round-off
    assert en["deflection_work_J"] == 0.0                              # rigid tool and work


def test_removed_volume_exact_and_against_the_slot(slot_run):
    drv, res, ap = slot_run
    # exact ledger: the booked removal equals the field's own volume change (integer quanta)
    assert res.removed_volume_m3 == pytest.approx(res.field_volume_change_m3, rel=1e-12)
    field = res.field
    top = float(field.lv_box_hi_m[0, 2])
    # the tool tip as moved (machine geometric errors put the floor a few um off the programmed depth)
    steady = [st for st in drv.steps if st.in_cut_s >= 0.0099 and st.x_range[0] > 6e-3 and st.x_range[1] < 28e-3]
    y_tip = float(np.mean([st.tip_yz[0] for st in steady]))
    depth = top - float(np.mean([st.tip_yz[1] for st in steady]))
    assert depth == pytest.approx(ap, abs=20e-6)
    R_eff = 5e-3 + float(np.max(drv.state.removal.tools[0].slabs.dr[:, 0]))     # widest tooth (runout)
    lattice = HF.lattice_slot_volume(field, -8e-3, 30e-3, y_tip, R_eff, depth, top)
    continuous = (30e-3 * 2 * R_eff + 0.5 * math.pi * R_eff ** 2) * depth
    print("removed", res.removed_volume_m3, "lattice", lattice, "continuous", continuous)
    a = float(field.lv_cell_m[0])
    assert res.removed_volume_m3 == pytest.approx(lattice, rel=2e-3)
    # the column model against the continuous slot: boundary columns of a half cell
    perimeter = 2 * 30e-3 + math.pi * R_eff
    assert abs(res.removed_volume_m3 - continuous) <= 0.5 * a * perimeter * ap
    # chip mass: rho (20 degC) x removed field volume; the analytic chip volume within the 2 % consistency band
    assert res.chip_mass_kg == pytest.approx(2700.0 * res.removed_volume_m3, rel=1e-12)
    assert res.analytic_chip_volume_m3 == pytest.approx(res.removed_volume_m3, rel=0.02)


def test_forces_against_the_stand_alone_kernel(slot_run):
    drv, res, ap = slot_run
    steady = [st for st in drv.steps if st.in_cut_s >= 0.0099 and st.x_range[0] > 6e-3 and st.x_range[1] < 28e-3]
    assert len(steady) >= 40
    W = np.concatenate([st.wrench_T0[0] for st in steady])
    P = np.concatenate([st.wrench_T0_peak[0] for st in steady])
    omega = float(np.mean([st.omega for st in steady]))
    vf = HF.SLOT_FEED_MM_MIN * 1e-3 / 60.0
    dt = drv.steps[-1].l0_ticks.get(0, 8) * 1e-6 if drv.steps[-1].l0_ticks else 8e-6
    l0 = max(st.l0_ticks.get(0, 0) for st in steady)
    mean_ref, peak_ref, rate_ref = HF.slot_reference(drv, ap, omega, vf, l0 * 1e-6)
    mean = W[:, :3].mean(axis=0)
    peak = float(np.max(np.linalg.norm(P[:, :3], axis=1)))
    print("slot mean F (N)", mean, "reference", mean_ref, "peak", peak, "reference", peak_ref)
    assert np.allclose(mean, mean_ref, atol=0.01 * np.linalg.norm(mean_ref)), (mean, mean_ref)
    assert peak == pytest.approx(peak_ref, rel=0.10)
    rate = sum(st.analytic_m3 for st in steady) / (0.01 * len(steady))
    assert rate == pytest.approx(rate_ref, rel=5e-3)
