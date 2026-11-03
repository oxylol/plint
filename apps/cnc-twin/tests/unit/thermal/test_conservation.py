"""Energy ledgers of the thermal solvers: cut cells, removal, deposits, every boundary class."""

import math

import numpy as np
import pytest

from cnctwin.core.bus import CuttingContact
from cnctwin.thermal import kernels as K
from cnctwin.thermal import props
from cnctwin.thermal.abi import N_L, L_U0, L_U1
from cnctwin.thermal.partition import FractionTable, split_contact
from cnctwin.thermal.protocols import BoxStock, SampledStock
from cnctwin.thermal.schema import ThermalSettings
from cnctwin.thermal.workpiece import WorkEnvironment, WorkThermal

import fixtures as F


def steel():
    return props.mat_table({"value": 7850.0}, {"table": [[20, 49.8], [500, 36.0]]},
                           {"table": [[20, 460.0], [300, 524.0], [500, 615.0]]}, {"value": 1.2e-5}, "steel")


def environment(coolant=True):
    return WorkEnvironment(T_air_K=293.15, T_table_K=294.0, coolant_on=coolant, T_fluid_K=298.0, h_far_W_m2K=2000.0,
                           near=np.array([[0.02, 0.015, 0.02, 0.006, 9000.0]]),
                           wet=np.array([[1, 0], [0, 1]], dtype=np.int8), wet_origin_S_m=np.zeros(3), wet_cell_m=0.02)


def test_ledger_closes_with_cut_cells_removal_and_deposits():
    geo = SampledStock(lambda p: (p[:, 0] ** 2 + (p[:, 2] - 0.0) ** 2 < 0.0215 ** 2) & (p[:, 1] < 0.0285),
                       [0.0, 0.0, 0.0], [0.021, 0.03, 0.021], samples=4)
    work = WorkThermal(0, geo, steel(), ThermalSettings(cell_m=3e-3), 293.15, "steel", 0.0)
    f = work.grid.f
    assert np.count_nonzero((f > 0) & (f < 1)) > 10            # a curved surface: many cut cells
    rng = np.random.default_rng(4)
    for step in range(12):
        rows = np.zeros((6, 8))
        rows[:, 0] = rng.uniform(0.0, 0.02, 6)
        rows[:, 1] = rng.uniform(0.0, 0.028, 6)
        rows[:, 2] = rng.uniform(0.0, 0.02, 6)
        rows[:, 5] = 1.0
        rows[:, 6] = rng.uniform(5.0, 50.0, 6)
        work.add_rows(rows)
        if step == 4:
            geo.inside = lambda p: (p[:, 0] ** 2 + p[:, 2] ** 2 < 0.0215 ** 2) & (p[:, 1] < 0.0285) & \
                ~((p[:, 0] > 0.006) & (p[:, 2] > 0.012))
            geo.mark_changed([0.0, 0.0, 0.0], [0.021, 0.03, 0.021])
        result = work.fv_step(0.2 if step % 2 else 3.0, environment(coolant=step % 3 != 0), 0.2 * (step + 1))
        terms = result["deposited_J"] - (result["air_J"] + result["coolant_J"] + result["radiation_J"]
                                         + result["contact_J"]) - (result["removed_J"] - result["added_J"])
        du = result["u_after_J"] - result["u_before_J"]
        assert du == pytest.approx(terms, rel=1e-11, abs=1e-9)
        if step == 4:
            assert result["removed_J"] > 0.0
    assert work.ledger.worst_rel < 1e-12
    assert work.fixture.ledger.worst_rel < 1e-12


def test_removed_enthalpy_is_the_enthalpy_of_the_removed_fractions():
    geo = BoxStock([0, 0, 0], [0.02, 0.02, 0.02])
    mat = steel()
    work = WorkThermal(0, geo, mat, ThermalSettings(cell_m=2e-3), 293.15, "steel", 0.0)
    work.add_rows(np.array([[0.01, 0.01, 0.02, 0, 0, 1, 200.0, 0.0]]))
    work.fv_step(1.0, environment(False), 1.0)
    before_f = work.grid.f.copy()
    before_h = work.grid.h.copy()
    geo.remove_box([0.004, 0.004, 0.015], [0.0161, 0.0139, 0.02])
    removed, added = work.apply_fill_changes()
    rho_v = mat.rho_kg_m3[0] * 8e-9
    expected = float(np.sum(rho_v * (before_f - work.grid.f) * before_h))
    assert removed == pytest.approx(expected, rel=1e-12)
    assert added == 0.0
    energy = np.zeros(4)
    K.work_energy(work.grid, mat, 1e-9, energy)
    assert energy[0] == pytest.approx(work.ledger.u_J - removed, rel=1e-12)


def test_partition_destinations_sum_to_the_generation():
    edges = F.edge_model(nt=3, ne=5)
    rng = np.random.default_rng(2)
    shape = (3, 5)
    in_cut = rng.uniform(0.0, 3e-3, shape)
    in_cut[0, 0] = 0.0
    cut = CuttingContact(contact=0, tool_slot=0, heat_shear_J=3.0, heat_rake_J=1.2, heat_flank_J=0.4, ploughing_J=0.05,
                         in_cut_time_s=in_cut, int_h_dt=in_cut * rng.uniform(1e-5, 1e-4, shape),
                         rake_sliding_m=in_cut * 2.0, flank_sliding_m=in_cut * 4.0, int_sigma_v_rake=in_cut * 1e9,
                         int_sigma_v_flank=in_cut * 2e8, entries=np.ones(shape, np.int32))
    for table in (FractionTable.from_coefficients(F.coefficients()), None):
        heat = split_contact(cut, 0, table, edges, None)
        total = heat.chip_J + heat.tool_J + heat.work_J
        assert total == pytest.approx(4.65, rel=1e-13)
        assert heat.tool_J == pytest.approx(heat.tool_rake_elem_J.sum() + heat.tool_flank_elem_J.sum(), rel=1e-13)
        assert heat.work_J == pytest.approx(heat.work_elem_J.sum(), rel=1e-13)
        assert heat.fallback == (table is None)
    # no per-element data at all: everything still goes somewhere
    bare = CuttingContact(contact=1, tool_slot=0, heat_shear_J=1.0, heat_rake_J=0.5, heat_flank_J=0.2)
    heat = split_contact(bare, 0, None, None, None)
    assert heat.chip_J + heat.tool_J + heat.work_J == pytest.approx(1.7, rel=1e-13)


def test_tool_ladder_and_lumped_bodies_close():
    from cnctwin.thermal.machine import LumpedBody
    from cnctwin.thermal.ledger import SolverLedger
    body = LumpedBody.at("b", 2.0e4, 5.0, 293.15, SolverLedger("b"))
    for dt in (1e-3, 0.01, 10.0, 3600.0):
        body.step(dt, 80.0, 295.0, 0.4, 1e-12, 0.0)
    assert body.ledger.worst_rel < 1e-13
