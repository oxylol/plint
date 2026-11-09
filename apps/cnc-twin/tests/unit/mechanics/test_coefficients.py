"""Process maps, coefficient rows, calibration and the ProcessCoefficients assembly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_CHISEL, EDGE_MARGIN
from cnctwin.mechanics import abi, coefficients as CF, oxley

H_GRID = np.geomspace(2e-6, 1e-3, 24)


@pytest.fixture(scope="module")
def steel_map(steel, carbide):
    return CF.build_process_map(steel, carbide, 0.0, speed_grid_m_s=np.array([1.5, 3.0]),
                                t1_grid_m=np.geomspace(5e-7, 1e-3, 10), tool_temp_grid_K=np.array([293.15, 800.0]))


def test_map_nodes_equal_direct_solutions(steel, carbide, steel_map):
    t1 = steel_map.t1_grid_m[6]
    row = CF.lookup(steel_map, t1, 3.0, 800.0)[0]
    direct = oxley.solve(steel, t1, 3.0, 0.0, tool=carbide, tool_temp_K=800.0)
    assert row[abi.OX_FC] == pytest.approx(direct.cutting_force_N_per_m, rel=1e-12)
    assert row[abi.OX_T_INT] == pytest.approx(direct.T_interface_K, rel=1e-12)
    assert row[abi.OX_STATUS] == abi.STATUS_OXLEY
    assert np.all(np.isfinite(steel_map.values))


def test_map_interpolation_error_between_nodes(steel, carbide, steel_map):
    """Midway between nodes of this coarse map (t1 factor 2.3, speed factor 2, tool 293/800 K):
    within 4 % next to the edge of the admissible Oxley region (nodes 6-7, where the shear angle
    changes fastest with t1) and within 2 % away from it (nodes 7-8)."""
    for (a, b), tol in (((6, 7), 0.04), ((7, 8), 0.02)):
        t1 = math.sqrt(steel_map.t1_grid_m[a] * steel_map.t1_grid_m[b])
        row = CF.lookup(steel_map, t1, math.sqrt(1.5 * 3.0), 550.0)[0]
        direct = oxley.solve(steel, t1, math.sqrt(1.5 * 3.0), 0.0, tool=carbide, tool_temp_K=550.0)
        assert row[abi.OX_FC] == pytest.approx(direct.cutting_force_N_per_m, rel=tol)
        assert row[abi.OX_T_INT] == pytest.approx(direct.T_interface_K, rel=tol)


def test_map_failed_nodes_use_fallback(steel):
    pmap = CF.build_process_map(steel, None, math.radians(-40), speed_grid_m_s=np.array([2.0]),
                                t1_grid_m=np.geomspace(1e-6, 1e-3, 6))
    status = pmap.values[..., abi.OX_STATUS]
    assert set(np.unique(status)) <= {abi.STATUS_OXLEY, abi.STATUS_MERCHANT, abi.STATUS_INDENTATION}
    assert np.all(np.isfinite(pmap.values))


def test_group_rows_follow_the_force_convention(steel, carbide, steel_map):
    group = CF.EdgeGroup(rake_rad=0.0, edge_radius_m=4e-5)
    state = CF.SlowState(speed_m_s=2.5, tool_temp_K=600.0)
    gc = CF.group_coefficients(steel, carbide, group, state, H_GRID, steel_map)
    h_min = gc.h_min_m
    assert h_min == pytest.approx(0.27 * 4e-5)
    # above h_min: K h + Ke = shearing force at t1 = h - h_min plus the edge force
    k = int(np.searchsorted(H_GRID, 2e-4))
    t1 = H_GRID[k] - h_min
    row = CF.lookup(steel_map, t1, 2.5, 600.0)[0]
    total_t = gc.Kt_Pa[k] * H_GRID[k] + gc.Kte_N_per_m
    total_r = gc.Kr_Pa[k] * H_GRID[k] + gc.Kre_N_per_m
    assert total_t == pytest.approx(row[abi.OX_FC] + gc.Kte_N_per_m, rel=1e-9)
    assert total_r == pytest.approx(row[abi.OX_FT] + gc.Kre_N_per_m, rel=1e-9)
    assert gc.Ka_Pa[k] == pytest.approx(0.0, abs=1e-6)
    # continuity of the total at h_min and ploughing below it
    forces = gc.forces_per_width(np.array([h_min * 0.999, h_min * 1.001]))
    assert forces[0, 0] == pytest.approx(forces[0, 1], rel=0.05)
    assert np.all(gc.regime[H_GRID <= h_min] == CF.REGIME_PLOUGHING)
    assert gc.K_sp_source == "model" and gc.K_sp_N_per_m3 > 0.0
    for name in ("Kt_Pa", "Kr_Pa", "frac_shear_to_chip", "frac_rake_to_tool", "frac_flank_to_tool",
                 "T_shear_K", "T_interface_K", "T_chip_K", "power_frac_shear"):
        assert np.all(np.isfinite(getattr(gc, name))), name
    shear = gc.regime == CF.REGIME_OXLEY
    assert np.all((gc.frac_rake_to_tool[shear] > 0) & (gc.frac_rake_to_tool[shear] < 1))
    assert np.all(gc.T_interface_K[shear] >= gc.T_shear_K[shear])


def test_oblique_group_has_axial_force(steel, carbide, steel_map):
    straight = CF.group_coefficients(steel, carbide, CF.EdgeGroup(0.0, 0.0, edge_radius_m=2e-5),
                                     CF.SlowState(2.5, 600.0), H_GRID, steel_map)
    helix = CF.group_coefficients(steel, carbide, CF.EdgeGroup(0.0, math.radians(35), edge_radius_m=2e-5),
                                  CF.SlowState(2.5, 600.0), H_GRID, steel_map)
    k = int(np.searchsorted(H_GRID, 1e-4))
    assert helix.Ka_Pa[k] > 0.0 and straight.Ka_Pa[k] == pytest.approx(0.0, abs=1e-3)
    assert helix.Kae_N_per_m > 0.0


def test_flank_wear_and_heat(steel, carbide, steel_map):
    group = CF.EdgeGroup(rake_rad=0.0, edge_radius_m=3e-5)
    fresh = CF.group_coefficients(steel, carbide, group, CF.SlowState(2.5, 600.0), H_GRID, steel_map)
    worn = CF.group_coefficients(steel, carbide, group, CF.SlowState(2.5, 600.0, VB_m=2e-4), H_GRID, steel_map)
    assert worn.flank_sigma_mean_Pa > 0.0 and fresh.flank_sigma_mean_Pa == 0.0
    assert worn.flank_sigma_mean_Pa < worn.flank_sigma0_Pa       # beyond VB* the land stress decays
    assert np.all((worn.frac_flank_to_tool >= 0) & (worn.frac_flank_to_tool <= 1))
    vb = np.array([[0.0, 1e-4], [2e-4, 3e-4]])
    flank = CF.flank_coefficients([worn], np.zeros((2, 2), np.int32), vb)
    assert flank[0, 0] == 0.0 and flank[0, 1] == pytest.approx(worn.flank_sigma0_Pa)


def test_chisel_and_margin_groups(steel, carbide, steel_map):
    chisel = CF.group_coefficients(steel, carbide, CF.EdgeGroup(math.radians(-55), 0.0, edge_radius_m=3e-5,
                                                                kind=EDGE_CHISEL), CF.SlowState(0.5, 500.0),
                                   H_GRID, steel_map)
    assert np.all(chisel.regime == CF.REGIME_INDENTATION)
    assert chisel.Kte_N_per_m == 0.0 and np.all(chisel.Kr_Pa > 0.0)
    margin = CF.group_coefficients(steel, carbide, CF.EdgeGroup(0.0, 0.0, edge_radius_m=2e-5, kind=EDGE_MARGIN),
                                   CF.SlowState(2.5, 500.0), H_GRID, steel_map)
    assert np.all(margin.Kt_Pa == 0.0) and margin.h_min_m == H_GRID[-1] and margin.K_sp_N_per_m3 > 0.0


def test_bue_blends_toward_the_bue_rake(steel, carbide, steel_map):
    bue_map = CF.build_process_map(steel, carbide, math.radians(25), speed_grid_m_s=np.array([1.5, 3.0]),
                                   t1_grid_m=steel_map.t1_grid_m, tool_temp_grid_K=steel_map.Ttool_grid_K)
    group = CF.EdgeGroup(rake_rad=0.0, edge_radius_m=2e-5)
    base = CF.group_coefficients(steel, carbide, group, CF.SlowState(2.0, 500.0), H_GRID, steel_map)
    bue = CF.group_coefficients(steel, carbide, group, CF.SlowState(2.0, 500.0, bue_height_m=1e-4), H_GRID,
                                steel_map, bue_map=bue_map)
    k = int(np.searchsorted(H_GRID, 1e-4))
    assert bue.Kt_Pa[k] < base.Kt_Pa[k]


def test_calibration_reproduces_kienzle(steel, carbide):
    group = CF.EdgeGroup(rake_rad=0.0, edge_radius_m=3.75e-5)
    cal = CF.build_calibration(steel, carbide, group, "turning")
    assert cal.kind == "kienzle" and cal.speeds_m_s[0] == pytest.approx(100 / 60)
    entry = steel.kienzle[0]
    h = np.geomspace(5e-5, 4e-4, 5)
    pmap = CF.build_process_map(steel, carbide, 0.0, speed_grid_m_s=np.array([100 / 60]),
                                t1_grid_m=np.geomspace(5e-7, 1e-3, 14), tool_temp_grid_K=np.array([293.15]))
    gc = CF.group_coefficients(steel, carbide, group, CF.SlowState(100 / 60, 293.15), h, pmap, calibration=cal)
    forces = gc.forces_per_width(h)
    # the map here is coarser than the calibration's own: agreement within the interpolation error
    assert forces[0] == pytest.approx(entry.cutting_force_per_width(h), rel=0.03)
    assert forces[1] == pytest.approx(entry.feed_force_per_width(h), rel=0.03)
    # away from the calibration speed Oxley carries the speed dependence
    fast = CF.group_coefficients(steel, carbide, group, CF.SlowState(5.0, 293.15), h,
                                 CF.build_process_map(steel, carbide, 0.0, speed_grid_m_s=np.array([5.0]),
                                                      t1_grid_m=np.geomspace(5e-7, 1e-3, 14),
                                                      tool_temp_grid_K=np.array([293.15])), calibration=cal)
    assert np.all(fast.forces_per_width(h)[0] < forces[0])
    assert gc.calibration == "kienzle"


def test_assembly(steel, carbide, steel_map):
    groups = [CF.group_coefficients(steel, carbide, CF.EdgeGroup(0.0, 0.0, edge_radius_m=2e-5),
                                    CF.SlowState(2.5, 500.0), H_GRID, steel_map),
              CF.group_coefficients(steel, carbide, CF.EdgeGroup(0.0, 0.5, edge_radius_m=2e-5),
                                    CF.SlowState(2.5, 500.0), H_GRID, steel_map)]
    flank = CF.flank_coefficients(groups, np.array([[0, 1]], np.int32), np.array([[1e-4, 2e-4]]))
    pc = CF.process_coefficients(groups, tool_slot=3, version=7, flank_coeff_N_per_m2=flank)
    assert pc.Kt_Pa.shape == (2, H_GRID.size) and pc.Kte_N_per_m.shape == (2,) and pc.version == 7
    table = CF.coeff_table(pc)
    assert table.K_Pa.shape == (2, H_GRID.size, 3) and table.frac.shape == (2, H_GRID.size, 3)
    assert table.K_Pa.flags.c_contiguous and table.v_min_m_s.shape == (1,)
    for arr in table:
        assert np.all(np.isfinite(arr))


def test_data_only_material(polymer, composite, carbide):
    group = CF.EdgeGroup(rake_rad=math.radians(10), edge_radius_m=1e-5)
    gc = CF.group_coefficients(polymer, carbide, group, CF.SlowState(5.0, 400.0), H_GRID, None)
    entry = polymer.kienzle[0]
    k = int(np.searchsorted(H_GRID, 2e-4))
    t1 = H_GRID[k] - gc.h_min_m
    assert gc.Kt_Pa[k] * H_GRID[k] == pytest.approx(float(entry.cutting_force_per_width(np.array([t1]))[0]))
    assert np.all(gc.regime[H_GRID > gc.h_min_m] == CF.REGIME_DATA)
    assert np.all(np.isfinite(gc.T_interface_K)) and np.max(gc.T_interface_K) > 400.0
    cf = CF.group_coefficients(composite, carbide, group, CF.SlowState(5.0, 400.0), H_GRID, None)
    assert np.all(np.isfinite(cf.Kt_Pa))


def test_map_matches(steel_map):
    assert CF.map_matches(steel_map, 0.0, 293.15, float(steel_map.context[abi.CTX_MU]), rake_tol_rad=1e-3,
                          temp_tol_K=5.0, mu_rel_tol=0.01)
    assert not CF.map_matches(steel_map, 0.1, 293.15, float(steel_map.context[abi.CTX_MU]), rake_tol_rad=1e-3,
                              temp_tol_K=5.0, mu_rel_tol=0.01)
    # the context holds the effective friction coefficient and the interface mode (Oxley's condition here)
    assert steel_map.context[abi.CTX_MU] == pytest.approx(0.5) and steel_map.context[abi.CTX_COULOMB] == 0.0
    assert CF.map_matches(steel_map, 0.0, 293.15, 0.5, rake_tol_rad=1e-3, temp_tol_K=5.0, mu_rel_tol=0.01,
                          coulomb_limit=False)
    assert not CF.map_matches(steel_map, 0.0, 293.15, 0.5, rake_tol_rad=1e-3, temp_tol_K=5.0, mu_rel_tol=0.01,
                              coulomb_limit=True)
    assert not CF.map_matches(steel_map, 0.0, 293.15, 0.3, rake_tol_rad=1e-3, temp_tol_K=5.0, mu_rel_tol=0.01)


def test_power_fractions_sum_to_one(steel, carbide, steel_map):
    group = CF.EdgeGroup(rake_rad=0.0, edge_radius_m=3e-5)
    state = CF.SlowState(speed_m_s=2.5, tool_temp_K=600.0)
    gc = CF.group_coefficients(steel, carbide, group, state, H_GRID, steel_map)
    total = gc.power_frac_shear + gc.power_frac_rake + gc.power_frac_edge
    assert np.allclose(total, 1.0, rtol=0, atol=1e-12)
    k = int(np.searchsorted(H_GRID, 2e-4))
    force_t = gc.Kt_Pa[k] * H_GRID[k] + gc.Kte_N_per_m
    assert gc.power_frac_edge[k] == pytest.approx(gc.Kte_N_per_m / force_t, rel=1e-9)
    row = CF.lookup(steel_map, H_GRID[k] - gc.h_min_m, 2.5, 600.0)[0]
    shear_share = gc.power_frac_shear[k] / (gc.power_frac_shear[k] + gc.power_frac_rake[k])
    assert shear_share == pytest.approx(row[abi.OX_P_SHEAR], rel=1e-9)
    assert np.all(gc.power_frac_edge[H_GRID <= gc.h_min_m] == 1.0)
