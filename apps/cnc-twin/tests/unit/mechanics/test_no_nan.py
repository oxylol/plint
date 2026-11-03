"""No NaN or infinity anywhere in the coefficient tables over the grid ranges (inline fixtures and,
when it loads, every database material with its own data)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_CHISEL, EDGE_PERIPHERAL
from cnctwin.mechanics import coefficients as CF
from cnctwin.mechanics.chipform import ChipConditions, ChipState, chip_form
from cnctwin.mechanics.materials import ToolSide, WorkMaterial

H_GRID = np.geomspace(1e-7, 2e-3, 20)
ARRAYS = ("Kt_Pa", "Kr_Pa", "Ka_Pa", "shear_angle_rad", "chip_ratio", "frac_shear_to_chip", "frac_rake_to_tool",
          "frac_flank_to_tool", "power_frac_shear", "T_shear_K", "T_interface_K", "T_chip_K", "contact_length_m",
          "sticking_fraction", "interface_shear_Pa", "interface_normal_Pa")


def _check(work, tool, rakes=(-0.2, 0.0, 0.3)):
    speeds = np.array([0.05, 1.0, 10.0])
    for rake in rakes:
        pmap = None
        if work.jc is not None:
            pmap = CF.build_process_map(work, tool, rake, speed_grid_m_s=speeds,
                                        t1_grid_m=np.geomspace(2e-7, 2e-3, 8),
                                        tool_temp_grid_K=np.array([293.15, 900.0]))
            assert np.all(np.isfinite(pmap.values))
        for incl in (0.0, 0.6):
            for kind in (EDGE_PERIPHERAL, EDGE_CHISEL):
                group = CF.EdgeGroup(rake_rad=rake if kind != EDGE_CHISEL else -0.95, incl_rad=incl,
                                     edge_radius_m=2.5e-5, kind=kind)
                for speed in (0.02, 3.0, 30.0):
                    for t_tool in (None, 293.15, 1100.0):
                        state = CF.SlowState(speed_m_s=speed, tool_temp_K=t_tool, VB_m=1.5e-4, bue_height_m=0.0)
                        gc = CF.group_coefficients(work, tool, group, state, H_GRID, pmap)
                        for name in ARRAYS:
                            assert np.all(np.isfinite(getattr(gc, name))), (work.id, name, rake, incl, kind, speed)
                        for value in (gc.Kte_N_per_m, gc.Kre_N_per_m, gc.Kae_N_per_m, gc.K_sp_N_per_m3,
                                      gc.flank_sigma_mean_Pa):
                            assert math.isfinite(value)
                        k = 12
                        st = ChipState.from_group(gc, H_GRID[k])
                        cond = ChipConditions(h_m=H_GRID[k], width_m=1e-3, speed_m_s=speed, rake_rad=group.rake_rad,
                                              incl_rad=incl)
                        cf = chip_form(work, tool, cond, st)
                        assert np.all(np.isfinite(cf.velocity_rel_tool_m_s))
                        chip_row = gc.regime[k] not in (CF.REGIME_PLOUGHING, CF.REGIME_NONE, CF.REGIME_INDENTATION)
                        assert (cf.form_code > 0) if (chip_row and st.contact_length_m > 0) else cf.form_code >= 0


def test_fixtures_finite(steel, polymer, composite, carbide):
    for work in (steel, polymer, composite):
        _check(work, carbide)


def test_database_materials_finite():
    try:
        from cnctwin.db.database import Database
        db = Database.load()
        materials = [WorkMaterial.from_db(db.material(mid)) for mid in db.ids("material")]
        tool = ToolSide.from_db(None, db.tool_material(db.ids("tool_material")[0]), None)
    except Exception as exc:          # the databases are still being researched
        pytest.skip(f"database not loadable: {exc}")
    for work in materials:
        _check(work, tool, rakes=(0.0,))
