"""ResultPart configurations from displacement grids, part files, settings, state round trip."""

import numpy as np
import pytest

from cnctwin.core.bus import Displacement
from cnctwin.geom.io import read_mesh
from cnctwin.inspect import schema as SC
from cnctwin.inspect.result import DisplacementGrid, ResultPart
from cnctwin.inspect.state import InspectState
from tests.unit.inspect import inspfix as fx


def _grid(vec, shape=(4, 4, 4)):
    return DisplacementGrid([0.0, 0.0, 0.0], 0.01, np.broadcast_to(np.asarray(vec, float), shape + (3,)).copy())


def test_configurations_compose_the_grids():
    X = fx.box_mesh_fine([0, 0, 0], [0.03, 0.02, 0.01], 2)
    grids = {"thermal": _grid([1e-5, 0, 0]), "clamp": _grid([0, 2e-6, 0]), "residual": _grid([0, 0, 3e-6]),
             "unload_thermal": _grid([4e-6, 0, 0]), "final": _grid([0, 0, -1e-6])}
    part = ResultPart("OP10", X, grids, unload_time_s=12.0, unload_mean_temp_K=300.0, unload_max_temp_K=305.0)
    shifts = {"material": [0, 0, 0], "as_cut": [1e-5, 2e-6, 3e-6], "cold": [0, 2e-6, 3e-6], "unload": [4e-6, 0, -1e-6],
              "final": [0, 0, -1e-6]}
    for name, shift in shifts.items():
        mesh = part.configuration(name)
        np.testing.assert_allclose(mesh.vertices - X.vertices, np.tile(shift, (X.n_vertices, 1)), atol=1e-18)
        np.testing.assert_allclose(part.material_points(mesh), X.vertices, atol=0)
    assert part.unload_record()["max_temp_K"] == 305.0
    partial = ResultPart("OP10", X, {"final": _grid([0, 0, 1e-6])})
    assert partial.assumptions("unload") == ["OP10: no unload_thermal displacement grid (taken as zero in the unload "
                                            "geometry)"]
    with pytest.raises(ValueError):
        ResultPart("OP10", X, {"spring": _grid([0, 0, 0])})


def test_trilinear_grid_matches_a_linear_field_and_the_bus_record():
    nodes = np.stack(np.meshgrid(np.arange(4), np.arange(3), np.arange(5), indexing="ij"), axis=-1) * 0.01
    field = nodes @ np.array([[1e-4, 0, 0], [0, 2e-4, 0], [0, 0, -3e-4]])     # linear: interpolation exact
    disp = Displacement(origin_S_m=np.zeros(3), cell_m=0.01, total_m=field, residual_m=field * 0.5)
    g = DisplacementGrid.from_bus(disp, "residual_m")
    P = np.array([[0.005, 0.013, 0.027], [0.029, 0.001, 0.039]])
    np.testing.assert_allclose(g.at(P), 0.5 * P @ np.diag([1e-4, 2e-4, -3e-4]), atol=1e-18)


def test_part_files(tmp_path):
    X = fx.box_mesh_fine([0, 0, 0], [0.03, 0.02, 0.01], 1)
    part = ResultPart("OP10", X, {"final": _grid([0, 0, 1e-5])})
    written = part.write_files(tmp_path)
    assert written == ["part_OP10.glb", "part_OP10.stl", "part_OP10.3mf", "part_OP10_unload.stl"]
    stl = read_mesh(tmp_path / "part_OP10.stl", unit="mm")
    assert stl.bounds()[1][2] == pytest.approx(0.01001, abs=1e-8)            # float32 STL in mm
    again = tmp_path / "again"
    part.write_files(again)
    for name in written:
        assert (again / name).read_bytes() == (tmp_path / name).read_bytes()


def test_settings_and_state_round_trip():
    s = SC.settings({"sections": 12, "probe_radius_mm": 0.75})
    assert s.sections == 12 and s.probe_radius_m == pytest.approx(7.5e-4)
    with pytest.raises(Exception):
        SC.settings({"sections": 1})
    st = InspectState(records=[{"op": "OP10", "tolerances": [{"id": "a", "pass": True}]}], next_stocks=[{"op": "OP10"}],
                      finding_keys=["tol:a:0"], files=["part_OP10.stl"])
    arrays, meta = st.to_arrays()
    again = InspectState.from_arrays(arrays, meta)
    assert again == st and again.records is not st.records
