"""Deviation map of known offset surfaces: every face of a box grown by delta, a cylinder grown radially."""

import numpy as np
import pytest

from cnctwin.geom.primitives import cylinder_mesh
from cnctwin.inspect.cadref import CadReference, cad_from_mesh
from cnctwin.inspect.deviation import brick_attributes, deviation_map, statistics, vertex_deviation
from cnctwin.inspect.sampling import assign, vertex_weights
from cnctwin.inspect.schema import defaults
from tests.unit.inspect import inspfix as fx

S = defaults()


def test_offset_box_deviation_per_face():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([0.03, 0.02, 0.01])
    cad = CadReference(fx.box_model(lo, hi, 2))
    delta = 2.5e-5
    grown = fx.box_mesh_fine(lo - delta, hi + delta, 16)
    a = assign(grown, cad, S)
    dev = deviation_map(a, cad, 1e-6)
    for face in dev["faces"]:
        usable = face["usable"]
        assert usable["points"] > 100
        assert usable["mean_m"] == pytest.approx(delta, abs=1e-15)
        assert usable["min_m"] == pytest.approx(delta, abs=1e-15) and usable["max_m"] == pytest.approx(delta, abs=1e-15)
        assert usable["missing_area_m2"] == 0.0
    # every vertex: + delta on faces, up to sqrt(3) delta at the corners (closest CAD point a corner)
    surface = dev["surface"]
    assert surface["min_m"] == pytest.approx(delta, abs=1e-15)
    assert surface["max_m"] == pytest.approx(np.sqrt(3) * delta, rel=1e-12)
    assert surface["area_m2"] == pytest.approx(grown.area(), rel=1e-12)
    # shrunk: material missing everywhere
    shrunk = fx.box_mesh_fine(lo + delta, hi - delta, 16)
    d2 = deviation_map(assign(shrunk, cad, S), cad, 1e-6)
    assert d2["usable"]["max_m"] == pytest.approx(-delta, abs=1e-15)
    assert d2["usable"]["missing_area_m2"] == pytest.approx(d2["usable"]["area_m2"])


def test_offset_cylinder_and_weights():
    cad_mesh = cylinder_mesh([0, 0, 0], [0, 0, 0.02], 0.005, chordal_tol_m=1e-6)
    cad = CadReference(fx.model(cad_mesh, [(0, "plane", {"point_m": [0, 0, 0], "normal": [0, 0, -1]}, None),
                                           (1, "cylinder", {"axis_point_m": [0, 0, 0], "axis": [0, 0, 1],
                                                            "radius_m": 0.005}, False),
                                           (2, "plane", {"point_m": [0, 0, 0.02], "normal": [0, 0, 1]}, None)]))
    grown = fx.subdivide(cylinder_mesh([0, 0, -1e-5], [0, 0, 0.02001], 0.005 + 1e-5, chordal_tol_m=1e-6), 1)
    a = assign(grown, cad, S)
    side = [f for f in deviation_map(a, cad, 1e-6)["faces"] if f["face"] == 1][0]["usable"]
    # vertices of the grown cylinder lie on its circle (+10 um); those on chords up to the sagitta of both facets
    assert side["max_m"] == pytest.approx(1e-5, abs=2e-6)
    assert side["mean_m"] == pytest.approx(1e-5, abs=2e-6)
    w = vertex_weights(grown)
    assert w.sum() == pytest.approx(grown.area(), rel=1e-12)
    st = statistics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0]), np.zeros((3, 3)), 0.0)
    assert st["mean_m"] == pytest.approx(2.25) and st["max_abs_m"] == 3.0


def test_vertex_deviation_bricks_and_plain_mesh_cad():
    lo, hi = np.zeros(3), np.array([0.01, 0.01, 0.01])
    cad = CadReference(cad_from_mesh(fx.box_mesh_fine(lo, hi, 1)))
    assert all(f.surface == "plane" for f in cad.faces)
    V = np.array([[0.005, 0.005, 0.0101], [0.005, 0.005, 0.0099]])
    np.testing.assert_allclose(vertex_deviation(cad, V), [1e-4, -1e-4], atol=1e-15)
    attrs = brick_attributes(cad, {"3": (V, np.array([[0, 1, 0]]))}, displace=lambda X: np.tile([0, 0, 1e-5], (len(X), 1)))
    np.testing.assert_allclose(attrs["3"]["deviation_m"], [1.1e-4, -0.9e-4], atol=1e-10)
