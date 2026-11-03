"""Remaining material and gouges of a known pocket: volumes, regions, locations."""

import numpy as np
import pytest

from cnctwin.inspect import abi as A
from cnctwin.inspect.cadref import CadReference
from cnctwin.inspect.columns import columns_from_mesh, compare, compare_meshes, difference, lattice_for, regions, volume_m3
from cnctwin.inspect.schema import defaults
from tests.unit.inspect import inspfix as fx

S = defaults()
LO, HI = [0.0, 0.0, 0.0], [0.040, 0.030, 0.015]
# pocket walls on cell boundaries of a 0.5 mm lattice with nodes at multiples of 0.5 mm: x 12.25-28.25, y 9.25-21.25
PLO, PHI = [0.01225, 0.00925], [0.02825, 0.02125]
CELL = 5e-4


def _cols(mesh):
    origin, cell, dims = lattice_for(LO, HI, CELL)
    return columns_from_mesh(mesh, origin, cell, dims)


def test_block_volume_is_exact():
    cols = _cols(fx.box_mesh_fine([0.00025, 0.00025, 0.0], [0.03975, 0.02975, 0.015], 2))
    assert volume_m3(cols) == pytest.approx(0.0395 * 0.0295 * 0.015, rel=1e-12)


def test_remaining_material_of_a_shallow_pocket():
    cad_mesh = fx.pocket_block(LO, HI, PLO, PHI, 0.002)
    cad = CadReference(fx.model(cad_mesh, fx.pocket_specs(LO, HI, PLO, PHI, 0.002)))
    machined = fx.pocket_block(LO, HI, PLO, PHI, 0.0015)          # 0.5 mm left on the floor
    out = compare(_cols(machined), _cols(cad_mesh), S, cad)
    assert len(out["remaining"]) == 1 and out["gouges"] == []
    region = out["remaining"][0]
    area = (PHI[0] - PLO[0]) * (PHI[1] - PLO[1])
    assert region["volume_m3"] == pytest.approx(area * 0.0005, rel=1e-12)            # 16 x 12 x 0.5 = 96 mm^3
    np.testing.assert_allclose(region["centroid_S_m"], [0.02025, 0.01525, 0.01325], atol=1e-12)
    assert region["max_thickness_m"] == pytest.approx(0.0005, rel=1e-12)
    np.testing.assert_allclose(region["box_lo_S_m"], [PLO[0], PLO[1], 0.013], atol=1e-12)
    assert region["nearest_face"] == 10                                              # the pocket floor
    assert out["totals"]["excess_volume_m3"] == pytest.approx(area * 0.0005, rel=1e-12)


def test_gouges_and_separate_regions():
    cad_mesh = fx.pocket_block(LO, HI, PLO, PHI, 0.002)
    deep = fx.pocket_block(LO, HI, PLO, PHI, 0.0022)               # 0.2 mm too deep
    out = compare(_cols(deep), _cols(cad_mesh), S)
    assert out["remaining"] == [] and len(out["gouges"]) == 1
    area = (PHI[0] - PLO[0]) * (PHI[1] - PLO[1])
    assert out["gouges"][0]["volume_m3"] == pytest.approx(area * 0.0002, rel=1e-12)
    # two pockets left on two separate islands of a field: two regions, ordered by volume
    field = _cols(fx.box_mesh_fine([0.00025, 0.00025, 0.0], [0.03975, 0.02975, 0.015], 2))
    target = _cols(fx.box_mesh_fine([0.00025, 0.00025, 0.0], [0.03975, 0.02975, 0.010], 2))
    extra = difference(field, target)
    assert volume_m3(extra) == pytest.approx(0.0395 * 0.0295 * 0.005, rel=1e-12)
    start = [0]
    segs = []
    for c in range(int(extra.dims[0] * extra.dims[1])):
        i, j = divmod(c, int(extra.dims[1]))
        if (5 <= i < 10 and 5 <= j < 8) or (30 <= i < 32 and 20 <= j < 30):
            segs.append([0.0, 0.001])
        start.append(len(segs))
    two = A.make_columns(extra.origin_m, extra.cell_m, extra.dims, start, np.array(segs))
    rows = regions(two)
    assert len(rows) == 2
    assert rows[0, A.CS_VOLUME] == pytest.approx(20 * CELL * CELL * 0.001, rel=1e-12)
    assert rows[1, A.CS_VOLUME] == pytest.approx(15 * CELL * CELL * 0.001, rel=1e-12)


def test_compare_meshes_finds_corner_material():
    cad_mesh = fx.pocket_block(LO, HI, PLO, PHI, 0.002)
    out = compare_meshes(fx.pocket_block(LO, HI, PLO, PHI, 0.0015), cad_mesh, 2.5e-4, S)   # nodes at 0.125 + k/4 mm
    assert out["totals"]["remaining"]["regions"] == 1
    area = (PHI[0] - PLO[0]) * (PHI[1] - PLO[1])
    assert out["totals"]["remaining"]["volume_m3"] == pytest.approx(area * 0.0005, rel=1e-9)
