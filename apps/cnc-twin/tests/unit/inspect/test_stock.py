"""The next operation's stock: removal's transfer conventions, the mislocation input, exact remap or rasterising."""

import numpy as np
import pytest

from cnctwin.inspect import stock as ST
from cnctwin.inspect.columns import volume_m3
from cnctwin.inspect.protocols import ColumnsStockBuilder
from cnctwin.inspect.result import DisplacementGrid, ResultPart
from tests.unit.inspect import inspfix as fx

LO, HI = np.array([0.0, 0.0, 0.0]), np.array([0.04, 0.03, 0.02])


def _part(grids=None):
    return ResultPart("OP10", fx.box_mesh_fine(LO, HI, 2), grids or {}, op_index=0)


@pytest.mark.parametrize("from_previous", [{"op": "OP10", "flip": {"axis": "X", "angle_deg": 180}},
                                           {"op": "OP10", "flip": {"axis": "Y", "angle_deg": 90}},
                                           {"op": "OP10", "flip": {"axis": "Z", "angle_deg": 30}},
                                           {"op": "OP10", "transfer": "sub_spindle"},
                                           {"op": "OP10", "transfer": "pallet"}])
def test_conventions_match_removal(from_previous):
    from cnctwin.removal import transfer as RT

    R = ST.transfer_rotation(from_previous)
    R_removal = RT.transform_of(from_previous, "block")
    if R_removal is None:
        assert R is None
        return
    np.testing.assert_allclose(R, R_removal, atol=1e-15)
    for kind in ("block", "cylinder"):
        mine = ST.new_box(LO, HI, R, kind)
        theirs = RT.new_box(LO, HI, R_removal, kind)
        for a, b in zip(mine, theirs):
            np.testing.assert_allclose(a, b, atol=1e-15)


def test_flip_moves_the_part_into_the_new_box():
    part = _part()
    ns = ST.next_stock(part, {"op": "OP10", "flip": {"axis": "X", "angle_deg": 180}}, stock_box=(LO, HI))
    lo, hi = ns.mesh.bounds()
    np.testing.assert_allclose(lo, [0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(hi, HI, atol=1e-15)
    assert ns.mesh.volume() == pytest.approx(part.material.volume(), rel=1e-12)
    # the old top (z = 0.02) is now the bottom: p_new = R p_old + t
    old_top = np.array([0.01, 0.01, 0.02])
    np.testing.assert_allclose((ns.transform_new_old @ np.append(old_top, 1))[:3], [0.01, 0.02, 0.0], atol=1e-15)
    assert ns.method == "mesh"


def test_mislocation_and_builders():
    part = _part()
    M = fx.rigid([0.0, 1.0, 0.0], 1e-3, [5e-5, 0.0, 1e-5])                 # tilted by a chip under a locator
    builder = ColumnsStockBuilder(5e-4)
    ns = ST.next_stock(part, {"op": "OP10", "flip": {"axis": "X", "angle_deg": 180}}, lattice_m=5e-4,
                       mislocation=M, builder=builder, stock_box=(LO, HI))
    np.testing.assert_allclose(ns.transform_new_old, M @ ns.nominal_transform, atol=1e-15)
    assert ns.method == "rasterised"
    assert volume_m3(ns.stock_field) == pytest.approx(part.material.volume(), rel=2e-3)
    assert "mislocation" in " ".join(ns.notes)

    class Exact:
        def remap(self, workpiece, rotation, kind):
            return ("remapped", tuple(np.round(rotation.reshape(-1), 12)))

        def from_mesh(self, *args):
            raise AssertionError("an exact remap is possible")

    ns2 = ST.next_stock(part, {"op": "OP10", "flip": {"axis": "X", "angle_deg": 180}}, builder=Exact(),
                        stock_box=(LO, HI))
    assert ns2.method == "exact remap" and ns2.stock_field[0] == "remapped"
    # a distorted part cannot be remapped: the next stock carries its spring-back
    bow = DisplacementGrid([0, 0, 0], 0.01, np.full((5, 4, 3, 3), [0.0, 0.0, 2e-6]))
    ns3 = ST.next_stock(_part({"final": bow}), None, lattice_m=5e-4, builder=builder)
    assert ns3.method == "rasterised"
    assert ns3.mesh.bounds()[0][2] == pytest.approx(2e-6, abs=1e-15)
    with pytest.raises(Exception):
        ST.next_stock(part, None, mislocation=np.diag([1.0, 1.0, 1.01, 1.0]))


def test_sub_spindle_keeps_a_turned_axis():
    part = _part()
    ns = ST.next_stock(part, {"op": "OP10", "transfer": "sub_spindle"}, kind="cylinder", stock_box=(LO, HI))
    np.testing.assert_allclose(ns.nominal_transform[:3, :3], np.diag([-1.0, 1.0, -1.0]), atol=1e-15)
    assert ns.box_lo_m[2] == pytest.approx(0.0, abs=1e-15)
    assert ns.nominal_transform[0, 3] == 0.0 and ns.nominal_transform[1, 3] == 0.0
