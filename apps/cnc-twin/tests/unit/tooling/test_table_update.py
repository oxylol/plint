"""Tool table from a project tool list, and the per-macro-step composition of the effective edge."""

import math

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.tooling import build_tool, build_tool_table, edge_state, effective_edge
from cnctwin.tooling.edges import edge_trig
from cnctwin.tooling.update import rephase
from tests.unit.tooling.fixtures import HOLDER, TOOLS, endmill, instance

MACHINE = {"spindles": [{"id": "main", "kind": "turning"}, {"id": "live", "kind": "live_tool"}]}


def test_tool_table_ids_paths_spindles_and_sisters():
    entries = [
        {"number": 1, "tool": "sandvik-2p342-1000-pa", "holder": "haas-cat40-er32", "life": {"group": 3}},
        {"number": 2, "tool": TOOLS["turning_insert"], "path": 2},
        {"number": 21, "tool": "sandvik-2p342-1000-pa", "holder": HOLDER, "life": {"sister_of": 1}},
        {"number": 5, "tool": TOOLS["twist_drill"], "life": {"sister_of": 6}},
        {"number": 6, "tool": TOOLS["twist_drill"], "spindle": "main"},
    ]
    table = build_tool_table(entries, MACHINE)
    s = table.slots
    assert [x.slot for x in s] == [0, 1, 2, 3, 4]
    assert s[0].tool_id == "sandvik-2p342-1000-pa" and s[0].holder_id == "haas-cat40-er32"
    assert s[1].tool_id == "inline:1" and s[1].holder_id == "" and s[1].path == 1
    assert s[2].holder_id == "inline:2"
    assert s[0].sister_group == 3 and s[2].sister_group == 3
    assert s[3].sister_group == s[4].sister_group == 4          # a new group for tool 6
    assert s[1].spindle_id == "main"                            # turning tool -> turning spindle
    assert s[3].spindle_id == "live"                            # rotating tool on a lathe -> live tool spindle
    assert s[4].spindle_id == "main"                            # given
    assert table.by_number(2, path=1).slot == 1
    with pytest.raises(InputError):
        build_tool_table([{"number": 1, "tool": "a"}, {"number": 1, "tool": "b"}])
    with pytest.raises(InputError):
        build_tool_table([{"number": 1, "tool": "a", "life": {"sister_of": 9}}])


class Rec:
    def __init__(self, **values):
        self.__dict__.update(values)


@pytest.fixture(scope="module")
def mill():
    return build_tool(endmill(), HOLDER, instance(stickout_mm=40))


def test_wear_and_growth_move_the_edges(mill):
    nt, ne = mill.edge.elem_r_m.shape
    wear = Rec(radial_wear_m=np.full((nt, ne), 4e-6), axial_wear_m=np.full((nt, ne), 3e-6),
               edge_radius_m=np.zeros((nt, ne)), broken=False)
    heat = Rec(radial_growth_m=np.full((nt, ne), 1e-6), length_growth_m=0.0, holder_growth_m=0.0)
    eff = effective_edge(mill.edge, edge_state(mill, wear, heat, None))
    assert np.allclose(eff.elem_r_m - mill.edge.elem_r_m, -3e-6, atol=1e-18)
    assert np.allclose(eff.elem_z_m - mill.edge.elem_z_m, 3e-6, atol=1e-18)
    assert np.array_equal(eff.elem_edge_radius_m, mill.edge.elem_edge_radius_m)   # no worn radius given
    worn = Rec(radial_wear_m=None, axial_wear_m=None, edge_radius_m=np.full((nt, ne), 25e-6), broken=False)
    assert np.allclose(effective_edge(mill.edge, edge_state(mill, worn)).elem_edge_radius_m, 25e-6)


def test_pullout_holder_and_length_growth(mill):
    heat = Rec(radial_growth_m=None, length_growth_m=10e-6, holder_growth_m=2e-6)
    hold = Rec(pullout_m=5e-6, slip_rad=0.0)
    eff = effective_edge(mill.edge, edge_state(mill, None, heat, hold))
    z = mill.edge.elem_z_m
    free = 0.040
    expect = -(2e-6 + 5e-6) - 10e-6 * (free - z) / free     # uniform strain over the free length
    assert np.allclose(eff.elem_z_m - z, expect, atol=1e-18)
    assert np.allclose(eff.elem_r_m, mill.edge.elem_r_m)
    assert np.allclose(eff.station_z_m - mill.edge.station_z_m,
                       -(7e-6) - 10e-6 * (free - mill.edge.station_z_m) / free, atol=1e-18)


def test_slip_rephases_and_broken_invalidates(mill):
    hold = Rec(pullout_m=0.0, slip_rad=0.02)
    eff = effective_edge(mill.edge, edge_state(mill, None, None, hold))
    assert np.allclose(eff.tooth_phase_rad, np.mod(mill.edge.tooth_phase_rad - 0.02, 2 * math.pi))
    assert np.all((eff.tooth_phase_rad >= 0) & (eff.tooth_phase_rad < 2 * math.pi))
    trig, frames = rephase(mill, eff)
    assert np.allclose(trig.cos_a, edge_trig(eff).cos_a)
    # the frame of tooth 0 turned by the slip (right-hand tool: +0.02 rad right-handed)
    c, s = math.cos(0.02), math.sin(0.02)
    t0 = mill.frames.tangent[0]
    assert np.allclose(frames.tangent[0, :, 0], c * t0[:, 0] - s * t0[:, 1], atol=1e-12)
    broken = Rec(radial_wear_m=None, axial_wear_m=None, edge_radius_m=None, broken=True)
    assert effective_edge(mill.edge, edge_state(mill, broken)).elem_valid.sum() == 0


def test_turning_tool_grows_toward_the_work():
    tool = build_tool(TOOLS["turning_insert"], None, {"number": 1})
    heat = Rec(radial_growth_m=None, length_growth_m=6e-6, holder_growth_m=0.0)
    eff = effective_edge(tool.edge, edge_state(tool, None, heat, Rec(pullout_m=0.0, slip_rad=0.5)))
    # shank along -x_T (away from the work): the tip moves along +x_T by the growth (strain over the overhang)
    dx = eff.elem_r_m - tool.edge.elem_r_m
    assert np.all(dx > 0) and dx.max() == pytest.approx(6e-6, rel=1e-3)
    assert np.allclose(eff.tooth_phase_rad, tool.edge.tooth_phase_rad)     # no slip for non-rotating tools


def test_shape_mismatch_is_an_error(mill):
    from cnctwin.core.errors import TwinError
    with pytest.raises(TwinError):
        edge_state(mill, Rec(radial_wear_m=np.zeros((2, 2)), axial_wear_m=None, edge_radius_m=None, broken=False))
