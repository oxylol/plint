"""Chordal deviation of a linearised contour around a cylindrical boss, from controller commands."""

import math

import numpy as np
import pytest

from cnctwin.controller.ir import ArcSpec, Command, Kind, LineRef, Space
from cnctwin.geom.primitives import cylinder_mesh
from cnctwin.inspect.cadref import CadReference
from cnctwin.inspect.chordal import chordal_deviation, segments_from_commands
from cnctwin.inspect.protocols import CartesianTips
from cnctwin.inspect.schema import defaults
from tests.unit.inspect import inspfix as fx

S = defaults().replace(chordal_max_distance_m=0.01)
R_BOSS, R_TOOL = 0.020, 0.005


def _cad():
    mesh = cylinder_mesh([0, 0, -0.01], [0, 0, 0.01], R_BOSS, chordal_tol_m=2e-7)
    return CadReference(fx.model(mesh, [(0, "plane", {"point_m": [0, 0, -0.01], "normal": [0, 0, -1]}, None),
                                        (1, "cylinder", {"axis_point_m": [0, 0, 0], "axis": [0, 0, 1],
                                                         "radius_m": R_BOSS}, False),
                                        (2, "plane", {"point_m": [0, 0, 0.01], "normal": [0, 0, 1]}, None)]))


def _contour(n):
    r = R_BOSS + R_TOOL
    cmds = []
    for k in range(n + 1):
        a = 2 * math.pi * k / n
        cmds.append(Command(Kind.LINEAR if k else Kind.RAPID, LineRef("op10.nc", 10 + k, f"G01 #{k}"), Space.TIP,
                            tip_target_m=(r * math.cos(a), r * math.sin(a), 0.0)))
    return cmds


def test_linear_contour_chord_error_is_the_sagitta():
    cad = _cad()
    n = 36
    segs = segments_from_commands(_contour(n), CartesianTips({}))
    assert len(segs) == n
    out = chordal_deviation(segs, cad, S)
    sagitta = (R_BOSS + R_TOOL) * (1 - math.cos(math.pi / n))
    assert out["evaluated"] == n
    # within the CAD's own tessellation (0.2 um chords): d0, d1 and d(P(s)) are distances to its facets
    assert out["max_m"] == pytest.approx(sagitta, abs=2.5e-7)
    worst = out["worst"][0]
    assert worst["toward_part"] is True and worst["face"] == 1
    assert worst["end_distances_m"][0] == pytest.approx(R_TOOL, abs=1e-6)
    assert out["exceeding"] == n                                  # 95 um >> the 10 um CAM tolerance


def test_machine_space_commands_and_arcs():
    cad = _cad()
    r = R_BOSS + R_TOOL
    offset = np.array([-0.3, -0.2, -0.4])
    h = 0.12
    modal = {"work_offset": "G54", "length_comp": True, "length_offset": 1}

    def machine(p):
        return {"X": p[0] + offset[0], "Y": p[1] + offset[1], "Z": p[2] + offset[2] + h}

    p0, p1 = (r, 0.0, 0.0), (0.0, r, 0.0)
    cmds = [Command(Kind.RAPID, LineRef("op10.nc", 1, "G0"), target=machine(p0), modal=modal),
            Command(Kind.ARC, LineRef("op10.nc", 2, "G3"), target=machine(p1), modal=modal,
                    arc=ArcSpec("XY", tuple(machine((0, 0, 0)).values()), (0, 0, 1), math.pi / 2, r, r)),
            Command(Kind.LINEAR, LineRef("op10.nc", 3, "G1"), target=machine(p0), modal=modal)]
    tips = CartesianTips({"G54": offset}, {1: h})
    segs = segments_from_commands(cmds, tips)
    assert [s.kind for s in segs] == [1, 0]
    np.testing.assert_allclose(segs[0].centre, [0, 0, 0], atol=1e-15)
    out = chordal_deviation(segs, cad, S)
    by_line = {w["line"]: w for w in out["worst"]}
    assert by_line[2]["chord_error_m"] == pytest.approx(0.0, abs=1e-6)      # the arc follows the offset circle
    assert by_line[3]["chord_error_m"] == pytest.approx(r * (1 - math.cos(math.pi / 4)), abs=2.5e-7)
