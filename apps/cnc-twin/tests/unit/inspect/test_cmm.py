"""Virtual CMM: probe plans, touch probing with stylus radius compensation, on-demand probes, CMM evaluation."""

import math

import numpy as np
import pytest

from cnctwin.inspect import abi as A
from cnctwin.inspect import gdt
from cnctwin.inspect.cadref import CadReference
from cnctwin.inspect.cmm import VirtualCMM, plan
from cnctwin.inspect.sampling import PointSet
from cnctwin.inspect.schema import defaults
from cnctwin.inspect.specs import read_tolerance
from tests.unit.inspect import inspfix as fx

S = defaults()


def test_plane_plan_stays_inside_with_inset():
    lo, hi = np.zeros(3), np.array([0.03, 0.02, 0.01])
    cad = CadReference(fx.box_model(lo, hi, 2))
    p = plan(cad, [5], S)                                    # +z face
    inset = S.probe_radius_m + S.probe_inset_m
    assert 3 <= len(p) <= S.probe_points_plane
    assert np.all(p.nominal[:, 0] >= inset - 1e-12) and np.all(p.nominal[:, 0] <= 0.03 - inset + 1e-12)
    assert np.all(p.nominal[:, 1] >= inset - 1e-12) and np.all(p.nominal[:, 1] <= 0.02 - inset + 1e-12)
    np.testing.assert_allclose(p.nominal[:, 2], 0.01, atol=1e-15)
    np.testing.assert_allclose(p.normals, np.tile([0, 0, 1.0], (len(p), 1)), atol=1e-15)


def test_probing_compensates_the_ball_radius():
    lo, hi = np.zeros(3), np.array([0.03, 0.02, 0.01])
    cad = CadReference(fx.box_model(lo, hi, 2))
    delta = 3e-5
    cmm = VirtualCMM(fx.box_mesh_fine(lo - delta, hi + delta, 4), S)
    for face in range(6):
        res = cmm.measure(plan(cad, [face], S))
        assert np.all(res.ok)
        np.testing.assert_allclose(res.deviation, delta, atol=1e-9)
        # the compensated point and the exact contact coincide on a surface normal to the vector
        np.testing.assert_allclose(res.measured, res.contact, atol=1e-9)
    hit = cmm.probe([0.015, 0.01, 0.02], [0, 0, -1])
    assert hit["status"] == "contact"
    np.testing.assert_allclose(hit["contact_S_m"], [0.015, 0.01, 0.01 + delta], atol=1e-12)
    np.testing.assert_allclose(hit["centre_S_m"], [0.015, 0.01, 0.01 + delta + S.probe_radius_m], atol=1e-9)
    miss = cmm.probe([0.1, 0.1, 0.1], [0, 0, 1])
    assert miss["status"] == "no contact"


def test_probing_a_hole_and_a_tilted_surface():
    size, t, c, r = (0.06, 0.04), 0.01, (0.04, 0.025), 0.004
    cad = CadReference(fx.model(fx.plate_with_hole(size, t, c, r, 256), fx.plate_specs(size, t, c, r)))
    result = fx.plate_with_hole(size, t, c, r + 1e-5, 256)
    cmm = VirtualCMM(result, S)
    p = plan(cad, [6], S)
    assert len(p) == S.probe_levels * S.probe_points_circle
    res = cmm.measure(p)
    assert np.all(res.ok)
    sagitta = (r + 1e-5) * (1 - math.cos(math.pi / 256))
    # the ball rests on the 256-gon's facets, between its corners (r + 10 um) and its apothem: the hole looks up to
    # a sagitta smaller (the deviation points into the hole, toward the axis)
    assert np.all(res.deviation >= -1e-5 - 1e-9) and np.all(res.deviation <= -1e-5 + sagitta + 1e-9)
    # a probe on a surface tilted by theta from its vector: compensation error r (1/cos(theta) - 1)
    tilt = math.radians(5.0)
    hit = cmm.probe([0.01, 0.01, 0.02], [math.sin(tilt), 0, -math.cos(tilt)], vector=[0, 0, 1])
    assert hit["status"] == "contact"
    err = hit["measured_S_m"][2] - 0.01
    assert err == pytest.approx(0.0, abs=1e-9)          # flat top: vector compensation along z is exact


def test_cmm_evaluation_matches_the_dense_one():
    size, t, c, r = (0.06, 0.04), 0.01, (0.04, 0.025), 0.004
    cad = CadReference(fx.model(fx.plate_with_hole(size, t, c, r, 128), fx.plate_specs(size, t, c, r)))
    result = fx.plate_with_hole(size, t, (c[0] + 2e-5, c[1]), r, 128)
    cmm = VirtualCMM(result, S)
    cache = {}

    def points_of(faces):
        out = []
        for f in sorted(set(faces)):
            if f not in cache:
                cache[f] = cmm.measure(plan(cad, [f], S)).points()
            out.append(cache[f])
        return PointSet.concat(out)

    datums = {lab: __import__("cnctwin.inspect.specs", fromlist=["read_datum"]).read_datum(d) for lab, d in (
        ("A", {"label": "A", "feature": {"face": "plane", "normal": [0, 0, -1], "near_mm": [30, 20, 0]}}),
        ("B", {"label": "B", "feature": {"face": "plane", "normal": [-1, 0, 0], "near_mm": [0, 20, 5]}}),
        ("C", {"label": "C", "feature": {"face": "plane", "normal": [0, -1, 0], "near_mm": [30, 0, 5]}}))}
    ctx = gdt.EvalContext(cad, points_of, S, datums, method="cmm")
    ev = gdt.evaluate(read_tolerance({"id": "pos", "kind": "position", "tolerance_mm": 0.1, "datums": ["A", "B", "C"],
                                      "feature": {"face": "cylinder", "axis": [0, 0, 1], "near_mm": [40, 25, 5]}}), ctx)
    assert ev.measured == pytest.approx(4e-5, abs=2e-7)
    assert ev.record["method"].startswith("cmm:")
