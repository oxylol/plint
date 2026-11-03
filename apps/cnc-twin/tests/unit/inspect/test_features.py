"""Feature recognition from B-rep faces (planes, slots, tabs, pockets) and measurement on the result."""

import pytest

from cnctwin.inspect.cadref import CadReference
from cnctwin.inspect.features import edge_is_concave, measure, recognise
from cnctwin.inspect.sampling import assign
from cnctwin.inspect.schema import defaults
from tests.unit.inspect import inspfix as fx

S = defaults()
LO, HI = [0.0, 0.0, 0.0], [0.040, 0.030, 0.015]
PLO, PHI = [0.012, 0.009], [0.028, 0.021]


@pytest.fixture(scope="module")
def cad():
    return CadReference(fx.model(fx.pocket_block(LO, HI, PLO, PHI, 0.004), fx.pocket_specs(LO, HI, PLO, PHI, 0.004)))


def test_recognises_pocket_slots_and_tabs(cad):
    feats = recognise(cad, S)
    by_kind = {}
    for f in feats:
        by_kind.setdefault(f.kind, []).append(f)
    assert len(by_kind["plane"]) == 11
    pockets = by_kind["pocket"]
    assert len(pockets) == 1
    p = pockets[0]
    assert p.nominal["floor"] == [10] and sorted(p.nominal["walls"]) == [6, 7, 8, 9] and p.nominal["top"] == [5]
    assert p.nominal["depth_m"] == pytest.approx(0.004) and p.nominal["open"] is False
    slots = sorted((f.nominal["width_m"], f.faces) for f in by_kind["slot"])
    assert [round(w * 1e3, 9) for w, _ in slots] == [12.0, 16.0]                       # pocket width and length
    # back-to-back pairs: the walls around the pocket (12, 12 in x; 9, 9 in y) and the floor's 11 mm to the bottom
    tabs = sorted(round(f.nominal["width_m"] * 1e3, 9) for f in by_kind["tab"])
    assert tabs == [9.0, 9.0, 11.0, 12.0, 12.0]
    assert edge_is_concave(cad, 10, 6) is True and edge_is_concave(cad, 5, 0) is False
    assert edge_is_concave(cad, 0, 1) is None


def test_measures_the_machined_pocket(cad):
    # the pocket came out 0.05 mm shallow and 0.02 mm wide in x
    result = fx.subdivide(fx.pocket_block(LO, HI, [PLO[0] - 1e-5, PLO[1]], [PHI[0] + 1e-5, PHI[1]], 0.00395), 2)
    a = assign(result, cad, S)
    feats = recognise(cad, S)

    def points_of(faces):
        return a.face_points(cad, faces, S.max_points_per_face)

    pocket = [f for f in feats if f.kind == "pocket"][0]
    m = measure(pocket, cad, points_of, S)
    assert m["status"] == "measured"
    assert m["depth_deviation_m"] == pytest.approx(-5e-5, abs=1e-12)
    assert m["floor_flatness_m"] == pytest.approx(0.0, abs=1e-12)
    slot16 = [f for f in feats if f.kind == "slot" and abs(f.nominal["width_m"] - 0.016) < 1e-9][0]
    w = measure(slot16, cad, points_of, S)
    assert w["width_deviation_m"] == pytest.approx(2e-5, abs=1e-12)
    assert w["two_point_min_m"] == pytest.approx(0.01602, abs=1e-12)
    floor = [f for f in feats if f.kind == "plane" and f.faces == [10]][0]
    mf = measure(floor, cad, points_of, S)
    assert mf["location_deviation_m"] == pytest.approx(5e-5, abs=1e-12)                 # floor 0.05 mm high
