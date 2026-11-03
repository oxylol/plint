"""Characteristics on a plate with a hole: position in a datum frame (with a rigid misplacement of the whole part and
an MMC bonus), size, form, orientation and profile against hand-computed values."""

import math

import numpy as np
import pytest

from cnctwin.inspect import datums as D
from cnctwin.inspect import gdt
from cnctwin.inspect.cadref import CadReference
from cnctwin.inspect.sampling import assign
from cnctwin.inspect.schema import defaults
from cnctwin.inspect.specs import read_datum, read_tolerance
from tests.unit.inspect import inspfix as fx

S = defaults()
SIZE, T, C, R = (0.06, 0.04), 0.01, (0.04, 0.025), 0.004
DX, DY = 2e-5, -1.5e-5                       # the hole is 25 um off its true position
HOLE = {"face": "cylinder", "axis": [0, 0, 1], "near_mm": [40, 25, 5]}
TOP = {"face": "plane", "normal": [0, 0, 1], "near_mm": [10, 10, 10]}
DATUMS = [{"label": "A", "feature": {"face": "plane", "normal": [0, 0, -1], "near_mm": [30, 20, 0]}},
          {"label": "B", "feature": {"face": "plane", "normal": [-1, 0, 0], "near_mm": [0, 20, 5]}},
          {"label": "C", "feature": {"face": "plane", "normal": [0, -1, 0], "near_mm": [30, 0, 5]}}]


@pytest.fixture(scope="module")
def part():
    cad = CadReference(fx.model(fx.plate_with_hole(SIZE, T, C, R, 64), fx.plate_specs(SIZE, T, C, R)))
    result = fx.subdivide(fx.plate_with_hole(SIZE, T, (C[0] + DX, C[1] + DY), R + 1e-5, 64), 3)
    placement = fx.rigid([0.3, -0.2, 1.0], 6e-4, [1e-4, 5e-5, 2e-5])      # the part sits elsewhere in S
    result = result.transformed(placement)
    a = assign(result, cad, S)
    ctx = gdt.EvalContext(cad, lambda faces: a.face_points(cad, faces, S.max_points_per_face), S,
                          {d["label"]: read_datum(d) for d in DATUMS})
    return cad, a, ctx, placement


def _eval(ctx, **entry):
    return gdt.evaluate(read_tolerance({"id": entry.pop("id", "t"), **entry}), ctx)


def test_datum_frame_undoes_the_rigid_placement(part):
    cad, a, ctx, placement = part
    nominal, measured, T_reg = D.drf_pair(cad, ctx.datum_specs, ["A", "B", "C"],
                                          lambda f: a.face_points(cad, f, 4000).points, S)
    np.testing.assert_allclose(T_reg @ placement, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(nominal.origin, [0, 0, 0], atol=1e-15)
    assert not nominal.free_translation.any() and not nominal.free_rotation.any()


def test_position_with_a_datum_frame(part):
    _, _, ctx, _ = part
    ev = _eval(ctx, kind="position", feature=HOLE, tolerance_mm=0.04, datums=["A", "B", "C"], basic_mm=[40, 25])
    assert ev.measured == pytest.approx(2 * math.hypot(DX, DY), abs=1e-9)          # 50 um (diameter zone)
    assert ev.passed is False                                                        # 40 um allowed


def test_position_mmc_bonus(part):
    _, _, ctx, _ = part
    # hole 8.02 mm (radius +10 um), limits 8 +0.05/-0: the envelope (MIC of the 64-gon, apothem) is 8.0152 mm;
    # bonus = envelope - MMC (8.000)
    ev = _eval(ctx, kind="position", feature=HOLE, tolerance_mm=0.04, datums=["A", "B", "C"], basic_mm=[40, 25],
               modifier="M", diameter_mm={"nominal": 8, "plus": 0.05, "minus": 0.0})
    apothem = (R + 1e-5) * math.cos(math.pi / 64)
    assert ev.record["envelope_diameter_m"] == pytest.approx(2 * apothem, abs=1e-9)
    assert ev.record["bonus_m"] == pytest.approx(2 * apothem - 0.008, abs=1e-9)
    assert ev.record["allowed_m"] == pytest.approx(4e-5 + 2 * apothem - 0.008, abs=1e-9)
    assert ev.passed is True


def test_size_two_point_and_envelope(part):
    _, _, ctx, _ = part
    ev = _eval(ctx, kind="diameter", feature=HOLE, diameter_mm={"nominal": 8, "fit": "H7"})
    size = ev.record["size"]
    rr = R + 1e-5
    # every vertex lies on the 64-gon between apothem and corner radius
    assert 2 * rr * math.cos(math.pi / 64) - 1e-9 <= size["two_point_min_m"] <= size["two_point_max_m"] <= 2 * rr + 1e-9
    assert ev.record["lower_m"] == pytest.approx(0.008, abs=1e-12)
    assert ev.record["upper_m"] == pytest.approx(0.008015, abs=1e-12)                # H7 at 6-10 mm: IT7 = 15 um
    assert ev.passed is (size["checks"]["envelope"] and size["checks"]["two_point"] and size["checks"]["least_squares"])


def test_form_orientation_and_profile(part):
    _, _, ctx, _ = part
    flat = _eval(ctx, kind="flatness", feature=TOP, tolerance_mm=0.001)
    assert flat.measured == pytest.approx(0.0, abs=1e-12)
    par = _eval(ctx, kind="parallelism", feature=TOP, tolerance_mm=0.001, datums=["A"])
    assert par.measured == pytest.approx(0.0, abs=1e-12)
    perp = _eval(ctx, kind="perpendicularity", feature=HOLE, tolerance_mm=0.001, datums=["A"])
    # the derived median line of a 64-gon prism: section centres of unevenly sampled polygon sections scatter by a
    # small fraction of the polygon's sagitta (4.8 um)
    assert perp.measured < 0.5e-6
    # the polygonal hole: cylindricity = corner radius - apothem
    cyl = _eval(ctx, kind="cylindricity", feature=HOLE, tolerance_mm=0.01)
    rr = R + 1e-5
    assert cyl.measured == pytest.approx(rr * (1 - math.cos(math.pi / 64)), rel=1e-6)
    # profile of the hole in A|B|C: 2 x (offset 25 um + radius 10 um + nothing inward beyond the apothem)
    prof = _eval(ctx, kind="profile_surface", feature=HOLE, tolerance_mm=0.1, datums=["A", "B", "C"])
    assert prof.measured == pytest.approx(2 * (math.hypot(DX, DY) + 1e-5), abs=2e-6)
    # without datums the zone floats: only size and form remain
    free = _eval(ctx, kind="profile_surface", feature=HOLE, tolerance_mm=0.1)
    assert free.measured < prof.measured
    assert free.measured == pytest.approx(2 * 1e-5, abs=2 * rr * (1 - math.cos(math.pi / 64)) + 1e-7)


def test_distance_and_unevaluable_specs(part):
    _, _, ctx, _ = part
    ev = _eval(ctx, kind="distance", between=[TOP, DATUMS[0]["feature"]], nominal_mm=10, plus_mm=0.01, minus_mm=0.01)
    assert ev.measured == pytest.approx(0.01, abs=1e-12)
    with pytest.raises(gdt.EvalError):
        _eval(ctx, kind="flatness", feature={"face": "sphere"}, tolerance_mm=0.01)
    with pytest.raises(gdt.EvalError):
        _eval(ctx, kind="perpendicularity", feature=HOLE, tolerance_mm=0.01)        # no datums
