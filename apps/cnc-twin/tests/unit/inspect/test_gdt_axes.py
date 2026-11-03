"""Axis datums and characteristics of turned features on a stepped shaft with an eccentric step; symmetry, LMC,
angularity and the thread class on prismatic parts."""

import math

import numpy as np
import pytest

from cnctwin.geom.primitives import revolved_mesh
from cnctwin.inspect import gdt
from cnctwin.inspect.cadref import CadReference
from cnctwin.inspect.features import recognise
from cnctwin.inspect.protocols import SurfaceJson
from cnctwin.inspect.sampling import assign
from cnctwin.inspect.schema import defaults
from cnctwin.inspect.specs import read_datum, read_tolerance
from tests.unit.inspect import inspfix as fx

S = defaults()
E = 1e-5                                         # eccentricity of the small step (10 um)
Z = [0.0, 0.0, 0.03, 0.03, 0.05, 0.05]
R = [0.0, 0.010, 0.010, 0.006, 0.006, 0.0]


def _shaft_specs():
    return [(0, "plane", {"point_m": [0, 0, 0], "normal": [0, 0, -1]}, None),
            (1, "cylinder", {"axis_point_m": [0, 0, 0], "axis": [0, 0, 1], "radius_m": 0.010}, False),
            (2, "plane", {"point_m": [0.008, 0, 0.03], "normal": [0, 0, 1]}, None),
            (3, "cylinder", {"axis_point_m": [0, 0, 0], "axis": [0, 0, 1], "radius_m": 0.006}, False),
            (4, "plane", {"point_m": [0, 0, 0.05], "normal": [0, 0, 1]}, None)]


@pytest.fixture(scope="module")
def shaft():
    cad_mesh = revolved_mesh(Z, R, chordal_tol_m=2e-5)
    cad = CadReference(fx.model(cad_mesh, _shaft_specs()))
    res = fx.subdivide(revolved_mesh(Z, R, chordal_tol_m=2e-5), 2)
    V = res.vertices.copy()
    g = res.face_group
    small = np.zeros(len(V), bool)
    small[np.unique(res.faces[(g == 3) | (g == 4)].reshape(-1))] = True
    V[small, 0] += E                                       # the small step (and its end) moved off the axis
    # every cylinder vertex exactly on its true cylinder (subdivision midpoints lie on chords)
    for grp, centre_x, radius in ((1, 0.0, 0.010), (3, E, 0.006)):
        idx = np.unique(res.faces[g == grp].reshape(-1))
        w = V[idx, :2] - [centre_x, 0.0]
        rho = np.linalg.norm(w, axis=1)
        on = rho > 0.5 * radius
        V[idx[on], :2] = [centre_x, 0.0] + w[on] * (radius / rho[on])[:, None]
    res.vertices = V
    a = assign(res, cad, S)
    datums = {"A": read_datum({"label": "A", "feature": {"face": "cylinder", "radius_mm": 10, "near_mm": [10, 0, 15]}}),
              "B": read_datum({"label": "B", "feature": {"face": "plane", "normal": [0, 0, -1], "near_mm": [5, 0, 0]}})}
    ctx = gdt.EvalContext(cad, lambda faces: a.face_points(cad, faces, S.max_points_per_face), S, datums)
    return ctx


SMALL = {"face": "cylinder", "radius_mm": 6, "near_mm": [6, 0, 40]}
BIG = {"face": "cylinder", "radius_mm": 10, "near_mm": [10, 0, 15]}


def _ev(ctx, **entry):
    return gdt.evaluate(read_tolerance({"id": "t", **entry}), ctx)


def test_coaxiality_and_runout_of_the_eccentric_step(shaft):
    co = _ev(shaft, kind="concentricity", feature=SMALL, tolerance_mm=0.05, datums=["A"])
    assert co.measured == pytest.approx(2 * E, rel=1e-3)
    ro = _ev(shaft, kind="runout", feature=SMALL, tolerance_mm=0.05, datums=["A"])
    assert ro.measured == pytest.approx(2 * E, rel=2e-3)                    # radial runout of an eccentric circle
    tr = _ev(shaft, kind="total_runout", feature=SMALL, tolerance_mm=0.05, datums=["A"])
    assert tr.measured == pytest.approx(2 * E, rel=2e-3)
    own = _ev(shaft, kind="runout", feature=BIG, tolerance_mm=0.05, datums=["A"])
    assert own.measured == pytest.approx(0.0, abs=1e-9)                      # the datum feature itself
    face = _ev(shaft, kind="total_runout", feature={"face": "plane", "normal": [0, 0, 1], "near_mm": [8, 0, 30]},
               tolerance_mm=0.05, datums=["A"])
    assert face.measured == pytest.approx(0.0, abs=1e-9)                     # the shoulder stays square


def test_form_of_the_turned_features(shaft):
    st = _ev(shaft, kind="straightness", feature=BIG, tolerance_mm=0.01)
    assert st.measured == pytest.approx(0.0, abs=1e-9)
    rd = _ev(shaft, kind="circularity", feature=SMALL, tolerance_mm=0.01)
    assert rd.measured == pytest.approx(0.0, abs=1e-9)
    dml = _ev(shaft, kind="straightness", feature=SMALL, tolerance_mm=0.01, modifier="axis")
    assert dml.measured == pytest.approx(0.0, abs=1e-8)
    pl = _ev(shaft, kind="profile_line", feature=BIG, tolerance_mm=0.01)
    assert pl.measured == pytest.approx(0.0, abs=1e-9)
    # in the frame of the big cylinder, the small one's line profiles show the eccentricity: 2 x max |d| = 2 e
    pl_a = _ev(shaft, kind="profile_line", feature=SMALL, tolerance_mm=0.01, datums=["A"])
    assert pl_a.measured == pytest.approx(2 * E, rel=2e-3)
    size = _ev(shaft, kind="diameter", feature=SMALL, diameter_mm={"nominal": 12, "fit": "h6"})
    assert size.measured == pytest.approx(0.012, abs=1e-9) and size.passed is True


SIZE, T, C, RH = (0.06, 0.04), 0.01, (0.04, 0.025), 0.004
HOLE = {"face": "cylinder", "axis": [0, 0, 1], "near_mm": [40, 25, 5]}


@pytest.fixture(scope="module")
def plate():
    cad = CadReference(fx.model(fx.plate_with_hole(SIZE, T, C, RH, 64), fx.plate_specs(SIZE, T, C, RH)))
    a = assign(fx.subdivide(fx.plate_with_hole(SIZE, T, C, RH + 1e-5, 64), 3), cad, S)
    datums = {d["label"]: read_datum(d) for d in (
        {"label": "A", "feature": {"face": "plane", "normal": [0, 0, -1], "near_mm": [30, 20, 0]}},
        {"label": "B", "feature": {"face": "plane", "normal": [-1, 0, 0], "near_mm": [0, 20, 5]}},
        {"label": "C", "feature": {"face": "plane", "normal": [0, -1, 0], "near_mm": [30, 0, 5]}})}
    threads = [{"id": "thread-T5-op0", "kind": "internal", "pitch_nominal_m": 1.25e-3, "pitch_m": 1.25e-3,
                "major_diameter_m": 8.1e-3, "minor_diameter_m": 6.75e-3, "simple_pitch_diameter_m": 7.25e-3,
                "pitch_error_m": 0.0, "flank_half_angles_rad": [math.pi / 6, math.pi / 6],
                "form_angle_nominal_rad": math.pi / 3}]
    return gdt.EvalContext(cad, lambda faces: a.face_points(cad, faces, S.max_points_per_face), S, datums,
                           surface=SurfaceJson({"features": [], "threads": threads}))


def test_lmc_bonus_angularity_and_thread(plate):
    lmc = _ev(plate, kind="position", feature=HOLE, tolerance_mm=0.01, datums=["A", "B", "C"], modifier="L",
              diameter_mm={"nominal": 8, "plus": 0.05, "minus": 0.0})
    # LMC of the hole 8.05; its minimum material envelope (the circumscribed 64-gon circle) 8.02: bonus 30 um
    assert lmc.record["minimum_material_envelope_m"] == pytest.approx(2 * (RH + 1e-5), abs=1e-9)
    assert lmc.record["bonus_m"] == pytest.approx(0.00805 - 2 * (RH + 1e-5), abs=1e-9)
    ang = _ev(plate, kind="angularity", feature={"face": "plane", "normal": [1, 0, 0], "near_mm": [60, 20, 5]},
              tolerance_mm=0.01, datums=["A"])
    assert ang.measured == pytest.approx(0.0, abs=1e-12)
    th = _ev(plate, kind="diameter", feature=HOLE, diameter_mm={"nominal": 8, "fit": "6H"})
    assert th.record["kind"] == "thread" and th.record["designation"] == "M8x1.25-6H"
    assert th.passed is True and th.record["gauges"] == {"GO": True, "NOT GO": True}


def test_symmetry_of_a_slot():
    lo, hi, plo, phi = [0.0, 0.0, 0.0], [0.040, 0.030, 0.015], [0.012, 0.009], [0.028, 0.021]
    cad = CadReference(fx.model(fx.pocket_block(lo, hi, plo, phi, 0.004), fx.pocket_specs(lo, hi, plo, phi, 0.004)))
    shift = 1e-5
    res = fx.subdivide(fx.pocket_block(lo, hi, [plo[0] + shift, plo[1]], [phi[0] + shift, phi[1]], 0.004), 2)
    a = assign(res, cad, S)
    datums = {"B": read_datum({"label": "B", "feature": {"face": "plane", "normal": [-1, 0, 0],
                                                         "near_mm": [0, 15, 7]}})}
    ctx = gdt.EvalContext(cad, lambda faces: a.face_points(cad, faces, S.max_points_per_face), S, datums,
                          recognise(cad, S))
    sym = _ev(ctx, kind="symmetry", feature={"face": "plane", "normal": [1, 0, 0], "near_mm": [12, 15, 13]},
              tolerance_mm=0.05, datums=["B"])
    assert sym.measured == pytest.approx(2 * shift, abs=1e-12)
    pos = _ev(ctx, kind="position", feature={"face": "plane", "normal": [1, 0, 0], "near_mm": [12, 15, 13]},
              tolerance_mm=0.05)
    assert pos.record["width_feature"] and pos.measured == pytest.approx(2 * shift, abs=1e-12)
