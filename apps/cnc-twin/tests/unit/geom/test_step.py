"""STEP import through OpenCASCADE: a box with a hole written with AP242 PMI through XCAF (datum, flatness,
position, diameter with a fit, distance), read back: face types, analytic parameters, watertight mesh, PMI
mapped to faces in the project's cad form (validated with the project schema), selectors, units, the
isolated (subprocess) reader and CadModel persistence."""

import math
import re

import numpy as np
import pytest

pytest.importorskip("OCP")

from cnctwin.db.schema import Ctx  # noqa: E402
from cnctwin.geom.cad import CadModel  # noqa: E402
from cnctwin.geom.io import step  # noqa: E402
from cnctwin.geom.selectors import select_faces, selector_for_face  # noqa: E402
from cnctwin.project.schema import FEATURE, TOLERANCE  # noqa: E402

CHORD = 5e-6


def write_part(path, unit="mm"):
    shape, ids = step.box_with_hole()
    step.write_step_pmi(
        path, shape,
        datums=[{"label": "A", "faces": [ids["bottom"]]}, {"label": "B", "faces": [ids["xmin"]]},
                {"label": "C", "faces": [ids["ymin"]]}],
        geometric=[{"kind": "flatness", "faces": [ids["top"]], "tolerance_mm": 0.02},
                   {"kind": "position", "faces": [ids["hole"]], "tolerance_mm": 0.05, "datums": ["A", "B", "C"],
                    "zone": "diameter", "material": "M"}],
        dimensions=[{"kind": "diameter", "faces": [ids["hole"]], "nominal_mm": 8.0, "plus_mm": 0.015, "minus_mm": 0.0,
                     "fit": "H7"},
                    {"kind": "distance", "faces": [ids["bottom"]], "faces2": [ids["top"]], "nominal_mm": 30.0,
                     "plus_mm": 0.05, "minus_mm": 0.03}],
        unit=unit)
    return ids


@pytest.fixture(scope="module")
def part(tmp_path_factory):
    path = tmp_path_factory.mktemp("step") / "box_hole.step"
    ids = write_part(path)
    return path, ids, step.read_step(path, chordal_tolerance_m=CHORD)


def test_file_is_ap242_with_semantic_pmi(part):
    path, _, _ = part
    text = path.read_text()
    assert "AP242" in text.upper() or "MANAGED_MODEL_BASED_3D_ENGINEERING" in text.upper()
    for entity in ("DATUM(", "FLATNESS_TOLERANCE", "POSITION_TOLERANCE", "DATUM_SYSTEM", "DIMENSIONAL_SIZE",
                   "DIMENSIONAL_LOCATION", "PLUS_MINUS_TOLERANCE"):
        assert entity in text, entity
    # the class of tolerance H7 is in the file as limits_and_fits (form 'H', zone 'hole', grade '7')
    assert re.search(r"LIMITS_AND_FITS\('H','hole','7'", text)


def test_faces_types_and_analytic_parameters(part):
    _, ids, model = part
    assert len(model.faces) == 7
    assert sorted(f.surface for f in model.faces) == ["cylinder"] + ["plane"] * 6
    outward = {"bottom": [0, 0, -1], "top": [0, 0, 1], "xmin": [-1, 0, 0], "xmax": [1, 0, 0], "ymin": [0, -1, 0],
               "ymax": [0, 1, 0]}
    for name, normal in outward.items():
        f = model.face(ids[name])
        assert f.surface == "plane" and np.allclose(f.params["normal"], normal)
        assert f.concave is None
    hole = model.face(ids["hole"])
    assert math.isclose(hole.params["radius_m"], 0.004, rel_tol=1e-12)
    assert np.allclose(hole.params["axis"], [0, 0, 1]) and hole.concave is True
    d = np.asarray(hole.params["axis_point_m"]) - [0.08, 0.06, 0.0]
    assert math.hypot(d[0], d[1]) < 1e-12
    assert math.isclose(model.face(ids["top"]).area_m2, 0.1 * 0.08 - math.pi * 0.004 ** 2, rel_tol=1e-3)
    assert model.unit["file_unit"] == "mm" and math.isclose(model.unit["file_scale_m"], 1e-3)


def test_mesh_is_watertight_with_face_ids(part):
    _, ids, model = part
    mesh = model.mesh
    assert mesh.is_watertight()
    assert sorted(set(mesh.face_group.tolist())) == list(range(7))
    exact = 0.1 * 0.08 * 0.03 - math.pi * 0.004 ** 2 * 0.03
    # the inscribed polygon of the hole removes less: the mesh is larger by at most hole wall area x chord
    assert 0.0 <= mesh.volume() - exact <= 2 * math.pi * 0.004 * 0.03 * CHORD * 2
    normals = mesh.face_normals()
    assert np.allclose(normals[mesh.face_group == ids["top"]], [0, 0, 1])
    # the hole wall's normals point toward its axis (into the hole, out of the material)
    sel = mesh.face_group == ids["hole"]
    cen = mesh.triangles()[sel].mean(axis=1)
    radial = cen[:, :2] - [0.08, 0.06]
    assert np.all(np.einsum("ij,ij->i", radial, normals[sel][:, :2]) < 0.0)


def check_feature(feature):
    ctx = Ctx()
    FEATURE.check(feature, (), None, ctx)
    assert not ctx.problems, ctx.problems


def test_pmi_round_trip_in_project_form(part):
    _, ids, model = part
    datums = {d["label"]: d for d in model.pmi["datums"]}
    assert sorted(datums) == ["A", "B", "C"]
    for label, face in (("A", "bottom"), ("B", "xmin"), ("C", "ymin")):
        assert datums[label]["feature"]["id"] == f"face:{ids[face]}"
        check_feature(datums[label]["feature"])
    for entry in model.pmi["tolerances"]:
        ctx = Ctx()
        TOLERANCE.check(entry, (), None, ctx)
        assert not ctx.problems, (entry, ctx.problems)
    by_kind = {t["kind"]: t for t in model.pmi["tolerances"]}
    flat = by_kind["flatness"]
    assert flat["feature"]["id"] == f"face:{ids['top']}" and flat["tolerance_mm"] == 0.02
    pos = by_kind["position"]
    assert pos["feature"]["face"] == "cylinder" and pos["feature"]["id"] == f"face:{ids['hole']}"
    assert pos["tolerance_mm"] == 0.05 and pos["datums"] == ["A", "B", "C"] and pos["modifier"] == "M"
    dia = by_kind["diameter"]
    assert dia["nominal_mm"] == 8.0 and dia["plus_mm"] == 0.015 and dia["minus_mm"] == 0.0
    assert dia["diameter_mm"] == {"nominal": 8.0, "fit": "H7"} and dia["feature"]["radius_mm"] == 4.0
    dist = by_kind["distance"]
    assert dist["nominal_mm"] == 30.0 and dist["plus_mm"] == 0.05 and dist["minus_mm"] == 0.03
    assert [b["id"] for b in dist["between"]] == [f"face:{ids['bottom']}", f"face:{ids['top']}"]
    assert model.pmi_unmapped == []
    raw_pos = next(r for r in model.pmi_raw if r.get("type") == "Position")
    assert raw_pos["zone"] == "Diameter" and [d["name"] for d in raw_pos["datums"]] == ["A", "B", "C"]
    assert math.isclose(raw_pos["value_m"], 5e-5, rel_tol=1e-12)


def test_selectors(part):
    _, ids, model = part
    s = select_faces(model, {"face": "plane", "normal": [0, 0, -1], "near_mm": [50, 40, 0]})
    assert s.ok and s.faces == [ids["bottom"]] and s.distance_m == 0.0
    s = select_faces(model, {"face": "cylinder", "axis": [0, 0, -1], "near_mm": [80, 60, 15]})
    assert s.faces == [ids["hole"]] and math.isclose(s.distance_m, 0.004, rel_tol=1e-3)
    assert select_faces(model, {"face": "cylinder", "radius_mm": 4.0}).faces == [ids["hole"]]
    assert not select_faces(model, {"face": "cylinder", "radius_mm": 5.0}).ok
    assert select_faces(model, {"face": "plane", "id": f"face:{ids['top']}"}).faces == [ids["top"]]
    assert not select_faces(model, {"face": "sphere"}).ok
    # a near point beside the part picks the nearest plane with that normal; a tilted normal within 1 deg still matches
    s = select_faces(model, {"face": "plane", "normal": [0.01, 0, 1], "near_mm": [10, 10, 31]})
    assert s.faces == [ids["top"]] and math.isclose(s.distance_m, 0.001, rel_tol=1e-9)
    assert not select_faces(model, {"face": "plane", "normal": [0.05, 0, 1], "near_mm": [10, 10, 31]}).ok
    # SI view (cnctwin.db.simap) and frame change: the selector written in a frame shifted by +10 mm in x
    from cnctwin.db.simap import SIMap
    from cnctwin.geom import transform as tf

    view = SIMap({"f": {"face": "plane", "normal": [1, 0, 0], "near_mm": [100, 40, 15]}})["f"]
    assert select_faces(model, view).faces == [ids["xmax"]]
    s = select_faces(model, {"face": "plane", "normal": [1, 0, 0], "near_mm": [110, 40, 15]},
                     pose_C_X=tf.translation([-0.01, 0, 0]))
    assert s.faces == [ids["xmax"]] and s.distance_m < 1e-12
    for f in model.faces:
        sel = selector_for_face(model, f.id)
        check_feature(sel)
        assert select_faces(model, {k: v for k, v in sel.items() if k != "id"}).best == f.id


def test_features_merge_split_faces(tmp_path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    a = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 50.0, 40.0, 10.0).Shape()
    b = BRepPrimAPI_MakeBox(gp_Pnt(50, 0, 0), 50.0, 40.0, 10.0).Shape()
    fused = BRepAlgoAPI_Fuse(a, b).Shape()
    step.write_step_pmi(tmp_path / "two.step", fused)
    model = step.read_step(tmp_path / "two.step", chordal_tolerance_m=1e-5)
    tops = [f.id for f in model.faces if f.surface == "plane" and np.allclose(f.params["normal"], [0, 0, 1])]
    assert len(tops) == 2 and model.face(tops[0]).feature == model.face(tops[1]).feature
    s = select_faces(model, {"face": "plane", "normal": [0, 0, 1], "near_mm": [20, 20, 10]})
    assert s.faces == sorted(tops)
    assert model.mesh.is_watertight()


def test_inch_file_units_and_pmi_values(tmp_path):
    path = tmp_path / "inch.step"
    write_part(path, unit="inch")
    assert "INCH" in path.read_text().upper()
    model = step.read_step(path, chordal_tolerance_m=CHORD)
    assert model.unit["file_unit"] == "in" and math.isclose(model.unit["file_scale_m"], 0.0254)
    lo, hi = model.mesh.bounds()
    assert np.allclose(hi - lo, [0.1, 0.08, 0.03], atol=1e-9)
    dia = next(t for t in model.pmi["tolerances"] if t["kind"] == "diameter")
    assert math.isclose(dia["nominal_mm"], 8.0, rel_tol=1e-9) and math.isclose(dia["plus_mm"], 0.015, rel_tol=1e-9)
    flat = next(t for t in model.pmi["tolerances"] if t["kind"] == "flatness")
    assert math.isclose(flat["tolerance_mm"], 0.02, rel_tol=1e-9)


def test_isolated_reader_and_persistence(tmp_path, part):
    path, _, model = part
    iso = step.read_step_isolated(path, chordal_tolerance_m=CHORD)
    assert iso.pmi == model.pmi and len(iso.faces) == len(model.faces)
    assert np.array_equal(iso.mesh.faces, model.mesh.faces) and np.allclose(iso.mesh.vertices, model.mesh.vertices)
    model.save(tmp_path / "m")
    back = CadModel.load(tmp_path / "m.json")
    assert back.pmi == model.pmi and back.faces == model.faces and back.features == model.features
    assert np.array_equal(back.mesh.face_group, model.mesh.face_group)
    with pytest.raises(Exception):
        step.read_step_isolated(tmp_path / "missing.step")


def test_bad_files_are_refused(tmp_path):
    from cnctwin.core.errors import InputError

    (tmp_path / "junk.step").write_text("this is not STEP\n")
    with pytest.raises(InputError):
        step.read_step(tmp_path / "junk.step")
    with pytest.raises(InputError):
        step.read_step(tmp_path / "absent.step")
