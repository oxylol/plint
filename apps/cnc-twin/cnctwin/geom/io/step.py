"""STEP import through OpenCASCADE (OCP 8 / OCCT 8, XCAF document) with B-rep face ids, analytic surfaces
and semantic PMI; and an AP242 writer with PMI (docs/models/geom.md "STEP import").

:func:`read_step` (pure: no global state is changed; meant to run before the simulation, in its own
process via :func:`read_step_isolated` or ``python -m cnctwin.geom.io.step in.step out``):

1. ``STEPCAFControl_Reader`` (names, colours, GD&T) into an XCAF document; shapes arrive in the session
   length unit (``UnitsMethods.GetCasCadeLengthUnit``, mm by default) and are scaled to metres; the file's
   own unit is reported (``STEPControl_Reader::FileUnits``: ``millimetre``, ``INCH`` ...).
2. ``BRepTools::Clean`` then ``BRepMesh_IncrementalMesh`` (sequential, so deterministic) at the chordal
   tolerance (linear deflection) and angular tolerance given (``geom.step_chordal_tolerance_m``,
   ``geom.step_angular_tolerance_rad``). Face ids are the 0-based order of ``TopExp::MapShapes(FACE)``;
   each triangle carries its face id (``Mesh.face_group``), REVERSED faces get flipped winding so normals
   point out of the material, and bit-identical nodes of shared edges are welded (a closed solid gives a
   watertight mesh).
3. Per face: the analytic surface (``BRepAdaptor_Surface``: plane, cylinder, cone, sphere, torus, B-spline,
   Bezier, revolution, extrusion, offset, other) with its parameters, the outward sense of planes and the
   concavity of curved faces from the oriented tessellation, area, bounds, a point on the face, the XCAF
   name; features = adjacent faces on the same analytic surface (``geom.feature_merge_*``).
4. PMI from ``XCAFDoc_DimTolTool``: dimensions (type, nominal, plus/minus, limits, class of tolerance),
   geometric tolerances (type, value, zone, material modifier, modifiers, datum references in order of
   precedence) and datums (label, modifiers, target flag), each mapped to the faces it applies to, then
   converted to the project's ``cad.datums`` / ``cad.tolerances`` form (mm). Items without a project form
   (dimensions on edges, datum targets, oriented/path dimensions ...) are kept in ``pmi_unmapped`` with
   the reason; everything read is in ``pmi_raw`` (SI).

The AP242 PMI subset read is the semantic representation of the CAx-IF recommended practices (2014):
``dimensional_size`` / ``dimensional_location`` with ``plus_minus_tolerance`` or ``limits_and_fits``,
``geometric_tolerance`` subtypes with ``geometric_tolerance_with_datum_reference`` (``datum_system``,
``datum_reference_compartment``) and ``..._with_modifiers``, ``datum`` / ``datum_feature``. OCCT returns the
lower tolerance as a magnitude, so a range entirely above the nominal (+a/+b) reads as +b/-a.
"""

from __future__ import annotations

import io as _io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ...core.errors import InputError
from ..cad import ANALYTIC, CadModel, FaceInfo
from ..mesh import Mesh
from ..selectors import selector_for_face
from ..settings import resolve
from .units import unit_name

# XCAF geometric tolerance type -> project tolerance kind
GT_KINDS = {"Flatness": "flatness", "Position": "position", "Perpendicularity": "perpendicularity",
            "Parallelism": "parallelism", "Angularity": "angularity", "Cylindricity": "cylindricity",
            "CircularityOrRoundness": "circularity", "Straightness": "straightness",
            "ProfileOfSurface": "profile_surface", "ProfileOfLine": "profile_line", "Concentricity": "concentricity",
            "Coaxiality": "concentricity", "Symmetry": "symmetry", "CircularRunout": "runout",
            "TotalRunout": "total_runout"}
SIZE_DIAMETER = ("Size_Diameter", "Size_SphericalDiameter")
SIZE_RADIUS = ("Size_Radius", "Size_SphericalRadius")
ANGULAR = ("Location_Angular", "Size_Angular")


def _enum_name(value) -> str:
    """``XCAFDimTolObjects_GeomToleranceType_Flatness`` -> ``Flatness``."""
    name = getattr(value, "name", str(value)).split(".")[-1]
    for prefix in ("XCAFDimTolObjects_GeomToleranceTypeValue_", "XCAFDimTolObjects_GeomToleranceType_",
                   "XCAFDimTolObjects_DimensionType_", "XCAFDimTolObjects_GeomToleranceMatReqModif_",
                   "XCAFDimTolObjects_GeomToleranceZoneModif_", "XCAFDimTolObjects_GeomToleranceModif_",
                   "XCAFDimTolObjects_DimensionQualifier_", "XCAFDimTolObjects_DimensionModif_",
                   "XCAFDimTolObjects_DatumSingleModif_", "XCAFDimTolObjects_DimensionFormVariance_",
                   "XCAFDimTolObjects_DimensionGrade_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _seq(seq) -> list:
    if seq is None:
        return []
    try:
        return [seq.Value(i) for i in range(1, seq.Length() + 1)]
    except AttributeError:
        return list(seq)


def _find_attr(label, cls):
    from OCP.TDF import TDF_AttributeIterator

    it = TDF_AttributeIterator(label)
    while it.More():
        a = it.Value()
        if isinstance(a, cls):
            return a
        it.Next()
    return None


def _dump(obj) -> dict:
    """Fields of an XCAFDimTolObjects object through DumpJson (the only route to out-parameters such as the
    class of tolerance from Python)."""
    buf = _io.BytesIO()
    obj.DumpJson(buf)
    text = buf.getvalue().decode("utf-8", "replace").strip().rstrip(",")
    try:
        return json.loads("{" + text + "}")
    except json.JSONDecodeError:
        return {}


def _canonical_axis(v) -> list[float]:
    a = np.asarray(v, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = int(np.argmax(np.abs(a)))
    if a[k] < 0.0:
        a = -a
    return [float(x) + 0.0 for x in a]


def _xyz(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)


# ============================================================================ reading

def _read_document(path: Path):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    # InitDocument sets up the XCAF tools on a document Python owns; NewDocument would register a second
    # document in the application session that a Python caller can never release
    app.InitDocument(doc)
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetGDTMode(True)
    reader.SetLayerMode(False)
    reader.SetPropsMode(False)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise InputError(f"STEP reader could not read the file ({_enum_name(status)})", str(path))
    if not reader.Transfer(doc):
        raise InputError("STEP reader could not transfer the file's shapes", str(path))
    return app, doc, reader


def read_step(path, chordal_tolerance_m: float | None = None, angular_tolerance_rad: float | None = None) -> CadModel:
    """CAD model of a STEP file (see the module doc)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
    from OCP.OCP.collections import Sequence_TDF_Label
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS, TopoDS_Compound
    from OCP.UnitsMethods import UnitsMethods
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    path = Path(path)
    if not path.is_file():
        raise InputError("STEP file not found", str(path))
    chord_m = float(resolve(chordal_tolerance_m, "step_chordal_tolerance_m"))
    ang_rad = float(resolve(angular_tolerance_rad, "step_angular_tolerance_rad"))
    if chord_m <= 0.0 or ang_rad <= 0.0:
        raise ValueError("STEP tessellation tolerances must be positive")
    app, doc, reader = _read_document(path)
    try:
        file_units = _file_length_units(reader)
        session_mm = float(UnitsMethods.GetCasCadeLengthUnit_s())
        to_m = session_mm * 1e-3
        stool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        free = Sequence_TDF_Label()
        stool.GetFreeShapes(free)
        roots = _seq(free)
        if not roots:
            raise InputError("STEP file holds no shape", str(path))
        shapes = [stool.GetShape_s(lab) for lab in roots]
        if len(shapes) == 1:
            root = shapes[0]
        else:
            root = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(root)
            for s in shapes:
                builder.Add(root, s)
        BRepTools.Clean_s(root)
        mesher = BRepMesh_IncrementalMesh(root, chord_m / to_m, False, ang_rad, False)
        if not mesher.IsDone():
            raise InputError("OpenCASCADE could not tessellate the shape", str(path))
        fmap = IMap()
        TopExp.MapShapes_s(root, TopAbs_FACE, fmap)
        faces = [TopoDS.Face(fmap.FindKey(i)) for i in range(1, fmap.Extent() + 1)]
        mesh = _tessellation(faces, to_m)
        infos = [_face_info(i, f, mesh, to_m) for i, f in enumerate(faces)]
        names = _face_names(stool, roots, faces)
        for i, name in names.items():
            infos[i].name = name
        features = _features(root, fmap, infos)
        mesh.group_names = {i: (infos[i].name or f"face{i}") for i in range(len(infos))}
        mesh.name = path.stem
        first = file_units[0] if file_units else ""
        scale = STEP_LENGTH_UNITS_M.get(first.lower())
        model = CadModel(mesh, infos, features, unit={"file_unit": unit_name(scale) if scale else first,
                                                      "file_scale_m": scale, "file_unit_names": file_units,
                                                      "session_unit_mm": session_mm},
                         source=str(path), chordal_tolerance_m=chord_m, angular_tolerance_rad=ang_rad,
                         names={str(k): v for k, v in names.items()})
        _read_pmi(doc, stool, fmap, model, to_m)
    finally:
        del reader, doc, app
    return model


# length unit names OCCT reports for a file's shape representations -> metres
STEP_LENGTH_UNITS_M = {"millimetre": 1e-3, "millimeter": 1e-3, "centimetre": 1e-2, "centimeter": 1e-2, "metre": 1.0,
                       "meter": 1.0, "kilometre": 1e3, "micrometre": 1e-6, "micron": 1e-6, "nanometre": 1e-9,
                       "inch": 0.0254, "foot": 0.3048, "feet": 0.3048, "mil": 2.54e-5, "yard": 0.9144,
                       "mile": 1609.344}


def _file_length_units(reader) -> list[str]:
    """Length unit names of the file's shape representations (``STEPControl_Reader::FileUnits``)."""
    from OCP.OCP.collections import Sequence_TCollection_AsciiString as Seq

    lengths, angles, solids = Seq(), Seq(), Seq()
    reader.Reader().FileUnits(lengths, angles, solids)
    return [lengths.Value(i).ToCString() for i in range(1, lengths.Length() + 1)]


def _tessellation(faces, to_m: float) -> Mesh:
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    verts, tris, groups = [], [], []
    offset = 0
    for fid, face in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nodes = np.array([_xyz(tri.Node(j).Transformed(trsf)) for j in range(1, tri.NbNodes() + 1)])
        t = np.array([tri.Triangle(j).Get() for j in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
        if face.Orientation() == TopAbs_REVERSED:
            t = t[:, [0, 2, 1]]
        verts.append(nodes * to_m)
        tris.append(t + offset)
        groups.append(np.full(len(t), fid, np.int64))
        offset += len(nodes)
    if not tris:
        raise InputError("the STEP shape has no faces to tessellate")
    mesh = Mesh(np.concatenate(verts), np.concatenate(tris), np.concatenate(groups))
    return mesh.weld(0.0, drop_degenerate=True)


def _surface(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import (GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder,
                             GeomAbs_OffsetSurface, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_SurfaceOfExtrusion,
                             GeomAbs_SurfaceOfRevolution, GeomAbs_Torus)

    s = BRepAdaptor_Surface(face)
    kind = s.GetType()
    table = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone", GeomAbs_Sphere: "sphere",
             GeomAbs_Torus: "torus", GeomAbs_BSplineSurface: "bspline", GeomAbs_BezierSurface: "bezier",
             GeomAbs_SurfaceOfRevolution: "revolution", GeomAbs_SurfaceOfExtrusion: "extrusion",
             GeomAbs_OffsetSurface: "offset"}
    return s, table.get(kind, "other")


def _face_info(fid: int, face, mesh: Mesh, to_m: float) -> FaceInfo:
    from OCP.TopAbs import TopAbs_REVERSED

    s, kind = _surface(face)
    sel = mesh.face_group == fid
    tris = mesh.vertices[mesh.faces[sel]]
    info = FaceInfo(id=fid, surface=kind, reversed=face.Orientation() == TopAbs_REVERSED, n_triangles=int(sel.sum()))
    if len(tris):
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        info.area_m2 = float(areas.sum())
        info.bounds_m = [tris.reshape(-1, 3).min(axis=0).tolist(), tris.reshape(-1, 3).max(axis=0).tolist()]
        centroids = tris.mean(axis=1)
        big = int(np.argmax(areas))
        x = centroids[big]
    else:
        cross = np.zeros((0, 3))
        centroids = np.zeros((0, 3))
        x = np.zeros(3)
    p: dict = {}
    if kind == "plane":
        pl = s.Plane()
        n = _xyz(pl.Axis().Direction())
        origin = _xyz(pl.Location()) * to_m
        if len(cross) and float(np.sum(cross @ n)) < 0.0:
            n = -n
        p = {"point_m": origin.tolist(), "normal": [float(v) + 0.0 for v in n]}
        x = x - np.dot(x - origin, n) * n
    elif kind == "cylinder":
        c = s.Cylinder()
        ax = c.Axis()
        a0 = _xyz(ax.Location()) * to_m
        u = np.asarray(_canonical_axis(_xyz(ax.Direction())))
        r = float(c.Radius()) * to_m
        p = {"axis_point_m": a0.tolist(), "axis": u.tolist(), "radius_m": r}
        info.concave = _concave_radial(centroids, cross, a0, u)
        x = _project_cylinder(x, a0, u, r)
    elif kind == "cone":
        k = s.Cone()
        apex = _xyz(k.Apex()) * to_m
        ref = _xyz(k.Location()) * to_m
        d = _xyz(k.Axis().Direction())
        semi = float(k.SemiAngle())
        ref_r = float(k.RefRadius()) * to_m
        u = (ref - apex) / np.linalg.norm(ref - apex) if np.linalg.norm(ref - apex) > 0.0 else d * math.copysign(1.0, semi)
        p = {"apex_m": apex.tolist(), "axis": [float(v) for v in u], "semi_angle_rad": abs(semi), "ref_point_m": ref.tolist(),
             "ref_radius_m": ref_r}
        info.concave = _concave_radial(centroids, cross, apex, u)
        x = _project_cone(x, apex, u, abs(semi))
    elif kind == "sphere":
        sp = s.Sphere()
        c0 = _xyz(sp.Location()) * to_m
        r = float(sp.Radius()) * to_m
        p = {"centre_m": c0.tolist(), "radius_m": r}
        if len(cross):
            info.concave = bool(np.sum(np.einsum("ij,ij->i", centroids - c0, cross)) < 0.0)
        v = x - c0
        x = c0 + r * v / np.linalg.norm(v) if np.linalg.norm(v) > 0.0 else x
    elif kind == "torus":
        t = s.Torus()
        ax = t.Axis()
        c0 = _xyz(ax.Location()) * to_m
        u = np.asarray(_canonical_axis(_xyz(ax.Direction())))
        R = float(t.MajorRadius()) * to_m
        r = float(t.MinorRadius()) * to_m
        p = {"centre_m": c0.tolist(), "axis": u.tolist(), "major_radius_m": R, "minor_radius_m": r}
        if len(cross):
            radial = centroids - c0 - np.outer((centroids - c0) @ u, u)
            rn = np.linalg.norm(radial, axis=1, keepdims=True)
            ring = c0 + np.divide(radial, rn, out=np.zeros_like(radial), where=rn > 0) * R
            info.concave = bool(np.sum(np.einsum("ij,ij->i", centroids - ring, cross)) < 0.0)
        x = _project_torus(x, c0, u, R, r)
    elif kind in ("bspline", "bezier"):
        g = s.BSpline() if kind == "bspline" else s.Bezier()
        p = {"u_degree": int(g.UDegree()), "v_degree": int(g.VDegree()), "n_poles_u": int(g.NbUPoles()),
             "n_poles_v": int(g.NbVPoles()), "rational": bool(g.IsURational() or g.IsVRational()),
             "periodic_u": bool(g.IsUPeriodic()), "periodic_v": bool(g.IsVPeriodic())}
    info.params = p
    info.point_m = [float(v) for v in x]
    return info


def _concave_radial(centroids, cross, a0, u) -> bool | None:
    if len(cross) == 0:
        return None
    w = centroids - a0
    radial = w - np.outer(w @ u, u)
    return bool(np.sum(np.einsum("ij,ij->i", radial, cross)) < 0.0)


def _project_cylinder(x, a0, u, r):
    w = x - a0
    h = np.dot(w, u)
    radial = w - h * u
    n = np.linalg.norm(radial)
    return a0 + h * u + (radial / n * r if n > 0 else radial)


def _project_cone(x, apex, u, semi):
    w = x - apex
    h = np.dot(w, u)
    radial = w - h * u
    n = np.linalg.norm(radial)
    if n == 0.0:
        return x
    e = radial / n
    g = math.cos(semi) * u + math.sin(semi) * e        # generator direction in the meridian plane
    s = max(0.0, float(np.dot(w, g)))
    return apex + s * g


def _project_torus(x, c0, u, R, r):
    w = x - c0
    radial = w - np.dot(w, u) * u
    n = np.linalg.norm(radial)
    if n == 0.0:
        return x
    ring = c0 + radial / n * R
    v = x - ring
    m = np.linalg.norm(v)
    return ring + v / m * r if m > 0 else x


def _face_names(stool, roots, faces) -> dict[int, str]:
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_Label

    out: dict[int, str] = {}
    for fid, face in enumerate(faces):
        for root in roots:
            label = TDF_Label()
            if stool.FindSubShape(root, face, label) and not label.IsNull():
                attr = _find_attr(label, TDataStd_Name)
                if attr is not None:
                    text = attr.Get().ToExtString() if hasattr(attr.Get(), "ToExtString") else str(attr.Get())
                    if text:
                        out[fid] = str(text)
                break
    return out


def _same_surface(a: FaceInfo, b: FaceInfo, dist: float, ang: float) -> bool:
    if a.surface != b.surface or a.surface not in ANALYTIC:
        return False
    pa, pb = a.params, b.params
    cos_tol = math.cos(ang)

    def parallel(u, v, signed=False):
        c = float(np.dot(u, v))
        return (c if signed else abs(c)) >= cos_tol

    def line_dist(point, origin, u):
        w = np.asarray(point) - np.asarray(origin)
        return float(np.linalg.norm(w - np.dot(w, u) * np.asarray(u)))

    if a.surface == "plane":
        n = np.asarray(pa["normal"])
        return parallel(n, pb["normal"], True) and abs(float(np.dot(np.asarray(pb["point_m"]) - pa["point_m"], n))) <= dist
    if a.surface == "cylinder":
        return (parallel(pa["axis"], pb["axis"]) and abs(pa["radius_m"] - pb["radius_m"]) <= dist
                and line_dist(pb["axis_point_m"], pa["axis_point_m"], np.asarray(pa["axis"])) <= dist)
    if a.surface == "cone":
        return (np.linalg.norm(np.asarray(pa["apex_m"]) - pb["apex_m"]) <= dist and parallel(pa["axis"], pb["axis"], True)
                and abs(pa["semi_angle_rad"] - pb["semi_angle_rad"]) <= ang)
    if a.surface == "sphere":
        return np.linalg.norm(np.asarray(pa["centre_m"]) - pb["centre_m"]) <= dist and abs(pa["radius_m"] - pb["radius_m"]) <= dist
    return (np.linalg.norm(np.asarray(pa["centre_m"]) - pb["centre_m"]) <= dist and parallel(pa["axis"], pb["axis"])
            and abs(pa["major_radius_m"] - pb["major_radius_m"]) <= dist
            and abs(pa["minor_radius_m"] - pb["minor_radius_m"]) <= dist)


def _features(root, fmap, infos: list[FaceInfo]) -> list[list[int]]:
    from OCP.OCP.collections import IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as EMap
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp

    dist = float(resolve(None, "feature_merge_distance_m"))
    ang = float(resolve(None, "feature_merge_angle_rad"))
    parent = list(range(len(infos)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    emap = EMap()
    TopExp.MapShapesAndAncestors_s(root, TopAbs_EDGE, TopAbs_FACE, emap)
    for e in range(1, emap.Extent() + 1):
        ids = sorted({fmap.FindIndex(f) - 1 for f in emap.FindFromIndex(e)} - {-1})
        for i in ids:
            for j in ids:
                if i < j and _same_surface(infos[i], infos[j], dist, ang):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(len(infos)):
        groups.setdefault(find(i), []).append(i)
    features = [sorted(g) for _, g in sorted(groups.items())]
    for k, g in enumerate(features):
        for i in g:
            infos[i].feature = k
    return features


# ---------------------------------------------------------------------------- PMI

def _ref_faces(dt, stool, fmap, label):
    from OCP.OCP.collections import Sequence_TDF_Label
    from OCP.TopAbs import TopAbs_FACE

    first, second = Sequence_TDF_Label(), Sequence_TDF_Label()
    dt.GetRefShapeLabel_s(label, first, second)
    out, other = [], []
    for seq, dest in ((first, out), (second, other)):
        for lab in _seq(seq):
            shape = stool.GetShape_s(lab)
            if shape.IsNull():
                continue
            if shape.ShapeType() != TopAbs_FACE:
                dest.append({"shape": _enum_name(shape.ShapeType()).replace("TopAbs_", "").lower()})
                continue
            idx = fmap.FindIndex(shape)
            if idx == 0:
                idx = next((i for i in range(1, fmap.Extent() + 1) if fmap.FindKey(i).IsPartner(shape)), 0)
            dest.append(idx - 1 if idx > 0 else {"shape": "face_not_in_model"})
    return out, other


def _faces_only(items) -> list[int]:
    return sorted({i for i in items if isinstance(i, int)})


def _read_pmi(doc, stool, fmap, model: CadModel, to_m: float) -> None:
    from OCP.OCP.collections import Sequence_TDF_Label
    from OCP.XCAFDoc import XCAFDoc_Datum, XCAFDoc_Dimension, XCAFDoc_DocumentTool, XCAFDoc_GeomTolerance

    dt = XCAFDoc_DocumentTool.DimTolTool_s(doc.Main())
    raw: list[dict] = []
    datums_by_name: dict[str, dict] = {}
    labels = Sequence_TDF_Label()
    dt.GetDatumLabels(labels)
    for lab in _seq(labels):
        attr = _find_attr(lab, XCAFDoc_Datum)
        if attr is None or attr.GetObject() is None:
            continue
        obj = attr.GetObject()
        name = obj.GetName().ToCString() if obj.GetName() is not None else ""
        faces, _ = _ref_faces(dt, stool, fmap, lab)
        rec = {"kind": "datum", "name": name, "position": int(obj.GetPosition()), "target": bool(obj.IsDatumTarget()),
               "modifiers": [_enum_name(m) for m in _seq(obj.GetModifiers())], "faces": _faces_only(faces),
               "other_shapes": [f for f in faces if not isinstance(f, int)], "entry": _entry(lab)}
        raw.append(rec)
        datums_by_name.setdefault(name, rec)
    labels = Sequence_TDF_Label()
    dt.GetGeomToleranceLabels(labels)
    for lab in _seq(labels):
        attr = _find_attr(lab, XCAFDoc_GeomTolerance)
        if attr is None or attr.GetObject() is None:
            continue
        obj = attr.GetObject()
        faces, _ = _ref_faces(dt, stool, fmap, lab)
        dl = Sequence_TDF_Label()
        dt.GetDatumWithObjectOfTolerLabels_s(lab, dl)
        refs = []
        for d in _seq(dl):
            da = _find_attr(d, XCAFDoc_Datum)
            if da is not None and da.GetObject() is not None:
                o = da.GetObject()
                refs.append({"name": o.GetName().ToCString() if o.GetName() is not None else "",
                             "position": int(o.GetPosition()), "modifiers": [_enum_name(m) for m in _seq(o.GetModifiers())]})
        refs.sort(key=lambda r: (r["position"], r["name"]))
        raw.append({"kind": "geometric", "type": _enum_name(obj.GetType()), "value_m": float(obj.GetValue()) * to_m,
                    "zone": _enum_name(obj.GetTypeOfValue()), "material": _enum_name(obj.GetMaterialRequirementModifier()),
                    "zone_modifier": _enum_name(obj.GetZoneModifier()),
                    "modifiers": [_enum_name(m) for m in _seq(obj.GetModifiers())], "datums": refs,
                    "faces": _faces_only(faces), "other_shapes": [f for f in faces if not isinstance(f, int)],
                    "entry": _entry(lab)})
    labels = Sequence_TDF_Label()
    dt.GetDimensionLabels(labels)
    for lab in _seq(labels):
        attr = _find_attr(lab, XCAFDoc_Dimension)
        if attr is None or attr.GetObject() is None:
            continue
        obj = attr.GetObject()
        f1, f2 = _ref_faces(dt, stool, fmap, lab)
        kind = _enum_name(obj.GetType())
        angular = kind in ANGULAR
        scale = 1.0 if angular else to_m
        rec = {"kind": "dimension", "type": kind, "value": float(obj.GetValue()) * scale,
               "unit": "rad" if angular else "m", "plus_minus": bool(obj.IsDimWithPlusMinusTolerance()),
               "upper": float(obj.GetUpperTolValue()) * scale, "lower": float(obj.GetLowerTolValue()) * scale,
               "range": [float(obj.GetLowerBound()) * scale, float(obj.GetUpperBound()) * scale]
               if obj.IsDimWithRange() else None,
               "qualifier": _enum_name(obj.GetQualifier()) if obj.HasQualifier() else "",
               "modifiers": [_enum_name(m) for m in _seq(obj.GetModifiers())], "fit": None,
               "faces": _faces_only(f1), "faces2": _faces_only(f2),
               "other_shapes": [f for f in f1 + f2 if not isinstance(f, int)], "entry": _entry(lab)}
        if obj.IsDimWithClassOfTolerance():
            rec["fit"] = _fit(obj)
        raw.append(rec)
    model.pmi_raw = raw
    _map_pmi(model)


def _entry(label) -> str:
    from OCP.TCollection import TCollection_AsciiString
    from OCP.TDF import TDF_Tool

    s = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, s)
    return s.ToCString()


def _fit(obj) -> str | None:
    """ISO 286 fit designation (``H7``, ``g6``) of a dimension's class of tolerance, read through DumpJson."""
    import OCP.XCAFDimTolObjects as X

    fields = _dump(obj)
    if "FormVariance" not in fields or "Grade" not in fields:
        return None
    variance = {int(v): _enum_name(k) for k, v in X.XCAFDimTolObjects_DimensionFormVariance.__members__.items()}
    grade = {int(v): _enum_name(k) for k, v in X.XCAFDimTolObjects_DimensionGrade.__members__.items()}
    letter = variance.get(int(fields["FormVariance"]), "")
    it = grade.get(int(fields["Grade"]), "")
    if not letter or letter == "None" or not it.startswith("IT"):
        return None
    hole = bool(int(fields.get("IsHole", 1)))
    return (letter.upper() if hole else letter.lower()) + it[2:]


def _r9(v: float) -> float:
    return float(round(float(v), 9)) + 0.0


def _map_pmi(model: CadModel) -> None:
    """``pmi_raw`` -> project-form ``pmi`` (mm) and ``pmi_unmapped``."""
    datums, tolerances, unmapped = [], [], []
    seen_labels = set()
    for k, rec in enumerate(model.pmi_raw):
        if rec["kind"] == "datum":
            if rec["target"]:
                unmapped.append({"index": k, "item": rec, "reason": "datum targets have no project form"})
                continue
            if not rec["faces"] or not rec["name"]:
                unmapped.append({"index": k, "item": rec, "reason": "datum without a named face"})
                continue
            if rec["name"] in seen_labels:
                continue
            seen_labels.add(rec["name"])
            datums.append({"label": rec["name"], "feature": selector_for_face(model, rec["faces"][0])})
        elif rec["kind"] == "geometric":
            kind = GT_KINDS.get(rec["type"])
            if kind is None or not rec["faces"]:
                unmapped.append({"index": k, "item": rec, "reason": f"geometric tolerance {rec['type']} "
                                                                    f"{'has no project kind' if kind is None else 'on no face'}"})
                continue
            entry = {"id": f"pmi-gt-{k}", "kind": kind, "feature": selector_for_face(model, rec["faces"][0]),
                     "tolerance_mm": _r9(rec["value_m"] * 1e3)}
            if rec["datums"]:
                entry["datums"] = [d["name"] for d in rec["datums"] if d["name"]]
            if rec["material"] in ("M", "L"):
                entry["modifier"] = rec["material"]
            tolerances.append(entry)
        else:
            entry = _map_dimension(model, k, rec)
            if isinstance(entry, str):
                unmapped.append({"index": k, "item": rec, "reason": entry})
            else:
                tolerances.append(entry)
    model.pmi = {"datums": datums, "tolerances": tolerances}
    model.pmi_unmapped = unmapped


def _limits(rec, to_unit: float):
    """(nominal, plus, minus) in output units; plus/minus are magnitudes (>= 0)."""
    if rec["range"] is not None:
        lo, hi = rec["range"]
        nominal = 0.5 * (lo + hi)
        return nominal * to_unit, 0.5 * (hi - lo) * to_unit, 0.5 * (hi - lo) * to_unit
    nominal = rec["value"] * to_unit
    if rec["plus_minus"]:
        return nominal, abs(rec["upper"]) * to_unit, abs(rec["lower"]) * to_unit
    return nominal, None, None


def _map_dimension(model: CadModel, k: int, rec: dict):
    t = rec["type"]
    if rec["other_shapes"] and not rec["faces"]:
        return f"dimension {t} refers to {', '.join(s['shape'] for s in rec['other_shapes'])}, not faces"
    if t in SIZE_DIAMETER or t in SIZE_RADIUS:
        if not rec["faces"]:
            return f"dimension {t} on no face"
        nominal, plus, minus = _limits(rec, 1e3)
        entry = {"id": f"pmi-dim-{k}", "kind": "diameter" if t in SIZE_DIAMETER else "radius",
                 "feature": selector_for_face(model, rec["faces"][0]), "nominal_mm": _r9(nominal)}
        if plus is not None:
            entry["plus_mm"] = _r9(plus)
            entry["minus_mm"] = _r9(minus)
        if rec["fit"] and t in SIZE_DIAMETER:
            entry["diameter_mm"] = {"nominal": _r9(nominal), "fit": rec["fit"]}
        return entry
    if t.startswith("Location_LinearDistance") or t in ("Location_CurvedDistance", "Size_Thickness") or t in ANGULAR:
        if not rec["faces"] or not rec["faces2"]:
            return f"dimension {t} needs two faces (has {len(rec['faces'])} and {len(rec['faces2'])})"
        angular = t in ANGULAR
        nominal, plus, minus = _limits(rec, 180.0 / math.pi if angular else 1e3)
        unit = "deg" if angular else "mm"
        entry = {"id": f"pmi-dim-{k}", "kind": "angle" if angular else "distance",
                 "between": [selector_for_face(model, rec["faces"][0]), selector_for_face(model, rec["faces2"][0])],
                 f"nominal_{unit}": _r9(nominal)}
        if plus is not None:
            entry[f"plus_{unit}"] = _r9(plus)
            entry[f"minus_{unit}"] = _r9(minus)
        return entry
    return f"dimension type {t} has no project tolerance kind"


# ============================================================================ isolated (subprocess) reading

def read_step_isolated(path, chordal_tolerance_m: float | None = None, angular_tolerance_rad: float | None = None,
                       timeout_s: float = 900.0, workdir=None) -> CadModel:
    """:func:`read_step` in a child process (OpenCASCADE's memory is returned to the system when it exits);
    the model comes back through :meth:`CadModel.save` files in ``workdir`` (a temporary directory if None)."""
    from ...paths import ROOT

    chord = float(resolve(chordal_tolerance_m, "step_chordal_tolerance_m"))
    ang = float(resolve(angular_tolerance_rad, "step_angular_tolerance_rad"))
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        out = Path(tmp) / "model"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        cmd = [sys.executable, "-m", "cnctwin.geom.io.step", str(Path(path).resolve()), str(out),
               "--chordal-m", repr(chord), "--angular-rad", repr(ang)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(ROOT))
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            raise InputError("STEP import failed: " + " | ".join(tail), str(path))
        return CadModel.load(out)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m cnctwin.geom.io.step", description="Import a STEP file into a "
                                     "cnctwin CAD model (<out>.npz + <out>.json)")
    parser.add_argument("step")
    parser.add_argument("out")
    parser.add_argument("--chordal-m", type=float, default=None)
    parser.add_argument("--angular-rad", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        model = read_step(args.step, args.chordal_m, args.angular_rad)
    except InputError as error:
        print(str(error), file=sys.stderr)
        return 2
    model.save(args.out)
    print(json.dumps(model.summary(), sort_keys=True))
    return 0


# ============================================================================ writing (test parts, round trips)

def box_with_hole(size_mm=(100.0, 80.0, 30.0), hole_xy_mm=(80.0, 60.0), hole_diameter_mm=8.0):
    """OpenCASCADE solid of a box (corner at the origin) with a through hole along z; returns
    ``(shape, faces)`` with the face ids (MapShapes order) of ``bottom top xmin xmax ymin ymax hole``."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    sx, sy, sz = (float(v) for v in size_mm)
    box = BRepPrimAPI_MakeBox(sx, sy, sz).Shape()
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(float(hole_xy_mm[0]), float(hole_xy_mm[1]), -1.0), gp_Dir(0, 0, 1)),
                                   0.5 * float(hole_diameter_mm), sz + 2.0).Shape()
    shape = BRepAlgoAPI_Cut(box, cyl).Shape()
    fmap = IMap()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
    names = {}
    for i in range(1, fmap.Extent() + 1):
        a = BRepAdaptor_Surface(TopoDS.Face(fmap.FindKey(i)))
        if a.GetType() == GeomAbs_Cylinder:
            names["hole"] = i - 1
        elif a.GetType() == GeomAbs_Plane:
            n = _xyz(a.Plane().Axis().Direction())
            o = _xyz(a.Plane().Location())
            k = int(np.argmax(np.abs(n)))
            level = float(np.dot(o, n) / n[k])
            axis = "xyz"[k]
            low_side = level < 0.5 * (sx, sy, sz)[k]
            names[{"x": ("xmin", "xmax"), "y": ("ymin", "ymax"), "z": ("bottom", "top")}[axis][0 if low_side else 1]] = i - 1
    return shape, names


def write_step_pmi(path, shape, datums=(), geometric=(), dimensions=(), unit: str = "mm", name: str = "part") -> None:
    """AP242 file of ``shape`` with semantic PMI through XCAF (OCCT's GD&T writer). Face ids index
    ``TopExp::MapShapes(shape, FACE)``; values in mm (deg for angles):

    * datums ``{label, faces: [ids]}`` (precedence = list order);
    * geometric ``{kind: flatness | position | perpendicularity | ..., faces, tolerance_mm, datums: [labels],
      zone: diameter | None, material: M | L | None}``;
    * dimensions ``{kind: diameter | radius | distance | angle, faces, faces2 (distance/angle), nominal_mm or
      nominal_deg, plus_mm, minus_mm (magnitudes), fit: "H7" (diameter/radius class of tolerance)}``.
    """
    import OCP.XCAFDimTolObjects as X
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
    from OCP.OCP.collections import Sequence_TDF_Label
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString, TCollection_HAsciiString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_Datum, XCAFDoc_Dimension, XCAFDoc_DocumentTool, XCAFDoc_GeomTolerance

    units = {"mm": ("MM", 1.0), "m": ("M", 1000.0), "inch": ("INCH", 25.4), "in": ("INCH", 25.4), "cm": ("CM", 10.0)}
    if unit not in units:
        raise ValueError(f"STEP unit {unit!r} is not one of {sorted(units)}")
    unit_text, unit_mm = units[unit]
    # OCCT 8's writer converts the geometry to the file unit but writes PMI measures as given, with the
    # file unit attached: lengths are divided here so the file states them in its own unit.
    to_file = 1.0 / unit_mm
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.InitDocument(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    dt = XCAFDoc_DocumentTool.DimTolTool_s(doc.Main())
    lab = st.AddShape(shape, False)
    TDataStd_Name.Set_s(lab, TCollection_ExtendedString(name))
    fmap = IMap()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
    sub = {}

    def face_label(fid):
        if fid not in sub:
            sub[fid] = st.AddSubShape(lab, fmap.FindKey(int(fid) + 1))
        return sub[fid]

    def labels(ids):
        seq = Sequence_TDF_Label()
        for f in ids:
            seq.Append(face_label(f))
        return seq

    datum_labels = {}
    for k, d in enumerate(datums):
        dl = dt.AddDatum()
        attr = XCAFDoc_Datum.Set_s(dl)
        obj = X.XCAFDimTolObjects_DatumObject()
        obj.SetName(TCollection_HAsciiString(str(d["label"])))
        obj.SetPosition(k + 1)
        attr.SetObject(obj)
        dt.SetDatum(labels(d["faces"]), dl)
        datum_labels[str(d["label"])] = dl
    gt_types = {v: k for k, v in GT_KINDS.items() if k != "Coaxiality"}
    for g in geometric:
        gl = dt.AddGeomTolerance()
        attr = XCAFDoc_GeomTolerance.Set_s(gl)
        obj = X.XCAFDimTolObjects_GeomToleranceObject()
        obj.SetType(getattr(X, "XCAFDimTolObjects_GeomToleranceType_" + gt_types[g["kind"]]))
        obj.SetValue(float(g["tolerance_mm"]) * to_file)
        if g.get("zone") == "diameter":
            obj.SetTypeOfValue(X.XCAFDimTolObjects_GeomToleranceTypeValue_Diameter)
        if g.get("material") in ("M", "L"):
            obj.SetMaterialRequirementModifier(getattr(X, "XCAFDimTolObjects_GeomToleranceMatReqModif_" + g["material"]))
        attr.SetObject(obj)
        dt.SetGeomTolerance(labels(g["faces"]), gl)
        for label in g.get("datums", []):
            dt.SetDatumToGeomTol(datum_labels[label], gl)
    dim_types = {"diameter": "Size_Diameter", "radius": "Size_Radius", "distance": "Location_LinearDistance",
                 "angle": "Location_Angular"}
    for d in dimensions:
        ml = dt.AddDimension()
        attr = XCAFDoc_Dimension.Set_s(ml)
        obj = X.XCAFDimTolObjects_DimensionObject()
        obj.SetType(getattr(X, "XCAFDimTolObjects_DimensionType_" + dim_types[d["kind"]]))
        angular = d["kind"] == "angle"
        scale = 1.0 if angular else to_file
        obj.SetValue(float(d["nominal_deg"] if angular else d.get("nominal_mm", 0.0)) * scale)
        plus = d.get("plus_deg") if angular else d.get("plus_mm")
        minus = d.get("minus_deg") if angular else d.get("minus_mm")
        if plus is not None or minus is not None:
            obj.SetUpperTolValue(float(plus or 0.0) * scale)
            obj.SetLowerTolValue(float(minus or 0.0) * scale)  # OCCT takes the lower deviation as a magnitude
        if d.get("fit"):
            fit = str(d["fit"])
            letters = fit.rstrip("0123456789")
            grade = int(fit[len(letters):])
            # OCCT 8's AP242 writer stores the grade as (enum value + 1) in limits_and_fits, while its reader
            # (and ISO 10303-41) take the text as the IT number; IT_n has enum value n + 1, so the enum of
            # value n - 1 makes the file say n. The test of this module checks the written text.
            by_value = {int(v): v for v in X.XCAFDimTolObjects_DimensionGrade.__members__.values()}
            if grade < 1 or grade - 1 not in by_value:
                raise ValueError(f"fit {fit!r}: grades IT1..IT18 can be written")
            obj.SetClassOfTolerance(letters.isupper(), getattr(X, "XCAFDimTolObjects_DimensionFormVariance_" + letters.upper()),
                                    by_value[grade - 1])
        attr.SetObject(obj)
        if d["kind"] in ("distance", "angle"):
            dt.SetDimension(labels(d["faces"]), labels(d["faces2"]), ml)
        else:
            dt.SetDimension(labels(d["faces"]), Sequence_TDF_Label(), ml)
    writer = STEPCAFControl_Writer()
    previous_unit = Interface_Static.CVal_s("write.step.unit")
    previous_schema = Interface_Static.CVal_s("write.step.schema")
    Interface_Static.SetCVal_s("write.step.schema", "AP242DIS")
    Interface_Static.SetCVal_s("write.step.unit", unit_text)
    try:
        writer.SetDimTolMode(True)
        writer.SetNameMode(True)
        if not writer.Transfer(doc, STEPControl_AsIs):
            raise InputError("OpenCASCADE could not transfer the document to STEP", str(path))
        status = writer.Write(str(path))
        if status != IFSelect_RetDone:
            raise InputError(f"OpenCASCADE could not write STEP ({_enum_name(status)})", str(path))
    finally:
        Interface_Static.SetCVal_s("write.step.unit", previous_unit)
        Interface_Static.SetCVal_s("write.step.schema", previous_schema)


if __name__ == "__main__":
    sys.exit(main())
