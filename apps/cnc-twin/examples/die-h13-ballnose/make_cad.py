"""CAD of the H13 die insert (part.step) and of the hardened blank it is finished from (stock.step).

Run: cd apps/cnc-twin && bin/tsafe python examples/die-h13-ballnose/make_cad.py

Frames: CAD frame = stock frame S, origin at the lower-left-bottom corner of the 100 x 80 x 40 insert, +z up,
top (parting) face at z = 40; the program datum (preset 1) is the centre of the top face, S (50, 40, 40).

Cavity: elliptic paraboloid z = 40 - d + d ((x - 50)^2 / a^2 + (y - 40)^2 / b^2), opening 2a x 2b at the
top face, depth d, convex rim fillet R_RIM between the cavity and the parting face. It is made as a
paraboloid of revolution (B-spline profile through the exact parabola) scaled by b/a in y, cut from the
block, the rim filleted.

Blank: the insert was rough- and semi-machined soft, then hardened to ~50 HRC; the blank is the same block
with the cavity drawn smaller by the machining allowance ALLOWANCE (bottom raised by it, the opening shrunk
so that the wall moves by it along the rim normal) and a rim fillet of R_RIM; the hard-milling program
starts from it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, \
    BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeRevol
from OCP.GC import GC_MakeSegment
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.OCP.collections import HArray1_gp_Pnt
from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.gp import gp_Ax1, gp_Dir, gp_GTrsf, gp_Mat, gp_Pnt, gp_XYZ

HERE = Path(__file__).resolve().parent
FIXED_STAMP = "2026-11-03T00:00:00"
BLOCK = (100.0, 80.0, 40.0)
CENTRE = (50.0, 40.0)
A_HALF, B_HALF, DEPTH = 24.0, 16.0, 9.6            # cavity 48 x 32 x 9.6 (0.8 x the 60 x 40 x 12 brief)
R_RIM = 2.0
ALLOWANCE = 0.3


def paraboloid_tool(a, b, d, top=5.0):
    """Solid that removes the cavity: paraboloid of revolution of radius a and depth d below z = 0,
    a cylinder above up to z = top, scaled by b/a in y; in a frame with the apex on the z axis at -d."""
    rs = np.linspace(0.0, a, 61)
    arr = HArray1_gp_Pnt(1, len(rs))
    for i, r in enumerate(rs, start=1):
        arr.SetValue(i, gp_Pnt(float(r), 0.0, float(-d + d * (r / a) ** 2)))
    interp = GeomAPI_Interpolate(arr, False, 1e-9)
    interp.Perform()
    profile = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
    wire = BRepBuilderAPI_MakeWire()
    wire.Add(profile)
    for p, q in (((a, 0, 0), (a, 0, top)), ((a, 0, top), (0, 0, top)), ((0, 0, top), (0, 0, -d))):
        wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeSegment(gp_Pnt(*p), gp_Pnt(*q)).Value()).Edge())
    face = BRepBuilderAPI_MakeFace(wire.Wire(), True).Face()
    solid = BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 2.0 * math.pi).Shape()
    g = gp_GTrsf()
    g.SetVectorialPart(gp_Mat(1.0, 0.0, 0.0, 0.0, b / a, 0.0, 0.0, 0.0, 1.0))
    g.SetTranslationPart(gp_XYZ(CENTRE[0], CENTRE[1], BLOCK[2]))
    return BRepBuilderAPI_GTransform(solid, g, True).Shape()


def fillet_rim(shape, radius):
    maker = BRepFilletAPI_MakeFillet(shape)
    edges = IMap()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edges)
    found = 0
    for i in range(1, edges.Extent() + 1):
        edge = TopoDS.Edge(edges.FindKey(i))
        curve = BRepAdaptor_Curve(edge)
        p = curve.Value(0.5 * (curve.FirstParameter() + curve.LastParameter()))
        inside = (abs(p.X() - CENTRE[0]) < 0.6 * BLOCK[0] - 10) and (abs(p.Y() - CENTRE[1]) < 0.5 * BLOCK[1] - 5)
        if abs(p.Z() - BLOCK[2]) < 1e-6 and inside:
            maker.Add(radius, edge)
            found += 1
    if found < 1:
        raise SystemExit("rim edge not found")
    maker.Build()
    if not maker.IsDone():
        raise SystemExit("rim fillet failed")
    return maker.Shape()


def insert(a, b, d):
    block = BRepPrimAPI_MakeBox(*BLOCK).Shape()
    body = BRepAlgoAPI_Cut(block, paraboloid_tool(a, b, d)).Shape()
    body = fillet_rim(body, R_RIM)
    if not BRepCheck_Analyzer(body).IsValid():
        raise SystemExit("insert solid is not valid")
    return body


def volume(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def write_step(shape, path: Path, name: str) -> None:
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.product.name", name)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise SystemExit(f"could not write {path}")
    text = path.read_text()
    text = re.sub(r"(FILE_NAME\('[^']*',')[^']*(')", rf"\g<1>{FIXED_STAMP}\g<2>", text, count=1)
    path.write_text(text)


def blank_params():
    """Opening and depth of the blank cavity: bottom raised by the allowance, the rim moved inwards so the
    wall moves by the allowance along its normal at the rim (slopes 2d/a and 2d/b)."""
    sx = math.sin(math.atan(2 * DEPTH / A_HALF))
    sy = math.sin(math.atan(2 * DEPTH / B_HALF))
    return A_HALF - ALLOWANCE / sx, B_HALF - ALLOWANCE / sy, DEPTH - ALLOWANCE


def main() -> None:
    part = insert(A_HALF, B_HALF, DEPTH)
    stock = insert(*blank_params())
    write_step(part, HERE / "part.step", "die-insert-h13")
    write_step(stock, HERE / "stock.step", "die-blank-h13-hardened")
    print(f"part {volume(part):.1f} mm3, blank {volume(stock):.1f} mm3, hard-milled {volume(stock) - volume(part):.1f} mm3; "
          f"blank cavity a {blank_params()[0]:.3f} b {blank_params()[1]:.3f} d {blank_params()[2]:.3f}")


if __name__ == "__main__":
    main()
