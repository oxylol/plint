"""CAD of the 4140 shaft (part.step) and of the bar it is turned from (stock.step).

Run: cd apps/cnc-twin && bin/tsafe python examples/lathe-shaft-4140/make_cad.py

Frames: CAD frame = stock frame S. The bar axis is S z through x = y = 0 (removal.stock convention for
cylinders); the bar spans z 0..230, its chucked end at z = 0, the free end (tailstock, program Z0) at
z = 230. Program Z = S z - 230. The finished shaft spans z 50..230 (program Z-180..0).

Profile (program Z, diameters):

| Z from .. to | feature |
|---|---|
| 0 .. -1.449 | chamfer 1.5 x 45 deg to the thread crest |
| -1.449 .. -20 | M24 x 1.5 - 6g thread: crest topped to d 23.898 by the full-profile insert, root d 22.058, right hand |
| -20 .. -23 | thread relief groove 3.0 wide, d 21.6, bottom corners R0.2 (DGM3 insert) |
| -23 .. -60 | d 30 g6 bearing seat, drawn at 29.983 (4 um below the middle of -0.007/-0.020: the expected push-off of the finishing pass), chamfer 0.5 x 45 |
| -60 .. -140 | d 35 h7 hub seat (drawn at 34.988, middle of 0/-0.025), chamfer 0.5 x 45, inner corner R0.4 |
| -140 .. -180 | d 42 collar, chamfer 1 x 45, inner corner R0.4; parted-off face at Z-180 |

The free end carries a DIN 332-1 A3.15 centre hole (60 deg cone, d2 6.7, pilot d 3.15 x 3) for the
live centre; it is in both the bar and the part. Sizes of the fits are drawn at the middle of their
zones because the program aims at the model.

The thread is a real helical groove (trapezoid: flanks 60 deg, crest flat 0.246, root flat 0.192 at
d 22.058; the insert's R0.19 round root leaves at most 0.02 mm in the root corners). If OpenCASCADE
cannot cut it, the script stops (no silent fallback).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, \
    BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCP.GC import GC_MakeSegment2d
from OCP.Geom import Geom_CylindricalSurface
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.gp import gp_Ax1, gp_Ax3, gp_Dir, gp_Pnt, gp_Pnt2d

HERE = Path(__file__).resolve().parent
FIXED_STAMP = "2026-11-03T00:00:00"
BAR_D, BAR_L = 45.0, 230.0
Z0 = 230.0                                      # program Z0 in S
PITCH = 1.5
D_CREST, D_ROOT = 23.898, 22.058                # 6g: d2 22.924 (middle of 22.854..22.994)
CREST_FLAT, ROOT_FLAT = 0.246, 0.192
THREAD_END_Z = -21.5                            # thread ends in the relief groove
CENTRE = {"d2": 6.7, "d1": 3.15, "pilot": 3.0}  # DIN 332-1 form A 3.15


def z(prog_z: float) -> float:
    return Z0 + prog_z


def centre_hole_points():
    """(r, z) of the centre hole from the face inwards (cone, pilot, pilot bottom on the axis)."""
    r2, r1 = CENTRE["d2"] / 2.0, CENTRE["d1"] / 2.0
    cone = (r2 - r1) / math.tan(math.radians(30.0))
    return [(r2, Z0), (r1, Z0 - cone), (r1, Z0 - cone - CENTRE["pilot"]), (0.0, Z0 - cone - CENTRE["pilot"])]


def revolve(points):
    """Solid of revolution about S z of a closed (r, z) polygon in the xz half-plane."""
    poly = BRepBuilderAPI_MakePolygon()
    for r, zz in points:
        poly.Add(gp_Pnt(float(r), 0.0, float(zz)))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
    return BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 2.0 * math.pi).Shape()


def shaft_profile():
    """Closed (r, z) profile of the finished shaft (thread at the crest diameter; the helix is cut later)."""
    rc = D_CREST / 2.0
    pts = [(0.0, z(-180.0)), (21.0, z(-180.0)), (21.0, z(-141.0)), (20.0, z(-140.0)),
           (17.494, z(-140.0)), (17.494, z(-60.5)), (16.994, z(-60.0)),
           (14.9915, z(-60.0)), (14.9915, z(-23.5)), (14.4915, z(-23.0)),
           (10.8, z(-23.0)), (10.8, z(-20.0)), (rc, z(-20.0)),
           (rc, z(-(rc - 10.5))), (10.5, Z0)]
    pts += centre_hole_points()
    return pts


def fillet_edges(shape, picks):
    """Fillet circular edges given by (radius at the edge, z of the edge, fillet radius)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle
    from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    maker = BRepFilletAPI_MakeFillet(shape)
    edges = IMap()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edges)
    found = 0
    for i in range(1, edges.Extent() + 1):
        edge = TopoDS.Edge(edges.FindKey(i))
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Circle:
            continue
        circle = curve.Circle()
        for r_edge, z_edge, r_fillet in picks:
            if abs(circle.Radius() - r_edge) < 1e-6 and abs(circle.Location().Z() - z_edge) < 1e-6:
                maker.Add(r_fillet, edge)
                found += 1
    if found != len(picks):
        raise SystemExit(f"fillet: expected {len(picks)} edges, found {found}")
    maker.Build()
    if not maker.IsDone():
        raise SystemExit("fillet failed")
    return maker.Shape()


def thread_groove():
    """Helical trapezoid groove (the material the threading insert removes), right hand, from 1 mm
    beyond the free end to THREAD_END_Z."""
    rc, rr = D_CREST / 2.0 + 0.02, D_ROOT / 2.0
    w_top = PITCH - CREST_FLAT + 2 * 0.02 * math.tan(math.radians(30.0))
    w_bot = ROOT_FLAT
    z_start, z_end = z(THREAD_END_Z), Z0 + 1.0
    turns = (z_end - z_start) / PITCH
    # spine: helix at the mean radius, on a cylindrical surface (u = angle, v = height)
    r_mid = 0.5 * (rc + rr)
    surface = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), r_mid)
    seg = GC_MakeSegment2d(gp_Pnt2d(0.0, z_start), gp_Pnt2d(2.0 * math.pi * turns, z_end)).Value()
    edge = BRepBuilderAPI_MakeEdge(seg, surface).Edge()
    BRepLib.BuildCurves3d_s(edge, 1e-6)
    spine = BRepBuilderAPI_MakeWire(edge).Wire()
    # profile at angle 0 (the xz plane), trapezoid centred on z_start
    poly = BRepBuilderAPI_MakePolygon()
    for r, dz in ((rr, -w_bot / 2), (rc, -w_top / 2), (rc, w_top / 2), (rr, w_bot / 2)):
        poly.Add(gp_Pnt(r, 0.0, z_start + dz))
    poly.Close()
    pipe = BRepOffsetAPI_MakePipeShell(spine)
    pipe.SetMode(gp_Dir(0, 0, 1))                 # binormal mode: the profile stays in planes through the axis
    pipe.Add(poly.Wire(), False, False)
    pipe.Build()
    if not pipe.IsDone() or not pipe.MakeSolid():
        raise SystemExit("thread sweep failed")
    return pipe.Shape()


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


def main() -> None:
    bar = revolve([(0.0, 0.0), (BAR_D / 2.0, 0.0), (BAR_D / 2.0, BAR_L)] + centre_hole_points())
    write_step(bar, HERE / "stock.step", "bar-4140-d45x230")
    shaft = revolve(shaft_profile())
    # inner corners: R0.4 from the finishing insert nose, groove bottom R0.2 from the DGM3 insert
    shaft = fillet_edges(shaft, [(14.9915, z(-60.0), 0.4), (17.494, z(-140.0), 0.4),
                                 (10.8, z(-23.0), 0.2), (10.8, z(-20.0), 0.2)])
    v_blank = volume(shaft)
    groove = thread_groove()
    threaded = BRepAlgoAPI_Cut(shaft, groove).Shape()
    if not BRepCheck_Analyzer(threaded).IsValid():
        raise SystemExit("threaded shaft is not a valid solid")
    v_part = volume(threaded)
    if not 0.0 < v_blank - v_part < 2000.0:
        raise SystemExit(f"thread cut removed {v_blank - v_part:.1f} mm3: not plausible")
    write_step(threaded, HERE / "part.step", "shaft-4140")
    print(f"bar {volume(bar):.0f} mm3, shaft before the thread {v_blank:.0f} mm3, part {v_part:.0f} mm3 "
          f"(thread groove {v_blank - v_part:.1f} mm3)")


if __name__ == "__main__":
    main()
