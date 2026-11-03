"""CAD model of the two-setup part: writes part.step (AP242, mm) next to this script.

Run: cd apps/cnc-twin && bin/tsafe python examples/two-setup-flip/make_cad.py

The model is the finished part in the stock frame S of OP10 (origin at the lower-left-bottom
corner of the 80 x 60 x 25 blank, +z up): OP10 faces the top to z 22.5, OP20 (flipped 180 deg
about X) faces the bottom to z 2.5, so the part is 80 x 60 x 20 between z 2.5 and z 22.5.

* top (OP10): circular pocket D30 x 6 deep at (40, 30); four D6.8 holes at (10, 10), (70, 10),
  (70, 50), (10, 50) with a 90 deg spot countersink to D7.8; 0.3 x 45 deg edge break round the top;
* bottom (OP20): counterbores D11 x 6.4 (DIN 974 for M6 cap screws) coaxial with the holes;
  0.3 x 45 deg edge break round the bottom.
The side faces are the sawn blank (datums B and C). Hand tolerances in project.yaml (no PMI).
"""

from __future__ import annotations

import re
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Line
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

HERE = Path(__file__).resolve().parent
SIZE_XY = (80.0, 60.0)
Z_BOTTOM, Z_TOP = 2.5, 22.5
POCKET = {"centre": (40.0, 30.0), "diameter": 30.0, "depth": 6.0}
HOLES = [(10.0, 10.0), (70.0, 10.0), (70.0, 50.0), (10.0, 50.0)]
HOLE_D = 6.8
SPOT_D = 7.8
CBORE = {"diameter": 11.0, "depth": 6.4}
EDGE_BREAK = 0.3
FIXED_STAMP = "2026-11-03T00:00:00"


def _perimeter_edges(shape, z):
    """The four straight outer edges of the block at height z."""
    edges = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    sx, sy = SIZE_XY
    while exp.More():
        edge = TopoDS.Edge(exp.Current())
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() == GeomAbs_Line:
            p0 = curve.Value(curve.FirstParameter())
            p1 = curve.Value(curve.LastParameter())
            on_level = abs(p0.Z() - z) < 1e-6 and abs(p1.Z() - z) < 1e-6
            on_rim = all(min(abs(p.X()), abs(p.X() - sx)) < 1e-6 or min(abs(p.Y()), abs(p.Y() - sy)) < 1e-6
                         for p in (p0, p1))
            if on_level and on_rim:
                edges.append(edge)
        exp.Next()
    return edges


def build():
    sx, sy = SIZE_XY
    h = Z_TOP - Z_BOTTOM
    part = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, Z_BOTTOM), gp_Pnt(sx, sy, Z_TOP)).Shape()
    chamfer = BRepFilletAPI_MakeChamfer(part)
    for z in (Z_TOP, Z_BOTTOM):
        for edge in _perimeter_edges(part, z):
            chamfer.Add(EDGE_BREAK, edge)
    part = chamfer.Shape()
    cx, cy = POCKET["centre"]
    pocket = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, Z_TOP - POCKET["depth"]), gp_Dir(0, 0, 1)),
                                      0.5 * POCKET["diameter"], POCKET["depth"] + 1.0).Shape()
    part = BRepAlgoAPI_Cut(part, pocket).Shape()
    for x, y in HOLES:
        hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, Z_BOTTOM - 1.0), gp_Dir(0, 0, 1)), 0.5 * HOLE_D, h + 2.0).Shape()
        part = BRepAlgoAPI_Cut(part, hole).Shape()
        step = 0.5 * (SPOT_D - HOLE_D)                       # 90 deg countersink: depth = radial step
        csk = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, y, Z_TOP - step), gp_Dir(0, 0, 1)), 0.5 * HOLE_D,
                                   0.5 * SPOT_D + 1.0, step + 1.0).Shape()
        part = BRepAlgoAPI_Cut(part, csk).Shape()
        cbore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, Z_BOTTOM - 1.0), gp_Dir(0, 0, 1)),
                                         0.5 * CBORE["diameter"], CBORE["depth"] + 1.0).Shape()
        part = BRepAlgoAPI_Cut(part, cbore).Shape()
    return part


def write_step(shape, path: Path) -> None:
    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.schema", "AP242DIS")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise SystemExit(f"could not write {path}")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(FILE_NAME\('[^']*',')[^']*(')", rf"\g<1>{FIXED_STAMP}\g<2>", text, count=1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    out = HERE / "part.step"
    write_step(build(), out)
    print(f"wrote {out}")
