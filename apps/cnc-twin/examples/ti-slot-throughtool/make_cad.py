"""CAD model of the Ti-6Al-4V slot plate: writes part.step (AP214/242 geometry, mm) next to this script.

Run: cd apps/cnc-twin && bin/tsafe python examples/ti-slot-throughtool/make_cad.py

Geometry (CAD frame = stock frame S: origin at the lower-left-bottom corner of the 100 x 70 x 30
plate, +z up, top face at z = 30):

* open slot along x through the whole plate, centred on y = 35;
* slot width 26.025 mm: the CAD model is drawn at the middle of the tolerance 26 +0.05/0, so the
  program (which targets the model) aims at the middle of the zone;
* slot floor at z = 22.0 (depth 8.0 from the top);
* floor corners R0.8 = the nose radius of the AVMT120408 inserts (drawing: "R0.8 max").

The file is deterministic: the timestamp OpenCASCADE writes into the STEP header is fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Line
from OCP.gp import gp_Pnt

HERE = Path(__file__).resolve().parent
PLATE_MM = (100.0, 70.0, 30.0)
SLOT_CENTRE_Y = 35.0
SLOT_WIDTH = 26.025                   # middle of 26.000 +0.050 / 0
SLOT_DEPTH = 8.0
FLOOR_RADIUS = 0.8                    # insert nose radius RE 0.8
FIXED_STAMP = "2026-11-03T00:00:00"


def slot_plate():
    sx, sy, sz = PLATE_MM
    plate = BRepPrimAPI_MakeBox(sx, sy, sz).Shape()
    y0 = SLOT_CENTRE_Y - SLOT_WIDTH / 2.0
    floor = sz - SLOT_DEPTH
    cutter = BRepPrimAPI_MakeBox(gp_Pnt(-1.0, y0, floor), gp_Pnt(sx + 1.0, y0 + SLOT_WIDTH, sz + 1.0)).Shape()
    body = BRepAlgoAPI_Cut(plate, cutter).Shape()
    fillet = BRepFilletAPI_MakeFillet(body)
    found = 0
    edges = IMap()
    TopExp.MapShapes_s(body, TopAbs_EDGE, edges)          # each edge once (an explorer visits it per face)
    for i in range(1, edges.Extent() + 1):
        edge = TopoDS.Edge(edges.FindKey(i))
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() == GeomAbs_Line:
            a = curve.Value(curve.FirstParameter())
            b = curve.Value(curve.LastParameter())
            along_x = abs(a.Y() - b.Y()) < 1e-9 and abs(a.Z() - b.Z()) < 1e-9 and abs(a.X() - b.X()) > 1.0
            at_floor = abs(a.Z() - floor) < 1e-9
            at_wall = min(abs(a.Y() - y0), abs(a.Y() - y0 - SLOT_WIDTH)) < 1e-9
            if along_x and at_floor and at_wall:
                fillet.Add(FLOOR_RADIUS, edge)
                found += 1
    if found != 2:
        raise SystemExit(f"expected 2 slot floor edges, found {found}")
    fillet.Build()
    if not fillet.IsDone():
        raise SystemExit("floor fillet failed")
    return fillet.Shape()


def write_step(shape, path: Path) -> None:
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.product.name", "ti-slot-plate")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise SystemExit(f"could not write {path}")
    text = path.read_text()
    text = re.sub(r"(FILE_NAME\('[^']*',')[^']*(')", rf"\g<1>{FIXED_STAMP}\g<2>", text, count=1)
    path.write_text(text)


def main() -> None:
    shape = slot_plate()
    out = HERE / "part.step"
    write_step(shape, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
