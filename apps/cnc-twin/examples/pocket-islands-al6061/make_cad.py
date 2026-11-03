"""CAD model of the pocket-with-islands part: writes part.step (AP242, mm) next to this script.

Run: cd apps/cnc-twin && bin/tsafe python examples/pocket-islands-al6061/make_cad.py

Geometry (stock frame S = CAD frame: origin at the lower-left-bottom corner of the 100 x 80 x 30
block, +z up, the top face at z = 30):

* pocket 84 x 64 (x 8..92, y 8..72), vertical corner radius R8, floor at z = 20 (10 deep);
* island 1: rectangle 18 x 32 (x 24..42, y 24..56), corner radius R4, top at z = 26;
* island 2: obround 18 x 32 (x 58..76, y 24..56, semicircular ends R9), top at z = 26.

Every channel between the walls and the islands is 16 mm wide, so one trochoidal path along the
channel centre lines clears the pocket at constant engagement (make_program.py).
The file is deterministic: the timestamp OpenCASCADE writes into the STEP header is fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.gp import gp_Pnt, gp_Vec

HERE = Path(__file__).resolve().parent
STOCK_MM = (100.0, 80.0, 30.0)
POCKET = {"x": (8.0, 92.0), "y": (8.0, 72.0), "r": 8.0, "floor_z": 20.0}
ISLAND_TOP_Z = 26.0
ISLAND_1 = {"x": (24.0, 42.0), "y": (24.0, 56.0), "r": 4.0}          # rounded rectangle
ISLAND_2 = {"x": (58.0, 76.0), "y": (24.0, 56.0)}                    # obround, ends R9
FIXED_STAMP = "2026-11-03T00:00:00"


def _p(x, y, z):
    return gp_Pnt(float(x), float(y), float(z))


def rounded_rect_wire(x0, x1, y0, y1, r, z):
    """Closed wire of a rectangle with corner radius r (counter-clockwise from the bottom edge)."""
    c = 1.0 - 2.0 ** -0.5                     # corner arc midpoint offset factor (r (1 - 1/sqrt 2))
    wire = BRepBuilderAPI_MakeWire()
    pieces = [
        ("seg", (x0 + r, y0), (x1 - r, y0)),
        ("arc", (x1 - r, y0), (x1 - r * c, y0 + r * c), (x1, y0 + r)),
        ("seg", (x1, y0 + r), (x1, y1 - r)),
        ("arc", (x1, y1 - r), (x1 - r * c, y1 - r * c), (x1 - r, y1)),
        ("seg", (x1 - r, y1), (x0 + r, y1)),
        ("arc", (x0 + r, y1), (x0 + r * c, y1 - r * c), (x0, y1 - r)),
        ("seg", (x0, y1 - r), (x0, y0 + r)),
        ("arc", (x0, y0 + r), (x0 + r * c, y0 + r * c), (x0 + r, y0)),
    ]
    for kind, *pts in pieces:
        if kind == "seg":
            curve = GC_MakeSegment(_p(*pts[0], z), _p(*pts[1], z)).Value()
        else:
            curve = GC_MakeArcOfCircle(_p(*pts[0], z), _p(*pts[1], z), _p(*pts[2], z)).Value()
        wire.Add(BRepBuilderAPI_MakeEdge(curve).Edge())
    return wire.Wire()


def obround_wire(x0, x1, y0, y1, z):
    """Closed wire of a stadium x0..x1 (width) by y0..y1 (length along y) with semicircular ends."""
    r = 0.5 * (x1 - x0)
    xc = 0.5 * (x0 + x1)
    wire = BRepBuilderAPI_MakeWire()
    wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeSegment(_p(x1, y0 + r, z), _p(x1, y1 - r, z)).Value()).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(_p(x1, y1 - r, z), _p(xc, y1, z),
                                                        _p(x0, y1 - r, z)).Value()).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeSegment(_p(x0, y1 - r, z), _p(x0, y0 + r, z)).Value()).Edge())
    wire.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(_p(x0, y0 + r, z), _p(xc, y0, z),
                                                        _p(x1, y0 + r, z)).Value()).Edge())
    return wire.Wire()


def prism(wire, height):
    face = BRepBuilderAPI_MakeFace(wire).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, float(height))).Shape()


def build():
    sx, sy, sz = STOCK_MM
    block = BRepPrimAPI_MakeBox(sx, sy, sz).Shape()
    floor = POCKET["floor_z"]
    cavity = prism(rounded_rect_wire(*POCKET["x"], *POCKET["y"], POCKET["r"], floor), sz - floor + 1.0)
    island_h = ISLAND_TOP_Z - floor
    island_1 = prism(rounded_rect_wire(*ISLAND_1["x"], *ISLAND_1["y"], ISLAND_1["r"], floor), island_h)
    island_2 = prism(obround_wire(*ISLAND_2["x"], *ISLAND_2["y"], floor), island_h)
    part = BRepAlgoAPI_Cut(block, cavity).Shape()
    part = BRepAlgoAPI_Fuse(part, island_1).Shape()
    part = BRepAlgoAPI_Fuse(part, island_2).Shape()
    return part


def write_step(shape, path: Path) -> None:
    writer = STEPControl_Writer()                    # creating the writer resets the write statics
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
