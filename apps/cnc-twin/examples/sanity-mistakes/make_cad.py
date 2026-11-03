"""CAD model of the part the sanity-mistakes program means to make: writes part.step (AP242, mm).

Run: cd apps/cnc-twin && bin/tsafe python examples/sanity-mistakes/make_cad.py

The program is a mistake program; this model is the intended result, so a run (or a reviewer) can
compare against it: a 100 x 80 x 20 plate (stock frame = CAD frame, top at z 20) with
* a slot 50 x 10, 15 deep (x 25..75, y 34..44, the D10 at y 39 from x 30 to 70, ends R5);
* spots D6 x 90 deg at (20, 68) and (80, 68) (3 mm deep);
* D6.8 holes 12 deep (EX-GDR 130 deg point) at (20, 15) and (80, 15), the first
  tapped M8 (modelled as its drill hole);
* a slot 6 wide, 14 deep at x 88, y 20..55 (ends R3).
No tolerances (cad.tolerances is not used by this example).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

HERE = Path(__file__).resolve().parent
TOP = 20.0
FIXED_STAMP = "2026-11-03T00:00:00"


def slot(x0, y0, x1, y1, r, depth):
    """Stadium-shaped slot (straight in x or y) from the top down to depth."""
    z0 = TOP - depth
    h = depth + 1.0
    body = None
    for x, y in ((x0, y0), (x1, y1)):
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z0), gp_Dir(0, 0, 1)), r, h).Shape()
        body = cyl if body is None else BRepAlgoAPI_Fuse(body, cyl).Shape()
    if abs(y1 - y0) < 1e-9:
        box = BRepPrimAPI_MakeBox(gp_Pnt(x0, y0 - r, z0), gp_Pnt(x1, y0 + r, z0 + h)).Shape()
    else:
        box = BRepPrimAPI_MakeBox(gp_Pnt(x0 - r, y0, z0), gp_Pnt(x0 + r, y1, z0 + h)).Shape()
    return BRepAlgoAPI_Fuse(body, box).Shape()


def drill(x, y, d, depth, point_deg):
    r = 0.5 * d
    cone_h = r / math.tan(math.radians(0.5 * point_deg))
    z_full = TOP - depth + cone_h
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z_full), gp_Dir(0, 0, 1)), r, TOP - z_full + 1.0).Shape()
    cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, y, TOP - depth), gp_Dir(0, 0, 1)), 0.0, r, cone_h).Shape()
    return BRepAlgoAPI_Fuse(cyl, cone).Shape()


def build():
    part = BRepPrimAPI_MakeBox(100.0, 80.0, TOP).Shape()
    part = BRepAlgoAPI_Cut(part, slot(30.0, 39.0, 70.0, 39.0, 5.0, 15.0)).Shape()
    part = BRepAlgoAPI_Cut(part, slot(88.0, 20.0, 88.0, 55.0, 3.0, 14.0)).Shape()
    for x in (20.0, 80.0):
        part = BRepAlgoAPI_Cut(part, drill(x, 68.0, 6.0, 3.0, 90.0)).Shape()
        part = BRepAlgoAPI_Cut(part, drill(x, 15.0, 6.8, 12.0, 130.0)).Shape()
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
