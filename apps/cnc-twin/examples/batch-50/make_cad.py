"""CAD of the 6061-T6 bracket of the batch-50 example: writes part.step next to this script.

Run: cd apps/cnc-twin && bin/tsafe python examples/batch-50/make_cad.py

Stock frame S = CAD frame: origin at the front-left-bottom corner of the 60 x 50 x 25 blank (6061-T6
extruded flat bar 50 x 25 sawn to 60), +x along the bar, +y away from the operator, +z up.

The bracket is an L section along x: an upright flange y 0..20 over the full height (z 0..25) and a
foot y 20..50, 8 mm thick (z 0..8), with two through holes d6.8 in the foot at x 15 and x 45, y 37.
The inner corner between the upright face (y = 20) and the foot top (z = 8) is sharp (square end mill).
"""

from __future__ import annotations

import re
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

HERE = Path(__file__).resolve().parent
FIXED_STAMP = "2026-11-03T00:00:00"
BLANK = (60.0, 50.0, 25.0)
UPRIGHT_Y = 20.0
FOOT_Z = 8.0
HOLES = ((15.0, 37.0), (45.0, 37.0))
HOLE_D = 6.8


def bracket():
    sx, sy, sz = BLANK
    body = BRepPrimAPI_MakeBox(sx, sy, sz).Shape()
    step = BRepPrimAPI_MakeBox(gp_Pnt(-1.0, UPRIGHT_Y, FOOT_Z), gp_Pnt(sx + 1.0, sy + 1.0, sz + 1.0)).Shape()
    body = BRepAlgoAPI_Cut(body, step).Shape()
    for x, y in HOLES:
        hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1)), HOLE_D / 2.0, FOOT_Z + 2.0).Shape()
        body = BRepAlgoAPI_Cut(body, hole).Shape()
    return body


def write_step(shape, path: Path) -> None:
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.product.name", "bracket-6061")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise SystemExit(f"could not write {path}")
    text = path.read_text()
    text = re.sub(r"(FILE_NAME\('[^']*',')[^']*(')", rf"\g<1>{FIXED_STAMP}\g<2>", text, count=1)
    path.write_text(text)


def main() -> None:
    write_step(bracket(), HERE / "part.step")
    print(f"wrote {HERE / 'part.step'}")


if __name__ == "__main__":
    main()
