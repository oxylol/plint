"""CAD model of the thin-wall part: writes part.step (AP242, mm) next to this script.

Run: cd apps/cnc-twin && bin/tsafe python examples/thin-wall-al7075/make_cad.py

Geometry (stock frame S = CAD frame: origin at the lower-left-bottom corner of the 90 x 60 x 32
block, +z up, top at z = 32): a 7 mm base (z 0..7) carrying two free-standing walls (fins)
1.0 mm thick, 25 mm tall (z 7..32) and 60 mm long (x 15..75), at y 19.5..20.5 and y 39.5..40.5.
The fin tops are the stock top. Hand tolerances in project.yaml (no PMI).
"""

from __future__ import annotations

import re
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.gp import gp_Pnt

HERE = Path(__file__).resolve().parent
STOCK_MM = (90.0, 60.0, 32.0)
BASE_Z = 7.0
FINS = [{"x": (15.0, 75.0), "y": (19.5, 20.5)}, {"x": (15.0, 75.0), "y": (39.5, 40.5)}]
FIXED_STAMP = "2026-11-03T00:00:00"


def build():
    sx, sy, sz = STOCK_MM
    part = BRepPrimAPI_MakeBox(sx, sy, BASE_Z).Shape()
    for fin in FINS:
        (x0, x1), (y0, y1) = fin["x"], fin["y"]
        box = BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, BASE_Z), gp_Pnt(x1, y1, sz)).Shape()
        part = BRepAlgoAPI_Fuse(part, box).Shape()
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
