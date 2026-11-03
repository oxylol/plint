"""CAD model of the drilled and tapped plate with semantic PMI: writes part.step (AP242, mm).

Run: cd apps/cnc-twin && bin/tsafe python examples/drill-tap-plate-1045/make_cad.py

Geometry (stock frame S = CAD frame: origin at the lower-left-bottom corner of the 160 x 80 x 16
plate, +z up, top at z = 16). Threaded holes are modelled as their tap-drill holes, through, with
the 90 deg countersink of the drawing; the thread itself is not modelled (cosmetic, as usual):

* 6 x M8x1.25-6H THRU on a 50 mm bolt circle about (80, 40), first hole at 0 deg: hole 6.8,
  countersink 90 deg x 9.0;
* 4 x M10x1.5-6H THRU at (35, 25), (125, 25), (125, 55), (35, 55): hole 8.5, countersink 90 deg x 11.0.

The plate ends (x < 10 and x > 150) are left free for the strap clamps, so no holder passes over
a clamp.

PMI (cnctwin.geom.io.step.write_step_pmi, OCCT's GD&T writer): datums A (bottom), B (left end
x = 0), C (front y = 0); for every hole a position tolerance of diameter 0.10 to A|B|C and a size
dimension of the drilled (thread minor) diameter with the 6H minor-diameter limits of ISO 965-2
(M8x1.25: 6.647-6.912; M10x1.5: 8.376-8.676). The thread class itself has no AP242 form in the
writer and is given by hand in project.yaml.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from OCP.BRepAdaptor import BRepAdaptor_Surface  # noqa: E402
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut  # noqa: E402
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder  # noqa: E402
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane  # noqa: E402
from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap  # noqa: E402
from OCP.TopAbs import TopAbs_FACE  # noqa: E402
from OCP.TopExp import TopExp  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt  # noqa: E402

from cnctwin.geom.io.step import write_step_pmi  # noqa: E402

PLATE_MM = (160.0, 80.0, 16.0)
M8 = {"drill": 6.8, "csk": 9.0, "minor": (6.647, 6.912)}
M10 = {"drill": 8.5, "csk": 11.0, "minor": (8.376, 8.676)}
BOLT_CIRCLE = {"centre": (80.0, 40.0), "radius": 25.0, "count": 6, "start_deg": 0.0}
M10_XY = [(35.0, 25.0), (125.0, 25.0), (125.0, 55.0), (35.0, 55.0)]
POSITION_TOL_MM = 0.10
FIXED_STAMP = "2026-11-03T00:00:00"


def m8_positions() -> list[tuple[float, float]]:
    cx, cy = BOLT_CIRCLE["centre"]
    r = BOLT_CIRCLE["radius"]
    out = []
    for k in range(BOLT_CIRCLE["count"]):
        a = math.radians(BOLT_CIRCLE["start_deg"] + 360.0 * k / BOLT_CIRCLE["count"])
        out.append((round(cx + r * math.cos(a), 6), round(cy + r * math.sin(a), 6)))
    return out


def holes():
    """(kind, x, y, spec) for every hole, M8 first (bolt-circle order), then M10."""
    return [("M8", x, y, M8) for x, y in m8_positions()] + [("M10", x, y, M10) for x, y in M10_XY]


def build():
    sx, sy, sz = PLATE_MM
    shape = BRepPrimAPI_MakeBox(sx, sy, sz).Shape()
    for _, x, y, spec in holes():
        r = 0.5 * spec["drill"]
        drill = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1)), r, sz + 2.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, drill).Shape()
        r_top = 0.5 * spec["csk"]
        depth = r_top - r                                   # 90 deg countersink: depth = radial step
        # cone from radius r at z = sz - depth up to r_top + 1 at z = sz + 1 (clears the top face)
        cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, y, sz - depth), gp_Dir(0, 0, 1)), r, r_top + 1.0,
                                    depth + 1.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, cone).Shape()
    return shape


def face_ids(shape):
    """Face ids (MapShapes order) of the datum planes and of each hole's cylinder."""
    fmap = IMap()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
    planes, cylinders = {}, {}
    for i in range(1, fmap.Extent() + 1):
        surf = BRepAdaptor_Surface(TopoDS.Face(fmap.FindKey(i)))
        if surf.GetType() == GeomAbs_Plane:
            pln = surf.Plane()
            n = pln.Axis().Direction()
            o = pln.Location()
            if abs(n.Z()) > 0.99 and abs(o.Z()) < 1e-6:
                planes["A"] = i - 1
            elif abs(n.X()) > 0.99 and abs(o.X()) < 1e-6:
                planes["B"] = i - 1
            elif abs(n.Y()) > 0.99 and abs(o.Y()) < 1e-6:
                planes["C"] = i - 1
        elif surf.GetType() == GeomAbs_Cylinder:
            loc = surf.Cylinder().Axis().Location()
            key = (round(loc.X(), 4), round(loc.Y(), 4))
            cylinders.setdefault(key, []).append(i - 1)
    return planes, cylinders


def write(path: Path) -> None:
    shape = build()
    planes, cylinders = face_ids(shape)
    datums = [{"label": k, "faces": [planes[k]]} for k in ("A", "B", "C")]
    geometric, dimensions = [], []
    for _, x, y, spec in holes():
        faces = cylinders[(round(x, 4), round(y, 4))]
        geometric.append({"kind": "position", "faces": faces, "tolerance_mm": POSITION_TOL_MM,
                          "datums": ["A", "B", "C"], "zone": "diameter"})
        lo, hi = spec["minor"]
        dimensions.append({"kind": "diameter", "faces": faces, "nominal_mm": spec["drill"],
                           "plus_mm": round(hi - spec["drill"], 3), "minus_mm": round(spec["drill"] - lo, 3)})
    write_step_pmi(path, shape, datums=datums, geometric=geometric, dimensions=dimensions, unit="mm",
                   name="drill-tap-plate-1045")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(FILE_NAME\('[^']*',')[^']*(')", rf"\g<1>{FIXED_STAMP}\g<2>", text, count=1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    out = HERE / "part.step"
    write(out)
    print(f"wrote {out}")
    for kind, x, y, _ in holes():
        print(f"  {kind} X{x:.3f} Y{y:.3f}")
