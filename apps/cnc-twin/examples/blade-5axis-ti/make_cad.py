"""CAD of the ruled Ti-6Al-4V blade (part.step) and of its pre-machined blank (stock.step).

Run: cd apps/cnc-twin && bin/tsafe python examples/blade-5axis-ti/make_cad.py

Geometry: blade_geom.py. The blade is one ruled B-spline face (ThruSections, ruled) between a section
below the platform (extrapolated along the rulings, so the fuse with the platform block is not coplanar)
and the tip section at z = 20, fused with the platform block 60 x 32 x 20 (z -24..-4). The blank is the same
block with a fin whose sections are the blade sections offset by 2.0 mm in their planes (the previous
3-axis roughing operation), so every flank pass starts from a known allowance.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.OCP.collections import HArray1_gp_Pnt
from OCP.gp import gp_Pnt

import blade_geom as G

HERE = Path(__file__).resolve().parent
FIXED_STAMP = "2026-11-03T00:00:00"
LOFT_BOTTOM_Z = G.ROOT_Z - 0.5         # inside the platform block
STOCK_ALLOWANCE = 2.0


def closed_wire(points_3d):
    arr = HArray1_gp_Pnt(1, len(points_3d))
    for i, (x, y, z) in enumerate(points_3d, start=1):
        arr.SetValue(i, gp_Pnt(float(x), float(y), float(z)))
    interp = GeomAPI_Interpolate(arr, True, 1e-7)
    interp.Perform()
    if not interp.IsDone():
        raise SystemExit("section interpolation failed")
    edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
    return BRepBuilderAPI_MakeWire(edge).Wire()


def sections(allowance=0.0):
    hub, tip = G.section_points(G.HUB), G.section_points(G.TIP)
    if allowance:
        hub, tip = G.offset_closed(hub, allowance), G.offset_closed(tip, allowance)
    bottom = G.ruled_point(hub, tip, LOFT_BOTTOM_Z)
    return (np.column_stack([bottom, np.full(len(bottom), LOFT_BOTTOM_Z)]),
            np.column_stack([tip, np.full(len(tip), G.TIP["z"])]))


def blade_solid(allowance=0.0):
    bottom, top = sections(allowance)
    loft = BRepOffsetAPI_ThruSections(True, True, 1e-6)
    loft.AddWire(closed_wire(bottom))
    loft.AddWire(closed_wire(top))
    loft.CheckCompatibility(False)
    loft.Build()
    if not loft.IsDone():
        raise SystemExit("ruled loft failed")
    return loft.Shape()


def platform():
    (x0, x1), (y0, y1), (z0, z1) = G.BASE["x"], G.BASE["y"], G.BASE["z"]
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x1, y1, z1)).Shape()


def fused(blade):
    shape = BRepAlgoAPI_Fuse(platform(), blade).Shape()
    unify = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    unify.Build()
    shape = unify.Shape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise SystemExit("fused solid is not valid")
    return shape


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
    part = fused(blade_solid())
    stock = fused(blade_solid(STOCK_ALLOWANCE))
    write_step(part, HERE / "part.step", "blade-ti64")
    write_step(stock, HERE / "stock.step", "blade-blank-ti64")
    print(f"part {volume(part):.0f} mm3, stock {volume(stock):.0f} mm3, removed {volume(stock) - volume(part):.0f} mm3")


if __name__ == "__main__":
    main()
