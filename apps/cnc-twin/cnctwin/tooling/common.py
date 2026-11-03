"""Pieces every builder shares: edge preparation, faces, flute phases and helices."""

from __future__ import annotations

import math

from ..core.errors import InputError
from . import params as P
from .edges import Face, ToothDef
from .spec import ToolSpec, clearance_primary, num, numlist, sub, text


def edge_prep(spec: ToolSpec, insert: bool = False) -> tuple[float, float, float]:
    """(hone radius, T-land width, T-land angle) from geometry.edge_radius_um and edge_prep.

    Without a stated radius: sharp -> sharp_edge_radius_m, hone/waterfall -> hone (or insert)
    default, chamfer -> chamfer_corner_radius_m; no preparation stated -> the hone default for
    coated or indexable tools, the sharp default for uncoated solid tools.
    """
    geometry = spec.geometry
    prep = sub(geometry, "edge_prep")
    kind = text(prep, "type") if prep is not None else None
    radius = num(geometry, "edge_radius_m")
    width, angle = 0.0, 0.0
    if kind == "chamfer":
        width = num(prep, "width_m", 0.0)
        angle = num(prep, "angle_rad", 0.0)
    if radius is None:
        if kind == "sharp":
            radius = P.pf("sharp_edge_radius_m")
        elif kind == "chamfer":
            radius = P.pf("chamfer_corner_radius_m")
        elif kind in ("hone", "waterfall"):
            radius = P.pf("insert_edge_radius_m") if insert else P.pf("hone_edge_radius_m")
        elif insert:
            radius = P.pf("insert_edge_radius_m")
        elif spec.coating_id not in (None, "none"):
            radius = P.pf("hone_edge_radius_m")
        else:
            radius = P.pf("sharp_edge_radius_m")
        spec.notes.append(f"edge radius not given: {radius * 1e6:.0f} um ({kind or 'no'} preparation default)")
    return float(radius), float(width), float(angle)


def radial_face(spec: ToolSpec, rake: float, clear: float, insert: bool = False, **kw) -> Face:
    radius, width, angle = edge_prep(spec, insert)
    return Face(mode="radial", rake=rake, clear=clear, edge_radius=radius, chamfer_width=width,
                chamfer_angle=angle, **kw)


def rake_clear(spec: ToolSpec, default_rake: float | None = None, default_clear: float | None = None):
    rake = num(spec.geometry, "rake_rad")
    if rake is None:
        if default_rake is None:
            raise InputError(f"tool {spec.tool_id} ({spec.type}): geometry needs rake_deg")
        rake = default_rake
        spec.notes.append(f"rake not given: {math.degrees(rake):.1f} deg")
    clear = clearance_primary(spec.geometry)
    if clear is None:
        if default_clear is None:
            raise InputError(f"tool {spec.tool_id} ({spec.type}): geometry needs clearance_deg")
        clear = default_clear
        spec.notes.append(f"clearance not given: {math.degrees(clear):.1f} deg")
    return float(rake), float(clear)


def flute_count(spec: ToolSpec, default: int | None = None) -> int:
    value = num(spec.geometry, "flutes")
    if value is None:
        if default is None:
            raise InputError(f"tool {spec.tool_id} ({spec.type}): geometry needs flutes")
        spec.notes.append(f"flutes not given: {default}")
        return int(default)
    count = int(round(value))
    if count < 1:
        raise InputError(f"tool {spec.tool_id}: flutes must be at least 1")
    return count


def phases_from_pitch(spec: ToolSpec, count: int, key: str = "pitch_rad") -> list[float]:
    """Cutting-sense phase of each flute: tooth j+1 trails tooth j by pitch[j] (pitch_deg is the
    spacing between consecutive flutes in the order they pass a fixed point)."""
    pitch = numlist(spec.geometry, key)
    if pitch is None:
        return [-2.0 * math.pi * j / count for j in range(count)]
    if len(pitch) != count:
        raise InputError(f"tool {spec.tool_id}: {key[:-4]}_deg has {len(pitch)} entries for {count} flutes")
    if abs(sum(pitch) - 2.0 * math.pi) > 1e-3:
        raise InputError(f"tool {spec.tool_id}: pitch_deg must sum to 360 (sums to {math.degrees(sum(pitch)):.3f})")
    phases, phase = [], 0.0
    for j in range(count):
        phases.append(phase)
        phase -= pitch[j]
    return phases


def helix_angles(spec: ToolSpec, count: int, default: float | None = None) -> list[float]:
    """Helix per flute (positive: same hand as the cut, chips go up the flutes)."""
    values = numlist(spec.geometry, "helix_rad")
    if values is None:
        if default is None:
            raise InputError(f"tool {spec.tool_id} ({spec.type}): geometry needs helix_deg")
        spec.notes.append(f"helix not given: {math.degrees(default):.1f} deg")
        values = [default]
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise InputError(f"tool {spec.tool_id}: helix_deg has {len(values)} entries for {count} flutes")
    return values


def helical_teeth(phases: list[float], helices: list[float], R_helix: float, z0: float = 0.0) -> list[ToothDef]:
    return [ToothDef(phase=ph, params={"tan_beta": math.tan(b), "R_helix": R_helix, "helix_z0": z0})
            for ph, b in zip(phases, helices)]


def core_diameter(spec: ToolSpec, diameter: float, flutes: int) -> float:
    value = num(spec.geometry, "core_diameter_m")
    if value is not None:
        return value
    ratio = P.core_ratio(flutes)
    spec.notes.append(f"core diameter not given: {ratio:.2f} D (core_ratio_by_flutes)")
    return ratio * diameter


def cutting_length(spec: ToolSpec, default: float | None = None) -> float:
    value = num(spec.geometry, "cutting_length_m")
    if value is None:
        if default is None:
            raise InputError(f"tool {spec.tool_id} ({spec.type}): geometry needs cutting_length_mm")
        spec.notes.append(f"cutting length not given: {default * 1e3:.3f} mm")
        return default
    return value


def extra(spec: ToolSpec, *keys):
    """A value under one of several keys (catalogue files may use names beyond the template)."""
    for key in keys:
        value = num(spec.geometry, key)
        if value is not None:
            return value
    return None


__all__ = ["edge_prep", "radial_face", "rake_clear", "flute_count", "phases_from_pitch", "helix_angles",
           "helical_teeth", "core_diameter", "cutting_length", "extra"]
