"""One tool instance as the physics sees it: :func:`build_tool` (docs/models/tooling.md).

``build_tool(tool, holder, instance, settings)`` reads a tool item (SIMap or file-key mapping), its
holder and the project's instance data (stickout, runout, tilt, offsets) and returns a
:class:`ToolGeometry`: the nominal :class:`cnctwin.core.abi.EdgeModel` (with the instance's runout
and tilt), its coefficient groups, angle tables and element frames, the revolved envelope, the
thread profile (taps, thread mills, threading inserts), the 2-D insert outline (non-rotating
tools) and the properties other domains read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EdgeModel
from ..core.errors import InputError
from . import holemaking, indexable, mills, turning
from .abi import EdgeFrameT, EdgeTrig, Envelope, GroupTable, InsertProfile, ThreadProfile, empty_envelope
from .built import Built
from .edges import ElementArrays, assemble
from .envelope import SpindleNose, build_envelope, thread_profile
from .props import ToolProps, compute
from .schema import ToolingSettings
from .spec import HolderSpec, InstanceSpec, ToolSpec, read_holder, read_instance, read_tool

BUILDERS = {
    "square_endmill": mills.build_endmill, "ball_endmill": mills.build_endmill,
    "toroidal_endmill": mills.build_endmill, "roughing_endmill": mills.build_endmill,
    "serrated_endmill": mills.build_endmill,
    "chamfer_mill": mills.build_chamfer, "engraver": mills.build_chamfer,
    "keyseat": mills.build_disc, "tslot": mills.build_disc, "slitting_saw": mills.build_disc,
    "dovetail": mills.build_dovetail, "lollipop": mills.build_lollipop, "thread_mill": mills.build_thread_mill,
    "face_mill": indexable.build_indexable_mill, "shoulder_mill": indexable.build_indexable_mill,
    "boring_head": indexable.build_boring_head,
    "spot_drill": holemaking.build_spot_drill, "centre_drill": holemaking.build_centre_drill,
    "twist_drill": holemaking.build_twist_drill, "gun_drill": holemaking.build_gun_drill,
    "indexable_drill": holemaking.build_indexable_drill, "reamer": holemaking.build_reamer,
    "tap_cutting": holemaking.build_tap, "tap_forming": holemaking.build_tap,
    "turning_insert": turning.build_turning, "boring_bar": turning.build_turning,
    "grooving_insert": turning.build_grooving, "parting_insert": turning.build_grooving,
    "threading_insert": turning.build_threading, "form_tool": turning.build_form_tool,
}


@dataclass
class ToolGeometry:
    """Everything tooling knows about one tool slot (static for the run)."""

    slot: int
    number: int
    tool_id: str
    holder_id: str
    type: str
    hand: str
    hand_sign: float
    rotating: bool
    edge: EdgeModel                    # nominal (instance runout and tilt included)
    groups: GroupTable
    trig: EdgeTrig
    frames: EdgeFrameT
    elements: ElementArrays            # tooth-frame vectors (C) for re-phasing the frames
    envelope: Envelope
    layout: dict
    thread: ThreadProfile | None
    insert: InsertProfile | None
    props: ToolProps
    info: dict
    notes: list[str] = field(default_factory=list)
    spec: ToolSpec | None = None
    holder: HolderSpec | None = None
    instance: InstanceSpec | None = None
    runout_tir_m: float = 0.0
    runout_source: str = "none"
    element_tags: tuple = ()           # layout piece tag of every element (station), e.g. "lip", "chisel"
    element_rows: np.ndarray | None = None   # thread row of every element (taps, thread mills), -1 elsewhere


def build_tool(tool, holder=None, instance=None, settings: ToolingSettings | None = None, *, slot: int = 0,
               tool_id: str = "", holder_id: str = "", db=None, spindle_nose: SpindleNose | None = None,
               spindle_runout_tir_m: float | None = None) -> ToolGeometry:
    settings = settings or ToolingSettings()
    spec = read_tool(tool, tool_id)
    hold = read_holder(holder, holder_id) if holder is not None else None
    if hold is not None:
        spec.notes.extend(hold.notes)
    inst = instance if isinstance(instance, InstanceSpec) else read_instance(instance)
    if spec.type == "boring_head" and spec.g("diameter_m") is None and inst.radius_offset_m is not None:
        spec.set_diameter_m = 2.0 * inst.radius_offset_m
        spec.notes.append(f"boring head set diameter from the radius offset: {spec.set_diameter_m * 1e3:.3f} mm")
    builder = BUILDERS.get(spec.type)
    if builder is None:
        raise InputError(f"tool {spec.tool_id}: no edge model for type {spec.type!r}")
    built: Built = builder(spec, settings)
    eb = assemble(built.segs, built.teeth, settings, spec.hand_sign)
    edge = eb.edge
    tir, source = _runout(spec, inst, hold, built.rotating, spindle_runout_tir_m)
    runout, tilt = runout_vectors(tir, inst.runout_phase_rad, inst.tilt_rad, spec.hand_sign)
    if not built.rotating:
        runout, tilt = np.zeros(3), np.zeros(2)
    edge = edge._replace(runout_T_m=runout, tilt_rad=tilt)
    if built.rotating:
        envelope, layout = build_envelope(spec, built, hold, inst.stickout_m, spindle_nose)
    else:
        envelope, layout = empty_envelope(), {"stickout": None, "z_gauge": None}
    thread = built.thread if built.thread is not None else thread_profile(built, edge.tooth_phase_rad)
    props = compute(spec, built.info, inst, hold, envelope if built.rotating else None, layout,
                    edge.tooth_phase_rad, spec.hand_sign, db)
    return ToolGeometry(
        slot=slot, number=inst.number, tool_id=spec.tool_id, holder_id=hold.holder_id if hold else (holder_id or ""),
        type=spec.type, hand=spec.hand, hand_sign=spec.hand_sign, rotating=built.rotating, edge=edge,
        groups=eb.groups, trig=eb.trig, frames=eb.frames, elements=eb.elements, envelope=envelope, layout=layout,
        thread=thread, insert=built.insert, props=props, info=built.info, notes=spec.notes, spec=spec, holder=hold,
        instance=inst, runout_tir_m=tir, runout_source=source,
        element_tags=tuple(eb.layout.segs[int(i)].tag for i in eb.layout.seg_index),
        element_rows=np.array([eb.layout.segs[int(i)].extra.get("row", -1) for i in eb.layout.seg_index], np.int64))


def _runout(spec: ToolSpec, inst: InstanceSpec, holder: HolderSpec | None, rotating: bool,
            spindle_tir: float | None) -> tuple[float, str]:
    """TIR at the cutting edge: measured (project) when given, else the holder's catalogue TIR and the
    spindle taper runout combined as independent errors (root sum of squares)."""
    if not rotating:
        if inst.runout_tir_m:
            spec.notes.append("runout_um has no meaning for a non-rotating tool and is not applied")
        return 0.0, "none"
    if inst.runout_tir_m is not None:
        return inst.runout_tir_m, "measured"
    parts = [v for v in ((holder.runout_tir_m if holder is not None else None), spindle_tir) if v is not None]
    if not parts:
        return 0.0, "none"
    tir = math.sqrt(sum(v * v for v in parts))
    spec.notes.append(f"runout not measured: {tir * 1e6:.1f} um TIR from the holder and spindle data (RSS), "
                      f"phase 0")
    return tir, "catalogue"


def runout_vectors(tir_m: float, phase_rad: float, tilt, hand_sign: float) -> tuple[np.ndarray, np.ndarray]:
    """Static radial runout at the tip (offset e = TIR / 2 toward the runout phase, measured from
    tooth 0 in the cutting sense) and the axis tilt about x_T and y_T. A scalar tilt tilts the axis
    toward the same azimuth (the offset grows with height z by tilt x z)."""
    alpha = hand_sign * phase_rad
    e = 0.5 * tir_m
    runout = np.array([e * math.cos(alpha), e * math.sin(alpha), 0.0])
    if tilt is None:
        tilt_xy = np.zeros(2)
    elif len(tilt) == 1:
        tau = float(tilt[0])
        tilt_xy = np.array([-tau * math.sin(alpha), tau * math.cos(alpha)])
    else:
        tilt_xy = np.array([float(tilt[0]), float(tilt[1])])
    return runout, tilt_xy


def element_positions_T(edge: EdgeModel, theta: float = 0.0, apply_runout: bool = True) -> np.ndarray:
    """Element midpoints in T0 at spindle angle theta (plain NumPy; the kernel version is
    cnctwin.tooling.kernels.edge_points_T0)."""
    h = float(edge.hand_sign[0])
    alpha = theta + h * (edge.tooth_phase_rad[:, None] + edge.elem_lag_rad)
    x = edge.elem_r_m * np.cos(alpha)
    y = edge.elem_r_m * np.sin(alpha)
    z = edge.elem_z_m.copy()
    if apply_runout:
        # offset and tilt are fixed in T: rotate them with the spindle
        c, s = math.cos(theta), math.sin(theta)
        xt = edge.elem_r_m * np.cos(alpha - theta)
        yt = edge.elem_r_m * np.sin(alpha - theta)
        ax, ay = float(edge.tilt_rad[0]), float(edge.tilt_rad[1])
        dx_t = ay * z + edge.runout_T_m[0]
        dy_t = -ax * z + edge.runout_T_m[1]
        dz_t = ax * yt - ay * xt
        x = x + c * dx_t - s * dy_t
        y = y + s * dx_t + c * dy_t
        z = z + dz_t
    return np.stack([x, y, z], axis=-1)
