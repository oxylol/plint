"""What the run loop writes besides the per-step series (docs/models/loop.md "Run directory"): the frames.bin
layout, scene.json and its meshes, the keyframe payloads (in-process workpiece bricks, coarse fields, chips) and
the run.json summary. Pure functions of the run's state and domain objects; :mod:`cnctwin.core.loop` calls them.
"""

from __future__ import annotations

import math

import numpy as np

from .bus import Bus


# =============================================================================== frames.bin

def axis_origins(motion) -> list[float]:
    """Per machine axis, the value written as the axis' origin in frames.bin: for linear axes the values that put
    the spindle nose (gauge line, a tool of length zero) on the stock origin with the rotary axes at home (least
    squares through the nominal chain, linear in the linear axes), 0 for rotary axes."""
    model = motion.model
    q_home = np.array([ax.home for ax in model.axes], dtype=float)
    linear = [a for a, ax in enumerate(model.axes) if not ax.rotary and a in set(motion.planners[0].nc_axes)]
    origins = np.zeros(len(model.axes))
    if not linear:
        return origins.tolist()
    from ..motion.abi import FK_S_T0
    kin = motion.kin[0]

    def tip(q):
        return kin.fk(q)[FK_S_T0][:3, 3]

    t0 = tip(q_home)
    step = 1e-3
    J = np.zeros((3, len(linear)))
    for j, a in enumerate(linear):
        q = q_home.copy()
        q[a] += step
        J[:, j] = (tip(q) - t0) / step
    dq, *_ = np.linalg.lstsq(J, -t0, rcond=None)
    for j, a in enumerate(linear):
        origins[a] = q_home[a] + float(dq[j])
    return origins.tolist()


def frames_layout(motion) -> tuple[list[dict], list[dict]]:
    """(axes, paths) of frames.bin: every machine axis of the model with its kind and origin, every path with the
    spindle its tool turns on."""
    model = motion.model
    origins = axis_origins(motion)
    axes = [{"name": ax.name, "kind": "rotary" if ax.rotary else "linear", "origin": float(origins[a])}
            for a, ax in enumerate(model.axes)]
    paths = [{"path": p, "spindle": str(getattr(model.paths[p], "tool_spindle", None) or "main")}
             for p in range(len(model.paths))]
    return axes, paths


def pose_M_S(motion, state) -> np.ndarray:
    """Pose of the stock frame S in M at the end of the last step (path 0): pose_M_T @ inv(pose_S_T)."""
    for bus in (state.bus_next, state.bus_prev):
        paths = list(getattr(bus.motion, "paths", []) or [])
        if paths and len(paths[0].tick):
            f = paths[0]
            return np.asarray(f.pose_M_T, float) @ np.linalg.inv(np.asarray(f.pose_S_T, float))
    from ..motion.abi import FK_M_T0, FK_S_T0
    q = np.array([ax.home for ax in motion.model.axes], dtype=float)
    poses = motion.kin[0].fk(q)
    return poses[FK_M_T0] @ np.linalg.inv(poses[FK_S_T0])


# =============================================================================== scene.json

def _tool_mesh(state, slot: int):
    """The whole tool envelope (cutting part, holder) of a slot as a revolved mesh in T (origin at the tip, +z
    toward the spindle), from removal's profile of the tool."""
    from ..geom.primitives import revolved_mesh
    from ..removal.profile import profile_polyline
    shape = (getattr(state.removal, "tools", {}) or {}).get(int(slot))
    if shape is None or not getattr(shape, "rotating", True):
        return None
    z, r = profile_polyline(shape.whole, 2.0e-5)          # closed to the axis at both ends
    z = np.asarray(z, float)
    r = np.asarray(r, float)
    if len(z) < 3 or float(np.max(r)) <= 0.0:
        return None
    return revolved_mesh(z, r, closed=False, chordal_tol_m=2.0e-5, name=f"tool_{slot}")


def box_mesh(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """(vertices [8, 3], faces [12, 3]) of the axis-aligned box [lo, hi], faces counter-clockwise from outside."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int64)
    return V, F


def write_scene(run) -> None:
    """scene.json + scene/*.glb: the stock as set up (S), the fixture bodies of every operation (S), every rotating
    tool's envelope (T), gravity, the frames.bin header and the keyframe brick layout (architecture 16)."""
    from ..holding.domain import gravity_M
    from ..removal import mesh as RM
    from ..removal.collision import fixture_meshes
    from .rundir import SceneWriter
    ctx, state = run.ctx, run.state
    project = ctx.project
    writer = SceneWriter(ctx.run_dir)
    wp = state.removal.workpieces[0]
    stock_rel = None
    size = [float(v) for v in (getattr(project.stock, "size_m", None) or [])]
    if str(getattr(project.stock, "kind", "") or "") == "block" and len(size) == 3:
        # the block as removal builds it (S from 0 to its size): exact in 12 triangles, where the field's own
        # mesh at a fine lattice is millions (0.075 mm on 100 x 80 x 30 mm: 9.5 M triangles, a 1.8 GB process)
        V, F = box_mesh(np.zeros(3), np.asarray(size, float))
    else:
        stock = RM.mesh_level(wp.field)
        V, F = np.asarray(stock.vertices), np.asarray(stock.faces)
    if len(F):
        stock_rel = writer.add_mesh("stock", (V, F), color_rgba=(0.72, 0.74, 0.76, 1.0), quantize=True)
    ops = list(project.operations)
    fixture = []
    for i, op in enumerate(ops):
        for k, (name, V, F) in enumerate(fixture_meshes(project, i)):
            rel = writer.add_mesh(f"fixture_{i}_{k}", (V, F), color_rgba=(0.35, 0.37, 0.42, 1.0))
            fixture.append({"id": f"{op.name}/{name}", "mesh": rel, "frame": "S", "pose": np.eye(4).tolist(),
                            "kind": "fixture", "op": str(op.name)})
    tools = []
    for slot in state.tools.slots:
        geo = state.tooling.geometry(slot.slot)
        entry = {"number": int(slot.number), "slot": int(slot.slot), "tool_id": slot.tool_id,
                 "diameter_m": float(geo.props.diameter_m), "length_m": float(geo.layout.get("stickout") or 0.0),
                 "teeth": int(geo.props.flutes), "rotating": bool(geo.rotating)}
        mesh = _tool_mesh(state, slot.slot)
        if mesh is not None:
            entry["envelope"] = writer.add_mesh(f"tool_T{slot.number}_{slot.slot}", mesh,
                                                color_rgba=(0.82, 0.62, 0.25, 1.0))
        tools.append(entry)
    op0 = ops[0]
    document = {"stock": {"mesh": stock_rel, "kind": str(project.stock.kind),
                          "pose_W_S": np.asarray(op0.stock_pose_W_S, float).tolist(),
                          "pose_M_S": pose_M_S(run.motion, state).tolist()},
                "machine": {"id": str(getattr(project.machine, "id", "") or ""),
                            "type": str(getattr(project.machine, "type", "") or ""),
                            "axes": [{"name": ax.name, "rotary": bool(ax.rotary)} for ax in run.motion.model.axes],
                            "components": []},
                "fixture": fixture, "tools": tools, "nozzles": [],
                "gravity_M": np.asarray(gravity_M(project.machine), float).tolist(),
                "operations": [{"name": str(op.name), "stock_pose_W_S": np.asarray(op.stock_pose_W_S, float).tolist()}
                               for op in ops]}
    writer.write(document, frames=run.frames, keyframes=run.keyframes.layout)


# =============================================================================== keyframes

KEYFRAME_MESH_CHUNK = 256          # bricks meshed per call for a keyframe

def keyframe_payload(run, full: bool = False) -> dict:
    """Arguments of KeyframeWriter.write for the current state: the re-meshed dirty bricks of every workpiece (all
    surface bricks for a full snapshot), coarse fields (workpiece temperatures, chip pile, wetted cells, coolant
    jets, tool deflection, part pose, pose_M_S) and the chips in flight."""
    from ..removal import mesh as RM
    from .keyframes import FieldGrid
    state = run.state
    rs = state.removal
    meshes, changed = {}, []
    for w, wp in enumerate(rs.workpieces):
        bricks = RM.surface_bricks(wp.field, 0) if full else RM.dirty_bricks(wp.field, 0)
        bricks = [int(b) for b in np.asarray(bricks).reshape(-1)]
        if not bricks:
            continue
        # meshed a chunk of bricks at a time and kept as float32 / int32 (the records quantise the vertices to
        # 16 bits per brick anyway): a full snapshot of a fine field is millions of triangles
        for i in range(0, len(bricks), KEYFRAME_MESH_CHUNK):
            part = bricks[i:i + KEYFRAME_MESH_CHUNK]
            built = RM.mesh_bricks(wp.field, part, 0)
            for b in part:
                key = b if w == 0 else (w, b)
                mesh = built.get(b) if isinstance(built, dict) else None
                changed.append(key)
                if mesh is not None and len(mesh.faces):
                    meshes[key] = (np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32))
            del built
        RM.clear_dirty(wp.field, bricks)
    bus = state.bus_prev
    fields: dict = {"pose_M_S": pose_M_S(run.motion, state)}
    grids = {}
    th = bus.thermal
    temps = getattr(th, "work_temp_K", {}) or {}
    if 0 in temps and np.size(temps[0]):
        grids["work_temp_K"] = FieldGrid(np.asarray(temps[0], float), np.asarray(th.work_temp_origin_S_m[0], float),
                                         float(th.work_temp_cell_m[0]))
    chips = bus.chips
    pile = getattr(chips, "pile_height_m", None)
    if pile is not None and np.size(pile) and getattr(chips, "pile_origin_S_m", None) is not None \
            and getattr(chips, "pile_cell_m", None) is not None:
        fields["pile_height_m"] = np.asarray(pile, np.float32)
        fields["pile_origin_S_m"] = np.asarray(chips.pile_origin_S_m, float)
        fields["pile_cell_m"] = np.array(float(np.asarray(chips.pile_cell_m, float).reshape(-1)[0]))
        axes = getattr(chips, "pile_axes_S", None)
        if axes is not None and np.size(axes):
            fields["pile_axes_S"] = np.asarray(axes, float)
    cool = bus.coolant
    wetted = getattr(cool, "wetted", None)
    if wetted is not None and np.size(wetted) and getattr(cool, "wetted_origin_S_m", None) is not None \
            and getattr(cool, "wetted_cell_m", None) is not None:
        fields["wetted"] = np.asarray(wetted, np.int8)
        fields["wetted_origin_S_m"] = np.asarray(cool.wetted_origin_S_m, float)
        fields["wetted_cell_m"] = np.array(float(np.asarray(cool.wetted_cell_m, float).reshape(-1)[0]))
    jets = list(getattr(cool, "jets", []) or [])
    if jets:
        fields["jet_origin_M_m"] = np.array([np.asarray(j.origin_M_m, float) for j in jets])
        fields["jet_direction_M"] = np.array([np.asarray(j.direction_M, float) for j in jets])
        fields["jet_speed_m_s"] = np.array([float(j.speed_m_s) for j in jets])
        fields["jet_flow_m3_s"] = np.array([float(j.flow_m3_s) for j in jets])
        fields["jet_half_angle_rad"] = np.array([float(j.half_angle_rad) for j in jets])
    # the tool's deflection at the step's peak per axis of T0, a 3-vector per contact (shape [3] for one contact,
    # [n, 3] for several: the form cnctwin.render and the API's keyframe fields read)
    dyn = [c for c in (getattr(bus.dynamics, "contacts", []) or []) if np.size(c.deflection_peak_T0_m) >= 3]
    if dyn:
        vectors = np.array([np.asarray(c.deflection_peak_T0_m, float).reshape(-1)[:3] for c in dyn])
        fields["tool_deflection_m"] = vectors[0] if len(vectors) == 1 else vectors
    parts = getattr(bus.holding, "parts", {}) or {}
    if 0 in parts and getattr(parts[0], "pose_delta", None) is not None:
        fields["part_pose_delta"] = np.asarray(parts[0].pose_delta, float)
    chips_arrays = run.domains["chips"].keyframe_arrays(state)
    return {"meshes": meshes, "changed": changed, "fields": fields, "grids": grids, "chips": chips_arrays,
            "full": True if full else None}


def write_keyframe(run, tick: int, full: bool = False) -> dict:
    payload = keyframe_payload(run, full)
    entry = run.keyframes.write(int(tick), payload["meshes"], payload["changed"], full=payload["full"],
                                fields=payload["fields"], grids=payload["grids"], chips=payload["chips"],
                                extra={"op": run.op_name(), "part": int(run.state.clock.part_index)})
    return entry


# =============================================================================== emission rows for chips

def emission_buffer(bus: Bus, width: int):
    """chips' emission EventBuffer from the step's cutting records: cutting's pass rows have the chips layout
    (cnctwin.cutting.abi.PASS_COLUMNS = cnctwin.chips.abi.EMISSION_COLUMNS)."""
    from .abi import EventBuffer
    rows = [np.asarray(c.emissions, float).reshape(-1, width) for c in bus.cutting.contacts
            if getattr(c, "emissions", None) is not None and np.size(c.emissions)]
    data = np.concatenate(rows, axis=0) if rows else np.zeros((0, width))
    rec = np.zeros((max(len(data), 1), width))
    rec[:len(data)] = data
    return EventBuffer(rec, np.array([len(data), 0], np.int64))


# =============================================================================== summary

def _sum_contacts(bus_records, name: str) -> float:
    return float(sum(float(getattr(c, name, 0.0) or 0.0) for c in bus_records))


def finite_or_none(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


__all__ = ["axis_origins", "box_mesh", "frames_layout", "pose_M_S", "write_scene", "keyframe_payload", "write_keyframe",
           "emission_buffer", "finite_or_none"]
