"""Reading a run directory (docs/architecture.md 16) for pictures and sound (docs/models/render.md "Inputs").

What is read and how:

* ``run.json``: the resolved project (machine, material, tools, coolant), the clock and the summary.
* ``scene.json``: machine components, fixture, stock, tools (tool envelope GLB in T, origin at the tip, +z_T
  toward the spindle), nozzles, and the frames.bin header (``scene["frames"]``: record size and fields
  ``{name, dtype, shape, offset}``). Meshes (``scene/*.glb``) are read with :func:`cnctwin.geom.io.read_glb`
  (node transforms applied). A component's mesh is in the frame named by its ``frame`` key: ``S`` (stock),
  ``M`` (machine, placed with ``pose_M_S``) or ``T`` (moves with the tool); ``pose`` (4 x 4) places it in that
  frame. Without ``frame``: fixture items and work-chain components are S, components linked to a tool-chain
  axis are T, the rest M.
* ``frames.bin``: one record per L1 tick (memory-mapped; a record still being written is ignored): tick,
  axis values, spindle angle wrapped to [0, 2 pi) with its revolution count, tool tip and tool axis in S per
  path. The unwrapped spindle angle is ``angle + 2 pi rev``.
* ``keyframes/index.jsonl``, ``keyframes/bricks.idx`` and the brick records of ``keyframes/kf_<k>.glb``: the
  in-process workpiece at keyframe k is, per brick, the newest record at or before k. A record is
  ``<4sIIffff`` (b"BRK1", n_vertices, n_indices, origin_S_m[3], cell_m), uint16 positions [n, 4] in cells
  (position = origin + cells x cell) and uint32 indices; files whose records are not BRK1 are read as glTF
  nodes named ``brick_<b>``. ``keyframes/attr/brick_<b>.npz`` (per-vertex deviation_m, Ra_m,
  generating_VB_m of the final brick mesh) is attached when the vertex count matches.
* ``keyframes/kf_<k>.npz`` (fields: surface/work temperatures, pile heights, wetted cells, tool deflection,
  part pose, ``pose_M_S``, coolant jets ``jet_*``) and ``chips/kf_<k>.npz`` (particles in M, pile grid).
* ``series/<tier>/part-<n>.parquet`` with ``series/index.jsonl``: columns by name, rows by tick range.
* ``timeline.json``: executed blocks (tick0, tick1, kind, line, block text, tool number).

Readers never write; missing files read as empty (a running or synthetic run may not have them all).
"""

from __future__ import annotations

import functools
import json
import math
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..units import degC_to_K

BRICK_MAGIC = b"BRK1"
BRICK_HEADER = struct.Struct("<4sIIffff")
WORK_CHAIN = "work"
NPZ_CACHE_ENTRIES = 8          # keyframe field and chip files kept in memory
TOOL_CHAIN = "tool"


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        return []
    out = []
    for raw in lines:
        try:
            out.append(json.loads(raw))
        except ValueError:
            break                       # a line still being written
    return out


def load_npz(path: Path) -> dict:
    """Arrays of an .npz file (``allow_pickle=False``), {} when missing or unreadable."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile):
        return {}


def frame_dtype(header: dict) -> np.dtype:
    fields = header.get("fields", [])
    return np.dtype({"names": [f["name"] for f in fields],
                     "formats": [(f["dtype"], tuple(f["shape"])) if f.get("shape") else f["dtype"] for f in fields],
                     "offsets": [int(f["offset"]) for f in fields], "itemsize": int(header["record_bytes"])})


@dataclass
class IPW:
    """In-process workpiece at one keyframe: one mesh in S, the brick of each vertex and per-vertex attributes
    (NaN where the brick's final attributes do not apply)."""

    k: int
    tick: int
    vertices: np.ndarray
    faces: np.ndarray
    brick_of_vertex: np.ndarray
    attrs: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return len(self.faces) == 0


class RunData:
    """Lazy, read-only view of one run directory."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.run = _json(self.path / "run.json") or {}
        self.scene = _json(self.path / "scene.json") or {}
        doc = _json(self.path / "channels.json")
        items = doc.get("channels", []) if isinstance(doc, dict) else (doc or [])
        self.channels = {c["name"]: c for c in items if isinstance(c, dict) and "name" in c}
        doc = _json(self.path / "timeline.json")
        self.timeline = (doc.get("blocks", []) if isinstance(doc, dict) else (doc or [])) or []
        self.keyframes = _jsonl(self.path / "keyframes" / "index.jsonl")
        self.keyframe_ticks = np.array([int(e["tick"]) for e in self.keyframes], np.int64)
        doc = _json(self.path / "keyframes" / "bricks.idx")
        self.bricks_index = self._bricks_index(doc)
        self._ipw_cache: dict = {}
        self._mesh_cache: dict = {}
        self._series_cache: dict = {}
        self._npz_cache: dict = {}

    # ------------------------------------------------------------------ clock and project
    @property
    def tick_s(self) -> float:
        clock = self.run.get("clock") or {}
        header = self.scene.get("frames") or {}
        return float(clock.get("tick_s") or header.get("tick_s") or 1e-6)

    @property
    def ipo_ticks(self) -> int:
        clock = self.run.get("clock") or {}
        header = self.scene.get("frames") or {}
        return int(clock.get("ipo_ticks") or header.get("ipo_ticks") or 1000)

    @property
    def project(self) -> dict:
        return self.run.get("project") or {}

    @property
    def end_tick(self) -> int:
        frames = self.frames
        if len(frames):
            return int(frames["tick"][-1]) + self.ipo_ticks
        if self.timeline:
            return int(max(int(b.get("tick1", 0)) for b in self.timeline))
        return 0

    # ------------------------------------------------------------------ frames.bin
    @functools.cached_property
    def frames(self) -> np.ndarray:
        header = self.scene.get("frames")
        path = self.path / (header.get("file", "frames.bin") if header else "frames.bin")
        if not header or not path.is_file():
            return np.zeros(0, np.dtype([("tick", "<i8")]))
        dtype = frame_dtype(header)
        count = path.stat().st_size // dtype.itemsize
        if count == 0:
            return np.zeros(0, dtype)
        return np.memmap(path, dtype=dtype, mode="r", shape=(count,))

    @property
    def frame_ticks(self) -> np.ndarray:
        frames = self.frames
        return np.asarray(frames["tick"], dtype=np.int64) if len(frames) else np.zeros(0, np.int64)

    def frame_index(self, tick: int) -> int:
        ticks = self.frame_ticks
        if len(ticks) == 0:
            return -1
        return int(np.clip(np.searchsorted(ticks, tick, side="right") - 1, 0, len(ticks) - 1))

    def frame_field(self, name: str, path: int = 0) -> np.ndarray | None:
        """A frames.bin field for one path as float64 ([n] or [n, k]); None when the layout has no such field."""
        frames = self.frames
        if not len(frames) or name not in frames.dtype.names:
            return None
        data = np.asarray(frames[name])
        if data.ndim >= 2 and data.shape[1] > path and name != "axis":
            data = data[:, path]
        return np.asarray(data, dtype=np.float64)

    def tip_S(self, path: int = 0) -> np.ndarray:
        data = self.frame_field("tip_S_m", path)
        return data.reshape(-1, 3) if data is not None else np.zeros((0, 3))

    def axis_T_S(self, path: int = 0) -> np.ndarray:
        data = self.frame_field("axis_T_S", path)
        if data is None:
            return np.tile(np.array([0.0, 0.0, 1.0]), (len(self.frames), 1))
        return data.reshape(-1, 3)

    def spindle_angle(self, path: int = 0) -> np.ndarray:
        """Unwrapped spindle angle (rad, right-handed about +z_T) per record."""
        angle = self.frame_field("spindle_angle_rad", path)
        if angle is None:
            return np.zeros(len(self.frames))
        rev = self.frame_field("spindle_rev", path)
        angle = angle.reshape(-1)
        return angle + 2.0 * math.pi * (rev.reshape(-1) if rev is not None else 0.0)

    def axis_values(self) -> tuple[list[str], np.ndarray]:
        header = self.scene.get("frames") or {}
        names = [a.get("name", f"A{i}") for i, a in enumerate(header.get("axes", []))]
        data = self.frame_field("axis")
        if data is None:
            return names, np.zeros((len(self.frames), len(names)))
        data = data.reshape(len(self.frames), -1)
        return names[: data.shape[1]] or [f"A{i}" for i in range(data.shape[1])], data

    # ------------------------------------------------------------------ timeline
    def block_at(self, tick: int) -> dict | None:
        for block in reversed(self.timeline):
            if int(block.get("tick0", 0)) <= tick:
                return block
        return None

    def tool_number_at(self, tick: int, path: int = 0) -> int | None:
        """Active tool number at ``tick``: frames field ``tool_number`` if the layout has it, else the timeline."""
        data = self.frame_field("tool_number", path)
        if data is not None:
            value = int(data.reshape(-1)[self.frame_index(tick)])
            return value if value >= 0 else None
        number = None
        for block in self.timeline:
            if int(block.get("path", 0)) != path:
                continue
            if int(block.get("tick0", 0)) > tick:
                break
            if block.get("tool") is not None:
                number = int(block["tool"])
        return number

    def tool_entry(self, number: int | None) -> dict | None:
        for entry in self.scene.get("tools", []) or []:
            if number is not None and int(entry.get("number", -1)) == int(number):
                return entry
        return None

    # ------------------------------------------------------------------ meshes
    def mesh(self, rel: str | None):
        """(vertices [n, 3], faces [m, 3]) of a GLB file of the run (node transforms applied), cached."""
        if not rel:
            return None
        if rel in self._mesh_cache:
            return self._mesh_cache[rel]
        from ..geom.io.gltf import read_glb

        path = self.path / rel
        result = None
        if path.is_file():
            mesh = read_glb(path).mesh(world=True)
            if mesh.n_faces:
                result = (np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64))
        self._mesh_cache[rel] = result
        return result

    def components(self) -> list[dict]:
        """Machine components, fixture items, nozzles with meshes: ``{id, mesh, frame, pose, kind, opacity}``."""
        chain = self.scene.get("chain") or {}
        tool_axes = {str(a) for a in chain.get(TOOL_CHAIN, [])}
        out = []
        machine = self.scene.get("machine") or {}
        for item in machine.get("components", []) or []:
            frame = item.get("frame")
            if frame is None:
                link = item.get("link")
                frame = "T" if link is not None and str(link) in tool_axes else ("S" if link is not None else "M")
            out.append(self._component(item, frame, item.get("kind", "machine")))
        for item in self.scene.get("fixture", []) or []:
            out.append(self._component(item, item.get("frame", "S"), item.get("kind", "fixture")))
        for item in self.scene.get("enclosure", []) or []:
            out.append(self._component(item, item.get("frame", "M"), item.get("kind", "enclosure")))
        return out

    @staticmethod
    def _component(item: dict, frame: str, kind: str) -> dict:
        pose = np.asarray(item.get("pose", np.eye(4)), dtype=np.float64).reshape(4, 4)
        return {"id": str(item.get("id", "")), "mesh": item.get("mesh"), "frame": str(frame).upper(), "pose": pose,
                "kind": kind, "opacity": item.get("opacity")}

    def stock_mesh(self):
        stock = self.scene.get("stock") or {}
        return self.mesh(stock.get("mesh"))

    # ------------------------------------------------------------------ keyframes
    @staticmethod
    def _bricks_index(doc) -> dict[int, list]:
        if not doc:
            return {}
        bricks = doc.get("bricks", {}) if isinstance(doc, dict) else {}
        out = {int(b): sorted([list(e) for e in entries], key=lambda e: int(e[0])) for b, entries in bricks.items()}
        return dict(sorted(out.items()))

    def keyframe_at(self, tick: int) -> dict | None:
        if not len(self.keyframe_ticks):
            return None
        i = int(np.searchsorted(self.keyframe_ticks, tick, side="right")) - 1
        return self.keyframes[i] if i >= 0 else None

    def _npz(self, rel: str) -> dict:
        """An .npz file of the run, kept for the last few keyframes asked (frames walk forward in time)."""
        if rel not in self._npz_cache:
            self._npz_cache[rel] = load_npz(self.path / rel)
            while len(self._npz_cache) > NPZ_CACHE_ENTRIES:
                self._npz_cache.pop(next(iter(self._npz_cache)))
        return self._npz_cache[rel]

    def fields(self, k: int) -> dict:
        entry = self.keyframes[k] if 0 <= k < len(self.keyframes) else {}
        return self._npz(entry.get("fields", f"keyframes/kf_{k}.npz"))

    def chips(self, k: int) -> dict:
        entry = self.keyframes[k] if 0 <= k < len(self.keyframes) else {}
        return self._npz(entry.get("chips", f"chips/kf_{k}.npz"))

    def ipw(self, k: int) -> IPW:
        """The in-process workpiece at keyframe ``k`` (cached for the last keyframe asked)."""
        if k in self._ipw_cache:
            return self._ipw_cache[k]
        final = {b: int(entries[-1][0]) for b, entries in self.bricks_index.items()}
        parts = []
        for brick, entries in self.bricks_index.items():
            chosen = [e for e in entries if int(e[0]) <= k]
            if not chosen:
                continue
            kk, name, start, length = chosen[-1][:4]
            mesh = self._brick_record(str(name), int(start), int(length), brick)
            if mesh is None:
                continue
            attrs = self._brick_attrs(brick, len(mesh[0])) if int(kk) == final[brick] else {}
            parts.append((brick, mesh[0], mesh[1], attrs))
        tick = int(self.keyframes[k]["tick"]) if 0 <= k < len(self.keyframes) else 0
        if not parts:
            ipw = IPW(k, tick, np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64), {})
        else:
            offsets = np.cumsum([0] + [len(v) for _, v, _, _ in parts])
            keys = sorted({key for _, _, _, attrs in parts for key in attrs})
            ipw = IPW(k, tick, np.concatenate([v for _, v, _, _ in parts]),
                      np.concatenate([f + offsets[i] for i, (_, _, f, _) in enumerate(parts)]),
                      np.concatenate([np.full(len(v), b, np.int64) for b, v, _, _ in parts]),
                      {key: np.concatenate([attrs.get(key, np.full(len(v), np.nan)) for _, v, _, attrs in parts])
                       for key in keys})
        self._ipw_cache = {k: ipw}               # one keyframe at a time: frames walk forward in time
        return ipw

    def _brick_record(self, name: str, start: int, length: int, brick: int):
        path = self.path / "keyframes" / name
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                record = handle.read(length)
        except OSError:
            return None
        if len(record) >= BRICK_HEADER.size and record[:4] == BRICK_MAGIC:
            magic, n_vert, n_idx, ox, oy, oz, cell = BRICK_HEADER.unpack_from(record, 0)
            pos = np.frombuffer(record, dtype="<u2", count=4 * n_vert, offset=BRICK_HEADER.size).reshape(-1, 4)
            idx = np.frombuffer(record, dtype="<u4", count=n_idx, offset=BRICK_HEADER.size + 8 * n_vert)
            vertices = np.array([ox, oy, oz], dtype=np.float64) + pos[:, :3].astype(np.float64) * float(cell)
            return vertices, idx.astype(np.int64).reshape(-1, 3)
        from ..geom.io.gltf import read_glb

        scene = read_glb(path)
        node = scene.find(f"brick_{brick}")
        if node is None or node["mesh"] is None:
            return None
        mesh = node["mesh"].transformed(node["world"])
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)

    def _brick_attrs(self, brick: int, n_vertices: int) -> dict:
        arrays = load_npz(self.path / "keyframes" / "attr" / f"brick_{brick}.npz")
        return {name: np.asarray(a, dtype=np.float64).reshape(-1) for name, a in arrays.items()
                if np.asarray(a).size == n_vertices and np.issubdtype(np.asarray(a).dtype, np.number)}

    def pose_M_S(self, tick: int) -> tuple[np.ndarray, bool]:
        """Pose of S in M at ``tick``: keyframe field ``pose_M_S`` at or before the tick, else ``scene.stock.pose_M_S``;
        (identity, False) when the run gives neither."""
        entry = self.keyframe_at(tick)
        if entry is not None:
            fields = self.fields(int(entry["k"]))
            pose = fields.get("pose_M_S")
            if pose is not None:
                pose = np.asarray(pose, dtype=np.float64)
                return (pose.reshape(-1, 4, 4)[0] if pose.size > 16 else pose.reshape(4, 4)), True
        stock = self.scene.get("stock") or {}
        if stock.get("pose_M_S") is not None:
            return np.asarray(stock["pose_M_S"], dtype=np.float64).reshape(4, 4), True
        return np.eye(4), False

    # ------------------------------------------------------------------ series
    def _parts(self, tier: str) -> list[dict]:
        index = [e for e in _jsonl(self.path / "series" / "index.jsonl") if e.get("tier") == tier]
        if index:
            return sorted(index, key=lambda e: int(e.get("part", 0)))
        folder = self.path / "series" / tier
        files = sorted(folder.glob("part-*.parquet"), key=lambda p: int(p.stem.split("-")[1])) if folder.is_dir() else []
        return [{"tier": tier, "part": int(p.stem.split("-")[1]), "file": p.relative_to(self.path).as_posix()}
                for p in files]

    def series_columns(self, tier: str) -> list[str]:
        import pyarrow.parquet as pq

        for part in self._parts(tier):
            try:
                return list(pq.ParquetFile(self.path / part["file"]).schema_arrow.names)
            except OSError:
                continue
        return []

    def series(self, tier: str, columns, tick0: int | None = None, tick1: int | None = None) -> dict:
        """``{"tick": int64 [n], column: float64 [n]}`` for the columns present, rows with tick0 <= tick < tick1."""
        import pyarrow.parquet as pq

        wanted = [c for c in dict.fromkeys(columns) if c != "tick"]
        ticks, data = [], {c: [] for c in wanted}
        present: set[str] = set()
        for part in self._parts(tier):
            if tick0 is not None and part.get("tick1") is not None and int(part["tick1"]) < tick0:
                continue
            if tick1 is not None and part.get("tick0") is not None and int(part["tick0"]) >= tick1:
                continue
            path = self.path / part["file"]
            try:
                names = set(pq.ParquetFile(path).schema_arrow.names)
            except OSError:
                continue
            cols = ["tick"] + [c for c in wanted if c in names]
            table = pq.read_table(path, columns=cols)
            t = table.column("tick").to_numpy().astype(np.int64)
            keep = np.ones(len(t), bool)
            if tick0 is not None:
                keep &= t >= tick0
            if tick1 is not None:
                keep &= t < tick1
            ticks.append(t[keep])
            for c in wanted:
                if c in names:
                    present.add(c)
                    data[c].append(table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)[keep])
                else:
                    data[c].append(np.full(int(keep.sum()), np.nan))
        out = {"tick": np.concatenate(ticks) if ticks else np.zeros(0, np.int64)}
        for c in wanted:
            if c in present:
                out[c] = np.concatenate(data[c]) if data[c] else np.zeros(0)
        return out

    def series_all(self, tier: str, columns) -> dict:
        """:meth:`series` over the whole run, cached per (tier, columns) for the life of this object."""
        key = (tier, tuple(columns))
        if key not in self._series_cache:
            self._series_cache[key] = self.series(tier, list(columns))
        return self._series_cache[key]

    def series_value_at(self, tier: str, columns, tick: int) -> dict:
        """The last row at or before ``tick`` of each column (NaN when there is none)."""
        data = self.series_all(tier, columns)
        ticks = data["tick"]
        i = int(np.searchsorted(ticks, tick, side="right")) - 1
        return {c: (float(data[c][i]) if c in data and i >= 0 else float("nan")) for c in columns}

    def channel_columns(self, name: str) -> list[str]:
        spec = self.channels.get(name)
        if spec is None:
            return []
        shape = spec.get("max_shape") or []
        if not shape:
            return [name]
        count = int(np.prod(shape))
        return [f"{name}[{i}]" for i in range(count)]

    # ------------------------------------------------------------------ material and coolant
    def material_family(self) -> str:
        material = self.project.get("material") or {}
        family = material.get("family")
        if family:
            return str(family)
        item = db_item("materials", material.get("id"))
        return str(item.get("family", "")) if item else ""

    def coolant(self) -> dict:
        """The run's fluid with its thresholds in SI where known: ``{id, kind, boiling_K, smoke_point_K}``."""
        coolant = self.project.get("coolant") or {}
        fluid = coolant.get("fluid") or {}
        if isinstance(fluid, str):
            fluid = {"id": fluid}
        out = {"id": fluid.get("id"), "kind": fluid.get("kind")}
        for si_key, file_key in (("boiling_K", "boiling_degC"), ("smoke_point_K", "smoke_point_degC")):
            if fluid.get(si_key) is not None:
                out[si_key] = float(_value(fluid[si_key]))
        item = db_item("coolants", fluid.get("id")) if fluid.get("id") else None
        if item:
            out.setdefault("kind", item.get("kind"))
            if out["kind"] is None:
                out["kind"] = item.get("kind")
            for si_key, file_key in (("boiling_K", "boiling_degC"), ("smoke_point_K", "smoke_point_degC")):
                if si_key not in out and _value(item.get(file_key)) is not None:
                    out[si_key] = degC_to_K(float(_value(item[file_key])))
        return out


def _value(entry):
    """A database scalar written plain or as ``{value, src, ...}``."""
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def db_item(folder: str, item_id) -> dict | None:
    """The raw data of ``db/<folder>/<id>.yaml`` (no validation; None if absent), for display thresholds and
    colours only."""
    if not item_id:
        return None
    from ..core.errors import InputError
    from ..db.yamlio import load_file
    from ..paths import DB

    path = DB / folder / f"{item_id}.yaml"
    if not path.is_file():
        return None
    try:
        data = load_file(path).data
    except (InputError, OSError):
        return None          # the caller uses its documented fall-back values (render-cues, material colours)
    return data if isinstance(data, dict) else None
