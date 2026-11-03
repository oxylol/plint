"""Keyframes of the in-process workpiece (docs/architecture.md 16, docs/formats.md "Keyframes").

At every keyframe (``recording.keyframe_period_s``) the run writes:

* ``keyframes/kf_<k>.glb``: a glTF 2.0 binary with one node per brick (32^3 lattice cells of the
  removal field) that changed since the last keyframe; every ``keyframe_full_every``-th keyframe
  (20) holds every non-empty brick (a full snapshot). Each brick's bytes in the BIN chunk are one
  self-describing **brick record** (:func:`brick_record`): a 28-byte header ``<4sIIffff`` (b"BRK1",
  n_vertices, n_indices, origin_S_m[3] float32, step_m float32), uint16 positions [n_vertices, 4]
  (x, y, z, 0: KHR_mesh_quantization, ``p = origin + q * step``) and uint32 indices [n_indices]
  (triangles, counter-clockwise from outside). The glTF node of the brick dequantises the same
  way (translation = origin, scale = step), so any glTF viewer shows the keyframe as it is; an
  emptied brick is a node without mesh and a record with no vertices.
* ``keyframes/bricks.idx``: JSON ``{bricks: {"<brick>": [[k, file, offset, length], ...]}}``, each
  brick's versions sorted by k (a full snapshot adds no entry for a brick whose record is
  byte-identical to its last version): the state of a brick at time t is one ranged read.
* ``keyframes/kf_<k>.npz``: coarse fields (temperatures as float16, 3-D grids block-averaged to cells
  of at most ``keyframe_field_max_cell_m`` = 8 mm; chip piles, wetted cells, deflection, part pose,
  coolant jets as given); ``chips/kf_<k>.npz``: the chips' particles and pile summary.
* ``keyframes/index.jsonl`` (append-only): ``{k, tick, time_s, glb, fields, chips, full, bricks, empty}``.
* ``keyframes/attr/brick_<brick>.npz``: per-brick final attributes, written at the end of the run.

Quantisation error: positions are rounded to ``step = extent / 65535`` of the brick's own vertex
bounding box, so every vertex is within ``step / 2`` (plus float32 rounding of origin and step) of its
float64 position: 0.06 um for an 8 mm brick (lattice 0.25 mm).

The GLB container is written here rather than with cnctwin.geom's general writer because each
brick must be one contiguous, self-describing record whose byte span the index can point to.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import FORMAT_VERSION
from ..db.models import params as model_params
from .rundir import (TICK_S, fsync_dir, fsync_file, jsonable, npz_bytes, read_json, tick_time_s, write_bytes_atomic,
                     write_json_atomic, write_npz)

BRICK_MAGIC = b"BRK1"
RECORD_HEADER = struct.Struct("<4sIIffff")      # magic, n_vertices, n_indices, origin_S_m[3], step_m
QUANT_MAX = int(np.iinfo(np.uint16).max)        # positions span 0 .. 65535 steps
POSITION_STRIDE = 8                             # uint16 x, y, z + one pad (4-byte vertex alignment)
GLB_MAGIC = 0x46546C67
GLB_JSON = 0x4E4F534A
GLB_BIN = 0x004E4942
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
UNSIGNED_SHORT, UNSIGNED_INT = 5123, 5125
INDEX_FORMAT = "cnctwin-bricks-index"
BRICK_CELLS = 32                                # removal's BRICK (architecture 16: 32^3 lattice cells)
_KEY = re.compile(r"^(?:w(\d+)\.)?(\d+)$")
RECORD_DOC = ("<4sIIffff header (b'BRK1', n_vertices, n_indices, origin_S_m[3], step_m), uint16 positions "
              "[n_vertices, 4] (x, y, z, 0; p = origin + q * step), uint32 indices [n_indices]")


def brick_key(brick, workpiece: int = 0) -> str:
    """The id of a brick in bricks.idx and the API: ``"<brick>"`` on workpiece 0, ``"w<w>.<brick>"`` else."""
    if isinstance(brick, str):
        if not _KEY.match(brick):
            raise ValueError(f"brick key {brick!r} is not '<brick>' or 'w<workpiece>.<brick>'")
        return brick
    if isinstance(brick, tuple):
        workpiece, brick = brick
    brick, workpiece = int(brick), int(workpiece)
    if brick < 0 or workpiece < 0:
        raise ValueError("brick and workpiece numbers are not negative")
    return str(brick) if workpiece == 0 else f"w{workpiece}.{brick}"


def key_order(key: str) -> tuple[int, int]:
    match = _KEY.match(key)
    return (int(match.group(1) or 0), int(match.group(2)))


# =============================================================================== brick records

def quantise(vertices) -> tuple[np.ndarray, np.ndarray, float]:
    """(uint16 positions [n, 4], float32 origin [3], float32 step) of float64 vertices (metres).

    One step for all axes (``max extent / 65535``, rounded up to float32), the origin at the bounding
    box minimum rounded down to float32, so ``origin + q * step`` is within step/2 of every vertex."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    origin = lo.astype(np.float32)
    origin = np.where(origin.astype(np.float64) > lo, np.nextafter(origin, np.float32(-np.inf)), origin)
    extent = float(np.max(hi - origin.astype(np.float64)))
    step = np.float32(extent / QUANT_MAX) if extent > 0.0 else np.float32(1.0)
    while float(step) * QUANT_MAX < extent:
        step = np.nextafter(step, np.float32(np.inf))
    q = np.rint((v - origin.astype(np.float64)) / float(step))
    q = np.clip(q, 0, QUANT_MAX).astype(np.uint16)
    out = np.zeros((len(v), 4), np.uint16)
    out[:, :3] = q
    return out, origin, float(step)


def brick_record(vertices, faces) -> bytes:
    """One brick's self-describing record (module docstring); no vertices: an empty brick."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3) if vertices is not None else np.zeros((0, 3))
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3) if faces is not None else np.zeros((0, 3), np.int64)
    if len(f) == 0 or len(v) == 0:
        return RECORD_HEADER.pack(BRICK_MAGIC, 0, 0, 0.0, 0.0, 0.0, 0.0)
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"brick faces index vertices outside 0..{len(v) - 1}")
    if not np.all(np.isfinite(v)):
        raise ValueError("brick vertices are not finite")
    q, origin, step = quantise(v)
    indices = f.astype("<u4").reshape(-1)
    header = RECORD_HEADER.pack(BRICK_MAGIC, len(q), len(indices), float(origin[0]), float(origin[1]),
                                float(origin[2]), step)
    return header + q.astype("<u2").tobytes() + indices.tobytes()


def decode_record(data: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    """(vertices float64 [n, 3] in S, faces int64 [m, 3], header) of one brick record."""
    if len(data) < RECORD_HEADER.size:
        raise ValueError("brick record shorter than its header")
    magic, nv, ni, ox, oy, oz, step = RECORD_HEADER.unpack_from(data, 0)
    if magic != BRICK_MAGIC:
        raise ValueError(f"not a brick record (magic {magic!r})")
    need = RECORD_HEADER.size + nv * POSITION_STRIDE + ni * 4
    if len(data) < need:
        raise ValueError(f"brick record of {len(data)} bytes, its header needs {need}")
    q = np.frombuffer(data, dtype="<u2", count=nv * 4, offset=RECORD_HEADER.size).reshape(nv, 4)
    indices = np.frombuffer(data, dtype="<u4", count=ni, offset=RECORD_HEADER.size + nv * POSITION_STRIDE)
    origin = np.array([ox, oy, oz], dtype=np.float64)
    vertices = origin[None, :] + q[:, :3].astype(np.float64) * float(step)
    return vertices, indices.astype(np.int64).reshape(-1, 3), {"n_vertices": nv, "n_indices": ni,
                                                                "origin_S_m": origin, "step_m": float(step)}


def _mesh_arrays(mesh):
    if mesh is None:
        return None, None
    if isinstance(mesh, tuple):
        return mesh
    return mesh.vertices, mesh.faces


def brick_glb(records: list[tuple[str, bytes]], name: str = "") -> tuple[bytes, dict[str, tuple[int, int]]]:
    """glTF 2.0 binary of brick records (key, record bytes) and each record's (offset, length) in the file.

    Every record sits 4-byte aligned in the BIN chunk; its positions and indices are buffer views
    into it, so viewers ignore the 28 header bytes and the record stays one contiguous byte range."""
    binary = bytearray()
    views, accessors, meshes, nodes = [], [], [], []
    starts: dict[str, int] = {}
    for key, record in records:
        while len(binary) % 4:
            binary.append(0)
        start = len(binary)
        starts[key] = start
        binary += record
        magic, nv, ni, ox, oy, oz, step = RECORD_HEADER.unpack_from(record, 0)
        node = {"name": f"brick_{key}", "extras": {"brick": key}}
        if nv and ni:
            q = np.frombuffer(record, dtype="<u2", count=nv * 4, offset=RECORD_HEADER.size).reshape(nv, 4)
            views.append({"buffer": 0, "byteOffset": start + RECORD_HEADER.size, "byteLength": nv * POSITION_STRIDE,
                          "byteStride": POSITION_STRIDE, "target": ARRAY_BUFFER})
            accessors.append({"bufferView": len(views) - 1, "componentType": UNSIGNED_SHORT, "count": int(nv),
                              "type": "VEC3", "min": [int(x) for x in q[:, :3].min(axis=0)],
                              "max": [int(x) for x in q[:, :3].max(axis=0)]})
            views.append({"buffer": 0, "byteOffset": start + RECORD_HEADER.size + nv * POSITION_STRIDE,
                          "byteLength": ni * 4, "target": ELEMENT_ARRAY_BUFFER})
            accessors.append({"bufferView": len(views) - 1, "componentType": UNSIGNED_INT, "count": int(ni),
                              "type": "SCALAR"})
            meshes.append({"name": f"brick_{key}", "primitives": [{"attributes": {"POSITION": len(accessors) - 2},
                                                                   "indices": len(accessors) - 1, "mode": 4}]})
            node.update({"mesh": len(meshes) - 1, "translation": [float(ox), float(oy), float(oz)],
                         "scale": [float(step)] * 3})
        else:
            node["extras"]["empty"] = True
        nodes.append(node)
    while len(binary) % 4:
        binary.append(0)
    document: dict = {"asset": {"version": "2.0", "generator": "cnctwin keyframes"}, "scene": 0,
                      "scenes": [{"name": name} if not nodes else {"name": name, "nodes": list(range(len(nodes)))}]}
    if nodes:
        document["nodes"] = nodes
    if meshes:
        document.update({"meshes": meshes, "accessors": accessors, "bufferViews": views,
                         "extensionsUsed": ["KHR_mesh_quantization"],
                         "extensionsRequired": ["KHR_mesh_quantization"]})
    if binary:
        document["buffers"] = [{"byteLength": len(binary)}]
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    text += b" " * ((4 - len(text) % 4) % 4)
    chunks = struct.pack("<II", len(text), GLB_JSON) + text
    bin_start = 12 + len(chunks) + 8
    if binary:
        chunks += struct.pack("<II", len(binary), GLB_BIN) + bytes(binary)
    data = struct.pack("<III", GLB_MAGIC, 2, 12 + len(chunks)) + chunks
    spans = {key: (bin_start + starts[key], len(record)) for key, record in records}
    return data, spans


# =============================================================================== coarse fields

@dataclass
class FieldGrid:
    """A grid of a keyframe's fields: ``values`` [nx, ny(, nz)] cell-centred, cell (0, 0, 0) with its
    lower corner at ``origin_S_m``; NaN where there is no material."""

    values: np.ndarray
    origin_S_m: np.ndarray
    cell_m: float


def coarsen(values, cell_m: float, max_cell_m: float) -> tuple[np.ndarray, int]:
    """Block mean (finite values only; a block without one is NaN) over the largest integer factor f with
    ``f * cell_m <= max_cell_m`` (at least 1); the grid is padded with NaN to a multiple of f."""
    values = np.asarray(values, dtype=np.float64)
    factor = max(1, int(math.floor(max_cell_m / cell_m * (1.0 + 1e-12))))
    if factor == 1:
        return values.copy(), 1
    padded_shape = tuple(int(math.ceil(n / factor)) * factor for n in values.shape)
    padded = np.full(padded_shape, np.nan)
    padded[tuple(slice(0, n) for n in values.shape)] = values
    blocks = []
    for n in padded_shape:
        blocks.extend((n // factor, factor))
    view = padded.reshape(blocks)
    axes = tuple(range(1, 2 * values.ndim, 2))
    finite = np.isfinite(view)
    sums = np.where(finite, view, 0.0).sum(axis=axes)
    counts = finite.sum(axis=axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return mean, factor


def _is_temperature(name: str) -> bool:
    return name.endswith("_K")


def _field_arrays(tick: int, tick_s: float, fields: Mapping | None, grids: Mapping | None,
                  max_cell_m: float) -> dict:
    arrays: dict = {"tick": np.array(int(tick), np.int64), "time_s": np.array(tick_time_s(tick, tick_s))}
    for name, value in (fields or {}).items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"keyframe field {name!r} has dtype object")
        if _is_temperature(name) and np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float16)
        arrays[str(name)] = array
    for name, grid in (grids or {}).items():
        coarse, factor = coarsen(grid.values, float(grid.cell_m), max_cell_m)
        arrays[str(name)] = coarse.astype(np.float16 if _is_temperature(name) else np.float32)
        arrays[f"{name}_origin_S_m"] = np.asarray(grid.origin_S_m, dtype=np.float64).reshape(-1)
        arrays[f"{name}_cell_m"] = np.array(float(grid.cell_m) * factor)
    return arrays


# =============================================================================== writer

class KeyframeWriter:
    """Writes keyframes (module docstring). ``period_ticks``: base ticks between keyframes; the first is
    due at tick 0. ``lattice_m``, ``origin_S_m`` and ``grid`` describe the brick layout for readers
    (bricks.idx header); they do not change what is written."""

    def __init__(self, run_dir, *, period_ticks: int, tick_s: float = TICK_S, full_every: int | None = None,
                 field_max_cell_m: float | None = None, lattice_m: float | None = None, origin_S_m=None,
                 grid=None) -> None:
        settings = model_params("recording")
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "keyframes"
        self.chips_dir = self.run_dir / "chips"
        self.period_ticks = int(period_ticks)
        if self.period_ticks <= 0:
            raise ValueError("the keyframe period is at least one tick")
        self.tick_s = float(tick_s)
        self.full_every = int(full_every if full_every is not None else settings["keyframe_full_every"])
        self.field_max_cell_m = float(field_max_cell_m if field_max_cell_m is not None
                                      else settings["keyframe_field_max_cell_m"])
        self.layout = {"cells": BRICK_CELLS, "lattice_m": lattice_m,
                       "origin_S_m": None if origin_S_m is None else [float(x) for x in origin_S_m],
                       "grid": None if grid is None else [int(x) for x in grid]}
        self.k = 0
        self.next_tick = 0
        self.bricks: dict[str, list] = {}       # key -> [[k, file, offset, length], ...]
        self.digest: dict[str, str] = {}        # key -> sha256 of its last indexed record
        self.live: set[str] = set()             # keys whose last record has a surface
        self.index_bytes = 0
        self.dir.mkdir(parents=True, exist_ok=True)

    def due(self, tick: int) -> bool:
        return int(tick) >= self.next_tick

    def write(self, tick: int, meshes: Mapping, changed=None, *, full: bool | None = None,
              fields: Mapping | None = None, grids: Mapping | None = None, chips: Mapping | None = None,
              extra: Mapping | None = None) -> dict:
        """Write keyframe k = the next number at ``tick``.

        ``meshes``: brick -> mesh (``vertices`` [n, 3] in S, metres, and ``faces`` [m, 3]; or a pair; None
        for an emptied brick), keyed by int (workpiece 0), (workpiece, brick) or a brick key. ``changed``:
        the bricks re-meshed for this keyframe (dirty bricks); a changed brick missing from ``meshes`` is
        empty now (default: the keys of ``meshes``). A full snapshot (every ``full_every``-th keyframe,
        or ``full=True``) expects a mesh for every non-empty brick; a brick that had a surface and is
        not given is empty now."""
        k = self.k
        full = (k % self.full_every == 0) if full is None else bool(full)
        given = {brick_key(b): m for b, m in meshes.items()}
        keys = set(given)
        keys.update(brick_key(b) for b in (changed if changed is not None else ()))
        if full:
            keys.update(self.live)
        records: list[tuple[str, bytes]] = []
        new_entries: list[str] = []
        emptied: list[str] = []
        for key in sorted(keys, key=key_order):
            vertices, faces = _mesh_arrays(given.get(key))
            record = brick_record(vertices, faces)
            digest = hashlib.sha256(record).hexdigest()
            empty = len(record) == RECORD_HEADER.size
            if empty and key not in self.live and key not in self.bricks:
                continue                            # never had a surface: nothing to record
            same = self.digest.get(key) == digest
            if same and not full:
                continue
            records.append((key, record))
            if not same:
                new_entries.append(key)
                self.digest[key] = digest
                if empty:
                    emptied.append(key)
                    self.live.discard(key)
                else:
                    self.live.add(key)
        glb_name = f"kf_{k}.glb"
        data, spans = brick_glb(records, name=f"kf_{k}")
        write_bytes_atomic(self.dir / glb_name, data)
        for key in new_entries:
            offset, length = spans[key]
            self.bricks.setdefault(key, []).append([k, glb_name, int(offset), int(length)])
        if new_entries:
            self._write_index()
        elif k == 0:
            self._write_index()
        write_npz(self.dir / f"kf_{k}.npz", _field_arrays(tick, self.tick_s, fields, grids, self.field_max_cell_m))
        chips_name = None
        if chips is not None:
            self.chips_dir.mkdir(parents=True, exist_ok=True)
            arrays = {str(n): np.asarray(v) for n, v in chips.items()}
            arrays.setdefault("tick", np.array(int(tick), np.int64))
            write_npz(self.chips_dir / f"kf_{k}.npz", arrays)
            chips_name = f"chips/kf_{k}.npz"
        entry = {"k": k, "tick": int(tick), "time_s": tick_time_s(tick, self.tick_s), "glb": f"keyframes/{glb_name}",
                 "fields": f"keyframes/kf_{k}.npz", "chips": chips_name, "full": full,
                 "bricks": [key for key, _ in records], "empty": emptied}
        if extra:
            entry.update(jsonable(dict(extra)))
        line = (json.dumps(jsonable(entry), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        with open(self.dir / "index.jsonl", "ab") as handle:
            handle.write(line)
        self.index_bytes += len(line)
        self.k += 1
        self.next_tick = (int(tick) // self.period_ticks + 1) * self.period_ticks
        return entry

    def index_document(self) -> dict:
        return {"format": INDEX_FORMAT, "version": FORMAT_VERSION, **jsonable(self.layout), "record": RECORD_DOC,
                "bricks": {key: [list(e) for e in self.bricks[key]] for key in sorted(self.bricks, key=key_order)}}

    def _write_index(self) -> None:
        write_json_atomic(self.dir / "bricks.idx", self.index_document(), indent=None)

    def write_attributes(self, attributes: Mapping) -> list[str]:
        """``keyframes/attr/brick_<brick>.npz`` per brick: final per-vertex attributes (deviation, finish,
        wear ...), in the vertex order of the brick's last record."""
        folder = self.dir / "attr"
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        for brick, arrays in sorted(((brick_key(b), a) for b, a in attributes.items()), key=lambda kv: key_order(kv[0])):
            name = f"attr/brick_{brick}.npz"
            write_npz(self.dir / name, {str(n): np.asarray(v) for n, v in arrays.items()})
            written.append(f"keyframes/{name}")
        return written

    # ------------------------------------------------------------------ checkpoint participant
    def checkpoint_cut(self) -> dict:
        index = self.dir / "index.jsonl"
        if index.exists():
            fsync_file(index)
        for name in ("bricks.idx",):
            if (self.dir / name).exists():
                fsync_file(self.dir / name)
        fsync_dir(self.dir)
        return {"keyframes": self.k, "index_bytes": self.index_bytes}

    def checkpoint_state(self) -> dict:
        return {"k": self.k, "next_tick": self.next_tick, "bricks": {k: [list(e) for e in v] for k, v in self.bricks.items()},
                "digest": dict(self.digest), "live": sorted(self.live, key=key_order), "index_bytes": self.index_bytes}

    def restore_state(self, state: Mapping) -> None:
        self.k = int(state["k"])
        self.next_tick = int(state["next_tick"])
        self.bricks = {str(k): [list(e) for e in v] for k, v in state["bricks"].items()}
        self.digest = {str(k): str(v) for k, v in state["digest"].items()}
        self.live = set(state["live"])
        self.index_bytes = int(state["index_bytes"])

    def truncate_outputs(self, cut: Mapping) -> None:
        count = int(cut["keyframes"])
        for folder in (self.dir, self.chips_dir):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("kf_*.*")):
                stem = path.name.split(".")[0]
                try:
                    number = int(stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if number >= count:
                    path.unlink()
        attr = self.dir / "attr"
        if attr.is_dir():
            for path in sorted(attr.glob("*")):
                path.unlink()
            attr.rmdir()
        index = self.dir / "index.jsonl"
        length = int(cut["index_bytes"])
        if index.exists():
            if index.stat().st_size < length:
                raise ValueError(f"keyframes/index.jsonl is shorter than its checkpoint cut ({length} bytes)")
            with open(index, "r+b") as handle:
                handle.truncate(length)
        elif length:
            raise ValueError("keyframes/index.jsonl is missing but the checkpoint cut has keyframes")
        for path in sorted(self.dir.glob(".*.tmp")):
            path.unlink()
        if self.bricks or count:
            self._write_index()
        elif (self.dir / "bricks.idx").exists():
            (self.dir / "bricks.idx").unlink()


# =============================================================================== reader

class KeyframeReader:
    """Reads keyframes back: the state of one brick at a time is one ranged read (``os.pread``) of the
    file bricks.idx names."""

    def __init__(self, run_dir) -> None:
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "keyframes"
        self._entries: list[dict] | None = None
        self._index: dict | None = None

    def entries(self) -> list[dict]:
        if self._entries is None:
            out = []
            path = self.dir / "index.jsonl"
            if path.is_file():
                for raw in path.read_bytes().splitlines(keepends=True):
                    if raw.endswith(b"\n") and raw.strip():
                        out.append(json.loads(raw))
            self._entries = sorted(out, key=lambda e: int(e["k"]))
        return self._entries

    def index(self) -> dict:
        if self._index is None:
            path = self.dir / "bricks.idx"
            self._index = read_json(path)["bricks"] if path.is_file() else {}
        return self._index

    def tick_of(self, k: int) -> int:
        for entry in self.entries():
            if int(entry["k"]) == int(k):
                return int(entry["tick"])
        raise KeyError(f"no keyframe {k}")

    def keyframe_at(self, tick: int | None = None, time_s: float | None = None) -> dict | None:
        """The newest keyframe at or before ``tick`` (or ``time_s``; None: the last one)."""
        entries = self.entries()
        if not entries:
            return None
        limit = self._limit(tick, time_s)
        chosen = None
        for entry in entries:
            if limit is None or int(entry["tick"]) <= limit:
                chosen = entry
        return chosen

    def playback(self, tick: int | None = None, time_s: float | None = None) -> list[int]:
        """Keyframes to load, in order, to show the workpiece at ``tick`` from the GLB files alone: the
        newest full snapshot at or before it and every keyframe after that up to it."""
        limit = self._limit(tick, time_s)
        chosen = [e for e in self.entries() if limit is None or int(e["tick"]) <= limit]
        start = 0
        for i, entry in enumerate(chosen):
            if entry.get("full"):
                start = i
        return [int(e["k"]) for e in chosen[start:]]

    def _limit(self, tick, time_s):
        if tick is not None:
            return int(tick)
        if time_s is not None:
            return int(round(float(time_s) / TICK_S))
        return None

    def brick_entry(self, key, tick: int | None = None, time_s: float | None = None):
        """[k, file, offset, length] of the brick's newest version at or before the time (None if none)."""
        entries = self.index().get(brick_key(key))
        if not entries:
            return None
        limit = self._limit(tick, time_s)
        if limit is None:
            return entries[-1]
        ticks = {int(e["k"]): int(e["tick"]) for e in self.entries()}
        chosen = None
        for entry in entries:
            if ticks.get(int(entry[0]), limit + 1) <= limit:
                chosen = entry
        return chosen

    def brick_bytes(self, key, tick: int | None = None, time_s: float | None = None) -> bytes | None:
        entry = self.brick_entry(key, tick, time_s)
        if entry is None:
            return None
        k, name, offset, length = entry
        path = self.run_dir / name if str(name).startswith("keyframes/") else self.dir / name
        fd = os.open(str(path), os.O_RDONLY)
        try:
            data = os.pread(fd, int(length), int(offset))
        finally:
            os.close(fd)
        if len(data) != int(length):
            raise ValueError(f"{name} is shorter than bricks.idx says (brick {key})")
        return data

    def brick(self, key, tick: int | None = None, time_s: float | None = None):
        """(vertices [n, 3], faces [m, 3]) of the brick at the time; None if it had no surface then."""
        data = self.brick_bytes(key, tick, time_s)
        if data is None:
            return None
        vertices, faces, header = decode_record(data)
        return None if header["n_vertices"] == 0 else (vertices, faces)

    def scene_at(self, tick: int | None = None, time_s: float | None = None) -> dict[str, tuple]:
        """Every brick with a surface at the time: key -> (vertices, faces)."""
        out = {}
        for key in sorted(self.index(), key=key_order):
            mesh = self.brick(key, tick, time_s)
            if mesh is not None:
                out[key] = mesh
        return out

    def fields(self, k: int) -> dict[str, np.ndarray]:
        with np.load(self.dir / f"kf_{k}.npz", allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def chips(self, k: int) -> dict[str, np.ndarray]:
        path = self.run_dir / "chips" / f"kf_{k}.npz"
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}


__all__ = ["KeyframeWriter", "KeyframeReader", "FieldGrid", "brick_record", "decode_record", "brick_glb", "quantise",
           "coarsen", "brick_key", "key_order", "npz_bytes", "BRICK_MAGIC", "RECORD_HEADER", "QUANT_MAX"]
