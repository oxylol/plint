"""Synthetic run directory (docs/architecture.md 16), deterministic from a seed.

The API, UI and report tests run against this while the simulation core is being written: every
file a run writes, with plausible content (a 16 x 12 mm pocket in a 40 x 30 x 15 mm Al 6061-T6
block, D10 4-flute end mill at about 10 000 rpm on a Haas VF-2SS, about 2.3 s simulated), the same
bytes for the same seed, also across a checkpoint and resume (:meth:`SyntheticRun.resume`).

The numbers are synthetic signals with the right shape (tooth-passing forces from the Altintas
mechanistic form, first-order heating, linear wear), not model output; nothing here is a model.

Layouts (what the API reads; docs/api.md "Run directory as read by the API"):

* ``run.json``      ``{format: cnctwin-run, project, inputs, versions, numba_cache_key, threads, clock,
                     verdict {safe, reasons}, summary, outputs [{path, bytes, sha256}]}`` (written last)
* ``status.json``   ``{format: cnctwin-run-status, status, progress {sim_time_s, est_total_s, wall_s,
                     eta_s}}`` (only when ``write_status``; the API job runner writes it otherwise)
* ``findings.json`` ``{format: cnctwin-findings, findings: [Finding.to_dict()]}``, rewritten as it grows
* ``trace.jsonl``   cnctwin.core.trace.TraceLog lines (id = line index)
* ``channels.json`` ``{format: cnctwin-channels, channels: [{name, unit, tier, domain, doc, max_shape,
                     dtype}]}``
* ``timeline.json`` ``{format: cnctwin-timeline, blocks: [{index, op, program, line, block, path,
                     kind, tick0, tick1, part, tool, ticks}]}`` (executed blocks so far; ``ticks`` is the
                     block's time by category, summing to tick1 - tick0: docs/models/production.md)
* ``series/<tier>/part-<n>.parquet`` columns ``tick`` (int64), ``time_s``, then channel columns;
  ``series/index.jsonl`` one line per closed part ``{tier, part, file, tick0, tick1, t0_s, t1_s, rows}``
* ``scene.json`` + ``scene/*.glb``; ``scene.json["frames"]`` is the frames.bin header
  ``{record_bytes, fields [{name, dtype, shape, offset}], axes, paths, tick_s, ipo_ticks}``
* ``frames.bin``    packed little-endian records, one per L1 tick
* ``keyframes/index.jsonl`` ``{k, tick, time_s, glb, fields, chips, full, bricks}`` per keyframe;
  ``keyframes/kf_<k>.glb`` glTF 2.0 binary, one node per changed brick, each brick's bytes a
  self-describing record (``<4sIIffff``: b"BRK1", n_vertices, n_indices, origin_S_m[3], cell_m, then
  uint16 positions [n, 4] in cells and uint32 indices); ``keyframes/bricks.idx`` JSON
  ``{bricks: {"<brick>": [[k, file, offset, length], ...]}}``; ``keyframes/kf_<k>.npz`` coarse
  fields including the coolant jets (``jet_*`` arrays); ``keyframes/attr/brick_<b>.npz``
* ``chips/kf_<k>.npz`` particles in flight and the pile summary
* ``part_<op>.glb|stl|3mf``, ``part_<op>_unload.stl``, ``inspection.json``, ``surface.json``,
  ``tools.json``, ``production.json`` (cnctwin.production over this run directory, with the
  inline machine/coolant/tool data of :data:`PRODUCTION_PROJECT`), ``sld/T<n>.json``, ``report.html|pdf``,
  ``setup_sheet.html|pdf``, ``images/*.png``, ``video.mp4``, ``audio.wav``
* ``checkpoints/ck_<tick>.npz`` + ``.json`` (the .json, written last, is the commit marker; it holds
  the byte lengths of trace.jsonl and frames.bin, the keyframe, timeline and finding counts and the
  closed series files)

Every container is written with fixed metadata (zip members dated 1980-01-01, no PNG/PDF dates,
ffmpeg ``-fflags +bitexact -flags +bitexact``), so the result set hashes identically run to run.

Usage::

    from make_synthetic_run import make_synthetic_run, write_synthetic_project
    manifest = make_synthetic_run(run_dir, seed=1)

    python tests/fixtures/make_synthetic_run.py RUN_DIR [--seed N] [--until 1.2] [--no-media]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import wave
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnctwin.core.findings import Category, Finding, Severity  # noqa: E402
from cnctwin.core.trace import TraceLog  # noqa: E402

FORMAT_VERSION = 1

# ----------------------------------------------------------------------------- clock (architecture 5.1)
TICK_S = 1e-6
IPO_TICKS = 1000                     # 1 ms L1 tick
MACRO_IPO = 10                       # 10 ms macro step
MACRO_TICKS = IPO_TICKS * MACRO_IPO
L0_TICKS = 8                         # 8 us L0 stride (0.5 deg rule at 10 000 rpm)
L0_PER_TICK = IPO_TICKS // L0_TICKS
CHECKPOINT_TICKS = 1_000_000         # checkpoint (and series part) every simulated second
KEYFRAME_TICKS = 500_000             # recording.keyframe_period_s = 0.5
FULL_SNAPSHOT_EVERY = 20
FAST_WINDOW_REVS = 4                 # recording.fast_window_revs

# ----------------------------------------------------------------------------- synthetic setup
N_TEETH = 4
MAX_TEETH = 16                       # channel max_shape for per-tooth channels
TOOL_RADIUS_M = 0.005
AP_M = 0.002
STOCK_SIZE_M = (0.040, 0.030, 0.015)
LATTICE_M = 0.0005
BRICK_CELLS = 32
WORK_ZERO_S_M = (0.010, 0.009, 0.015)        # G54 zero in the stock frame (pocket corner, top face)
G54_M_M = (-0.3810, -0.2032, -0.4064)        # machine coordinates of the work zero
HOME_W_M = (0.050, 0.040, 0.050)             # where the program starts (work coordinates)
RAPID_M_S = 35.6 / 60.0
NOMINAL_RPM = 10000.0
KT_PA = 700e6                        # Al 6061-T6 order of magnitude (synthetic)
KR_RATIO = 0.3
KTE_N_PER_M = 20e3
TOOL_STIFFNESS_N_PER_M = 2.0e7
AVAILABLE_POWER_W = 15e3
AMBIENT_K = 293.15
EDGE_RISE_K = 180.0
EDGE_TAU_S = 0.25
VB0_M = 20e-6
VB_RATE_M_S = 4e-6
DENSITY_KG_M3 = 2700.0
SURFACE_CELL_M = 0.004
AUDIO_RATE_HZ = 8000
VIDEO_SIZE = (64, 48)
VIDEO_FPS = 10
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SOURCE_DATE_EPOCH = "315532800"

OP = "OP10"
PROGRAM = "op10.nc"
TOOL_NUMBER = 1
TOOL_ID = "sandvik-2p342-1000-pa"
HOLDER_ID = "haas-cat40-er32"
DEFAULT_SEED = 1234

# What cnctwin.production needs resolved for production.json (the run.json project names only ids):
# a subset of db/machines/haas-vf2ss.yaml, db/coolants/emulsion-8pct.yaml and the tool template's
# price and regrinds, inline so the fixture does not depend on the database files' current state.
PRODUCTION_PROJECT = {
    "machine": {
        "id": "haas-vf2ss", "type": "vertical_mill",
        "controller": {"dialect": "haas", "least_input_increment_mm": 0.001},
        "axes": {"X": {"type": "linear", "travel_mm": [-762, 0], "moving_mass_kg": 300},
                 "Y": {"type": "linear", "travel_mm": [-406, 0], "moving_mass_kg": 650},
                 "Z": {"type": "linear", "travel_mm": [-508, 0], "moving_mass_kg": 400}},
        "kinematics": {"workpiece_chain": [{"link": "saddle", "axis": "Y", "direction": [0, -1, 0]},
                                           {"link": "table", "axis": "X", "direction": [-1, 0, 0]}],
                       "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]}],
                       "home_mm": {"X": 0, "Y": 0, "Z": 0}},
        "spindles": [{"id": "main", "kind": "milling", "max_rpm": 12000, "rotor_inertia_kgm2": 0.05,
                      "efficiency": 0.85, "idle_power_W": 700}],
        "coolant": {"tank_l": 208, "pumps": [{"id": "flood", "kind": "flood", "pressure_bar": 2.0, "flow_l_min": 45,
                                              "power_kW": 0.75}],
                    "air_blast": {"flow_l_min": 170, "pressure_bar": 6.9}},
        "chip_conveyor": {"type": "auger"},
        "doors": {"count": 2, "auto": False},
        "power": {"base_W": 900, "servo_standby_W": 300, "air_consumption_l_min": 113},
    },
    "coolant": {"fluid": {"id": "emulsion-8pct", "kind": "emulsion", "concentration_pct": 8,
                          "drag_out_ml_per_kg_chips": 42, "evaporation_l_per_h_per_m2": 0.3,
                          "mist_loss_fraction": 2.0e-6, "cost_eur_per_l": 4.0},
                "concentration_pct": 8},
    "tools": {"list": [{"number": TOOL_NUMBER, "tool": {"id": TOOL_ID, "type": "square_endmill", "hand": "right",
                                                        "geometry": {"diameter_mm": 10, "flutes": 4},
                                                        "price_eur": 85, "regrinds": 2},
                        "stickout_mm": 35, "length_offset_mm": 120.53}]},
    "production": {"parts": 1},
}
BLOCK_CATEGORY = {"rapid": "rapid", "tool_change": "tool_change", "spindle": "spindle_ramp",
                  "coolant": "coolant_delay", "stop": "program_stop", "mode": "auxiliary"}


# ----------------------------------------------------------------------------- the program and its blocks

@dataclass(frozen=True)
class Block:
    line: int
    text: str
    kind: str                         # rapid | linear | spindle | tool_change | coolant | stop | mode
    end_W_mm: tuple | None = None     # target in work coordinates (mm), None = no motion
    feed_mm_min: float = 0.0
    dwell_s: float = 0.0
    ae_mm: float = 0.0                # radial immersion while cutting (0 = not a cutting block)


def program_blocks() -> list[Block]:
    return [
        Block(2, "O1000 (SYNTHETIC POCKET 16X12)", "mode", dwell_s=0.001),
        Block(3, "G21 G17 G40 G49 G80 G90", "mode", dwell_s=0.001),
        Block(4, "T1 M06 (D10 4FL CARBIDE)", "tool_change", dwell_s=0.15),
        Block(5, "G54", "mode", dwell_s=0.001),
        Block(6, "S10000 M03", "spindle", dwell_s=0.10),
        Block(7, "G00 X0. Y0.", "rapid", end_W_mm=(0.0, 0.0, HOME_W_M[2] * 1e3)),
        Block(8, "G43 H1 Z20. M08", "rapid", end_W_mm=(0.0, 0.0, 20.0)),
        Block(9, "G00 Z2.", "rapid", end_W_mm=(0.0, 0.0, 2.0)),
        Block(10, "G01 Z-2. F1200.", "linear", end_W_mm=(0.0, 0.0, -2.0), feed_mm_min=1200.0),
        Block(11, "G01 X16. F3000.", "linear", end_W_mm=(16.0, 0.0, -2.0), feed_mm_min=3000.0, ae_mm=10.0),
        Block(12, "G01 Y4.", "linear", end_W_mm=(16.0, 4.0, -2.0), feed_mm_min=3000.0, ae_mm=10.0),
        Block(13, "G01 X0.", "linear", end_W_mm=(0.0, 4.0, -2.0), feed_mm_min=3000.0, ae_mm=4.0),
        Block(14, "G01 Y8.", "linear", end_W_mm=(0.0, 8.0, -2.0), feed_mm_min=3000.0, ae_mm=10.0),
        Block(15, "G01 X16.", "linear", end_W_mm=(16.0, 8.0, -2.0), feed_mm_min=3000.0, ae_mm=4.0),
        Block(16, "G01 Y12.", "linear", end_W_mm=(16.0, 12.0, -2.0), feed_mm_min=3000.0, ae_mm=10.0),
        Block(17, "G01 X0.", "linear", end_W_mm=(0.0, 12.0, -2.0), feed_mm_min=3000.0, ae_mm=4.0),
        Block(18, "G00 Z20.", "rapid", end_W_mm=(0.0, 12.0, 20.0)),
        Block(19, "M09", "coolant", dwell_s=0.01),
        Block(20, "M05", "spindle", dwell_s=0.05),
        Block(21, "G91 G28 Z0.", "rapid", end_W_mm=(0.0, 12.0, HOME_W_M[2] * 1e3)),
        Block(22, "M30", "stop", dwell_s=0.001),
    ]


def program_text() -> str:
    lines = {1: "%"}
    for block in program_blocks():
        lines[block.line] = block.text
    last = max(lines) + 1
    lines[last] = "%"
    return "\n".join(lines[i] for i in range(1, last + 1)) + "\n"


SPINDLE_ON_LINE = 6
SPINDLE_OFF_LINE = 20
COOLANT_ON_LINE = 8
COOLANT_OFF_LINE = 19


@dataclass
class Schedule:
    """Block timing in base ticks: block i runs [tick0[i], tick1[i])."""

    blocks: list
    tick0: np.ndarray
    tick1: np.ndarray
    start_W_m: np.ndarray             # [n_blocks, 3]
    end_W_m: np.ndarray
    end_tick: int


def make_schedule() -> Schedule:
    blocks = program_blocks()
    tick = 0
    pos = np.array(HOME_W_M, float)
    t0, t1, starts, ends = [], [], [], []
    for block in blocks:
        start = pos.copy()
        if block.end_W_mm is not None:
            end = np.array(block.end_W_mm, float) * 1e-3
            speed = RAPID_M_S if block.kind == "rapid" else block.feed_mm_min * 1e-3 / 60.0
            duration_s = float(np.linalg.norm(end - start)) / speed
        else:
            end = start.copy()
            duration_s = block.dwell_s
        ticks = max(1, int(round(duration_s / (IPO_TICKS * TICK_S)))) * IPO_TICKS
        t0.append(tick)
        tick += ticks
        t1.append(tick)
        starts.append(start)
        ends.append(end)
        pos = end
    end_tick = int(math.ceil(tick / MACRO_TICKS)) * MACRO_TICKS
    t1[-1] = end_tick                  # the last block absorbs the padding to the macro boundary
    return Schedule(blocks, np.array(t0, np.int64), np.array(t1, np.int64), np.array(starts), np.array(ends),
                    end_tick)


# ----------------------------------------------------------------------------- deterministic containers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_bytes_atomic(path, (json.dumps(obj, indent=1, sort_keys=False) + "\n").encode("utf-8"))


def write_npz(path: Path, arrays: dict) -> None:
    """Like numpy.savez, but with fixed member dates (numpy stamps the wall clock)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in arrays:
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    write_bytes_atomic(path, buffer.getvalue())


def box_mesh(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.uint32)
    return corners, faces


def prism_mesh(radius: float, z0: float, z1: float, sides: int = 8) -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(sides) * 2.0 * math.pi / sides
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    vertices = np.concatenate([np.column_stack([ring, np.full(sides, z0)]), np.column_stack([ring, np.full(sides, z1)]),
                               [[0.0, 0.0, z0], [0.0, 0.0, z1]]])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[i, j, sides + j], [i, sides + j, sides + i], [2 * sides, j, i], [2 * sides + 1, sides + i, sides + j]]
    return vertices, np.array(faces, np.uint32)


def _pad4(data: bytearray, fill: bytes) -> None:
    while len(data) % 4:
        data += fill


def glb_bytes(meshes: list[dict], generator: str = "cnctwin synthetic run") -> tuple[bytes, list[tuple[int, int]]]:
    """glTF 2.0 binary. Each mesh: {name, positions float32 [n,3] | uint16 [n,4], indices uint32 [m],
    translation, scale, header bytes}. Returns the file and each mesh's (offset, length) in the file
    (header + positions + indices)."""
    binary = bytearray()
    views, accessors, gl_meshes, nodes, spans = [], [], [], [], []
    quantised = False
    for mesh in meshes:
        start = len(binary)
        if mesh.get("header"):
            binary += mesh["header"]
            _pad4(binary, b"\x00")
        positions = np.ascontiguousarray(mesh["positions"])
        indices = np.ascontiguousarray(mesh["indices"], dtype="<u4").reshape(-1)
        if positions.dtype == np.uint16:
            quantised = True
            pos_bytes = positions.astype("<u2").tobytes()
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(pos_bytes), "byteStride": 8,
                          "target": 34962})
            mins = positions[:, :3].min(axis=0).astype(int).tolist()
            maxs = positions[:, :3].max(axis=0).astype(int).tolist()
            accessors.append({"bufferView": len(views) - 1, "componentType": 5123, "count": int(len(positions)),
                              "type": "VEC3", "min": mins, "max": maxs})
        else:
            pos32 = positions.astype("<f4")
            pos_bytes = pos32.tobytes()
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(pos_bytes), "target": 34962})
            accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "count": int(len(pos32)),
                              "type": "VEC3", "min": [float(v) for v in pos32.min(axis=0)],
                              "max": [float(v) for v in pos32.max(axis=0)]})
        binary += pos_bytes
        _pad4(binary, b"\x00")
        index_bytes = indices.tobytes()
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(index_bytes), "target": 34963})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5125, "count": int(len(indices)),
                          "type": "SCALAR"})
        binary += index_bytes
        _pad4(binary, b"\x00")
        gl_meshes.append({"name": mesh["name"], "primitives": [{"attributes": {"POSITION": len(accessors) - 2},
                                                                 "indices": len(accessors) - 1, "mode": 4}]})
        node = {"name": mesh["name"], "mesh": len(gl_meshes) - 1}
        if mesh.get("translation") is not None:
            node["translation"] = [float(v) for v in mesh["translation"]]
        if mesh.get("scale") is not None:
            node["scale"] = [float(v) for v in mesh["scale"]]
        nodes.append(node)
        spans.append((start, len(binary) - start))
    document = {"asset": {"version": "2.0", "generator": generator}, "scene": 0,
                "scenes": [{"nodes": list(range(len(nodes)))}]}
    if nodes:
        document.update({"nodes": nodes, "meshes": gl_meshes, "accessors": accessors, "bufferViews": views,
                         "buffers": [{"byteLength": len(binary)}]})
    if quantised:
        document["extensionsUsed"] = ["KHR_mesh_quantization"]
        document["extensionsRequired"] = ["KHR_mesh_quantization"]
    json_chunk = bytearray(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    _pad4(json_chunk, b" ")
    header_bytes = 12 + 8 + len(json_chunk) + (8 if binary else 0)
    total = header_bytes + len(binary)
    out = bytearray(struct.pack("<4sII", b"glTF", 2, total))
    out += struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
    if binary:
        out += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    return bytes(out), [(header_bytes + start, length) for start, length in spans]


def stl_bytes(vertices: np.ndarray, faces: np.ndarray, title: str) -> bytes:
    header = title.encode("ascii")[:80].ljust(80, b" ")
    out = bytearray(header + struct.pack("<I", len(faces)))
    for face in faces:
        a, b, c = (np.asarray(vertices[i], float) for i in face)
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        normal = normal / norm if norm > 0 else normal
        out += struct.pack("<12fH", *normal, *a, *b, *c, 0)
    return bytes(out)


def threemf_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    mm = np.asarray(vertices) * 1e3
    verts = "".join(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in mm)
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces)
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>'
             f'<object id="1" type="model"><mesh><vertices>{verts}</vertices><triangles>{tris}</triangles>'
             '</mesh></object></resources><build><item objectid="1"/></build></model>\n')
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/'
                     'package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.'
                     'openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="'
                     'application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://'
            'schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>\n')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in (("[Content_Types].xml", content_types), ("_rels/.rels", rels), ("3D/3dmodel.model", model)):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, text)
    return buffer.getvalue()


def pdf_bytes(lines: list[str]) -> bytes:
    """A one-page PDF 1.4 with the given text lines (Helvetica), no dates."""
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = ["BT", "/F1 11 Tf", "50 790 Td", "14 TL"]
    for text in lines:
        stream.append(f"({escape(text)}) Tj T*")
    stream.append("ET")
    content = "\n".join(stream).encode("latin-1", "replace")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> "
               b"/Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def wav_bytes(samples: np.ndarray, rate_hz: int) -> bytes:
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate_hz)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    """MPEG-4 Part 2 video through imageio-ffmpeg's ffmpeg, bit-exact, moov first (seekable streaming)."""
    import imageio_ffmpeg
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.mp4")
    height, width = frames.shape[1:3]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p", "-threads", "1",
               "-fflags", "+bitexact", "-flags", "+bitexact", "-map_metadata", "-1",
               "-movflags", "+faststart", "-f", "mp4", "-y", str(tmp)]
    env = dict(os.environ, SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH)
    result = subprocess.run(command, input=np.ascontiguousarray(frames, np.uint8).tobytes(), capture_output=True,
                            env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')}")
    os.replace(tmp, path)


def zip_dir_bytes(directory: Path) -> bytes:
    """Deterministic zip of a directory tree (sorted members, fixed dates)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(directory).as_posix(), date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


# ----------------------------------------------------------------------------- the synthetic project

def project_document(seed: int, name: str = "Synthetic pocket, Al 6061-T6") -> str:
    return f"""# Synthetic project for the API, UI and report tests (tests/fixtures/make_synthetic_run.py)
format: cnctwin-project
version: 1
name: "{name}"
description: "16 x 12 mm pocket, 2 mm deep, in a 40 x 30 x 15 mm block (synthetic)"
seed: {seed}
machine: haas-vf2ss
controller_overrides: {{}}
material:
  id: al6061-t6
environment:
  ambient_degC: 20
  coolant_degC: 22
  machine_state: warm
  time_between_ops_min: 10
stock:
  kind: block
  size_mm: [40, 30, 15]
  tolerance_mm: [0.1, 0.1, 0.1]
  form: rolled_plate
cad:
  path: null
  units: mm
  tolerances:
    - {{id: pocket-depth, kind: distance, nominal_mm: 2, plus_mm: 0.05, minus_mm: 0.05}}
    - {{id: floor-finish, kind: roughness, Ra_um: 1.6}}
tools:
  library: null
  list:
    - number: {TOOL_NUMBER}
      tool: {TOOL_ID}
      holder: {HOLDER_ID}
      stickout_mm: 35
      length_offset_mm: 120.53
      radius_offset_mm: 5.0
      runout_um: 5
      comment: "D10 4FL CARBIDE"
coolant:
  fluid: emulsion-8pct
  concentration_pct: 8
operations:
  - name: {OP}
    program: {PROGRAM}
    setup:
      workholding:
        - {{device: lang-makro-grip-77, pose: {{translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}}}}
      stock_pose: {{translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}}
      work_offsets_mm: {{G54: [-381.0, -203.2, -406.4]}}
production:
  parts: 1
recording:
  fast: windows
  fast_window_revs: {FAST_WINDOW_REVS}
  keyframe_period_s: 0.5
outputs:
  report: [html, pdf]
  images: true
  video: {{fps: {VIDEO_FPS}, speed: 1}}
  audio: true
"""


def write_synthetic_project(directory, seed: int = DEFAULT_SEED, name: str = "Synthetic pocket, Al 6061-T6") -> Path:
    """A project folder (project.yaml + op10.nc) the synthetic run belongs to. Returns project.yaml."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "project.yaml").write_text(project_document(seed, name), encoding="utf-8")
    (directory / PROGRAM).write_text(program_text(), encoding="utf-8")
    return directory / "project.yaml"


def seed_from_project(project_path) -> int:
    for line in Path(project_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("seed:"):
            return int(line.split(":", 1)[1].strip())
    return DEFAULT_SEED


# ----------------------------------------------------------------------------- channels

@dataclass(frozen=True)
class Chan:
    name: str
    unit: str
    tier: str
    domain: str
    doc: str
    max_shape: tuple = ()
    dtype: str = "float64"

    def columns(self) -> list[str]:
        if not self.max_shape:
            return [self.name]
        return [f"{self.name}[{i}]" for i in range(int(np.prod(self.max_shape)))]

    def to_dict(self) -> dict:
        return {"name": self.name, "unit": self.unit, "tier": self.tier, "domain": self.domain, "doc": self.doc,
                "max_shape": list(self.max_shape), "dtype": self.dtype}


FAST_CHANNELS = (
    Chan("cutting.force_N", "N", "fast", "cutting", "cutting force on the tool in T0 (x, y, z)", (3,)),
    Chan("cutting.torque_Nm", "Nm", "fast", "cutting", "cutting torque about the tool axis"),
    Chan("cutting.chip_thickness_m", "m", "fast", "cutting", "uncut chip thickness per tooth", (MAX_TEETH,)),
    Chan("dynamics.tool_displacement_m", "m", "fast", "dynamics", "tool tip deflection in T0 (x, y)", (2,)),
)
TICK_CHANNELS = (
    Chan("motion.feed_m_s", "m/s", "tick", "motion", "tool-tip path speed"),
    Chan("motion.following_error_m", "m", "tick", "motion", "following error per axis (X, Y, Z)", (3,)),
    Chan("cutting.spindle_power_W", "W", "tick", "cutting", "spindle cutting power"),
    Chan("cutting.spindle_load_fraction", "-", "tick", "cutting", "demand over available spindle power"),
)
MACRO_CHANNELS = (
    Chan("thermal.edge_temp_K", "K", "macro", "thermal", "hottest cutting-edge temperature"),
    Chan("thermal.work_bulk_mean_K", "K", "macro", "thermal", "mean workpiece temperature"),
    Chan("wear.VB_m", "m", "macro", "wear", "flank wear land width per tooth", (MAX_TEETH,)),
    Chan("chips.flute_fill", "-", "macro", "chips", "chip volume in the flutes over flute volume"),
    Chan("coolant.reach_fraction", "-", "macro", "coolant", "flow-weighted share of coolant reaching the cut"),
    Chan("removal.volume_consistency", "-", "macro", "removal", "field volume over analytic chip volume"),
    Chan("removal.removed_volume_m3", "m3", "macro", "removal", "material removed since the start"),
)
ALL_CHANNELS = FAST_CHANNELS + TICK_CHANNELS + MACRO_CHANNELS


def tier_columns(tier: str) -> list[str]:
    columns = ["tick", "time_s"]
    if tier == "fast":
        for chan in sorted(FAST_CHANNELS, key=lambda c: c.name):
            columns += chan.columns()
    elif tier == "tick":
        plain = [c for chan in sorted(TICK_CHANNELS, key=lambda c: c.name) for c in chan.columns()]
        envelope = [f"{c}.{s}" for chan in sorted(FAST_CHANNELS, key=lambda c: c.name) for c in chan.columns()
                    for s in ("min", "mean", "max")]
        columns += plain + envelope
    else:
        for chan in sorted(MACRO_CHANNELS, key=lambda c: c.name):
            columns += chan.columns()
    return columns


# ----------------------------------------------------------------------------- the run

FRAME_DTYPE = np.dtype([("tick", "<i8"), ("axis", "<f4", (3,)), ("spindle_angle_rad", "<f4", (1,)),
                        ("spindle_rev", "<i4", (1,)), ("tip_S_m", "<f4", (1, 3)), ("axis_T_S", "<f4", (1, 3))])


def frames_header() -> dict:
    fields = []
    for name in FRAME_DTYPE.names:
        sub, offset = FRAME_DTYPE.fields[name][:2]
        base = sub.base if sub.shape else sub
        fields.append({"name": name, "dtype": base.str, "shape": list(sub.shape), "offset": int(offset)})
    units = {"tick": "tick", "axis": "m", "spindle_angle_rad": "rad", "spindle_rev": "rev", "tip_S_m": "m",
             "axis_T_S": "-"}
    for entry in fields:
        entry["unit"] = units[entry["name"]]
    return {"format": "cnctwin-frames", "version": FORMAT_VERSION, "file": "frames.bin",
            "record_bytes": FRAME_DTYPE.itemsize, "byte_order": "little", "tick_s": TICK_S, "ipo_ticks": IPO_TICKS,
            "axes": [{"name": "X", "kind": "linear", "unit": "m"}, {"name": "Y", "kind": "linear", "unit": "m"},
                     {"name": "Z", "kind": "linear", "unit": "m"}],
            "paths": [{"path": 0, "spindle": "main"}], "fields": fields,
            "doc": "axis: X Y Z relative to the stock origin; spindle angle wrapped to [0, 2 pi) with its "
                   "revolution count; tip_S_m and axis_T_S per path"}


@dataclass
class _Planned:
    tick: int
    order: int
    event: str
    domain: str
    line: int
    quantities: dict
    parents: tuple = ()
    finding: dict | None = None


class SyntheticRun:
    """Writes the run directory step by step (one macro step per :meth:`step`)."""

    def __init__(self, run_dir, seed: int = DEFAULT_SEED, *, write_status: bool = True, media: bool = True,
                 project_path=None):
        self.dir = Path(run_dir)
        self.seed = int(seed)
        self.write_status = write_status
        self.media = media
        self.project_path = Path(project_path) if project_path is not None else None
        self.schedule = make_schedule()
        rng = self._rng("setup")
        self.rpm = NOMINAL_RPM * (1.0 + 0.1 * float(rng.uniform(-1.0, 1.0)))
        self.omega_rad_s = -self.rpm * 2.0 * math.pi / 60.0          # M03: negative about +z_T
        self.kt_pa = KT_PA * (1.0 + 0.05 * float(rng.uniform(-1.0, 1.0)))
        self.chatter_freq_hz = 1800.0 + 200.0 * float(rng.uniform(-1.0, 1.0))
        self.include_error = (self.seed % 2 == 0)
        self.end_tick = self.schedule.end_tick
        self.macro = 0
        self.keyframes = 0
        self.part_counter = {"fast": 0, "tick": 0, "macro": 0}
        self.closed_series: dict[str, list[str]] = {"fast": [], "tick": [], "macro": []}
        self.rows: dict[str, list[dict]] = {"fast": [], "tick": [], "macro": []}
        self.findings: list[dict] = []
        self.timeline_count = 0
        self.bricks_index: dict[str, list] = {}
        self.last_checkpoint: dict | None = None
        self.trace: TraceLog | None = None
        self.plan = self._plan_events()
        self.plan_cursor = 0
        self.finished = False
        self._spindle_on = int(self._block_end(SPINDLE_ON_LINE))
        self._spindle_off = int(self._block_start(SPINDLE_OFF_LINE))
        self._coolant_on = int(self._block_end(COOLANT_ON_LINE))
        self._coolant_off = int(self._block_start(COOLANT_OFF_LINE))
        self._ae_m = np.array([b.ae_mm for b in self.schedule.blocks]) * 1e-3
        self._is_plunge = np.array([b.line == 10 for b in self.schedule.blocks])

    # ------------------------------------------------------------------ deterministic draws
    def _rng(self, name: str, *counters: int) -> np.random.Generator:
        key = zlib.crc32(name.encode("utf-8"))
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed & 0xFFFFFFFF, key,
                                                                              *[int(c) for c in counters]])))

    # ------------------------------------------------------------------ schedule helpers
    def _block_index(self, line: int) -> int:
        for index, block in enumerate(self.schedule.blocks):
            if block.line == line:
                return index
        raise KeyError(line)

    def _block_start(self, line: int) -> int:
        return int(self.schedule.tick0[self._block_index(line)])

    def _block_end(self, line: int) -> int:
        return int(self.schedule.tick1[self._block_index(line)])

    @property
    def time_s(self) -> float:
        return self.macro * MACRO_TICKS * TICK_S

    @property
    def total_s(self) -> float:
        return self.end_tick * TICK_S

    def block_at(self, ticks: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.schedule.tick0, ticks, side="right") - 1, 0,
                       len(self.schedule.blocks) - 1)

    def tip_W(self, ticks: np.ndarray) -> np.ndarray:
        ticks = np.asarray(ticks, np.int64)
        index = self.block_at(ticks)
        t0 = self.schedule.tick0[index]
        t1 = self.schedule.tick1[index]
        frac = np.clip((ticks - t0) / np.maximum(t1 - t0, 1), 0.0, 1.0)[:, None]
        start = self.schedule.start_W_m[index]
        end = self.schedule.end_W_m[index]
        return start + frac * (end - start)

    def tip_S(self, ticks) -> np.ndarray:
        return self.tip_W(ticks) + np.array(WORK_ZERO_S_M)

    def tip_M(self, ticks) -> np.ndarray:
        return self.tip_W(ticks) + np.array(G54_M_M)

    def feed_speed(self, ticks: np.ndarray) -> np.ndarray:
        index = self.block_at(ticks)
        delta = self.schedule.end_W_m[index] - self.schedule.start_W_m[index]
        duration = (self.schedule.tick1[index] - self.schedule.tick0[index]) * TICK_S
        return np.linalg.norm(delta, axis=1) / duration

    def velocity_W(self, ticks: np.ndarray) -> np.ndarray:
        index = self.block_at(ticks)
        delta = self.schedule.end_W_m[index] - self.schedule.start_W_m[index]
        duration = (self.schedule.tick1[index] - self.schedule.tick0[index]) * TICK_S
        return delta / duration[:, None]

    def cut_ae(self, ticks: np.ndarray) -> np.ndarray:
        """Radial immersion (m) where the tool is cutting, 0 elsewhere (the plunge cuts axially)."""
        return self._ae_m[self.block_at(ticks)]

    def plunging(self, ticks: np.ndarray) -> np.ndarray:
        return self._is_plunge[self.block_at(ticks)] & (self.tip_W(ticks)[:, 2] < 0.0)

    def spindle_angle(self, ticks: np.ndarray) -> np.ndarray:
        """Unwrapped spindle angle (rad, right-handed about +z_T)."""
        running = np.clip(np.asarray(ticks, np.int64), self._spindle_on, self._spindle_off) - self._spindle_on
        return self.omega_rad_s * running * TICK_S

    def omega(self, ticks: np.ndarray) -> np.ndarray:
        ticks = np.asarray(ticks)
        return np.where((ticks >= self._spindle_on) & (ticks < self._spindle_off), self.omega_rad_s, 0.0)

    def cut_time_s(self, tick: int) -> float:
        """Cumulative time in cut (cutting blocks) up to ``tick``."""
        total = 0.0
        for index, block in enumerate(self.schedule.blocks):
            if block.ae_mm <= 0.0:
                continue
            t0, t1 = int(self.schedule.tick0[index]), int(self.schedule.tick1[index])
            total += max(0, min(tick, t1) - t0) * TICK_S
        return total

    def removed_volume_m3(self, tick: int) -> float:
        volume = 0.0
        for index, block in enumerate(self.schedule.blocks):
            if block.ae_mm <= 0.0:
                continue
            t0, t1 = int(self.schedule.tick0[index]), int(self.schedule.tick1[index])
            seconds = max(0, min(tick, t1) - t0) * TICK_S
            volume += block.ae_mm * 1e-3 * AP_M * block.feed_mm_min * 1e-3 / 60.0 * seconds
        return volume

    # ------------------------------------------------------------------ L0 signals
    def l0(self, ticks: np.ndarray, noise_counter: int) -> dict:
        """Forces, torque, chip thickness and deflection at L0 sample ticks."""
        ticks = np.asarray(ticks, np.int64)
        n = len(ticks)
        ae = self.cut_ae(ticks)
        cutting = ae > 0.0
        phi = np.abs(self.spindle_angle(ticks))
        feed = self.feed_speed(ticks)
        fz = feed / (N_TEETH * self.rpm / 60.0)
        phi_start = math.pi - np.arccos(np.clip(1.0 - ae / TOOL_RADIUS_M, -1.0, 1.0))
        force = np.zeros((n, 3))
        torque = np.zeros(n)
        chip = np.full((n, MAX_TEETH), np.nan)
        chip[:, :N_TEETH] = 0.0
        for tooth in range(N_TEETH):
            angle = np.mod(phi + tooth * 2.0 * math.pi / N_TEETH, 2.0 * math.pi)
            engaged = cutting & (angle >= phi_start) & (angle <= math.pi)
            h = np.where(engaged, fz * np.sin(angle), 0.0)
            ft = np.where(engaged, self.kt_pa * AP_M * h + KTE_N_PER_M * AP_M, 0.0)
            fr = KR_RATIO * ft
            force[:, 0] += -ft * np.cos(angle) - fr * np.sin(angle)
            force[:, 1] += ft * np.sin(angle) - fr * np.cos(angle)
            force[:, 2] += 0.15 * ft
            torque += ft * TOOL_RADIUS_M
            chip[:, tooth] = h
        plunge = self.plunging(ticks)
        force[:, 2] += np.where(plunge, 150.0, 0.0)
        torque += np.where(plunge, 0.4, 0.0)
        rng = self._rng("l0-noise", noise_counter)
        active = (cutting | plunge)[:, None]
        force += np.where(active, rng.normal(0.0, 2.0, size=(n, 3)), 0.0)
        displacement = force[:, :2] / TOOL_STIFFNESS_N_PER_M
        return {"cutting.force_N": force, "cutting.torque_Nm": torque, "cutting.chip_thickness_m": chip,
                "dynamics.tool_displacement_m": displacement}

    def fast_windows(self) -> list[tuple[int, int]]:
        """The first FAST_WINDOW_REVS revolutions of each engagement (cutting block start)."""
        revolution_ticks = int(round(2.0 * math.pi / abs(self.omega_rad_s) / TICK_S))
        windows = []
        for index, block in enumerate(self.schedule.blocks):
            if block.ae_mm > 0.0:
                start = int(self.schedule.tick0[index])
                windows.append((start, min(start + FAST_WINDOW_REVS * revolution_ticks, int(self.schedule.tick1[index]))))
        return windows

    # ------------------------------------------------------------------ planned trace events and findings
    def _plan_events(self) -> list[_Planned]:
        plan: list[_Planned] = []
        order = 0

        def add(tick, event, domain, line, quantities, parents=(), finding=None):
            nonlocal order
            plan.append(_Planned(int(tick), order, event, domain, line, quantities, tuple(parents), finding))
            order += 1

        rng = self._rng("plan")
        spindle_on = self._block_end(SPINDLE_ON_LINE)
        add(spindle_on, "spindle_on", "motion", SPINDLE_ON_LINE, {"omega_rad_s": self.omega_rad_s})
        add(self._block_end(COOLANT_ON_LINE), "coolant_on", "coolant", COOLANT_ON_LINE, {"flow_m3_s": 2.5e-4})
        plunge_start = self._block_start(10)
        add(plunge_start, "air_feed", "removal", 10, {"air_cut_time_s": 0.1}, ["spindle_on"],
            finding={"category": Category.AIR_CUTTING, "severity": Severity.INFO,
                     "message": "Feed motion out of contact for 0.100 s before the tool reaches the stock"})
        for index, block in enumerate(self.schedule.blocks):
            if block.ae_mm <= 0.0:
                continue
            t0, t1 = int(self.schedule.tick0[index]), int(self.schedule.tick1[index])
            add(t0, f"engagement:{block.line}", "removal", block.line,
                {"ae_m": block.ae_mm * 1e-3, "ap_m": AP_M}, ["spindle_on"])
            for tick in range(t0 + 100_000, t1, 100_000):
                add(tick, f"force_peak:{tick}", "cutting", block.line,
                    {"force_N": float(250.0 + 40.0 * rng.uniform())}, [f"engagement:{block.line}"])
        slot = self._block_index(11)
        t_pack = int(self.schedule.tick0[slot] + 0.6 * (self.schedule.tick1[slot] - self.schedule.tick0[slot]))
        t_pack -= t_pack % IPO_TICKS
        add(t_pack, "flute_fill", "chips", 11, {"flute_fill": 0.72}, ["engagement:11", "coolant_on"])
        add(t_pack, "finding:chip_packing", "chips", 11, {"flute_fill": 0.72}, ["flute_fill"],
            finding={"category": Category.CHIP_PACKING, "severity": Severity.WARNING,
                     "message": "Flutes 72 % full while slotting: chips may pack and break the tool"})
        side = self._block_index(13)
        t_chat = int((self.schedule.tick0[side] + self.schedule.tick1[side]) // 2)
        t_chat -= t_chat % IPO_TICKS
        index_value = float(0.35 + 0.1 * rng.uniform())
        add(t_chat, "modal_energy", "dynamics", 13, {"chatter_index": index_value}, [self._last_force_before(plan, t_chat)])
        add(t_chat, "finding:chatter", "dynamics", 13, {"chatter_freq_Hz": self.chatter_freq_hz,
                                                       "chatter_index": index_value}, ["modal_energy"],
            finding={"category": Category.CHATTER, "severity": Severity.WARNING,
                     "message": f"Chatter at {self.chatter_freq_hz:.0f} Hz (index {index_value:.2f}) at ae 4.0 mm"})
        exit_block = self._block_index(15)
        t_burr = int(self.schedule.tick1[exit_block]) - 20 * IPO_TICKS
        add(t_burr, "exit_angle", "cutting", 15, {"exit_angle_rad": math.pi}, ["engagement:15"])
        add(t_burr, "finding:burr", "cutting", 15, {"burr_height_m": 4.0e-5}, ["exit_angle"],
            finding={"category": Category.BURR, "severity": Severity.INFO,
                     "message": "Exit burr about 0.04 mm on the pocket wall at X16"})
        if self.include_error:
            floor_block = self._block_index(17)
            t_tol = int(self.schedule.tick1[floor_block]) - IPO_TICKS
            add(1_000_000, "thermal_growth", "thermal", 0, {"length_growth_m": 1.2e-5},
                [self._last_force_before(plan, 1_000_000)])
            add(t_tol, "floor_position", "inspect", 17, {"deviation_m": 6.2e-5},
                ["thermal_growth", self._last_force_before(plan, t_tol)])
            add(t_tol, "finding:out_of_tolerance", "inspect", 17,
                {"measured_m": 0.002062, "nominal_m": 0.002, "deviation_m": 6.2e-5}, ["floor_position"],
                finding={"category": Category.OUT_OF_TOLERANCE, "severity": Severity.ERROR,
                         "message": "pocket-depth 2.062 mm, tolerance 2 +/- 0.05 mm (tool growth 12 um)"})
        plan.sort(key=lambda p: (p.tick, p.order))
        return plan

    @staticmethod
    def _last_force_before(plan: list[_Planned], tick: int) -> str:
        candidates = [p for p in plan if p.event.startswith("force_peak:") and p.tick <= tick]
        return candidates[-1].event if candidates else "spindle_on"

    # ------------------------------------------------------------------ lifecycle
    def begin(self) -> None:
        """Fresh run: clear the directory's run files and write the static files."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for name in ("frames.bin", "trace.jsonl", "findings.json", "timeline.json", "run.json"):
            if (self.dir / name).exists():
                (self.dir / name).unlink()
        for sub in ("series", "keyframes", "chips", "checkpoints"):
            target = self.dir / sub
            if target.exists():
                for path in sorted(target.rglob("*"), reverse=True):
                    path.unlink() if path.is_file() else path.rmdir()
        (self.dir / "frames.bin").write_bytes(b"")
        self.trace = TraceLog(self.dir / "trace.jsonl")
        self._write_static()
        self._write_findings()
        self._write_timeline()
        if self.write_status:
            self._write_status("running")

    def resume(self, checkpoint_json) -> None:
        """Continue after the checkpoint: everything written after its cut is dropped first."""
        marker = json.loads(Path(checkpoint_json).read_text(encoding="utf-8"))
        tick = int(marker["tick"])
        self.macro = tick // MACRO_TICKS
        with open(self.dir / "frames.bin", "r+b") as handle:
            handle.truncate(int(marker["frames_bytes"]))
        self.trace = TraceLog.resume(self.dir / "trace.jsonl", int(marker["trace_bytes"]))
        self.plan_cursor = int(marker["trace_events"])
        self.keyframes = int(marker["keyframes"])
        self._truncate_jsonl(self.dir / "keyframes" / "index.jsonl", self.keyframes)
        for path in sorted((self.dir / "keyframes").glob("kf_*.*")):
            if int(path.stem.split("_")[1]) >= self.keyframes:
                path.unlink()
        for path in sorted((self.dir / "chips").glob("kf_*.npz")):
            if int(path.stem.split("_")[1]) >= self.keyframes:
                path.unlink()
        index_path = self.dir / "keyframes" / "bricks.idx"
        if index_path.exists():
            document = json.loads(index_path.read_text(encoding="utf-8"))
            self.bricks_index = {b: [e for e in entries if e[0] < self.keyframes]
                                 for b, entries in document["bricks"].items()}
            self.bricks_index = {b: e for b, e in self.bricks_index.items() if e}
            self._write_bricks_index()
        self.closed_series = {tier: list(files) for tier, files in marker["series"].items()}
        self.part_counter = {tier: len(files) for tier, files in self.closed_series.items()}
        keep = {f for files in self.closed_series.values() for f in files}
        for path in sorted((self.dir / "series").rglob("part-*.parquet")):
            if path.relative_to(self.dir).as_posix() not in keep:
                path.unlink()
        self._truncate_jsonl(self.dir / "series" / "index.jsonl", int(marker["series_index_lines"]))
        document = json.loads((self.dir / "findings.json").read_text(encoding="utf-8"))
        self.findings = document["findings"][: int(marker["findings"])]
        self._write_findings()
        self.timeline_count = int(marker["timeline_blocks"])
        self._write_timeline()
        self.last_checkpoint = {"tick": tick, "file": f"checkpoints/ck_{tick}.json"}
        self.rows = {"fast": [], "tick": [], "macro": []}
        if self.write_status:
            self._write_status("running")

    @staticmethod
    def _truncate_jsonl(path: Path, lines: int) -> None:
        if not path.exists():
            return
        kept = path.read_bytes().splitlines(keepends=True)[:lines]
        write_bytes_atomic(path, b"".join(kept))

    def run_all(self, until_s: float | None = None) -> None:
        while not self.finished:
            if until_s is not None and self.time_s >= until_s - 1e-12:
                return
            self.step()
        return

    def step(self) -> None:
        """Advance one macro step; at the end of the run, write the results."""
        if self.finished:
            return
        tick0 = self.macro * MACRO_TICKS
        tick1 = tick0 + MACRO_TICKS
        if tick0 % KEYFRAME_TICKS == 0:
            self._write_keyframe(tick0)
        l1_ticks = tick0 + IPO_TICKS * np.arange(MACRO_IPO, dtype=np.int64)
        l0_ticks = tick0 + L0_TICKS * np.arange(MACRO_TICKS // L0_TICKS, dtype=np.int64)
        signals = self.l0(l0_ticks, self.macro)
        self._append_frames(l1_ticks)
        self._tick_rows(l1_ticks, signals)
        self._fast_rows(l0_ticks, signals)
        self._macro_row(tick1)
        self._timeline_to(tick1)
        self._events_to(tick1)
        self.macro += 1
        if tick1 % CHECKPOINT_TICKS == 0 or tick1 >= self.end_tick:
            self._close_parts()
            self._checkpoint(tick1)
        if tick1 >= self.end_tick:
            self.finish()
        elif self.write_status:
            self._write_status("running")

    # ------------------------------------------------------------------ static files
    def _write_static(self) -> None:
        write_json(self.dir / "channels.json", {"format": "cnctwin-channels", "version": FORMAT_VERSION,
                                                "channels": [c.to_dict() for c in sorted(ALL_CHANNELS,
                                                                                          key=lambda c: c.name)]})
        scene_dir = self.dir / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        meshes = {
            "base": box_mesh((-0.6, -0.4, -0.9), (0.6, 0.4, -0.5)),
            "table": box_mesh((-0.45, -0.2, -0.5), (0.45, 0.2, -0.45)),
            "vise": box_mesh((-0.03, -0.08, -0.45), (0.07, 0.08, -0.41)),
            "stock": box_mesh((0.0, 0.0, 0.0), STOCK_SIZE_M),
            "tool_T1": prism_mesh(TOOL_RADIUS_M, 0.0, 0.035),
            "spindle": prism_mesh(0.05, 0.1, 0.4),
        }
        for name, (vertices, faces) in meshes.items():
            data, _ = glb_bytes([{"name": name, "positions": vertices, "indices": faces}])
            write_bytes_atomic(scene_dir / f"{name}.glb", data)
        nx, ny, nz = (int(math.ceil(s / LATTICE_M / BRICK_CELLS)) for s in STOCK_SIZE_M)
        scene = {
            "format": "cnctwin-scene", "version": FORMAT_VERSION, "units": "m",
            "machine": {"id": "haas-vf2ss", "components": [
                {"id": "base", "mesh": "scene/base.glb", "link": None},
                {"id": "table", "mesh": "scene/table.glb", "link": "X"},
                {"id": "spindle", "mesh": "scene/spindle.glb", "link": "Z"}]},
            "chain": {"work": ["base", "Y", "X", "table"], "tool": ["base", "Z", "spindle"]},
            "axes": [{"name": "X", "kind": "linear", "travel_m": [-0.762, 0.0]},
                     {"name": "Y", "kind": "linear", "travel_m": [-0.406, 0.0]},
                     {"name": "Z", "kind": "linear", "travel_m": [-0.508, 0.0]}],
            "fixture": [{"id": "lang-makro-grip-77", "mesh": "scene/vise.glb"}],
            "stock": {"kind": "block", "size_m": list(STOCK_SIZE_M), "mesh": "scene/stock.glb",
                      "pose_W_S": np.eye(4).tolist(), "lattice_m": LATTICE_M,
                      "bricks": {"cells": BRICK_CELLS, "grid": [nx, ny, nz], "origin_S_m": [0.0, 0.0, 0.0],
                                 "brick_m": BRICK_CELLS * LATTICE_M}},
            "work_zero_S_m": list(WORK_ZERO_S_M),
            "tools": [{"slot": 0, "number": TOOL_NUMBER, "tool_id": TOOL_ID, "holder_id": HOLDER_ID,
                       "envelope": "scene/tool_T1.glb", "diameter_m": 2 * TOOL_RADIUS_M, "length_m": 0.035,
                       "teeth": N_TEETH}],
            "nozzles": [{"id": "n1", "kind": "flood", "origin_M_m": [0.08, 0.05, -0.25],
                         "direction_M": [-0.6, -0.4, -0.69]}],
            "frames": frames_header(),
        }
        write_json(self.dir / "scene.json", scene)

    # ------------------------------------------------------------------ per-step writers
    def _append_frames(self, ticks: np.ndarray) -> None:
        records = np.zeros(len(ticks), FRAME_DTYPE)
        tip = self.tip_S(ticks)
        angle = self.spindle_angle(ticks)
        records["tick"] = ticks
        records["axis"] = tip.astype(np.float32)
        records["spindle_angle_rad"][:, 0] = np.mod(angle, 2.0 * math.pi).astype(np.float32)
        records["spindle_rev"][:, 0] = np.floor(angle / (2.0 * math.pi)).astype(np.int32)
        records["tip_S_m"][:, 0, :] = tip.astype(np.float32)
        records["axis_T_S"][:, 0, :] = np.array([0.0, 0.0, 1.0], np.float32)
        with open(self.dir / "frames.bin", "ab") as handle:
            handle.write(records.tobytes())

    def _tick_rows(self, ticks: np.ndarray, signals: dict) -> None:
        rng = self._rng("tick-noise", self.macro)
        velocity = self.velocity_W(ticks)
        speed = np.linalg.norm(velocity, axis=1)
        follow = velocity / 33.3 + rng.normal(0.0, 1e-7, size=velocity.shape)
        torque_mean = signals["cutting.torque_Nm"].reshape(len(ticks), L0_PER_TICK).mean(axis=1)
        power = np.abs(torque_mean * self.omega(ticks))
        envelopes = {}
        for chan in FAST_CHANNELS:
            values = signals[chan.name].reshape(len(ticks), L0_PER_TICK, -1)
            stats = {"min": values.min(axis=1), "mean": values.mean(axis=1), "max": values.max(axis=1)}
            for j, column in enumerate(chan.columns()):
                for stat, array in stats.items():
                    envelopes[f"{column}.{stat}"] = array[:, j]   # NaN padding stays NaN
        for i, tick in enumerate(ticks):
            row = {"tick": int(tick), "time_s": float(tick * TICK_S), "motion.feed_m_s": float(speed[i]),
                   "cutting.spindle_power_W": float(power[i]),
                   "cutting.spindle_load_fraction": float(power[i] / AVAILABLE_POWER_W)}
            for axis in range(3):
                row[f"motion.following_error_m[{axis}]"] = float(follow[i, axis])
            for column, array in envelopes.items():
                row[column] = float(array[i])
            self.rows["tick"].append(row)

    def _fast_rows(self, ticks: np.ndarray, signals: dict) -> None:
        inside = np.zeros(len(ticks), bool)
        for start, end in self.fast_windows():
            inside |= (ticks >= start) & (ticks < end)
        for i in np.flatnonzero(inside):
            row = {"tick": int(ticks[i]), "time_s": float(ticks[i] * TICK_S)}
            for chan in FAST_CHANNELS:
                values = np.asarray(signals[chan.name][i]).reshape(-1)
                for j, column in enumerate(chan.columns()):
                    row[column] = float(values[j])
            self.rows["fast"].append(row)

    def _macro_row(self, tick: int) -> None:
        rng = self._rng("macro-noise", self.macro)
        cut_s = self.cut_time_s(tick)
        in_cut = bool(self.cut_ae(np.array([tick - 1]))[0] > 0.0)
        row = {"tick": int(tick), "time_s": float(tick * TICK_S),
               "thermal.edge_temp_K": AMBIENT_K + EDGE_RISE_K * (1.0 - math.exp(-cut_s / EDGE_TAU_S)),
               "thermal.work_bulk_mean_K": AMBIENT_K + 2.0 * cut_s,
               "chips.flute_fill": (0.15 + 0.55 * min(1.0, cut_s / 0.6)) if in_cut else 0.05,
               "coolant.reach_fraction": float(0.8 + 0.05 * rng.uniform(-1.0, 1.0)) if tick >= self._coolant_on else 0.0,
               "removal.volume_consistency": float(1.0 + 0.002 * rng.uniform(-1.0, 1.0)),
               "removal.removed_volume_m3": self.removed_volume_m3(tick)}
        for tooth in range(MAX_TEETH):
            row[f"wear.VB_m[{tooth}]"] = (VB0_M + VB_RATE_M_S * cut_s * (1.0 + 0.05 * tooth)) if tooth < N_TEETH \
                else float("nan")
        self.rows["macro"].append(row)

    def _timeline_to(self, tick: int) -> None:
        count = int(np.searchsorted(self.schedule.tick0, tick, side="left"))
        if count != self.timeline_count:
            self.timeline_count = count
            self._write_timeline()

    def _timeline_blocks(self, count: int) -> list[dict]:
        blocks = []
        tool = None
        for index in range(count):
            block = self.schedule.blocks[index]
            if block.kind == "tool_change":
                tool = TOOL_NUMBER
            blocks.append({"index": index, "op": OP, "program": PROGRAM, "line": block.line, "block": block.text,
                           "path": 0, "kind": block.kind, "tick0": int(self.schedule.tick0[index]),
                           "tick1": int(self.schedule.tick1[index]), "part": 0, "tool": tool,
                           "ticks": self._block_ticks(index)})
        return blocks

    def _block_ticks(self, index: int) -> dict:
        """The block's ticks by category (cutting where the tool is in the stock, below Z0)."""
        block = self.schedule.blocks[index]
        duration = int(self.schedule.tick1[index] - self.schedule.tick0[index])
        if block.kind != "linear":
            return {BLOCK_CATEGORY.get(block.kind, "auxiliary"): duration}
        if block.ae_mm > 0.0:
            return {"cutting": duration}
        z0, z1 = float(self.schedule.start_W_m[index][2]), float(self.schedule.end_W_m[index][2])
        if min(z0, z1) >= 0.0 or z0 == z1:
            return {"cutting": duration} if max(z0, z1) < 0.0 else {"air_cutting": duration}
        below = (0.0 - min(z0, z1)) / abs(z1 - z0) if max(z0, z1) > 0.0 else 1.0
        cut = int(round(duration * min(max(below, 0.0), 1.0)))
        return {k: v for k, v in (("cutting", cut), ("air_cutting", duration - cut)) if v}

    def _write_timeline(self) -> None:
        write_json(self.dir / "timeline.json", {"format": "cnctwin-timeline", "version": FORMAT_VERSION,
                                                "tick_s": TICK_S, "blocks": self._timeline_blocks(self.timeline_count)})

    def _events_to(self, tick: int) -> None:
        changed = False
        while self.plan_cursor < len(self.plan) and self.plan[self.plan_cursor].tick < tick:
            planned = self.plan[self.plan_cursor]
            ids = {p.event: i for i, p in enumerate(self.plan[: self.plan_cursor])}
            parents = [ids[name] for name in planned.parents if name in ids]
            event_id = self.trace.record(planned.tick * TICK_S, planned.tick, planned.domain, planned.event,
                                         planned.quantities, parents, planned.line)
            self.plan_cursor += 1
            if planned.finding is not None:
                self._add_finding(planned, event_id)
                changed = True
        self.trace.flush()
        if changed:
            self._write_findings()

    def _add_finding(self, planned: _Planned, event_id: int) -> None:
        tick = np.array([planned.tick])
        tip_S = self.tip_S(tick)[0]
        tip_M = self.tip_M(tick)[0]
        block = self.schedule.blocks[self._block_index(planned.line)] if planned.line else None
        finding = Finding(category=planned.finding["category"], severity=planned.finding["severity"],
                          message=planned.finding["message"], time_s=planned.tick * TICK_S, tick=planned.tick,
                          domain=planned.domain, op=OP, program=PROGRAM, line=planned.line,
                          block=block.text if block else "",
                          pos_W_m=tuple(float(v) for v in tip_S), pos_M_m=tuple(float(v) for v in tip_M),
                          quantities={k: float(v) for k, v in sorted(planned.quantities.items())},
                          cause_chain=self.trace.chain_ids(event_id), id=len(self.findings))
        self.findings.append(finding.to_dict())

    def _write_findings(self) -> None:
        write_json(self.dir / "findings.json", {"format": "cnctwin-findings", "version": FORMAT_VERSION,
                                                "findings": self.findings})

    def _close_parts(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        for tier in ("fast", "tick", "macro"):
            rows = self.rows[tier]
            if not rows:
                continue
            columns = tier_columns(tier)
            arrays = {}
            for column in columns:
                values = [row[column] for row in rows]
                arrays[column] = pa.array(values, type=pa.int64() if column == "tick" else pa.float64())
            table = pa.table(arrays)
            relative = f"series/{tier}/part-{self.part_counter[tier]}.parquet"
            target = self.dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.tmp")
            pq.write_table(table, tmp, row_group_size=4096, compression="zstd")
            os.replace(tmp, target)
            entry = {"tier": tier, "part": self.part_counter[tier], "file": relative, "tick0": rows[0]["tick"],
                     "tick1": rows[-1]["tick"], "t0_s": rows[0]["time_s"], "t1_s": rows[-1]["time_s"],
                     "rows": len(rows)}
            with open(self.dir / "series" / "index.jsonl", "ab") as handle:
                handle.write((json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"))
            self.closed_series[tier].append(relative)
            self.part_counter[tier] += 1
            self.rows[tier] = []

    def _checkpoint(self, tick: int) -> None:
        directory = self.dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        npz = directory / f"ck_{tick}.npz"
        write_npz(npz, {"tick": np.array(tick, np.int64), "macro": np.array(self.macro, np.int64),
                        "rng_counters": np.array([self.macro, self.plan_cursor], np.int64)})
        index_path = self.dir / "series" / "index.jsonl"
        marker = {"format": "cnctwin-checkpoint", "version": FORMAT_VERSION, "tick": tick, "time_s": tick * TICK_S,
                  "macro": self.macro, "op": OP, "npz": npz.name, "npz_sha256": sha256_file(npz),
                  "trace_bytes": self.trace.byte_length(), "trace_events": self.plan_cursor,
                  "frames_bytes": (self.dir / "frames.bin").stat().st_size, "keyframes": self.keyframes,
                  "timeline_blocks": self.timeline_count, "findings": len(self.findings),
                  "series": {tier: list(files) for tier, files in self.closed_series.items()},
                  "series_index_lines": len(index_path.read_bytes().splitlines()) if index_path.exists() else 0}
        write_json(directory / f"ck_{tick}.json", marker)
        self.last_checkpoint = {"tick": tick, "file": f"checkpoints/ck_{tick}.json"}
        markers = sorted(directory.glob("ck_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        for old in markers[:-2]:
            old.unlink()
            stale = old.with_suffix(".npz")
            if stale.exists():
                stale.unlink()

    # ------------------------------------------------------------------ keyframes
    def _brick_boxes(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        brick_m = BRICK_CELLS * LATTICE_M
        counts = [int(math.ceil(s / brick_m)) for s in STOCK_SIZE_M]
        boxes = []
        for kz in range(counts[2]):
            for j in range(counts[1]):
                for i in range(counts[0]):
                    lo = np.array([i, j, kz]) * brick_m
                    hi = np.minimum(lo + brick_m, STOCK_SIZE_M)
                    boxes.append((i + counts[0] * (j + counts[1] * kz), lo, hi))
        return boxes

    def _machined_until(self, tick: int) -> np.ndarray:
        """Tool-tip positions in S of the cutting moves up to ``tick`` (1 ms resolution)."""
        ticks = np.arange(0, max(tick, 1), IPO_TICKS, dtype=np.int64)
        cutting = self.cut_ae(ticks) > 0.0
        return self.tip_S(ticks[cutting]) if cutting.any() else np.zeros((0, 3))

    def _brick_mesh(self, lo: np.ndarray, hi: np.ndarray, cut_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        top = hi[2]
        if len(cut_points):
            inside = ((cut_points[:, 0] >= lo[0] - TOOL_RADIUS_M) & (cut_points[:, 0] <= hi[0] + TOOL_RADIUS_M)
                      & (cut_points[:, 1] >= lo[1] - TOOL_RADIUS_M) & (cut_points[:, 1] <= hi[1] + TOOL_RADIUS_M))
            if inside.any() and hi[2] >= STOCK_SIZE_M[2] - 1e-9:
                top = hi[2] - AP_M
        vertices, faces = box_mesh(lo, np.array([hi[0], hi[1], top]))
        cells = np.round((vertices - lo) / LATTICE_M).astype(np.uint16)
        quantised = np.zeros((len(cells), 4), np.uint16)
        quantised[:, :3] = cells
        return quantised, faces.reshape(-1)

    def _write_keyframe(self, tick: int) -> None:
        k = self.keyframes
        full = (k % FULL_SNAPSHOT_EVERY == 0)
        cut_now = self._machined_until(tick)
        changed_points = cut_now
        if not full:
            ticks = np.arange(max(0, tick - KEYFRAME_TICKS), tick, IPO_TICKS, dtype=np.int64)
            mask = self.cut_ae(ticks) > 0.0
            changed_points = self.tip_S(ticks[mask]) if mask.any() else np.zeros((0, 3))
        meshes, ids = [], []
        for brick, lo, hi in self._brick_boxes():
            touched = len(changed_points) and np.any(
                (changed_points[:, 0] >= lo[0] - TOOL_RADIUS_M) & (changed_points[:, 0] <= hi[0] + TOOL_RADIUS_M)
                & (changed_points[:, 1] >= lo[1] - TOOL_RADIUS_M) & (changed_points[:, 1] <= hi[1] + TOOL_RADIUS_M))
            if not (full or touched):
                continue
            positions, indices = self._brick_mesh(lo, hi, cut_now)
            header = struct.pack("<4sIIffff", b"BRK1", len(positions), len(indices), *lo.astype(np.float32),
                                 LATTICE_M)
            meshes.append({"name": f"brick_{brick}", "positions": positions, "indices": indices,
                           "translation": lo, "scale": [LATTICE_M] * 3, "header": header})
            ids.append(brick)
        directory = self.dir / "keyframes"
        directory.mkdir(parents=True, exist_ok=True)
        glb_name = f"kf_{k}.glb"
        data, spans = glb_bytes(meshes)
        write_bytes_atomic(directory / glb_name, data)
        for brick, (offset, length) in zip(ids, spans):
            self.bricks_index.setdefault(str(brick), []).append([k, glb_name, int(offset), int(length)])
        self._write_bricks_index()
        self._write_fields(directory / f"kf_{k}.npz", tick)
        self._write_chips(self.dir / "chips" / f"kf_{k}.npz", tick, k)
        entry = {"k": k, "tick": tick, "time_s": tick * TICK_S, "glb": f"keyframes/{glb_name}",
                 "fields": f"keyframes/kf_{k}.npz", "chips": f"chips/kf_{k}.npz", "full": full, "bricks": ids}
        with open(directory / "index.jsonl", "ab") as handle:
            handle.write((json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"))
        self.keyframes += 1

    def _write_bricks_index(self) -> None:
        brick_m = BRICK_CELLS * LATTICE_M
        counts = [int(math.ceil(s / brick_m)) for s in STOCK_SIZE_M]
        document = {"format": "cnctwin-bricks-index", "version": FORMAT_VERSION, "cells": BRICK_CELLS,
                    "lattice_m": LATTICE_M, "grid": counts, "origin_S_m": [0.0, 0.0, 0.0],
                    "record": "<4sIIffff header (BRK1, n_vertices, n_indices, origin_S_m[3], cell_m), uint16 "
                              "positions [n_vertices, 4] in cells, uint32 indices [n_indices]",
                    "bricks": {b: self.bricks_index[b] for b in sorted(self.bricks_index, key=int)}}
        write_json(self.dir / "keyframes" / "bricks.idx", document)

    def _write_fields(self, path: Path, tick: int) -> None:
        nx = int(round(STOCK_SIZE_M[0] / SURFACE_CELL_M))
        ny = int(round(STOCK_SIZE_M[1] / SURFACE_CELL_M))
        centres_x = (np.arange(nx) + 0.5) * SURFACE_CELL_M
        centres_y = (np.arange(ny) + 0.5) * SURFACE_CELL_M
        tip = self.tip_S(np.array([tick]))[0]
        distance = np.hypot(centres_x[None, :] - tip[0], centres_y[:, None] - tip[1])
        cut_s = self.cut_time_s(tick)
        heat = (EDGE_RISE_K * 0.2) * (1.0 - math.exp(-cut_s / EDGE_TAU_S))
        surface = AMBIENT_K + heat * np.exp(-distance / 0.006)
        pile = np.clip(self.removed_volume_m3(tick) * 2e4 * np.exp(-distance / 0.01), 0.0, None)
        coolant_on = self._coolant_on <= tick < self._coolant_off
        wetted = distance < (0.02 if coolant_on else 0.0)
        force = self.l0(np.array([tick]), 10_000_000 + tick // MACRO_TICKS)["cutting.force_N"][0]
        arrays = {"tick": np.array(tick, np.int64), "time_s": np.array(tick * TICK_S),
                  "surface_origin_S_m": np.array([0.0, 0.0, STOCK_SIZE_M[2]]),
                  "surface_cell_m": np.array(SURFACE_CELL_M),
                  "surface_temp_K": surface.astype(np.float16),
                  "pile_height_m": pile.astype(np.float32), "wetted": wetted,
                  "tool_deflection_m": force / TOOL_STIFFNESS_N_PER_M, "part_pose": np.eye(4)}
        jets = 1 if coolant_on else 0
        arrays.update({
            "jet_nozzle_id": np.array(["n1"][:jets], dtype="<U8"),
            "jet_kind": np.array(["flood"][:jets], dtype="<U16"),
            "jet_origin_M_m": np.array([[0.08, 0.05, -0.25]][:jets], float).reshape(jets, 3),
            "jet_direction_M": np.array([[-0.6, -0.4, -0.69]][:jets], float).reshape(jets, 3),
            "jet_speed_m_s": np.array([3.2][:jets], float), "jet_flow_m3_s": np.array([2.5e-4][:jets], float),
            "jet_half_angle_rad": np.array([0.12][:jets], float),
            "jet_reach_fraction": np.array([0.82][:jets], float)})
        write_npz(path, arrays)

    def _write_chips(self, path: Path, tick: int, k: int) -> None:
        cutting = bool(self.cut_ae(np.array([tick]))[0] > 0.0)
        rng = self._rng("chips", k)
        n = 24 if cutting else 0
        tip = self.tip_M(np.array([tick]))[0]
        removed_kg = self.removed_volume_m3(tick) * DENSITY_KG_M3
        arrays = {"tick": np.array(tick, np.int64),
                  "pos_M_m": tip + rng.normal(0.0, 0.01, size=(n, 3)),
                  "vel_M_m_s": rng.normal(0.0, 3.0, size=(n, 3)),
                  "mass_kg": np.full(n, 2.0e-7), "temp_K": AMBIENT_K + 120.0 * rng.uniform(size=n),
                  "form_code": np.full(n, 61, np.int16),
                  "colour_rgb": np.tile(np.array([[196, 198, 204]], np.uint8), (n, 1)),
                  "pile_mass_kg": np.array(removed_kg * 0.9), "conveyed_kg": np.array(removed_kg * 0.05),
                  "pile_height_max_m": np.array(removed_kg * 0.8)}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_npz(path, arrays)

    # ------------------------------------------------------------------ the end of the run
    def finish(self) -> None:
        if self.finished:
            return
        if self.end_tick % KEYFRAME_TICKS == 0 and self.keyframes * KEYFRAME_TICKS <= self.end_tick:
            self._write_keyframe(self.end_tick)
        self._write_attributes()
        self._write_results()
        if self.media:
            self._write_media()
        self.trace.flush()
        self._write_run_json()
        self.finished = True
        if self.write_status:
            self._write_status("done")

    def _write_attributes(self) -> None:
        directory = self.dir / "keyframes" / "attr"
        directory.mkdir(parents=True, exist_ok=True)
        cut = self._machined_until(self.end_tick)
        for brick, lo, hi in self._brick_boxes():
            positions, _ = self._brick_mesh(lo, hi, cut)
            rng = self._rng("attr", brick)
            n = len(positions)
            write_npz(directory / f"brick_{brick}.npz", {
                "deviation_m": rng.normal(2e-5, 1e-5, size=n).astype(np.float32),
                "Ra_m": (0.8e-6 + 0.3e-6 * rng.uniform(size=n)).astype(np.float32),
                "generating_VB_m": np.full(n, VB0_M + VB_RATE_M_S * self.cut_time_s(self.end_tick), np.float32)})

    def _part_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        outer_v, outer_f = box_mesh((0.0, 0.0, 0.0), STOCK_SIZE_M)
        lo = np.array(WORK_ZERO_S_M) - [TOOL_RADIUS_M, TOOL_RADIUS_M, AP_M]
        hi = np.array(WORK_ZERO_S_M) + [0.016 + TOOL_RADIUS_M, 0.012 + TOOL_RADIUS_M, 0.0]
        pocket_v, pocket_f = box_mesh(lo, hi)
        return np.concatenate([outer_v, pocket_v]), np.concatenate([outer_f, pocket_f[:, ::-1] + len(outer_v)])

    def _write_results(self) -> None:
        cut_s = self.cut_time_s(self.end_tick)
        removed = self.removed_volume_m3(self.end_tick)
        measured = 0.002062 if self.include_error else 0.002011
        write_json(self.dir / "inspection.json", {
            "format": "cnctwin-inspection", "version": FORMAT_VERSION, "temperature_K": AMBIENT_K,
            "tolerances": [
                {"id": "pocket-depth", "kind": "distance", "nominal_m": 0.002, "plus_m": 5e-5, "minus_m": 5e-5,
                 "measured_m": measured, "deviation_m": measured - 0.002, "pass": abs(measured - 0.002) <= 5e-5},
                {"id": "floor-flatness", "kind": "flatness", "tolerance_m": 2e-5, "measured_m": 8e-6, "pass": True}],
            "summary": {"checked": 2, "failed": 1 if self.include_error else 0}})
        write_json(self.dir / "surface.json", {
            "format": "cnctwin-surface", "version": FORMAT_VERSION,
            "features": [{"id": "floor-finish", "Ra_m": 0.92e-6, "Rz_m": 5.1e-6, "spec_Ra_m": 1.6e-6, "pass": True}],
            "burrs": [{"edge": "pocket wall X16", "height_m": 4.0e-5}]})
        vb = [VB0_M + VB_RATE_M_S * cut_s * (1.0 + 0.05 * t) for t in range(N_TEETH)]
        write_json(self.dir / "tools.json", {
            "format": "cnctwin-tools", "version": FORMAT_VERSION,
            "tools": [{"slot": 0, "number": TOOL_NUMBER, "tool_id": TOOL_ID, "holder_id": HOLDER_ID,
                       "cut_time_s": cut_s, "VB_m": vb, "VB_max_m": max(vb), "life_used_fraction": cut_s / 3600.0,
                       "broken": False,
                       "max_edge_temp_K": AMBIENT_K + EDGE_RISE_K * (1.0 - math.exp(-cut_s / EDGE_TAU_S))}]})
        from cnctwin.production import production_from_run_dir
        production = production_from_run_dir(self.dir, project=PRODUCTION_PROJECT, tick_s=TICK_S,
                                             removed_mass_kg=removed * DENSITY_KG_M3, write=False)
        production["removed_volume_m3"] = removed
        write_json(self.dir / "production.json", production)
        speeds = np.linspace(500.0, 2500.0, 81)
        lobes = AP_M * (1.2 + 0.8 * np.abs(np.sin(speeds / 180.0)))
        write_json(self.dir / "sld" / "T1.json", {
            "format": "cnctwin-sld", "version": FORMAT_VERSION, "tool": "T1", "slot": 0, "number": TOOL_NUMBER,
            "speed_rad_s": speeds.tolist(), "depth_limit_m": lobes.tolist(), "radial_immersion_m": 0.004,
            "operating_points": [{"op": OP, "line": 13, "speed_rad_s": abs(self.omega_rad_s), "ap_m": AP_M,
                                  "stable": False}]})
        vertices, faces = self._part_mesh()
        glb, _ = glb_bytes([{"name": f"part_{OP}", "positions": vertices, "indices": faces.reshape(-1)}])
        write_bytes_atomic(self.dir / f"part_{OP}.glb", glb)
        write_bytes_atomic(self.dir / f"part_{OP}.stl", stl_bytes(vertices, faces, f"cnctwin part {OP}"))
        write_bytes_atomic(self.dir / f"part_{OP}.3mf", threemf_bytes(vertices, faces))
        unloaded = vertices + np.array([0.0, 0.0, 2e-6]) * (vertices[:, 2:3] / STOCK_SIZE_M[2])
        write_bytes_atomic(self.dir / f"part_{OP}_unload.stl", stl_bytes(unloaded, faces, f"cnctwin part {OP} unload"))

    def _write_media(self) -> None:
        verdict = self.verdict()
        rows = "".join(f"<tr><td>{f['severity']}</td><td>{f['category']}</td><td>{f['line']}</td>"
                       f"<td>{f['message']}</td></tr>" for f in self.findings)
        safe = "SAFE TO RUN" if verdict["safe"] else "NOT SAFE TO RUN"
        report = (f"<!doctype html><html><head><meta charset='utf-8'><title>CNC Twin report</title></head><body>"
                  f"<h1>{safe}</h1><p>Cycle time {self.end_tick * TICK_S:.3f} s</p>"
                  f"<table><tr><th>severity</th><th>category</th><th>line</th><th>message</th></tr>{rows}</table>"
                  f"<img src='images/forces.png' alt='forces'></body></html>\n")
        write_bytes_atomic(self.dir / "report.html", report.encode("utf-8"))
        write_bytes_atomic(self.dir / "report.pdf", pdf_bytes(
            ["CNC Twin report (synthetic run)", safe, f"Cycle time {self.end_tick * TICK_S:.3f} s"]
            + [f"{f['severity']}: {f['message']}" for f in self.findings]))
        sheet = (f"<!doctype html><html><head><meta charset='utf-8'><title>Setup sheet</title></head><body>"
                 f"<h1>Setup sheet {OP}</h1><p>T{TOOL_NUMBER} {TOOL_ID} in {HOLDER_ID}, stickout 35 mm</p>"
                 f"<p>Stock 40 x 30 x 15 mm, G54 at the pocket corner</p></body></html>\n")
        write_bytes_atomic(self.dir / "setup_sheet.html", sheet.encode("utf-8"))
        write_bytes_atomic(self.dir / "setup_sheet.pdf", pdf_bytes([f"Setup sheet {OP}", f"T{TOOL_NUMBER} {TOOL_ID}"]))
        image = np.full((60, 80, 3), 200, np.uint8)
        image[18:42, 20:52] = (120, 130, 140)
        write_bytes_atomic(self.dir / "images" / "part.png", png_bytes(image))
        chart = np.full((48, 96, 3), 255, np.uint8)
        ticks = np.linspace(0, self.end_tick - 1, 96).astype(np.int64)
        power = np.abs(self.l0(ticks, 20_000_000)["cutting.torque_Nm"] * self.omega(ticks))
        heights = (power / max(power.max(), 1e-9) * 40).astype(int)
        for x, height in enumerate(heights):
            chart[47 - height:48, x] = (40, 90, 160)
        write_bytes_atomic(self.dir / "images" / "forces.png", png_bytes(chart))
        n_frames = max(1, int(self.end_tick * TICK_S * VIDEO_FPS))
        frames = np.zeros((n_frames, VIDEO_SIZE[1], VIDEO_SIZE[0], 3), np.uint8)
        frames[:] = (30, 34, 40)
        for i in range(n_frames):
            tip = self.tip_S(np.array([int(i / VIDEO_FPS / TICK_S)]))[0]
            x = int(np.clip(tip[0] / STOCK_SIZE_M[0] * VIDEO_SIZE[0], 0, VIDEO_SIZE[0] - 4))
            y = int(np.clip(tip[1] / STOCK_SIZE_M[1] * VIDEO_SIZE[1], 0, VIDEO_SIZE[1] - 4))
            frames[i, 8:40, 8:56] = (150, 155, 160)
            frames[i, y:y + 4, x:x + 4] = (230, 120, 40)
        write_mp4(self.dir / "video.mp4", frames, VIDEO_FPS)
        t = np.arange(int(self.end_tick * TICK_S * AUDIO_RATE_HZ)) / AUDIO_RATE_HZ
        tooth_hz = N_TEETH * self.rpm / 60.0
        rng = self._rng("audio")
        audio = 0.3 * np.sin(2 * math.pi * tooth_hz * t) * (self.cut_ae((t / TICK_S).astype(np.int64)) > 0) \
            + 0.05 * np.sin(2 * math.pi * self.rpm / 60.0 * t) + 0.01 * rng.normal(size=len(t))
        write_bytes_atomic(self.dir / "audio.wav", wav_bytes(audio, AUDIO_RATE_HZ))

    def verdict(self) -> dict:
        reasons = [f"{f['category']} ({f['severity']}) at {f['op']} line {f['line']}: {f['message']}"
                   for f in self.findings if f["severity"] in ("error", "fatal")]
        return {"safe": not reasons, "reasons": reasons}

    def _write_run_json(self) -> None:
        outputs = []
        for path in sorted(p for p in self.dir.rglob("*") if p.is_file()):
            relative = path.relative_to(self.dir).as_posix()
            if relative in ("run.json", "status.json") or path.name.startswith("."):
                continue
            outputs.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        counts = {s: sum(1 for f in self.findings if f["severity"] == s) for s in ("info", "warning", "error", "fatal")}
        removed = self.removed_volume_m3(self.end_tick)
        project_text = project_document(self.seed)
        program = program_text()
        project_file = self.project_path.name if self.project_path else "project.yaml"
        document = {
            "format": "cnctwin-run", "version": FORMAT_VERSION,
            "project": {"name": "Synthetic pocket, Al 6061-T6", "seed": self.seed, "machine": {"id": "haas-vf2ss"},
                        "material": {"id": "al6061-t6"},
                        "stock": {"kind": "block", "size_m": list(STOCK_SIZE_M)},
                        "tools": {"list": [{"number": TOOL_NUMBER, "tool": {"id": TOOL_ID},
                                            "holder": {"id": HOLDER_ID}, "stickout_m": 0.035}]},
                        "operations": [{"name": OP, "program": PROGRAM, "program_text": program,
                                        "program_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest()}]},
            "project_file": project_file,
            "project_dir": str(self.project_path.parent) if self.project_path else None,
            "inputs": {project_file: {"sha256": hashlib.sha256(project_text.encode("utf-8")).hexdigest(),
                                      "bytes": len(project_text.encode("utf-8"))},
                       PROGRAM: {"sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
                                 "bytes": len(program.encode("utf-8"))}},
            "versions": {"cnctwin": "1.0.0", "format": FORMAT_VERSION, "python": "3.12", "synthetic": True},
            "numba_cache_key": hashlib.sha256(f"synthetic-{self.seed}".encode()).hexdigest()[:16],
            "threads": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "NUMBA_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
            "clock": {"tick_s": TICK_S, "ipo_ticks": IPO_TICKS, "macro_ipo": MACRO_IPO, "l0_ticks": {OP: L0_TICKS}},
            "seed": self.seed,
            "verdict": self.verdict(),
            "summary": {"cycle_time_s": self.end_tick * TICK_S, "cut_time_s": self.cut_time_s(self.end_tick),
                        "removed_volume_m3": removed, "removed_mass_kg": removed * DENSITY_KG_M3,
                        "findings": counts, "spindle_speed_rad_s": abs(self.omega_rad_s),
                        "macro_steps": self.end_tick // MACRO_TICKS},
            "outputs": outputs,
        }
        write_json(self.dir / "run.json", document)

    def _write_status(self, status: str) -> None:
        progress = {"sim_time_s": self.time_s, "est_total_s": self.total_s, "wall_s": 0.0, "eta_s": 0.0}
        write_json(self.dir / "status.json", {"format": "cnctwin-run-status", "version": FORMAT_VERSION,
                                              "status": status, "progress": progress, "op": OP,
                                              "checkpoint": self.last_checkpoint, "error": None})


# ----------------------------------------------------------------------------- convenience

def make_synthetic_run(run_dir, seed: int = DEFAULT_SEED, *, until_s: float | None = None, media: bool = True,
                       write_status: bool = True, project_path=None) -> dict:
    """Write a whole run (or the part up to ``until_s`` of simulated time, status running).

    Returns a manifest for tests: seed, end tick, the counts written and the file list.
    """
    run = SyntheticRun(run_dir, seed, write_status=write_status, media=media, project_path=project_path)
    run.begin()
    run.run_all(until_s)
    files = sorted(p.relative_to(run.dir).as_posix() for p in run.dir.rglob("*") if p.is_file())
    return {"run_dir": str(run.dir), "seed": seed, "end_tick": run.end_tick, "finished": run.finished,
            "macro_steps": run.macro, "keyframes": run.keyframes, "findings": len(run.findings),
            "trace_events": run.plan_cursor, "timeline_blocks": run.timeline_count, "files": files,
            "verdict": run.verdict(), "omega_rad_s": run.omega_rad_s}


RESULT_SET = ("findings.json", "trace.jsonl", "series", "frames.bin", "keyframes", "inspection.json", "surface.json",
              "tools.json", "production.json")


def result_set_digest(run_dir) -> dict[str, str]:
    """sha256 of every file of the declared result set (architecture 9)."""
    run_dir = Path(run_dir)
    digests = {}
    for name in RESULT_SET:
        target = run_dir / name
        paths = sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else \
            ([target] if target.exists() else [])
        for path in paths:
            if path.name.startswith("."):
                continue
            digests[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    for path in sorted(run_dir.glob("part_*")):
        digests[path.name] = sha256_file(path)
    return digests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--until", type=float, default=None, help="stop after this much simulated time (s)")
    parser.add_argument("--no-media", action="store_true", help="skip video, audio, images and reports")
    args = parser.parse_args(argv)
    manifest = make_synthetic_run(args.run_dir, args.seed, until_s=args.until, media=not args.no_media)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
