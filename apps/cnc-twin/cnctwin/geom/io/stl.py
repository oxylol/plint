"""STL (3D Systems 1989): ASCII and binary, read and write (docs/models/geom.md "Files").

STL has no unit: ``unit`` (default mm, the CAD convention) scales file numbers to metres on read and back
on write. Reading welds bit-identical corners into shared vertices (``weld=False`` keeps the soup); an
ASCII file with several ``solid`` blocks gives one face group per block (``group_names`` from the solid
names); a binary file's 16-bit attribute words are kept as the face attribute ``stl_attribute`` when any
is non-zero (some exporters store a colour there). Binary output: an 80-byte header that never starts
with ``solid``, float32 facet normals computed from the geometry, attribute words 0. ASCII output writes
the shortest round-trip decimal of every float64. Output bytes depend only on the mesh.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ...core.errors import InputError
from ..mesh import Mesh
from .units import unit_scale

BINARY_HEADER = b"cnctwin binary STL".ljust(80, b" ")
_RECORD = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_VERTEX = re.compile(rb"vertex\s+(" + _FLOAT.encode() + rb")\s+(" + _FLOAT.encode() + rb")\s+(" + _FLOAT.encode() + rb")")
_SOLID = re.compile(rb"^\s*solid\b[ \t]*([^\r\n]*)", re.MULTILINE)


def _is_binary(data: bytes) -> bool:
    if len(data) >= 84:
        n = int(np.frombuffer(data[80:84], "<u4")[0])
        if len(data) == 84 + 50 * n:
            return True
    return not data.lstrip()[:5].lower() == b"solid"


def read_stl(path, unit="mm", weld: bool = True) -> Mesh:
    """Mesh (metres) of an ASCII or binary STL file."""
    path = Path(path)
    data = path.read_bytes()
    scale = unit_scale(unit)
    if _is_binary(data):
        if len(data) < 84:
            raise InputError("not an STL file (shorter than the 84-byte binary header)", str(path))
        n = int(np.frombuffer(data[80:84], "<u4")[0])
        if len(data) < 84 + 50 * n:
            raise InputError(f"binary STL declares {n} facets but holds {(len(data) - 84) // 50}", str(path))
        rec = np.frombuffer(data, _RECORD, count=n, offset=84)
        tris = rec["v"].astype(np.float64) * scale
        face_attr = {"stl_attribute": rec["attr"].astype(np.int32)} if np.any(rec["attr"]) else {}
        mesh = Mesh(tris.reshape(-1, 3), np.arange(3 * n).reshape(-1, 3), face_attr=face_attr, name=path.stem)
        return mesh.weld(0.0, drop_degenerate=False) if weld else mesh
    starts = [m.start() for m in _SOLID.finditer(data)] or [0]
    names = [m.group(1).decode("utf-8", "replace").strip() for m in _SOLID.finditer(data)]
    blocks = [data[a:b] for a, b in zip(starts, starts[1:] + [len(data)])]
    tris, groups = [], []
    for g, block in enumerate(blocks):
        coords = np.array(_VERTEX.findall(block), dtype=np.float64).reshape(-1, 3)
        if len(coords) % 3:
            raise InputError(f"ASCII STL solid {g} has {len(coords)} vertices, not a multiple of 3", str(path))
        tris.append(coords.reshape(-1, 3, 3))
        groups.append(np.full(len(coords) // 3, g if len(blocks) > 1 else -1, np.int64))
    soup = np.concatenate(tris) * scale if tris else np.zeros((0, 3, 3))
    group_names = {g: n for g, n in enumerate(names) if n} if len(blocks) > 1 else {}
    mesh = Mesh(soup.reshape(-1, 3), np.arange(3 * len(soup)).reshape(-1, 3), np.concatenate(groups),
                group_names=group_names, name=(names[0] if names and names[0] else path.stem))
    return mesh.weld(0.0, drop_degenerate=False) if weld else mesh


def stl_bytes(mesh: Mesh, unit="mm", ascii: bool = False) -> bytes:
    scale = unit_scale(unit)
    tris = mesh.triangles() / scale
    normals = mesh.face_normals()
    if not ascii:
        rec = np.zeros(len(tris), _RECORD)
        rec["normal"] = normals.astype("<f4")
        rec["v"] = tris.astype("<f4")
        return BINARY_HEADER + np.uint32(len(tris)).astype("<u4").tobytes() + rec.tobytes()
    name = (mesh.name or "mesh").replace("\n", " ")
    lines = [f"solid {name}"]
    for n, t in zip(normals.tolist(), tris.tolist()):
        lines.append(f"  facet normal {n[0]!r} {n[1]!r} {n[2]!r}")
        lines.append("    outer loop")
        for v in t:
            lines.append(f"      vertex {v[0]!r} {v[1]!r} {v[2]!r}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_stl(path, mesh: Mesh, unit="mm", ascii: bool = False) -> None:
    Path(path).write_bytes(stl_bytes(mesh, unit, ascii))


__all__ = ["read_stl", "write_stl", "stl_bytes", "BINARY_HEADER"]
