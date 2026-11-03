"""Wavefront OBJ: read and write with groups (docs/models/geom.md "Files").

Read: ``v`` (x y z, an optional w is a homogeneous weight and divides), ``f`` with any of the index forms
``i``, ``i/t``, ``i//n``, ``i/t/n`` (negative indices count back from the last vertex), polygons split into
triangle fans, and ``g``/``o`` names as face groups (in order of first appearance, ``group_names`` kept;
faces before any group are group -1). Texture coordinates, normals, smoothing groups, materials, free-form
geometry and line/point elements carry no surface of the mesh: they are not read, and when a ``report``
dict is passed it receives the count of each skipped statement keyword. Write: ``v`` lines, ``vn`` when the
mesh has a ``normal`` vertex attribute, faces in mesh order with a ``g`` line wherever the group changes
(the mesh name goes into a comment, so reading back numbers the groups exactly as they appear).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.errors import InputError
from ..mesh import Mesh
from .units import unit_scale


def read_obj(path, unit="mm", report: dict | None = None) -> Mesh:
    path = Path(path)
    scale = unit_scale(unit)
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    groups: list[int] = []
    group_ids: dict[str, int] = {}
    current = -1
    skipped: dict[str, int] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            while line.endswith("\\"):
                line = line[:-1] + " " + next(handle, "").split("#", 1)[0].strip()
            parts = line.split()
            key = parts[0]
            if key == "v":
                try:
                    xyz = [float(p) for p in parts[1:4]]
                    w = float(parts[4]) if len(parts) > 4 else 1.0
                except ValueError as error:
                    raise InputError(f"bad vertex: {error}", str(path), line_no) from None
                if len(xyz) != 3 or w == 0.0:
                    raise InputError("a vertex needs x y z (and a non-zero w)", str(path), line_no)
                verts.append([c / w for c in xyz])
            elif key == "f":
                idx = []
                for token in parts[1:]:
                    head = token.split("/", 1)[0]
                    try:
                        i = int(head)
                    except ValueError:
                        raise InputError(f"bad face index {token!r}", str(path), line_no) from None
                    i = i - 1 if i > 0 else len(verts) + i
                    if not 0 <= i < len(verts):
                        raise InputError(f"face index {token!r} refers to a missing vertex", str(path), line_no)
                    idx.append(i)
                if len(idx) < 3:
                    raise InputError("a face needs at least three vertices", str(path), line_no)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
                    groups.append(current)
            elif key in ("g", "o"):
                name = " ".join(parts[1:]) or "default"
                if name not in group_ids:
                    group_ids[name] = len(group_ids)
                current = group_ids[name]
            else:
                skipped[key] = skipped.get(key, 0) + 1
    if report is not None:
        report.update({"skipped": dict(sorted(skipped.items()))})
    v = np.array(verts, dtype=np.float64).reshape(-1, 3) * scale
    return Mesh(v, np.array(faces, dtype=np.int64).reshape(-1, 3), np.array(groups, dtype=np.int64),
                group_names={i: n for n, i in group_ids.items()}, name=path.stem)


def obj_text(mesh: Mesh, unit="mm") -> str:
    scale = unit_scale(unit)
    lines = [f"# cnctwin OBJ, units {unit}", f"# name {mesh.name or 'mesh'}"]
    for x, y, z in (mesh.vertices / scale).tolist():
        lines.append(f"v {x!r} {y!r} {z!r}")
    normals = mesh.vertex_attr.get("normal")
    if normals is not None:
        for x, y, z in np.asarray(normals, dtype=np.float64).tolist():
            lines.append(f"vn {x!r} {y!r} {z!r}")
    current = None
    for (a, b, c), g in zip((mesh.faces + 1).tolist(), mesh.face_group.tolist()):
        if g != current:
            if g >= 0 or current is not None:
                lines.append(f"g {mesh.group_names.get(g, f'group{g}') if g >= 0 else 'default'}")
            current = g
        if normals is not None:
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
        else:
            lines.append(f"f {a} {b} {c}")
    return "\n".join(lines) + "\n"


def write_obj(path, mesh: Mesh, unit="mm") -> None:
    Path(path).write_text(obj_text(mesh, unit), encoding="utf-8")


__all__ = ["read_obj", "write_obj", "obj_text"]
