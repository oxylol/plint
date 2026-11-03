"""PLY (Turk 1994): write, and read the same subset back (docs/models/geom.md "Files").

Write: ``binary_little_endian`` (default) or ``ascii``; vertex ``x y z`` (double by default, ``float`` on
request), ``nx ny nz`` from a ``normal`` vertex attribute, every other vertex attribute as float properties
(``name`` or ``name_0 .. name_k``), faces as ``list uchar int vertex_indices``, the face group as ``int
face_group`` when any face has one, face attributes as float properties. Read: ascii and both binary
encodings, any scalar property types, the face list (polygons as fans) and scalar face properties; other
elements (edges, materials) are not geometry of the mesh and are listed in ``report['skipped']``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.errors import InputError
from ..mesh import Mesh
from .units import unit_scale

_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
          "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
          "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _attr_columns(name: str, arr: np.ndarray):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return [name], a[:, None]
    flat = a.reshape(len(a), -1)
    return [f"{name}_{k}" for k in range(flat.shape[1])], flat


def ply_bytes(mesh: Mesh, unit="mm", ascii: bool = False, double: bool = True) -> bytes:
    scale = unit_scale(unit)
    vtype = "double" if double else "float"
    vcols = [("x", vtype), ("y", vtype), ("z", vtype)]
    vdata = [mesh.vertices / scale]
    if "normal" in mesh.vertex_attr:
        vcols += [("nx", "float"), ("ny", "float"), ("nz", "float")]
        vdata.append(np.asarray(mesh.vertex_attr["normal"], dtype=np.float64))
    for name, arr in mesh.vertex_attr.items():
        if name == "normal":
            continue
        cols, block = _attr_columns(name, arr)
        vcols += [(c, "float") for c in cols]
        vdata.append(block)
    fcols = []
    fdata = []
    if np.any(mesh.face_group >= 0):
        fcols.append(("face_group", "int"))
        fdata.append(mesh.face_group.astype(np.float64)[:, None])
    for name, arr in mesh.face_attr.items():
        cols, block = _attr_columns(name, arr)
        fcols += [(c, "float") for c in cols]
        fdata.append(block)
    header = ["ply", f"format {'ascii' if ascii else 'binary_little_endian'} 1.0", "comment cnctwin PLY, units " + str(unit),
              f"element vertex {mesh.n_vertices}"]
    header += [f"property {t} {n}" for n, t in vcols]
    header += [f"element face {mesh.n_faces}", "property list uchar int vertex_indices"]
    header += [f"property {t} {n}" for n, t in fcols]
    header.append("end_header")
    head = ("\n".join(header) + "\n").encode("ascii")
    V = np.concatenate(vdata, axis=1) if vdata else np.zeros((0, 0))
    F = np.concatenate(fdata, axis=1) if fdata else np.zeros((mesh.n_faces, 0))
    if ascii:
        lines = []
        for row in V.tolist():
            lines.append(" ".join(repr(v) if t in ("double",) else repr(float(np.float32(v)))
                                  for v, (_, t) in zip(row, vcols)))
        for f, row in zip(mesh.faces.tolist(), F.tolist()):
            extra = [str(int(v)) if t == "int" else repr(float(np.float32(v))) for v, (_, t) in zip(row, fcols)]
            lines.append(" ".join(["3"] + [str(i) for i in f] + extra))
        return head + ("\n".join(lines) + "\n").encode("ascii")
    vdt = np.dtype([(n, "<" + _TYPES[t]) for n, t in vcols])
    vrec = np.zeros(mesh.n_vertices, vdt)
    for k, (n, _) in enumerate(vcols):
        vrec[n] = V[:, k]
    fdt = np.dtype([("count", "u1"), ("idx", "<i4", (3,))] + [(n, "<" + _TYPES[t]) for n, t in fcols])
    frec = np.zeros(mesh.n_faces, fdt)
    frec["count"] = 3
    frec["idx"] = mesh.faces
    for k, (n, _) in enumerate(fcols):
        frec[n] = F[:, k]
    return head + vrec.tobytes() + frec.tobytes()


def write_ply(path, mesh: Mesh, unit="mm", ascii: bool = False, double: bool = True) -> None:
    Path(path).write_bytes(ply_bytes(mesh, unit, ascii, double))


def read_ply(path, unit="mm", report: dict | None = None) -> Mesh:
    path = Path(path)
    data = path.read_bytes()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise InputError("not a PLY file", str(path))
    body_start = data.index(b"\n", end) + 1
    header = data[:body_start].decode("ascii", "replace").splitlines()
    fmt = None
    elements: list[dict] = []
    for line in header[1:]:
        parts = line.split()
        if not parts or parts[0] in ("comment", "obj_info", "end_header"):
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append({"name": parts[1], "count": int(parts[2]), "props": []})
        elif parts[0] == "property":
            if parts[1] == "list":
                elements[-1]["props"].append((parts[4], "list", parts[2], parts[3]))
            else:
                elements[-1]["props"].append((parts[2], parts[1]))
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise InputError(f"unsupported PLY format {fmt!r}", str(path))
    endian = "<" if fmt == "binary_little_endian" else ">"
    scale = unit_scale(unit)
    pos = body_start
    tokens = iter(data[body_start:].split()) if fmt == "ascii" else None
    out: dict[str, dict] = {}
    skipped = []
    for el in elements:
        cols: dict[str, list] = {p[0]: [] for p in el["props"]}
        if fmt == "ascii":
            for _ in range(el["count"]):
                for p in el["props"]:
                    if p[1] == "list":
                        k = int(next(tokens))
                        cols[p[0]].append([int(next(tokens)) for _ in range(k)])
                    else:
                        cols[p[0]].append(float(next(tokens)))
        elif all(p[1] != "list" for p in el["props"]):
            dt = np.dtype([(p[0], endian + _TYPES[p[1]]) for p in el["props"]])
            rec = np.frombuffer(data, dt, count=el["count"], offset=pos)
            pos += dt.itemsize * el["count"]
            cols = {p[0]: rec[p[0]].astype(np.float64) for p in el["props"]}
        else:
            for _ in range(el["count"]):
                for p in el["props"]:
                    if p[1] == "list":
                        ct = np.dtype(endian + _TYPES[p[2]])
                        it = np.dtype(endian + _TYPES[p[3]])
                        k = int(np.frombuffer(data, ct, 1, pos)[0])
                        pos += ct.itemsize
                        cols[p[0]].append(np.frombuffer(data, it, k, pos).astype(np.int64).tolist())
                        pos += it.itemsize * k
                    else:
                        st = np.dtype(endian + _TYPES[p[1]])
                        cols[p[0]].append(float(np.frombuffer(data, st, 1, pos)[0]))
                        pos += st.itemsize
        if el["name"] in ("vertex", "face"):
            out[el["name"]] = cols
        else:
            skipped.append(el["name"])
    if report is not None:
        report.update({"skipped": skipped})
    vcols = out.get("vertex", {})
    V = np.column_stack([np.asarray(vcols.get(k, []), dtype=np.float64) for k in ("x", "y", "z")]) * scale
    vertex_attr = {}
    if all(k in vcols for k in ("nx", "ny", "nz")):
        vertex_attr["normal"] = np.column_stack([np.asarray(vcols[k], dtype=np.float64) for k in ("nx", "ny", "nz")])
    vertex_attr.update(_group_columns({k: v for k, v in vcols.items() if k not in ("x", "y", "z", "nx", "ny", "nz")}))
    fcols = out.get("face", {})
    list_key = "vertex_indices" if "vertex_indices" in fcols else ("vertex_index" if "vertex_index" in fcols else None)
    faces, origin = [], []
    for fi, poly in enumerate(fcols.get(list_key, []) if list_key else []):
        for k in range(1, len(poly) - 1):
            faces.append((poly[0], poly[k], poly[k + 1]))
            origin.append(fi)
    origin = np.asarray(origin, dtype=np.int64)
    groups = np.asarray(fcols["face_group"], dtype=np.int64)[origin] if "face_group" in fcols else None
    face_attr = {k: np.asarray(v)[origin] for k, v in _group_columns(
        {k: v for k, v in fcols.items() if k not in (list_key, "face_group")}).items()}
    return Mesh(V, np.asarray(faces, dtype=np.int64).reshape(-1, 3), groups, vertex_attr, face_attr, name=path.stem)


def _group_columns(cols: dict) -> dict:
    """Columns ``a_0 .. a_k`` back into one ``a`` [n, k+1] array; single columns stay [n]."""
    out: dict[str, np.ndarray] = {}
    stems: dict[str, list] = {}
    for name in cols:
        stem, _, idx = name.rpartition("_")
        if stem and idx.isdigit():
            stems.setdefault(stem, []).append((int(idx), name))
        else:
            out[name] = np.asarray(cols[name], dtype=np.float64)
    for stem, items in stems.items():
        items.sort()
        if [i for i, _ in items] == list(range(len(items))) and stem not in out:
            out[stem] = np.column_stack([np.asarray(cols[n], dtype=np.float64) for _, n in items])
        else:
            for _, n in items:
                out[n] = np.asarray(cols[n], dtype=np.float64)
    return out


__all__ = ["write_ply", "read_ply", "ply_bytes"]
