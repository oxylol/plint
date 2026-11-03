"""glTF 2.0 binary (GLB) writer and reader (Khronos glTF 2.0; KHR_mesh_quantization) for the run
directory's scene and keyframe files (docs/architecture.md 16, docs/models/geom.md "Files").

Write (:func:`glb_bytes`): a list of :class:`GlbNode` (name, mesh, local 4x4 pose, children, colour,
extras) becomes nodes, meshes, accessors and buffer views in one binary buffer. Per mesh: ``POSITION``
(float32, accessor min/max as the specification requires), ``NORMAL`` (the mesh's ``normal`` vertex
attribute, else angle-weighted vertex normals; ``normals=False`` omits it), ``indices`` (uint16 when the
vertices fit, else uint32) and every float vertex attribute as an application-specific attribute
``_<name>`` (float32, 1-4 components; the specification reserves the leading underscore for them).
``split_groups`` writes one primitive per face group with ``extras.face_group`` (B-rep face ids,
machine component patches). ``quantize`` stores positions as uint16 with the
``KHR_mesh_quantization`` extension: ``p = lo + q * step`` with one step for all axes
(``max extent / 65535``, so the dequantisation is a uniform scale and normals need no correction), put
in an extra child node ``<name>#q`` carrying the dequantisation matrix; normals as normalised int8. The
position error is at most ``step / 2`` per axis (0.06 um for an 8 mm brick). Output bytes depend only on
the input: JSON with sorted keys, fixed separators, shortest round-trip floats, zero padding.

Read (:func:`read_glb`): GLB files with one embedded buffer: nodes (matrix or TRS), meshes, triangle
primitives (strips and fans converted), float and (normalised) integer accessors with byte strides, the
quantization extension (dequantised through the node transforms). Sparse accessors, external buffers and
compression extensions (Draco, meshopt) are refused with an error; point and line primitives carry no
surface and are counted in ``GlbScene.skipped``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.errors import InputError
from .. import transform as tf
from ..mesh import Mesh, merge

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
FLOAT, UBYTE, BYTE, USHORT, SHORT, UINT = 5126, 5121, 5120, 5123, 5122, 5125
_COMP = {FLOAT: ("<f4", 4), UBYTE: ("u1", 1), BYTE: ("i1", 1), USHORT: ("<u2", 2), SHORT: ("<i2", 2), UINT: ("<u4", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_TYPE_OF = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}
QUANT_LEVELS = 65535


@dataclass
class GlbNode:
    """One node: ``matrix`` is its pose in the parent (column vectors, metres); ``children`` index the list."""

    name: str = ""
    mesh: Mesh | None = None
    matrix: np.ndarray | None = None
    children: list = field(default_factory=list)
    color_rgba: tuple | None = None
    extras: dict | None = None


class _Builder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, data: bytes, target: int | None, stride: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        entry = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            entry["target"] = target
        if stride is not None:
            entry["byteStride"] = stride
        self.blob.extend(data)
        self.views.append(entry)
        return len(self.views) - 1

    def accessor(self, view: int, comp: int, count: int, kind: str, lo=None, hi=None, normalized=False) -> int:
        entry = {"bufferView": view, "componentType": comp, "count": int(count), "type": kind}
        if lo is not None:
            entry["min"] = [_num(v) for v in lo]
            entry["max"] = [_num(v) for v in hi]
        if normalized:
            entry["normalized"] = True
        self.accessors.append(entry)
        return len(self.accessors) - 1


def _num(v):
    f = float(v)
    return int(f) if f.is_integer() and abs(f) < 2 ** 53 else f


def _padded(arr: np.ndarray, stride: int) -> bytes:
    """Rows of ``arr`` (C order) each padded with zeros to ``stride`` bytes."""
    rows = np.ascontiguousarray(arr).view(np.uint8).reshape(len(arr), -1)
    out = np.zeros((len(arr), stride), np.uint8)
    out[:, :rows.shape[1]] = rows
    return out.tobytes()


def glb_bytes(nodes, quantize: bool = False, normals: bool = True, split_groups: bool = False,
              roots=None, extras: dict | None = None) -> bytes:
    """GLB of a list of :class:`GlbNode` (or a single Mesh). ``roots`` lists the scene's root nodes (default:
    every node that is no other node's child)."""
    if isinstance(nodes, Mesh):
        nodes = [GlbNode(name=nodes.name or "mesh", mesh=nodes)]
    nodes = list(nodes)
    b = _Builder()
    gl_nodes: list[dict] = []
    gl_meshes: list[dict] = []
    materials: list[dict] = []
    material_of: dict[tuple, int] = {}
    extra_children: dict[int, list[int]] = {}
    for i, node in enumerate(nodes):
        entry: dict = {"name": node.name or f"node{i}"}
        if node.matrix is not None:
            M = np.asarray(node.matrix, dtype=np.float64)
            if not np.allclose(M, np.eye(4), rtol=0.0, atol=0.0):
                entry["matrix"] = [float(v) for v in M.T.reshape(-1)]
        if node.extras:
            entry["extras"] = node.extras
        gl_nodes.append(entry)
    for i, node in enumerate(nodes):
        if node.mesh is None or node.mesh.n_faces == 0:
            continue
        mat = None
        if node.color_rgba is not None:
            key = tuple(float(c) for c in node.color_rgba)
            if len(key) != 4:
                raise ValueError("color_rgba needs four components in [0, 1]")
            if key not in material_of:
                material_of[key] = len(materials)
                materials.append({"pbrMetallicRoughness": {"baseColorFactor": list(key), "metallicFactor": 0.0,
                                                           "roughnessFactor": 0.8}, "doubleSided": False})
            mat = material_of[key]
        mesh_index, dequant = _write_mesh(b, node.mesh, quantize, normals, split_groups, mat)
        gl_meshes.append(mesh_index)
        if dequant is None:
            gl_nodes[i]["mesh"] = len(gl_meshes) - 1
        else:
            child = {"name": f"{gl_nodes[i]['name']}#q", "mesh": len(gl_meshes) - 1,
                     "matrix": [float(v) for v in dequant.T.reshape(-1)]}
            gl_nodes.append(child)
            extra_children.setdefault(i, []).append(len(gl_nodes) - 1)
    child_set: set[int] = set()
    for i, node in enumerate(nodes):
        kids = [int(c) for c in node.children] + extra_children.get(i, [])
        for c in node.children:
            if not 0 <= int(c) < len(nodes) or int(c) == i:
                raise ValueError(f"node {i} has an invalid child {c}")
        if kids:
            gl_nodes[i]["children"] = kids
            child_set.update(int(c) for c in node.children)
    root_list = [int(r) for r in roots] if roots is not None else [i for i in range(len(nodes)) if i not in child_set]
    doc: dict = {"asset": {"version": "2.0", "generator": "cnctwin"}, "scene": 0,
                 "scenes": [{"nodes": root_list}], "nodes": gl_nodes}
    if gl_meshes:
        doc["meshes"] = gl_meshes
        doc["accessors"] = b.accessors
        doc["bufferViews"] = b.views
    if materials:
        doc["materials"] = materials
    if quantize and gl_meshes:
        doc["extensionsUsed"] = ["KHR_mesh_quantization"]
        doc["extensionsRequired"] = ["KHR_mesh_quantization"]
    if extras:
        doc["extras"] = extras
    while len(b.blob) % 4:
        b.blob.append(0)
    if b.blob:
        doc["buffers"] = [{"byteLength": len(b.blob)}]
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    text += b" " * ((4 - len(text) % 4) % 4)
    chunks = struct.pack("<II", len(text), CHUNK_JSON) + text
    if b.blob:
        chunks += struct.pack("<II", len(b.blob), CHUNK_BIN) + bytes(b.blob)
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(chunks)) + chunks


def _write_mesh(b: _Builder, mesh: Mesh, quantize: bool, normals: bool, split_groups: bool, material):
    attrs: dict[str, int] = {}
    dequant = None
    v = mesh.vertices
    if quantize:
        lo = v.min(axis=0)
        extent = float(np.max(v.max(axis=0) - lo))
        step = extent / QUANT_LEVELS if extent > 0.0 else 1.0
        q = np.clip(np.rint((v - lo) / step), 0, QUANT_LEVELS).astype("<u2")
        view = b.view(_padded(q, 8), ARRAY_BUFFER, stride=8)
        attrs["POSITION"] = b.accessor(view, USHORT, len(q), "VEC3", q.min(axis=0), q.max(axis=0))
        dequant = np.diag([step, step, step, 1.0])
        dequant[:3, 3] = lo
    else:
        p = v.astype("<f4")
        view = b.view(p.tobytes(), ARRAY_BUFFER)
        attrs["POSITION"] = b.accessor(view, FLOAT, len(p), "VEC3", p.min(axis=0), p.max(axis=0))
    if normals:
        n = mesh.vertex_attr.get("normal")
        n = mesh.vertex_normals() if n is None else np.asarray(n, dtype=np.float64)
        if quantize:
            nq = np.clip(np.rint(n * 127.0), -127, 127).astype("i1")
            view = b.view(_padded(nq, 4), ARRAY_BUFFER, stride=4)
            attrs["NORMAL"] = b.accessor(view, BYTE, len(nq), "VEC3", normalized=True)
        else:
            view = b.view(n.astype("<f4").tobytes(), ARRAY_BUFFER)
            attrs["NORMAL"] = b.accessor(view, FLOAT, len(n), "VEC3")
    for name, arr in mesh.vertex_attr.items():
        if name == "normal":
            continue
        a = np.asarray(arr)
        if not np.issubdtype(a.dtype, np.floating) and not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"vertex attribute {name!r} is not numeric")
        a = a.astype("<f4").reshape(len(a), -1)
        if a.shape[1] > 4:
            raise ValueError(f"vertex attribute {name!r} has {a.shape[1]} components (glTF allows 1-4)")
        semantic = name if name.startswith("_") else "_" + name
        view = b.view(a.tobytes(), ARRAY_BUFFER)
        attrs[semantic] = b.accessor(view, FLOAT, len(a), _TYPE_OF[a.shape[1]])
    index_type, comp = ("<u2", USHORT) if mesh.n_vertices <= 65535 else ("<u4", UINT)
    prims = []
    groups = [None] if not split_groups else sorted(set(mesh.face_group.tolist()))
    for g in groups:
        faces = mesh.faces if g is None else mesh.faces[mesh.face_group == g]
        idx = faces.astype(index_type).reshape(-1)
        view = b.view(idx.tobytes(), ELEMENT_ARRAY_BUFFER)
        prim = {"attributes": dict(attrs), "indices": b.accessor(view, comp, len(idx), "SCALAR"), "mode": 4}
        if material is not None:
            prim["material"] = material
        if g is not None:
            prim["extras"] = {"face_group": int(g)}
            if g in mesh.group_names:
                prim["extras"]["group_name"] = mesh.group_names[g]
        prims.append(prim)
    return {"name": mesh.name or "mesh", "primitives": prims}, dequant


def write_glb(path, nodes, **kwargs) -> None:
    Path(path).write_bytes(glb_bytes(nodes, **kwargs))


# ============================================================================ reader

@dataclass
class GlbScene:
    """Nodes of a GLB file: each ``{name, index, world (4x4), local (4x4), mesh (Mesh in node coordinates or
    None), extras}``; ``skipped`` counts non-triangle primitives."""

    nodes: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    json: dict = field(default_factory=dict)

    def mesh(self, world: bool = True) -> Mesh:
        """All node meshes in one mesh (in world coordinates by default); face group = node order."""
        parts = []
        for k, node in enumerate(n for n in self.nodes if n["mesh"] is not None):
            m = node["mesh"].transformed(node["world"]) if world else node["mesh"].copy()
            parts.append(m)
        if not parts:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        return merge(parts, offset_groups=False) if len(parts) > 1 else parts[0]

    def find(self, name: str):
        for node in self.nodes:
            if node["name"] == name:
                return node
        return None


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    T = np.eye(4)
    if "scale" in node:
        T = np.diag(list(node["scale"]) + [1.0])
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        T = tf.pose(tf.matrix_from_quat([w, x, y, z])) @ T
    if "translation" in node:
        T = tf.translation(node["translation"]) @ T
    return T


def _accessor_array(doc: dict, binary: bytes, index: int, path: str) -> np.ndarray:
    acc = doc["accessors"][index]
    if "sparse" in acc:
        raise InputError("sparse glTF accessors are not supported", path)
    if "bufferView" not in acc:
        raise InputError("a glTF accessor without bufferView (all zeros) is not supported", path)
    view = doc["bufferViews"][acc["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise InputError("glTF data outside the GLB binary chunk is not supported", path)
    dtype, size = _COMP[acc["componentType"]]
    ncomp = _NCOMP[acc["type"]]
    count = int(acc["count"])
    elem = size * ncomp
    stride = int(view.get("byteStride", elem))
    offset = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    if count and offset + stride * (count - 1) + elem > len(binary):
        raise InputError("a glTF accessor reaches past the binary chunk", path)
    raw = np.ndarray((count, ncomp), dtype=np.dtype(dtype), buffer=binary, offset=offset, strides=(stride, size))
    out = np.array(raw, dtype=np.float64)
    if acc.get("normalized"):
        if acc["componentType"] == BYTE:
            out = np.maximum(out / 127.0, -1.0)
        elif acc["componentType"] == SHORT:
            out = np.maximum(out / 32767.0, -1.0)
        elif acc["componentType"] == UBYTE:
            out = out / 255.0
        elif acc["componentType"] == USHORT:
            out = out / 65535.0
    return out


def read_glb(source) -> GlbScene:
    """Scene of a GLB file (path or bytes)."""
    data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    path = "<bytes>" if isinstance(source, (bytes, bytearray)) else str(source)
    if len(data) < 20:
        raise InputError("not a GLB file", path)
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise InputError("not a glTF 2.0 binary file", path)
    pos = 12
    doc = None
    binary = b""
    while pos < min(length, len(data)):
        clen, ctype = struct.unpack_from("<II", data, pos)
        chunk = data[pos + 8: pos + 8 + clen]
        if ctype == CHUNK_JSON:
            doc = json.loads(chunk.decode("utf-8"))
        elif ctype == CHUNK_BIN and not binary:
            binary = bytes(chunk)
        pos += 8 + clen
    if doc is None:
        raise InputError("GLB file without a JSON chunk", path)
    unsupported = [e for e in doc.get("extensionsRequired", []) if e != "KHR_mesh_quantization"]
    if unsupported:
        raise InputError(f"GLB needs unsupported extensions {unsupported}", path)
    skipped: dict[str, int] = {}
    meshes = []
    for gm in doc.get("meshes", []):
        parts = []
        for prim in gm.get("primitives", []):
            mode = prim.get("mode", 4)
            if mode not in (4, 5, 6):
                skipped[f"mode{mode}"] = skipped.get(f"mode{mode}", 0) + 1
                continue
            attrs = prim["attributes"]
            pos_arr = _accessor_array(doc, binary, attrs["POSITION"], path)
            if "indices" in prim:
                idx = _accessor_array(doc, binary, prim["indices"], path).reshape(-1).astype(np.int64)
            else:
                idx = np.arange(len(pos_arr), dtype=np.int64)
            if mode == 4:
                faces = idx.reshape(-1, 3)
            elif mode == 5:
                faces = np.array([(idx[i], idx[i + 1], idx[i + 2]) if i % 2 == 0 else (idx[i + 1], idx[i], idx[i + 2])
                                  for i in range(len(idx) - 2)], dtype=np.int64).reshape(-1, 3)
            else:
                faces = np.array([(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)],
                                 dtype=np.int64).reshape(-1, 3)
            vattr = {}
            for sem, a in attrs.items():
                if sem == "NORMAL":
                    vattr["normal"] = _accessor_array(doc, binary, a, path)
                elif sem.startswith("_"):
                    arr = _accessor_array(doc, binary, a, path)
                    vattr[sem[1:]] = arr[:, 0] if arr.shape[1] == 1 else arr
            group = int(prim.get("extras", {}).get("face_group", -1))
            names = {group: prim["extras"]["group_name"]} if group >= 0 and "group_name" in prim.get("extras", {}) \
                else {}
            parts.append(Mesh(pos_arr, faces, np.full(len(faces), group, np.int64), vattr, group_names=names,
                              name=gm.get("name", "")))
        if not parts:
            meshes.append(None)
        elif len(parts) == 1:
            meshes.append(parts[0])
        else:
            same = all(np.array_equal(p.vertices, parts[0].vertices) for p in parts[1:])
            if same:
                faces = np.concatenate([p.faces for p in parts])
                groups = np.concatenate([p.face_group for p in parts])
                names = {}
                for p in parts:
                    names.update(p.group_names)
                meshes.append(Mesh(parts[0].vertices, faces, groups, parts[0].vertex_attr, group_names=names,
                                   name=parts[0].name))
            else:
                meshes.append(merge(parts, name=gm.get("name", "")))
    gl_nodes = doc.get("nodes", [])
    scene_index = doc.get("scene", 0)
    scenes = doc.get("scenes", [{"nodes": list(range(len(gl_nodes)))}])
    roots = scenes[scene_index]["nodes"] if scenes else []
    out_nodes = []
    stack = [(int(r), np.eye(4)) for r in reversed(roots)]
    visited = set()
    while stack:
        index, parent = stack.pop()
        if index in visited:
            raise InputError("glTF node hierarchy has a cycle", path)
        visited.add(index)
        node = gl_nodes[index]
        local = _node_matrix(node)
        world = parent @ local
        mesh = meshes[node["mesh"]] if "mesh" in node else None
        out_nodes.append({"name": node.get("name", ""), "index": index, "world": world, "local": local,
                          "mesh": mesh, "extras": node.get("extras", {})})
        for c in reversed(node.get("children", [])):
            stack.append((int(c), world))
    return GlbScene(out_nodes, doc.get("extras", {}), skipped, doc)


__all__ = ["GlbNode", "GlbScene", "glb_bytes", "write_glb", "read_glb", "QUANT_LEVELS"]
