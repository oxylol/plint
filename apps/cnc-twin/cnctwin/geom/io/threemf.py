"""3MF (3MF Consortium core specification 1.3): read and write meshes with units, objects, components and
build items (docs/models/geom.md "Files").

Read: the model part named by ``_rels/.rels`` (StartPart relationship), the ``unit`` of the model, every
``object`` of type model/solidsupport/support/other with its ``mesh`` or ``components`` (transforms
composed recursively), and every build ``item`` with its transform. Transforms are the specification's
3x4 row-vector matrices ``m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32`` (p' = p M); here they become
4x4 column-vector poses (R[i][j] = m_ji, t = (m30, m31, m32)). ``read_3mf`` returns one mesh per build item
(``merge=True``: one mesh, face group = build item, group name = object name). Material and colour
properties (pid/p1..p3, the materials extension) are appearance, not geometry: not read. External model
parts of the production extension (components with ``p:path``) are refused with an error.
Write: one object per mesh, a build item per (mesh, transform), unit ``millimeter`` by default; the zip
entries have fixed timestamps and order, so the bytes depend only on the input.
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ...core.errors import InputError
from ..mesh import Mesh, merge
from .units import THREEMF_UNITS, unit_scale

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PROD_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
CONTENT_TYPES = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                 '</Types>\n')
RELS = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Relationships xmlns="{REL_NS}">'
        f'<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="{MODEL_REL}"/>'
        '</Relationships>\n')


def _q(tag: str) -> str:
    return f"{{{CORE_NS}}}{tag}"


def _transform(text: str | None) -> np.ndarray:
    if not text:
        return np.eye(4)
    m = [float(v) for v in text.split()]
    if len(m) != 12:
        raise ValueError(f"a 3MF transform has 12 numbers, not {len(m)}")
    T = np.eye(4)
    T[:3, :3] = np.array(m[:9]).reshape(3, 3).T
    T[:3, 3] = m[9:12]
    return T


def _transform_text(T: np.ndarray) -> str:
    R = T[:3, :3]
    vals = [R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1], R[0, 2], R[1, 2], R[2, 2], T[0, 3], T[1, 3], T[2, 3]]
    return " ".join(repr(float(v)) for v in vals)


def _model_part(zf: zipfile.ZipFile, path: Path) -> str:
    names = set(zf.namelist())
    if "_rels/.rels" in names:
        root = ET.fromstring(zf.read("_rels/.rels"))
        for rel in root.iter(f"{{{REL_NS}}}Relationship"):
            if rel.get("Type") == MODEL_REL:
                target = rel.get("Target", "").lstrip("/")
                if target in names:
                    return target
    if "3D/3dmodel.model" in names:
        return "3D/3dmodel.model"
    raise InputError("no 3D model part in the 3MF package", str(path))


def read_3mf(path, merge_items: bool = True):
    """Meshes (metres) of a 3MF file: one per build item, or one merged mesh (``merge_items``)."""
    path = Path(path)
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise InputError("not a 3MF (zip) package", str(path)) from None
    with zf:
        part = _model_part(zf, path)
        objects: dict[str, dict] = {}
        items: list[tuple[str, np.ndarray]] = []
        unit = "millimeter"
        current: dict | None = None
        verts: list = []
        tris: list = []
        with zf.open(part) as handle:
            for event, el in ET.iterparse(handle, events=("start", "end")):
                tag = el.tag
                if event == "start":
                    if tag == _q("model"):
                        unit = el.get("unit", "millimeter")
                    elif tag == _q("object"):
                        current = {"id": el.get("id"), "name": el.get("name", ""), "type": el.get("type", "model"),
                                   "mesh": None, "components": []}
                        verts, tris = [], []
                    continue
                if tag == _q("vertex"):
                    verts.append((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
                    el.clear()
                elif tag == _q("triangle"):
                    tris.append((int(el.get("v1")), int(el.get("v2")), int(el.get("v3"))))
                    el.clear()
                elif tag == _q("mesh") and current is not None:
                    current["mesh"] = (np.array(verts, dtype=np.float64).reshape(-1, 3),
                                       np.array(tris, dtype=np.int64).reshape(-1, 3))
                    el.clear()
                elif tag == _q("component") and current is not None:
                    if el.get(f"{{{PROD_NS}}}path"):
                        raise InputError("3MF components in external model parts (production extension) are not "
                                         "supported: export the part as a single model", str(path))
                    current["components"].append((el.get("objectid"), _transform(el.get("transform"))))
                elif tag == _q("object") and current is not None:
                    objects[current["id"]] = current
                    current = None
                    el.clear()
                elif tag == _q("item"):
                    if el.get(f"{{{PROD_NS}}}path"):
                        raise InputError("3MF build items in external model parts are not supported", str(path))
                    items.append((el.get("objectid"), _transform(el.get("transform"))))
    scale = unit_scale(unit)
    meshes = []
    for k, (oid, T) in enumerate(items):
        if oid not in objects:
            raise InputError(f"build item {k} refers to the missing object {oid}", str(path))
        mesh = _resolve(objects, oid, T, 0, path)
        mesh = mesh.transformed(np.diag([scale, scale, scale, 1.0]))
        mesh.name = objects[oid]["name"] or f"object{oid}"
        meshes.append(mesh)
    if not merge_items:
        return meshes
    for k, m in enumerate(meshes):
        m.face_group = np.full(m.n_faces, k, np.int32)
        m.group_names = {k: m.name}
    out = merge(meshes, name=path.stem)
    return out


def _resolve(objects: dict, oid: str, T: np.ndarray, depth: int, path: Path) -> Mesh:
    if depth > 64:
        raise InputError("3MF components nest deeper than 64 levels (a cycle?)", str(path))
    obj = objects.get(oid)
    if obj is None:
        raise InputError(f"3MF component refers to the missing object {oid}", str(path))
    parts = []
    if obj["mesh"] is not None:
        v, f = obj["mesh"]
        if len(f) and (f.min() < 0 or f.max() >= len(v)):
            raise InputError(f"3MF object {oid} has a triangle index outside its vertices", str(path))
        parts.append(Mesh(v, f).transformed(T))
    for cid, CT in obj["components"]:
        parts.append(_resolve(objects, cid, T @ CT, depth + 1, path))
    if not parts:
        raise InputError(f"3MF object {oid} has neither a mesh nor components", str(path))
    return merge(parts)


def threemf_bytes(meshes, transforms=None, names=None, unit: str = "millimeter") -> bytes:
    """3MF package of one or more meshes (metres), each placed by an optional 4x4 transform."""
    if isinstance(meshes, Mesh):
        meshes = [meshes]
    meshes = list(meshes)
    transforms = [None] * len(meshes) if transforms is None else list(transforms)
    names = [m.name for m in meshes] if names is None else list(names)
    scale = unit_scale(unit)
    unit_attr = THREEMF_UNITS.get(scale) or {v: k for k, v in THREEMF_UNITS.items()}.get(unit)
    if unit_attr is None:
        raise ValueError(f"3MF supports micron, millimeter, centimeter, inch, foot and meter, not {unit!r}")
    out = [f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="{unit_attr}" xml:lang="en-US" xmlns="{CORE_NS}">',
           "<resources>"]
    for k, mesh in enumerate(meshes):
        name = (names[k] or f"object{k + 1}").replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
        out.append(f'<object id="{k + 1}" type="model" name="{name}"><mesh><vertices>')
        out.extend(f'<vertex x="{x!r}" y="{y!r}" z="{z!r}"/>' for x, y, z in (mesh.vertices / scale).tolist())
        out.append("</vertices><triangles>")
        out.extend(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces.tolist())
        out.append("</triangles></mesh></object>")
    out.append("</resources><build>")
    for k, T in enumerate(transforms):
        if T is None:
            out.append(f'<item objectid="{k + 1}"/>')
        else:
            M = np.asarray(T, dtype=np.float64).copy()
            M[:3, 3] = M[:3, 3] / scale
            out.append(f'<item objectid="{k + 1}" transform="{_transform_text(M)}"/>')
    out.append("</build></model>\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in (("[Content_Types].xml", CONTENT_TYPES), ("_rels/.rels", RELS),
                           ("3D/3dmodel.model", "\n".join(out))):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, text.encode("utf-8"), compresslevel=6)
    return buffer.getvalue()


def write_3mf(path, meshes, transforms=None, names=None, unit: str = "millimeter") -> None:
    Path(path).write_bytes(threemf_bytes(meshes, transforms, names, unit))


__all__ = ["read_3mf", "write_3mf", "threemf_bytes"]
