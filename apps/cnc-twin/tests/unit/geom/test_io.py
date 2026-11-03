"""Mesh files: round trips, units, groups, transforms, quantisation, custom attributes, deterministic bytes."""

import math
import struct
import zipfile

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.geom import io as gio
from cnctwin.geom import transform as tf
from cnctwin.geom.io import gltf, obj, ply, stl, threemf
from cnctwin.geom.mesh import merge
from cnctwin.geom.primitives import box_mesh, cylinder_mesh


@pytest.fixture()
def part():
    m = merge([box_mesh([0, 0, 0], [0.1, 0.08, 0.03]), cylinder_mesh([0.2, 0, 0], [0.2, 0, 0.05], 0.01, 1e-4)],
              offset_groups=True, name="part")
    m.vertex_attr["deviation_m"] = np.linspace(0.0, 1e-5, m.n_vertices)
    return m


def same_geometry(a, b, tol):
    assert a.n_faces == b.n_faces
    assert math.isclose(a.volume(), b.volume(), rel_tol=tol)
    assert np.allclose(np.sort(a.triangles().reshape(-1, 3), axis=0), np.sort(b.triangles().reshape(-1, 3), axis=0),
                       atol=tol * 0.2)


def test_stl_binary_and_ascii(tmp_path, part):
    stl.write_stl(tmp_path / "b.stl", part)
    data = (tmp_path / "b.stl").read_bytes()
    assert data[:80] == stl.BINARY_HEADER and not data.startswith(b"solid")
    assert struct.unpack("<I", data[80:84])[0] == part.n_faces and len(data) == 84 + 50 * part.n_faces
    back = stl.read_stl(tmp_path / "b.stl")
    same_geometry(part, back, 1e-6)
    assert back.n_vertices == part.n_vertices and back.is_watertight() == part.is_watertight()
    stl.write_stl(tmp_path / "a.stl", part, ascii=True)
    back = stl.read_stl(tmp_path / "a.stl")
    same_geometry(part, back, 1e-12)
    assert stl.read_stl(tmp_path / "a.stl", unit="m").bounds()[1][0] > 100.0      # numbers read as metres
    assert stl.stl_bytes(part) == stl.stl_bytes(part.copy())


def test_stl_multiple_ascii_solids_become_groups(tmp_path):
    text = ("solid first\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\n"
            "endsolid first\nsolid second\nfacet normal 0 0 1\nouter loop\nvertex 0 0 1\nvertex 1 0 1\nvertex 0 1 1\n"
            "endloop\nendfacet\nendsolid second\n")
    (tmp_path / "two.stl").write_text(text)
    m = stl.read_stl(tmp_path / "two.stl")
    assert m.face_group.tolist() == [0, 1] and m.group_names == {0: "first", 1: "second"}
    (tmp_path / "bad.stl").write_bytes(b"\x00" * 90)
    with pytest.raises(InputError):
        stl.read_stl(tmp_path / "bad.stl")


def test_obj_groups_and_index_forms(tmp_path, part):
    obj.write_obj(tmp_path / "p.obj", part)
    back = obj.read_obj(tmp_path / "p.obj")
    same_geometry(part, back, 1e-12)
    assert np.array_equal(back.face_group, part.face_group)
    assert back.group_names[0] == part.group_names[0]
    text = ("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\ng quad\nusemtl steel\nf 1/1/1 2/1/1 3/1/1 4/1/1\n"
            "g tri\nf -4//1 -3//1 -1//1\n")
    (tmp_path / "q.obj").write_text(text)
    report = {}
    m = obj.read_obj(tmp_path / "q.obj", report=report)
    assert m.n_faces == 3 and m.face_group.tolist() == [0, 0, 1] and m.group_names == {0: "quad", 1: "tri"}
    assert report["skipped"] == {"usemtl": 1, "vn": 1, "vt": 1}
    (tmp_path / "e.obj").write_text("v 0 0 0\nf 1 2 3\n")
    with pytest.raises(InputError):
        obj.read_obj(tmp_path / "e.obj")


def test_ply_binary_and_ascii_with_attributes(tmp_path, part):
    part.vertex_attr["normal"] = part.vertex_normals()
    part.face_attr["wear_m"] = np.arange(part.n_faces, dtype=np.float64) * 1e-6
    for ascii_ in (False, True):
        ply.write_ply(tmp_path / "p.ply", part, ascii=ascii_)
        back = ply.read_ply(tmp_path / "p.ply")
        same_geometry(part, back, 1e-12)
        assert np.array_equal(back.face_group, part.face_group)
        assert np.allclose(back.vertex_attr["normal"], part.vertex_attr["normal"], atol=1e-7)
        assert np.allclose(back.vertex_attr["deviation_m"], part.vertex_attr["deviation_m"], rtol=1e-6)
        assert np.allclose(back.face_attr["wear_m"], part.face_attr["wear_m"], rtol=1e-6)
    assert ply.ply_bytes(part) == ply.ply_bytes(part.copy())


def test_3mf_units_items_transforms_and_components(tmp_path, part):
    T = tf.pose(tf.rot_z(0.5), [0.01, 0.02, 0.03])
    box = box_mesh([0, 0, 0], [0.01, 0.01, 0.01], name="cube")
    threemf.write_3mf(tmp_path / "a.3mf", [part, box], transforms=[None, T], unit="inch")
    items = threemf.read_3mf(tmp_path / "a.3mf", merge_items=False)
    assert [m.name for m in items] == ["part", "cube"]
    same_geometry(part, items[0], 1e-12)
    assert np.allclose(items[1].vertices, tf.apply(T, box.vertices), atol=1e-15)
    merged = threemf.read_3mf(tmp_path / "a.3mf")
    assert set(merged.face_group.tolist()) == {0, 1} and merged.group_names == {0: "part", 1: "cube"}
    with zipfile.ZipFile(tmp_path / "a.3mf") as zf:
        assert zf.namelist() == ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]
        assert 'unit="inch"' in zf.read("3D/3dmodel.model").decode()
    assert threemf.threemf_bytes(part) == threemf.threemf_bytes(part.copy())
    # a hand-made package with components, the default millimetre unit and a mirroring item
    model = ('<?xml version="1.0"?><model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             '<resources><object id="1" type="model"><mesh><vertices><vertex x="0" y="0" z="0"/>'
             '<vertex x="10" y="0" z="0"/><vertex x="0" y="10" z="0"/><vertex x="0" y="0" z="10"/></vertices>'
             '<triangles><triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/><triangle v1="0" v2="3" '
             'v3="2"/><triangle v1="1" v2="2" v3="3"/></triangles></mesh></object>'
             '<object id="2" type="model" name="pair"><components><component objectid="1"/>'
             '<component objectid="1" transform="1 0 0 0 1 0 0 0 1 20 0 0"/></components></object></resources>'
             '<build><item objectid="2" transform="-1 0 0 0 1 0 0 0 1 0 0 0"/></build></model>')
    with zipfile.ZipFile(tmp_path / "c.3mf", "w") as zf:
        zf.writestr("3D/3dmodel.model", model)
    m = threemf.read_3mf(tmp_path / "c.3mf")
    assert m.n_faces == 8 and math.isclose(m.volume(), 2 * 1000.0 / 6.0 * 1e-9, rel_tol=1e-12)
    assert math.isclose(m.bounds()[0][0], -0.03)


def test_glb_nodes_attributes_and_determinism(tmp_path, part):
    nodes = [gio.GlbNode("machine", None, np.eye(4), [1, 2], extras={"kind": "root"}),
             gio.GlbNode("part", part, tf.translation([0.5, 0, 0]), color_rgba=(0.6, 0.6, 0.65, 1.0)),
             gio.GlbNode("box", box_mesh([0, 0, 0], [1, 1, 1]), tf.pose(tf.rot_x(0.3), [0, 1, 0]))]
    data = gio.glb_bytes(nodes, extras={"run": "t"})
    assert data == gio.glb_bytes(nodes, extras={"run": "t"})
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == gltf.GLB_MAGIC and version == 2 and length == len(data) and len(data) % 4 == 0
    scene = gio.read_glb(data)
    assert [n["name"] for n in scene.nodes] == ["machine", "part", "box"] and scene.extras == {"run": "t"}
    p = scene.find("part")
    assert np.allclose(p["world"], tf.translation([0.5, 0, 0]))
    assert np.allclose(p["mesh"].vertices, part.vertices, atol=1e-7)                 # float32 positions
    assert np.allclose(p["mesh"].vertex_attr["deviation_m"], part.vertex_attr["deviation_m"], rtol=1e-6)
    assert "normal" in p["mesh"].vertex_attr
    doc = scene.json
    assert doc["meshes"][0]["primitives"][0]["attributes"]["_deviation_m"] >= 0
    assert doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [0.6, 0.6, 0.65, 1.0]
    world = scene.mesh()
    assert math.isclose(world.volume(), part.volume() + 1.0, rel_tol=1e-6)
    gio.write_glb(tmp_path / "s.glb", nodes)
    assert (tmp_path / "s.glb").read_bytes() == gio.glb_bytes(nodes)


def test_glb_quantised_positions_and_groups(part):
    data = gio.glb_bytes([gio.GlbNode("part", part, None)], quantize=True, split_groups=True)
    scene = gio.read_glb(data)
    doc = scene.json
    assert doc["extensionsRequired"] == ["KHR_mesh_quantization"]
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert acc["componentType"] == gltf.USHORT and doc["bufferViews"][acc["bufferView"]]["byteStride"] == 8
    world = scene.mesh()
    extent = float(np.max(part.vertices.max(axis=0) - part.vertices.min(axis=0)))
    assert np.max(np.abs(world.vertices - part.vertices)) <= 0.5 * extent / gltf.QUANT_LEVELS * (1 + 1e-6)
    assert np.array_equal(world.face_group, np.sort(part.face_group, kind="stable")) or \
        sorted(world.face_group.tolist()) == sorted(part.face_group.tolist())
    assert [n["name"] for n in scene.nodes] == ["part", "part#q"]


def test_read_write_dispatch(tmp_path, part):
    for suffix in (".stl", ".obj", ".ply", ".3mf", ".glb"):
        gio.write_mesh(tmp_path / f"m{suffix}", part)
        back = gio.read_mesh(tmp_path / f"m{suffix}")
        assert back.n_faces == part.n_faces
        assert math.isclose(back.volume(), part.volume(), rel_tol=1e-6)
    with pytest.raises(ValueError):
        gio.read_mesh(tmp_path / "m.xyz")
    with pytest.raises(InputError):
        gio.read_glb(b"not a glb file at all")
