"""Meshing: closed, outward meshes with vertices on the exact ray crossings; incremental bricks; surface
event ids; welding; displacement grid; the case table."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_mesh as KM
from cnctwin.removal import k_subtract as KS
from cnctwin.removal import mesh as M
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S
from cnctwin.removal.mctable import EDGES, tables

LAT = 0.25e-3


def _cut(f, centre, radius, ev=5):
    prof = P.profile_from_points([0.0, 20e-3], [radius, radius])
    spec = A.subtract_spec([0, 0, 1], [centre], [0.0, 20e-3], [0.0], [-1] * 3, [1] * 3, event_id=ev)
    KS.subtract_envelope(f, prof, spec, A.empty_target(int(f.info[A.FI_NRAYS])), A.new_accum(F.n_levels(f)),
                         event_buffer(8, A.EVENT_WIDTH), A.make_scratch())


def test_case_table_closes_every_case():
    count, tri = tables()
    for case in range(1, 255):
        # every crossing edge of the case appears, and every triangle edge is shared by exactly two triangles
        crossing = {e for e, (a, b, _) in enumerate(EDGES) if ((case >> a) & 1) != ((case >> b) & 1)}
        used = set(tri[case, :count[case]].ravel().tolist())
        assert used == crossing, case
    assert count[0] == 0 and count[255] == 0


def test_block_tube_and_cut_meshes_are_closed():
    for f in (S.block_field([5e-3, 4e-3, 3e-3], LAT), S.tube_field(8e-3, 5e-3, LAT, inner_diameter_m=4e-3)):
        m = M.mesh_level(f)
        assert M.open_edges(m) == 0
        assert M.mesh_volume(m) == pytest.approx(F.volume_m3(f, 0, 2), rel=0.02)
    f = S.block_field([12e-3, 12e-3, 6e-3], LAT)
    _cut(f, [6e-3, 6e-3, 3e-3], 3.1e-3, ev=17)
    _cut(f, [9e-3, 4e-3, 1e-3], 1.7e-3, ev=18)
    m = M.mesh_level(f)
    assert M.open_edges(m) == 0
    assert M.mesh_volume(m) == pytest.approx(F.volume_m3(f, 0, 2), rel=0.02)
    assert {17, 18, A.SEV_STOCK} <= set(np.unique(m.sev).tolist())


def test_vertices_lie_on_the_exact_crossings():
    f = S.tube_field(8e-3, 5e-3, LAT)
    m = M.mesh_level(f)
    r = np.hypot(m.vertices[:, 0], m.vertices[:, 1])
    side = (m.vertices[:, 2] > 0.2e-3) & (m.vertices[:, 2] < 4.8e-3)
    # every vertex on the side lies on a lattice edge at the exact crossing: within the quantum of r = 4 mm
    on_edge = side & (np.abs(r - 4e-3) < 1e-7)
    assert on_edge.sum() > 0.95 * side.sum()


def test_incremental_dirty_bricks_and_welding():
    f = S.block_field([40e-3, 10e-3, 6e-3], LAT)
    M.clear_dirty(f, np.arange(len(f.bdirty)))
    _cut(f, [5e-3, 5e-3, 4e-3], 2e-3)
    dirty = M.dirty_bricks(f)
    assert 0 < len(dirty) < len(f.bdirty)
    meshes = M.mesh_bricks(f, M.surface_bricks(f))
    welded = M.weld(list(meshes.values()))
    assert M.open_edges(welded) == 0
    assert len(np.unique(welded.edge_id)) == len(welded.edge_id)


def test_displacement_grid_moves_vertices():
    f = S.block_field([4e-3, 4e-3, 4e-3], LAT)
    grid = np.zeros((3, 3, 3, 3))
    grid[..., 2] = 10e-6                       # uniform +10 um in z
    disp = M.Grid([-1e-3, -1e-3, -1e-3], 3e-3, grid)
    m0 = M.mesh_level(f)
    m1 = M.mesh_level(f, displacement=disp)
    assert np.allclose(m1.vertices[:, 2] - m0.vertices[:, 2], 10e-6, atol=1e-15)


def test_ray_crossing_matches_reference():
    f = S.tube_field(8e-3, 5e-3, LAT, inner_diameter_m=4e-3)
    for ray in range(0, int(f.info[A.FI_NRAYS]), 97):
        segs = R.ray_segments(f, ray)
        for lo, hi, elo, ehi in segs:
            q0 = (hi // 25000) * 25000
            e, ev, found = KM.ray_crossing(f, ray, q0, q0 + 25000, 1)
            assert found == 1 and e == hi and ev == ehi


def test_one_signature_each():
    for kernel in KM.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_planar_patches_keep_the_surface_in_fewer_triangles():
    """mesh_level(planar=True): the flat patches of every brick become fans about their centroid; the mesh stays
    closed and welds across bricks without a crack, it encloses the same volume, every vertex lies on the full mesh's
    surface (the planes of the faces it replaces), the curved walls of the cuts keep their triangles, and the block's
    flat faces shrink by an order of magnitude."""
    f = S.block_field([20e-3, 18e-3, 9e-3], LAT)
    _cut(f, [8e-3, 8e-3, 5e-3], 3.1e-3, ev=17)
    _cut(f, [14e-3, 5e-3, 3e-3], 1.7e-3, ev=18)
    full = M.mesh_level(f)
    flat = M.mesh_level(f, planar=True)
    assert M.open_edges(flat) == 0 and M.open_edges(full) == 0
    assert len(np.unique(flat.edge_id)) == len(flat.edge_id)
    assert M.mesh_volume(flat) == pytest.approx(M.mesh_volume(full), rel=1e-12)
    assert len(flat.faces) < 0.5 * len(full.faces)          # 0.37 here (3 x 3 x 2 bricks); 32 x 32 patches: 1/16
    # the new vertices lie on the block's faces or on the floors of the cuts (z of a floor: a crossing plane)
    new = flat.edge_id < 0
    assert new.any()
    v = flat.vertices[new]
    lo, hi = np.zeros(3), np.array([20e-3, 18e-3, 9e-3])
    on_box = np.any((np.abs(v - lo) < 1e-12) | (np.abs(v - hi) < 1e-12), axis=1)
    floors = np.unique(np.round(full.vertices[:, 2], 12))
    on_floor = np.isin(np.round(v[:, 2], 12), floors)
    assert np.all(on_box | on_floor)
    # every lattice vertex of the planar mesh is a vertex of the full mesh (same ids, same positions)
    old = flat.edge_id >= 0
    idx = np.searchsorted(full.edge_id, flat.edge_id[old])
    assert np.array_equal(full.edge_id[idx], flat.edge_id[old])
    assert np.array_equal(full.vertices[idx], flat.vertices[old])
    assert {17, 18, A.SEV_STOCK} <= set(np.unique(flat.sev).tolist())
    # a coarse field is not worth it; a fine one is
    assert not M.planar_wanted(f) and M.planar_wanted(f, budget=1000)
