"""The rasteriser kernels: coverage against the analytic triangle, the fill rule, the z-buffer order,
perspective-correct interpolation, near-plane clipping, section caps, shadows, blending and sprites, and every
kernel against its plain-Python reference (architecture 5.5)."""

import math

import numpy as np
import pytest

from cnctwin.render import kernels as K
from cnctwin.render import reference as R
from cnctwin.render.camera import Camera, look_at, orthographic, perspective
from cnctwin.render.raster import corner_normals, render
from cnctwin.render.scene import Material, RenderMesh, Scene, Sprites

RNG = np.random.default_rng(7)


def ndc_clip(points_ndc, w=None):
    """Clip coordinates of NDC points (w = 1 unless given: then x, y, z are multiplied by w)."""
    p = np.asarray(points_ndc, dtype=np.float64).reshape(-1, 3)
    w = np.ones(len(p)) if w is None else np.asarray(w, dtype=np.float64)
    return np.ascontiguousarray(np.concatenate([p * w[:, None], w[:, None]], axis=1))


def run_opaque(clip, faces, width, height, attrs=None, flags=None, planes=None, cap=None):
    n = len(clip)
    attrs = np.zeros((n, K.N_ATTR)) if attrs is None else attrs
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int32).reshape(-1, 3))
    flags = np.zeros(len(faces), np.int32) if flags is None else np.asarray(flags, np.int32)
    cap_index = np.zeros(len(faces), np.int32)
    cap_rgb = np.ascontiguousarray(np.asarray([[0.9, 0.1, 0.1]] if cap is None else cap, dtype=np.float64))
    planes = np.zeros((0, 4)) if planes is None else np.ascontiguousarray(planes, dtype=np.float64)
    depth = np.full((height, width), np.inf)
    color = np.zeros((height, width, 3), np.float32)
    fid = np.full((height, width), -1, np.int32)
    K.raster_opaque(clip, np.ascontiguousarray(attrs), faces, flags, cap_index, cap_rgb, planes, np.ones((1, 1)),
                    np.eye(4), np.zeros(K.N_SHADOW), depth, color, fid)
    return depth, color, fid


def centres_inside(tri_px, width, height):
    """Brute force: pixel centres strictly inside the screen triangle or on a top-left edge."""
    (x0, y0), (x1, y1), (x2, y2) = tri_px
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if area < 0:
        (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    ok = np.ones(xs.shape, bool)
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y2)), ((x2, y2), (x0, y0)), ((x0, y0), (x1, y1))):
        dx, dy = bx - ax, by - ay
        e = dx * (ys - ay) - dy * (xs - ax)
        top_left = dy < 0 or (dy == 0 and dx > 0)
        ok &= (e > 0) | ((e == 0) & top_left)
    return ok


def to_ndc(px, width, height):
    px = np.asarray(px, dtype=np.float64)
    return np.stack([px[:, 0] / width * 2.0 - 1.0, 1.0 - px[:, 1] / height * 2.0, np.zeros(len(px))], axis=1)


def test_coverage_of_random_triangles_is_the_pixel_centres_inside_and_close_to_the_area():
    width, height = 64, 48
    for _ in range(40):
        tri = RNG.uniform([-5, -5], [width + 5, height + 5], size=(3, 2))
        _, _, fid = run_opaque(ndc_clip(to_ndc(tri, width, height)), [[0, 1, 2]], width, height)
        covered = fid >= 0
        assert np.array_equal(covered, centres_inside(tri, width, height))
    # a large triangle inside the picture: the count of covered centres is the area within the perimeter bound
    tri = np.array([[3.3, 2.1], [60.7, 9.4], [17.2, 45.6]])
    _, _, fid = run_opaque(ndc_clip(to_ndc(tri, width, height)), [[0, 1, 2]], width, height)
    area = 0.5 * abs((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1]))
    perimeter = sum(np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3))
    assert abs(int((fid >= 0).sum()) - area) <= perimeter / 2.0


def test_shared_edges_cover_every_pixel_exactly_once():
    width, height = 64, 32          # powers of two: the half-pixel corners survive the trip through NDC exactly
    quad = np.array([[4.5, 3.5], [35.5, 3.5], [35.5, 26.5], [4.5, 26.5]])   # centres on the edges: the tie rule
    first = run_opaque(ndc_clip(to_ndc(quad[[0, 1, 2]], width, height)), [[0, 1, 2]], width, height)[2] >= 0
    second = run_opaque(ndc_clip(to_ndc(quad[[0, 2, 3]], width, height)), [[0, 1, 2]], width, height)[2] >= 0
    assert not np.any(first & second)
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    expected = (xs >= 4.5) & (xs < 35.5) & (ys >= 3.5) & (ys < 26.5)        # top-left edges in, bottom-right out
    assert np.array_equal(first | second, expected)
    # a fan of triangles around a centre also tiles without gaps or overlaps
    centre = np.array([20.3, 15.1])
    angles = np.linspace(0.0, 2.0 * math.pi, 9)[:-1]
    ring = centre + 12.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    count = np.zeros((height, width), int)
    for i in range(8):
        tri = np.array([centre, ring[i], ring[(i + 1) % 8]])
        count += run_opaque(ndc_clip(to_ndc(tri, width, height)), [[0, 1, 2]], width, height)[2] >= 0
    assert count.max() == 1
    polygon_inside = np.zeros((height, width), bool)
    for i in range(8):
        polygon_inside |= centres_inside(np.array([centre, ring[i], ring[(i + 1) % 8]]), width, height)
    assert np.array_equal(count == 1, polygon_inside)


def test_z_buffer_keeps_the_nearest_surface_whatever_the_order_and_ties_keep_the_first():
    width, height = 32, 32
    tri = np.array([[-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0]])
    near, far = tri.copy(), tri.copy()
    near[:, 2], far[:, 2] = -0.2, 0.5
    for order in ((near, far), (far, near)):
        clip = ndc_clip(np.concatenate(order))
        depth, _, fid = run_opaque(clip, [[0, 1, 2], [3, 4, 5]], width, height)
        nearest_face = 0 if order[0] is near else 1
        assert np.all(fid[fid >= 0] == nearest_face)
        assert np.allclose(depth[fid >= 0], -0.2)
    clip = ndc_clip(np.concatenate([tri, tri]))
    _, _, fid = run_opaque(clip, [[0, 1, 2], [3, 4, 5]], width, height)
    assert np.all(fid[fid >= 0] == 0)


def test_back_faces_are_culled_when_asked():
    width, height = 16, 16
    ccw = ndc_clip([[-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0]])
    cw = ccw[[0, 2, 1]]
    assert (run_opaque(ccw, [[0, 1, 2]], width, height, flags=[K.FLAG_CULL_BACK])[2] >= 0).any()
    assert not (run_opaque(cw, [[0, 1, 2]], width, height, flags=[K.FLAG_CULL_BACK])[2] >= 0).any()
    assert (run_opaque(cw, [[0, 1, 2]], width, height, flags=[0])[2] >= 0).any()


def test_perspective_correct_interpolation_matches_the_ray_plane_intersection():
    width, height = 96, 64
    cam = Camera(np.array([0.3, -2.0, 1.2]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), fov_y_rad=0.9,
                 near_m=0.1, far_m=10.0)
    matrix = cam.matrix(width / height)
    quad = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    clip = np.ascontiguousarray(np.concatenate([quad, np.ones((4, 1))], axis=1) @ matrix.T)
    attrs = np.zeros((4, K.N_ATTR))
    attrs[:, K.A_AMB] = quad[:, 0]                       # colour channel 0 carries world x
    attrs[:, K.A_WORLD:K.A_WORLD + 3] = quad
    _, color, fid = run_opaque(clip, [[0, 1, 2], [0, 2, 3]], width, height, attrs=attrs)
    inv = np.linalg.inv(matrix)
    errors = []
    for y, x in zip(*np.nonzero(fid >= 0)):
        ndc = np.array([(x + 0.5) / width * 2 - 1, 1 - (y + 0.5) / height * 2])
        a = inv @ np.array([ndc[0], ndc[1], -1.0, 1.0])
        b = inv @ np.array([ndc[0], ndc[1], 1.0, 1.0])
        a, b = a[:3] / a[3], b[:3] / b[3]
        hit = a + (b - a) * (a[2] / (a[2] - b[2]))
        errors.append(abs(float(color[y, x, 0]) - hit[0]))
    assert len(errors) > 500 and max(errors) < 1e-5        # float32 colour buffer


def test_triangles_through_the_near_plane_are_clipped_not_lost():
    width, height = 64, 64
    cam = Camera(np.array([0.0, -0.05, 0.3]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), fov_y_rad=1.2,
                 near_m=0.1, far_m=20.0)
    floor = np.array([[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 6.0, 0.0], [-2.0, 6.0, 0.0]])  # starts behind the eye
    clip = np.ascontiguousarray(np.concatenate([floor, np.ones((4, 1))], axis=1) @ cam.matrix(1.0).T)
    assert np.any(clip[:, 3] < 0.0)
    depth, _, fid = run_opaque(clip, [[0, 1, 2], [0, 2, 3]], width, height)
    covered = fid >= 0
    assert covered[height - 1].all() and not covered[0].any()            # floor fills the bottom up to the horizon
    assert np.all((depth[covered] >= -1.0) & (depth[covered] <= 1.0))
    ref_depth, _, ref_fid = R.raster_opaque_ref(clip, np.zeros((4, K.N_ATTR)), np.array([[0, 1, 2], [0, 2, 3]]),
                                                np.zeros(2, np.int32), np.zeros(2, np.int32), np.zeros((1, 3)),
                                                np.zeros((0, 4)), np.ones((1, 1)), np.eye(4), np.zeros(K.N_SHADOW),
                                                width, height)
    assert np.array_equal(ref_fid, fid) and np.allclose(ref_depth, depth, rtol=0, atol=1e-12)


def random_scene(n_tri, rng):
    clip = np.ascontiguousarray(np.concatenate([rng.uniform(-1.2, 1.2, (3 * n_tri, 2)), rng.uniform(-0.9, 0.9, (3 * n_tri, 1)),
                                                rng.uniform(0.6, 1.6, (3 * n_tri, 1))], axis=1))
    clip[:, :3] *= clip[:, 3:4]
    clip[::7, 2] = -1.5 * clip[::7, 3]                  # some vertices in front of the near plane
    attrs = np.ascontiguousarray(rng.uniform(0.0, 1.0, (3 * n_tri, K.N_ATTR)))
    faces = np.arange(3 * n_tri, dtype=np.int32).reshape(-1, 3)
    flags = np.ascontiguousarray(rng.choice([0, K.FLAG_CULL_BACK, K.FLAG_CLIP | K.FLAG_CAP, K.FLAG_RECEIVE_SHADOW],
                                            n_tri).astype(np.int32))
    return clip, attrs, faces, flags


def test_opaque_depth_and_blend_kernels_match_their_references():
    width, height = 40, 30
    clip, attrs, faces, flags = random_scene(25, np.random.default_rng(3))
    planes = np.array([[0.3, -0.2, 1.0, 0.1]])
    shadow = np.ascontiguousarray(np.random.default_rng(4).uniform(-1.0, 1.0, (16, 16)))
    light = np.ascontiguousarray(orthographic(2.0, 1.0, 0.1, 3.0) @ look_at([1.0, 1.0, 2.0], [0, 0, 0], [0, 0, 1]))
    par = np.array([1.0, 0.01, 1.0, 0.0])
    cap_index = np.zeros(len(faces), np.int32)
    cap_rgb = np.array([[0.8, 0.2, 0.1]])
    depth = np.full((height, width), np.inf)
    color = np.zeros((height, width, 3), np.float32)
    fid = np.full((height, width), -1, np.int32)
    K.raster_opaque(clip, attrs, faces, flags, cap_index, cap_rgb, planes, shadow, light, par, depth, color, fid)
    ref = R.raster_opaque_ref(clip, attrs, faces, flags, cap_index, cap_rgb, planes, shadow, light, par, width, height)
    assert np.array_equal(ref[2], fid)
    assert np.allclose(ref[0], depth, rtol=0, atol=1e-12)
    assert np.allclose(ref[1], color, rtol=0, atol=1e-6)
    sdepth = np.full((height, width), np.inf)
    cast = np.full(len(faces), K.FLAG_CAST_SHADOW, np.int32)
    K.raster_depth(clip, faces, cast, sdepth)
    assert np.allclose(R.raster_depth_ref(clip, faces, cast, width, height), sdepth, rtol=0, atol=1e-12)
    rgbaw = np.ascontiguousarray(np.concatenate([attrs[:, :4], attrs[:, 6:9]], axis=1))
    order = np.arange(len(faces) - 1, -1, -1).astype(np.int32)
    blended = color.copy()
    behind = np.full_like(depth, 0.95)
    K.raster_blend(clip, rgbaw, faces, order, flags & (K.FLAG_CULL_BACK | K.FLAG_CLIP), planes, behind, blended)
    expected = R.raster_blend_ref(clip, rgbaw, faces, order, flags & (K.FLAG_CULL_BACK | K.FLAG_CLIP), planes, behind,
                                  color)
    assert np.allclose(expected, blended, rtol=0, atol=1e-6)


def test_sprite_kernel_matches_its_reference():
    rng = np.random.default_rng(9)
    width, height = 48, 32
    k = 30
    s0 = np.ascontiguousarray(np.concatenate([rng.uniform(0, width, (k, 1)), rng.uniform(0, height, (k, 1)),
                                              rng.uniform(-0.5, 0.9, (k, 1))], axis=1))
    s1 = np.ascontiguousarray(s0 + np.concatenate([rng.normal(0, 6, (k, 2)), np.zeros((k, 1))], axis=1))
    radius = np.ascontiguousarray(rng.uniform(0.5, 5.0, k))
    rgba = np.ascontiguousarray(rng.uniform(0, 1, (k, 4)))
    mode = np.ascontiguousarray(rng.integers(0, 2, k).astype(np.int32))
    order = np.arange(k, dtype=np.int32)
    depth = np.ascontiguousarray(rng.uniform(-0.2, 1.0, (height, width)))
    color = np.ascontiguousarray(rng.uniform(0, 1, (height, width, 3)).astype(np.float32))
    out = color.copy()
    K.raster_sprites(s0, s1, radius, rgba, mode, order, depth, out)
    assert np.allclose(R.raster_sprites_ref(s0, s1, radius, rgba, mode, order, depth, color), out, atol=1e-5)


def test_corner_normals_are_crease_aware():
    from cnctwin.geom.primitives import box_mesh, cylinder_mesh

    box = box_mesh([0, 0, 0], [1, 2, 3])
    normals = corner_normals(box.vertices, box.faces, math.radians(35))
    face_n = np.cross(box.triangles()[:, 1] - box.triangles()[:, 0], box.triangles()[:, 2] - box.triangles()[:, 0])
    face_n /= np.linalg.norm(face_n, axis=1, keepdims=True)
    assert np.allclose(normals, face_n[:, None, :])                  # a box keeps sharp edges
    cyl = cylinder_mesh([0, 0, 0], [0, 0, 1], 0.5, chordal_tol_m=1e-3)
    smooth = corner_normals(cyl.vertices, cyl.faces, math.radians(35))
    assert np.allclose(smooth, R.corner_normals_ref(cyl.vertices, cyl.faces.astype(np.int64), math.radians(35)),
                       atol=1e-12)
    side = np.abs(smooth[..., 2]) < 1e-9                              # side corners: radial normals
    radial = cyl.vertices[cyl.faces][side][:, :2]
    radial = radial / np.linalg.norm(radial, axis=-1, keepdims=True)
    assert side.sum() > 20 and np.allclose(smooth[side][:, :2], radial, atol=1e-6)


def test_section_plane_caps_a_cut_solid_and_shadows_darken_the_floor():
    from cnctwin.geom.primitives import box_mesh

    box = box_mesh([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])
    floor = RenderMesh("floor", [[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]], [[0, 1, 2], [0, 2, 3]],
                       Material(srgb=(0.8, 0.8, 0.8), cull_back=False, clip=False))
    cube = RenderMesh("cube", box.vertices, box.faces, Material(srgb=(0.2, 0.4, 0.8), cap=True, cap_srgb=(1.0, 0.0, 0.0)))
    cam = Camera(np.array([0.0, -4.0, 3.0]), np.array([0.0, 0.0, 0.4]), fov_y_rad=0.7, near_m=0.5, far_m=20.0)
    scene = Scene([floor, cube], shadows=True)
    lit = render(scene, cam, 120, 90, ssaa=1)
    unlit = render(Scene([floor, cube], shadows=False), cam, 120, 90, ssaa=1)
    on_floor = lit.mesh_of_face[np.maximum(lit.face_id, 0)] == 0
    on_floor &= lit.face_id >= 0
    darker = (lit.linear.sum(axis=2) < unlit.linear.sum(axis=2) - 1e-4) & on_floor
    assert darker.sum() > 50                                          # the cube casts a shadow on the floor
    cut = Scene([floor, cube], planes=np.array([[0.0, 1.0, 0.0, 0.0]]), shadows=False)   # keep y >= 0
    result = render(cut, cam, 120, 90, ssaa=1)
    red = (result.image[..., 0] > 200) & (result.image[..., 1] < 60) & (result.image[..., 2] < 60)
    assert red.sum() > 100                                            # the inside of the cube shows the cap colour


def test_transparent_surfaces_blend_over_the_opaque_ones():
    quad = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    back = RenderMesh("back", quad, [[0, 1, 2], [0, 2, 3]], Material(srgb=(0.0, 0.0, 0.0), cull_back=False,
                                                                    receive_shadow=False))
    glass = RenderMesh("glass", np.asarray(quad, float) * 0.5 + [0, 0, 0.5], [[0, 1, 2], [0, 2, 3]],
                       Material(srgb=(1.0, 1.0, 1.0), opacity=0.25, cull_back=False, two_sided=True, specular=0.0))
    cam = Camera(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                 projection="orthographic", view_height_m=3.0, near_m=1.0, far_m=10.0)
    params = None
    alone = render(Scene([back], shadows=False), cam, 60, 60, params=params)
    both = render(Scene([back, glass], shadows=False), cam, 60, 60, params=params)
    centre = both.linear[30, 30]
    expected = 0.25 * (np.full(3, both.stats["blend_fragments"] > 0) * 0 + _lit_white(cam)) + 0.75 * alone.linear[30, 30]
    assert np.allclose(centre, expected, atol=2e-3)


def _lit_white(cam):
    from cnctwin.render.raster import Lighting

    light = Lighting.for_camera(cam)
    n = np.array([0.0, 0.0, 1.0])
    return np.full(3, light.ambient + light.fill * max(float(n @ light.fill_dir), 0.0)
                   + light.key * max(float(n @ light.key_dir), 0.0))


def test_sprites_add_glow_and_hide_behind_surfaces():
    cam = Camera(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                 projection="orthographic", view_height_m=2.0, near_m=1.0, far_m=10.0)
    wall = RenderMesh("wall", [[-1, -1, 0], [0, -1, 0], [0, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]],
                      Material(srgb=(0.1, 0.1, 0.1), cull_back=False))
    glow = Sprites([[-0.5, 0.0, -0.5], [0.5, 0.0, 0.5]], [[-0.5, 0.0, -0.5], [0.5, 0.0, 0.5]], 0.2, [1.0, 0.5, 0.1],
                   1.0, 1)
    base = render(Scene([wall], shadows=False), cam, 80, 80)
    with_glow = render(Scene([wall], [glow], shadows=False), cam, 80, 80)
    behind = with_glow.linear[40, 20] - base.linear[40, 20]           # sprite behind the wall: hidden
    front = with_glow.linear[40, 60] - base.linear[40, 60]            # sprite in the open: added
    assert np.allclose(behind, 0.0) and front[0] > 0.5


def test_every_kernel_has_one_compiled_signature():
    for kernel in K.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
