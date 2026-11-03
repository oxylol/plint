"""Drawing a :class:`~cnctwin.render.scene.Scene` with a :class:`~cnctwin.render.camera.Camera` into an sRGB
picture (docs/models/render.md "Rasteriser").

Passes, all at ``ssaa`` times the output size and box-filtered down at the end:

1. **Vertex shading** (NumPy): per triangle corner, a base colour (material, per-vertex layer colour or
   per-face colour, sRGB -> linear), a normal (flat: the face normal; Gouraud: crease-aware angle-weighted
   corner normals) and two lit colours: ambient + fill light (+ emission), and the key light (Lambert +
   Blinn-Phong) that shadows may darken. Lights are fixed to the camera (key upper left, fill right).
2. **Shadow map** (Williams 1978): depth of every shadow caster seen from the key light (orthographic,
   fitted to the scene bounds), percentage-closer filtered (Reeves et al. 1987) per fragment.
3. **Opaque pass**: z-buffer, near-plane clipping, top-left fill rule, perspective-correct interpolation,
   section planes with capped cuts (kernels.raster_opaque).
4. **Transparent pass**: enclosure, doors, coolant streams: triangles sorted back to front by view depth of
   their centroid (stable, ties by index), depth-tested against the opaque surfaces, over-blended
   (Porter & Duff 1984).
5. **Sprites**: smoke/steam over-blended back to front, sparks and glow added.
6. Box filter, sRGB encoding, then labels and colour bars at the output resolution (Pillow).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import colour as C
from . import kernels as K
from . import params as P
from . import text as T
from .camera import Camera, look_at, orthographic
from .scene import FLAT, Label, Material, RenderMesh, Scene, Sprites


@dataclass
class RenderResult:
    image: np.ndarray                    # [H, W, 3] uint8 sRGB at the output size
    depth: np.ndarray                    # [H * ssaa, W * ssaa] z/w of the nearest opaque surface (inf: none)
    face_id: np.ndarray                  # [H * ssaa, W * ssaa] index into the concatenated opaque faces (-1)
    linear: np.ndarray                   # [H, W, 3] float32 linear RGB before encoding
    stats: dict = field(default_factory=dict)
    mesh_of_face: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))   # opaque face -> mesh index


@dataclass
class Lighting:
    key_dir: np.ndarray                  # unit, from the scene toward the key light (world)
    fill_dir: np.ndarray
    key: float
    fill: float
    ambient: float
    specular: float
    exponent: float

    @classmethod
    def for_camera(cls, camera: Camera, params: dict | None = None) -> "Lighting":
        p = P.view() if params is None else params
        f = camera.forward()
        up = np.asarray(camera.up, dtype=np.float64)
        s = np.cross(f, up)
        s /= float(np.linalg.norm(s))
        u = np.cross(s, f)

        def light(az, el):
            v = math.cos(el) * (math.cos(az) * (-f) - math.sin(az) * s) + math.sin(el) * u
            return v / float(np.linalg.norm(v))

        return cls(light(float(p["key_light_azimuth_rad"]), float(p["key_light_elevation_rad"])),
                   light(float(p["fill_light_azimuth_rad"]), float(p["fill_light_elevation_rad"])),
                   float(p["key_light_intensity"]), float(p["fill_light_intensity"]),
                   float(p["ambient_intensity"]), float(p["specular_intensity"]), float(p["specular_exponent"]))


# ----------------------------------------------------------------------------- normals
def face_normals(corners: np.ndarray) -> np.ndarray:
    c = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    n = np.linalg.norm(c, axis=1, keepdims=True)
    return np.divide(c, n, out=np.zeros_like(c), where=n > 0.0)


def _corner_angles(corners: np.ndarray) -> np.ndarray:
    out = np.zeros(corners.shape[:2])
    for k in range(3):
        a = corners[:, (k + 1) % 3] - corners[:, k]
        b = corners[:, (k + 2) % 3] - corners[:, k]
        out[:, k] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))
    return out


def corner_normals(vertices: np.ndarray, faces: np.ndarray, crease_rad: float) -> np.ndarray:
    """Normals [m, 3, 3] at the triangle corners: angle-weighted over the faces around each vertex whose normal
    is within ``crease_rad`` of the face's own (kernels.corner_normals)."""
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    corners = vertices[faces]
    fn = np.ascontiguousarray(face_normals(corners))
    ang = np.ascontiguousarray(_corner_angles(corners))
    flat = faces.reshape(-1).astype(np.int64)
    order = np.argsort(flat, kind="stable").astype(np.int32)
    counts = np.bincount(flat, minlength=len(vertices))
    offsets = np.zeros(len(vertices) + 1, np.int32)
    np.cumsum(counts, out=offsets[1:])
    out = np.zeros((len(faces), 3, 3))
    K.corner_normals(fn, faces, ang, offsets, np.ascontiguousarray(order // 3), np.ascontiguousarray(order % 3),
                     float(math.cos(crease_rad)), out)
    return out


# ----------------------------------------------------------------------------- shading
def _base_linear(mesh: RenderMesh) -> np.ndarray:
    """Base colour (linear) at the corners [m, 3, 3]."""
    m = len(mesh.faces)
    if mesh.vertex_srgb is not None:
        return C.srgb_to_linear(mesh.vertex_srgb)[mesh.faces]
    if mesh.face_srgb is not None:
        return np.repeat(C.srgb_to_linear(mesh.face_srgb)[:, None, :], 3, axis=1)
    return np.broadcast_to(C.srgb_to_linear(np.asarray(mesh.material.srgb, dtype=np.float64)), (m, 3, 3)).copy()


def _emission(mesh: RenderMesh) -> np.ndarray | None:
    if mesh.vertex_emission is not None:
        return mesh.vertex_emission[mesh.faces]
    if mesh.face_emission is not None:
        return np.repeat(mesh.face_emission[:, None, :], 3, axis=1)
    return None


def shade(mesh: RenderMesh, camera: Camera, light: Lighting, crease_rad: float, normal_offset_m: float = 0.0):
    """Corner world positions [m, 3, 3], ambient [m, 3, 3] and key-light [m, 3, 3] linear colours, and the shadow
    lookup positions (corners moved ``normal_offset_m`` along their normal toward the lit side)."""
    corners = mesh.vertices[mesh.faces]
    if mesh.material.shading == FLAT:
        normals = np.repeat(face_normals(corners)[:, None, :], 3, axis=1)
    else:
        normals = corner_normals(mesh.vertices, mesh.faces, crease_rad)
    if camera.projection == "orthographic":
        view = np.broadcast_to(-camera.forward(), corners.shape)
    else:
        view = camera.eye - corners
        view = view / np.maximum(np.linalg.norm(view, axis=2, keepdims=True), 1e-300)
    if mesh.material.two_sided:
        side = np.sign(np.einsum("mki,mki->mk", normals, view))
        normals = normals * np.where(side == 0.0, 1.0, side)[..., None]
    base = _base_linear(mesh)
    ndl = np.maximum(normals @ light.key_dir, 0.0)
    ndf = np.maximum(normals @ light.fill_dir, 0.0)
    half = light.key_dir + view
    half = half / np.maximum(np.linalg.norm(half, axis=2, keepdims=True), 1e-300)
    ndh = np.maximum(np.einsum("mki,mki->mk", normals, half), 0.0)
    ks = light.specular if mesh.material.specular is None else float(mesh.material.specular)
    spec = ks * np.where(ndl > 0.0, ndh ** light.exponent, 0.0)
    amb = base * (light.ambient + light.fill * ndf)[..., None]
    emit = _emission(mesh)
    if emit is not None:
        amb = amb + emit
    direct = base * (light.key * ndl)[..., None] + spec[..., None]
    lookup = corners + normals * normal_offset_m
    return corners, amb, direct, lookup


def _flags(material: Material, planes_active: bool) -> int:
    flags = 0
    if material.cull_back and not material.two_sided:
        flags |= K.FLAG_CULL_BACK
    if material.clip and planes_active:
        flags |= K.FLAG_CLIP
        if material.cap:
            flags |= K.FLAG_CAP
    if material.receive_shadow:
        flags |= K.FLAG_RECEIVE_SHADOW
    if material.cast_shadow:
        flags |= K.FLAG_CAST_SHADOW
    return flags


def _to_clip(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return np.ascontiguousarray(homo @ matrix.T)


# ----------------------------------------------------------------------------- passes
@dataclass
class _Opaque:
    world: np.ndarray        # [3m, 3]
    attrs: np.ndarray        # [3m, 9]
    faces: np.ndarray        # [m, 3] int32
    flags: np.ndarray        # [m] int32
    cap_index: np.ndarray    # [m] int32
    cap_rgb: np.ndarray      # [k, 3]
    mesh_of_face: np.ndarray


def _gather_opaque(scene: Scene, camera: Camera, light: Lighting, crease: float, default_cap,
                   normal_offset_m: float = 0.0) -> _Opaque:
    worlds, attrs, flags, caps, owners = [], [], [], [], []
    cap_table = [C.srgb_to_linear(np.asarray(default_cap, dtype=np.float64))]
    planes_active = len(scene.planes) > 0
    for index, mesh in enumerate(scene.meshes):
        if mesh.transparent or len(mesh.faces) == 0:
            continue
        corners, amb, direct, lookup = shade(mesh, camera, light, crease, normal_offset_m)
        m = len(mesh.faces)
        worlds.append(corners.reshape(-1, 3))
        attrs.append(np.concatenate([amb.reshape(-1, 3), direct.reshape(-1, 3), corners.reshape(-1, 3),
                                     lookup.reshape(-1, 3)], axis=1))
        flags.append(np.full(m, _flags(mesh.material, planes_active), np.int32))
        cap = 0
        if mesh.material.cap_srgb is not None:
            cap_table.append(C.srgb_to_linear(np.asarray(mesh.material.cap_srgb, dtype=np.float64)))
            cap = len(cap_table) - 1
        caps.append(np.full(m, cap, np.int32))
        owners.append(np.full(m, index, np.int32))
    if not worlds:
        return _Opaque(np.zeros((0, 3)), np.zeros((0, K.N_ATTR)), np.zeros((0, 3), np.int32), np.zeros(0, np.int32),
                       np.zeros(0, np.int32), np.asarray(cap_table), np.zeros(0, np.int32))
    world = np.concatenate(worlds)
    n_faces = len(world) // 3
    return _Opaque(world, np.ascontiguousarray(np.concatenate(attrs)),
                   np.ascontiguousarray(np.arange(3 * n_faces, dtype=np.int32).reshape(-1, 3)),
                   np.concatenate(flags), np.concatenate(caps), np.ascontiguousarray(np.asarray(cap_table)),
                   np.concatenate(owners))


SHADOW_SPHERE_MARGIN = 1.01      # the light's view volume encloses the focus sphere with 1 % to spare
SHADOW_NORMAL_OFFSET_TEXELS = 1.5   # shadow lookups move 1.5 texels along the normal (no self-shadowing acne)


def shadow_map(world: np.ndarray, faces: np.ndarray, flags: np.ndarray, key_dir: np.ndarray, size: int,
               bounds, reach_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Depth map of the casters seen from the key light: orthographic, ``size`` texels across the bounding sphere
    of ``bounds`` (the focus), its depth range reaching ``reach_m`` toward the light so tall casters outside the
    focus (spindle, column) still shade it; returns (map [size, size], world -> light clip)."""
    lo, hi = bounds
    centre = 0.5 * (lo + hi)
    radius = max(0.5 * float(np.linalg.norm(hi - lo)), 1e-6) * SHADOW_SPHERE_MARGIN
    reach = max(float(reach_m if reach_m is not None else radius), radius)
    up = np.array([0.0, 0.0, 1.0]) if abs(float(key_dir[2])) < 0.95 else np.array([1.0, 0.0, 0.0])
    eye_distance = reach + radius
    view = look_at(centre + eye_distance * key_dir, centre, up)
    matrix = orthographic(2.0 * radius, 1.0, 0.5 * radius, eye_distance + radius) @ view
    depth = np.full((size, size), np.inf)
    if len(faces):
        K.raster_depth(_to_clip(matrix, world), faces, flags, depth)
    return depth, matrix


def _background(height: int, width: int, params: dict) -> np.ndarray:
    top = C.srgb_to_linear(np.asarray(params["background_top_srgb"], dtype=np.float64))
    bottom = C.srgb_to_linear(np.asarray(params["background_bottom_srgb"], dtype=np.float64))
    t = ((np.arange(height) + 0.5) / height)[:, None]
    column = (1.0 - t) * top + t * bottom
    return np.ascontiguousarray(np.broadcast_to(column[:, None, :], (height, width, 3)).astype(np.float32))


def _transparent(scene: Scene, camera: Camera, light: Lighting, crease: float, matrix: np.ndarray,
                 depth: np.ndarray, color: np.ndarray) -> int:
    rows, faces, flags, keys = [], [], [], []
    planes_active = len(scene.planes) > 0
    offset = 0
    view = camera.view()
    for mesh in scene.meshes:
        if not mesh.transparent or len(mesh.faces) == 0:
            continue
        corners, amb, direct, _ = shade(mesh, camera, light, crease)
        rgb = (amb + direct).reshape(-1, 3)
        if mesh.vertex_alpha is not None:
            alpha = (mesh.vertex_alpha[mesh.faces] * mesh.material.opacity).reshape(-1, 1)
        else:
            alpha = np.full((len(rgb), 1), mesh.material.opacity)
        rows.append(np.concatenate([rgb, alpha, corners.reshape(-1, 3)], axis=1))
        m = len(mesh.faces)
        faces.append(offset + np.arange(3 * m, dtype=np.int32).reshape(-1, 3))
        flag = K.FLAG_CULL_BACK if (mesh.material.cull_back and not mesh.material.two_sided) else 0
        if mesh.material.clip and planes_active:
            flag |= K.FLAG_CLIP
        flags.append(np.full(m, flag, np.int32))
        centroid = corners.mean(axis=1)
        keys.append(centroid @ view[2, :3] + view[2, 3])          # view-space z (more negative = farther)
        offset += 3 * m
    if not rows:
        return 0
    rgbaw = np.ascontiguousarray(np.concatenate(rows))
    all_faces = np.ascontiguousarray(np.concatenate(faces))
    order = np.ascontiguousarray(np.argsort(np.concatenate(keys), kind="stable").astype(np.int32))
    planes = np.ascontiguousarray(np.asarray(scene.planes, dtype=np.float64).reshape(-1, 4))
    return int(K.raster_blend(_to_clip(matrix, rgbaw[:, K.B_WORLD:K.B_WORLD + 3]), rgbaw, all_faces, order,
                              np.concatenate(flags), planes, depth, color))


def project_points(matrix: np.ndarray, points: np.ndarray, width: int, height: int):
    """Screen x, y (pixels), z/w and w of world points (w <= 0: behind the eye)."""
    clip = _to_clip(matrix, np.asarray(points, dtype=np.float64).reshape(-1, 3))
    w = clip[:, 3]
    safe = np.where(np.abs(w) > 1e-300, w, 1e-300)
    sx = (clip[:, 0] / safe + 1.0) * 0.5 * width
    sy = (1.0 - clip[:, 1] / safe) * 0.5 * height
    return np.stack([sx, sy, clip[:, 2] / safe], axis=1), w


def _sprites(scene: Scene, camera: Camera, matrix: np.ndarray, aspect: float, depth: np.ndarray,
             color: np.ndarray, ssaa: int) -> int:
    groups = [s for s in scene.sprites if isinstance(s, Sprites) and len(s)]
    if not groups:
        return 0
    height, width = depth.shape
    proj_yy = camera.proj(aspect)[1, 1]
    s0s, s1s, radii, rgbas, modes, zs = [], [], [], [], [], []
    for group in groups:
        a, wa = project_points(matrix, group.p0, width, height)
        b, wb = project_points(matrix, group.p1, width, height)
        visible = (wa > 0.0) & (wb > 0.0)
        if not np.any(visible):
            continue
        r_px = group.radius_m * proj_yy * height / (2.0 * np.maximum(wa, 1e-300))
        r_px = np.maximum(r_px, group.min_radius_px * ssaa)
        s0s.append(a[visible])
        s1s.append(b[visible])
        radii.append(r_px[visible])
        rgbas.append(np.concatenate([group.rgb[visible], group.alpha[visible, None]], axis=1))
        modes.append(group.mode[visible])
        zs.append(a[visible, 2])
    if not s0s:
        return 0
    s0 = np.ascontiguousarray(np.concatenate(s0s))
    s1 = np.ascontiguousarray(np.concatenate(s1s))
    radius = np.ascontiguousarray(np.concatenate(radii))
    rgba = np.ascontiguousarray(np.concatenate(rgbas))
    mode = np.ascontiguousarray(np.concatenate(modes).astype(np.int32))
    z = np.concatenate(zs)
    index = np.arange(len(z))
    alpha_first = np.lexsort((index, -z, mode))      # over-blended far to near, then the additive ones
    return int(K.raster_sprites(s0, s1, radius, rgba, mode, np.ascontiguousarray(alpha_first.astype(np.int32)),
                                depth, color))


def downsample(color: np.ndarray, factor: int) -> np.ndarray:
    """Box filter over factor x factor samples (fixed summation order: deterministic)."""
    if factor == 1:
        return color
    h, w, _ = color.shape
    blocks = color.reshape(h // factor, factor, w // factor, factor, 3)
    return (blocks.sum(axis=3).sum(axis=1) * np.float32(1.0 / (factor * factor))).astype(np.float32)


# ----------------------------------------------------------------------------- the frame
def render(scene: Scene, camera: Camera, width: int, height: int, ssaa: int = 1, shadows: bool | None = None,
           params: dict | None = None, shadow_focus=None) -> RenderResult:
    """Draw ``scene`` (see the module docstring); ``shadows`` None follows ``scene.shadows``; the shadow map
    covers ``shadow_focus`` (lo, hi), default the opaque scene bounds."""
    p = P.view() if params is None else params
    width, height, ssaa = int(width), int(height), max(int(ssaa), 1)
    W, H = width * ssaa, height * ssaa
    aspect = width / height
    light = Lighting.for_camera(camera, p)
    crease = float(p["crease_angle_rad"])
    use_shadows = scene.shadows if shadows is None else bool(shadows)
    bounds = scene.bounds(opaque_only=True) if shadow_focus is None else (np.asarray(shadow_focus[0], float),
                                                                          np.asarray(shadow_focus[1], float))
    size = int(round(float(p["shadow_map_size"])))
    offset = 0.0
    if use_shadows and bounds is not None:
        texel = 2.0 * max(0.5 * float(np.linalg.norm(bounds[1] - bounds[0])), 1e-6) * SHADOW_SPHERE_MARGIN / size
        offset = SHADOW_NORMAL_OFFSET_TEXELS * texel
    opaque = _gather_opaque(scene, camera, light, crease, p["cap_srgb"], offset)
    matrix = camera.matrix(aspect)
    depth = np.full((H, W), np.inf)
    color = _background(H, W, p)
    face_id = np.full((H, W), -1, np.int32)
    shadow_par = np.zeros(K.N_SHADOW)
    smap = np.ones((1, 1))
    lmat = np.eye(4)
    stats = {"opaque_faces": int(len(opaque.faces))}
    if use_shadows and len(opaque.faces) and bounds is not None:
        full = scene.bounds(opaque_only=True)
        reach = float(np.linalg.norm(full[1] - full[0])) if full is not None else None
        smap, lmat = shadow_map(opaque.world, opaque.faces, opaque.flags, light.key_dir, size, bounds, reach)
        shadow_par[K.SH_ON] = 1.0
        shadow_par[K.SH_BIAS] = float(p["shadow_bias"])
        shadow_par[K.SH_PCF] = float(round(float(p["shadow_pcf_radius"])))
    planes = np.ascontiguousarray(np.asarray(scene.planes, dtype=np.float64).reshape(-1, 4))
    if len(opaque.faces):
        stats["opaque_fragments"] = int(K.raster_opaque(_to_clip(matrix, opaque.world), opaque.attrs, opaque.faces,
                                                        opaque.flags, opaque.cap_index, opaque.cap_rgb, planes,
                                                        np.ascontiguousarray(smap), np.ascontiguousarray(lmat),
                                                        shadow_par, depth, color, face_id))
    stats["blend_fragments"] = _transparent(scene, camera, light, crease, matrix, depth, color)
    stats["sprite_fragments"] = _sprites(scene, camera, matrix, aspect, depth, color, ssaa)
    linear = downsample(color, ssaa)
    image = C.to_uint8(C.linear_to_srgb(linear))
    image = overlay(image, scene, matrix, p)
    return RenderResult(image, depth, face_id, linear, stats, opaque.mesh_of_face)


def overlay(image: np.ndarray, scene: Scene, matrix: np.ndarray, params: dict) -> np.ndarray:
    """Labels, colour bars and the title on the finished picture."""
    if not (scene.labels or scene.colorbars or scene.title or scene.subtitle):
        return image
    height, width = image.shape[:2]
    size = max(8, int(round(float(params["label_font_size"]) * height / 720.0)))
    picture = Image.fromarray(image, "RGB")
    if scene.title:
        T.draw_label(picture, (size, 1.2 * size), Label(scene.title, srgb=(0.05, 0.05, 0.05)), int(1.25 * size))
    if scene.subtitle:
        T.draw_label(picture, (size, 2.6 * size), Label(scene.subtitle, srgb=(0.2, 0.2, 0.2)), size)
    for label in scene.labels:
        scale = height / 720.0 if label.scale_with_height else 1.0
        if label.anchor_m is not None:
            xy, w = project_points(matrix, np.asarray(label.anchor_m)[None, :], width, height)
            if w[0] <= 0.0:
                continue
            x, y = float(xy[0, 0]), float(xy[0, 1])
            if label.xy_px is not None:
                x, y = x + label.xy_px[0], y + label.xy_px[1]
        elif label.xy_px is not None:
            x, y = float(label.xy_px[0]) * scale, float(label.xy_px[1]) * scale
        else:
            continue
        T.draw_label(picture, (x, y), label, int(round(label.size * scale)) if label.size else size)
    bar_h = int(round(float(params["colorbar_height_fraction"]) * height))
    bar_w = max(4, int(round(float(params["colorbar_width_fraction"]) * width)))
    for i, bar in enumerate(scene.colorbars):
        x0 = width - (i + 1) * (bar_w + 6 * size) - size
        y0 = (height - bar_h) // 2 + size
        T.draw_colorbar(picture, bar, x0, y0, bar_w, bar_h, size)
    return np.asarray(picture, dtype=np.uint8)
