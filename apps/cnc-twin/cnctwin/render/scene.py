"""What a picture is made of: triangle meshes with materials and optional per-vertex colours, sprites
(sparks, smoke, steam), text labels, colour bars and section planes, all in one world frame (the stock
frame S for run pictures). Plain data; :func:`cnctwin.render.raster.render` draws it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GOURAUD = "gouraud"
FLAT = "flat"


@dataclass
class Material:
    srgb: tuple = (0.75, 0.75, 0.76)    # base colour (sRGB in [0, 1]) when the mesh has no per-vertex colours
    opacity: float = 1.0                # < 1: drawn in the transparent pass (sorted, blended, not in shadow maps)
    shading: str = GOURAUD              # gouraud (crease-aware vertex normals) | flat (face normals)
    cull_back: bool = True              # closed solid: back faces are never seen
    clip: bool = True                   # section planes cut it
    cap: bool = False                   # back faces seen through a section plane drawn in cap_srgb (solids)
    cap_srgb: tuple | None = None       # None: render.cap_srgb
    cast_shadow: bool = True
    receive_shadow: bool = True
    two_sided: bool = False             # open surfaces: lit on the side facing the eye
    specular: float | None = None       # None: render.specular_intensity

    def __post_init__(self) -> None:
        if self.shading not in (GOURAUD, FLAT):
            raise ValueError(f"shading {self.shading!r} is not gouraud or flat")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise ValueError("opacity must lie in [0, 1]")


@dataclass
class RenderMesh:
    """A triangle mesh already placed in the world frame (metres)."""

    name: str
    vertices: np.ndarray                 # [n, 3]
    faces: np.ndarray                    # [m, 3], counter-clockwise seen from outside
    material: Material = field(default_factory=Material)
    vertex_srgb: np.ndarray | None = None      # [n, 3] layer colours (sRGB)
    face_srgb: np.ndarray | None = None        # [m, 3] per-face colours (sRGB), e.g. instanced chips
    vertex_emission: np.ndarray | None = None  # [n, 3] linear emitted light (incandescence), not shadowed
    face_emission: np.ndarray | None = None    # [m, 3] the same per face
    vertex_alpha: np.ndarray | None = None     # [n] opacity per vertex (transparent pass), e.g. fading jets

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3))
        self.faces = np.ascontiguousarray(np.asarray(self.faces, dtype=np.int64).reshape(-1, 3))
        n, m = len(self.vertices), len(self.faces)
        if m and (self.faces.min() < 0 or self.faces.max() >= n):
            raise ValueError(f"mesh {self.name!r}: a face refers to a missing vertex")
        for name, rows, width in (("vertex_srgb", n, 3), ("face_srgb", m, 3), ("vertex_emission", n, 3),
                                  ("face_emission", m, 3)):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=np.float64).reshape(-1, width)
                if len(value) != rows:
                    raise ValueError(f"mesh {self.name!r}: {name} has {len(value)} rows for {rows}")
                setattr(self, name, value)
        if self.vertex_alpha is not None:
            self.vertex_alpha = np.asarray(self.vertex_alpha, dtype=np.float64).reshape(-1)
            if len(self.vertex_alpha) != n:
                raise ValueError(f"mesh {self.name!r}: vertex_alpha has {len(self.vertex_alpha)} rows for {n}")

    @property
    def transparent(self) -> bool:
        return self.material.opacity < 1.0 or self.vertex_alpha is not None

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self.faces) == 0:
            return None
        used = self.vertices[np.unique(self.faces)]
        return used.min(axis=0), used.max(axis=0)


SPRITE_ALPHA = 0
SPRITE_ADD = 1


@dataclass
class Sprites:
    """Round sprites or streaks (capsules from p0 to p1) of world radius, linear colour and opacity."""

    p0: np.ndarray                      # [k, 3] world
    p1: np.ndarray                      # [k, 3] world (= p0 for round sprites)
    radius_m: np.ndarray                # [k]
    rgb: np.ndarray                     # [k, 3] linear
    alpha: np.ndarray                   # [k] opacity (alpha mode) or intensity (additive)
    mode: np.ndarray                    # [k] SPRITE_ALPHA | SPRITE_ADD
    min_radius_px: float = 0.75
    name: str = ""

    def __post_init__(self) -> None:
        self.p0 = np.asarray(self.p0, dtype=np.float64).reshape(-1, 3)
        k = len(self.p0)
        self.p1 = np.asarray(self.p1, dtype=np.float64).reshape(-1, 3)
        self.radius_m = np.broadcast_to(np.asarray(self.radius_m, dtype=np.float64), (k,)).copy()
        self.rgb = np.broadcast_to(np.asarray(self.rgb, dtype=np.float64), (k, 3)).copy()
        self.alpha = np.broadcast_to(np.asarray(self.alpha, dtype=np.float64), (k,)).copy()
        self.mode = np.broadcast_to(np.asarray(self.mode, dtype=np.int32), (k,)).copy()
        if len(self.p1) != k:
            raise ValueError("sprites: p0 and p1 differ in length")

    def __len__(self) -> int:
        return len(self.p0)


@dataclass
class Label:
    text: str
    anchor_m: np.ndarray | None = None  # world point (projected), or None for a fixed screen position
    xy_px: tuple | None = None          # output-pixel position when no anchor (or the offset from the anchor)
    srgb: tuple = (0.08, 0.08, 0.08)
    size: float | None = None           # text height in output pixels (None: render.label_font_size)
    align: str = "left"                 # left | centre | right
    scale_with_height: bool = False     # xy_px (and size) are given for a 720-row picture and scale with it


@dataclass
class ColorBar:
    title: str                          # "Temperature"
    unit: str                           # display unit text ("degC", "um", "mm", "%")
    vmin: float                         # display units
    vmax: float
    cmap: str
    ticks: list | None = None


@dataclass
class Scene:
    meshes: list = field(default_factory=list)
    sprites: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    colorbars: list = field(default_factory=list)
    planes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))   # a x + b y + c z + d >= 0 is kept
    shadows: bool = True
    title: str = ""
    subtitle: str = ""

    def add(self, mesh: RenderMesh | None) -> None:
        if mesh is not None and len(mesh.faces):
            self.meshes.append(mesh)

    def bounds(self, opaque_only: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
        boxes = [m.bounds() for m in self.meshes if not (opaque_only and m.transparent)]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)


def section_plane(point, normal) -> np.ndarray:
    """Plane row keeping the half-space the normal points into: n . x - n . p >= 0."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / float(np.linalg.norm(n))
    return np.array([n[0], n[1], n[2], -float(n @ np.asarray(point, dtype=np.float64))])
