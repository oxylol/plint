"""What the wear domain needs from other parts, as small Protocols (wired by the integration).

* :class:`ToolInfo` - per tool slot: the nominal EdgeModel (the per-element arrays on the bus have its
  [n_teeth, n_elem] shape), the critical sections (tooling's ``ToolProps.sections``), the radius of the
  finishing edge (corner / nose / ball) and of the tool, and whether it rotates.
  :class:`GeometryToolInfo` wraps cnctwin.tooling ``ToolGeometry`` objects (the integration passes
  ``state.tooling.tools``); :class:`StaticToolInfo` takes the arrays directly (tests, inline data).
* :class:`SkinQuery` - share of an element's cut that lies in the stock's skin layer (scale, cast skin).
  :class:`StockShell` is the analytic default for block, cylinder and tube stock in the stock frame S
  (conventions of cnctwin.removal.stock); removal can answer it from its field's stock-surface ids.
* :class:`ToolTolerances` - the tightest surface displacement the tool's wear may cause, and the finish
  specification, of the surfaces a tool finishes. :class:`ProjectTolerances` reads the project's
  ``cad.tolerances``; :meth:`ProjectTolerances.finishing` places the surface events on the tolerances' plane
  features, and the domain assigns each slot the tightest tolerance and finish of the features it generates (a
  tolerance whose features cannot be placed applies to every tool that writes surface events).

The Taylor-calibration response is the fourth protocol, :class:`cnctwin.wear.calibration.TaylorResponse`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.abi import EdgeModel
from ..core.errors import InputError
from .params import as_map, field_of, num, plain, text


@dataclass(frozen=True)
class SectionInfo:
    """A critical section of the tool (ascending z from the tip)."""

    name: str
    z_m: float
    Z_bending_m3: float
    Z_polar_m3: float
    d_eq_m: float


@runtime_checkable
class ToolInfo(Protocol):
    def edge_model(self, slot: int) -> EdgeModel:
        """Nominal edge elements of a tool slot."""

    def sections(self, slot: int) -> list[SectionInfo]:
        """Critical sections, ascending z (empty: none known)."""

    def finish_radius_m(self, slot: int) -> float:
        """Radius of the edge that generates the finished surface (corner, nose, ball); 0 when sharp."""

    def tool_radius_m(self, slot: int) -> float:
        """Largest cutting radius."""

    def rotating(self, slot: int) -> bool:
        """True for milling, drilling and tapping tools, False for the turning family."""


class StaticToolInfo:
    """ToolInfo from explicit data (tests, inline tools)."""

    def __init__(self, edges: dict, sections: dict | None = None, finish_radius_m: dict | None = None,
                 tool_radius_m: dict | None = None, rotating: dict | None = None):
        self._edges = dict(edges)
        self._sections = dict(sections or {})
        self._finish = dict(finish_radius_m or {})
        self._radius = dict(tool_radius_m or {})
        self._rotating = dict(rotating or {})

    def edge_model(self, slot: int) -> EdgeModel:
        if slot not in self._edges:
            raise InputError(f"wear: no edge model for tool slot {slot}")
        return self._edges[slot]

    def sections(self, slot: int) -> list[SectionInfo]:
        return sorted(self._sections.get(slot, []), key=lambda s: s.z_m)

    def finish_radius_m(self, slot: int) -> float:
        return float(self._finish.get(slot, 0.0))

    def tool_radius_m(self, slot: int) -> float:
        if slot in self._radius:
            return float(self._radius[slot])
        edge = self.edge_model(slot)
        valid = np.asarray(edge.elem_valid) != 0
        return float(np.max(np.asarray(edge.elem_r_m)[valid])) if valid.any() else 0.0

    def rotating(self, slot: int) -> bool:
        return bool(self._rotating.get(slot, True))


class GeometryToolInfo:
    """ToolInfo over cnctwin.tooling ``ToolGeometry`` objects indexed by slot."""

    def __init__(self, geometries):
        self._geos = {int(g.slot): g for g in geometries}

    def _geo(self, slot: int):
        if slot not in self._geos:
            raise InputError(f"wear: tooling has no geometry for tool slot {slot}")
        return self._geos[slot]

    def edge_model(self, slot: int) -> EdgeModel:
        return self._geo(slot).edge

    def sections(self, slot: int) -> list[SectionInfo]:
        out = []
        for s in self._geo(slot).props.sections:
            out.append(SectionInfo(str(s.name), float(s.z_m), float(s.Z_bending_m3), float(s.Z_polar_m3),
                                   float(s.d_eq_m)))
        return sorted(out, key=lambda s: s.z_m)

    def finish_radius_m(self, slot: int) -> float:
        geo = self._geo(slot)
        geometry = geo.spec.geometry if geo.spec is not None else None
        if geo.type == "ball_endmill":
            return 0.5 * float(geo.props.diameter_m)
        for holder in (field_of(geometry, "insert"), field_of(geometry, "inserts")):
            radius = num(holder, "nose_radius_m") if holder is not None else None
            if radius:
                return float(radius)
        radius = num(geometry, "corner_radius_m") if geometry is not None else None
        return float(radius) if radius else 0.0

    def tool_radius_m(self, slot: int) -> float:
        return 0.5 * float(self._geo(slot).props.diameter_m)

    def rotating(self, slot: int) -> bool:
        return bool(self._geo(slot).rotating)


# ------------------------------------------------------------------------------------------ skin
@runtime_checkable
class SkinQuery(Protocol):
    def skin_fraction(self, slot: int, edge: EdgeModel, rotating: bool, pose_S_T: np.ndarray,
                      in_cut_s: np.ndarray) -> np.ndarray:
        """[n_teeth, n_elem] share in [0, 1] of each in-cut element's cut that lies in the skin layer."""


class StockShell:
    """Skin layer of a block ([0, size] in S), cylinder or tube stock (axis S z, z in [0, length]).

    Rotating tools: each station's circle (radius, height in T, at the step-end pose) is sampled at
    ``samples`` angles; the share of the samples inside the stock that lie within ``thickness`` of the
    stock's outer boundary is the element's skin share. Non-rotating tools on a rotating stock: the
    element's radius and height in S decide (the work turns past it)."""

    def __init__(self, kind: str, size_m, thickness_m: float, samples: int = 64):
        self.kind = str(kind)
        self.size = np.asarray(size_m, dtype=float).reshape(-1)
        self.thickness = float(thickness_m)
        self.samples = int(samples)
        if self.kind not in ("block", "cylinder", "tube"):
            raise InputError(f"wear: no skin model for {self.kind} stock")

    def _depth(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(inside, depth below the outer boundary) of points [n, 3] in S."""
        if self.kind == "block":
            lo = p
            hi = self.size[None, :3] - p
            inside = np.all(lo >= 0.0, axis=1) & np.all(hi >= 0.0, axis=1)
            depth = np.minimum(np.min(lo, axis=1), np.min(hi, axis=1))
            return inside, depth
        rho = np.hypot(p[:, 0], p[:, 1])
        outer = 0.5 * self.size[0]
        length = self.size[-1]
        inner = 0.5 * self.size[1] if self.kind == "tube" else 0.0
        inside = (rho <= outer) & (rho >= inner) & (p[:, 2] >= 0.0) & (p[:, 2] <= length)
        depth = np.minimum(np.minimum(outer - rho, p[:, 2]), length - p[:, 2])
        if self.kind == "tube":
            depth = np.minimum(depth, rho - inner)
        return inside, depth

    def skin_fraction(self, slot, edge, rotating, pose_S_T, in_cut_s):
        in_cut = np.asarray(in_cut_s, dtype=float)
        out = np.zeros(in_cut.shape)
        if self.thickness <= 0.0 or not np.any(in_cut > 0.0):
            return out
        pose = np.asarray(pose_S_T, dtype=float)
        rot, trans = pose[:3, :3], pose[:3, 3]
        if rotating:
            radii = np.asarray(edge.station_r_m, dtype=float)
            heights = np.asarray(edge.station_z_m, dtype=float)
            angles = 2.0 * math.pi * (np.arange(self.samples) + 0.5) / self.samples
            share = np.zeros(radii.shape[0])
            for s in range(radii.shape[0]):
                pts = np.stack([radii[s] * np.cos(angles), radii[s] * np.sin(angles),
                                np.full(angles.shape, heights[s])], axis=1) @ rot.T + trans
                inside, depth = self._depth(pts)
                count = int(np.count_nonzero(inside))
                share[s] = float(np.count_nonzero(inside & (depth < self.thickness))) / count if count else 0.0
            station = np.asarray(edge.elem_station, dtype=np.int64)
            out = np.where(in_cut > 0.0, share[np.clip(station, 0, max(radii.shape[0] - 1, 0))], 0.0)
            return out
        psi = np.asarray(edge.tooth_phase_rad, dtype=float)[:, None] + np.asarray(edge.elem_lag_rad, dtype=float)
        r = np.asarray(edge.elem_r_m, dtype=float)
        pts = np.stack([r * np.cos(psi), r * np.sin(psi), np.asarray(edge.elem_z_m, dtype=float)], axis=-1)
        flat = pts.reshape(-1, 3) @ rot.T + trans
        # cylinder and tube: the element sweeps the circle of its radius about S z, so its radius and
        # height decide; block (planing, shaping): the point itself
        inside, depth = self._depth(flat)
        skin = (inside & (depth < self.thickness)).reshape(in_cut.shape)
        return np.where((in_cut > 0.0) & skin, 1.0, 0.0)


# ------------------------------------------------------------------------------------------ tolerances
@runtime_checkable
class ToolTolerances(Protocol):
    def surface_tolerance_m(self, slot: int) -> float | None:
        """Largest displacement of a finished surface the tool's wear may cause (None: no tolerance)."""

    def finish_spec(self, slot: int) -> tuple[float | None, float | None]:
        """(Ra, Rt or Rz) the tool's finished surfaces must meet (None: no specification)."""

    def applies(self, slot: int, finishes: bool) -> bool:
        """Whether the tolerances apply to the tool (default: tools that write surface events)."""


class ProjectTolerances:
    """The project's ``cad.tolerances`` (see the module doc for the rules).

    Surface displacement allowed by each tolerance: distance +/- band / 2; diameter band / 4 (the
    radius moves by half the diameter change, within +/- half the band); radius band / 2; zone
    tolerances (position, flatness, profile, runout ...) zone / 2. Fits without explicit plus/minus
    (ISO 286 grades) are not converted and are listed in :attr:`unused`."""

    def __init__(self, project=None):
        self.allowed_m: float | None = None
        self.Ra_m: float | None = None
        self.Rt_m: float | None = None
        self.sources: list[str] = []
        self.unused: list[str] = []
        self._assigned: dict[int, tuple[float | None, float | None, float | None]] = {}
        # per tolerance: (allowed displacement, Ra, Rt, [(point_S, unit normal_S)] of its plane features, every feature
        # a plane with a point and a normal); see finishing()
        self.features: list[tuple] = []
        cad = field_of(project, "cad") if project is not None else None
        self._pose = _placement(cad)
        tolerances = field_of(cad, "tolerances") if cad is not None else None
        for tol in tolerances or []:
            self._read(tol)

    def assign(self, slot: int, allowed_m: float | None, Ra_m: float | None = None, Rt_m: float | None = None):
        """The integration's per-tool assignment (which features a tool finishes) overrides the default."""
        self._assigned[int(slot)] = (allowed_m, Ra_m, Rt_m)

    def _tighten(self, value: float | None, ident: str) -> None:
        if value is None or value <= 0.0:
            return
        self._last_allowed = value
        if self.allowed_m is None or value < self.allowed_m:
            self.allowed_m = value
        self.sources.append(ident)

    def _read(self, tol) -> None:
        tol = as_map(tol)
        self._read_values(tol)
        kind = text(tol, "kind")
        if kind == "roughness":
            ra, rz, rt = num(tol, "Ra_m"), num(tol, "Rz_m"), num(tol, "Rt_m")
            rt = min(v for v in (rz, rt) if v is not None) if (rz is not None or rt is not None) else None
            allowed = None
        else:
            ra = rt = None
            allowed = self._last_allowed          # set by _tighten for the kinds that give a displacement
        if allowed is None and ra is None and rt is None:
            return
        self.features.append((allowed, ra, rt, self._planes(tol)))

    def _planes(self, tol) -> list | None:
        """The tolerance's features as planes (point and unit normal in S); None when one is not a plane with both."""
        feats = []
        feature = field_of(tol, "feature")
        if feature is not None:
            feats.append(feature)
        feats.extend(field_of(tol, "between") or [])
        if not feats:
            return None
        planes = []
        R, t = self._pose[:3, :3], self._pose[:3, 3]
        for feat in feats:
            if str(plain(field_of(feat, "face")) or "") != "plane":
                return None
            near, normal = _vector(feat, "near_m", "near_mm", 1e-3), _vector(feat, "normal", "normal", 1.0)
            if near is None or normal is None or not np.linalg.norm(normal) > 0.0:
                return None
            n = R @ normal
            planes.append((R @ near + t, n / np.linalg.norm(n)))
        return planes

    def finishing(self, points_S_m, normals_S, band_m: float):
        """Which surface events generate a toleranced feature's final surface, and the tightest (allowed
        displacement, Ra, Rt) of the tolerances they generate: an event belongs to a plane feature when its normal is
        within 30 deg of the feature's and it lies within max(band_m, 2 x the tolerance's allowed displacement) of
        the plane. A tolerance whose features are not all planes with a point and a normal cannot be placed and
        applies to every event, as before. (mask [m] bool, allowed, Ra, Rt), None where nothing applies."""
        pts = np.asarray(points_S_m, dtype=float).reshape(-1, 3)
        nrm = np.asarray(normals_S, dtype=float).reshape(-1, 3)
        mask = np.zeros(len(pts), bool)
        best = [None, None, None]
        cos_tol = math.cos(math.radians(30.0))
        for allowed, ra, rt, planes in self.features:
            if planes is None:
                on = np.ones(len(pts), bool)
            else:
                band = max(float(band_m), 2.0 * allowed if allowed else 0.0)
                on = np.zeros(len(pts), bool)
                for point, normal in planes:
                    on |= (np.abs((pts - point) @ normal) <= band) & (nrm @ normal >= cos_tol)
            if not on.any():
                continue
            mask |= on
            for i, value in enumerate((allowed, ra, rt)):
                if value is not None and value > 0.0:
                    best[i] = value if best[i] is None else min(best[i], value)
        return mask, best[0], best[1], best[2]

    def _read_values(self, tol) -> None:
        self._last_allowed = None
        kind = text(tol, "kind")
        ident = text(tol, "id", kind)
        plus, minus = num(tol, "plus_m"), num(tol, "minus_m")
        if kind == "roughness":
            ra, rz, rt = num(tol, "Ra_m"), num(tol, "Rz_m"), num(tol, "Rt_m")
            if ra is not None:
                self.Ra_m = ra if self.Ra_m is None else min(self.Ra_m, ra)
            for value in (rz, rt):
                if value is not None:
                    self.Rt_m = value if self.Rt_m is None else min(self.Rt_m, value)
            return
        if kind in ("distance", "radius"):
            if plus is not None or minus is not None:
                self._tighten(0.5 * ((plus or 0.0) + (minus or 0.0)), ident)
            else:
                self.unused.append(ident)
            return
        if kind == "diameter":
            dia = field_of(tol, "diameter_m")
            if plus is None and minus is None and dia is not None and not isinstance(plain(dia), (int, float)):
                plus, minus = num(dia, "plus"), num(dia, "minus")
            if plus is not None or minus is not None:
                self._tighten(0.25 * ((plus or 0.0) + (minus or 0.0)), ident)
            else:
                self.unused.append(ident)
            return
        if kind == "angle":
            self.unused.append(ident)
            return
        zone = num(tol, "tolerance_m")
        if zone is not None:
            self._tighten(0.5 * zone, ident)
        else:
            self.unused.append(ident)

    def surface_tolerance_m(self, slot: int) -> float | None:
        if slot in self._assigned:
            return self._assigned[slot][0]
        return self.allowed_m

    def finish_spec(self, slot: int) -> tuple[float | None, float | None]:
        if slot in self._assigned:
            return self._assigned[slot][1], self._assigned[slot][2]
        return self.Ra_m, self.Rt_m

    def applies(self, slot: int, finishes: bool) -> bool:
        return slot in self._assigned or bool(finishes)


def _placement(cad) -> np.ndarray:
    """The CAD placement in the stock frame (identity without one)."""
    placement = field_of(cad, "placement") if cad is not None else None
    if placement is not None and hasattr(placement, "matrix"):
        return np.asarray(placement.matrix(), dtype=float)
    return np.eye(4)


def _vector(obj, si_key, raw_key, scale):
    value = field_of(obj, si_key)
    if value is None:
        value = field_of(obj, raw_key)
        if value is None:
            return None
        return np.asarray([float(plain(x)) for x in plain(value)], dtype=float) * scale
    return np.asarray([float(plain(x)) for x in plain(value)], dtype=float)


__all__ = ["SectionInfo", "ToolInfo", "StaticToolInfo", "GeometryToolInfo", "SkinQuery", "StockShell",
           "ToolTolerances", "ProjectTolerances"]
