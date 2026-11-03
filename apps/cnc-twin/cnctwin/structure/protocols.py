"""What the structure domain needs from removal, as a small Protocol (wired by the integration to
``cnctwin.removal.views.FieldGeometry``), and complete analytic implementations for stocks that are not cut
(and for the tests).

Stock frame conventions (removal's): a block of size (sx, sy, sz) spans [0, sx] x [0, sy] x [0, sz] in S; a
cylinder [diameter, length] or tube [od, id, length] has its axis on S z through x = y = 0, z in [0, length].
All geometry is in the material configuration of S (architecture 4.1).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.errors import InputError


@runtime_checkable
class WorkGeometry(Protocol):
    def stock_bounds_S(self, workpiece: int) -> tuple[np.ndarray, np.ndarray]:
        """(lo [3], hi [3]) of the stock box in S (m)."""

    def fill_fraction(self, workpiece: int, origin_S_m: np.ndarray, cell_m: float,
                      shape: tuple[int, int, int]) -> np.ndarray:
        """Material fraction [nx, ny, nz] in [0, 1] of the cells of a cubic grid."""


def _axis_overlap(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None) / (edges[1:] - edges[:-1])


class BoxGeometry:
    """A block stock [lo, hi] from which axis-aligned boxes are removed: exact material fractions (the union of
    the removed boxes by a partition at every box face), exact wall thickness along axis directions."""

    def __init__(self, lo_S_m, hi_S_m, workpiece: int = 0):
        self.lo = np.asarray(lo_S_m, dtype=float)
        self.hi = np.asarray(hi_S_m, dtype=float)
        self.workpiece = workpiece
        self.removed: list[tuple[np.ndarray, np.ndarray]] = []
        self._version = 0

    def stock_bounds_S(self, workpiece: int):
        return self.lo.copy(), self.hi.copy()

    def version(self, workpiece: int) -> int:
        return self._version

    def remove_box(self, lo_S_m, hi_S_m) -> None:
        lo = np.maximum(np.asarray(lo_S_m, dtype=float), self.lo)
        hi = np.minimum(np.asarray(hi_S_m, dtype=float), self.hi)
        if np.any(hi <= lo):
            return
        self.removed.append((lo, hi))
        self._version += 1

    def _box_fraction(self, edges, lo, hi):
        o = [_axis_overlap(edges[a], lo[a], hi[a]) for a in range(3)]
        return o[0][:, None, None] * o[1][None, :, None] * o[2][None, None, :]

    def fill_fraction(self, workpiece, origin_S_m, cell_m, shape):
        origin = np.asarray(origin_S_m, dtype=float)
        edges = [origin[a] + cell_m * np.arange(int(shape[a]) + 1) for a in range(3)]
        full = self._box_fraction(edges, self.lo, self.hi)
        if not self.removed:
            return full
        cuts = []
        for a in range(3):
            pts = [self.lo[a], self.hi[a]] + [b[0][a] for b in self.removed] + [b[1][a] for b in self.removed]
            cuts.append(np.unique(np.asarray(pts)))
        mids = [0.5 * (c[1:] + c[:-1]) for c in cuts]
        covered = np.zeros((len(mids[0]), len(mids[1]), len(mids[2])), dtype=bool)
        for lo, hi in self.removed:
            m = [(mids[a] > lo[a]) & (mids[a] < hi[a]) for a in range(3)]
            covered |= m[0][:, None, None] & m[1][None, :, None] & m[2][None, None, :]
        holes = np.zeros(tuple(int(s) for s in shape))
        for a, b, c in zip(*np.nonzero(covered)):
            holes += self._box_fraction(edges, np.array([cuts[0][a], cuts[1][b], cuts[2][c]]),
                                        np.array([cuts[0][a + 1], cuts[1][b + 1], cuts[2][c + 1]]))
        return np.clip(full - holes, 0.0, 1.0)

    def inside(self, points) -> np.ndarray:
        P = np.atleast_2d(np.asarray(points, dtype=float))
        ok = np.all((P >= self.lo) & (P <= self.hi), axis=1)
        for lo, hi in self.removed:
            ok &= ~np.all((P > lo) & (P < hi), axis=1)
        return ok

    def wall_thickness(self, workpiece, point_S_m, normal, max_dist: float) -> float:
        """Material thickness behind p along -n: first entry into material to the next exit (removal's
        semantics), by bisection on a 1e-3 relative sampling of the ray."""
        p = np.asarray(point_S_m, dtype=float)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        s = np.linspace(0.0, max_dist, 4001)
        ins = self.inside(p[None, :] - s[:, None] * n[None, :])
        idx = np.nonzero(ins)[0]
        if len(idx) == 0:
            return 0.0
        a = idx[0]
        out = np.nonzero(~ins[a:])[0]
        if len(out) == 0:
            return float(max_dist)
        b = a + out[0]

        def refine(i0, i1, want):
            lo_s, hi_s = s[i0], s[i1]
            for _ in range(60):
                mid = 0.5 * (lo_s + hi_s)
                if self.inside(p - mid * n)[0] == want:
                    hi_s = mid
                else:
                    lo_s = mid
            return hi_s

        t_in = refine(max(a - 1, 0), a, True) if a > 0 else 0.0
        t_out = refine(b - 1, b, False)
        return float(t_out - t_in)


class CylinderGeometry:
    """A cylinder or tube stock on S z (axis x = y = 0), turned to a radius profile r(z) (callable of z, at
    most the stock radius): fractions by s^3 midpoint samples per cell (deterministic, error <= 1 / (2 s)
    of a cell layer at the surface)."""

    def __init__(self, radius_m: float, length_m: float, r_in_m: float = 0.0, samples: int = 4, workpiece: int = 0):
        self.radius = float(radius_m)
        self.length = float(length_m)
        self.r_in = float(r_in_m)
        self.samples = int(samples)
        self.profile = None
        self._version = 0

    def stock_bounds_S(self, workpiece: int):
        r = self.radius
        return np.array([-r, -r, 0.0]), np.array([r, r, self.length])

    def version(self, workpiece: int) -> int:
        return self._version

    def turn(self, profile) -> None:
        """r(z) of the turned outer surface (applied as min(r(z), stock radius))."""
        self.profile = profile
        self._version += 1

    def inside(self, points) -> np.ndarray:
        P = np.atleast_2d(np.asarray(points, dtype=float))
        r = np.hypot(P[:, 0], P[:, 1])
        r_out = np.full(len(P), self.radius)
        if self.profile is not None:
            r_out = np.minimum(r_out, np.asarray(self.profile(P[:, 2]), dtype=float))
        return (r <= r_out) & (r >= self.r_in) & (P[:, 2] >= 0.0) & (P[:, 2] <= self.length)

    def fill_fraction(self, workpiece, origin_S_m, cell_m, shape):
        s = self.samples
        off = (np.arange(s) + 0.5) / s
        origin = np.asarray(origin_S_m, dtype=float)
        shape = tuple(int(v) for v in shape)
        out = np.zeros(shape)
        gx = origin[0] + cell_m * (np.arange(shape[0])[:, None] + off[None, :]).ravel()
        gy = origin[1] + cell_m * (np.arange(shape[1])[:, None] + off[None, :]).ravel()
        for k in range(shape[2]):
            zs = origin[2] + cell_m * (k + off)
            X, Y, Z = np.meshgrid(gx, gy, zs, indexing="ij")
            ins = self.inside(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)).reshape(len(gx), len(gy), s)
            out[:, :, k] = ins.reshape(shape[0], s, shape[1], s, s).sum(axis=(1, 3, 4)) / float(s ** 3)
        return out


def _plain(v):
    return getattr(v, "value", v)


def stock_geometry(stock, workpiece: int = 0):
    """The analytic geometry of the project's stock (block, cylinder, tube), not cut."""
    if stock is None:
        raise InputError("structure: the project has no stock")
    get = stock.get if hasattr(stock, "get") else (lambda k, d=None: getattr(stock, k, d))
    kind = str(_plain(get("kind")) or "")
    size = _plain(get("size_m"))
    if size is None:
        raw = _plain(get("size_mm"))
        size = [float(_plain(v)) * 1e-3 for v in raw] if raw is not None else None
    if size is None:
        raise InputError(f"structure: a {kind or 'stock'} needs its size")
    size = [float(_plain(v)) for v in size]
    if kind == "block":
        return BoxGeometry([0.0, 0.0, 0.0], size, workpiece)
    if kind == "cylinder":
        return CylinderGeometry(0.5 * size[0], size[-1], 0.0, workpiece=workpiece)
    if kind == "tube":
        return CylinderGeometry(0.5 * size[0], size[-1], 0.5 * size[1], workpiece=workpiece)
    raise InputError(f"structure: stock kind {kind!r} needs removal's geometry query")


__all__ = ["WorkGeometry", "BoxGeometry", "CylinderGeometry", "stock_geometry"]
