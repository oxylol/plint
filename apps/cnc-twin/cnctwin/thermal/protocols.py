"""What the thermal domain needs from other parts, as small Protocols (wired by the integration).

* :class:`WorkGeometry` - material of a workpiece on the thermal grid (removal's fill query);
* :class:`ToolEdges` - the EdgeModel of a tool slot (tooling);
* :class:`FixtureContact` - which grid faces touch the workholding (holding / project setup).

All geometry is in the stock lattice frame S, material configuration (docs/architecture.md 4.1).
:class:`BoxStock` and :class:`SampledStock` are complete implementations for an analytic stock
(used by the tests and for parts that are only heated, never cut).
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from ..core.abi import EdgeModel


@runtime_checkable
class WorkGeometry(Protocol):
    def stock_bounds_S(self, workpiece: int) -> tuple[np.ndarray, np.ndarray]:
        """(lo [3], hi [3]) of the stock's bounding box in S (m)."""

    def fill_fraction(self, workpiece: int, origin_S_m: np.ndarray, cell_m: float,
                      shape: tuple[int, int, int]) -> np.ndarray:
        """Material volume fraction [nx, ny, nz] in [0, 1] of the cells of the given grid."""

    def changed_box_S(self, workpiece: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Axis-aligned box in S containing every material change since the previous call (None if
        nothing changed). The thermal domain calls it once per finite-volume step."""


@runtime_checkable
class ToolEdges(Protocol):
    def edge_model(self, tool_slot: int) -> EdgeModel:
        """The edge elements of a tool slot (shapes match the per-element arrays on the bus)."""


@runtime_checkable
class FixtureContact(Protocol):
    def contact_faces(self, workpiece: int, origin_S_m: np.ndarray, cell_m: float,
                      fill: np.ndarray) -> np.ndarray:
        """uint8 bitmask [nx, ny, nz] of cell faces (bit b: -x, +x, -y, +y, -z, +z) that touch the
        workholding or the table."""


class BoxStock:
    """A block stock [lo, hi] in S from which axis-aligned boxes can be removed (exact fractions)."""

    def __init__(self, lo_S_m, hi_S_m, workpiece: int = 0):
        self.workpiece = workpiece
        self.lo = np.asarray(lo_S_m, dtype=float)
        self.hi = np.asarray(hi_S_m, dtype=float)
        self.removed: list[tuple[np.ndarray, np.ndarray]] = []
        self._changed: tuple[np.ndarray, np.ndarray] | None = None

    def stock_bounds_S(self, workpiece: int):
        return self.lo.copy(), self.hi.copy()

    def remove_box(self, lo_S_m, hi_S_m) -> None:
        lo = np.maximum(np.asarray(lo_S_m, dtype=float), self.lo)
        hi = np.minimum(np.asarray(hi_S_m, dtype=float), self.hi)
        if np.any(hi <= lo):
            return
        self.removed.append((lo, hi))
        if self._changed is None:
            self._changed = (lo.copy(), hi.copy())
        else:
            self._changed = (np.minimum(self._changed[0], lo), np.maximum(self._changed[1], hi))

    def changed_box_S(self, workpiece: int):
        out = self._changed
        self._changed = None
        return out

    def fill_fraction(self, workpiece, origin_S_m, cell_m, shape):
        origin = np.asarray(origin_S_m, dtype=float)
        edges = [origin[a] + cell_m * np.arange(shape[a] + 1) for a in range(3)]
        full = self._overlap(edges, self.lo, self.hi)
        holes = np.zeros(tuple(shape))
        # removed boxes may overlap one another: subtract their union by inclusion on a fine grid
        if self.removed:
            holes = self._union_overlap(edges, shape)
        return np.clip(full - holes, 0.0, 1.0)

    @staticmethod
    def _axis_overlap(edges, lo, hi):
        return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None) / (edges[1:] - edges[:-1])

    def _overlap(self, edges, lo, hi):
        ox = self._axis_overlap(edges[0], lo[0], hi[0])
        oy = self._axis_overlap(edges[1], lo[1], hi[1])
        oz = self._axis_overlap(edges[2], lo[2], hi[2])
        return ox[:, None, None] * oy[None, :, None] * oz[None, None, :]

    def _union_overlap(self, edges, shape):
        # exact union of boxes: partition every axis at all box faces, mark covered sub-boxes
        cuts = []
        for a in range(3):
            points = [self.lo[a], self.hi[a]] + [b[0][a] for b in self.removed] + [b[1][a] for b in self.removed]
            cuts.append(np.unique(np.asarray(points)))
        covered = np.zeros((cuts[0].size - 1, cuts[1].size - 1, cuts[2].size - 1), dtype=bool)
        mids = [0.5 * (c[1:] + c[:-1]) for c in cuts]
        for lo, hi in self.removed:
            mx = (mids[0] > lo[0]) & (mids[0] < hi[0])
            my = (mids[1] > lo[1]) & (mids[1] < hi[1])
            mz = (mids[2] > lo[2]) & (mids[2] < hi[2])
            covered |= mx[:, None, None] & my[None, :, None] & mz[None, None, :]
        holes = np.zeros(tuple(shape))
        for a, b, c in zip(*np.nonzero(covered)):
            holes += self._overlap(edges, np.array([cuts[0][a], cuts[1][b], cuts[2][c]]),
                                   np.array([cuts[0][a + 1], cuts[1][b + 1], cuts[2][c + 1]]))
        return holes


class SampledStock:
    """A stock given by an inside test, fractions by s^3 midpoint samples per cell (deterministic)."""

    def __init__(self, inside: Callable[[np.ndarray], np.ndarray], lo_S_m, hi_S_m, samples: int = 4):
        self.inside = inside
        self.lo = np.asarray(lo_S_m, dtype=float)
        self.hi = np.asarray(hi_S_m, dtype=float)
        self.samples = int(samples)
        self._changed = None

    def stock_bounds_S(self, workpiece: int):
        return self.lo.copy(), self.hi.copy()

    def mark_changed(self, lo_S_m, hi_S_m) -> None:
        self._changed = (np.asarray(lo_S_m, dtype=float), np.asarray(hi_S_m, dtype=float))

    def changed_box_S(self, workpiece: int):
        out = self._changed
        self._changed = None
        return out

    def fill_fraction(self, workpiece, origin_S_m, cell_m, shape):
        s = self.samples
        offsets = (np.arange(s) + 0.5) / s
        origin = np.asarray(origin_S_m, dtype=float)
        out = np.zeros(tuple(shape))
        for i in range(shape[0]):
            xs = origin[0] + cell_m * (i + offsets)
            for j in range(shape[1]):
                ys = origin[1] + cell_m * (j + offsets)
                zs = origin[2] + cell_m * (np.arange(shape[2])[:, None] + offsets[None, :])
                X, Y, Z = np.meshgrid(xs, ys, zs.ravel(), indexing="ij")
                inside = self.inside(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1))
                counts = inside.reshape(s, s, shape[2], s).sum(axis=(0, 1, 3))
                out[i, j, :] = counts / float(s ** 3)
        return out
