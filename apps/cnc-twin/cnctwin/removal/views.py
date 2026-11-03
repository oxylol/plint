"""The in-process workpiece as other domains ask for it (thermal.protocols.WorkGeometry and
chips.protocols.ChipGeometry), backed by the removal field (docs/models/removal.md "Queries").

:class:`FieldGeometry` answers ``stock_bounds_S``, ``fill_fraction`` (thermal finite volumes, exact for the
column model), ``changed_box_S`` (per caller: the box of every brick changed since that caller's previous
call), ``surface_height`` (chip piles: part surface only, kind SURF_PART / SURF_NONE), ``version`` (the field
version) and ``rotation_radius_S`` (largest material radius about S z). The integration wires one instance
per run (``RemovalDomain.geometry(state)``); it reads the live field of the state, so it always answers for
the latest geometry.
"""

from __future__ import annotations

import numpy as np

from . import abi as A
from .field import brick_box, n_levels

SURF_NONE = 0
SURF_PART = 1


class FieldGeometry:
    def __init__(self, state_getter, caller: str = "default"):
        self._state = state_getter
        self._seen: dict[tuple[str, int], np.ndarray] = {}
        self.caller = caller

    def _wp(self, workpiece: int):
        st = self._state()
        if workpiece < 0 or workpiece >= len(st.workpieces):
            raise IndexError(f"removal has no workpiece {workpiece}")
        return st.workpieces[workpiece]

    def stock_bounds_S(self, workpiece: int):
        f = self._wp(workpiece).field
        return f.lv_box_lo_m[0].copy(), f.lv_box_hi_m[0].copy()

    def fill_fraction(self, workpiece: int, origin_S_m, cell_m: float, shape) -> np.ndarray:
        from .k_query import fill_fraction

        out = np.zeros(tuple(int(v) for v in shape))
        fill_fraction(self._wp(workpiece).field, np.ascontiguousarray(np.asarray(origin_S_m, dtype=float)),
                      float(cell_m), np.asarray(shape, dtype=np.int64), out)
        return np.clip(out, 0.0, 1.0)

    def changed_box_S(self, workpiece: int, caller: str | None = None):
        f = self._wp(workpiece).field
        key = (caller or self.caller, workpiece)
        prev = self._seen.get(key)
        cur = np.array(f.bver, copy=True)
        self._seen[key] = cur
        if prev is None or prev.shape != cur.shape:
            return f.lv_box_lo_m[0].copy(), f.lv_box_hi_m[0].copy()
        changed = np.flatnonzero(cur != prev)
        if len(changed) == 0:
            return None
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for b in changed:
            lvl, coords, n_lo, n_hi = brick_box(f, int(b))
            o = f.lv_origin_m[lvl]
            a = f.lv_cell_m[lvl]
            lo = np.minimum(lo, o + n_lo * a)
            hi = np.maximum(hi, o + n_hi * a)
        return lo, hi

    def surface_height(self, workpiece: int, axis: int, sign: int, a_S_m, b_S_m):
        from .k_query import surface_height

        av = np.ascontiguousarray(np.asarray(a_S_m, dtype=float).reshape(-1))
        bv = np.ascontiguousarray(np.asarray(b_S_m, dtype=float).reshape(-1))
        h = np.zeros(len(av))
        kind = np.zeros(len(av), np.int64)
        surface_height(self._wp(workpiece).field, int(axis), int(sign), av, bv, h, kind)
        kind = np.where(kind > 0, SURF_PART, SURF_NONE).astype(np.int64)
        return h, kind

    def version(self, workpiece: int) -> int:
        return int(self._wp(workpiece).field.info[A.FI_VERSION])

    def rotation_radius_S(self, workpiece: int):
        wp = self._wp(workpiece)
        f = wp.field
        nx, ny = int(f.lv_dims[0, 0]), int(f.lv_dims[0, 1])
        r0 = int(f.lv_ray0[0, 2])
        cnt = f.cnt[r0:r0 + nx * ny].reshape(nx, ny)
        o = f.lv_origin_m[0]
        a = float(f.lv_cell_m[0])
        ii, jj = np.nonzero(cnt > 0)
        if len(ii) == 0:
            return 0.0, 0.0, 0.0
        x = o[0] + ii * a
        y = o[1] + jj * a
        r = float(np.sqrt(x * x + y * y).max() + 0.5 * a * np.sqrt(2.0))
        lo, hi = self.stock_bounds_S(workpiece)
        return r, float(lo[2]), float(hi[2])

    def point_inside(self, workpiece: int, points_S_m) -> np.ndarray:
        from .k_query import points_inside

        P = np.ascontiguousarray(np.asarray(points_S_m, dtype=float).reshape(-1, 3))
        out = np.zeros(len(P), np.int8)
        points_inside(self._wp(workpiece).field, P, out)
        return out.astype(bool)

    def ray_cast(self, workpiece: int, origin_S_m, direction, t_max: float, exit_: bool = False):
        """(hit, t, point, outward normal) of the first entry (or exit) along a ray / segment."""
        from .k_query import ray_cast

        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        o = np.asarray(origin_S_m, dtype=float)
        out = np.zeros(8)
        hit = ray_cast(self._wp(workpiece).field, o[0], o[1], o[2], d[0], d[1], d[2], float(t_max), 1 if exit_ else 0,
                       1e-8, out)
        return bool(hit), float(out[0]), out[1:4].copy(), out[4:7].copy()

    def wall_thickness(self, workpiece: int, point_S_m, normal, max_dist: float) -> float:
        from .k_query import wall_thickness

        p = np.asarray(point_S_m, dtype=float)
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        return float(wall_thickness(self._wp(workpiece).field, p[0], p[1], p[2], n[0], n[1], n[2], float(max_dist)))

    def levels(self, workpiece: int) -> int:
        return n_levels(self._wp(workpiece).field)
