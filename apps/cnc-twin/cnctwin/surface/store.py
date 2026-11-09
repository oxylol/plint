"""The per-face patch store (bounded memory: aggregates per face patch, docs/models/surface.md section 7).

A **patch** is one facet of one brick (32^3 lattice cells, removal's level-0 numbering) of one workpiece:
the events of a brick whose outward normals lie within ``face_angle`` of the facet normal and whose plane
offset lies within ``plane_tol`` belong to it. It keeps running sums (count, centroid, normal), the
aggregated roughness of its evaluated lines per direction (feed: along the feed marks; cross: across
the passes), lay, marks, integrity and flags. A **pass** is one uninterrupted generation of a patch by
one tool (a gap longer than ``pass_gap`` opens a new one); it keeps the tool-tip and feed sums for the
step-over (cross) profile and witness marks. **Lines** are the open profile lines (cnctwin.surface.lines).

Everything is NumPy arrays plus JSON metadata (to_arrays / from_arrays round-trip bit-identically).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lines import N_ROW, LineBuffer

# ------------------------------------------------------------------------------------------- patch columns
PF_NAMES = (
    "ox", "oy", "oz",                          # origin: first event (S)
    "nsx", "nsy", "nsz",                       # sum of outward normals
    "xsx", "xsy", "xsz",                       # sum of event positions
    "count",
    "f_len", "f_ra", "f_rq", "f_rz", "f_rt_max", "f_rz_max", "f_ra_max", "f_lines", "f_cutoff", "f_short",
    "f_pt_max", "f_rsm",
    "c_len", "c_ra", "c_rq", "c_rz", "c_rt_max", "c_rz_max", "c_ra_max", "c_lines", "c_cutoff", "c_short",
    "lx", "ly", "lz",                          # lay direction (unit, S) of the roughest direction
    "dx", "dy", "dz",                          # feed direction (S) of the last line
    "marks", "mark_max",
    "T_peak", "sigma", "sigma_depth", "hard_change", "hard_depth", "white", "burn", "overtemper", "melt",
    "chatter", "chatter_freq", "chatter_index",
    "recut", "recut_max",
    "bue",
    "vb", "u_n",
    "spec_ra", "spec_rz", "spec_rt",           # roughness specification of the feature the patch belongs to (0: none)
)
PI_NAMES = ("workpiece", "brick", "op", "slot", "block_first", "block_last", "cls", "first_tick", "last_tick",
            "feature", "finding")
PF = {n: i for i, n in enumerate(PF_NAMES)}
PI = {n: i for i, n in enumerate(PI_NAMES)}

QF_NAMES = ("csx", "csy", "csz", "dsx", "dsy", "dsz", "ax", "ay", "az", "w_min", "w_sum", "un_sum", "count",
            "hmin_sum")
QI_NAMES = ("patch", "slot", "cls", "tick0", "tick1", "block_first", "block_last", "op", "closed", "lines")
QF = {n: i for i, n in enumerate(QF_NAMES)}
QI = {n: i for i, n in enumerate(QI_NAMES)}


@dataclass
class Store:
    pf: np.ndarray = field(default_factory=lambda: np.zeros((0, len(PF_NAMES))))
    pi: np.ndarray = field(default_factory=lambda: np.zeros((0, len(PI_NAMES)), np.int64))
    n_patch: int = 0
    qf: np.ndarray = field(default_factory=lambda: np.zeros((0, len(QF_NAMES))))
    qi: np.ndarray = field(default_factory=lambda: np.zeros((0, len(QI_NAMES)), np.int64))
    n_pass: int = 0
    lines: dict = field(default_factory=dict)          # (patch, pass, sub) -> LineBuffer
    facets: dict = field(default_factory=dict)         # (workpiece, brick) -> [patch]
    active: dict = field(default_factory=dict)         # (patch, slot) -> pass

    # ------------------------------------------------------------------ patches
    def new_patch(self, workpiece: int, brick: int, X: np.ndarray, normal: np.ndarray, op: int) -> int:
        if self.n_patch == self.pf.shape[0]:
            size = max(16, 2 * self.pf.shape[0])
            pf = np.zeros((size, len(PF_NAMES)))
            pi = np.zeros((size, len(PI_NAMES)), np.int64)
            pf[:self.n_patch] = self.pf[:self.n_patch]
            pi[:self.n_patch] = self.pi[:self.n_patch]
            self.pf, self.pi = pf, pi
        i = self.n_patch
        self.n_patch += 1
        self.pf[i] = 0.0
        self.pi[i] = 0
        self.pf[i, PF["ox"]:PF["oz"] + 1] = X
        self.pf[i, PF["nsx"]:PF["nsz"] + 1] = normal
        self.pi[i, PI["workpiece"]] = workpiece
        self.pi[i, PI["brick"]] = brick
        self.pi[i, PI["op"]] = op
        self.pi[i, PI["slot"]] = -1
        self.pi[i, PI["block_first"]] = -1
        self.pi[i, PI["block_last"]] = -1
        self.pi[i, PI["first_tick"]] = -1
        self.pi[i, PI["last_tick"]] = -1
        self.pi[i, PI["feature"]] = -1
        self.facets.setdefault((int(workpiece), int(brick)), []).append(i)
        return i

    def normal(self, i: int) -> np.ndarray:
        n = self.pf[i, PF["nsx"]:PF["nsz"] + 1]
        norm = float(np.linalg.norm(n))
        return n / norm if norm > 0.0 else np.array([0.0, 0.0, 1.0])

    def origin(self, i: int) -> np.ndarray:
        return self.pf[i, PF["ox"]:PF["oz"] + 1].copy()

    def centroid(self, i: int) -> np.ndarray:
        c = self.pf[i, PF["count"]]
        return self.pf[i, PF["xsx"]:PF["xsz"] + 1] / c if c > 0 else self.origin(i)

    def find_patch(self, workpiece: int, brick: int, X: np.ndarray, normal: np.ndarray, cos_tol: float,
                   plane_tol_m: float) -> int:
        """First facet of the brick (creation order) matching the normal and the plane, -1 if none."""
        for i in self.facets.get((int(workpiece), int(brick)), ()):
            n = self.normal(i)
            if float(n @ normal) < cos_tol:
                continue
            if abs(float((X - self.origin(i)) @ n)) > plane_tol_m:
                continue
            return i
        return -1

    def patch_list(self) -> range:
        return range(self.n_patch)

    # ------------------------------------------------------------------ passes
    def new_pass(self, patch: int, slot: int, cls: int, tick: int, block: int, op: int) -> int:
        if self.n_pass == self.qf.shape[0]:
            size = max(16, 2 * self.qf.shape[0])
            qf = np.zeros((size, len(QF_NAMES)))
            qi = np.zeros((size, len(QI_NAMES)), np.int64)
            qf[:self.n_pass] = self.qf[:self.n_pass]
            qi[:self.n_pass] = self.qi[:self.n_pass]
            self.qf, self.qi = qf, qi
        q = self.n_pass
        self.n_pass += 1
        self.qf[q] = 0.0
        self.qf[q, QF["w_min"]] = np.inf
        self.qi[q] = 0
        self.qi[q, QI["patch"]] = patch
        self.qi[q, QI["slot"]] = slot
        self.qi[q, QI["cls"]] = cls
        self.qi[q, QI["tick0"]] = tick
        self.qi[q, QI["tick1"]] = tick
        self.qi[q, QI["block_first"]] = block
        self.qi[q, QI["block_last"]] = block
        self.qi[q, QI["op"]] = op
        self.active[(int(patch), int(slot))] = q
        return q

    def passes_of(self, patch: int) -> np.ndarray:
        return np.nonzero(self.qi[:self.n_pass, QI["patch"]] == patch)[0]

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self, prefix: str = "store.") -> tuple[dict, dict]:
        arrays = {f"{prefix}pf": self.pf[:self.n_patch].copy(), f"{prefix}pi": self.pi[:self.n_patch].copy(),
                  f"{prefix}qf": self.qf[:self.n_pass].copy(), f"{prefix}qi": self.qi[:self.n_pass].copy()}
        meta_lines = []
        for k, (key, line) in enumerate(sorted(self.lines.items())):
            arrays[f"{prefix}line.{k}.rows"] = line.rows[:line.count].copy()
            arrays[f"{prefix}line.{k}.vec"] = np.concatenate([line.origin, line.d, line.n, line.axis])
            meta_lines.append({"key": [int(v) for v in key], "d_len": int(len(line.d)), "n_len": int(len(line.n)),
                               **line.frame_dict()})
        meta = {"lines": meta_lines,
                "active": [[int(p), int(s), int(q)] for (p, s), q in sorted(self.active.items())]}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str = "store.") -> "Store":
        st = cls()
        st.pf = np.ascontiguousarray(arrays[f"{prefix}pf"], dtype=float).copy()
        st.pi = np.ascontiguousarray(arrays[f"{prefix}pi"], dtype=np.int64).copy()
        st.n_patch = st.pf.shape[0]
        qf = np.ascontiguousarray(arrays[f"{prefix}qf"], dtype=float)
        if qf.ndim == 2 and qf.shape[1] < len(QF_NAMES):           # a checkpoint written before a column was added
            qf = np.concatenate([qf, np.zeros((qf.shape[0], len(QF_NAMES) - qf.shape[1]))], axis=1)
        st.qf = qf.copy()
        st.qi = np.ascontiguousarray(arrays[f"{prefix}qi"], dtype=np.int64).copy()
        st.n_pass = st.qf.shape[0]
        for i in range(st.n_patch):
            st.facets.setdefault((int(st.pi[i, PI["workpiece"]]), int(st.pi[i, PI["brick"]])), []).append(i)
        for k, m in enumerate(meta["lines"]):
            rows = np.ascontiguousarray(arrays[f"{prefix}line.{k}.rows"], dtype=float).reshape(-1, N_ROW).copy()
            vec = np.asarray(arrays[f"{prefix}line.{k}.vec"], dtype=float)
            dl, nl = int(m["d_len"]), int(m["n_len"])
            origin, d, n, axis = vec[:3], vec[3:3 + dl], vec[3 + dl:3 + dl + nl], vec[3 + dl + nl:3 + dl + nl + 3]
            line = LineBuffer(patch=int(m["patch"]), pass_id=int(m["pass_id"]), sub=int(m["sub"]), slot=int(m["slot"]),
                              cls=str(m["cls"]), cylindrical=bool(m["cylindrical"]), origin=origin.copy(),
                              d=d.copy(), n=n.copy(), axis=axis.copy(), rho0=float(m["rho0"]), z0=float(m["z0"]),
                              rows=rows, count=int(m["count"]), first_tick=int(m["first_tick"]),
                              last_tick=int(m["last_tick"]), gap_sum=float(m["gap_sum"]), gap_n=int(m["gap_n"]),
                              block_first=int(m["block_first"]), block_last=int(m["block_last"]), op=int(m["op"]),
                              hold_ticks=float(m["hold_ticks"]))
            st.lines[tuple(int(v) for v in m["key"])] = line
        st.active = {(int(p), int(s)): int(q) for p, s, q in meta["active"]}
        return st


__all__ = ["Store", "PF", "PI", "QF", "QI", "PF_NAMES", "PI_NAMES", "QF_NAMES", "QI_NAMES"]
