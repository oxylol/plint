"""Marching-cubes case table built from per-face rules (docs/models/removal.md "Meshing").

The classic 15-case table (Lorensen & Cline 1987) resolves the ambiguous faces (two diagonal corners
inside) differently in neighbouring cubes and leaves holes (Durst 1988; Nielson & Hamann 1991). Here every
case is derived from its six faces with one fixed face rule, so the two cubes sharing a face always draw
the same contour segments on it and the mesh is closed:

* on a face, every edge whose corners differ carries one vertex;
* the vertices are paired around the inside corners: one inside corner, or two adjacent ones, give one
  segment; three inside corners one segment around the outside corner; two diagonal inside corners two
  segments that cut each inside corner off ("separate the inside", the face rule of Montani, Scateni &
  Scopigno 1994);
* the segments of the six faces close into loops (every vertex lies on two faces); each loop is a polygon
  oriented so its normal points from the inside corners to the outside corners, and fanned into triangles.

Corner i of a cell sits at offset (i & 1, (i >> 1) & 1, (i >> 2) & 1); edge e = 4 * axis + j joins
``EDGE_CORNERS[e]`` along ``axis`` from the lower to the upper corner.
"""

from __future__ import annotations

import functools

import numpy as np

CORNER_OFFSET = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], np.int64)


def _edges():
    edges = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for j in range(4):
            lo = [0, 0, 0]
            lo[others[0]] = j & 1
            lo[others[1]] = (j >> 1) & 1
            hi = list(lo)
            hi[axis] = 1
            ca = lo[0] + 2 * lo[1] + 4 * lo[2]
            cb = hi[0] + 2 * hi[1] + 4 * hi[2]
            edges.append((ca, cb, axis))
    return edges


EDGES = _edges()
EDGE_CORNERS = np.array([[a, b] for a, b, _ in EDGES], np.int64)
EDGE_AXIS = np.array([ax for _, _, ax in EDGES], np.int64)
EDGE_LOWER_OFFSET = np.array([CORNER_OFFSET[a] for a, _, _ in EDGES], np.int64)


def _edge_of(ca: int, cb: int) -> int:
    for e, (a, b, _) in enumerate(EDGES):
        if {a, b} == {ca, cb}:
            return e
    raise KeyError((ca, cb))


def _faces():
    """Six faces, corners in cyclic order."""
    faces = []
    for axis in range(3):
        for side in (0, 1):
            others = [a for a in range(3) if a != axis]
            cyc = [(0, 0), (1, 0), (1, 1), (0, 1)]
            corners = []
            for u, v in cyc:
                c = [0, 0, 0]
                c[axis] = side
                c[others[0]] = u
                c[others[1]] = v
                corners.append(c[0] + 2 * c[1] + 4 * c[2])
            faces.append(corners)
    return faces


FACES = _faces()


def _face_segments(case: int, corners: list[int]) -> list[tuple[int, int]]:
    inside = [(case >> c) & 1 for c in corners]
    n_in = sum(inside)
    if n_in in (0, 4):
        return []
    edges = [_edge_of(corners[k], corners[(k + 1) % 4]) for k in range(4)]   # edge k joins corner k, k+1
    cross = [inside[k] != inside[(k + 1) % 4] for k in range(4)]
    if n_in == 2 and inside[0] == inside[2]:
        # diagonal: cut every inside corner off
        segs = []
        for k in range(4):
            if inside[k]:
                segs.append((edges[(k - 1) % 4], edges[k]))
        return segs
    ids = [edges[k] for k in range(4) if cross[k]]
    return [(ids[0], ids[1])]


def _loops(segs: list[tuple[int, int]]) -> list[list[int]]:
    adj: dict[int, list[int]] = {}
    for a, b in segs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for e, nb in adj.items():
        if len(nb) != 2:
            raise AssertionError(f"edge {e} in {len(nb)} face segments")
    seen = set()
    loops = []
    for start in sorted(adj):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nb = adj[cur]
            nxt = nb[0] if nb[0] != prev else nb[1]
            if nxt == start:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def _orient(loop: list[int], case: int) -> list[int]:
    mid = {e: 0.5 * (CORNER_OFFSET[EDGES[e][0]] + CORNER_OFFSET[EDGES[e][1]]) for e in loop}
    normal = np.zeros(3)
    for k in range(len(loop)):
        p, q = mid[loop[k]], mid[loop[(k + 1) % len(loop)]]
        normal += np.cross(p, q)
    out = np.zeros(3)
    for e in loop:
        a, b, _ = EDGES[e]
        ia, ib = (case >> a) & 1, (case >> b) & 1
        inside, outside = (a, b) if ia else (b, a)
        out += CORNER_OFFSET[outside] - CORNER_OFFSET[inside]
    return loop if float(normal @ out) > 0.0 else loop[::-1]


@functools.lru_cache(maxsize=1)
def tables() -> tuple[np.ndarray, np.ndarray]:
    """(tri_count [256] int64, tri_edges [256, 12, 3] int64)."""
    count = np.zeros(256, np.int64)
    tri = np.full((256, 12, 3), -1, np.int64)
    for case in range(256):
        segs = []
        for face in FACES:
            segs.extend(_face_segments(case, face))
        k = 0
        for loop in _loops(segs):
            loop = _orient(loop, case)
            for i in range(1, len(loop) - 1):
                tri[case, k] = (loop[0], loop[i], loop[i + 1])
                k += 1
        count[case] = k
    return count, np.ascontiguousarray(tri)
