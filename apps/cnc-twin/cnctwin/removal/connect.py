"""Loose bodies after subtraction (docs/models/removal.md "Loose bodies").

:func:`find_bodies` labels the level-0 column graph (k_connect); :func:`release_loose` keeps the held
material (components meeting a held region: the workholding's contact boxes in S; without any, the largest
component) and removes every other component from the field, returning what came loose.
:func:`release_loose_incremental` is the check of every macro step: when the material was one component at the
last check and only removals happened since, the columns around the bricks those removals changed decide
(k_connect.local_connected); the whole field is labelled only when they cannot prove that it is still one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import abi as A


@dataclass
class Body:
    root: int
    volume_m3: float
    centroid_S_m: np.ndarray
    box_lo_S_m: np.ndarray
    box_hi_S_m: np.ndarray
    held: bool = False

    def to_dict(self) -> dict:
        return {"volume_m3": self.volume_m3, "centroid_S_m": [float(v) for v in self.centroid_S_m],
                "box_lo_S_m": [float(v) for v in self.box_lo_S_m], "box_hi_S_m": [float(v) for v in self.box_hi_S_m]}


def find_bodies(field: A.DexelField):
    """(bodies sorted by volume descending (ties by root), first, parent, total) of level 0."""
    from .k_connect import component_stats, label_z

    nx, ny = int(field.lv_dims[0, 0]), int(field.lv_dims[0, 1])
    r0 = int(field.lv_ray0[0, 2])
    nseg = int(field.cnt[r0:r0 + nx * ny].sum())
    first = np.zeros(nx * ny + 1, np.int64)
    parent = np.zeros(max(nseg, 1), np.int32)
    total = int(label_z(field, first, parent))
    if total == 0:
        return [], first, parent, 0
    stats = np.zeros((total, 10))
    component_stats(field, first, parent, total, stats)
    roots = np.flatnonzero(parent[:total] == np.arange(total))
    bodies = []
    for r in roots:
        v = float(stats[r, 0])
        cen = stats[r, 1:4] / v if v > 0 else np.zeros(3)
        bodies.append(Body(int(r), v, cen, stats[r, 4:7].copy(), stats[r, 7:10].copy()))
    bodies.sort(key=lambda b: (-b.volume_m3, b.root))
    return bodies, first, parent, total


def _meets(body: Body, boxes) -> bool:
    for lo, hi in boxes:
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        if np.all(body.box_hi_S_m >= lo) and np.all(body.box_lo_S_m <= hi):
            return True
    return False


def release_loose(field: A.DexelField, held_boxes=()):
    """Remove the components not held; returns (released bodies, held bodies, removed level-0 z length)."""
    from .k_connect import remove_loose

    bodies, first, parent, total = find_bodies(field)
    if len(bodies) <= 1:
        for b in bodies:
            b.held = True
        return [], bodies, 0
    from .k_connect import components_in_box

    held_boxes = list(held_boxes or ())
    held_any = False
    if held_boxes:
        flags = np.zeros(max(total, 1), np.int8)
        for lo, hi in held_boxes:
            components_in_box(field, first, parent, total, np.ascontiguousarray(np.asarray(lo, dtype=float)),
                              np.ascontiguousarray(np.asarray(hi, dtype=float)), flags)
        for b in bodies:
            b.held = bool(flags[b.root]) and _meets(b, held_boxes)
            held_any = held_any or b.held
    if not held_any:
        for b in bodies:
            b.held = False
        bodies[0].held = True
    loose = np.zeros(max(total, 1), np.int8)
    released = [b for b in bodies if not b.held]
    for b in released:
        loose[b.root] = 1
    removed = int(remove_loose(field, first, parent, total, loose, 0.0, 0.0, 0.0, A.SEV_NONE, A.make_scratch()))
    field.info[A.FI_TOPO] = 0
    return released, [b for b in bodies if b.held], removed


def release_loose_incremental(field: A.DexelField, held_boxes=(), one_body: bool = False):
    """The connectivity check after removals: (released bodies, held bodies, removed level-0 z length, one_body after,
    local). ``one_body``: the material was one component at the last check and only subtractions changed it since
    (the caller's bookkeeping); then the changed region alone is labelled, and when it proves the material is still
    one component nothing is released (local = True). Otherwise :func:`release_loose`. Either way the change marks
    (``bconn``) and the topology counter (``FI_TOPO``) start again from zero."""
    from .k_connect import local_connected

    if one_body:
        box = np.zeros(6, np.int64)
        if int(local_connected(field, box)) != 0:
            field.bconn[:] = 0
            field.info[A.FI_TOPO] = 0
            return [], [], 0, True, True
    released, held, removed = release_loose(field, held_boxes)
    field.bconn[:] = 0
    field.info[A.FI_TOPO] = 0
    return released, held, removed, len(held) == 1, False
