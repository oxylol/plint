"""Revolved profiles for the subtraction and collision kernels (docs/models/removal.md "Envelope profile").

:func:`profile_from_envelope` turns tooling's :class:`cnctwin.tooling.abi.Envelope` (closed chains of
frustum and arc pieces in the meridian half-plane) into the exact max-radius profile R(z) the kernels
use: every arc is split where z is extremal (a = k pi) into z-monotone halves, discs (z0 = z1) bound the
solid but add no radius, and the upper envelope max_i R_i(z) is resolved exactly: breakpoints at all
piece ends and at every crossing of two pieces (closed form for line-line, a quadratic for line-arc and
arc-arc, refined with Brent's method to 1e-15 m). Selection by cutting flag, chain and part gives the
cutting envelope (subtracted), the whole envelope (collision), or a single part.

:func:`profile_from_thread` gives a thread mill's annular rows as a profile; :func:`profile_polyline`
returns (z, r) points for cnctwin.geom's revolved-solid overlap test; :func:`tooth_slabs` turns one
tooth of an EdgeModel into the slab heights and radius offsets of a SubtractSpec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..tooling.abi import ENV_ARC, ENV_FRUSTUM, PART_BODY, PART_EDGE, Envelope
from . import abi as A

_TINY = 1e-15


@dataclass(frozen=True)
class Piece:
    kind: int           # PK_LINE / PK_ARC
    z0: float
    z1: float
    c0: float
    c1: float
    c2: float
    c3: float
    part: int
    index: int          # order of appearance (tie breaker)

    def radius(self, z: float) -> float:
        if self.kind == A.PK_LINE:
            return self.c0 + self.c1 * z
        d = self.c2 * self.c2 - (z - self.c0) ** 2
        return self.c1 + self.c3 * math.sqrt(max(d, 0.0))


def _envelope_pieces(env: Envelope, select) -> list[Piece]:
    out: list[Piece] = []
    for i in range(len(env.kind)):
        if not select(int(env.cutting[i]), int(env.chain[i]), int(env.part[i])):
            continue
        part = int(env.part[i])
        if int(env.kind[i]) == ENV_FRUSTUM:
            z0, r0, z1, r1 = float(env.z0_m[i]), float(env.r0_m[i]), float(env.z1_m[i]), float(env.r1_m[i])
            if abs(z1 - z0) <= _TINY:
                continue
            slope = (r1 - r0) / (z1 - z0)
            out.append(Piece(A.PK_LINE, min(z0, z1), max(z0, z1), r0 - slope * z0, slope, 0.0, 0.0, part, len(out)))
        elif int(env.kind[i]) == ENV_ARC:
            zc, rc, rho = float(env.zc_m[i]), float(env.rc_m[i]), float(env.rho_m[i])
            a0, a1 = float(env.a0_rad[i]), float(env.a1_rad[i])
            lo, hi = min(a0, a1), max(a0, a1)
            cuts = [lo]
            k = math.floor(lo / math.pi) + 1
            while k * math.pi < hi - 1e-15:
                cuts.append(k * math.pi)
                k += 1
            cuts.append(hi)
            for s in range(len(cuts) - 1):
                aa, bb = cuts[s], cuts[s + 1]
                if bb - aa <= 1e-15:
                    continue
                sigma = 1.0 if math.sin(0.5 * (aa + bb)) > 0.0 else -1.0
                za, zb = zc - rho * math.cos(aa), zc - rho * math.cos(bb)
                if abs(zb - za) <= _TINY:
                    continue
                out.append(Piece(A.PK_ARC, min(za, zb), max(za, zb), zc, rc, rho, sigma, part, len(out)))
    return out


def _crossings(p: Piece, q: Piece, lo: float, hi: float) -> list[float]:
    """z in (lo, hi) where R_p = R_q (sampled sign changes refined by Brent's method, plus tangencies
    resolved by the sample minimum of |R_p - R_q|)."""
    def diff(z):
        return p.radius(z) - q.radius(z)

    n = 64
    zs = np.linspace(lo, hi, n + 1)
    ds = np.array([diff(z) for z in zs])
    roots = []
    for k in range(n):
        a, b = zs[k], zs[k + 1]
        da, db = ds[k], ds[k + 1]
        if da == 0.0:
            roots.append(float(a))
        elif da * db < 0.0:
            roots.append(float(brentq(diff, a, b, xtol=1e-15, rtol=4.5e-16, maxiter=200)))
    return [z for z in roots if lo < z < hi]


def build_profile(pieces: list[Piece], r_dead: float = 0.0) -> A.Profile:
    """Upper envelope of the pieces as a Profile."""
    if not pieces:
        return A.empty_profile()
    bps = set()
    for p in pieces:
        bps.add(p.z0)
        bps.add(p.z1)
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            lo = max(pieces[i].z0, pieces[j].z0)
            hi = min(pieces[i].z1, pieces[j].z1)
            if hi - lo > 1e-13:
                bps.update(_crossings(pieces[i], pieces[j], lo, hi))
    zs = sorted(bps)
    rows = []
    for a, b in zip(zs[:-1], zs[1:]):
        if b - a <= 1e-13:
            continue
        mid = 0.5 * (a + b)
        best, best_r = None, -math.inf
        for p in pieces:
            if p.z0 <= mid <= p.z1:
                r = p.radius(mid)
                if r > best_r + 1e-15:
                    best, best_r = p, r
        if best is None or best_r <= 0.0 and best.radius(a) <= 0.0 and best.radius(b) <= 0.0:
            continue
        if rows and rows[-1][0] is best and abs(rows[-1][2] - a) <= 1e-13:
            rows[-1][2] = b
        else:
            rows.append([best, a, b])
    m = len(rows)
    prof = A.Profile(np.zeros(m), np.zeros(m), np.zeros(m, np.int32), np.zeros(m), np.zeros(m), np.zeros(m),
                     np.zeros(m), np.zeros(m, np.int32), np.zeros(m), np.zeros(A.N_PI))
    rmax_all = 0.0
    for k, (p, a, b) in enumerate(rows):
        prof.z0_m[k] = a
        prof.z1_m[k] = b
        prof.kind[k] = p.kind
        prof.c0[k] = p.c0
        prof.c1[k] = p.c1
        prof.c2[k] = p.c2
        prof.c3[k] = p.c3
        prof.part[k] = p.part
        rm = max(p.radius(a), p.radius(b))
        if p.kind == A.PK_ARC and p.c3 > 0 and a < p.c0 < b:
            rm = max(rm, p.radius(p.c0))
        prof.rmax_m[k] = max(rm, 0.0)
        rmax_all = max(rmax_all, rm)
    prof.info[A.PI_ZMIN] = rows[0][1] if rows else 0.0
    prof.info[A.PI_ZMAX] = rows[-1][2] if rows else 0.0
    prof.info[A.PI_RMAX] = rmax_all
    prof.info[A.PI_RDEAD] = r_dead
    return prof


def dead_radius(env: Envelope) -> float:
    """Radius of the non-cutting centre of the end face: where the lowest cutting piece starts when the
    envelope closes the bottom with a non-cutting disc from the axis (tooling: non-centre-cutting tools)."""
    best_z, best_r = math.inf, 0.0
    for i in range(len(env.kind)):
        if not env.cutting[i]:
            continue
        for z, r in ((float(env.z0_m[i]), float(env.r0_m[i])), (float(env.z1_m[i]), float(env.r1_m[i]))):
            if z < best_z - 1e-12 or (abs(z - best_z) <= 1e-12 and r < best_r):
                best_z, best_r = z, r
    if not math.isfinite(best_z):
        return 0.0
    # a non-cutting disc from the axis at the bottom of the cutting head
    for i in range(len(env.kind)):
        if env.cutting[i] or int(env.kind[i]) != ENV_FRUSTUM:
            continue
        if abs(float(env.z0_m[i]) - float(env.z1_m[i])) <= 1e-12 and abs(float(env.z0_m[i]) - best_z) <= 1e-9 \
                and min(float(env.r0_m[i]), float(env.r1_m[i])) <= 1e-12:
            return best_r
    return 0.0 if best_r <= 1e-9 else best_r


def profile_from_envelope(env: Envelope, which: str = "cutting", chains=None, parts=None) -> A.Profile:
    """``which``: "cutting" (pieces swept by cutting edges), "all" (every piece of the selected chains),
    "noncutting" (pieces without the cutting flag). ``chains`` / ``parts``: optional sets to keep."""
    def select(cutting, chain, part):
        if chains is not None and chain not in chains:
            return False
        if parts is not None and part not in parts:
            return False
        if which == "cutting":
            return cutting == 1
        if which == "noncutting":
            return cutting == 0
        return True

    pieces = _envelope_pieces(env, select)
    return build_profile(pieces, dead_radius(env) if which in ("cutting", "all") else 0.0)


def profile_from_points(z, r, part: int = PART_EDGE, r_dead: float = 0.0) -> A.Profile:
    """Profile of a z-ascending polyline (z, r) (tests, simple solids, stylus shanks)."""
    pieces = []
    for k in range(len(z) - 1):
        z0, z1, r0, r1 = float(z[k]), float(z[k + 1]), float(r[k]), float(r[k + 1])
        if z1 - z0 <= _TINY:
            continue
        slope = (r1 - r0) / (z1 - z0)
        pieces.append(Piece(A.PK_LINE, z0, z1, r0 - slope * z0, slope, 0.0, 0.0, part, len(pieces)))
    return build_profile(pieces, r_dead)


def profile_ball(radius_m: float, length_m: float, part: int = PART_EDGE) -> A.Profile:
    """Ball end (sphere of radius R about (R, 0)) then a cylinder to ``length_m`` (tests, probe styli)."""
    R = float(radius_m)
    pieces = [Piece(A.PK_ARC, 0.0, R, R, 0.0, R, 1.0, part, 0)]
    if length_m > R:
        pieces.append(Piece(A.PK_LINE, R, float(length_m), R, 0.0, 0.0, 0.0, part, 1))
    return build_profile(pieces)


def profile_torus(radius_m: float, corner_m: float, length_m: float, part: int = PART_EDGE) -> A.Profile:
    """Toroidal end mill: flat end to R - rc, corner arc of radius rc, cylinder to ``length_m``."""
    R, rc = float(radius_m), float(corner_m)
    pieces = [Piece(A.PK_ARC, 0.0, rc, rc, R - rc, rc, 1.0, part, 0)]
    if length_m > rc:
        pieces.append(Piece(A.PK_LINE, rc, float(length_m), R, 0.0, 0.0, 0.0, part, 1))
    return build_profile(pieces)


def profile_from_thread(thread, tooth: int = 0, part: int = PART_EDGE) -> A.Profile:
    """Upper envelope of a thread mill's (or tap land's) row polygons (z, r) as a profile."""
    pieces = []
    for m in range(thread.pts_zr_m.shape[1]):
        n = int(thread.n_pts[tooth, m])
        if n < 3:
            continue
        pts = np.asarray(thread.pts_zr_m[tooth, m, :n], dtype=float)
        for k in range(n):
            (z0, r0), (z1, r1) = pts[k], pts[(k + 1) % n]
            if abs(z1 - z0) <= _TINY:
                continue
            slope = (r1 - r0) / (z1 - z0)
            pieces.append(Piece(A.PK_LINE, min(z0, z1), max(z0, z1), r0 - slope * z0, slope, 0.0, 0.0, part,
                                len(pieces)))
    # the tool body below the rows (filled to the axis) up to the root radius
    if pieces:
        z_lo = min(p.z0 for p in pieces)
        z_hi = max(p.z1 for p in pieces)
        root = float(thread.major_r_m[0] - thread.depth_m[0])
        if root > 0.0:
            pieces.append(Piece(A.PK_LINE, z_lo, z_hi, root, 0.0, 0.0, 0.0, PART_BODY, len(pieces)))
    return build_profile(pieces)


def radius_at(prof: A.Profile, z: float) -> float:
    """R(z) of a profile (0 outside it)."""
    for k in range(len(prof.z0_m)):
        if prof.z0_m[k] - 1e-15 <= z <= prof.z1_m[k] + 1e-15:
            if prof.kind[k] == A.PK_LINE:
                return float(prof.c0[k] + prof.c1[k] * z)
            d = prof.c2[k] ** 2 - (z - prof.c0[k]) ** 2
            return float(prof.c1[k] + prof.c3[k] * math.sqrt(max(d, 0.0)))
    return 0.0


def profile_polyline(prof: A.Profile, chord_tol_m: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """(z, r) points of a profile with arcs chorded to ``chord_tol_m`` (inscribed chords lifted by half the
    sagitta so the polyline error is +- chord_tol / 2), closed to the axis at both ends."""
    zs, rs = [], []
    for k in range(len(prof.z0_m)):
        a, b = float(prof.z0_m[k]), float(prof.z1_m[k])
        if prof.kind[k] == A.PK_LINE:
            pts = [a, b]
            lift = 0.0
        else:
            rho = float(prof.c2[k])
            n = max(2, int(math.ceil((b - a) / max(math.sqrt(8.0 * rho * chord_tol_m), 1e-9))) + 1)
            pts = list(np.linspace(a, b, n))
            lift = 0.5 * chord_tol_m
        for z in pts:
            r = radius_at(prof, z) + lift
            if zs and abs(z - zs[-1]) <= 1e-15:
                rs[-1] = max(rs[-1], r)
            else:
                zs.append(z)
                rs.append(r)
    if not zs:
        return np.zeros(2), np.zeros(2)
    z = np.array([zs[0]] + zs + [zs[-1]])
    r = np.array([0.0] + rs + [0.0])
    return z, r


def tooth_slabs(edge, prof: A.Profile, tooth: int, slab_height_m: float, max_slabs: int = 64,
                z_lo: float | None = None, z_hi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(slab_z [ns + 1], slab_dr [ns]) of one tooth of an EdgeModel against a profile.

    The tooth's valid elements are sorted by height and grouped greedily into slabs at least
    ``slab_height_m`` high (boundaries half way between neighbouring elements; the first slab starts at the
    profile's lowest point, the last ends at its top), so every slab holds elements of the tooth. A slab's
    offset is the largest (element distance from the spindle axis - R(z)) of its elements: runout offset,
    tilt, wear and growth of the effective edge are included. A tooth without valid elements gives one
    slab with offset -R_max (it does not cut)."""
    zmin = float(prof.info[A.PI_ZMIN]) if z_lo is None else float(z_lo)
    zmax = float(prof.info[A.PI_ZMAX]) if z_hi is None else float(z_hi)
    rmax = float(prof.info[A.PI_RMAX])
    hand = float(np.asarray(edge.hand_sign).reshape(-1)[0])
    ox, oy, _ = (float(v) for v in np.asarray(edge.runout_T_m).reshape(3))
    tx, ty = (float(v) for v in np.asarray(edge.tilt_rad).reshape(2))
    zs, ds = [], []
    for k in range(edge.elem_r_m.shape[1]):
        if not edge.elem_valid[tooth, k]:
            continue
        r = float(edge.elem_r_m[tooth, k])
        z = float(edge.elem_z_m[tooth, k])
        alpha = hand * (float(edge.tooth_phase_rad[tooth]) + float(edge.elem_lag_rad[tooth, k]))
        x, y = r * math.cos(alpha), r * math.sin(alpha)
        px = x + ty * z + ox
        py = y - tx * z + oy
        zs.append(z)
        ds.append(math.hypot(px, py) - radius_at(prof, z))
    if not zs:
        return np.array([zmin, zmax]), np.array([-rmax])
    order = np.argsort(np.asarray(zs), kind="stable")
    zs = np.asarray(zs)[order]
    ds = np.asarray(ds)[order]
    height = max(float(slab_height_m), (zmax - zmin) / max(1, int(max_slabs)))
    bounds = [zmin]
    drs = []
    cur = -math.inf
    start = zmin
    for k in range(len(zs)):
        cur = max(cur, float(ds[k]))
        last = k == len(zs) - 1
        if not last:
            cut = 0.5 * (float(zs[k]) + float(zs[k + 1]))
            if cut - start >= height and cut < zmax:
                bounds.append(cut)
                drs.append(cur)
                cur = -math.inf
                start = cut
    bounds.append(zmax)
    drs.append(cur)
    return np.asarray(bounds, dtype=float), np.asarray(drs, dtype=float)


def helix_from_thread(thread, hand_sign: float) -> A.HelixSweep:
    """HelixSweep of a tap from tooling's ThreadProfile: every (land, row) polygon, ascending in z per land,
    oriented counter-clockwise in (z, r)."""
    nt, n_rows = thread.pts_zr_m.shape[0], thread.pts_zr_m.shape[1]
    polys, counts, boxes, first, phases = [], [], [], [0], []
    nmax = 3
    for j in range(nt):
        rows = []
        for m in range(n_rows):
            n = int(thread.n_pts[j, m])
            if n < 3:
                continue
            pts = np.asarray(thread.pts_zr_m[j, m, :n], dtype=float)
            area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
            if area < 0.0:
                pts = pts[::-1]
            rows.append(pts)
        rows.sort(key=lambda q: float(q[:, 0].min()))
        for pts in rows:
            polys.append(pts)
            counts.append(len(pts))
            boxes.append([pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()])
            nmax = max(nmax, len(pts))
        first.append(len(polys))
        phases.append(float(thread.tooth_phase_rad[j]))
    poly = np.zeros((len(polys), nmax, 2))
    for k, pts in enumerate(polys):
        poly[k, :len(pts)] = pts
    boxes_a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    info = np.zeros(4)
    info[0] = float(hand_sign)
    if len(polys):
        info[1] = float(boxes_a[:, 0].min())
        info[2] = float(boxes_a[:, 1].max())
        info[3] = float(boxes_a[:, 3].max())
    return A.HelixSweep(np.ascontiguousarray(poly), np.asarray(counts, np.int32), np.ascontiguousarray(boxes_a),
                        np.asarray(first, np.int32), np.asarray(phases, dtype=float), info)
