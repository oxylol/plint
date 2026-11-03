"""Hard-milling program for the H13 die cavity (die_h13.h, HEIDENHAIN Klartext), computed from part.step.

Run: cd apps/cnc-twin/examples/die-h13-ballnose && TSAFE_MEM=1800M ../../bin/tsafe python make_program.py
(about 5 minutes on 4 cores)

The cavity is machined in two 3+2 halves: x >= 0 with PLANE SPATIAL SPB+15 (tool leaning towards +x),
x <= 0 with SPB-15, 1 mm overlap. With the tool leaning away from the half it cuts, the angle between the
tool axis and the surface normal is the surface slope plus 15 deg, so the ball never cuts with its centre
(Tungaloy p. I054: tilt the tool about 15 deg to avoid cutting at zero speed at the axis).

1. The cavity, its rim fillet and the parting face around them are tessellated from part.step (chordal
   deviation 2 um: on the concave cavity the facets lie on the tool side of the surface, so the error is
   material left, never a gouge) and expressed in the tilted working plane of each half.
2. Ball drop-cutter on the triangles (vertex, edge and facet tests): ball centre on the vertical (tool
   axis) through each raster point at the highest contact, for a ball of radius R + s (s = stock left);
   the program point is the ball tip, centre - R along the tool axis. Each point keeps its contact normal.
   Points every DX along the line; where the chord between two points would leave the ball-centre path
   by more than SAG_TOL (over the convex rim: into the part) the midpoint is added.
3. Raster lines parallel to x' (cut direction along the long axis of the cavity), zig-zag. A line runs
   while the ball touches the cavity or the rim fillet and continues until the contact is MARGIN onto the
   parting face, so the rim is rolled over completely; the lines cover y from one tangent line of the rim
   with the parting face to the other. Links: LINK_UP along the tool axis and across at F_LINK (inside the
   concave cavity and over the flat parting face the straight link between two lifted positions is free),
   down into the cut at the cutting feed.
4. Stepover for a constant scallop height h (TOOLS[...]["h"]) from a probe grid of ball-centre heights
   z'(x', y) of both halves (drop cutter every PROBE_DY x PROBE_DX): in the normal section across the
   lines, h = dc^2 / (8 R) * (1 + R k), dc the spacing of neighbouring ball centres across the path and
   k the normal curvature of the centre surface in that direction (positive in the concave cavity,
   negative over the convex rim), both from the probe grid (scallop_factor); the stepover at y is the
   smallest dy over the line and the next PROBE_DY. The scallop between the final lines is predicted from
   the same grid and written to toolpath_stats.json.
5. Spindle speed per tool from the distribution of the effective cutting diameter D sin(psi + kappa)
   (psi = contact angle from the tool axis, kappa = acos(1 - ap/R), ap the largest engagement): S puts vc
   at the maker's value at the 80th percentile; the vc range is reported.

Writes die_h13.h and toolpath_stats.json.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
DATUM_S = np.array([50.0, 40.0, 40.0])          # preset 1: centre of the parting face
R = 5.0                                         # TEB100E4L ball radius
TILT_DEG = 15.0
OVERLAP = 0.5                                   # each half runs 0.5 mm past x = 0 (contact point)
MARGIN = 0.3                                    # mm the contact runs onto the parting face past the rim
TOOLS = {
    # rough: blank allowance 0.3 -> 0.06 + scallop <= 0.04, so the finishing engagement is 0.06-0.10 mm
    "rough": {"T": 1, "flutes": 4, "fz": 0.028, "vc": 75.0, "stock": 0.06, "ap": 0.24, "h": 0.040,
              "name": "TEB100E4L-20/30C10CFH72 AH710", "tol": "32.1 T0.02", "hsc": "32.2 HSC-MODE:1",
              "title": "HARD ROUGHING, STOCK 0.3 -> 0.06, SCALLOP 0.04"},
    "finish": {"T": 2, "flutes": 4, "fz": 0.024, "vc": 100.0, "stock": 0.0, "ap": 0.10, "h": 0.0045,
               "name": "TEB100E4L-20/30C10CFH72 AH710", "tol": "32.1 T0.005", "hsc": "32.2 HSC-MODE:0",
               "title": "FINISHING RASTER, SCALLOP 0.0045"},
}
MAX_RPM = 15000
DX = 0.25                                       # mm between raster points along a line
SAG_TOL = 0.0003                                # mm, chord vs ball-centre path (vertical, estimated)
PROBE_DY, PROBE_DX, CURV_SPAN = 0.25, 0.5, 1.0  # probe grid; second difference over +-CURV_SPAN
LINK_UP = 0.5
F_LINK = 1500
POOL = 4
# machine (dmg-dmu50-3rd): table point q (table frame at B0 C0) -> machine coords (q_x, q_y, q_z - 625 + L)
PIVOT_B = np.array([0.0, 0.0, -120.0])
PRESET_F = np.array([0.0, 0.0, 67.0])           # datum in the table frame: grid plate 27 + insert 40
TOOL_L = {1: 65.0 + 40.0, 2: 65.0 + 40.0}       # HAIMER 40.445.10.3 A 65 + stickout
TRAVEL = {"X": (-325.0, 325.0), "Y": (-260.0, 260.0), "Z": (-475.0, 0.0)}


# ---------------------------------------------------------------------------------------------- geometry
def triangles_from_step(path: Path, deflection=0.002):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise SystemExit(f"cannot read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.05, False)
    tris = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            nodes = np.array([[tri.Node(i).Transformed(trsf).X(), tri.Node(i).Transformed(trsf).Y(),
                               tri.Node(i).Transformed(trsf).Z()] for i in range(1, tri.NbNodes() + 1)])
            rev = face.Orientation() == TopAbs_REVERSED
            for k in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(k).Get()
                t = nodes[[a - 1, c - 1, b - 1]] if rev else nodes[[a - 1, b - 1, c - 1]]
                tris.append(t)
        explorer.Next()
    tris = np.array(tris)
    keep = tris[:, :, 2].max(axis=1) > DATUM_S[2] - 15.0     # cavity, rim fillet, parting face (sides are far away)
    return tris[keep]


def rot_y(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])


# ---------------------------------------------------------------------------------------------- drop cutter
class DropCutter:
    """Ball drop-cutter against a triangle set in a frame whose +z is the tool axis."""

    CHUNK = 8                                                # raster points per vectorised block

    def __init__(self, tris):
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norm = np.linalg.norm(n, axis=1)
        good = norm > 1e-12                                   # drop degenerate triangles
        tris, n, norm = tris[good], n[good], norm[good]
        n = n / norm[:, None]
        n[n[:, 2] < 0] *= -1.0                               # every facet faces the tool (+z side)
        self.tris, self.n = tris, n
        self.d = np.einsum("ij,ij->i", n, tris[:, 0])
        self.lo = tris[:, :, :2].min(axis=1)
        self.hi = tris[:, :, :2].max(axis=1)
        edges = []
        for a_i, b_i in ((0, 1), (1, 2), (2, 0)):
            A, B = tris[:, a_i], tris[:, b_i]
            L = np.linalg.norm(B - A, axis=1)
            U = (B - A) / np.where(L > 1e-12, L, 1.0)[:, None]
            edges.append((A, U, L))
        self.edges = edges

    def line(self, y, xs, radius):
        """Ball centre z and contact normal for the points (xs, y)."""
        strip = np.nonzero((self.lo[:, 1] - radius <= y) & (self.hi[:, 1] + radius >= y))[0]
        best = np.full(len(xs), -np.inf)
        normal = np.zeros((len(xs), 3))
        r2 = radius * radius
        lo_x, hi_x = self.lo[strip, 0], self.hi[strip, 0]
        for j0 in range(0, len(xs), self.CHUNK):
            x = xs[j0:j0 + self.CHUNK]
            near = (lo_x - radius <= x[-1]) & (hi_x + radius >= x[0])
            idx = strip[near]
            if idx.size == 0:
                continue
            T, N, Dn = self.tris[idx], self.n[idx], self.d[idx]
            px, py = x[:, None], y
            # facet test: centre c with n.(c - v0) = r, contact c - r n inside the triangle
            nz = N[:, 2]
            ok = nz > 1e-9
            zc = (radius + Dn[None, :] - N[None, :, 0] * px - N[None, :, 1] * py) / np.where(ok, nz, 1.0)[None, :]
            inside = _inside_xy(T, px - radius * N[None, :, 0], py - radius * N[None, :, 1]) & ok[None, :]
            zf = np.where(inside, zc, -np.inf)
            _take(best, normal, j0, zf, lambda k, m: N[m])
            # vertex tests
            for v in range(3):
                V = T[:, v]
                d2 = (px - V[None, :, 0]) ** 2 + (py - V[None, :, 1]) ** 2
                zv = np.where(d2 <= r2, V[None, :, 2] + np.sqrt(np.maximum(r2 - d2, 0.0)), -np.inf)
                _take(best, normal, j0, zv, lambda k, m, V=V, zv=zv: np.stack(
                    [x[k] - V[m, 0], np.full(len(k), y) - V[m, 1], zv[k, m] - V[m, 2]], axis=1) / radius)
            # edge tests
            for A_all, U_all, L_all in self.edges:
                A, U, L = A_all[idx], U_all[idx], L_all[idx]
                dx = px - A[None, :, 0]
                dy = py - A[None, :, 1]
                p = dx * U[None, :, 0] + dy * U[None, :, 1]
                qa = 1.0 - U[None, :, 2] ** 2
                qb = -2.0 * p * U[None, :, 2]
                qc = dx * dx + dy * dy - p * p - r2
                disc = qb * qb - 4 * qa * qc
                valid = (qa > 1e-9) & (disc >= 0) & (L[None, :] > 1e-12)
                t = np.where(valid, (-qb + np.sqrt(np.maximum(disc, 0.0))) / (2 * np.where(valid, qa, 1.0)), 0.0)
                sp = p + t * U[None, :, 2]
                valid &= (sp >= 0) & (sp <= L[None, :])
                ze = np.where(valid, A[None, :, 2] + t, -np.inf)

                def edge_normal(k, m, A=A, U=U, sp=sp, ze=ze):
                    foot = A[m] + sp[k, m][:, None] * U[m]
                    c = np.stack([x[k], np.full(len(k), y), ze[k, m]], axis=1)
                    return (c - foot) / radius
                _take(best, normal, j0, ze, edge_normal)
        return best, normal


def _inside_xy(T, cx, cy):
    """Point (cx, cy) [k, m] inside triangle m's xy projection."""
    ax, ay = T[None, :, 0, 0], T[None, :, 0, 1]
    bx, by = T[None, :, 1, 0], T[None, :, 1, 1]
    qx, qy = T[None, :, 2, 0], T[None, :, 2, 1]
    d1 = (cx - bx) * (ay - by) - (ax - bx) * (cy - by)
    d2 = (cx - qx) * (by - qy) - (bx - qx) * (cy - qy)
    d3 = (cx - ax) * (qy - ay) - (qx - ax) * (cy - ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _take(best, normal, j0, z, normal_of):
    """Keep, per raster point, the highest z of this test and its contact normal (computed only there)."""
    m = np.argmax(z, axis=1)
    k = np.arange(z.shape[0])
    zbest = z[k, m]
    seg = slice(j0, j0 + z.shape[0])
    better = zbest > best[seg]
    if not better.any():
        return
    kb = k[better]
    best[j0 + kb] = zbest[better]
    normal[j0 + kb] = normal_of(kb, m[better])


_WORKER = {}


def _worker_init(tris):
    _WORKER["cutter"] = DropCutter(tris)


def _worker_line(args):
    y, xs, radius = args
    return _WORKER["cutter"].line(y, xs, radius)


def drop_lines(tilted, jobs, radius):
    """Ball centres z' and contact normals for jobs [(y, xs), ...] in the tilted frame (POOL processes)."""
    with mp.get_context("fork").Pool(POOL, initializer=_worker_init, initargs=(tilted,)) as pool:
        return pool.map(_worker_line, [(y, xs, radius) for y, xs in jobs], chunksize=2)


def drop_grid(tilted, ys, xs, radius):
    res = drop_lines(tilted, [(y, xs) for y in ys], radius)
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


# ---------------------------------------------------------------------------------------------- raster
class Cavity:
    """Where a contact counts as cavity: below the parting face, or on it within MARGIN of a cavity facet."""

    def __init__(self, tris_d):
        n = np.cross(tris_d[:, 1] - tris_d[:, 0], tris_d[:, 2] - tris_d[:, 0])
        nz = np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-30)
        cav = (tris_d[:, :, 2] < -1e-6).any(axis=1) & (nz > 0.2)       # not the vertical sides of the block
        verts = tris_d[cav].reshape(-1, 3)
        self.tree = cKDTree(verts[:, :2])
        self.y_lo, self.y_hi = float(verts[:, 1].min()), float(verts[:, 1].max())

    def contains(self, contact):
        flat = contact.reshape(-1, 3)
        below = np.isfinite(flat[:, 2]) & (flat[:, 2] < -1e-4)
        xy = np.where(np.isfinite(flat[:, :2]), flat[:, :2], 1e9)
        dist, _ = self.tree.query(xy, distance_upper_bound=MARGIN + 1e-9)
        return (below | (dist <= MARGIN)).reshape(contact.shape[:-1])


def contacts_part(frame, xs, ys, zc, nrm, radius):
    centre = np.stack(np.broadcast_arrays(xs[None, :], ys[:, None], zc), axis=-1)
    return np.einsum("ij,abj->abi", frame, centre - radius * nrm)


def _shift(a, di, dj):
    """a[i + di, j + dj] with NaN outside."""
    out = np.full_like(a, np.nan)
    ny, nx = a.shape
    out[max(-di, 0):ny - max(di, 0), max(-dj, 0):nx - max(dj, 0)] = \
        a[max(di, 0):ny - max(-di, 0) or None, max(dj, 0):nx - max(-dj, 0) or None]
    return out


def scallop_factor(z):
    """fac(y, x) with h = dy^2 fac for raster lines along x' spaced dy, z = ball-centre height on the probe grid.
    Path direction t = (1, 0, p), centre-surface normal ~ (-p, -q, 1), cross direction b = n x t ~ (-pq, 1 + p^2, q):
    the centres of neighbouring lines are dc = dy sqrt((1 + p^2 + q^2) / (1 + p^2)) apart along b, and
    h = dc^2 / (8 R) (1 + R k_b) with k_b the normal curvature of the centre surface along b (concave > 0).
    Derivatives by central differences over +-CURV_SPAN / 2 (slopes) and +-CURV_SPAN (curvatures)."""
    hx, hy = PROBE_DX, PROBE_DY
    jx, iy = int(round(0.5 * CURV_SPAN / hx)), int(round(0.5 * CURV_SPAN / hy))
    p = (_shift(z, 0, jx) - _shift(z, 0, -jx)) / (2 * jx * hx)
    q = (_shift(z, iy, 0) - _shift(z, -iy, 0)) / (2 * iy * hy)
    zxx = (_shift(z, 0, 2 * jx) - 2 * z + _shift(z, 0, -2 * jx)) / (2 * jx * hx) ** 2
    zyy = (_shift(z, 2 * iy, 0) - 2 * z + _shift(z, -2 * iy, 0)) / (2 * iy * hy) ** 2
    zxy = (_shift(z, iy, jx) - _shift(z, iy, -jx) - _shift(z, -iy, jx) + _shift(z, -iy, -jx)) / (4 * iy * hy * jx * hx)
    P = 1.0 + p * p
    Q = P + q * q
    k_b = (zxx * p * p * q * q - 2.0 * zxy * p * q * P + zyy * P * P) / (np.sqrt(Q) * P * Q)
    return (Q / P) * np.maximum(1.0 + R * k_b, 0.25) / (8.0 * R)


def line_positions(halves, kind, cavity):
    """y of the raster lines for a constant scallop height (module docstring, step 4), and the probe data."""
    spec = TOOLS[kind]
    radius = R + spec["stock"]
    ys = np.arange(cavity.y_lo - 1.5, cavity.y_hi + 1.5 + 1e-9, PROBE_DY)
    xs = np.arange(-30.0, 30.0 + 1e-9, PROBE_DX)
    w_flat = math.sqrt(8.0 * R * spec["h"])
    need = np.full(len(ys), w_flat)
    probes = []
    for side, frame, tilted in halves:
        zc, nrm = drop_grid(tilted, ys, xs, radius)
        contact = contacts_part(frame, xs, ys, zc, nrm, radius)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(zc) & cavity.contains(contact) & (contact[..., 0] * side >= -OVERLAP)
        fac = scallop_factor(np.where(np.isfinite(zc), zc, np.nan))           # h = dy^2 * fac
        valid &= np.isfinite(fac)
        dy_ok = np.where(valid, np.sqrt(spec["h"] / np.where(valid, fac, 1.0)), np.inf)
        need = np.minimum(need, dy_ok.min(axis=1))
        probes.append((fac, valid))
    lines = [cavity.y_lo]
    while lines[-1] < cavity.y_hi - 1e-6:
        y = lines[-1]
        w = w_flat
        for _ in range(4):                                     # w only shrinks: conservative
            sel = (ys >= y - PROBE_DY) & (ys <= y + w + PROBE_DY)
            w = min(w, float(need[sel].min()))
        lines.append(min(y + w, cavity.y_hi))
    return lines, (ys, probes)


def predicted_scallop(lines, probe):
    ys, probes = probe
    hs = []
    for a, b in zip(lines[:-1], lines[1:]):
        i = int(np.argmin(np.abs(ys - 0.5 * (a + b))))
        for fac, valid in probes:
            if valid[i].any():
                hs.append((b - a) ** 2 * fac[i][valid[i]])
    h = np.concatenate(hs) * 1000.0
    return {"p50": round(float(np.percentile(h, 50)), 2), "p95": round(float(np.percentile(h, 95)), 2),
            "max": round(float(h.max()), 2)}


def raster(tilted, frame, side, kind, ys, cavity):
    """CL points of one half: lines with tips, contact angles and part-frame points. Points every DX, plus
    the midpoint of every segment whose chord would leave the ball-centre path by more than SAG_TOL
    (estimated from the second difference of z'; over the convex rim the chord would cut into it)."""
    spec = TOOLS[kind]
    radius = R + spec["stock"]
    ys = np.asarray(ys)
    xs = np.arange(-30.0, 30.0 + 1e-9, DX)
    zc, nrm = drop_grid(tilted, ys, xs, radius)
    contact = contacts_part(frame, xs, ys, zc, nrm, radius)
    with np.errstate(invalid="ignore"):
        inside = cavity.contains(contact) & (contact[..., 0] * side >= -OVERLAP) & np.isfinite(zc)
    rows, jobs = [], []
    for i, y in enumerate(ys):
        if not inside[i].any():
            continue
        idx = np.nonzero(inside[i])[0]
        idx = np.arange(max(idx[0] - 1, 0), min(idx[-1] + 2, len(xs)))      # one point more at each end
        idx = idx[np.isfinite(zc[i, idx])]
        z = zc[i, idx]
        d2 = np.zeros(len(idx))
        d2[1:-1] = np.abs(z[:-2] - 2.0 * z[1:-1] + z[2:])
        sag = np.maximum(d2[:-1], d2[1:]) / 8.0
        rows.append((i, idx))
        jobs.append((y, xs[idx[:-1][sag > SAG_TOL]] + 0.5 * DX))
    extra = drop_lines(tilted, jobs, radius)
    lines = []
    for (i, idx), (y, xm), (zm, nm) in zip(rows, jobs, extra):
        x = np.concatenate([xs[idx], xm])
        z = np.concatenate([zc[i, idx], zm])
        n = np.concatenate([nrm[i, idx], nm.reshape(-1, 3)])
        order = np.argsort(x, kind="stable")
        x, z, n = x[order], z[order], n[order]
        ok = np.isfinite(z)
        x, z, n = x[ok], z[ok], n[ok]
        tip = np.column_stack([x, np.full(len(x), y), z - R])
        c_part = (frame @ (np.column_stack([x, np.full(len(x), y), z]) - radius * n).T).T
        psi = np.degrees(np.arccos(np.clip(n[:, 2], -1, 1)))
        lines.append({"y": float(y), "tip": tip, "psi": psi, "part": (frame @ tip.T).T,
                      "cutting": c_part[:, 2] < -1e-4, "refined": int(len(xm))})
    return lines


def effective_vc(lines, n_rpm, ap):
    kappa = math.degrees(math.acos(1.0 - ap / R))
    psi = np.concatenate([ln["psi"][ln["cutting"]] for ln in lines])
    d_eff = 2 * R * np.sin(np.radians(np.minimum(psi + kappa, 90.0)))
    return math.pi * d_eff * n_rpm / 1000.0, d_eff, psi


def choose_rpm(lines, spec):
    _, d_eff, _ = effective_vc(lines, 1000.0, spec["ap"])
    d80 = float(np.percentile(d_eff, 80))
    return int(min(MAX_RPM, round(spec["vc"] * 1000.0 / (math.pi * d80) / 10.0) * 10))


# ---------------------------------------------------------------------------------------------- output
def fmt(v):
    return f"{v:+.4f}"                           # TNC 640 input resolution 0.1 um


def rot_axis(axis, deg):
    a = np.asarray(axis, float)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def machine_xyz(part_pts, b_deg, tool):
    """Linear machine axes for tool tips at part points (datum frame) with the table at B (C 0)."""
    q = PRESET_F + part_pts
    q = PIVOT_B + (rot_axis([0, -1, 0], b_deg) @ (q - PIVOT_B).T).T
    return np.column_stack([q[:, 0], q[:, 1], q[:, 2] - 625.0 + TOOL_L[tool]])


def main():
    tris_d = triangles_from_step(HERE / "part.step") - DATUM_S
    cavity = Cavity(tris_d)
    halves = []
    for side in (+1, -1):
        frame = rot_y(TILT_DEG * side)                        # tilted frame -> datum frame
        halves.append((side, frame, np.einsum("ij,mkj->mki", frame.T, tris_d)))
    out = ["0 BEGIN PGM DIE_H13 MM"]

    def add(text):
        out.append(f"{len(out)} {text}")

    add("BLK FORM 0.1 Z X-50 Y-40 Z-40")
    add("BLK FORM 0.2 X+50 Y+40 Z+0")
    for c in ("DIE-H13-BALLNOSE OP10 - CNC TWIN EXAMPLE - WRITTEN BY MAKE_PROGRAM.PY FROM PART.STEP",
              "DMG MORI DMU 50 3RD GEN - HEIDENHAIN TNC 640 - B/C SWIVEL ROTARY TABLE",
              "H13 50 HRC DIE INSERT 100 X 80 X 40 ON LANG QUICK-POINT 45600 (2 CLAMPING STUDS)",
              "CAVITY 48 X 32 X 9.6 (ELLIPTIC PARABOLOID, RIM R2) - BLANK SOFT-ROUGHED TO 0.3 MM, HARDENED",
              "PRESET 1 = CENTRE OF THE PARTING FACE",
              "T1 TUNGALOY TEB100E4L-20/30C10CFH72 AH710 D10 BALL 4FL - HAIMER 40.445.10.3 - STICKOUT 40",
              "T2 TUNGALOY TEB100E4L-20/30C10CFH72 AH710 D10 BALL 4FL (NEW) - HAIMER 40.445.10.3 - STICKOUT 40",
              "TWO 3+2 HALVES: X>=0 PLANE SPATIAL SPB+15, X<=0 SPB-15 (NO CUTTING AT THE BALL CENTRE)",
              "AIR BLAST THROUGH THE COOLANT NOZZLES (M8, COOLANT/AIR SWITCH ON AIR) PER TUNGALOY"):
        add(f"; {c}")
    add("CYCL DEF 247 PRESETTING ~")
    out[-1] += "\n    Q339=+1 ;PRESET NUMBER"
    stats = {"tools": {}, "travel": {}}
    travel = []
    for kind in ("rough", "finish"):
        spec = TOOLS[kind]
        ys, probe = line_positions(halves, kind, cavity)
        done = [(side, frame, raster(tilted, frame, side, kind, ys, cavity)) for side, frame, tilted in halves]
        all_lines = [ln for _, _, lines in done for ln in lines]
        rpm = choose_rpm(all_lines, spec)
        feed = int(round(rpm * spec["flutes"] * spec["fz"]))
        vc, _, psi = effective_vc(all_lines, rpm, spec["ap"])
        add(f"; ---- T{spec['T']} {spec['name']} - {spec['title']}")
        add(f"TOOL CALL {spec['T']} Z S{rpm} F{feed}")
        add("L Z-10 R0 FMAX M91")
        cut = link = plunge = 0.0
        for side, frame, lines in done:
            b_deg = TILT_DEG * side
            add(f"PLANE SPATIAL SPA+0 SPB{b_deg:+.0f} SPC+0 TURN MB MAX FMAX {'SEQ+' if side > 0 else 'SEQ-'}")
            add("CYCL DEF 32.0 TOLERANCE")
            add(f"CYCL DEF {spec['tol']}")
            add(f"CYCL DEF {spec['hsc']}")
            first = lines[0]["tip"][0]
            add(f"L X{fmt(first[0])} Y{fmt(first[1])} R0 FMAX M3")
            add(f"L Z{fmt(first[2] + 10.0)} R0 FMAX M8")
            add(f"L Z{fmt(first[2] + LINK_UP)} FMAX")
            prev = None
            for j, ln in enumerate(lines):
                pts = ln["tip"] if j % 2 == 0 else ln["tip"][::-1]
                if prev is not None:
                    add(f"L Z{fmt(prev[2] + LINK_UP)} F{F_LINK}")
                    add(f"L X{fmt(pts[0][0])} Y{fmt(pts[0][1])} Z{fmt(pts[0][2] + LINK_UP)}")
                    link += LINK_UP + float(np.linalg.norm(pts[0] - prev))
                plunge += LINK_UP
                add(f"L X{fmt(pts[0][0])} Y{fmt(pts[0][1])} Z{fmt(pts[0][2])} F{feed}")
                for p in pts[1:]:
                    add(f"L X{fmt(p[0])} Y{fmt(p[1])} Z{fmt(p[2])}")
                cut += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                prev = pts[-1]
                travel.append(machine_xyz(ln["part"], b_deg, spec["T"]))
            add(f"L Z{fmt(prev[2] + 10.0)} R0 FMAX")
            add("PLANE RESET TURN MB MAX FMAX")
        add("L Z-10 R0 FMAX M91 M9")
        dys = np.diff(ys)
        stats["tools"][kind] = {
            "T": spec["T"], "S": rpm, "F": feed, "stock_mm": spec["stock"], "scallop_target_um": spec["h"] * 1000,
            "lines": len(all_lines), "points": int(sum(len(ln["tip"]) for ln in all_lines)),
            "points_added_for_chord": int(sum(ln["refined"] for ln in all_lines)),
            "cut_path_mm": round(cut, 1), "link_path_mm": round(link, 1), "plunge_path_mm": round(plunge, 1),
            "feed_time_s": round((cut + plunge) / feed * 60.0 + link / F_LINK * 60.0, 1),
            "vc_m_min": {"p5": round(float(np.percentile(vc, 5)), 1), "p50": round(float(np.percentile(vc, 50)), 1),
                         "p80": round(float(np.percentile(vc, 80)), 1), "max": round(float(vc.max()), 1)},
            "contact_angle_deg": {"min": round(float(psi.min()), 1), "p50": round(float(np.percentile(psi, 50)), 1),
                                  "max": round(float(psi.max()), 1)},
            "stepover_mm": {"min": round(float(dys.min()), 3), "max": round(float(dys.max()), 3)},
            "scallop_um": predicted_scallop(ys, probe),
            "y_range_mm": [round(ys[0], 3), round(ys[-1], 3)],
        }
    add("M30")
    out.append(f"{len(out)} END PGM DIE_H13 MM")
    travel = np.concatenate(travel)
    for i, key in enumerate("XYZ"):
        lo, hi = float(travel[:, i].min()), float(travel[:, i].max())
        stats["travel"][key] = [round(lo, 1), round(hi, 1)]
        if not (TRAVEL[key][0] <= lo and hi <= TRAVEL[key][1]):
            raise SystemExit(f"axis {key} out of travel: {lo:.1f} .. {hi:.1f}")
    (HERE / "die_h13.h").write_text("\n".join(out) + "\n")
    (HERE / "toolpath_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
