"""3-D transient field of the tool's cutting region (docs/models/thermal.md 2.4).

The region is a wedge at the cutting edge, meshed in its own edge frame E: x along the rake face
normal to the edge (into the tool), y along the (unrolled) edge, z the rake face's outward normal
(the tool is at z < 0). The cross-section normal to the edge is a polar grid about the edge line,
rings r_k = r_1 g^(k-1) (geometric, so the contact scale and the body scale are both resolved),
equal sectors over the wedge angle beta (rake face theta = 0, flank face theta = -beta), clipped
by the section of the tool body:

* **insert**: the quadrilateral edge - rake back (x = ic) - bottom back - flank bottom (depth = the
  insert thickness): the bottom is the seat on the holder bar, coupled to the bar's first ladder
  node by the seat contact (db/models/thermal-tool.yaml shank contact conductance over the seat
  area) in series with the bar's quasi-steady internal resistance H / 3k (slab heated on one face,
  Carslaw & Jaeger 1959 ch. III); the field replaces the ladder's insert nodes, which become views
  (mass-weighted means of the cells by distance from the edge), so the ladder API keeps working;
* **tooth** (solid rotating tools): the wedge of one tooth out to the flute depth R_w (core
  diameter from the tool item, else from the radius whose disc has the fluted section's area), its outer
  arc coupled to the ladder node at the same height through the quasi-steady parabolic profile of
  the fluted section, 8 pi k per unit length shared by the teeth (cylinder with a uniform surface
  flux: T_s - T_mean = q R / 4k, Carslaw & Jaeger 1959 ch. VII); one representative tooth carries the
  sum of the teeth's heat with every capacity and conductance multiplied by the number of teeth
  (the system stays symmetric), the overlapped ladder section, mass and wetted perimeter are
  removed from the ladder.

Heat enters as uniform fluxes over each element's rake strip (l_c from the edge) and flank strip
(l_f), mapped onto the rake and flank faces by overlap (exact energy). Exposed faces convect to air
(the ladder's correlation of that tool, plus linearised radiation) and to the coolant over the
reach. The field and the ladder are one implicit system (kernels.field_step: backward Euler,
conjugate gradients with a block symmetric Gauss-Seidel preconditioner); energy is booked by the
tool's ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.abi import EDGE_NOSE, EdgeModel
from ..core.errors import InputError
from . import convection as cv
from . import kernels as K
from .abi import (FD_ITER, FD_RES, FS_MAX_ITER, FS_TOL, N_FD, FieldMesh, FieldSystem, Ladder, MatTable,
                  new_field_system)

# boundary face kinds
FACE_RAKE = 0
FACE_FLANK = 1
FACE_END = 2            # the two ends of the prism along the edge
FACE_BACK = 3           # other exposed faces of the section (insert back)
FACE_SEAT = 4           # insert bottom on the holder (coupled)
FACE_ROOT = 5           # tooth root on the fluted core (coupled)

ARC_POINTS = 32         # polygon points per arc of a clipped cell (area error ~ (dtheta / 32)^2)
MIN_CELL_FRACTION = 1e-9


# ------------------------------------------------------------------ plane geometry of the section
def _halfplanes(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normals n and offsets c of a counter-clockwise convex polygon: inside is n.p <= c."""
    v = np.asarray(poly, dtype=float)
    e = np.roll(v, -1, axis=0) - v
    length = np.linalg.norm(e, axis=1)
    nrm = np.stack([e[:, 1], -e[:, 0]], axis=1) / length[:, None]
    return nrm, np.einsum("ij,ij->i", nrm, v)


def _clip(subject: np.ndarray, nrm: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon by the convex region n.p <= c (any simple subject)."""
    out = [np.asarray(p, dtype=float) for p in subject]
    for k in range(c.size):
        if not out:
            break
        src = out
        out = []
        for i in range(len(src)):
            p = src[i - 1]
            q = src[i]
            dp = float(nrm[k] @ p - c[k])
            dq = float(nrm[k] @ q - c[k])
            if dq <= 0.0:
                if dp > 0.0:
                    out.append(p + dp / (dp - dq) * (q - p))
                out.append(q)
            elif dp <= 0.0:
                out.append(p + dp / (dp - dq) * (q - p))
    return np.array(out).reshape(-1, 2)


def _area_centroid(poly: np.ndarray) -> tuple[float, float, float]:
    if poly.shape[0] < 3:
        return 0.0, 0.0, 0.0
    x, z = poly[:, 0], poly[:, 1]
    x1, z1 = np.roll(x, -1), np.roll(z, -1)
    cross = x * z1 - x1 * z
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-300:
        return 0.0, float(x.mean()), float(z.mean())
    return area, float(((x + x1) * cross).sum() / (6.0 * area)), float(((z + z1) * cross).sum() / (6.0 * area))


def _sector_polygon(r0: float, r1: float, th_lo: float, th_hi: float) -> np.ndarray:
    t = np.linspace(th_lo, th_hi, ARC_POINTS + 1)
    outer = np.stack([r1 * np.cos(t), r1 * np.sin(t)], axis=1)
    if r0 <= 0.0:
        return np.vstack([np.zeros((1, 2)), outer])
    inner = np.stack([r0 * np.cos(t[::-1]), r0 * np.sin(t[::-1])], axis=1)
    return np.vstack([outer, inner])


def _ray_reach(theta: float, nrm: np.ndarray, c: np.ndarray) -> float:
    """Distance from the origin (a vertex of the polygon) to the boundary along direction theta."""
    e = np.array([math.cos(theta), math.sin(theta)])
    proj = nrm @ e
    sel = proj > 1e-12
    if not np.any(sel):
        return math.inf
    return float(np.min(c[sel] / proj[sel]))


def _inside(p: np.ndarray, nrm: np.ndarray, c: np.ndarray, tol: float) -> bool:
    return bool(np.all(nrm @ p <= c + tol))


def _arc_inside(r: float, th_lo: float, th_hi: float, nrm: np.ndarray, c: np.ndarray) -> float:
    """Angle (rad) of the arc of radius r over [th_lo, th_hi] that lies inside the polygon (exact)."""
    pts = [th_lo, th_hi]
    for k in range(c.size):
        s = c[k] / r
        if abs(s) <= 1.0:
            phi = math.atan2(nrm[k, 1], nrm[k, 0])
            d = math.acos(s)
            for t in (phi + d, phi - d):
                for m in (-1, 0, 1):
                    tt = t + 2.0 * math.pi * m
                    if th_lo < tt < th_hi:
                        pts.append(tt)
    pts = sorted(pts)
    total = 0.0
    tol = 1e-12 * max(r, 1e-12)
    for a, b in zip(pts, pts[1:]):
        mid = 0.5 * (a + b)
        if b > a and _inside(np.array([r * math.cos(mid), r * math.sin(mid)]), nrm, c, tol):
            total += b - a
    return total


def _angle(p: np.ndarray) -> float:
    th = math.atan2(p[1], p[0])
    return th - 2.0 * math.pi if th > 1e-12 else th


def _segment_in_sector(p: np.ndarray, q: np.ndarray, r0: float, r1: float, th_lo: float, th_hi: float) -> float:
    """Length of the part of segment p-q inside the annular sector."""
    d = q - p
    seg = float(np.linalg.norm(d))
    if seg <= 0.0:
        return 0.0
    ts = [0.0, 1.0]
    a = float(d @ d)
    bq = 2.0 * float(p @ d)
    for r in (r0, r1):
        if r <= 0.0:
            continue
        cc = float(p @ p) - r * r
        disc = bq * bq - 4.0 * a * cc
        if disc >= 0.0:
            root = math.sqrt(disc)
            for t in ((-bq - root) / (2.0 * a), (-bq + root) / (2.0 * a)):
                if 0.0 < t < 1.0:
                    ts.append(t)
    for th in (th_lo, th_hi):
        ex, ez = math.cos(th), math.sin(th)
        den = ex * d[1] - ez * d[0]
        if abs(den) > 1e-300:
            t = (ez * p[0] - ex * p[1]) / den
            if 0.0 < t < 1.0:
                ts.append(t)
    ts = sorted(ts)
    length = 0.0
    for ta, tb in zip(ts, ts[1:]):
        if tb <= ta:
            continue
        m = p + 0.5 * (ta + tb) * d
        rm = float(np.linalg.norm(m))
        thm = _angle(m)
        if r0 - 1e-15 <= rm <= r1 + 1e-15 and th_lo - 1e-12 <= thm <= th_hi + 1e-12:
            length += (tb - ta) * seg
    return length


def _line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    e = b - a
    return abs(float(e[0] * (point[1] - a[1]) - e[1] * (point[0] - a[0]))) / float(np.linalg.norm(e))


@dataclass
class Section:
    """Polar cross-section mesh of the wedge (one slab), clipped by the tool's section polygon."""

    beta: float
    r_edges: np.ndarray            # [nk + 1], r_edges[0] = 0
    th_edges: np.ndarray           # [nj + 1], 0 .. -beta
    index: np.ndarray              # [nk, nj] cross-section cell or -1
    ring: np.ndarray               # [n2]
    sector: np.ndarray             # [n2]
    area: np.ndarray               # [n2] m2
    cx: np.ndarray                 # [n2] centroid (x, z) in E
    cz: np.ndarray
    lnk_a: np.ndarray              # in-plane internal faces
    lnk_b: np.ndarray
    lnk_len: np.ndarray
    lnk_da: np.ndarray
    lnk_db: np.ndarray
    bnd_cell: np.ndarray           # in-plane boundary faces
    bnd_len: np.ndarray
    bnd_d: np.ndarray
    bnd_kind: np.ndarray
    bnd_r0: np.ndarray             # radial interval of rake / flank faces (distance from the edge)
    bnd_r1: np.ndarray


def ring_edges(r1: float, growth: float, r_max: float) -> np.ndarray:
    """0, r1, r1 g, ... with the last edge exactly r_max (a last ring thinner than 30 % of the one
    before is merged into it)."""
    edges = [0.0]
    r = r1
    while r < r_max * (1.0 - 1e-9):
        edges.append(r)
        r *= growth
    if len(edges) >= 3 and (r_max - edges[-1]) < 0.3 * (edges[-1] - edges[-2]):
        edges.pop()
    edges.append(r_max)
    return np.array(edges)


def build_section(beta: float, r_edges: np.ndarray, n_sector: int, poly: np.ndarray | None,
                  edge_kinds: list[int] | None) -> Section:
    """Cells, internal faces and boundary faces of the clipped polar wedge.

    ``poly``: counter-clockwise convex section with the edge point at the origin and its first edge
    along the flank (origin -> flank bottom) and its last along the rake face (-> origin); ``edge_kinds``
    the FACE_* kind of every polygon edge. None: the full sector, its outer arc a coupled root."""
    th = np.linspace(0.0, -beta, n_sector + 1)
    nk = r_edges.size - 1
    nj = n_sector
    if poly is not None:
        nrm, c = _halfplanes(poly)
    index = -np.ones((nk, nj), dtype=np.int64)
    ring, sector, area, cx, cz = [], [], [], [], []
    for k in range(nk):
        r0, r1 = r_edges[k], r_edges[k + 1]
        for j in range(nj):
            th_hi, th_lo = th[j], th[j + 1]
            dth = th_hi - th_lo
            full = 0.5 * (r1 * r1 - r0 * r0) * dth
            poly_s = _sector_polygon(r0, r1, th_lo, th_hi)
            inside = poly is None or all(_inside(p, nrm, c, 1e-12 * r1) for p in poly_s)
            if inside:
                rc = (2.0 / 3.0) * (r1 ** 3 - r0 ** 3) / (r1 * r1 - r0 * r0) * math.sin(0.5 * dth) / (0.5 * dth)
                tc = 0.5 * (th_hi + th_lo)
                a, x0, z0 = full, rc * math.cos(tc), rc * math.sin(tc)
            else:
                a, x0, z0 = _area_centroid(_clip(poly_s, nrm, c))
            if a <= MIN_CELL_FRACTION * full:
                continue
            index[k, j] = len(area)
            ring.append(k)
            sector.append(j)
            area.append(a)
            cx.append(x0)
            cz.append(z0)
    ring = np.array(ring, dtype=np.int64)
    sector = np.array(sector, dtype=np.int64)
    area = np.array(area)
    cx = np.array(cx)
    cz = np.array(cz)
    cr = np.hypot(cx, cz)
    la, lb, ll, lda, ldb = [], [], [], [], []
    ba, bl, bd, bk, b0, b1 = [], [], [], [], [], []
    for k in range(nk):
        r0, r1 = r_edges[k], r_edges[k + 1]
        width = r1 - r0
        for j in range(nj):
            a = index[k, j]
            if a < 0:
                continue
            th_hi, th_lo = th[j], th[j + 1]
            # arc face to the next ring
            if k + 1 < nk:
                bcell = index[k + 1, j]
                ang = (th_hi - th_lo) if poly is None else _arc_inside(r1, th_lo, th_hi, nrm, c)
                if bcell >= 0 and ang > 0.0:
                    la.append(a)
                    lb.append(bcell)
                    ll.append(r1 * ang)
                    lda.append(max(r1 - cr[a], 0.05 * width))
                    ldb.append(max(cr[bcell] - r1, 0.05 * (r_edges[k + 2] - r1)))
            elif poly is None:
                ba.append(a)
                bl.append(r1 * (th_hi - th_lo))
                bd.append(max(r1 - cr[a], 0.05 * width))
                bk.append(FACE_ROOT)
                b0.append(r0)
                b1.append(r1)
            # radial faces: to the next sector, or the rake / flank face
            for side, th_f in ((0, th_hi), (1, th_lo)):
                reach = math.inf if poly is None else _ray_reach(th_f, nrm, c)
                length = max(0.0, min(r1, reach) - r0)
                if length <= 0.0:
                    continue
                ux, uz = math.cos(th_f), math.sin(th_f)
                da = max(abs(cx[a] * uz - cz[a] * ux), 1e-3 * width)
                if side == 0 and j == 0:
                    kind = FACE_RAKE
                elif side == 1 and j == nj - 1:
                    kind = FACE_FLANK
                else:
                    if side == 1:
                        bcell = index[k, j + 1]
                        if bcell >= 0:
                            la.append(a)
                            lb.append(bcell)
                            ll.append(length)
                            lda.append(da)
                            ldb.append(max(abs(cx[bcell] * uz - cz[bcell] * ux), 1e-3 * width))
                    continue
                ba.append(a)
                bl.append(length)
                bd.append(da)
                bk.append(kind)
                b0.append(r0)
                b1.append(r0 + length)
            # polygon edges that are not the rake or flank face
            if poly is not None:
                for e in range(poly.shape[0]):
                    kind = edge_kinds[e]
                    if kind in (FACE_RAKE, FACE_FLANK):
                        continue
                    p = poly[e]
                    q = poly[(e + 1) % poly.shape[0]]
                    length = _segment_in_sector(p, q, r0, r1, th_lo, th_hi)
                    if length <= 1e-9 * r1:
                        continue
                    ba.append(a)
                    bl.append(length)
                    bd.append(max(_line_distance(np.array([cx[a], cz[a]]), p, q), 1e-3 * width))
                    bk.append(kind)
                    b0.append(r0)
                    b1.append(r1)
    arr = lambda v, t=float: np.array(v, dtype=t)
    order = np.lexsort((arr(lb, np.int64), arr(la, np.int64)))
    return Section(beta=float(beta), r_edges=np.asarray(r_edges, dtype=float), th_edges=th, index=index, ring=ring,
                   sector=sector, area=area, cx=cx, cz=cz, lnk_a=arr(la, np.int64)[order],
                   lnk_b=arr(lb, np.int64)[order], lnk_len=arr(ll)[order], lnk_da=arr(lda)[order],
                   lnk_db=arr(ldb)[order], bnd_cell=arr(ba, np.int64), bnd_len=arr(bl), bnd_d=arr(bd),
                   bnd_kind=arr(bk, np.int64), bnd_r0=arr(b0), bnd_r1=arr(b1))


def insert_polygon(beta: float, back_m: float, thickness_m: float) -> tuple[np.ndarray, list[int]]:
    """Counter-clockwise section of an insert at its cutting edge: edge, flank bottom, back bottom,
    rake back (flank, seat, back and rake faces)."""
    flank_x = thickness_m / math.tan(beta) if abs(math.tan(beta)) > 1e-12 else 0.0
    if beta > 0.5 * math.pi:
        flank_x = -thickness_m * math.tan(beta - 0.5 * math.pi)
    poly = np.array([[0.0, 0.0], [flank_x, -thickness_m], [back_m, -thickness_m], [back_m, 0.0]])
    return poly, [FACE_FLANK, FACE_SEAT, FACE_BACK, FACE_RAKE]


def slab_edges(boundaries: np.ndarray, length_m: float, dy_max: float, required=()) -> np.ndarray:
    """Edges along the edge axis: the element boundaries in [0, length] (closer ones than dy_max / 2
    merged) and the ``required`` ones (contact ends, never merged), intervals longer than dy_max split
    equally."""
    req = np.unique(np.clip(np.asarray(required, dtype=float), 0.0, length_m))
    cand = np.unique(np.clip(np.concatenate([[0.0, length_m], np.asarray(boundaries, dtype=float), req]), 0.0,
                             length_m))
    kept = [0.0]
    for y in cand[1:]:
        if y - kept[-1] >= 0.5 * dy_max or y >= length_m or np.any(np.abs(req - y) <= 1e-12):
            kept.append(float(y))
    if kept[-1] < length_m:
        kept.append(length_m)
    if len(kept) >= 3 and kept[-1] - kept[-2] < 0.5 * dy_max and not np.any(np.abs(req - kept[-2]) <= 1e-12):
        kept.pop(-2)
    out = [kept[0]]
    for a, b in zip(kept, kept[1:]):
        n = max(1, int(math.ceil((b - a) / dy_max - 1e-9)))
        out.extend(a + (b - a) * (i + 1) / n for i in range(n))
    return np.array(out)


# ------------------------------------------------------------------ elements along the field edge
@dataclass
class FieldElements:
    """Which edge elements the field carries and where (representative tooth = tooth 0)."""

    elem: np.ndarray               # [nE] element index e of tooth 0
    side: np.ndarray               # [nE] 0: along the field edge (y0..y1), 1: minor edge at the end face (x0..x1)
    lo: np.ndarray                 # [nE] y0 (side 0) or x0 (side 1)
    hi: np.ndarray                 # [nE]
    in_field: np.ndarray           # [nt, ne] bool


def unroll_elements(edges: EdgeModel, turning: bool, length_m: float) -> FieldElements:
    """Element positions along the field's straight edge axis by cumulative chip width.

    Turning tools with a nose: the axis starts at the nose's end toward the minor edge and runs
    through the nose along the major edge (the one with the smaller kappa); the minor edge's elements
    lie along x on the prism's end face (a 90 deg corner). Otherwise all valid elements of tooth 0 in
    their order (rotating tools: from the tip), up to ``length_m``."""
    valid = np.asarray(edges.elem_valid[0]) != 0
    idx = np.flatnonzero(valid)
    db = np.asarray(edges.elem_db_m[0], dtype=float)
    nt, ne = np.shape(edges.elem_valid)
    elem, side, lo, hi = [], [], [], []
    nose = idx[np.asarray(edges.elem_kind[0])[idx] == EDGE_NOSE] if turning else np.zeros(0, np.int64)
    if nose.size:
        before = idx[idx < nose.min()]
        after = idx[idx > nose.max()]
        kappa = np.asarray(edges.elem_kappa_rad[0], dtype=float)
        mean = lambda s: float(np.mean(kappa[s])) if s.size else math.inf
        if mean(after) < mean(before):
            major, minor, nose_order = after, before[::-1], nose
        else:
            major, minor, nose_order = before[::-1], after, nose[::-1]
        y = 0.0
        for e in list(nose_order) + list(major):
            elem.append(int(e))
            side.append(0)
            lo.append(y)
            y += db[e]
            hi.append(y)
        x = 0.0
        for e in minor:
            elem.append(int(e))
            side.append(1)
            lo.append(x)
            x += db[e]
            hi.append(x)
    else:
        y = 0.0
        for e in idx:
            if y >= length_m:
                break
            elem.append(int(e))
            side.append(0)
            lo.append(y)
            y += db[e]
            hi.append(y)
    elem = np.array(elem, dtype=np.int64)
    in_field = np.zeros((nt, ne), dtype=bool)
    in_field[:, elem] = np.asarray(edges.elem_valid)[:, elem] != 0
    return FieldElements(elem, np.array(side, dtype=np.int64), np.array(lo), np.array(hi), in_field)


def _overlap(a0, a1, b0, b1):
    """Overlap length of intervals (broadcasting)."""
    return np.maximum(np.minimum(a1, b1) - np.maximum(a0, b0), 0.0)


# ------------------------------------------------------------------ the field
class ToolField:
    """Cutting-region field of one tool slot (built by :meth:`build`, stepped with the ladder)."""

    def __init__(self):
        self.kind = "insert"
        self.beta = 0.5 * math.pi
        self.mult = 1.0
        self.section: Section | None = None
        self.y_edges = np.zeros(2)
        self.mesh: FieldMesh | None = None
        self.sys: FieldSystem | None = None
        self.face_kind = np.zeros(0, np.int64)
        self.face_slab = np.zeros(0, np.int64)
        self.face_r0 = np.zeros(0)
        self.face_r1 = np.zeros(0)
        self.face_T = np.zeros(0)
        self.face_cover = np.zeros(0)       # fraction of the face under a contact (no convection)
        self.face_exposed = np.zeros(0, np.int8)   # convects to air / coolant (not coupled, not a cut)
        self.elements: FieldElements | None = None
        self.view_of_cell = np.zeros(0, np.int64)
        self.view_nodes = np.zeros(0, np.int64)
        self.view_cell = np.zeros(0, np.int64)
        self.view_node = np.zeros(0, np.int64)
        self.view_w = np.zeros(0)
        self.slab_node = np.zeros(0, np.int64)
        self.ref_diameter_m = 0.01
        self.rho = 1.0
        self.poly = np.zeros((0, 2))
        self._face_energy = np.zeros(0)
        self.iterations = 0
        self.residual = 0.0
        self.frames: dict = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, *, kind: str, beta: float, ring1_m: float, growth: float, sector_rad: float, dy_max_m: float,
              length_m: float, section_back_m: float, thickness_m: float, root_radius_m: float, mult: float,
              mat_tool: MatTable, mat_holder: MatTable, ladder: Ladder, layout,
              elements: FieldElements | None, elem_z_m: np.ndarray | None, seat: dict | None,
              ref_diameter_m: float, tol: float, max_iter: int, frames: dict | None = None,
              contact_edges_m=()) -> "ToolField":
        obj = cls()
        obj.kind = kind
        obj.beta = float(beta)
        obj.mult = float(mult)
        obj.rho = float(mat_tool.rho_kg_m3[0])
        obj.ref_diameter_m = float(ref_diameter_m)
        obj.elements = elements
        obj.frames = frames or {}
        n_sector = max(3, int(math.ceil(beta / sector_rad - 1e-9)))
        if kind == "insert":
            poly, kinds = insert_polygon(beta, section_back_m, thickness_m)
            r_max = float(np.max(np.hypot(poly[:, 0], poly[:, 1])))
            obj.poly = poly
        else:
            poly, kinds = None, None
            r_max = root_radius_m
            obj.poly = np.zeros((0, 2))
        sec = build_section(beta, ring_edges(ring1_m, growth, r_max), n_sector, poly, kinds)
        obj.section = sec
        bounds = np.zeros(0)
        if elements is not None and elements.elem.size:
            sel = elements.side == 0
            bounds = np.concatenate([elements.lo[sel], elements.hi[sel]])
        obj.y_edges = slab_edges(bounds, length_m, dy_max_m, contact_edges_m)
        obj._extrude(ladder, layout, mat_tool, elem_z_m, seat, root_radius_m)
        obj.sys.scal[FS_TOL] = tol
        obj.sys.scal[FS_MAX_ITER] = float(max_iter)
        return obj

    def _extrude(self, ladder: Ladder, layout, mat_tool: MatTable, elem_z_m, seat, root_radius_m):
        sec = self.section
        n2 = sec.area.size
        dy = np.diff(self.y_edges)
        ny = dy.size
        n = n2 * ny
        mult = self.mult
        rho = self.rho
        cell = lambda iy, c2: iy * n2 + c2
        m = (np.tile(sec.area, ny) * np.repeat(dy, n2)) * rho * mult
        la, lb, area, da, db = [], [], [], [], []
        for iy in range(ny):
            la.append(cell(iy, sec.lnk_a))
            lb.append(cell(iy, sec.lnk_b))
            area.append(sec.lnk_len * dy[iy] * mult)
            da.append(sec.lnk_da)
            db.append(sec.lnk_db)
            if iy + 1 < ny:
                c2 = np.arange(n2)
                la.append(cell(iy, c2))
                lb.append(cell(iy + 1, c2))
                area.append(sec.area * mult)
                da.append(np.full(n2, 0.5 * dy[iy]))
                db.append(np.full(n2, 0.5 * dy[iy + 1]))
        la = np.concatenate(la).astype(np.int32)
        lb = np.concatenate(lb).astype(np.int32)
        # faces: in-plane boundary faces per slab, then the two prism ends
        fc, fa, fd, fk, fs_, f0, f1 = [], [], [], [], [], [], []
        for iy in range(ny):
            fc.append(cell(iy, sec.bnd_cell))
            fa.append(sec.bnd_len * dy[iy] * mult)
            fd.append(sec.bnd_d)
            fk.append(sec.bnd_kind)
            fs_.append(np.full(sec.bnd_cell.size, iy))
            f0.append(sec.bnd_r0)
            f1.append(sec.bnd_r1)
        for iy, d_half in ((0, 0.5 * dy[0]), (ny - 1, 0.5 * dy[-1])):
            c2 = np.arange(n2)
            fc.append(cell(iy, c2))
            fa.append(sec.area * mult)
            fd.append(np.full(n2, d_half))
            fk.append(np.full(n2, FACE_END))
            fs_.append(np.full(n2, iy))
            f0.append(sec.r_edges[sec.ring])
            f1.append(sec.r_edges[sec.ring + 1])
        face_cell = np.concatenate(fc).astype(np.int32)
        face_area = np.concatenate(fa)
        face_d = np.concatenate(fd)
        self.face_kind = np.concatenate(fk).astype(np.int64)
        self.face_slab = np.concatenate(fs_).astype(np.int64)
        self.face_r0 = np.concatenate(f0)
        self.face_r1 = np.concatenate(f1)
        nf = face_cell.size
        face_node = -np.ones(nf, dtype=np.int32)
        face_rc = np.zeros(nf)
        face_lk = np.zeros(nf)
        nn = ladder.m.size
        active = np.ones(nn, dtype=np.int8)
        self.slab_node = -np.ones(ny, dtype=np.int64)
        if self.kind == "insert":
            seat_faces = self.face_kind == FACE_SEAT
            bottom = float(face_area[seat_faces].sum())
            if bottom <= 0.0:
                raise InputError("thermal: the insert field has no seat face")
            # seat contact over its area fraction of the bottom, in series with the bar's internal resistance
            face_node[seat_faces] = int(seat["node"])
            face_rc[seat_faces] = bottom / (mult * seat["conductance_W_m2K"] * seat["contact_area_m2"])
            face_lk[seat_faces] = seat["bar_height_m"] / 3.0
            tool_nodes = np.asarray(layout.tool_nodes, dtype=np.int64)
            self.view_nodes = tool_nodes.copy()
            active[tool_nodes] = 0
            zs = np.asarray(layout.z_m, dtype=float)[tool_nodes]
            bounds = 0.5 * (zs[1:] + zs[:-1])
            rad = np.tile(np.hypot(sec.cx, sec.cz), ny)
            self.view_of_cell = tool_nodes[np.searchsorted(bounds, rad, side="right")]
            # every cell's mass goes to the nodes its ring overlaps (by radial overlap), so each node is
            # the mean over the cells at its distance from the edge and no node is left empty
            lo = np.concatenate([[0.0], bounds])
            hi = np.concatenate([bounds, [np.inf]])
            r0 = np.tile(sec.r_edges[sec.ring], ny)
            r1 = np.tile(sec.r_edges[sec.ring + 1], ny)
            ov = _overlap(r0[:, None], r1[:, None], lo[None, :], hi[None, :]) / (r1 - r0)[:, None]
            vc, vn = np.nonzero(ov > 0.0)
            self.view_cell = vc.astype(np.int64)
            self.view_node = tool_nodes[vn]
            self.view_w = ov[vc, vn]
        else:
            # tooth root: the ladder node at the height of the slab (field elements' z along the edge axis)
            ys = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
            el = self.elements
            if el is not None and el.elem.size and elem_z_m is not None:
                mids = 0.5 * (el.lo + el.hi)
                z_slab = np.interp(ys, mids, np.asarray(elem_z_m, dtype=float)[0, el.elem])
            else:
                z_slab = ys
            tool_nodes = np.asarray(layout.tool_nodes, dtype=np.int64)
            zs = np.asarray(layout.z_m, dtype=float)[tool_nodes]
            bounds = 0.5 * (zs[1:] + zs[:-1])
            self.slab_node = tool_nodes[np.searchsorted(bounds, z_slab, side="right")]
            root = self.face_kind == FACE_ROOT
            face_node[root] = self.slab_node[self.face_slab[root]].astype(np.int32)
            # 8 pi k per unit length for all teeth: per unit root area of the representative tooth
            # (multiplicity mult) the resistance is mult R_w beta / (8 pi k)
            face_lk[root] = mult * root_radius_m * self.beta / (8.0 * math.pi)
            self.view_of_cell = -np.ones(n, dtype=np.int64)
            self._remove_overlap(ladder, layout, ny, dy, root_radius_m)
        self.mesh, self.sys = _assemble(n, ny, m, la, lb, np.concatenate(area), np.concatenate(da), np.concatenate(db),
                                        face_cell, face_area, face_d, face_node, face_rc, face_lk, ladder, active)
        # cells start at the temperature of the ladder node they replace or sit on
        node_of_cell = self.view_of_cell if self.kind == "insert" else self.slab_node[np.repeat(np.arange(ny), n2)]
        for node in np.unique(node_of_cell):
            sel = node_of_cell == node
            self.sys.T[sel] = float(ladder.T[node])
            self.sys.h[sel] = float(K.enthalpy(mat_tool, float(ladder.T[node])))
        self.face_T = self.sys.T[face_cell].copy()
        self.face_cover = np.zeros(nf)
        self._face_energy = np.zeros(nf)
        # an insert's prism ends are its faces; a tooth's are cuts through the tooth (it continues beyond the
        # field, and the teeth meet at the tip): adiabatic
        cut = (self.face_kind == FACE_END) if self.kind == "tooth" else np.zeros(nf, dtype=bool)
        self.face_exposed = ((face_node < 0) & ~cut).astype(np.int8)
        if self.kind == "insert":
            self.update_views(ladder, mat_tool, set_mass=True)

    def _remove_overlap(self, ladder: Ladder, layout, ny: int, dy: np.ndarray, root_radius_m: float) -> None:
        """Take the field's teeth (volume, mass and wetted surface) out of the ladder nodes they lie in
        (each node keeps at least a fifth of its section and perimeter)."""
        sec = self.section
        tooth_area = float(sec.area.sum())
        volume = {}
        surface = {}
        for iy in range(ny):
            node = int(self.slab_node[iy])
            volume[node] = volume.get(node, 0.0) + self.mult * tooth_area * float(dy[iy])
            surface[node] = surface.get(node, 0.0) + self.mult * 2.0 * root_radius_m * float(dy[iy])
        for node in sorted(volume):
            dz = float(ladder.dz[node])
            area = float(ladder.area[node])
            keep = max(area * dz - volume[node], 0.2 * area * dz) / dz
            ladder.m[node] *= keep / area
            ladder.area[node] = keep
            perim = float(layout.perimeter_m[node])
            layout.perimeter_m[node] = max(perim * dz - surface[node], 0.2 * perim * dz) / dz

    # ------------------------------------------------------------------ views (insert)
    def update_views(self, ladder: Ladder, mat_tool: MatTable, set_mass: bool = False) -> None:
        """Ladder insert nodes = mass-weighted means of the cells at their distance from the edge (each
        cell shared by radial overlap; the views together hold the field's mass and energy)."""
        if self.view_nodes.size == 0:
            return
        wm = self.view_w * self.mesh.m[self.view_cell]
        for node in self.view_nodes:
            sel = self.view_node == node
            mass = math.fsum(wm[sel].tolist())
            if set_mass:
                ladder.m[node] = mass
            if mass > 0.0:
                ladder.h[node] = math.fsum((wm[sel] * self.sys.h[self.view_cell[sel]]).tolist()) / mass
                ladder.T[node] = float(K.temperature(mat_tool, float(ladder.h[node])))

    # ------------------------------------------------------------------ sources
    def _strip_weights(self, faces: np.ndarray, r_len: np.ndarray, y0: np.ndarray, y1: np.ndarray,
                       along_x: bool = False) -> np.ndarray:
        """[nE, nfaces] overlap areas of strips [0, r_len] x [y0, y1] with the faces (side 0), or of
        strips [x0, x1] x [0, r_len] (side 1: minor edge on the end face, ``along_x``)."""
        r0 = self.face_r0[faces][None, :]
        r1 = self.face_r1[faces][None, :]
        ya = self.y_edges[self.face_slab[faces]][None, :]
        yb = self.y_edges[self.face_slab[faces] + 1][None, :]
        if not along_x:
            return _overlap(r0, r1, 0.0, r_len[:, None]) * _overlap(ya, yb, y0[:, None], y1[:, None])
        return _overlap(r0, r1, y0[:, None], y1[:, None]) * _overlap(ya, yb, 0.0, r_len[:, None])

    def deposit_strips(self, rake_J, flank_J, l_c, l_f, lo, hi, side) -> float:
        """Uniform fluxes over rake strips [0, l_c] and flank strips [0, l_f] from the edge, over
        [lo, hi] along the edge (side 0) or along x on the end face (side 1, minor edge, whose flank
        heat enters the same corner cells); energy into the face cells by overlap, renormalised so
        every joule is placed (a range outside the prism is moved to its nearest end). Returns the
        energy placed."""
        rake_J = np.atleast_1d(np.asarray(rake_J, dtype=float))
        flank_J = np.atleast_1d(np.asarray(flank_J, dtype=float))
        if rake_J.size == 0:
            return 0.0
        l_c = np.atleast_1d(np.asarray(l_c, dtype=float))
        l_f = np.atleast_1d(np.asarray(l_f, dtype=float))
        lo = np.atleast_1d(np.asarray(lo, dtype=float))
        hi = np.atleast_1d(np.asarray(hi, dtype=float))
        side = np.atleast_1d(np.asarray(side))
        src = self.sys.src
        rf = np.flatnonzero(self.face_kind == FACE_RAKE)
        ff = np.flatnonzero(self.face_kind == FACE_FLANK)
        tiny = 1e-3 * float(self.section.r_edges[1])
        placed = 0.0
        for s in (0, 1):
            sel = side == s
            if not np.any(sel):
                continue
            if s == 0:
                a, b = _clip_range(lo[sel], hi[sel], float(self.y_edges[0]), float(self.y_edges[-1]))
                targets = ((rf, rake_J[sel], l_c[sel], False), (ff, flank_J[sel], l_f[sel], False))
            else:
                a, b = _clip_range(lo[sel], hi[sel], 0.0, float(self.face_r1[rf].max()))
                targets = ((rf, rake_J[sel], l_c[sel], True), (rf, flank_J[sel], l_f[sel], True))
            for faces, energy, length, along_x in targets:
                if faces.size == 0 or not np.any(energy != 0.0):
                    continue
                w = self._strip_weights(faces, np.maximum(length, tiny), a, b, along_x)
                tot = w.sum(axis=1)
                share = w / np.where(tot > 0.0, tot, 1.0)[:, None]
                per_face = (energy[:, None] * share).sum(axis=0)
                missed = float(energy[tot <= 0.0].sum())
                if missed:
                    per_face[int(np.argmin(self.face_r0[faces] + self.face_slab[faces]))] += missed
                np.add.at(src, self.mesh.face_cell[faces], per_face)
                self._face_energy[faces] += per_face
                hot = energy > 0.0
                cover = w[hot].sum(axis=0) / np.maximum(self.mesh.face_area[faces] / self.mult, 1e-300)
                self.face_cover[faces] = np.minimum(self.face_cover[faces] + cover, 1.0)
                placed += float(per_face.sum())
        return placed

    # ------------------------------------------------------------------ environment
    def set_environment(self, env, emissivity: float) -> None:
        """Film coefficients and areas of the exposed faces for the coming step."""
        sys = self.sys
        exposed = self.face_exposed != 0
        t_face = self.face_T
        if env.in_spindle and env.omega_rad_s > 0.0:
            h_conv = cv.h_rotating_cylinder(env.omega_rad_s, self.ref_diameter_m, t_face, env.T_air_K)
        else:
            h_conv = cv.h_horizontal_cylinder(t_face, env.T_air_K, self.ref_diameter_m)
        h_rad = np.asarray(cv.h_radiation(emissivity, t_face, env.T_air_K), dtype=float)
        h_air = np.asarray(h_conv, dtype=float) + h_rad
        open_area = self.mesh.face_area * (1.0 - self.face_cover)
        reach = min(max(env.reach, 0.0), 1.0) if (env.coolant_on and env.h_coolant_W_m2K > 0.0) else 0.0
        sys.fa_h[:] = np.where(exposed, h_air, 0.0)
        sys.fa_w[:] = np.where(exposed, open_area * (1.0 - reach), 0.0)
        sys.fc_h[:] = np.where(exposed, env.h_coolant_W_m2K if reach > 0.0 else 0.0, 0.0)
        sys.fc_w[:] = np.where(exposed, open_area * reach, 0.0)

    # ------------------------------------------------------------------ step
    def step(self, ladder: Ladder, mat_tool: MatTable, mat_holder: MatTable, dt_s: float) -> np.ndarray:
        out = np.zeros(N_FD)
        K.field_step(self.mesh, self.sys, ladder, mat_tool, mat_holder, float(dt_s), out)
        self.iterations = int(out[FD_ITER])
        self.residual = float(out[FD_RES])
        self._update_faces(ladder, mat_tool, mat_holder, dt_s)
        self.sys.src[:] = 0.0
        self.face_cover[:] = 0.0
        if self.kind == "insert":
            self.update_views(ladder, mat_tool)
        return out

    def _update_faces(self, ladder: Ladder, mat_tool: MatTable, mat_holder: MatTable, dt_s: float) -> None:
        """Face temperatures from the cell temperatures and the face's flux (linear half cell)."""
        mesh = self.mesh
        sys = self.sys
        cellT = sys.T[mesh.face_cell]
        k = np.interp(cellT, mat_tool.T_K, mat_tool.k_W_mK)
        area = mesh.face_area
        energy = self._face_energy
        flux_in = energy / np.maximum(area * dt_s, 1e-300) if dt_s > 0.0 else np.zeros(area.size)
        d = mesh.face_d
        t_air, t_cool = float(ladder.scal[0]), float(ladder.scal[1])
        ga = np.where((sys.fa_w > 0) & (sys.fa_h > 0), sys.fa_w / (d / k + 1.0 / np.maximum(sys.fa_h, 1e-300)), 0.0)
        gc = np.where((sys.fc_w > 0) & (sys.fc_h > 0), sys.fc_w / (d / k + 1.0 / np.maximum(sys.fc_h, 1e-300)), 0.0)
        loss = (ga * (cellT - t_air) + gc * (cellT - t_cool)) / np.maximum(area, 1e-300)
        temp = cellT + (flux_in - loss) * d / k
        node = mesh.face_node
        coupled = node >= 0
        if np.any(coupled):
            idx = node[coupled]
            mats = ladder.mat[idx]
            tn = ladder.T[idx]
            kn = np.where(mats == 0, np.interp(tn, mat_tool.T_K, mat_tool.k_W_mK),
                          np.interp(tn, mat_holder.T_K, mat_holder.k_W_mK))
            r_in = d[coupled] / k[coupled]
            r_out = mesh.face_rc[coupled] + mesh.face_lk[coupled] / kn
            temp[coupled] = cellT[coupled] - (cellT[coupled] - tn) * r_in / (r_in + r_out)
        self.face_T = temp
        self._face_energy = np.zeros(area.size)

    # ------------------------------------------------------------------ reading the field
    def strip_means(self, l_c, l_f, lo, hi, side) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rake strip mean, flank strip mean, edge line) face temperatures per strip."""
        l_c = np.asarray(l_c, dtype=float)
        l_f = np.asarray(l_f, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        side = np.asarray(side)
        n = l_c.size
        rake = np.zeros(n)
        flank = np.zeros(n)
        edge = np.zeros(n)
        rf = np.flatnonzero(self.face_kind == FACE_RAKE)
        ff = np.flatnonzero(self.face_kind == FACE_FLANK)
        inner_r = self.section.r_edges[1]
        for s in (0, 1):
            sel = np.flatnonzero(side == s)
            if sel.size == 0:
                continue
            # ranges outside the prism read its nearest end (as deposit_strips places them)
            if s == 0:
                a, b = _clip_range(lo[sel], hi[sel], float(self.y_edges[0]), float(self.y_edges[-1]))
            else:
                a, b = _clip_range(lo[sel], hi[sel], 0.0, float(self.face_r1[rf].max()))
            wr = self._strip_weights(rf, l_c[sel], a, b, s == 1)
            rake[sel] = _wmean(wr, self.face_T[rf])
            if s == 0:
                wf = self._strip_weights(ff, l_f[sel], a, b)
                flank[sel] = _wmean(wf, self.face_T[ff])
                we_r = self._strip_weights(rf, np.full(sel.size, inner_r), a, b)
                we_f = self._strip_weights(ff, np.full(sel.size, inner_r), a, b)
                edge[sel] = 0.5 * (_wmean(we_r, self.face_T[rf]) + _wmean(we_f, self.face_T[ff]))
            else:
                ends = np.flatnonzero((self.face_kind == FACE_END) & (self.face_slab == 0)
                                      & (self.section.sector[self.mesh.face_cell % self.section.area.size] == 0))
                wf = _overlap(self.face_r0[ends][None, :], self.face_r1[ends][None, :], a[:, None], b[:, None])
                flank[sel] = _wmean(wf, self.face_T[ends])
                edge[sel] = np.maximum(rake[sel], flank[sel])
        return rake, flank, edge

    def nearest_solid_point(self, points_E) -> np.ndarray:
        """Points moved onto the nearest point of the field's solid when outside it (a sensor spot that
        overhangs the tool reads the tool's surface): y into the prism, then the rake plane, the flank
        ray and, for inserts, the section polygon."""
        pts = np.atleast_2d(np.asarray(points_E, dtype=float)).copy()
        pts[:, 1] = np.clip(pts[:, 1], self.y_edges[0], self.y_edges[-1])
        e_f = np.array([math.cos(-self.beta), math.sin(-self.beta)])
        nrm, c = _halfplanes(self.poly) if self.poly.shape[0] else (np.zeros((0, 2)), np.zeros(0))
        for p in pts:
            xz = np.array([p[0], min(p[2], 0.0)])
            th = math.atan2(xz[1], xz[0]) if np.any(xz != 0.0) else 0.0
            if th > 0.0:
                th -= 2.0 * math.pi
            if th < -self.beta:
                xz = max(float(xz @ e_f), 0.0) * e_f
            for _ in range(4):
                viol = nrm @ xz - c
                if not np.any(viol > 0.0):
                    break
                kk = int(np.argmax(viol))
                xz = xz - viol[kk] * nrm[kk]
            r = float(np.hypot(*xz))
            r_max = float(self.section.r_edges[-1])
            if r > r_max:
                xz *= r_max / r
            p[0], p[2] = xz
        return pts

    def temperature_at(self, points_E, outside: str = "nan") -> np.ndarray:
        """Temperatures (K) at points (x, y, z) of the edge frame E; outside the field's solid NaN, or with
        ``outside="nearest"`` the temperature of its nearest point (nearest_solid_point).

        Interpolation: linear in log r, theta and y between cell centroids of the polar lattice, the
        boundary values (face temperatures) at the rake face, flank face and prism ends."""
        pts = np.atleast_2d(np.asarray(points_E, dtype=float))
        if outside == "nearest":
            pts = self.nearest_solid_point(pts)
        out = np.full(pts.shape[0], np.nan)
        sec = self.section
        n2 = sec.area.size
        T = self.sys.T
        re = sec.r_edges
        nk = re.size - 1
        nj = sec.th_edges.size - 1
        rc_lat = np.array([(2.0 / 3.0) * (re[k + 1] ** 3 - re[k] ** 3) / (re[k + 1] ** 2 - re[k] ** 2)
                           for k in range(nk)])
        th_c = 0.5 * (sec.th_edges[1:] + sec.th_edges[:-1])
        yc = 0.5 * (self.y_edges[1:] + self.y_edges[:-1])
        rake_T = self._face_lookup(FACE_RAKE)
        flank_T = self._face_lookup(FACE_FLANK)
        end_T = {0: self._end_lookup(0), 1: self._end_lookup(len(yc) - 1)}
        if self.poly.shape[0]:
            nrm, c = _halfplanes(self.poly)
        for i, (x, y, z) in enumerate(pts):
            scale = 1e-9 * max(float(re[-1]), 1e-6)
            if z > scale or y < self.y_edges[0] - scale or y > self.y_edges[-1] + scale:
                continue
            r = math.hypot(x, z)
            th = math.atan2(min(z, 0.0), x) if r > 0.0 else 0.0
            if th > 0.0:
                th -= 2.0 * math.pi
            if th < -self.beta - 1e-9 or r > re[-1] * (1.0 + 1e-9):
                continue
            if self.poly.shape[0] and not _inside(np.array([x, min(z, 0.0)]), nrm, c, scale):
                continue
            th = min(max(th, -self.beta), 0.0)
            k = int(min(max(np.searchsorted(re, r, side="right") - 1, 0), nk - 1))
            j = int(min(max(np.searchsorted(-sec.th_edges, -th, side="right") - 1, 0), nj - 1))
            if sec.index[k, j] < 0:
                continue
            out[i] = self._interp(r, th, y, rc_lat, th_c, yc, rake_T, flank_T, end_T, T, n2)
        return out

    def _face_lookup(self, kind: int) -> np.ndarray:
        """[nk, ny] face temperature of the rake (or flank) face of ring k in slab iy (NaN if none)."""
        nk = self.section.r_edges.size - 1
        ny = self.y_edges.size - 1
        table = np.full((nk, ny), np.nan)
        sel = np.flatnonzero(self.face_kind == kind)
        cells2 = self.mesh.face_cell[sel] % self.section.area.size
        table[self.section.ring[cells2], self.face_slab[sel]] = self.face_T[sel]
        return table

    def _end_lookup(self, slab: int) -> np.ndarray:
        n2 = self.section.area.size
        table = np.full(n2, np.nan)
        sel = np.flatnonzero((self.face_kind == FACE_END) & (self.face_slab == slab))
        table[self.mesh.face_cell[sel] % n2] = self.face_T[sel]
        return table

    def _interp(self, r, th, y, rc_lat, th_c, yc, rake_T, flank_T, end_T, T, n2) -> float:
        sec = self.section
        nk = rc_lat.size
        nj = th_c.size
        ny = yc.size
        # radial bracket (log r between ring centroids; constant inside the first ring centroid)
        if r <= rc_lat[0]:
            ks, wk = (0, 0), (1.0, 0.0)
        elif r >= rc_lat[-1]:
            ks, wk = (nk - 1, nk - 1), (1.0, 0.0)
        else:
            k0 = int(np.searchsorted(rc_lat, r, side="right") - 1)
            t = (math.log(r) - math.log(rc_lat[k0])) / (math.log(rc_lat[k0 + 1]) - math.log(rc_lat[k0]))
            ks, wk = (k0, k0 + 1), (1.0 - t, t)
        # angular bracket: nodes th_c (descending from ~0 to ~-beta) plus the two faces
        nodes = np.concatenate([[0.0], th_c, [-self.beta]])
        a = int(min(max(np.searchsorted(-nodes, -th, side="right") - 1, 0), nodes.size - 2))
        tt = (nodes[a] - th) / (nodes[a] - nodes[a + 1]) if nodes[a] != nodes[a + 1] else 0.0
        js, wj = (a - 1, a), (1.0 - tt, tt)        # node index -1: rake face, nj: flank face
        ynodes = np.concatenate([[self.y_edges[0]], yc, [self.y_edges[-1]]])
        b = int(min(max(np.searchsorted(ynodes, y, side="right") - 1, 0), ynodes.size - 2))
        ty = (y - ynodes[b]) / (ynodes[b + 1] - ynodes[b]) if ynodes[b + 1] > ynodes[b] else 0.0
        ys, wy = (b - 1, b), (1.0 - ty, ty)        # node -1: end at y0, ny: end at y_end
        total = 0.0
        weight = 0.0
        for kk, w1 in zip(ks, wk):
            if w1 == 0.0:
                continue
            for jj, w2 in zip(js, wj):
                if w2 == 0.0:
                    continue
                for iy, w3 in zip(ys, wy):
                    if w3 == 0.0:
                        continue
                    iyc = min(max(iy, 0), ny - 1)
                    if jj < 0:
                        val = rake_T[kk, iyc]
                    elif jj >= nj:
                        val = flank_T[kk, iyc]
                    else:
                        c2 = sec.index[kk, jj]
                        if c2 < 0:
                            continue
                        if iy < 0:
                            val = end_T[0][c2]
                        elif iy >= ny:
                            val = end_T[1][c2]
                        else:
                            val = T[iy * n2 + c2]
                    if not np.isfinite(val):
                        continue
                    w = w1 * w2 * w3
                    total += w * val
                    weight += w
        return total / weight if weight > 0.0 else math.nan

    # ------------------------------------------------------------------ energy
    def energy_J(self) -> float:
        return math.fsum((self.mesh.m * self.sys.h).tolist())

    # ------------------------------------------------------------------ checkpoint
    _MESH_FIELDS = FieldMesh._fields
    _SYS_FIELDS = ("h", "T", "src", "fa_h", "fa_w", "fc_h", "fc_w", "active", "x", "scal")
    _ARRAYS = ("face_kind", "face_slab", "face_r0", "face_r1", "face_T", "face_cover", "face_exposed", "view_of_cell",
               "view_nodes", "view_cell", "view_node", "view_w", "slab_node", "y_edges", "poly", "_face_energy")
    _SECTION = ("r_edges", "th_edges", "index", "ring", "sector", "area", "cx", "cz", "lnk_a", "lnk_b", "lnk_len",
                "lnk_da", "lnk_db", "bnd_cell", "bnd_len", "bnd_d", "bnd_kind", "bnd_r0", "bnd_r1")
    _ELEMENTS = ("elem", "side", "lo", "hi", "in_field")

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = {}
        for name in self._MESH_FIELDS:
            arrays[f"{prefix}mesh.{name}"] = np.array(getattr(self.mesh, name))
        for name in self._SYS_FIELDS:
            arrays[f"{prefix}sys.{name}"] = np.array(getattr(self.sys, name))
        for name in self._ARRAYS:
            arrays[f"{prefix}{name}"] = np.array(getattr(self, name))
        for name in self._SECTION:
            arrays[f"{prefix}section.{name}"] = np.array(getattr(self.section, name))
        if self.elements is not None:
            for name in self._ELEMENTS:
                arrays[f"{prefix}elements.{name}"] = np.array(getattr(self.elements, name))
        for key, value in sorted(self.frames.items()):
            arrays[f"{prefix}frames.{key}"] = np.array(value)
        meta = {"kind": self.kind, "beta": self.beta, "mult": self.mult, "ref_diameter_m": self.ref_diameter_m,
                "rho": self.rho, "iterations": self.iterations, "residual": self.residual,
                "has_elements": self.elements is not None, "frames": sorted(self.frames),
                "bw": int(self.sys.band.shape[1] - 1)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str) -> "ToolField":
        obj = cls()
        get = lambda key: np.ascontiguousarray(arrays[f"{prefix}{key}"]).copy()
        obj.kind = str(meta["kind"])
        obj.beta = float(meta["beta"])
        obj.mult = float(meta["mult"])
        obj.ref_diameter_m = float(meta["ref_diameter_m"])
        obj.rho = float(meta["rho"])
        obj.iterations = int(meta["iterations"])
        obj.residual = float(meta["residual"])
        obj.mesh = FieldMesh(*(get(f"mesh.{name}") for name in cls._MESH_FIELDS))
        sysvals = {name: get(f"sys.{name}") for name in cls._SYS_FIELDS}
        n = sysvals["h"].size
        nf = sysvals["fa_h"].size
        sys = new_field_system(n, nf, sysvals["active"].size, obj.mesh.col.size, int(meta["bw"]))
        for name, value in sysvals.items():
            getattr(sys, name)[:] = value
        obj.sys = sys
        for name in cls._ARRAYS:
            setattr(obj, name, get(name))
        obj.section = Section(beta=obj.beta, **{name: get(f"section.{name}") for name in cls._SECTION})
        if meta["has_elements"]:
            el = {name: get(f"elements.{name}") for name in cls._ELEMENTS}
            obj.elements = FieldElements(el["elem"], el["side"], el["lo"], el["hi"], el["in_field"].astype(bool))
        obj.frames = {key: get(f"frames.{key}") for key in meta["frames"]}
        return obj


def _clip_range(a: np.ndarray, b: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Intervals [a, b] clipped into [lo, hi]; one that falls outside keeps its width (at most the
    whole range) at the nearest end."""
    width = np.minimum(np.maximum(b - a, 0.0), hi - lo)
    a2 = np.clip(a, lo, hi)
    b2 = np.clip(b, lo, hi)
    empty = b2 - a2 <= 0.0
    a2 = np.where(empty & (b <= lo), lo, np.where(empty, hi - width, a2))
    b2 = np.where(empty & (b <= lo), lo + width, np.where(empty, hi, b2))
    return a2, b2


def _wmean(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    tot = weights.sum(axis=1)
    return np.where(tot > 0.0, (weights * values[None, :]).sum(axis=1) / np.where(tot > 0.0, tot, 1.0), np.nan)


def _assemble(n, n_slabs, m, la, lb, area, da, db, face_cell, face_area, face_d, face_node, face_rc, face_lk,
              ladder: Ladder, active: np.ndarray) -> tuple[FieldMesh, FieldSystem]:
    """Strictly lower CSR pattern of the combined system and the slot of every coupling."""
    nn = ladder.m.size
    big = n + nn
    pairs = []                      # (row, col) with row > col
    pairs.append(np.stack([np.maximum(la, lb), np.minimum(la, lb)], axis=1))
    coupled = np.flatnonzero(face_node >= 0)
    if coupled.size:
        pairs.append(np.stack([n + face_node[coupled].astype(np.int64), face_cell[coupled].astype(np.int64)], axis=1))
    lad_pairs = []
    lad_ok = np.zeros(ladder.link_a.size, dtype=bool)
    for li, (a, b) in enumerate(zip(ladder.link_a, ladder.link_b)):
        if active[a] and active[b]:
            lad_ok[li] = True
            lad_pairs.append((n + max(a, b), n + min(a, b)))
    if lad_pairs:
        pairs.append(np.array(lad_pairs, dtype=np.int64))
    allp = np.concatenate(pairs, axis=0).astype(np.int64)
    key = allp[:, 0] * big + allp[:, 1]
    uniq = np.unique(key)
    rows = uniq // big
    cols = uniq % big
    row_ptr = np.zeros(big + 1, dtype=np.int32)
    np.add.at(row_ptr, rows + 1, 1)
    row_ptr = np.cumsum(row_ptr).astype(np.int32)
    col = cols.astype(np.int32)

    def slot_of(r, c):
        keys = np.asarray(r, dtype=np.int64) * big + np.asarray(c, dtype=np.int64)
        return np.searchsorted(uniq, keys).astype(np.int32)

    link_slot = slot_of(np.maximum(la, lb), np.minimum(la, lb))
    face_slot = -np.ones(face_cell.size, dtype=np.int32)
    if coupled.size:
        face_slot[coupled] = slot_of(n + face_node[coupled].astype(np.int64), face_cell[coupled].astype(np.int64))
    lad_slot = -np.ones(ladder.link_a.size, dtype=np.int32)
    for li in np.flatnonzero(lad_ok):
        a, b = int(ladder.link_a[li]), int(ladder.link_b[li])
        lad_slot[li] = slot_of(n + max(a, b), n + min(a, b))
    # diagonal blocks of the preconditioner: every slab (n2 cells, contiguous) and the ladder
    n2 = n // n_slabs
    blk_ptr = np.array(list(range(0, n + 1, n2)) + [big], dtype=np.int32)
    blk_of = np.searchsorted(blk_ptr, np.arange(big), side="right") - 1
    inside = blk_of[rows] == blk_of[cols]
    bw = int(np.max(rows[inside] - cols[inside])) if np.any(inside) else 1
    mesh = FieldMesh(np.ascontiguousarray(m, dtype=float), np.ascontiguousarray(la, dtype=np.int32),
                     np.ascontiguousarray(lb, dtype=np.int32), np.ascontiguousarray(area, dtype=float),
                     np.ascontiguousarray(da, dtype=float), np.ascontiguousarray(db, dtype=float),
                     np.ascontiguousarray(face_cell, dtype=np.int32), np.ascontiguousarray(face_area, dtype=float),
                     np.ascontiguousarray(face_d, dtype=float), np.ascontiguousarray(face_node, dtype=np.int32),
                     np.ascontiguousarray(face_rc, dtype=float), np.ascontiguousarray(face_lk, dtype=float), row_ptr,
                     col, link_slot, face_slot, lad_slot, blk_ptr)
    sys = new_field_system(n, face_cell.size, nn, col.size, max(bw, 1))
    sys.active[:] = active
    return mesh, sys


__all__ = ["ToolField", "Section", "FieldElements", "build_section", "insert_polygon", "ring_edges", "slab_edges",
           "unroll_elements", "FACE_RAKE", "FACE_FLANK", "FACE_END", "FACE_BACK", "FACE_SEAT", "FACE_ROOT"]
