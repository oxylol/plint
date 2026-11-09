"""Kernel form of a tool's cutting edges (docs/models/cutting.md "Elements"): tooling's effective EdgeModel
and element frames -> :class:`cnctwin.cutting.abi.CutTool`, and the table rows of the tool.

Per element (flat index e = tooth * n_elem + elem, T at spindle angle 0): midpoint p (runout and tilt
included; wear, growth, pull-out and slip are already in the effective edge, architecture 4.1), the
chip-thickness normal n (into the material), the cutting direction c, the force directions on the
tool d_t = -c, d_r = -n, d_a = a = -(t_edge + sin(i) c) / cos(i) (cnctwin.mechanics' (t, r, a)
frame with tooling's profile tangent t_edge and inclination i), the moment arms about the station's
axis point, chip width db and edge length dS, the table row, the wear-land forces (Smithey et al.
2001), the process-damping constants and the chip flow direction (Stabler's rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.abi import EDGE_CHISEL, EDGE_INSERT, EDGE_LIP, EDGE_MARGIN, EDGE_NOSE, EDGE_PERIPHERAL, EdgeModel
from .abi import (N_PK, N_PKI, N_TC, TC_TANGENT_VN, TC_FLOAT_K, TC_FLOAT_TRAVEL, TC_H_CAP, TC_HAND, TC_HOLE_MAKING, TC_N_ELEM,
                  TC_N_TEETH, TC_PD_MAX_LEN, TC_POLAR_SIDE, TC_R_CUT, TC_RECOVERY_RATIO, TC_RECUT_ROW, TC_ROTATING,
                  TC_SURF_EVENT_RAD, TC_TURN_COLWELL, TC_Z_HI, TC_Z_LO, CutTool, with_packed)
from .coefficients import RowGeometry, Rows, flank_stress

HOLE_MAKING_TYPES = ("twist_drill", "indexable_drill", "gun_drill", "spot_drill", "centre_drill", "reamer",
                     "tap_cutting", "tap_forming")
CUTTING_KINDS = (EDGE_PERIPHERAL, EDGE_INSERT, EDGE_LIP, EDGE_NOSE)


@dataclass(frozen=True)
class ToolOptions:
    """Numerical settings and model constants the tool layout carries (from physics.cutting and
    db/models/cutting-contact.yaml)."""

    tangent_vn: float = 0.0872
    turn_colwell: bool = False
    surface_event_rad: float = math.radians(10.0)
    h_cap_m: float = 4e-3
    recovery_ratio: float = 1.0
    pd_max_flank_m: float = 5e-4
    pd_edge_land_per_radius: float = 1.0


def flat(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a).reshape(-1))


def element_positions(edge: EdgeModel) -> np.ndarray:
    """[nt, ne, 3] element midpoints in T at spindle angle 0 with runout and tilt (tooling convention)."""
    h = float(edge.hand_sign[0])
    alpha = h * (np.asarray(edge.tooth_phase_rad)[:, None] + np.asarray(edge.elem_lag_rad))
    r = np.asarray(edge.elem_r_m)
    z = np.asarray(edge.elem_z_m)
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    ax, ay = float(edge.tilt_rad[0]), float(edge.tilt_rad[1])
    px = x + ay * z + float(edge.runout_T_m[0])
    py = y - ax * z + float(edge.runout_T_m[1])
    pz = z + ax * y - ay * x + float(edge.runout_T_m[2])
    return np.stack([px, py, pz], axis=-1)


def row_plan(geometry, edge: EdgeModel, omega_rad_s: float, v_min_m_s: float) -> tuple[list[RowGeometry], np.ndarray]:
    """Table rows of a tool and the row of every element [nt, ne]: one row per coefficient group of
    tooling's GroupTable (in group order, so rows 0..ng-1 match the bus ProcessCoefficients), then a
    low-speed (indentation) variant of every group with elements whose speed |omega| r is below v_min
    (rotating tools; drill centres)."""
    groups = geometry.groups
    ng = int(np.asarray(groups.kind).shape[0])
    elem_group = np.asarray(edge.elem_group, np.int64)
    valid = np.asarray(edge.elem_valid) != 0
    radius = np.abs(np.asarray(edge.elem_r_m, float))
    edge_r = np.asarray(edge.elem_edge_radius_m, float)
    lobes = int(np.asarray(edge.tooth_phase_rad).shape[0])
    diameter = float(getattr(geometry.props, "diameter_m", 0.0) or 0.0)
    rows: list[RowGeometry] = []
    for g in range(ng):
        members = valid & (elem_group == g)
        er = float(np.mean(edge_r[members])) if members.any() else float(groups.edge_radius_m[g])
        rr = float(np.mean(radius[members])) if members.any() else 0.0
        rows.append(RowGeometry(kind=int(groups.kind[g]), rake_rad=float(groups.rake_rad[g]),
                                incl_rad=float(groups.incl_rad[g]), clear_rad=float(groups.clear_rad[g]),
                                edge_radius_m=er, forming=bool(groups.forming[g]), radius_m=rr, lobes=lobes,
                                diameter_m=diameter))
    elem_row = np.where(valid, elem_group, 0).astype(np.int32)
    if geometry.rotating and abs(omega_rad_s) > 0.0 and v_min_m_s > 0.0:
        slow = valid & (np.abs(omega_rad_s) * radius < v_min_m_s)
        for g in range(ng):
            members = slow & (elem_group == g)
            if not members.any() or rows[g].forming or rows[g].kind in (EDGE_CHISEL, EDGE_MARGIN):
                continue
            base = rows[g]
            rows.append(RowGeometry(kind=base.kind, rake_rad=base.rake_rad, incl_rad=base.incl_rad,
                                    clear_rad=base.clear_rad, edge_radius_m=base.edge_radius_m, low_speed=True,
                                    radius_m=float(np.mean(radius[members])), lobes=base.lobes,
                                    diameter_m=base.diameter_m))
            elem_row[members] = len(rows) - 1
    return rows, elem_row


def _holder_float(geometry) -> tuple[float, float]:
    """Axial float stiffness (N/m) and travel (m) of a floating or synchro tapping holder (0, 0: rigid)."""
    holder = getattr(geometry, "holder", None)
    if holder is None:
        return 0.0, 0.0
    k = getattr(holder, "axial_stiffness_N_per_m", None)
    travel = getattr(holder, "axial_float_m", None)
    raw = getattr(holder, "raw", None)
    if (k is None or travel is None) and isinstance(raw, dict):
        if k is None and raw.get("axial_stiffness_N_um") is not None:
            k = float(raw["axial_stiffness_N_um"]) * 1e6
        if travel is None and raw.get("axial_float_mm") is not None:
            travel = float(raw["axial_float_mm"]) * 1e-3
    if k is None or travel is None:
        return 0.0, 0.0
    return float(k), float(travel)


def build_cut_tool(geometry, edge: EdgeModel, frames, rows: list[Rows], elem_row: np.ndarray, VB_m: np.ndarray | None,
                   options: ToolOptions, float_holder: tuple[float, float] | None = None) -> CutTool:
    """CutTool of one tool slot (see the module doc). ``rows[r]`` are the table rows' coefficients (for
    the flank and process-damping constants), ``elem_row`` the row of each element [nt, ne]."""
    from ..mechanics.oblique import chip_flow_direction
    nt, ne = np.asarray(edge.elem_r_m).shape
    n = nt * ne
    P = element_positions(edge).reshape(n, 3)
    if not geometry.rotating:
        P = np.stack([flat(edge.elem_r_m), np.zeros(n), flat(edge.elem_z_m)], axis=-1)
    normal = np.asarray(frames.normal, float).reshape(n, 3)
    cut = np.asarray(frames.cut_dir, float).reshape(n, 3)
    tangent = np.asarray(frames.tangent, float).reshape(n, 3)
    incl = flat(edge.elem_incl_rad).astype(float)
    rake = flat(edge.elem_rake_rad).astype(float)
    clear = flat(edge.elem_clear_rad).astype(float)
    ci = np.cos(incl)
    si = np.sin(incl)
    a_dir = -(tangent + si[:, None] * cut) / np.where(np.abs(ci) > 1e-9, ci, 1e-9)[:, None]
    a_norm = np.linalg.norm(a_dir, axis=1)
    a_dir = a_dir / np.where(a_norm > 0.0, a_norm, 1.0)[:, None]
    dirs = np.stack([-cut, -normal, a_dir], axis=1)
    station = flat(edge.elem_station).astype(np.int32)
    st_z = np.ascontiguousarray(np.asarray(edge.station_z_m, float))
    nst = st_z.shape[0]
    z_of = st_z[np.clip(station, 0, max(nst - 1, 0))] if nst else np.zeros(n)
    pr = P.copy()
    pr[:, 2] -= z_of
    arms = np.cross(pr[:, None, :], dirs)
    valid = flat(edge.elem_valid).astype(np.int8)
    row = flat(elem_row).astype(np.int32)
    kind = flat(edge.elem_kind).astype(np.int32)
    db = flat(edge.elem_db_m).astype(float)
    dS = flat(edge.elem_len_m).astype(float)
    er = flat(edge.elem_edge_radius_m).astype(float)
    vb = np.zeros(n) if VB_m is None or np.asarray(VB_m).size != n else flat(VB_m).astype(float)
    forming = np.array([1 if rows[r].forming else 0 for r in row], np.int8) if rows else np.zeros(n, np.int8)
    flank = np.zeros((n, 2))
    sig_flank = np.zeros(n)
    pd = np.zeros((n, 4))
    chip = np.zeros((n, 3))
    for r, rr in enumerate(rows):
        members = row == r
        if not members.any():
            continue
        sigma, tau = flank_stress(rr, vb[members])
        flank[members, 0] = tau * vb[members] * dS[members]
        flank[members, 1] = sigma * vb[members] * dS[members]
        sig_flank[members] = np.where(vb[members] > 0.0, sigma, rr.flank_sigma0_Pa)
        pd[members, 0] = rr.K_sp_N_per_m3 * dS[members]
        pd[members, 1] = vb[members] + options.pd_edge_land_per_radius * er[members]
        pd[members, 2] = np.tan(np.clip(clear[members], 0.0, 1.4))
        pd[members, 3] = rr.flank_tau0_Pa / rr.flank_sigma0_Pa if rr.flank_sigma0_Pa > 0.0 else rr.mu_eff
    for e in range(n):
        v = chip_flow_direction(float(rake[e]), float(incl[e]), float(incl[e]))
        # (t, r, a) components -> T: t = c, r = -n (away from the machined surface), a
        chip[e] = v[0] * cut[e] - v[1] * normal[e] + v[2] * a_dir[e]
    alpha0 = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2.0 * math.pi) if geometry.rotating else np.zeros(n)
    r_xy = np.hypot(P[:, 0], P[:, 1]) if geometry.rotating else np.abs(flat(edge.elem_r_m))
    tooth = np.repeat(np.arange(nt, dtype=np.int32), ne)
    elem = np.tile(np.arange(ne, dtype=np.int32), nt)
    order = [e for e in np.lexsort((tooth, station)) if valid[e] != 0]
    st_elem = np.array(order, np.int32) if order else np.zeros(1, np.int32)
    st_ptr = np.zeros(nst + 1, np.int32)
    for e in order:
        st_ptr[station[e] + 1] += 1
    st_ptr = np.cumsum(st_ptr).astype(np.int32)
    sections = sorted(getattr(geometry.props, "sections", []) or [], key=lambda s: s.z_m)
    sec_z = np.array([s.z_m for s in sections], float) if sections else np.array([np.inf])
    sec_zb = np.array([s.Z_bending_m3 for s in sections], float) if sections else np.zeros(1)
    sec_zp = np.array([s.Z_polar_m3 for s in sections], float) if sections else np.zeros(1)
    st_sec = np.searchsorted(sec_z, st_z, side="right").astype(np.int32) if nst else np.zeros(1, np.int32)
    hole = geometry.type in HOLE_MAKING_TYPES
    st_centre = np.zeros(max(nst, 1), np.int8)
    if hole and nst:
        chisel = [station[e] for e in order if kind[e] == EDGE_CHISEL]
        if chisel:
            st_centre[np.unique(chisel)] = 1
        else:
            cutting = [e for e in order if kind[e] in CUTTING_KINDS]
            if cutting:
                inner = min(cutting, key=lambda e: r_xy[e])
                st_centre[station[inner]] = 1
    scal = np.zeros(N_TC)
    scal[TC_N_TEETH] = nt
    scal[TC_ROTATING] = 1.0 if geometry.rotating else 0.0
    scal[TC_HOLE_MAKING] = 1.0 if hole else 0.0
    scal[TC_HAND] = float(edge.hand_sign[0])
    k_float, travel = float_holder if float_holder is not None else _holder_float(geometry)
    scal[TC_FLOAT_K] = k_float
    scal[TC_FLOAT_TRAVEL] = travel
    cutting_rows = [r for r in range(len(rows)) if rows[r].h_min_m < np.inf and not rows[r].forming
                    and not rows[r].rubbing]
    counts = {r: int(np.sum((row == r) & (valid != 0))) for r in cutting_rows}
    scal[TC_RECUT_ROW] = float(max(counts, key=lambda r: (counts[r], -r))) if counts else 0.0
    vmask = valid != 0
    scal[TC_R_CUT] = float(r_xy[vmask].max()) if vmask.any() else 0.0
    scal[TC_Z_LO] = float(P[vmask, 2].min()) if vmask.any() else 0.0
    scal[TC_Z_HI] = float(P[vmask, 2].max()) if vmask.any() else 0.0
    scal[TC_TANGENT_VN] = options.tangent_vn
    scal[TC_TURN_COLWELL] = 1.0 if options.turn_colwell else 0.0
    scal[TC_SURF_EVENT_RAD] = options.surface_event_rad
    scal[TC_H_CAP] = options.h_cap_m
    scal[TC_POLAR_SIDE] = 0.0 if geometry.rotating else 1.0
    scal[TC_RECOVERY_RATIO] = options.recovery_ratio
    scal[TC_PD_MAX_LEN] = options.pd_max_flank_m
    scal[TC_N_ELEM] = ne
    tooth_a0 = (np.mod(float(edge.hand_sign[0]) * np.asarray(edge.tooth_phase_rad, float), 2.0 * math.pi)
                if geometry.rotating else np.zeros(nt))
    c = np.ascontiguousarray
    read_row, write_row = helix_rows(geometry, edge, P, normal, dS, station, valid, tooth, nt, ne, nst)
    tool = CutTool(c(P), c(normal), c(cut), c(dirs), c(arms), c(alpha0), c(r_xy), c(db), c(dS), c(row), c(tooth),
                   c(elem), c(station), c(read_row), c(write_row), c(kind), c(forming), c(valid), c(flank), c(sig_flank), c(pd),
                   c(er), c(vb),
                   c(chip), c(rake), c(incl), c(st_z if nst else np.zeros(1)), c(st_ptr), c(st_elem), c(st_sec),
                   c(st_centre), c(sec_z), c(sec_zb), c(sec_zp), c(tooth_a0), c(scal), np.zeros((1, N_PK)),
                   np.zeros((1, N_PKI), np.int64))
    return with_packed(tool)


def _meridian_normal(n_T: np.ndarray, p_T: np.ndarray) -> tuple[float, float]:
    """Unit (radial, axial) components of an element normal in its meridian plane."""
    r = math.hypot(float(p_T[0]), float(p_T[1]))
    nr = (float(n_T[0]) * float(p_T[0]) + float(n_T[1]) * float(p_T[1])) / r if r > 0.0 else 0.0
    nz = float(n_T[2])
    nm = math.hypot(nr, nz)
    return (nr / nm, nz / nm) if nm > 0.0 else (0.0, 0.0)


def _profile_hit(r0: float, z0: float, n0: tuple[float, float], rc: np.ndarray, zc: np.ndarray, nc: list,
                 dSc: np.ndarray) -> int:
    """Which element of a profile polyline (meridian points (rc, zc) in profile order, each covering its edge
    length dSc) the chip-thickness line through (r0, z0) along n0 crosses: index of the polyline vertex
    nearest to the crossing closest to the point, -1 when the line misses the profile (the predecessor's
    land does not reach there)."""
    m = rc.shape[0]
    pts = np.stack([rc, zc], axis=-1)
    segs = []
    for k in range(m - 1):
        segs.append((pts[k], pts[k + 1], k, k + 1))
    for k, nb_k in ((0, 1), (m - 1, m - 2)):
        tan = np.array([nc[k][1], -nc[k][0]])                     # meridian edge tangent of the end element
        if m > 1 and float(np.dot(tan, pts[k] - pts[nb_k])) < 0.0:
            tan = -tan
        segs.append((pts[k], pts[k] + 0.5 * float(dSc[k]) * tan, k, k))
        if m == 1:
            segs.append((pts[k], pts[k] - 0.5 * float(dSc[k]) * tan, k, k))
            break
    p = np.array([r0, z0])
    d = -np.array(n0)
    best_t = math.inf
    best_k = -1
    for a, b, ka, kb in segs:
        u_vec = b - a
        den = d[0] * (-u_vec[1]) + d[1] * u_vec[0]                  # det [d, -u]
        if abs(den) < 1e-18:
            continue
        rhs = a - p
        t = (rhs[0] * (-u_vec[1]) + rhs[1] * u_vec[0]) / den
        u = (d[0] * rhs[1] - d[1] * rhs[0]) / den
        if not (-1e-12 <= u <= 1.0 + 1e-12):
            continue
        if abs(t) < abs(best_t):
            best_t = t
            best_k = ka if u <= 0.5 else kb
    return best_k


def helix_rows(geometry, edge: EdgeModel, P: np.ndarray, normal: np.ndarray, dS: np.ndarray, station: np.ndarray,
               valid: np.ndarray, tooth: np.ndarray, nt: int, ne: int, nst: int) -> tuple[np.ndarray, np.ndarray]:
    """Memory rows each element reads and writes (docs/models/cutting.md "Surface memory").

    An element follows its *predecessor*: the element that passed the same T0 angle just before it at the same
    S height. For tools that do not advance a lead per revolution it is the element of the same station on the
    tooth that passed just before; the element reads and writes its station row. A tap advances one lead per
    revolution: the tooth that passed the angle g / omega earlier (smallest positive cutting-sense gap g) was
    lead g / 2 pi higher. Its element at the same S height is the one whose lifted profile the
    chip-thickness line of the element crosses (actual positions, chamfer clipping included). For the lands
    after the first that is the element of the same station; for the first land it is the last land's
    element one thread row lower, which passes the angle at the same moment as the first land's element of
    its own station, which writes the station row: it writes a handover row (nst + its station) instead,
    which only its first-land successors read. An element whose chip line misses the predecessor's profile
    (below the tap's first teeth, or where the chamfer clipped the predecessor's land) reads the field:
    read row -1 (the kernel takes the refill plane of its station, uncached)."""
    read = station.astype(np.int32).copy()
    write = station.astype(np.int32).copy()
    info = getattr(geometry, "info", {}) or {}
    lead = info.get("lead") if isinstance(info, dict) else None
    if not lead or getattr(geometry, "element_rows", None) is None or not geometry.rotating or nt < 2:
        return read, write
    n = nt * ne
    h = float(edge.hand_sign[0])
    two_pi = 2.0 * math.pi
    psi = np.mod(h * np.arctan2(P[:, 1], P[:, 0]), two_pi)             # cutting-sense angle of each element
    # tooth reference: right-handed angle h phase (as tooth_a0), cutting-sense angle h (h phase) = phase
    psi_tooth = np.mod(np.asarray(edge.tooth_phase_rad, float), two_pi)
    r_e = np.hypot(P[:, 0], P[:, 1])
    mer = [_meridian_normal(normal[e], P[e]) for e in range(n)]
    same_station_pred = np.zeros(n, bool)
    handover: dict[int, list[int]] = {}
    for e in range(n):
        if valid[e] == 0 or not 0 <= int(station[e]) < nst:
            continue
        j = int(tooth[e])
        gaps = [((psi_tooth[k] - psi_tooth[j]) % two_pi, k) for k in range(nt) if k != j]
        gaps = [g for g in gaps if g[0] > 0.0]
        if not gaps:
            continue
        jt = min(gaps)[1]
        cand = np.array([jt * ne + kk for kk in range(ne) if valid[jt * ne + kk] != 0], np.int64)
        if cand.size == 0:
            read[e] = -1
            continue
        gap_c = np.mod(psi[cand] - psi[e], two_pi)
        zc = P[cand, 2] + lead * gap_c / two_pi                    # S height when it passed this angle
        k_hit = _profile_hit(r_e[e], P[e, 2], mer[e], r_e[cand], zc, [mer[c_] for c_ in cand], dS[cand])
        if k_hit < 0:
            read[e] = -1
            continue
        pe = int(cand[k_hit])
        if int(station[pe]) == int(station[e]):
            same_station_pred[pe] = True
        else:
            handover.setdefault(pe, []).append(e)
    for pe, readers in handover.items():
        if same_station_pred[pe]:
            # it also hands its surface on within its own station (not a tap layout): the readers take that row
            for e in readers:
                read[e] = int(station[pe])
            continue
        write[pe] = nst + int(station[pe])
        for e in readers:
            read[e] = nst + int(station[pe])
    return read, write


__all__ = ["ToolOptions", "row_plan", "build_cut_tool", "element_positions", "helix_rows",
           "HOLE_MAKING_TYPES"]
