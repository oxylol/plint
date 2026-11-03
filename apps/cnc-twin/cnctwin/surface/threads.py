"""Thread profile accuracy of turned, milled and tapped threads (docs/models/surface.md section 8).

The surface events of a threading tool are the generated flank points. About the thread axis (S z for
turning; the tool axis through the mean tool position for tapping and thread milling, whose orbit is centred
on the hole) each point has (rho, z, phi); unwrapping the helix, zeta = z - s L phi / 2 pi folded into one
pitch, gives the axial profile (s = +-1 is the hand, chosen as the unwrapping with the smaller scatter; the
lead L is refined by minimising the flank-line scatter within +-1 % of the nominal). On the folded profile:

* flank lines rho = c0 + c1 zeta fitted to each flank (points whose normals are within 10 deg of radial are
  crest and root, excluded): half-angles atan(1 / |c1|), included angle, the flank-angle error;
* pitch and lead (the refined L) and the pitch error;
* major and minor diameters (medians of the crest and root points);
* the pitch diameter by the three-wire method on the simulated profile: a wire of the best size
  d_w = P / (2 cos(alpha / 2)) rests on both flanks of a groove (cnctwin.surface.kernels.wire_centre on the
  binned profile, the wire's axial position chosen where it sits deepest), M = 2 (rho_c + d_w / 2) for an
  external thread (M = 2 (rho_c - d_w / 2) inside an internal one) and
  E = M - d_w (1 + 1 / sin(alpha / 2)) + (P / 2) cot(alpha / 2)  (internal: the signs reversed)
  (Machinery's Handbook, Oberg et al. 2016; lead-angle correction neglected: it is below 0.1 um for the
  helix angles of standard metric threads);
* the simple pitch diameter (where the ridge and the groove are equally wide) and the ISO 724 basic pitch
  diameter d - 0.649519 P of the measured major diameter, for reference.

The thread class (6g, 6H ...) is decided by inspect from these values against ISO 965-1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.rng import hash_uniform, stream_key
from . import kernels as K

RADIAL_COS = math.cos(math.radians(10.0))
ISO_D2_FACTOR = 0.649519          # d2 = d - 0.649519 P (iso-724-1993 / iso-68-1-1998: 3 H / 4, H = 0.866025 P)


@dataclass
class ThreadFeature:
    slot: int
    op: int
    pitch_m: float
    lead_m: float
    form_angle_rad: float
    hand: float
    starts: int
    axis_point: np.ndarray
    axis_dir: np.ndarray
    rotating: bool
    pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))   # X (3), n (3), arrival index
    keep_every: int = 1
    seen: int = 0
    tip_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tip_n: int = 0
    block_first: int = -1
    block_last: int = -1
    first_tick: int = -1
    last_tick: int = -1

    def add(self, X: np.ndarray, N: np.ndarray, cap: int) -> None:
        """Keep the points whose counter-based draw (stream surface.thread_thinning, the arrival index) lies
        below 1 / keep_every; when the budget is exceeded keep_every doubles and the kept points are thinned by
        the same rule (no aliasing with the periodic event sampling)."""
        X = np.asarray(X, float).reshape(-1, 3)
        N = np.asarray(N, float).reshape(-1, 3)
        idx = np.arange(self.seen, self.seen + len(X), dtype=np.int64)
        self.seen += len(X)
        u = _draws(idx)
        keep = u < 1.0 / self.keep_every
        rows = np.concatenate([X[keep], N[keep], idx[keep, None].astype(float)], axis=1)
        self.pts = np.concatenate([self.pts, rows]) if len(self.pts) else rows
        while len(self.pts) > cap:
            self.keep_every *= 2
            self.pts = self.pts[_draws(self.pts[:, 6].astype(np.int64)) < 1.0 / self.keep_every].copy()


STREAM_THIN = stream_key("surface.thread_thinning")


def _draws(idx: np.ndarray) -> np.ndarray:
    return np.array([hash_uniform(0, STREAM_THIN, int(i), 0, 0) for i in idx])


def _axis(feature: ThreadFeature) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(feature.axis_dir, dtype=float)
    a = a / np.linalg.norm(a)
    if feature.rotating and feature.tip_n > 0:
        c = feature.tip_sum / feature.tip_n
        return c - (c @ a) * a, a
    return np.asarray(feature.axis_point, dtype=float), a


def cylindrical(X, N, point, a):
    rel = X - point[None, :]
    z = rel @ a
    radial = rel - z[:, None] * a[None, :]
    rho = np.linalg.norm(radial, axis=1)
    rho_hat = radial / np.where(rho > 0.0, rho, 1.0)[:, None]
    e1 = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    phi = np.arctan2(radial @ e2, radial @ e1)
    n_rho = np.einsum("ij,ij->i", N, rho_hat)
    n_z = N @ a
    return rho, z, phi, n_rho, n_z


def fold(z, phi, lead, pitch, sign):
    return np.mod(z - sign * lead * phi / (2.0 * math.pi), pitch)


def _flank_fit(zeta, rho, mask):
    if np.count_nonzero(mask) < 3:
        return None
    A = np.stack([np.ones(np.count_nonzero(mask)), zeta[mask]], axis=1)
    coef, res, *_ = np.linalg.lstsq(A, rho[mask], rcond=None)
    resid = rho[mask] - A @ coef
    return coef, float(np.sqrt(np.mean(resid * resid)))


def _centre_fold(zeta, rho, n_rho, pitch, external):
    """Shift the folded coordinate so the groove centre (the circular mean of the groove-bottom points: radial
    normals on the inner side for an external thread, the outer side for an internal one) sits at P/2."""
    bottom = (np.abs(n_rho) >= RADIAL_COS) & ((rho < np.median(rho)) if external else (rho > np.median(rho)))
    if not bottom.any():
        return zeta
    ang = 2.0 * math.pi * zeta[bottom] / pitch
    centre = math.atan2(float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang)))) / (2.0 * math.pi) * pitch
    return np.mod(zeta - centre + 0.5 * pitch, pitch)


def _scatter(zeta, rho, pitch) -> float:
    bins = 128
    idx = np.minimum((zeta / pitch * bins).astype(int), bins - 1)
    total, count = 0.0, 0
    for b in np.unique(idx):
        r = rho[idx == b]
        if len(r) > 1:
            total += float(np.sum((r - np.median(r)) ** 2))
            count += len(r)
    return total / count if count else math.inf


def measure(feature: ThreadFeature) -> dict | None:
    """Thread measurements of a feature (None when it has too few points)."""
    if len(feature.pts) < 50 or feature.pitch_m <= 0.0:
        return None
    point, a = _axis(feature)
    X, N = feature.pts[:, :3], feature.pts[:, 3:6]
    rho, z, phi, n_rho, n_z = cylindrical(X, N, point, a)
    external = float(np.median(n_rho)) > 0.0
    lead0 = feature.lead_m if feature.lead_m > 0.0 else feature.pitch_m * max(feature.starts, 1)
    pitch = feature.pitch_m
    best_sign = min((1.0, -1.0), key=lambda s: _scatter(fold(z, phi, lead0, pitch, s), rho, pitch))
    flank = np.abs(n_rho) < RADIAL_COS

    def objective(lead):
        zeta = fold(z, phi, lead, lead / max(feature.starts, 1), best_sign)
        return _scatter(zeta, rho, lead / max(feature.starts, 1))

    lo, hi = 0.99 * lead0, 1.01 * lead0
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = objective(c), objective(d)
    for _ in range(40):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = objective(d)
    lead = 0.5 * (lo + hi)
    pitch_meas = lead / max(feature.starts, 1)
    zeta = _centre_fold(fold(z, phi, lead, pitch_meas, best_sign), rho, n_rho, pitch_meas, external)
    # the two flanks: the sign of the normal's axial component
    left = flank & (n_z > 0.0)
    right = flank & (n_z < 0.0)
    fit_l, fit_r = _flank_fit(zeta, rho, left), _flank_fit(zeta, rho, right)
    halves = []
    for fit in (fit_l, fit_r):
        halves.append(math.atan(1.0 / abs(fit[0][1])) if fit is not None and abs(fit[0][1]) > 0.0 else math.nan)
    crest = np.abs(n_rho) >= RADIAL_COS
    outer = crest & (rho > np.median(rho))
    inner = crest & (rho <= np.median(rho))
    major = 2.0 * float(np.median(rho[outer])) if outer.any() else 2.0 * float(np.max(rho))
    minor = 2.0 * float(np.median(rho[inner])) if inner.any() else 2.0 * float(np.min(rho))
    alpha = feature.form_angle_rad
    d_w = pitch_meas / (2.0 * math.cos(0.5 * alpha))
    prof_z, prof_r = binned_profile(zeta, rho, pitch_meas)
    # the wires rest on the simulated points themselves (the highest asperities carry them)
    order = np.argsort(zeta, kind="stable")
    raw_z = np.ascontiguousarray(np.concatenate([zeta[order] - pitch_meas, zeta[order], zeta[order] + pitch_meas]))
    raw_r = np.ascontiguousarray(np.concatenate([rho[order]] * 3))
    rho_c = wire_rest(raw_z, raw_r, pitch_meas, 0.5 * d_w, external)
    if external:
        M = 2.0 * (rho_c + 0.5 * d_w)
        E = M - d_w * (1.0 + 1.0 / math.sin(0.5 * alpha)) + 0.5 * pitch_meas / math.tan(0.5 * alpha)
    else:
        M = 2.0 * (rho_c - 0.5 * d_w)
        E = M + d_w * (1.0 + 1.0 / math.sin(0.5 * alpha)) - 0.5 * pitch_meas / math.tan(0.5 * alpha)
    simple = simple_pitch_diameter(prof_z, prof_r, pitch_meas)
    return {
        "kind": "external" if external else "internal", "slot": feature.slot, "op": feature.op,
        "pitch_nominal_m": feature.pitch_m, "lead_nominal_m": lead0, "form_angle_nominal_rad": alpha,
        "pitch_m": pitch_meas, "lead_m": lead, "pitch_error_m": pitch_meas - feature.pitch_m,
        "flank_half_angles_rad": halves, "included_angle_rad": float(np.nansum(halves)) if halves else math.nan,
        "flank_angle_error_rad": (float(np.nansum(halves)) - alpha) if halves else math.nan,
        "flank_straightness_m": [fit[1] if fit is not None else None for fit in (fit_l, fit_r)],
        "major_diameter_m": major, "minor_diameter_m": minor,
        "wire_diameter_m": d_w, "M_over_wires_m": M, "pitch_diameter_3wire_m": E,
        "simple_pitch_diameter_m": simple, "iso724_pitch_diameter_m": major - ISO_D2_FACTOR * pitch_meas,
        "points": int(len(feature.pts)), "hand": "right" if best_sign * feature.hand >= 0 else "left",
        "first_tick": feature.first_tick, "last_tick": feature.last_tick, "block_first": feature.block_first,
        "block_last": feature.block_last,
    }


def binned_profile(zeta, rho, pitch, bins: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Median profile rho(zeta) over three pitches (the folded pitch repeated), for the wires."""
    idx = np.minimum((zeta / pitch * bins).astype(int), bins - 1)
    med = np.full(bins, np.nan)
    for b in range(bins):
        sel = idx == b
        if sel.any():
            med[b] = float(np.median(rho[sel]))
    centres = (np.arange(bins) + 0.5) / bins * pitch
    ok = ~np.isnan(med)
    med = np.interp(centres, centres[ok], med[ok], period=pitch) if ok.sum() >= 2 else np.full(bins, np.nanmean(rho))
    pz = np.concatenate([centres - pitch, centres, centres + pitch])
    pr = np.concatenate([med, med, med])
    return np.ascontiguousarray(pz), np.ascontiguousarray(pr)


def wire_rest(pz, pr, pitch, radius, external) -> float:
    """Centre radius of a wire settled in the groove centred at P/2: the axial position where it sits deepest
    (external: smallest centre radius; internal: largest), a grid over the middle half of the pitch refined by
    golden-section search (the resting height is V-shaped about the groove centre)."""
    sign = 1.0 if external else -1.0

    def height(zc):
        c = K.wire_centre(pz, pr, float(zc), float(radius), 1 if external else 0)
        return sign * c if math.isfinite(c) else math.inf

    grid = np.linspace(0.25 * pitch, 0.75 * pitch, 201)
    vals = np.array([height(z) for z in grid])
    if not np.isfinite(vals).any():
        return math.nan
    i = int(np.argmin(vals))
    step = grid[1] - grid[0]
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = height(c), height(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = height(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = height(d)
        if hi - lo < 1e-4 * step:
            break
    best = min(fc, fd, float(vals[i]))
    return sign * best


def simple_pitch_diameter(pz, pr, pitch) -> float:
    """Diameter at which the groove (below the level) takes exactly half the pitch."""
    one = (pz >= 0.0) & (pz < pitch)
    r = pr[one]
    lo, hi = float(np.min(r)), float(np.max(r))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        frac = float(np.mean(r < mid))
        if frac > 0.5:
            hi = mid
        else:
            lo = mid
    return lo + hi


def ideal_iso_profile(d_m: float, pitch_m: float, external: bool = True, n_turns: int = 6, per_turn: int = 720,
                      flank_points: int = 60, alpha: float = math.radians(60.0)):
    """Points and normals of an ideal ISO 68-1 basic profile (flats P/8 at the crest, P/4 at the root of the
    external thread), right-hand helix about S z: (X [n, 3], N [n, 3]) for tests and the doc's check."""
    H = pitch_m / (2.0 * math.tan(0.5 * alpha))
    r_maj = 0.5 * d_m
    # basic profile radius over one pitch (external: crest flat P/8 at d/2, root flat P/4 at d/2 - 5H/8)
    r_min = r_maj - 5.0 * H / 8.0
    half_crest = pitch_m / 16.0
    half_root = pitch_m / 8.0
    zs, rs, nzs, nrs = [], [], [], []
    t = math.tan(0.5 * alpha)
    # one pitch from the crest centre at zeta = 0 to zeta = P: crest, falling flank, root, rising flank
    segments = [(-half_crest, half_crest, "crest"), (half_crest, 0.5 * pitch_m - half_root, "down"),
                (0.5 * pitch_m - half_root, 0.5 * pitch_m + half_root, "root"),
                (0.5 * pitch_m + half_root, pitch_m - half_crest, "up")]
    for z0, z1, kind in segments:
        for u in np.linspace(z0, z1, flank_points, endpoint=False):
            if kind == "crest":
                r, nr, nz = r_maj, 1.0, 0.0
            elif kind == "root":
                r, nr, nz = r_min, 1.0, 0.0
            elif kind == "down":
                r = r_maj - (u - half_crest) / t
                nr, nz = math.sin(0.5 * alpha), math.cos(0.5 * alpha)
            else:
                r = r_min + (u - 0.5 * pitch_m - half_root) / t
                nr, nz = math.sin(0.5 * alpha), -math.cos(0.5 * alpha)
            zs.append(u)
            rs.append(r)
            nrs.append(nr)
            nzs.append(nz)
    zs, rs, nrs, nzs = map(np.asarray, (zs, rs, nrs, nzs))
    if not external:
        # internal thread (nut): the same basic profile surface, its material outside: the normal points inward
        nrs = -nrs
        nzs = -nzs
    X, N = [], []
    for k in range(n_turns * per_turn):
        phi = 2.0 * math.pi * k / per_turn
        j = k % len(zs)
        z = zs[j] + pitch_m * phi / (2.0 * math.pi)
        c, s = math.cos(phi), math.sin(phi)
        X.append((rs[j] * c, rs[j] * s, z))
        N.append((nrs[j] * c, nrs[j] * s, nzs[j]))
    return np.array(X), np.array(N)


__all__ = ["ThreadFeature", "measure", "binned_profile", "wire_rest", "simple_pitch_diameter", "ideal_iso_profile",
           "cylindrical", "fold", "ISO_D2_FACTOR"]
