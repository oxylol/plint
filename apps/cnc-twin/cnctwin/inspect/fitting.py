"""Associated features: least-squares fits and minimum-zone / envelope fits (docs/models/inspect.md "Fits").

**Least squares** (ISO 10360-6 Gaussian association; Forbes 1989; Shakarji 1998): geometric distances, solved by
Levenberg-Marquardt from the CAD nominal (a deterministic start), central-difference Jacobians in a local
parametrisation that is re-centred after every step:

* plane: centroid and the smallest singular vector (exact, no iteration); line: centroid, largest singular vector;
* circle (in a given plane): ``|p - c| - R`` over (c, R), started from the algebraic fit (Kasa 1976);
* sphere ``|p - c| - R``; cylinder ``|(p - a) x u| - R`` over the axis point (2), direction (2) and R;
* cone ``(rho - r0 - z tan psi) cos psi`` over the axis point (2), direction (2), the radius r0 at the axis point
  and the half angle psi (the parametrisation of Shakarji 1998: no apex singularity);
* torus ``sqrt((rho - R)^2 + z^2) - r`` over the centre (3), axis (2), R and r.

Standard uncertainties of the parameters follow from ``s^2 (J^T J)^-1`` with ``s^2 = sum r^2 / (n - p)``.

**Minimum zone, inscribed, circumscribed and tangent** (Chebyshev) associations are solved as sequential linear
programs (Chetwynd 1985; Carr & Ferreira 1995): the distances are linearised in a small displacement of the
feature (the small-displacement torsor of Bourdet et al. 1996), the LP (HiGHS) minimises the zone (or maximises
the inscribed / minimises the circumscribed radius) within a trust region, the feature is moved by the solution
and the exact distances are evaluated again; a step that does not improve the exact zone halves the trust region.
The converged solution is the exact minimum zone of the point set to the solver tolerance.

* MZ plane (ISO 12781), MZ line in a plane (ISO 12780), MZ circle (ISO 12181), MZ cylinder (ISO 12180),
  MZ sphere; MIC / MCC circle, cylinder and sphere (ISO 14405 GX / GN, ASME actual mating envelope);
* the constrained-minimax (tangent) plane of ISO 5459 / ASME Y14.5.1 is the outer plane of the minimum zone (for
  a fixed orientation the tangent plane touches the highest point; minimising the largest gap over the
  orientation is minimising the zone width);
* :func:`min_enclosing_circle` is Welzl's construction in the fixed input order (exact).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from ..core.errors import InputError
from ..geom import transform as tf
from .schema import InspectSettings


class FitError(InputError):
    """A fit could not be made (too few points, a degenerate configuration, the solver failed)."""


@dataclass
class Fit:
    kind: str                          # plane line circle sphere cylinder cone torus
    method: str                        # LS | MZ | MIC | MCC | tangent
    params: dict                       # in S, SI (cnctwin.geom.cad parameter names)
    residuals_m: np.ndarray            # signed geometric distance of each point to the fitted surface
    iterations: int = 0
    converged: bool = True
    uncertainty: dict = field(default_factory=dict)   # standard uncertainties of the parameters (LS)
    note: str = ""

    @property
    def zone_m(self) -> float:
        r = self.residuals_m
        return float(r.max() - r.min()) if len(r) else 0.0

    @property
    def rms_m(self) -> float:
        r = self.residuals_m
        return float(np.sqrt(np.mean(r * r))) if len(r) else 0.0

    def to_json(self) -> dict:
        out = {"kind": self.kind, "method": self.method, "points": int(len(self.residuals_m)),
               "zone_m": self.zone_m, "rms_m": self.rms_m, "iterations": self.iterations,
               "converged": self.converged}
        for k, v in self.params.items():
            out[k] = [float(x) for x in v] if isinstance(v, np.ndarray) else float(v) if isinstance(
                v, (float, np.floating)) else v
        if self.uncertainty:
            out["uncertainty"] = {k: float(v) for k, v in self.uncertainty.items()}
        if self.note:
            out["note"] = self.note
        return out


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise FitError("a zero direction")
    return v / n


def _frame(u) -> tuple[np.ndarray, np.ndarray]:
    R = tf.frame_from_z(u)
    return R[:, 0], R[:, 1]


def _need(P, n, what):
    if len(P) < n:
        raise FitError(f"a {what} fit needs at least {n} points, got {len(P)}")


# ============================================================================ least squares

def _lm(residual, x0, scale, iterations, tol):
    """Levenberg-Marquardt with central-difference Jacobians; ``scale`` [p] are the parameter magnitudes."""
    x = np.asarray(x0, dtype=np.float64).copy()
    r = residual(x)
    cost = float(r @ r)
    lam = 1e-3
    it = 0
    converged = False
    h = 1e-6 * np.asarray(scale, dtype=np.float64)
    J = None
    for it in range(1, iterations + 1):
        J = np.empty((len(r), len(x)))
        for k in range(len(x)):
            e = np.zeros(len(x))
            e[k] = h[k]
            J[:, k] = (residual(x + e) - residual(x - e)) / (2.0 * h[k])
        A_ = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(30):
            try:
                step = -np.linalg.solve(A_ + lam * np.diag(np.diag(A_) + 1e-30), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = x + step
            rn = residual(xn)
            cn = float(rn @ rn)
            if cn <= cost:
                x, r, cost = xn, rn, cn
                lam = max(lam / 10.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved or float(np.max(np.abs(step) / np.asarray(scale))) < tol:
            converged = True
            break
    return x, r, J, it, converged


def _uncertainty(J, r, names) -> dict:
    n, p = J.shape
    if n <= p:
        return {}
    s2 = float(r @ r) / (n - p)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return {}
    return {name: math.sqrt(max(cov[k, k], 0.0)) for k, name in enumerate(names)}


def fit_plane(P, normal0) -> Fit:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 3, "plane")
    c = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    if np.dot(n, normal0) < 0.0:
        n = -n
    r = (P - c) @ n
    unc = {}
    if len(P) > 3:
        s2 = float(r @ r) / (len(P) - 3)
        unc = {"offset_m": math.sqrt(s2 / len(P)), "tilt_rad": math.sqrt(s2 / max(s[0] ** 2, 1e-300)) if len(s) else 0.0}
    return Fit("plane", "LS", {"point_m": c, "normal": n}, r, 1, True, unc)


def fit_line(P, direction0) -> Fit:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 2, "line")
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    u = vt[0]
    if np.dot(u, direction0) < 0.0:
        u = -u
    w = P - c
    r = np.linalg.norm(w - np.outer(w @ u, u), axis=1)
    return Fit("line", "LS", {"point_m": c, "direction": u}, r)


def _circle_algebraic(xy):
    A_ = np.column_stack([xy, np.ones(len(xy))])
    b = -(xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A_, b, rcond=None)
    c = -0.5 * sol[:2]
    R = math.sqrt(max(c @ c - sol[2], 0.0))
    return c, R


def fit_circle(P, centre0, normal, settings: InspectSettings) -> Fit:
    """Circle in the plane through centre0 with the given normal (points are projected onto it)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 3, "circle")
    n = _unit(normal)
    e1, e2 = _frame(n)
    o = np.asarray(centre0, dtype=np.float64)
    xy = np.column_stack([(P - o) @ e1, (P - o) @ e2])
    c0, R0 = _circle_algebraic(xy)
    scale = max(R0, 1e-6)

    def res(x):
        return np.hypot(xy[:, 0] - x[0], xy[:, 1] - x[1]) - x[2]

    x, r, J, it, ok = _lm(res, [c0[0], c0[1], R0], [scale] * 3, settings.ls_iterations, settings.ls_step_tolerance)
    centre = o + x[0] * e1 + x[1] * e2 + ((P - o) @ n).mean() * n
    unc = _uncertainty(J, r, ("centre_x_m", "centre_y_m", "radius_m")) if J is not None else {}
    return Fit("circle", "LS", {"centre_m": centre, "normal": n, "radius_m": float(x[2])}, r, it, ok, unc)


def fit_sphere(P, centre0, radius0, settings: InspectSettings) -> Fit:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 4, "sphere")
    o = np.asarray(centre0, dtype=np.float64)
    scale = max(float(radius0), 1e-6)

    def res(x):
        return np.linalg.norm(P - (o + x[:3]), axis=1) - x[3]

    x, r, J, it, ok = _lm(res, [0.0, 0.0, 0.0, radius0], [scale] * 4, settings.ls_iterations,
                          settings.ls_step_tolerance)
    unc = _uncertainty(J, r, ("centre_x_m", "centre_y_m", "centre_z_m", "radius_m")) if J is not None else {}
    return Fit("sphere", "LS", {"centre_m": o + x[:3], "radius_m": float(x[3])}, r, it, ok, unc)


def _axis_from(x, a0, e1, e2, u0):
    a = a0 + x[0] * e1 + x[1] * e2
    u = u0 + x[2] * e1 + x[3] * e2
    return a, u / np.linalg.norm(u)


def _recentre(P, a, u):
    """The axis point nearest to the points' centroid (keeps the parametrisation well conditioned)."""
    c = P.mean(axis=0)
    return a + float((c - a) @ u) * u


def cylinder_distance(P, a, u):
    w = P - a
    return np.linalg.norm(w - np.outer(w @ u, u), axis=1)


def fit_cylinder(P, axis_point0, axis0, radius0, settings: InspectSettings) -> Fit:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 5, "cylinder")
    a, u = _recentre(P, np.asarray(axis_point0, float), _unit(axis0)), _unit(axis0)
    scale = max(float(radius0), 1e-6)
    total = 0
    ok = False
    J = r = None
    for _ in range(4):                  # outer re-centring of the local parametrisation
        e1, e2 = _frame(u)

        def res(x, a=a, u=u, e1=e1, e2=e2):
            aa, uu = _axis_from(x, a, e1, e2, u)
            return cylinder_distance(P, aa, uu) - x[4]

        x, r, J, it, ok = _lm(res, [0, 0, 0, 0, radius0 if total == 0 else radius], [scale, scale, 1, 1, scale],
                              settings.ls_iterations, settings.ls_step_tolerance)
        total += it
        a, u = _axis_from(x, a, e1, e2, u)
        a = _recentre(P, a, u)
        radius = float(x[4])
        if float(np.max(np.abs(x[:4]) / np.array([scale, scale, 1, 1]))) < 1e3 * settings.ls_step_tolerance:
            break
    if np.dot(u, axis0) < 0.0:
        u = -u
    unc = _uncertainty(J, r, ("axis_x_m", "axis_y_m", "tilt_x_rad", "tilt_y_rad", "radius_m")) if J is not None else {}
    return Fit("cylinder", "LS", {"axis_point_m": a, "axis": u, "radius_m": radius}, r, total, ok, unc)


def cone_distance(P, a, u, r0, psi):
    w = P - a
    z = w @ u
    rho = np.linalg.norm(w - np.outer(z, u), axis=1)
    return (rho - r0 - z * math.tan(psi)) * math.cos(psi)


def fit_cone(P, apex0, axis0, semi_angle0, settings: InspectSettings) -> Fit:
    """Cone fit in the (axis point, axis, radius there, half angle) parametrisation; returns geom cone params."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 6, "cone")
    u = _unit(axis0)
    apex = np.asarray(apex0, dtype=np.float64)
    a = _recentre(P, apex, u)
    psi = float(semi_angle0)
    r0 = float((a - apex) @ u) * math.tan(psi)
    scale = max(abs(r0), float(np.ptp(P, axis=0).max()), 1e-6)
    total = 0
    ok = False
    J = r = None
    for _ in range(4):
        e1, e2 = _frame(u)

        def res(x, a=a, u=u, e1=e1, e2=e2):
            aa, uu = _axis_from(x, a, e1, e2, u)
            return cone_distance(P, aa, uu, x[4], x[5])

        x, r, J, it, ok = _lm(res, [0, 0, 0, 0, r0, psi], [scale, scale, 1, 1, scale, 1], settings.ls_iterations,
                              settings.ls_step_tolerance)
        total += it
        a_new, u = _axis_from(x, a, e1, e2, u)
        r0, psi = float(x[4]), float(x[5])
        a = _recentre(P, a_new, u)
        r0 += float((a - a_new) @ u) * math.tan(psi)
        if float(np.max(np.abs(x[:4]) / np.array([scale, scale, 1, 1]))) < 1e3 * settings.ls_step_tolerance:
            break
    if np.dot(u, axis0) < 0.0:          # keep the axis from the apex into the cone
        u, psi, r0 = -u, -psi, r0
    apex = a - (r0 / math.tan(psi)) * u if abs(math.tan(psi)) > 1e-12 else a
    unc = _uncertainty(J, r, ("axis_x_m", "axis_y_m", "tilt_x_rad", "tilt_y_rad", "ref_radius_m",
                              "semi_angle_rad")) if J is not None else {}
    return Fit("cone", "LS", {"apex_m": apex, "axis": u, "semi_angle_rad": psi, "ref_point_m": a, "ref_radius_m": r0},
               r, total, ok, unc)


def torus_distance(P, c, u, R, r):
    w = P - c
    z = w @ u
    rho = np.linalg.norm(w - np.outer(z, u), axis=1)
    return np.hypot(rho - R, z) - r


def fit_torus(P, centre0, axis0, major0, minor0, settings: InspectSettings) -> Fit:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 7, "torus")
    c = np.asarray(centre0, dtype=np.float64)
    u = _unit(axis0)
    scale = max(float(major0), 1e-6)
    e1, e2 = _frame(u)

    def res(x):
        cc = c + x[0] * e1 + x[1] * e2 + x[2] * u
        uu = u + x[3] * e1 + x[4] * e2
        uu = uu / np.linalg.norm(uu)
        return torus_distance(P, cc, uu, x[5], x[6])

    x, r, J, it, ok = _lm(res, [0, 0, 0, 0, 0, major0, minor0], [scale, scale, scale, 1, 1, scale, scale],
                          settings.ls_iterations, settings.ls_step_tolerance)
    cc = c + x[0] * e1 + x[1] * e2 + x[2] * u
    uu = _unit(u + x[3] * e1 + x[4] * e2)
    unc = _uncertainty(J, r, ("centre_x_m", "centre_y_m", "centre_z_m", "tilt_x_rad", "tilt_y_rad",
                              "major_radius_m", "minor_radius_m")) if J is not None else {}
    return Fit("torus", "LS", {"centre_m": cc, "axis": uu, "major_radius_m": float(x[5]),
                               "minor_radius_m": float(x[6])}, r, it, ok, unc)


# ============================================================================ Chebyshev by sequential LP

def lp(c, A_ub, b_ub, bounds):
    """One linear program (HiGHS); raises FitError when it has no optimum."""
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        raise FitError(f"linear program failed: {res.message}")
    return res.x


def _zone_value(rho, mode):
    if mode == "MZ":
        return float(rho.max() - rho.min())
    if mode == "MIC":
        return -float(rho.min())
    return float(rho.max())                 # MCC


def _line_search(f, iterations: int = 48) -> float:
    """argmin of f on [0, 1] by golden-section search (the LP direction overshoots in directions the
    linearisation cannot see, such as the free side of two diametral points), 1 when f decreases to the end."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = 0.0, 1.0
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iterations):
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = f(x2)
    best = x1 if f1 <= f2 else x2
    return 1.0 if f(1.0) <= min(f1, f2) else best


def slp(distance, x0, steps, mode: str, settings: InspectSettings):
    """Sequential LP of a Chebyshev association: ``distance(x) -> rho [n]`` (exact), ``steps`` [p] the trust
    region of each parameter. mode MZ minimises max - min, MIC maximises min, MCC minimises max.
    Returns (x, rho, iterations, converged, at_trust_bound)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    trust = np.asarray(steps, dtype=np.float64).copy()
    rho = distance(x)
    value = _zone_value(rho, mode)
    p = len(x)
    it = 0
    converged = False
    bound_hits = 0
    for it in range(1, settings.lp_iterations + 1):
        h = 1e-7 * np.maximum(trust, 1e-12)
        J = np.empty((len(rho), p))
        for k in range(p):
            e = np.zeros(p)
            e[k] = h[k]
            J[:, k] = (distance(x + e) - distance(x - e)) / (2.0 * h[k])
        bounds = [(-t, t) for t in trust]
        if mode == "MZ":
            # variables [dx, t_lo, t_hi]: min t_hi - t_lo, t_lo <= rho + J dx <= t_hi
            cvec = np.concatenate([np.zeros(p), [-1.0, 1.0]])
            A_ub = np.vstack([np.hstack([J, np.zeros((len(rho), 1)), -np.ones((len(rho), 1))]),
                              np.hstack([-J, np.ones((len(rho), 1)), np.zeros((len(rho), 1))])])
            b_ub = np.concatenate([-rho, rho])
            sol = lp(cvec, A_ub, b_ub, bounds + [(None, None), (None, None)])
        elif mode == "MIC":
            cvec = np.concatenate([np.zeros(p), [-1.0]])
            A_ub = np.hstack([-J, np.ones((len(rho), 1))])
            sol = lp(cvec, A_ub, rho, bounds + [(None, None)])
        else:
            cvec = np.concatenate([np.zeros(p), [1.0]])
            A_ub = np.hstack([J, -np.ones((len(rho), 1))])
            sol = lp(cvec, A_ub, -rho, bounds + [(None, None)])
        dx = sol[:p]
        alpha = _line_search(lambda a: _zone_value(distance(x + a * dx), mode))
        xn = x + alpha * dx
        rn = distance(xn)
        vn = _zone_value(rn, mode)
        if vn <= value + settings.lp_tolerance_m * 1e-3:
            gain = value - vn
            at_bound = alpha > 0.999 and bool(np.any(np.abs(dx) >= 0.999 * trust))
            bound_hits = bound_hits + 1 if at_bound else 0
            x, rho, value = xn, rn, vn
            if gain < settings.lp_tolerance_m and not at_bound:
                converged = True
                break
        else:
            trust *= 0.5
            if float(np.max(trust / np.asarray(steps))) < 1e-9:
                converged = True
                break
    return x, rho, it, converged, bound_hits >= 3


def zone_plane(P, normal0, settings: InspectSettings) -> Fit:
    """Minimum-zone plane pair (ISO 12781 MZPL); params: the middle plane, ``zone_m`` its width, and the outer
    (tangent, ISO 5459 / ASME) plane offset ``outer_offset_m`` along the normal from ``point_m``."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 3, "plane")
    ls = fit_plane(P, normal0)
    c = ls.params["point_m"]
    n0 = ls.params["normal"]
    e1, e2 = _frame(n0)
    size = max(float(np.ptp(P, axis=0).max()), 1e-9)

    def heights(x):
        n = n0 + x[0] * e1 + x[1] * e2
        return (P - c) @ (n / np.linalg.norm(n))

    trust = settings.lp_trust_m / size
    x, h, it, ok, _ = slp(heights, [0.0, 0.0], [trust, trust], "MZ", settings)
    n = n0 + x[0] * e1 + x[1] * e2
    n = n / np.linalg.norm(n)
    mid = 0.5 * (h.max() + h.min())
    return Fit("plane", "MZ", {"point_m": c + mid * n, "normal": n, "outer_offset_m": float(h.max() - mid)},
               h - mid, it, ok)


def tangent_plane(P, normal0, settings: InspectSettings, fixed_axis=None) -> Fit:
    """Datum feature simulator of a plane (ISO 5459 constrained minimax = ASME tangent plane): the outer plane of
    the minimum zone, touching the outermost points; ``fixed_axis`` keeps the normal perpendicular to it (a
    secondary datum perpendicular to the primary: one rotation left)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 1 if fixed_axis is not None else 3, "datum plane")
    n0 = _unit(normal0)
    if fixed_axis is not None:
        a = _unit(fixed_axis)
        n0 = _unit(n0 - (n0 @ a) * a)
        e = np.cross(a, n0)
        size = max(float(np.ptp(P, axis=0).max()), 1e-9)

        def heights(x):
            n = n0 + x[0] * e
            return P @ (n / np.linalg.norm(n))

        if len(P) >= 2:
            x, h, it, ok, _ = slp(heights, [0.0], [settings.lp_trust_m / size], "MZ", settings)
        else:
            x, h, it, ok = np.zeros(1), heights(np.zeros(1)), 0, True
        n = n0 + x[0] * e
        n = n / np.linalg.norm(n)
        top = float(h.max())
        return Fit("plane", "tangent", {"point_m": top * n, "normal": n}, h - top, it, ok)
    mz = zone_plane(P, n0, settings)
    n = mz.params["normal"]
    point = mz.params["point_m"] + mz.params["outer_offset_m"] * n
    return Fit("plane", "tangent", {"point_m": point, "normal": n}, (P - point) @ n, mz.iterations, mz.converged)


def zone_line(points_2d, settings: InspectSettings) -> tuple[float, float, np.ndarray]:
    """Minimum-zone straightness of a profile ``[(s, h)]`` in its plane: (zone, LS zone, MZ residuals)."""
    sh = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if len(sh) < 2:
        raise FitError("a straightness evaluation needs two points")
    s = sh[:, 0] - sh[:, 0].mean()
    h = sh[:, 1] - sh[:, 1].mean()
    slope_ls = float((s @ h) / (s @ s)) if float(s @ s) > 0 else 0.0
    ls = (h - slope_ls * s) / math.sqrt(1.0 + slope_ls ** 2)
    size = max(float(np.ptp(s)), 1e-9)

    def heights(x):
        k = slope_ls + x[0]
        return (h - k * s) / math.sqrt(1.0 + k * k)

    x, r, _, _, _ = slp(heights, [0.0], [settings.lp_trust_m / size], "MZ", settings)
    return float(r.max() - r.min()), float(ls.max() - ls.min()), r - 0.5 * (r.max() + r.min())


def zone_circle(P, centre0, normal, mode: str, settings: InspectSettings) -> Fit:
    """MZ, MIC or MCC circle in a plane (ISO 12181 MZCI / MICI / MCCI)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 3, "circle")
    ls = fit_circle(P, centre0, normal, settings)
    n = ls.params["normal"]
    e1, e2 = _frame(n)
    c0 = ls.params["centre_m"]
    xy = np.column_stack([(P - c0) @ e1, (P - c0) @ e2])
    R = ls.params["radius_m"]

    def radii(x):
        return np.hypot(xy[:, 0] - x[0], xy[:, 1] - x[1])

    x, rho, it, ok, unbounded = slp(radii, [0.0, 0.0], [settings.lp_trust_m] * 2, mode, settings)
    centre = c0 + x[0] * e1 + x[1] * e2
    if mode == "MZ":
        radius, note = 0.5 * float(rho.max() + rho.min()), ""
    elif mode == "MIC":
        radius, note = float(rho.min()), ""
    else:
        radius, note = float(rho.max()), ""
    if unbounded:
        note = "the points do not surround the centre (open arc): the envelope is not bounded"
        ok = False
    return Fit("circle", mode, {"centre_m": centre, "normal": n, "radius_m": radius,
                                "inner_radius_m": float(rho.min()), "outer_radius_m": float(rho.max())},
               rho - radius, it, ok, note=note)


def zone_cylinder(P, axis_point0, axis0, mode: str, settings: InspectSettings, fixed_direction: bool = False) -> Fit:
    """MZ, MIC or MCC cylinder (ISO 12180 MZCY / MICY / MCCY); ``fixed_direction`` keeps the axis direction
    (a datum cylinder perpendicular to a primary plane)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 3 if fixed_direction else 5, "cylinder")
    u0 = _unit(axis0)
    a0 = _recentre(P, np.asarray(axis_point0, float), u0)
    if not fixed_direction:
        ls = fit_cylinder(P, a0, u0, float(np.mean(cylinder_distance(P, a0, u0))), settings)
        a0, u0 = ls.params["axis_point_m"], ls.params["axis"]
    e1, e2 = _frame(u0)
    length = max(float(np.ptp((P - a0) @ u0)), 1e-9)

    def radii(x):
        a = a0 + x[0] * e1 + x[1] * e2
        if fixed_direction:
            return cylinder_distance(P, a, u0)
        u = u0 + x[2] * e1 + x[3] * e2
        return cylinder_distance(P, a, u / np.linalg.norm(u))

    tr = settings.lp_trust_m
    steps = [tr, tr] if fixed_direction else [tr, tr, tr / length, tr / length]
    x, rho, it, ok, unbounded = slp(radii, np.zeros(len(steps)), steps, mode, settings)
    a = a0 + x[0] * e1 + x[1] * e2
    u = u0 if fixed_direction else _unit(u0 + x[2] * e1 + x[3] * e2)
    if np.dot(u, axis0) < 0.0:
        u = -u
    radius = 0.5 * float(rho.max() + rho.min()) if mode == "MZ" else float(rho.min()) if mode == "MIC" \
        else float(rho.max())
    note = "the points do not surround the axis: the envelope is not bounded" if unbounded else ""
    return Fit("cylinder", mode, {"axis_point_m": _recentre(P, a, u), "axis": u, "radius_m": radius,
                                  "inner_radius_m": float(rho.min()), "outer_radius_m": float(rho.max())},
               rho - radius, it, ok and not unbounded, note=note)


def zone_sphere(P, centre0, mode: str, settings: InspectSettings) -> Fit:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    _need(P, 4, "sphere")
    c0 = np.asarray(centre0, dtype=np.float64)
    ls = fit_sphere(P, c0, float(np.mean(np.linalg.norm(P - c0, axis=1))), settings)
    c0 = ls.params["centre_m"]

    def radii(x):
        return np.linalg.norm(P - (c0 + x), axis=1)

    x, rho, it, ok, unbounded = slp(radii, np.zeros(3), [settings.lp_trust_m] * 3, mode, settings)
    radius = 0.5 * float(rho.max() + rho.min()) if mode == "MZ" else float(rho.min()) if mode == "MIC" \
        else float(rho.max())
    return Fit("sphere", mode, {"centre_m": c0 + x, "radius_m": radius, "inner_radius_m": float(rho.min()),
                                "outer_radius_m": float(rho.max())}, rho - radius, it, ok and not unbounded,
               note="the points do not surround the centre" if unbounded else "")


# ============================================================================ zones linear in free displacements

def symmetric_zone(d0, J, settings: InspectSettings | None = None) -> tuple[float, np.ndarray]:
    """min over xi of max |d0 + J xi| (profile with free degrees of freedom); (half zone, xi)."""
    d0 = np.asarray(d0, dtype=np.float64)
    if J is None or J.shape[1] == 0:
        return float(np.max(np.abs(d0))) if len(d0) else 0.0, np.zeros(0)
    p = J.shape[1]
    n = len(d0)
    cvec = np.concatenate([np.zeros(p), [1.0]])
    A_ub = np.vstack([np.hstack([J, -np.ones((n, 1))]), np.hstack([-J, -np.ones((n, 1))])])
    b_ub = np.concatenate([-d0, d0])
    sol = lp(cvec, A_ub, b_ub, [(None, None)] * p + [(0.0, None)])
    xi = sol[:p]
    return float(np.max(np.abs(d0 + J @ xi))), xi


def band_zone(d0, J) -> tuple[float, np.ndarray]:
    """min over xi of (max - min) of d0 + J xi (orientation zones with free rotations); (width, xi)."""
    d0 = np.asarray(d0, dtype=np.float64)
    if J is None or J.shape[1] == 0:
        return float(d0.max() - d0.min()) if len(d0) else 0.0, np.zeros(0)
    p = J.shape[1]
    n = len(d0)
    cvec = np.concatenate([np.zeros(p), [-1.0, 1.0]])
    A_ub = np.vstack([np.hstack([J, np.zeros((n, 1)), -np.ones((n, 1))]),
                      np.hstack([-J, np.ones((n, 1)), np.zeros((n, 1))])])
    b_ub = np.concatenate([-d0, d0])
    sol = lp(cvec, A_ub, b_ub, [(None, None)] * (p + 2))
    xi = sol[:p]
    v = d0 + J @ xi
    return float(v.max() - v.min()), xi


def radial_zone(o0, G, sides: int, settings: InspectSettings | None = None) -> tuple[float, np.ndarray]:
    """min over xi of max_k |o0_k + G_k xi| for 2-D offsets o0 [m, 2] with G [m, 2, p] (a cylindrical or circular
    zone with free degrees of freedom). A linear program on a regular polygon of ``sides`` gives the start (its
    radius is at most 1/cos(pi/sides) too large); the exact minimax is then reached by the sequential LP of the
    circumscribed association on the exact norms. (radius, xi)."""
    o0 = np.asarray(o0, dtype=np.float64).reshape(-1, 2)
    if G is None or G.shape[2] == 0:
        return float(np.max(np.linalg.norm(o0, axis=1))) if len(o0) else 0.0, np.zeros(0)
    p = G.shape[2]
    ang = 2.0 * math.pi * np.arange(sides) / sides
    U = np.column_stack([np.cos(ang), np.sin(ang)])
    rows, rhs = [], []
    for k in range(len(o0)):
        rows.append(np.hstack([U @ G[k], -np.ones((sides, 1))]))
        rhs.append(-(U @ o0[k]))
    cvec = np.concatenate([np.zeros(p), [1.0]])
    sol = lp(cvec, np.vstack(rows), np.concatenate(rhs), [(None, None)] * p + [(0.0, None)])
    xi = sol[:p]

    def norms(x):
        return np.linalg.norm(o0 + np.einsum("kap,p->ka", G, x), axis=1)

    if settings is None:
        from .schema import defaults

        settings = defaults()
    r0 = float(np.max(norms(xi)))
    scale = max(r0, float(np.max(np.abs(o0))) if len(o0) else 0.0, 1e-12)
    gnorm = np.maximum(np.max(np.abs(G), axis=(0, 1)), 1e-300)
    xi, rho, _, _, _ = slp(norms, xi, scale / gnorm, "MCC", settings)
    return float(np.max(rho)), xi


def min_enclosing_circle(xy) -> tuple[np.ndarray, float]:
    """Smallest circle containing 2-D points (Welzl 1991, incremental in the given order): (centre, radius)."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(2), 0.0

    def inside(p, c, r):
        return math.hypot(p[0] - c[0], p[1] - c[1]) <= r * (1.0 + 1e-12) + 1e-15

    def circum(a, b, c):
        d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-300:
            far = max(((a, b), (b, c), (a, c)), key=lambda q: math.dist(q[0], q[1]))
            m = 0.5 * (far[0] + far[1])
            return m, 0.5 * math.dist(far[0], far[1])
        ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
        centre = np.array([ux, uy])
        return centre, math.dist(centre, a)

    c, r = pts[0].copy(), 0.0
    for i in range(1, len(pts)):
        if inside(pts[i], c, r):
            continue
        c, r = pts[i].copy(), 0.0
        for j in range(i):
            if inside(pts[j], c, r):
                continue
            c, r = 0.5 * (pts[i] + pts[j]), 0.5 * math.dist(pts[i], pts[j])
            for k in range(j):
                if not inside(pts[k], c, r):
                    c, r = circum(pts[i], pts[j], pts[k])
    return c, float(r)


__all__ = ["Fit", "FitError", "fit_plane", "fit_line", "fit_circle", "fit_sphere", "fit_cylinder", "fit_cone",
           "fit_torus", "zone_plane", "tangent_plane", "zone_line", "zone_circle", "zone_cylinder", "zone_sphere",
           "symmetric_zone", "band_zone", "radial_zone", "min_enclosing_circle", "slp", "lp", "cylinder_distance",
           "cone_distance", "torus_distance"]
