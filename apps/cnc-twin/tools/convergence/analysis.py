"""Convergence analysis of one quantity over the refinement levels of a study (docs/convergence.md "Method").

A level is a value of the refined setting expressed as a step size ``h`` (smaller is finer: degrees per L0 step,
lattice spacing, element length, 1 / memory bins, macro step, cell size ...). For the three finest levels
h1 < h2 < h3 with values f1, f2, f3 the observed order p and the Richardson estimate follow Celik et al. (2008,
"Procedure for estimation and reporting of uncertainty due to discretization in CFD applications", J. Fluids Eng.
130, 078001) for refinement ratios that need not be equal:

    r21 = h2 / h1, r32 = h3 / h2, e21 = f2 - f1, e32 = f3 - f2, s = sign(e32 / e21)
    p = |ln|e32 / e21| + q(p)| / ln r21,  q(p) = ln((r21^p - s) / (r32^p - s))     (fixed point)
    f_ext = (r21^p f1 - f2) / (r21^p - 1),  GCI_fine = 1.25 |e21| / (r21^p - 1)

The extrapolated value is the reference when the convergence is monotone (s > 0) with an order in [P_MIN, P_MAX];
otherwise (oscillating, noise-dominated or non-asymptotic) the finest level is the reference and its uncertainty is
the largest difference between the finest level and the next two. A quantity whose levels all agree to within its
noise floor is "converged at every level" (no order). The error of a level is |f(h) - f_ref|, relative to |f_ref|
or absolute (quantities with an absolute tolerance). The shipped default is inside the converged range when its
error is within the documented tolerance; otherwise the coarsest measured level within it is the recommended one.
"""

from __future__ import annotations

import math

P_MIN = 0.5          # observed orders outside [P_MIN, P_MAX] are not trusted for extrapolation
P_MAX = 4.0          # the schemes are first or second order; a larger fitted order is a jump, not asymptotic


def observed_order(h, f, max_iter: int = 200):
    """(p, s) of the three levels (h1 < h2 < h3, values f1, f2, f3); p None when undefined (a zero difference)."""
    (h1, h2, h3), (f1, f2, f3) = h, f
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0.0 or e32 == 0.0:
        return None, 0
    r21, r32 = h2 / h1, h3 / h2
    s = 1.0 if e32 / e21 > 0.0 else -1.0
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(max_iter):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None, int(s)
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    return p, int(s)


def analyse(levels_h, values, *, default_h, tol, relative: bool = True, noise: float = 0.0, exact=None):
    """Convergence of one quantity. ``levels_h``: the step sizes, ``values``: f at each level (same order),
    ``default_h``: the shipped default's step size (one of the levels), ``tol``: the documented tolerance (relative
    when ``relative``, else absolute in the quantity's unit), ``noise``: differences at or below this (in the same
    sense as ``tol``) are noise, ``exact``: the exact value when the case has one (it is then the reference; the
    observed order and the Richardson estimate are still reported). Returns a JSON-able dict."""
    pairs = sorted((float(h), float(v)) for h, v in zip(levels_h, values) if v is not None and math.isfinite(v))
    out = {"h": [p[0] for p in pairs], "values": [p[1] for p in pairs], "tol": tol, "relative": relative,
           "default_h": float(default_h)}
    if len(pairs) < 2:
        out.update(status="insufficient", reference=None)
        return out
    hs = [p[0] for p in pairs]
    fs = [p[1] for p in pairs]
    finest = fs[0]
    scale = abs(finest) if relative else 1.0
    if relative and scale == 0.0:
        scale = max(abs(v) for v in fs) or 1.0

    def err(v, ref):
        return abs(v - ref) / scale

    spread = max(err(v, finest) for v in fs)
    p = s = None
    ref = finest
    unc = err(fs[1], finest) if len(fs) > 1 else 0.0
    status = "finest"
    if spread <= noise:
        status = "flat"
        unc = spread
    elif len(fs) >= 3:
        p, s = observed_order(hs[:3], fs[:3])
        e21 = fs[1] - fs[0]
        unc = max(err(fs[1], finest), err(fs[2], finest))
        if p is not None and s > 0 and P_MIN <= p <= P_MAX and err(fs[1], fs[0]) > noise:
            r21 = hs[1] / hs[0]
            ref = (r21 ** p * fs[0] - fs[1]) / (r21 ** p - 1.0)
            unc = 1.25 * abs(e21) / (r21 ** p - 1.0) / scale
            status = "richardson"
        elif s is not None and s < 0:
            status = "oscillating"
        elif err(fs[1], fs[0]) <= noise:
            status = "finest"
            unc = err(fs[1], finest)
    if exact is not None:
        out["richardson"] = ref if status == "richardson" else None
        ref, status, unc = float(exact), "exact", 0.0
        if relative and ref != 0.0:
            scale = abs(ref)
    errors = [err(v, ref) for v in fs]
    out.update(status=status, order=p, sign=s, reference=ref, reference_uncertainty=unc, errors=errors)
    d = [i for i, h in enumerate(hs) if abs(h - default_h) <= 1e-12 * max(1.0, abs(default_h))]
    if d:
        e_def = errors[d[0]]
        out["default_value"] = fs[d[0]]
        out["default_error"] = e_def
        out["default_inside"] = bool(e_def <= tol)
    ok = [i for i, e in enumerate(errors) if e <= tol]
    # the converged range: every level at least as fine as the coarsest one within the tolerance, provided all finer
    # levels are within it too (a level inside the tolerance between two outside it is not a range)
    coarsest_ok = None
    for i in range(len(hs)):
        if all(errors[j] <= tol for j in range(i + 1)):
            coarsest_ok = i
    out["coarsest_within_tol_h"] = hs[coarsest_ok] if coarsest_ok is not None else None
    out["within_tol_h"] = [hs[i] for i in ok]
    out["conclusive"] = bool(unc <= tol)
    return out
