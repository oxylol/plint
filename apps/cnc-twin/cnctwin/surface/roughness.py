"""ISO profile evaluation of a simulated profile (docs/models/surface.md section 3).

The kinematic profile z(u) (cnctwin.surface.kernels.envelope_profile) is evaluated the way a stylus
instrument would: nominal form removed by a least-squares quadratic (ISO 4287 primary profile after form
removal), short-wave cut-off lambda_s and the Gaussian roughness filter lambda_c (ISO 16610-21), the cut-off
chosen by ISO 4288 (periodic profiles by RSm, otherwise by Ra), the parameters of ISO 4287 over up to five
sampling lengths after discarding half a cut-off at each end. A profile shorter than two cut-offs cannot be
filtered: its primary-profile parameters over the whole length are reported instead (``method`` "P").
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import abi as A
from . import kernels as K
from . import params as PRM

MEAN_LINE_POINTS_PER_CUTOFF = 50        # the lambda_c mean line is evaluated at lambda_c / 50 and interpolated


@dataclass
class Roughness:
    Ra_m: float = 0.0
    Rq_m: float = 0.0
    Rz_m: float = 0.0
    Rt_m: float = 0.0
    Rp_m: float = 0.0
    Rv_m: float = 0.0
    Rsk: float = 0.0
    Rku: float = 0.0
    RSm_m: float = 0.0
    Rdq: float = 0.0
    cutoff_m: float = 0.0
    lambda_s_m: float = 0.0
    sampling_lengths: int = 0
    evaluation_m: float = 0.0
    profile_m: float = 0.0
    method: str = "R"                   # R: filtered roughness profile; P: primary profile (too short to filter)

    def to_dict(self) -> dict:
        return asdict(self)


class IsoRules:
    """The ISO 4288 / 3274 tables of db/models/surface-roughness.yaml."""

    def __init__(self, values: dict | None = None):
        p = values if values is not None else PRM.roughness()
        self.alpha = float(p["gauss_alpha"])
        self.n_sl = int(round(float(p["sampling_lengths"])))
        self.by_rsm = [(float(a), float(b)) for a, b in p["cutoff_by_rsm_m"]]
        self.by_ra = [(float(a), float(b)) for a, b in p["cutoff_by_ra_m"]]
        self.stylus = [(float(a), float(b), float(c)) for a, b, c in p["stylus_table_m"]]
        self.hd = float(p["rsm_height_discrimination"])
        self.wd = float(p["rsm_width_discrimination"])

    def cutoff_for_rsm(self, rsm_m: float) -> float:
        for upper, cutoff in self.by_rsm:
            if rsm_m <= upper:
                return cutoff
        return self.by_rsm[-1][1]

    def cutoff_for_ra(self, ra_m: float) -> float:
        for upper, cutoff in self.by_ra:
            if ra_m <= upper:
                return cutoff
        return self.by_ra[-1][1]

    def stylus_for(self, cutoff_m: float) -> tuple[float, float]:
        """(lambda_s, largest sampling spacing) of the cut-off (the nearest tabulated cut-off)."""
        best = min(self.stylus, key=lambda row: abs(math.log(row[0] / cutoff_m)))
        return best[1], best[2]

    def spacing_for(self, rsm_m: float) -> float:
        """Sampling spacing for a profile whose marks repeat at rsm_m: the instrument's spacing for the cut-off
        ISO 4288 assigns, and at most rsm / 40 so every mark is resolved."""
        cutoff = self.cutoff_for_rsm(rsm_m) if rsm_m > 0.0 else self.by_ra[2][1]
        _, spacing = self.stylus_for(cutoff)
        return min(spacing, rsm_m / 40.0) if rsm_m > 0.0 else spacing


def remove_form(u: np.ndarray, z: np.ndarray, degree: int = 2) -> np.ndarray:
    """z minus its least-squares polynomial of ``degree`` (in centred, scaled coordinates)."""
    if len(z) <= degree + 1:
        return z - np.mean(z) if len(z) else z
    x = (u - u.mean()) / max(float(np.ptp(u)), 1e-300)
    coef = np.polynomial.polynomial.polyfit(x, z, degree)
    return z - np.polynomial.polynomial.polyval(x, coef)


def params_of(r: np.ndarray, du: float, i0: int, lsl: int, n_sl: int, rules: IsoRules) -> np.ndarray:
    out = np.zeros(A.N_P)
    K.profile_params(np.ascontiguousarray(r, dtype=float), float(du), int(i0), int(lsl), int(n_sl), rules.hd,
                     rules.wd, out)
    return out


def choose_cutoff(p: np.ndarray, du: float, rules: IsoRules, hint_rsm_m: float = 0.0) -> float:
    """ISO 4288 cut-off of a form-removed profile: periodic profiles by RSm, tried from the shortest cut-off
    up (RSm evaluated with the width discrimination of that cut-off's sampling length) until the RSm table
    returns the cut-off tried; otherwise by Ra. ``hint_rsm_m`` (the feed spacing of a kinematic profile) puts
    its cut-off first in the trial order (the result is the same, it is found sooner)."""
    n = len(p)
    out = np.zeros(A.N_P)
    trials = [c for _, c in rules.by_rsm]
    if hint_rsm_m > 0.0:
        first = rules.cutoff_for_rsm(hint_rsm_m)
        trials = [first] + [c for c in trials if c != first]
    for cutoff in trials:
        lsl = int(round(cutoff / du))
        if lsl < 4 or lsl > n:
            continue
        n_sl = max(1, n // lsl)
        K.profile_params(np.ascontiguousarray(p), float(du), 0, lsl, n_sl, rules.hd, rules.wd, out)
        if out[A.P_N_ELEM] >= 2 and out[A.P_RSM] > 0.0 and rules.cutoff_for_rsm(float(out[A.P_RSM])) == cutoff:
            return cutoff
    est = params_of(p, du, 0, n, 1, rules)
    return rules.cutoff_for_ra(float(est[A.P_RA]))


def evaluate(z: np.ndarray, du: float, rules: IsoRules, cutoff_m: float | None = None,
             hint_rsm_m: float = 0.0) -> Roughness:
    """ISO evaluation of a finite, uniformly sampled profile z (spacing du); see the module doc."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    res = Roughness(profile_m=n * du)
    if n < 8:
        res.method = "none"
        return res
    u = np.arange(n) * du
    p = remove_form(u, z)
    if cutoff_m is None:
        cutoff_m = choose_cutoff(p, du, rules, hint_rsm_m)
    lam_s, _ = rules.stylus_for(cutoff_m)
    primary = np.empty(n)
    if lam_s > 2.0 * du:
        K.gaussian_mean(np.ascontiguousarray(p), float(du), float(lam_s), rules.alpha, 1, primary)
    else:
        primary[:] = p
    res.cutoff_m = float(cutoff_m)
    res.lambda_s_m = float(lam_s)
    lsl = int(round(cutoff_m / du))
    margin = int(math.ceil(0.5 * cutoff_m / du))
    available = n - 2 * margin
    n_sl = min(rules.n_sl, available // lsl) if lsl > 0 else 0
    if n_sl >= 1:
        mean = np.empty(n)
        stride = max(1, int(cutoff_m / (MEAN_LINE_POINTS_PER_CUTOFF * du)))
        K.gaussian_mean(np.ascontiguousarray(primary), float(du), float(cutoff_m), rules.alpha, stride, mean)
        r = primary - mean
        i0 = margin + (available - n_sl * lsl) // 2
        out = params_of(r, du, i0, lsl, n_sl, rules)
        res.method = "R"
        res.sampling_lengths = int(n_sl)
        res.evaluation_m = n_sl * lsl * du
    else:
        r = primary - float(np.mean(primary))
        out = params_of(r, du, 0, n, 1, rules)
        res.method = "P"
        res.sampling_lengths = 1
        res.evaluation_m = n * du
    res.Ra_m, res.Rq_m = float(out[A.P_RA]), float(out[A.P_RQ])
    res.Rz_m, res.Rt_m = float(out[A.P_RZ]), float(out[A.P_RT])
    res.Rp_m, res.Rv_m = float(out[A.P_RP]), float(out[A.P_RV])
    res.Rsk, res.Rku = float(out[A.P_RSK]), float(out[A.P_RKU])
    res.RSm_m, res.Rdq = float(out[A.P_RSM]), float(out[A.P_RDQ])
    return res


def primary_params(z: np.ndarray, du: float, rules: IsoRules) -> Roughness:
    """Parameters of the unfiltered profile after form removal over its whole length (no lambda_s, no
    lambda_c): the kinematic values Pt, Pa ... the analytic formulas (f^2 / 8 r) describe."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    res = Roughness(profile_m=n * du, method="P")
    if n < 8:
        res.method = "none"
        return res
    p = remove_form(np.arange(n) * du, z, degree=1)
    r = p - float(np.mean(p))
    out = params_of(r, du, 0, n, 1, rules)
    res.sampling_lengths = 1
    res.evaluation_m = n * du
    res.Ra_m, res.Rq_m = float(out[A.P_RA]), float(out[A.P_RQ])
    res.Rz_m, res.Rt_m = float(out[A.P_RZ]), float(out[A.P_RT])
    res.Rp_m, res.Rv_m = float(out[A.P_RP]), float(out[A.P_RV])
    res.Rsk, res.Rku = float(out[A.P_RSK]), float(out[A.P_RKU])
    res.RSm_m, res.Rdq = float(out[A.P_RSM]), float(out[A.P_RDQ])
    return res


__all__ = ["Roughness", "IsoRules", "evaluate", "choose_cutoff", "primary_params", "remove_form", "params_of"]
