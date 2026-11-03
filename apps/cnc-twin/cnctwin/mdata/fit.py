"""Bounded least squares through a simulate(params) callback, with intervals from the Jacobian.

Problem (docs/models/calibration.md "Calibration"): parameters theta (SI, each with bounds; some
fitted in log space), a callback ``simulate(theta) -> {observation name: predicted array}`` and
observation sets ``y_i`` with standard uncertainties ``sigma_i``. The weighted residual

    r(theta) = [(simulate(theta)[i] - y_i) / sigma_i  for every set i, finite samples only]

is minimised by ``scipy.optimize.least_squares`` (trust region reflective, branch-coleman-li-1999,
virtanen-2020-scipy) inside the bounds. Sets without a stated sigma are weighted by their own
scatter: after a fit each set's sigma becomes its RMS residual (floored at its resolution
/ sqrt(12), jcgm-100-2008) and the fit is repeated (``reweight_passes``).

Inference (bates-watts-1988 ch. 2.3, seber-wild-2003 ch. 5): with the weighted Jacobian J at the
solution (SVD J = U S V'), s^2 = |r|^2 / (m - p) and

    cov = s^2 V S^-2 V'   (singular values below sv_rel_cutoff * s_max dropped),

the interval of each parameter is theta_k +- t(1 - alpha/2, m - p) sqrt(cov_kk) (log-space
parameters: the interval of ln theta, exponentiated: asymmetric). A parameter loading on a dropped
direction is **not identifiable** (no interval); pairs with |correlation| > correlation_flag are
**confounded**; a parameter on a bound has an interval that is only one-sided. Because these are
linearisation (Wald) intervals, the cost is re-evaluated at both ends of each interval: when the
actual increase differs from the predicted one by more than ci_check_ratio_max the interval is
flagged approximate (curvature, bates-watts-1988 ch. 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.errors import InputError
from .align import error_metrics
from .params import calibration_params

Simulate = Callable[[dict], dict]


class FitError(InputError):
    """The fit cannot be set up or the simulation failed during it."""


@dataclass(frozen=True)
class Param:
    name: str
    value: float                     # initial value = prior (SI)
    lower: float
    upper: float
    unit: str = "-"                  # SI unit text of the value
    free: bool = True
    log: bool = False                # fitted as ln(value) (positive parameters spanning decades)
    scale: float | None = None       # typical magnitude (x_scale); default from the value and the bounds
    targets: tuple = ()              # overlay targets (cnctwin.mdata.overlay.Target)
    doc: str = ""

    def internal(self, value: float) -> float:
        return math.log(value) if self.log else float(value)

    def external(self, x: float) -> float:
        return math.exp(x) if self.log else float(x)


@dataclass
class ObservationSet:
    name: str
    measured: np.ndarray
    sigma: float | np.ndarray | None = None   # standard uncertainty; None: estimated from the scatter
    unit: str = ""
    t_s: np.ndarray | None = None
    floor: float = 0.0                         # resolution / sqrt(12): sigma never goes below it
    doc: str = ""

    def __post_init__(self) -> None:
        self.measured = np.asarray(self.measured, dtype=np.float64).ravel()


@dataclass
class FitSettings:
    tol: float
    diff_step: float
    max_nfev_per_param: int
    confidence: float
    sv_cutoff: float
    correlation_flag: float
    ci_check: bool
    ci_check_ratio_max: float
    reweight_passes: int
    central_jacobian: bool = True
    loss: str = "linear"
    absolute_sigma: bool = False

    @classmethod
    def for_simulator(cls, kind: str = "surrogate", **overrides) -> "FitSettings":
        p = calibration_params()
        loop = kind == "loop"
        values = dict(tol=float(p["tol_loop" if loop else "tol_surrogate"]),
                      diff_step=float(p["diff_step_loop" if loop else "diff_step_surrogate"]),
                      max_nfev_per_param=int(p["max_nfev_per_param"]), confidence=float(p["confidence_level"]),
                      sv_cutoff=float(p["sv_rel_cutoff"]), correlation_flag=float(p["correlation_flag"]),
                      ci_check=not loop, ci_check_ratio_max=float(p["ci_check_ratio_max"]),
                      reweight_passes=int(p["reweight_passes"]), central_jacobian=not loop)
        values.update(overrides)
        return cls(**values)


@dataclass
class FitResult:
    params: list[dict]
    names: list[str]
    values: dict[str, float]
    correlation: list[list[float | None]]
    confounded: list[dict]
    singular_values: list[float]
    condition_number: float | None
    s2: float
    dof: int
    n_obs: int
    cost_before: float
    cost_after: float
    nfev: int
    success: bool
    status: int
    message: str
    sigma: dict[str, float]
    metrics: dict
    predictions: dict = field(default_factory=dict)    # {set: {"prior": array, "fit": array}}

    def to_dict(self, *, predictions: bool = False) -> dict:
        out = {"params": self.params, "free": self.names, "values": self.values, "correlation": self.correlation,
               "confounded": self.confounded, "singular_values": self.singular_values,
               "condition_number": self.condition_number, "s2": self.s2, "dof": self.dof, "n_obs": self.n_obs,
               "cost_before": self.cost_before, "cost_after": self.cost_after, "nfev": self.nfev,
               "success": self.success, "status": self.status, "message": self.message, "sigma": self.sigma,
               "metrics": self.metrics}
        if predictions:
            out["predictions"] = {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in self.predictions.items()}
        return out


def _evaluate(simulate: Simulate, values: dict, observations: list[ObservationSet]) -> dict[str, np.ndarray]:
    try:
        predicted = simulate(dict(values))
    except InputError:
        raise
    except (ArithmeticError, ValueError) as error:
        raise FitError(f"the simulation failed at {values}: {type(error).__name__}: {error}") from None
    out = {}
    for obs in observations:
        if obs.name not in predicted:
            raise FitError(f"the simulation returned no {obs.name!r} (it returned {', '.join(sorted(predicted))})")
        array = np.asarray(predicted[obs.name], dtype=np.float64).ravel()
        if array.shape != obs.measured.shape:
            raise FitError(f"{obs.name}: the simulation gives {array.size} values for {obs.measured.size} measured")
        out[obs.name] = array
    return out


def fit(params: list[Param], simulate: Simulate, observations: list[ObservationSet],
        settings: FitSettings | None = None, progress: Callable | None = None) -> FitResult:
    """Fit the free parameters (see the module docstring); fixed ones keep their value."""
    from scipy.optimize import least_squares
    from scipy.stats import t as student_t

    settings = settings or FitSettings.for_simulator()
    if len({p.name for p in params}) != len(params):
        raise FitError("parameter names must be unique")
    for p in params:
        if p.free and not p.lower < p.upper:
            raise FitError(f"parameter {p.name}: a free parameter needs lower < upper, not [{p.lower:g}, {p.upper:g}]")
        if not p.lower <= p.value <= p.upper:
            raise FitError(f"parameter {p.name}: initial value {p.value:g} is outside its bounds "
                           f"[{p.lower:g}, {p.upper:g}]")
        if p.log and p.lower <= 0:
            raise FitError(f"parameter {p.name}: a log-space parameter needs a positive lower bound")
    free = [p for p in params if p.free]
    fixed = {p.name: p.value for p in params if not p.free}
    masks = {o.name: np.isfinite(o.measured) for o in observations}
    m = int(sum(mask.sum() for mask in masks.values()))
    n = len(free)
    if m == 0:
        raise FitError("no finite measured values to fit")
    if n and m <= n:
        raise FitError(f"{m} observations cannot determine {n} parameters (need more than {n})")
    evaluations = [0]

    def values_of(x: np.ndarray) -> dict:
        out = dict(fixed)
        for p, xi in zip(free, x):
            out[p.name] = p.external(float(xi))
        return out

    prior_values = {p.name: p.value for p in params}
    prior_pred = _evaluate(simulate, prior_values, observations)
    evaluations[0] += 1
    for obs in observations:
        bad = masks[obs.name] & ~np.isfinite(prior_pred[obs.name])
        if bad.any():
            raise FitError(f"{obs.name}: the simulation is not finite at {int(bad.sum())} measured points "
                           f"with the initial parameters")
    sigma = {}
    for obs in observations:
        if obs.sigma is not None:
            sigma[obs.name] = obs.sigma
        else:
            residual = prior_pred[obs.name][masks[obs.name]] - obs.measured[masks[obs.name]]
            scatter = float(np.std(obs.measured[masks[obs.name]]))
            sigma[obs.name] = max(float(np.sqrt(np.mean(residual ** 2))), scatter, obs.floor, 1e-300)

    def residuals(x: np.ndarray, sig: dict) -> np.ndarray:
        evaluations[0] += 1
        predicted = _evaluate(simulate, values_of(x), observations)
        parts = []
        for obs in observations:
            mask = masks[obs.name]
            s = sig[obs.name]
            s = s[mask] if isinstance(s, np.ndarray) else s
            r = (predicted[obs.name][mask] - obs.measured[mask]) / s
            if not np.all(np.isfinite(r)):
                raise FitError(f"{obs.name}: the simulation became non-finite at {values_of(x)}")
            parts.append(r)
        if progress is not None:
            progress(phase="fit", evaluations=evaluations[0])
        return np.concatenate(parts)

    def cost(x, sig) -> float:
        r = residuals(x, sig)
        return 0.5 * float(r @ r)

    lower = np.array([p.internal(p.lower) for p in free])
    upper = np.array([p.internal(p.upper) for p in free])
    x0 = np.array([p.internal(p.value) for p in free])
    span = upper - lower
    x0 = np.clip(x0, lower + 1e-9 * span, upper - 1e-9 * span) if n else x0
    x_scale = np.array([1.0 if p.log else (p.scale or max(abs(p.value), 1e-3 * (p.upper - p.lower), 1e-300))
                        for p in free])
    estimated = [o.name for o in observations if o.sigma is None]
    passes = settings.reweight_passes if (estimated and len(observations) > 1) else 0
    result = None
    x = x0.copy()
    cost_before = cost(x0, sigma) if n else 0.0
    if n:
        for attempt in range(passes + 1):
            result = least_squares(residuals, x, jac="2-point", bounds=(lower, upper), method="trf",
                                   x_scale=x_scale, diff_step=settings.diff_step, ftol=settings.tol,
                                   xtol=settings.tol, gtol=settings.tol, loss=settings.loss,
                                   max_nfev=settings.max_nfev_per_param * n, args=(sigma,))
            x = result.x
            if attempt == passes:
                break
            fitted = _evaluate(simulate, values_of(x), observations)
            for obs in observations:
                if obs.sigma is None:
                    mask = masks[obs.name]
                    rms = float(np.sqrt(np.mean((fitted[obs.name][mask] - obs.measured[mask]) ** 2)))
                    sigma[obs.name] = max(rms, obs.floor, 1e-300)
            cost_before = cost(x0, sigma)
    values = values_of(x) if n else dict(prior_values)
    fit_pred = _evaluate(simulate, values, observations)
    r_final = residuals(x, sigma)
    cost_after = 0.5 * float(r_final @ r_final)
    if not n:
        cost_before = cost_after
    dof = m - n
    s2 = float(r_final @ r_final) / dof if dof > 0 else math.nan
    variance_scale = 1.0 if settings.absolute_sigma else s2
    # Jacobian at the solution (internal coordinates)
    if n:
        if settings.central_jacobian:
            J = np.empty((m, n))
            for k in range(n):
                h = settings.diff_step ** (1.0 / 1.5) * max(abs(x[k]), x_scale[k] if not free[k].log else 1.0)
                lo, hi = max(x[k] - h, lower[k]), min(x[k] + h, upper[k])
                xp, xm = x.copy(), x.copy()
                xp[k], xm[k] = hi, lo
                J[:, k] = (residuals(xp, sigma) - residuals(xm, sigma)) / (hi - lo)
        else:
            J = np.asarray(result.jac, dtype=np.float64)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        keep = S > settings.sv_cutoff * (S[0] if len(S) else 0.0)
        inv = np.zeros_like(S)
        inv[keep] = 1.0 / S[keep] ** 2
        cov = (Vt.T * inv) @ Vt * variance_scale
        dropped = Vt[~keep] if (~keep).any() else np.zeros((0, n))
        not_identifiable = np.any(np.abs(dropped) > 0.1, axis=0) if len(dropped) else np.zeros(n, bool)
        condition = float(S[0] / S[keep][-1]) if keep.any() else None
        t_quantile = float(student_t.ppf(0.5 + settings.confidence / 2.0, dof)) if dof > 0 else math.nan
        std_internal = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(std_internal, std_internal)
        active = result.active_mask if result is not None else np.zeros(n, int)
    else:
        S, keep, cov, not_identifiable, condition, t_quantile = np.zeros(0), np.zeros(0, bool), np.zeros((0, 0)), \
            np.zeros(0, bool), None, math.nan
        std_internal, corr, active = np.zeros(0), np.zeros((0, 0)), np.zeros(0, int)
    out_params = []
    free_index = {p.name: k for k, p in enumerate(free)}
    for p in params:
        record = {"name": p.name, "unit": p.unit, "prior": p.value, "value": values[p.name], "free": p.free,
                  "bounds": [p.lower, p.upper], "log": p.log, "doc": p.doc,
                  "targets": [t.to_dict() if hasattr(t, "to_dict") else t for t in p.targets]}
        if p.free:
            k = free_index[p.name]
            half = t_quantile * std_internal[k]
            identifiable = not bool(not_identifiable[k]) and math.isfinite(half)
            record["identifiable"] = identifiable
            record["at_bound"] = {-1: "lower", 1: "upper"}.get(int(active[k]), None)
            if identifiable:
                lo, hi = x[k] - half, x[k] + half
                record["ci"] = [p.external(max(lo, -700.0) if p.log else lo), p.external(min(hi, 700.0) if p.log else hi)]
                record["std_error"] = float(std_internal[k] * (values[p.name] if p.log else 1.0))
                record["std_error_internal"] = float(std_internal[k])
            else:
                record["ci"], record["std_error"], record["std_error_internal"] = None, None, None
            if settings.ci_check and identifiable and half > 0:
                record["ci_check"] = _ci_check(k, x, half, cost, sigma, cost_after, J, lower, upper,
                                               variance_scale, settings.ci_check_ratio_max)
        out_params.append(record)
    names = [p.name for p in free]
    confounded = []
    for i in range(n):
        for j in range(i + 1, n):
            value = corr[i, j]
            if math.isfinite(value) and abs(value) > settings.correlation_flag:
                confounded.append({"a": names[i], "b": names[j], "correlation": float(value)})
    metrics = {}
    for obs in observations:
        metrics[obs.name] = {"unit": obs.unit, "before": error_metrics(obs.measured, prior_pred[obs.name]),
                             "after": error_metrics(obs.measured, fit_pred[obs.name])}
    return FitResult(
        params=out_params, names=names, values={k: float(v) for k, v in values.items()},
        correlation=[[float(c) if math.isfinite(c) else None for c in row] for row in corr],
        confounded=confounded, singular_values=[float(s) for s in S], condition_number=condition,
        s2=s2, dof=dof, n_obs=m, cost_before=float(cost_before), cost_after=cost_after, nfev=evaluations[0],
        success=bool(result.success) if result is not None else True,
        status=int(result.status) if result is not None else 0,
        message=str(result.message) if result is not None else "no free parameter: evaluated only",
        sigma={k: float(np.mean(v)) if isinstance(v, np.ndarray) else float(v) for k, v in sigma.items()},
        metrics=metrics, predictions={o.name: {"prior": prior_pred[o.name], "fit": fit_pred[o.name]}
                                      for o in observations})


def _ci_check(k, x, half, cost, sigma, cost_min, J, lower, upper, variance_scale, ratio_max) -> dict:
    """Actual vs linearised cost increase at both ends of parameter k's interval (others fixed)."""
    predicted = 0.5 * half ** 2 * float(J[:, k] @ J[:, k])
    out = {}
    for side, sign in (("lower", -1.0), ("upper", 1.0)):
        xs = x.copy()
        xs[k] = min(max(x[k] + sign * half, lower[k]), upper[k])
        moved = abs(xs[k] - x[k])
        if moved < 0.5 * half:
            out[side] = {"ratio": None, "note": "interval end outside the bounds"}
            continue
        expected = predicted * (moved / half) ** 2
        actual = cost(xs, sigma) - cost_min
        ratio = actual / expected if expected > 0 else None
        out[side] = {"ratio": ratio, "approximate": ratio is None or not (1.0 / ratio_max <= ratio <= ratio_max)}
    out["approximate"] = any(v.get("approximate", False) for v in out.values() if isinstance(v, dict))
    return out


__all__ = ["Param", "ObservationSet", "FitSettings", "FitResult", "FitError", "fit"]
