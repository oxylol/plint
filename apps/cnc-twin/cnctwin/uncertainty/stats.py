"""Statistics of an uncertainty study: distributions, capability, probabilities, rank correlations.

Equations (docs/models/uncertainty.md for validity and error bounds):

* quantiles: linear interpolation between order statistics (Hyndman & Fan 1996, definition 7);
  mean with its Student-t interval ``mean +- t(1 - a/2, n - 1) s / sqrt(n)``;
* capability (Kotz & Johnson 1993, AIAG SPC 2005, ISO 22514-2): ``Cp = (USL - LSL) / 6 s``,
  ``Cpk = min(USL - mean, mean - LSL) / 3 s``; one-sided zones (form, orientation, location: the
  lower bound 0 is natural, not a specification) use ``Cpu = (USL - mean) / 3 s`` as Cpk. Interval
  of Cpk (Bissell 1990) ``Cpk +- z sqrt(1/(9 n) + Cpk^2 / (2 (n - 1)))``, of Cp from chi-square
  ``Cp sqrt(chi2(a/2, n - 1)/(n - 1)) .. Cp sqrt(chi2(1 - a/2, n - 1)/(n - 1))``; the quantile
  indices of ISO 22514-2 ``Cp_q = (USL - LSL)/(X99.865 - X0.135)``,
  ``Cpk_q = min((USL - X50)/(X99.865 - X50), (X50 - LSL)/(X50 - X0.135))``; the normal fraction out
  of tolerance ``Phi((LSL - mean)/s) + 1 - Phi((USL - mean)/s)``;
* probabilities: the Wilson score interval (Wilson 1927; Brown, Cai & DasGupta 2001);
* sensitivity: Spearman's rank correlation (Spearman 1904) with the t test of Zar (1972), and the
  partial rank correlation (Iman & Conover 1980; Marino et al. 2008): the Pearson correlation of
  the residuals of the ranks of the factor and of the output, each regressed (least squares, with
  intercept) on the ranks of every other factor, tested with t on n - 2 - (k - 1) degrees of freedom.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def _z(confidence: float) -> float:
    return float(sps.norm.ppf(0.5 + 0.5 * confidence))


def finite(values) -> np.ndarray:
    data = np.asarray([v for v in values if v is not None], float)
    return data[np.isfinite(data)]


def describe(values, levels, confidence: float) -> dict:
    data = finite(values)
    n = len(data)
    out = {"n": n, "mean": None, "std": None, "min": None, "max": None, "quantiles": {}, "mean_ci": None}
    if not n:
        return out
    out.update({"mean": float(data.mean()), "min": float(data.min()), "max": float(data.max())})
    out["quantiles"] = {f"{level:.6g}": float(np.quantile(data, level, method="linear")) for level in levels}
    if n > 1:
        std = float(data.std(ddof=1))
        out["std"] = std
        half = float(sps.t.ppf(0.5 + 0.5 * confidence, n - 1)) * std / math.sqrt(n)
        out["mean_ci"] = [out["mean"] - half, out["mean"] + half]
    else:
        out["std"] = 0.0
    return out


def wilson(successes: int, n: int, confidence: float) -> dict:
    """Probability ``successes / n`` with its Wilson score interval."""
    if n <= 0:
        return {"p": None, "low": None, "high": None, "k": int(successes), "n": int(n)}
    z = _z(confidence)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    low = 0.0 if successes <= 0 else max(0.0, centre - half)        # exact at the ends (the score interval
    high = 1.0 if successes >= n else min(1.0, centre + half)       # reaches 0 and 1 there; rounding would not)
    return {"p": p, "low": low, "high": high, "k": int(successes), "n": int(n)}


def capability(values, lower, upper, *, one_sided: bool, confidence: float, target: float, min_samples: int,
               levels) -> dict:
    """Capability indices of the measured values against [lower, upper] (SI)."""
    data = finite(values)
    n = len(data)
    out = {"n": n, "lower": None if one_sided else lower, "upper": upper, "one_sided": one_sided, "Cp": None,
           "Cpk": None, "Cpu": None, "Cpl": None, "Cpk_ci": None, "Cp_ci": None, "Cp_q": None, "Cpk_q": None,
           "expected_out_frac": None, "capable": None, "reported": n >= min_samples, "target": target}
    if n < 2 or upper is None and lower is None:
        return out
    mean = float(data.mean())
    std = float(data.std(ddof=1))
    low = None if one_sided else lower
    if std <= 0.0:
        inside = (low is None or mean >= low) and (upper is None or mean <= upper)
        out["expected_out_frac"] = 0.0 if inside else 1.0
        return out
    cpu = (upper - mean) / (3.0 * std) if upper is not None else None
    cpl = (mean - low) / (3.0 * std) if low is not None else None
    cpk = min(v for v in (cpu, cpl) if v is not None)
    z = _z(confidence)
    out.update({"Cpu": cpu, "Cpl": cpl, "Cpk": cpk})
    half = z * math.sqrt(1.0 / (9.0 * n) + cpk * cpk / (2.0 * (n - 1)))
    out["Cpk_ci"] = [cpk - half, cpk + half]
    if low is not None and upper is not None:
        cp = (upper - low) / (6.0 * std)
        alpha = 1.0 - confidence
        out["Cp"] = cp
        out["Cp_ci"] = [cp * math.sqrt(sps.chi2.ppf(alpha / 2.0, n - 1) / (n - 1)),
                        cp * math.sqrt(sps.chi2.ppf(1.0 - alpha / 2.0, n - 1) / (n - 1))]
    q_lo, q_mid, q_hi = (float(np.quantile(data, q, method="linear")) for q in (levels[0], 0.5, levels[-1]))
    if q_hi > q_lo:
        if low is not None and upper is not None:
            out["Cp_q"] = (upper - low) / (q_hi - q_lo)
        parts = []
        if upper is not None and q_hi > q_mid:
            parts.append((upper - q_mid) / (q_hi - q_mid))
        if low is not None and q_mid > q_lo:
            parts.append((q_mid - low) / (q_mid - q_lo))
        out["Cpk_q"] = min(parts) if parts else None
    expected = 0.0
    if low is not None:
        expected += float(sps.norm.cdf((low - mean) / std))
    if upper is not None:
        expected += float(sps.norm.sf((upper - mean) / std))
    out["expected_out_frac"] = expected
    out["capable"] = cpk >= target
    return out


def ranks(values: np.ndarray) -> np.ndarray:
    return sps.rankdata(values, method="average")


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3 or np.ptp(x[mask]) == 0.0 or np.ptp(y[mask]) == 0.0:
        return {"rho": None, "p": None, "n": n}
    rx, ry = ranks(x[mask]), ranks(y[mask])
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rho = max(-1.0, min(1.0, rho))
    if abs(rho) >= 1.0:
        return {"rho": rho, "p": 0.0, "n": n}
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    return {"rho": rho, "p": float(2.0 * sps.t.sf(abs(t), n - 2)), "n": n}


def prcc(X: np.ndarray, y: np.ndarray, min_dof: int) -> list[dict]:
    """Partial rank correlation of each column of X with y (None where undefined)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    n, k = X.shape if X.ndim == 2 else (0, 0)
    varied = [j for j in range(k) if n and np.ptp(X[:, j]) > 0.0]
    dof = n - 2 - (len(varied) - 1)
    out = [{"prcc": None, "p": None, "dof": dof} for _ in range(k)]
    if n == 0 or dof < min_dof or np.ptp(y) == 0.0:
        return out
    R = np.column_stack([ranks(X[:, j]) for j in varied])
    ry = ranks(y)
    for position, j in enumerate(varied):
        others = np.column_stack([np.ones(n)] + [R[:, i] for i in range(len(varied)) if i != position])
        res_x = R[:, position] - others @ np.linalg.lstsq(others, R[:, position], rcond=None)[0]
        res_y = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
        if np.std(res_x) == 0.0 or np.std(res_y) == 0.0:
            continue
        r = float(np.clip(np.corrcoef(res_x, res_y)[0, 1], -1.0, 1.0))
        p = 0.0 if abs(r) >= 1.0 else float(2.0 * sps.t.sf(abs(r * math.sqrt(dof / (1.0 - r * r))), dof))
        out[j] = {"prcc": r, "p": p, "dof": dof}
    return out


__all__ = ["describe", "wilson", "capability", "spearman", "prcc", "ranks", "finite"]
