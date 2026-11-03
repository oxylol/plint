"""Alignment of measured machine data with a simulated run, resampling, overlays and error metrics.

Model (docs/models/calibration.md "Alignment"): one affine map between the two time bases,

    t_sim = offset_s + scale * t_meas

``scale`` absorbs the drift between the machine's clock and the simulated clock (ppm for
quartz clocks; up to ``max_scale_deviation`` for a mis-stated sampling rate).

* **Markers** (preferred): the measured block markers (changes of ``program.line``, else
  ``program.sequence``, else ``program.block`` text) are matched to the simulated block starts with
  the same key. The offset is found by voting (every pair of equal keys votes for
  ``T_sim - t_meas``; the densest bin wins, robust to repeated lines), the pairs are then matched
  in time order to the nearest simulated start within a window, and ``(offset, scale)`` are fitted
  by Huber-weighted least squares with outlier rejection (huber-1981). The scale is fitted only
  when enough markers span ``scale_min_span_s``. Polled markers lag the true block start by a
  uniform 0..dt_poll; when the marker times lie on a polling grid the offset is corrected by
  dt_poll / 2 (the mean lag) and the residual scale is then about dt_poll / sqrt(12).
* **Cross-correlation** (no usable markers): the measured spindle load / power and the
  simulated one are resampled on a common grid and the masked normalised cross-correlation is
  evaluated at every lag by FFT (padfield-2012-masked-ncc); the peak, refined by a parabola
  through its neighbours, is the offset (knapp-carter-1976). The scale comes from local offsets
  in ``xcorr_scale_windows`` windows. A secondary peak above ``xcorr_ambiguity_ratio`` of the main
  one marks the delay ambiguous (periodic passes).
* **Given**: ``offset_s`` (and ``scale``) stated by the user.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from .log import Channel, MachineLog
from .params import align_params
from .simrun import Marker

FORMAT = "cnctwin-overlay"


@dataclass
class Alignment:
    offset_s: float = 0.0
    scale: float = 1.0
    method: str = "none"                 # markers | xcorr | given | none
    quality: dict = field(default_factory=dict)

    def to_sim(self, t_meas) -> np.ndarray:
        return self.offset_s + self.scale * np.asarray(t_meas, dtype=np.float64)

    def to_meas(self, t_sim) -> np.ndarray:
        return (np.asarray(t_sim, dtype=np.float64) - self.offset_s) / self.scale

    def to_dict(self) -> dict:
        return {"offset_s": float(self.offset_s), "scale": float(self.scale), "method": self.method,
                "quality": _plain(self.quality)}

    @classmethod
    def from_dict(cls, data: dict) -> "Alignment":
        return cls(float(data.get("offset_s", 0.0)), float(data.get("scale", 1.0)), str(data.get("method", "given")),
                   dict(data.get("quality") or {}))


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ----------------------------------------------------------------------------- robust line fit

def robust_line(x: np.ndarray, y: np.ndarray, *, fit_slope: bool, p: dict | None = None,
                sigma_floor: float = 0.0) -> dict:
    """Huber-IRLS fit of y = a + b x (b = 1 when not fit_slope) with outlier rejection.

    Returns {a, b, sigma, inliers (bool mask)}. sigma is the robust residual scale (MAD * mad_to_sigma)
    of the inliers, never below ``sigma_floor`` (the resolution of quantised times, q / sqrt(12),
    jcgm-100-2008): a MAD of zero from quantised times must not make every other residual an outlier."""
    p = p or align_params()
    k = float(p["huber_k"])
    mad_to_sigma = float(p["mad_to_sigma"])
    reject = float(p["marker_outlier_sigma"])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inliers = np.ones(len(x), bool)
    a, b, sigma = 0.0, 1.0, 0.0
    x0 = float(np.median(x)) if len(x) else 0.0           # centre x for conditioning
    for _ in range(10):
        xs, ys = x[inliers] - x0, y[inliers]
        weights = np.ones(len(xs))
        for _ in range(50):
            if fit_slope and len(xs) >= 2:
                w = weights
                sw = w.sum()
                mx, my = (w * xs).sum() / sw, (w * ys).sum() / sw
                sxx = (w * (xs - mx) ** 2).sum()
                b_new = (w * (xs - mx) * (ys - my)).sum() / sxx if sxx > 0 else 1.0
                a_new = my - b_new * mx
            else:
                b_new = 1.0
                a_new = float(np.sum(weights * (ys - xs)) / weights.sum())
            r = ys - (a_new + b_new * xs)
            scale = max(float(np.median(np.abs(r - np.median(r)))) * mad_to_sigma, sigma_floor)
            if scale <= 0:
                scale = float(np.sqrt(np.mean(r ** 2))) if len(r) else 0.0
            new_weights = np.ones(len(r)) if scale <= 0 else np.minimum(1.0, k * scale / np.maximum(np.abs(r), 1e-300))
            done = abs(a_new - a) <= 1e-12 * (1 + abs(a)) and abs(b_new - b) <= 1e-15 and np.allclose(new_weights, weights)
            a, b, weights = a_new, b_new, new_weights
            if done:
                break
        residual = y - (a + b * (x - x0))
        sigma = max(float(np.median(np.abs(residual[inliers] - np.median(residual[inliers])))) * mad_to_sigma,
                    sigma_floor) if inliers.any() else sigma_floor
        floor = sigma if sigma > 0 else float(np.sqrt(np.mean(residual[inliers] ** 2))) if inliers.any() else 0.0
        new_inliers = np.abs(residual) <= reject * floor if floor > 0 else np.ones(len(x), bool)
        if np.array_equal(new_inliers, inliers) or new_inliers.sum() < (2 if fit_slope else 1):
            break
        inliers = new_inliers
    return {"a": a - b * x0, "b": b, "sigma": sigma, "inliers": inliers}


# ----------------------------------------------------------------------------- markers

def measured_markers(log: MachineLog, key: str) -> tuple[np.ndarray, np.ndarray]:
    """(times, keys) where the measured ``program.<key>`` channel changes value."""
    channel = log.channels.get(f"program.{key}")
    if channel is None:
        return np.zeros(0), np.zeros(0, dtype=object)
    data = channel.finite()
    if len(data) == 0:
        return np.zeros(0), np.zeros(0, dtype=object)
    values = data.values
    if data.numeric:
        changed = np.concatenate([[True], values[1:] != values[:-1]])
        keys = np.asarray(values[changed], dtype=np.float64)
    else:
        normal = np.array([_normal_block(v) for v in values], dtype=object)
        changed = np.concatenate([[True], normal[1:] != normal[:-1]])
        keys = normal[changed]
    return data.t_s[changed], keys


def _normal_block(text) -> str:
    return " ".join(str(text).upper().split())


def _sim_keys(markers: list[Marker], key: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([m.t_s for m in markers])
    if key == "line":
        return times, np.array([float(m.line) for m in markers])
    if key == "sequence":
        keep = np.array([m.sequence is not None for m in markers], bool)
        return times[keep], np.array([float(m.sequence) for m in markers if m.sequence is not None])
    return times, np.array([_normal_block(m.block) for m in markers], dtype=object)


def _poll_interval(log: MachineLog, times: np.ndarray) -> float:
    """The polling interval of the log when the marker times lie on its sampling grid (else 0)."""
    if len(times) < 3:
        return 0.0
    best = 0.0
    for channel in log.channels.values():
        if channel.kind != "sample" or not channel.numeric or len(channel) < 10:
            continue
        dt = channel.median_dt_s()
        if not math.isfinite(dt) or dt <= 0:
            continue
        steps = np.diff(channel.t_s)
        if np.mean(np.abs(steps - dt) <= 1e-3 * dt) < 0.9:
            continue                                   # not a periodic poll
        phase = (times - channel.t_s[0]) / dt
        on_grid = np.abs(phase - np.round(phase)) <= 1e-3
        if np.mean(on_grid) >= 0.9:
            best = max(best, dt)
    return best


def align_markers(sim_markers: list[Marker], log: MachineLog, *, key: str | None = None,
                  poll_interval_s: float | None = None) -> Alignment | None:
    """Alignment from block markers, or None when no key gives at least one match."""
    p = align_params()
    keys = [key] if key else ["line", "sequence", "block"]
    for name in keys:
        t_meas, k_meas = measured_markers(log, name)
        t_sim, k_sim = _sim_keys(sim_markers, name)
        if len(t_meas) == 0 or len(t_sim) == 0:
            continue
        pairs = _match(t_meas, k_meas, t_sim, k_sim, p)
        if pairs is None or len(pairs[0]) == 0:
            continue
        i_meas, j_sim = pairs
        x, y = t_meas[i_meas], t_sim[j_sim]
        span = float(x.max() - x.min()) if len(x) else 0.0
        fit_slope = len(x) >= int(p["marker_min_for_scale"]) and span >= float(p["scale_min_span_s"])
        poll = _poll_interval(log, t_meas) if poll_interval_s is None else float(poll_interval_s)
        sim_step = float(np.min(np.diff(np.unique(t_sim)))) if len(np.unique(t_sim)) > 1 else 0.0
        fit = robust_line(x, y, fit_slope=fit_slope, p=p, sigma_floor=max(poll, sim_step) / math.sqrt(12.0))
        if abs(fit["b"] - 1.0) > float(p["max_scale_deviation"]):
            raise InputError(f"the markers give a time scale of {fit['b']:.6f}, more than "
                             f"{float(p['max_scale_deviation']):g} from 1 (mdata-align max_scale_deviation): the "
                             f"measured and simulated programs do not match")
        offset = fit["a"] + fit["b"] * poll / 2.0
        inl = fit["inliers"]
        return Alignment(offset, fit["b"], "markers", {
            "key": name, "measured_markers": int(len(t_meas)), "simulated_markers": int(len(t_sim)),
            "matched": int(len(x)), "inliers": int(inl.sum()), "residual_sigma_s": fit["sigma"],
            "poll_interval_s": poll, "poll_correction_s": fit["b"] * poll / 2.0, "scale_fitted": bool(fit_slope),
            "span_s": span, "pairs": [[float(a), float(b)] for a, b, ok in zip(x, y, inl) if ok]})
    return None


def _match(t_meas, k_meas, t_sim, k_sim, p) -> tuple[np.ndarray, np.ndarray] | None:
    """Pairs (measured index, simulated index) of equal keys, in time order, near one offset."""
    # 1. offset votes from every equal-key pair (capped per key so a hot loop line cannot dominate the cost)
    votes = []
    sim_by_key: dict = {}
    for j, key in enumerate(k_sim):
        sim_by_key.setdefault(key, []).append(j)
    for i, key in enumerate(k_meas):
        js = sim_by_key.get(key)
        if js:
            votes.append(t_sim[np.asarray(js[:2000])] - t_meas[i])
    if not votes:
        return None
    d = np.concatenate(votes)
    gaps = np.diff(np.unique(t_sim))
    sim_gap = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 0.0
    meas_gaps = np.diff(np.unique(t_meas))
    poll = float(np.percentile(meas_gaps[meas_gaps > 0], 5)) if np.any(meas_gaps > 0) else 0.0
    width = max(sim_gap / 2.0, poll / 2.0, 1e-6)
    d_sorted = np.sort(d)
    # densest window of +-width: counts by two pointers
    right = np.searchsorted(d_sorted, d_sorted + 2 * width, side="right")
    counts = right - np.arange(len(d_sorted))
    best = int(np.argmax(counts))
    window_votes = d_sorted[best:right[best]]
    offset = float(np.median(window_votes))
    # 2. greedy matching in time order to the nearest simulated start with the same key
    tolerance = max(4.0 * width, 3.0 * float(np.std(window_votes)) if len(window_votes) > 1 else 0.0, 1e-3)
    i_list, j_list = [], []
    last_j = -1
    a, b = offset, 1.0
    for i in np.argsort(t_meas, kind="stable"):
        js = sim_by_key.get(k_meas[i])
        if not js:
            continue
        js_arr = np.asarray(js)
        js_arr = js_arr[js_arr > last_j]
        if len(js_arr) == 0:
            continue
        predicted = a + b * t_meas[i]
        error = np.abs(t_sim[js_arr] - predicted)
        k = int(np.argmin(error))
        if error[k] > tolerance + abs(b - 1.0) * abs(t_meas[i]):
            continue
        i_list.append(int(i))
        j_list.append(int(js_arr[k]))
        last_j = int(js_arr[k])
        if len(i_list) >= 8 and len(i_list) % 8 == 0:
            span = t_meas[i_list[-1]] - t_meas[i_list[0]]
            if span >= float(p["scale_min_span_s"]):
                fit = robust_line(t_meas[i_list], t_sim[j_list], fit_slope=True, p=p)
                if abs(fit["b"] - 1.0) <= float(p["max_scale_deviation"]):
                    a, b = fit["a"], fit["b"]
    return np.asarray(i_list, int), np.asarray(j_list, int)


# ----------------------------------------------------------------------------- cross-correlation

def _grid_values(channel: Channel, t0: float, dt: float, n: int, max_gap: float | None) -> np.ndarray:
    return channel.at(t0 + dt * np.arange(n), max_gap_s=max_gap)


def masked_ncc(x: np.ndarray, y: np.ndarray, min_overlap: int) -> tuple[np.ndarray, np.ndarray]:
    """(lags, coefficient): c(L) = corr(x[k], y[k - L]) over the samples valid in both (NaN = gap),
    every lag with at least ``min_overlap`` common samples (others NaN). Padfield's six-FFT form."""
    mx, my = np.isfinite(x).astype(float), np.isfinite(y).astype(float)
    xv, yv = np.where(mx > 0, x, 0.0), np.where(my > 0, y, 0.0)
    n = len(x) + len(y) - 1
    size = 1 << int(math.ceil(math.log2(max(n, 2))))

    def corr(a, b):
        # c[L] = sum_k a[k] b[k - L] for L = -(len(b)-1) .. len(a)-1
        fa = np.fft.rfft(a, size)
        fb = np.fft.rfft(b[::-1], size)
        full = np.fft.irfft(fa * fb, size)[: n]
        return full                                  # index L + len(b) - 1

    count = corr(mx, my)
    sxy = corr(xv, yv)
    sx = corr(xv, my)
    sy = corr(mx, yv)
    sxx = corr(xv * xv, my)
    syy = corr(mx, yv * yv)
    lags = np.arange(n) - (len(y) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        count_r = np.round(count)
        cov = sxy - sx * sy / count_r
        vx = sxx - sx * sx / count_r
        vy = syy - sy * sy / count_r
        c = cov / np.sqrt(vx * vy)
    c[(count_r < min_overlap) | ~np.isfinite(c) | (vx <= 0) | (vy <= 0)] = np.nan
    return lags, np.clip(c, -1.0, 1.0)


def _peak(lags: np.ndarray, c: np.ndarray, ratio_limit: float) -> dict:
    valid = np.isfinite(c)
    if not valid.any():
        raise InputError("cross-correlation: the series do not overlap enough (mdata-align xcorr_min_overlap_frac)")
    k = int(np.nanargmax(np.where(valid, c, -np.inf)))
    peak = float(c[k])
    shift = 0.0
    if 0 < k < len(c) - 1 and np.isfinite(c[k - 1]) and np.isfinite(c[k + 1]):
        denom = c[k - 1] - 2 * c[k] + c[k + 1]
        if denom < 0:
            shift = float(0.5 * (c[k - 1] - c[k + 1]) / denom)
    lo = k
    while lo > 0 and np.isfinite(c[lo - 1]) and c[lo - 1] > 0.5 * peak:
        lo -= 1
    hi = k
    while hi < len(c) - 1 and np.isfinite(c[hi + 1]) and c[hi + 1] > 0.5 * peak:
        hi += 1
    outside = np.ones(len(c), bool)
    outside[max(0, lo - (k - lo)): min(len(c), hi + (hi - k) + 1)] = False
    second = float(np.nanmax(np.where(outside & valid, c, np.nan))) if (outside & valid).any() else -1.0
    ratio = second / peak if peak > 0 else 1.0
    return {"lag": float(lags[k]) + shift, "peak": peak, "second": second, "ambiguous": bool(ratio >= ratio_limit),
            "ratio": float(ratio), "lobe_samples": int(hi - lo + 1)}


def align_xcorr(sim: Channel, measured: Channel, *, scale_search: bool = True) -> Alignment:
    """Alignment from the cross-correlation of a simulated and a measured signal (the same physical
    quantity up to an affine transform: shaft power and spindle load, power and power)."""
    p = align_params()
    sim, measured = sim.finite(), measured.finite()
    if len(sim) < 4 or len(measured) < 4:
        raise InputError("cross-correlation needs at least four samples of both signals")
    dt_meas = measured.median_dt_s()
    dt_sim = sim.median_dt_s()
    span_sim = float(sim.t_s[-1] - sim.t_s[0])
    span_meas = float(measured.t_s[-1] - measured.t_s[0])
    dt = max(min(dt_meas, dt_sim), (span_sim + span_meas) / float(p["xcorr_max_points"]))
    gap = float(p["resample_max_gap_factor"])
    n_sim = int(span_sim / dt) + 1
    n_meas = int(span_meas / dt) + 1
    x = _grid_values(sim, sim.t_s[0], dt, n_sim, gap * dt_sim)
    y = _grid_values(measured, measured.t_s[0], dt, n_meas, gap * dt_meas)
    min_overlap = max(4, int(float(p["xcorr_min_overlap_frac"]) * min(np.isfinite(x).sum(), np.isfinite(y).sum())))
    lags, c = masked_ncc(x, y, min_overlap)
    peak = _peak(lags, c, float(p["xcorr_ambiguity_ratio"]))
    if peak["peak"] < float(p["xcorr_min_correlation"]):
        raise InputError(f"cross-correlation peak {peak['peak']:.2f} is below {float(p['xcorr_min_correlation']):g} "
                         f"(mdata-align xcorr_min_correlation): the measured and simulated signals do not show the "
                         f"same process; give markers or offset_s")
    offset = float(sim.t_s[0] - measured.t_s[0] + peak["lag"] * dt)
    quality = {"signal_sim": sim.name, "signal_measured": measured.name, "grid_dt_s": dt, **peak,
               "scale_fitted": False}
    scale = 1.0
    overlap_s = min(span_meas, span_sim)
    windows = int(p["xcorr_scale_windows"])
    if scale_search and overlap_s >= float(p["scale_min_span_s"]) and windows >= 2:
        local = _local_offsets(sim, measured, offset, dt, windows, p)
        if len(local) >= 2:
            centres = np.array([w[0] for w in local])
            offsets = np.array([w[1] for w in local])
            weights = np.array([max(w[2], 1e-3) for w in local])
            A = np.column_stack([np.ones(len(centres)), centres])
            W = np.sqrt(weights)[:, None]
            coef, *_ = np.linalg.lstsq(A * W, offsets * W[:, 0], rcond=None)
            b = 1.0 + float(coef[1])
            if abs(b - 1.0) <= float(p["max_scale_deviation"]):
                scale = b
                offset = float(coef[0])
                quality.update({"scale_fitted": True, "windows": [[float(c_), float(o), float(w)] for c_, o, w in local]})
    return Alignment(offset, scale, "xcorr", quality)


def _local_offsets(sim: Channel, measured: Channel, offset: float, dt: float, windows: int, p) -> list:
    t0, t1 = float(measured.t_s[0]), float(measured.t_s[-1])
    edges = np.linspace(t0, t1, windows + 1)
    margin = float(p["max_scale_deviation"]) * (t1 - t0) + 10 * dt
    out = []
    gap = float(p["resample_max_gap_factor"])
    for a, b in zip(edges[:-1], edges[1:]):
        n = int((b - a) / dt) + 1
        y = _grid_values(measured, a, dt, n, gap * measured.median_dt_s())
        start = a + offset - margin
        m = int((b - a + 2 * margin) / dt) + 1
        x = _grid_values(sim, start, dt, m, gap * sim.median_dt_s())
        if np.isfinite(y).sum() < 8 or np.isfinite(x).sum() < 8:
            continue
        lags, c = masked_ncc(x, y, max(8, int(0.8 * np.isfinite(y).sum())))
        try:
            peak = _peak(lags, c, 1.0)
        except InputError:
            continue
        if peak["peak"] < float(p["xcorr_min_correlation"]):
            continue
        local = start - a + peak["lag"] * dt
        out.append((0.5 * (a + b), local, peak["peak"]))
    return out


# ----------------------------------------------------------------------------- one call

def align(sim_log: MachineLog, measured: MachineLog, sim_markers: list[Marker] | None = None, *,
          method: str = "auto", offset_s: float | None = None, scale: float | None = None,
          key: str | None = None, poll_interval_s: float | None = None, sim_signal: Channel | None = None,
          measured_signal: str | None = None) -> Alignment:
    """``method``: auto (markers, else cross-correlation), markers, xcorr, given (offset_s [, scale]), none.

    ``sim_signal`` / ``measured_signal``: the pair to cross-correlate (default: the first of spindle load,
    power, torque, current, speed that both logs have); ``sim_signal`` is typically the measured quantity
    predicted with the nominal machine parameters (idle power included)."""
    if method == "given" or offset_s is not None:
        return Alignment(float(offset_s or 0.0), float(scale or 1.0), "given", {})
    if method == "none":
        return Alignment(0.0, 1.0, "none", {})
    if method in ("auto", "markers") and sim_markers:
        found = align_markers(sim_markers, measured, key=key, poll_interval_s=poll_interval_s)
        if found is not None:
            return found
        if method == "markers":
            raise InputError("no measured block marker (program.line, program.sequence, program.block) matches "
                             "a simulated block")
    if method in ("auto", "xcorr"):
        if sim_signal is not None and measured_signal is not None:
            if measured_signal not in measured:
                raise InputError(f"the machine data has no {measured_signal} to cross-correlate")
            return align_xcorr(sim_signal, measured.channels[measured_signal])
        pairs = [("spindle.load_frac", "spindle.load_frac"), ("spindle.shaft_power_W", "spindle.power_W"),
                 ("spindle.shaft_power_W", "spindle.load_frac"), ("spindle.shaft_power_W", "spindle.torque_Nm"),
                 ("spindle.shaft_power_W", "spindle.current_A"), ("spindle.speed_rad_s", "spindle.speed_rad_s")]
        for sim_name, meas_name in pairs:
            if sim_name in sim_log and meas_name in measured:
                return align_xcorr(sim_log.channels[sim_name], measured.channels[meas_name])
        raise InputError("no signal both the run and the machine data have for a cross-correlation (spindle load, "
                         "power, torque, current or speed)")
    raise InputError(f"alignment method {method!r}: use auto, markers, xcorr, given or none")


# ----------------------------------------------------------------------------- resampling and overlays

def resample(channel: Channel, t_sim: np.ndarray, alignment: Alignment) -> np.ndarray:
    """Measured channel values at simulated times (gaps longer than resample_max_gap_factor x its
    sampling interval stay NaN)."""
    gap = float(align_params()["resample_max_gap_factor"])
    dt = channel.median_dt_s()
    return channel.at(alignment.to_meas(t_sim), max_gap_s=gap * dt if math.isfinite(dt) else None)


def error_metrics(measured, predicted) -> dict:
    """n, bias (mean of predicted - measured), RMSE, MAE, max |error|, NRMSE (RMSE / range of the
    measured), R^2 and Pearson r over the samples where both are finite."""
    m = np.asarray(measured, dtype=np.float64)
    q = np.asarray(predicted, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(q)
    m, q = m[ok], q[ok]
    n = int(len(m))
    if n == 0:
        return {"n": 0, "bias": None, "rmse": None, "mae": None, "max_abs": None, "nrmse": None, "r2": None, "r": None}
    e = q - m
    rmse = float(np.sqrt(np.mean(e ** 2)))
    span = float(m.max() - m.min())
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    r = None
    if n > 1 and np.std(m) > 0 and np.std(q) > 0:
        r = float(np.corrcoef(m, q)[0, 1])
    return {"n": n, "bias": float(e.mean()), "rmse": rmse, "mae": float(np.mean(np.abs(e))),
            "max_abs": float(np.max(np.abs(e))), "nrmse": rmse / span if span > 0 else None,
            "r2": 1.0 - float(np.sum(e ** 2)) / ss_tot if ss_tot > 0 else None, "r": r}


def reduce_m4(t: np.ndarray, columns: dict[str, np.ndarray], max_points: int) -> tuple[np.ndarray, dict, bool]:
    """At most ``max_points`` per series: per bucket the first, last, minimum and maximum samples of the
    first column (indices shared by every column), so peaks survive."""
    n = len(t)
    if n <= max_points:
        return t, columns, False
    buckets = max(1, max_points // 4)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    key = next(iter(columns.values()))
    keep = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = key[a:b]
        idx = {a, b - 1}
        if np.isfinite(seg).any():
            idx.add(a + int(np.nanargmin(seg)))
            idx.add(a + int(np.nanargmax(seg)))
        keep.extend(sorted(idx))
    keep = np.asarray(sorted(set(keep)), int)
    return t[keep], {k: v[keep] for k, v in columns.items()}, True


def block_comparison(sim_markers: list[Marker], alignment: Alignment) -> list[dict]:
    """Per matched block: simulated and measured start and duration (from consecutive matched markers)."""
    pairs = alignment.quality.get("pairs") or []
    out = []
    by_time = {round(m.t_s, 9): m for m in sim_markers}
    for (tm0, ts0), (tm1, ts1) in zip(pairs[:-1], pairs[1:]):
        marker = by_time.get(round(ts0, 9))
        measured_duration = alignment.scale * (tm1 - tm0)
        out.append({"line": marker.line if marker else None, "block": marker.block if marker else None,
                    "sim_start_s": ts0, "sim_duration_s": ts1 - ts0,
                    "measured_start_s": float(alignment.to_sim(tm0)), "measured_duration_s": measured_duration,
                    "error_s": measured_duration - (ts1 - ts0)})
    return out


def measured_cycle_time(log: MachineLog, alignment: Alignment, sim_markers: list[Marker]) -> dict:
    """Cycle time from the execution state (first ACTIVE to the next non-ACTIVE state), else from the
    first and last matched markers (then compared with the simulated span between the same blocks)."""
    execution = log.channels.get("program.execution")
    if execution is not None and not execution.numeric:
        values = [str(v).upper() if v is not None else "" for v in execution.values]
        start = next((i for i, v in enumerate(values) if v == "ACTIVE"), None)
        if start is not None:
            end = next((i for i in range(start + 1, len(values)) if values[i] not in ("ACTIVE", "")), None)
            if end is not None:
                duration = alignment.scale * float(execution.t_s[end] - execution.t_s[start])
                return {"method": "execution", "measured_s": duration,
                        "measured_start_s": float(alignment.to_sim(execution.t_s[start]))}
    pairs = alignment.quality.get("pairs") or []
    if len(pairs) >= 2:
        return {"method": "markers", "measured_s": alignment.scale * (pairs[-1][0] - pairs[0][0]),
                "simulated_between_markers_s": pairs[-1][1] - pairs[0][1]}
    return {"method": None, "measured_s": None}


def overlay(sim_log: MachineLog, measured: MachineLog, alignment: Alignment, pairs: list[dict], *,
            dt_s: float | None = None, t0_s: float | None = None, t1_s: float | None = None,
            sim_markers: list[Marker] | None = None, extra: dict | None = None,
            simulated_cycle_s: float | None = None) -> dict:
    """Overlay data for the UI and report: per pair ``{name, sim, measured, unit, transform}``
    (``sim`` a channel of the simulated log or a callable t_sim -> values, ``measured`` a channel of
    the machine log), the series on a common grid in simulated time, and error metrics.

    ``extra``: {name: callable(t_sim) -> values} added as further columns of every series (the
    calibrated prediction)."""
    p = align_params()
    starts, ends = [], []
    for pair in pairs:
        sim = pair["sim"]
        if isinstance(sim, str) and sim in sim_log.channels:
            c = sim_log.channels[sim].finite()
            if len(c):
                starts.append(c.t_s[0])
                ends.append(c.t_s[-1])
        m = measured.channels.get(pair["measured"])
        if m is not None and len(m.finite()):
            mm = m.finite()
            starts.append(float(alignment.to_sim(mm.t_s[0])))
            ends.append(float(alignment.to_sim(mm.t_s[-1])))
    if not starts:
        raise InputError("nothing to overlay: neither series has data")
    t0 = float(t0_s if t0_s is not None else max(min(starts), 0.0) if sim_log.channels else min(starts))
    t1 = float(t1_s if t1_s is not None else max(ends))
    if dt_s is None:
        candidates = [measured.channels[p_["measured"]].median_dt_s() for p_ in pairs if p_["measured"] in measured.channels]
        candidates = [c for c in candidates if math.isfinite(c) and c > 0]
        dt_s = min(candidates) if candidates else (t1 - t0) / 1000.0
    n = max(2, int(round((t1 - t0) / dt_s)) + 1)
    grid = t0 + dt_s * np.arange(n)
    series, metrics = [], {}
    for pair in pairs:
        sim = pair["sim"]
        simulated = sim(grid) if callable(sim) else (sim_log.channels[sim].at(grid) if sim in sim_log.channels
                                                       else np.full(n, np.nan))
        m = measured.channels.get(pair["measured"])
        meas = resample(m, grid, alignment) if m is not None else np.full(n, np.nan)
        transform = pair.get("transform")
        if transform is not None:
            meas = transform(meas)
        columns = {"measured": meas, "simulated": np.asarray(simulated, dtype=np.float64)}
        for name, function in (extra or {}).get(pair["name"], {}).items():
            columns[name] = np.asarray(function(grid), dtype=np.float64)
        metrics[pair["name"]] = {k: error_metrics(meas, v) for k, v in columns.items() if k != "measured"}
        t_red, reduced, was_reduced = reduce_m4(grid, columns, int(p["overlay_max_points"]))
        series.append({"name": pair["name"], "unit": pair.get("unit", ""),
                       "sim_channel": sim if isinstance(sim, str) else pair.get("sim_label", "model"),
                       "measured_channel": pair["measured"], "t_s": t_red,
                       **{k: v for k, v in reduced.items()}, "reduced": "m4" if was_reduced else "none"})
    document = {"format": FORMAT, "version": 1, "time_base": "simulation", "alignment": alignment.to_dict(),
                "t0_s": t0, "t1_s": t1, "dt_s": dt_s, "series": series, "metrics": metrics}
    if sim_markers:
        document["blocks"] = block_comparison(sim_markers, alignment)
        cycle = measured_cycle_time(measured, alignment, sim_markers)
        error = None
        if cycle.get("method") == "execution" and simulated_cycle_s:
            error = cycle["measured_s"] - simulated_cycle_s
        document["cycle_time"] = {**cycle, "simulated_s": simulated_cycle_s, "error_s": error,
                                  "error_frac": error / simulated_cycle_s if error is not None else None}
    return _plain(document)


__all__ = ["Alignment", "align", "align_markers", "align_xcorr", "masked_ncc", "robust_line", "measured_markers",
           "resample", "overlay", "error_metrics", "reduce_m4", "block_comparison", "measured_cycle_time", "FORMAT"]
