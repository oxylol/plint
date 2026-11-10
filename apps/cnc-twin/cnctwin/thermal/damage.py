"""Thermal damage of the machined surface (docs/models/thermal.md 3.4 "Thermal damage").

The machined surface's temperature history (surfacetemp.py: the work's share of the shear-zone, flank-contact and
edge-ploughing heat as moving sources over the new surface, on the bulk and feed-scale temperature) is judged with
the criterion of each kind of damage, from the material database (``thermal_damage``):

* **white layer** (rehardening, error): austenite must form. The start of austenite formation on continuous heating
  rises with the heating rate (continuous-heating TTA diagrams, Orlich & Pietrzeniuk 1976; AISI 4140 to 1e4 K/s:
  Miokovic et al. 2004, 2005): Ac1(rate) is interpolated in log(rate) between the data sheet's Ac1
  (``white_layer_degC``, slow heating) and ``white_layer_fast_degC`` at
  ``white_layer_fast_rate_kelvin_per_second`` where the material gives it (clamped beyond: no extrapolation to
  faster rates, which keeps the lower threshold), else the data sheet's Ac1 alone. The peak temperature at the minimum affected depth (``min_depth_m``) must exceed Ac1 at
  the heating rate there.
* **over-tempering** (soft layer of a hardened and tempered steel, warning): time at temperature, Hollomon & Jaffe's
  (1945) tempering parameter P = T (C + log10 t[h]) accumulated over the history by the additivity rule (at each
  sample the accumulated P is turned into the equivalent time at the sample's temperature) at the minimum depth,
  against the tempering the material received (``overtemper_degC`` held ``tempering_hold_min``).
* **burn** (temper / oxide colours, warning): the colour table's temperature (``burn_degC``) is a tempering-practice
  temperature, and tempering temperature and time are interchangeable: the same parameter at the surface against
  ``burn_degC`` held ``burn_hold_s`` (thermal-damage.yaml).
* **melt / smear** (error): the peak at the minimum depth above the solidus (``melt_smear_degC``).

The history is evaluated where a cheap screen passes: the surface peak above the lowest temperature at which any
criterion can hold (a tempering parameter cannot exceed the reference while T stays below the reference temperature
for less than the reference time). One ``thermal_damage`` finding per (operation, workpiece, kind, program block),
with the location in S and the tick; repeats in the same block are counted in the report. Materials without a
history (legacy callers of :meth:`DamageMonitor.check`) fall back to the threshold crossing of the given temperature.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from ..core.findings import Category, Severity
from ..db.models import params
from ..units import ZERO_DEGC_K
from .inputs import plain, raw

KINDS = (
    # kind, database key, severity, what happens
    ("burn", "burn_degC", Severity.WARNING, "temper / oxide colours (burn marks) on the surface"),
    ("overtemper", "overtemper_degC", Severity.WARNING, "the hardened surface over-tempers (soft layer)"),
    ("white_layer", "white_layer_degC", Severity.ERROR, "the surface rehardens to a brittle white layer"),
    ("melt_smear", "melt_smear_degC", Severity.ERROR, "the surface melts or smears"),
)
_SECONDS_PER_HOUR = 3600.0


@functools.lru_cache(maxsize=1)
def damage_params() -> dict:
    return dict(params("thermal-damage"))


@dataclass(frozen=True)
class Threshold:
    kind: str
    temp_K: float
    severity: Severity
    effect: str
    hold_s: float = 0.0              # burn / overtemper: the reference time at temp_K (0: threshold crossing)
    fast_K: float = 0.0              # white layer: Ac1 at fast_rate (0: none)
    fast_rate: float = 0.0               # its heating rate (K/s)


def _float(value) -> float | None:
    value = plain(value)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def thresholds(material) -> list[Threshold]:
    """Criteria of a material (resolved SI view or raw mapping), lowest reference temperature first."""
    data = raw(material) or {}
    section = data.get("thermal_damage") if isinstance(data, dict) else None
    out = []
    if not isinstance(section, dict):
        return out
    p = damage_params()
    for kind, key, severity, effect in KINDS:
        value = _float(section.get(key))
        if value is None:
            continue
        temp_K = value + ZERO_DEGC_K
        hold = 0.0
        fast = 0.0
        rate = 0.0
        if kind == "overtemper":
            minutes = _float(section.get("tempering_hold_min"))
            hold = 60.0 * minutes if minutes and minutes > 0.0 else float(p["tempering_hold_default_s"])
        elif kind == "burn":
            hold = float(p["burn_hold_s"])
        elif kind == "white_layer":
            f = _float(section.get("white_layer_fast_degC"))
            r = _float(section.get("white_layer_fast_rate_kelvin_per_second"))
            if f is not None and r is not None and r > float(p["slow_heating_rate_kelvin_per_second"]):
                fast, rate = f + ZERO_DEGC_K, r
        out.append(Threshold(kind, temp_K, severity, effect, hold, fast, rate))
    return sorted(out, key=lambda t: t.temp_K)


def hj_constant(material) -> float:
    data = raw(material) or {}
    section = data.get("thermal_damage") if isinstance(data, dict) else None
    value = _float(section.get("hollomon_jaffe_C")) if isinstance(section, dict) else None
    return float(value) if value and value > 0.0 else float(damage_params()["hollomon_jaffe_C"])


# ----------------------------------------------------------------------------------------------- criteria
def tempering_parameter(times_s, temps_K, C: float, hold_K: float = 0.0, hold_s: float = 0.0) -> float:
    """Hollomon-Jaffe parameter P = T (C + log10 t[h]) of a history (K), accumulated by additivity: each interval at
    its mean temperature adds its duration to the time equivalent, at that temperature, of the P so far; ``hold_K``
    for ``hold_s`` is appended (the bulk and feed-scale temperature that stays after the edge has passed)."""
    t = np.asarray(times_s, dtype=float)
    T = np.asarray(temps_K, dtype=float)
    P = None
    segments = [(0.5 * (T[i] + T[i + 1]), t[i + 1] - t[i]) for i in range(t.size - 1) if t[i + 1] > t[i]]
    if hold_s > 0.0 and hold_K > 0.0:
        segments.append((hold_K, hold_s))
    for temp, dt in segments:
        if temp <= 0.0 or dt <= 0.0:
            continue
        t_eq_h = 10.0 ** (P / temp - C) if P is not None else 0.0
        P = temp * (C + math.log10(t_eq_h + dt / _SECONDS_PER_HOUR))
    return -math.inf if P is None else P


def reference_parameter(temp_K: float, hold_s: float, C: float) -> float:
    return temp_K * (C + math.log10(max(hold_s, 1e-12) / _SECONDS_PER_HOUR))


def heating_rate(times_s, temps_K) -> float:
    """Mean heating rate (K/s) between rise_fraction and 1 - rise_fraction of the rise to the peak."""
    t = np.asarray(times_s, dtype=float)
    T = np.asarray(temps_K, dtype=float)
    j = int(np.argmax(T))
    base = float(T[0])
    rise = float(T[j]) - base
    if j == 0 or rise <= 0.0:
        return 0.0
    f = float(damage_params()["heating_rise_fraction"])

    def crossing(level):
        for i in range(j):
            if T[i] <= level < T[i + 1] or (T[i] < level <= T[i + 1]):
                w = (level - T[i]) / (T[i + 1] - T[i])
                return t[i] + w * (t[i + 1] - t[i])
        return t[0]

    lo, hi = crossing(base + f * rise), crossing(base + (1.0 - f) * rise)
    return (1.0 - 2.0 * f) * rise / (hi - lo) if hi > lo else math.inf


def ac1_at_rate(limit: Threshold, rate_K_s: float) -> float:
    """Austenite start (K) at a heating rate: log-linear between the slow data-sheet value and the fast point."""
    if limit.fast_K <= 0.0 or limit.fast_rate <= 0.0 or not rate_K_s > 0.0:
        return limit.temp_K
    slow = float(damage_params()["slow_heating_rate_kelvin_per_second"])
    if rate_K_s <= slow:
        return limit.temp_K
    w = min(math.log10(rate_K_s / slow) / math.log10(limit.fast_rate / slow), 1.0)
    return limit.temp_K + w * (limit.fast_K - limit.temp_K)


@dataclass
class Verdict:
    kind: str
    damaged: bool
    metric: float                    # peak K (white layer, melt) or tempering parameter K (burn, overtemper)
    limit: float                     # the criterion's value in the same unit
    depth_m: float = 0.0             # deepest sampled depth where the criterion holds
    details: dict = field(default_factory=dict)


def screen_K(limit: Threshold) -> float:
    """Lowest surface peak at which the criterion can hold (all criteria need T above the reference temperature,
    tempering within less than the reference time; histories here last seconds at most)."""
    return limit.temp_K


def judge(limit: Threshold, hist, C: float, hold_K: float = 0.0, hold_s: float = 0.0) -> Verdict:
    """Apply one criterion to a surfacetemp.SurfaceHistory."""
    zmin = float(damage_params()["min_depth_m"])
    depths = np.asarray(hist.depth_m, dtype=float)

    def evaluate(z):
        series = hist.at_depth(z)
        if limit.kind in ("burn", "overtemper"):
            P = tempering_parameter(hist.time_s, series, C, hold_K, hold_s)
            return P, reference_parameter(limit.temp_K, limit.hold_s, C), {}
        peak = float(np.max(series))
        if limit.kind == "white_layer":
            rate = heating_rate(hist.time_s, series)
            return peak, ac1_at_rate(limit, rate), {"heating_rate_K_s": rate}
        return peak, limit.temp_K, {}

    z_eval = 0.0 if limit.kind == "burn" else zmin
    metric, crit, info = evaluate(z_eval)
    damaged = metric > crit
    depth = 0.0
    if damaged:
        for z in depths:
            if z >= z_eval:
                m, c, _ = evaluate(float(z))
                if m > c:
                    depth = float(z)
    info.update(depth_evaluated_m=z_eval, surface_peak_K=float(np.max(hist.temp_K[0])))
    return Verdict(limit.kind, bool(damaged), float(metric), float(crit), depth, info)


class DamageMonitor:
    """Raises thermal_damage findings once per (op, workpiece, kind, block)."""

    def __init__(self, material_name: str, limits: list[Threshold], hj_C: float | None = None):
        self.material_name = material_name
        self.limits = list(limits)
        self.hj_C = float(hj_C) if hj_C else float(damage_params()["hollomon_jaffe_C"])
        self.seen: set[tuple[int, int, str, int]] = set()
        self.counts: dict[str, int] = {}
        self.max_K: dict[str, float] = {}

    def screen(self) -> float:
        """The lowest surface peak at which any criterion can hold (inf without criteria)."""
        return min((screen_K(t) for t in self.limits), default=math.inf)

    def verdicts(self, hist, hold_K: float = 0.0, hold_s: float = 0.0) -> list[Verdict]:
        peak = float(np.max(hist.temp_K))
        return [judge(limit, hist, self.hj_C, hold_K, hold_s) for limit in self.limits if peak > screen_K(limit)]

    def check(self, ctx, state, op: int, workpiece: int, block: int, tick: int, path: int, surface_K: float,
              quantities: dict, parents: list[int], verdicts: list[Verdict] | None = None) -> list:
        """Findings of one contact evaluation: from ``verdicts`` (the history's criteria) when given, else the
        threshold crossing of ``surface_K`` (legacy)."""
        if verdicts is None:
            verdicts = [Verdict(t.kind, surface_K > t.temp_K, surface_K, t.temp_K) for t in self.limits]
        by_kind = {t.kind: t for t in self.limits}
        raised = []
        for verdict in verdicts:
            limit = by_kind.get(verdict.kind)
            if limit is None or not verdict.damaged:
                continue
            self.counts[limit.kind] = self.counts.get(limit.kind, 0) + 1
            self.max_K[limit.kind] = max(self.max_K.get(limit.kind, 0.0), surface_K)
            key = (int(op), int(workpiece), limit.kind, int(block))
            if key in self.seen:
                continue
            self.seen.add(key)
            message = self._message(limit, verdict, surface_K)
            values = dict(quantities)
            values["surface_temp_K"] = surface_K
            values["threshold_K"] = limit.temp_K
            values["criterion_value"] = verdict.metric
            values["criterion_limit"] = verdict.limit
            values["affected_depth_m"] = verdict.depth_m
            for name, value in verdict.details.items():
                values[name] = float(value)
            raised.append(ctx.finding(state, "thermal", Category.THERMAL_DAMAGE, limit.severity, message, tick=tick,
                                      path=path, quantities=values, parents=parents))
        return raised

    def _message(self, limit: Threshold, verdict: Verdict, surface_K: float) -> str:
        name = limit.kind.replace("_", " ")
        c = lambda k: k - ZERO_DEGC_K
        if verdict.depth_m <= 0.0 and verdict.metric == surface_K and verdict.limit == limit.temp_K:
            return (f"surface temperature {c(surface_K):.0f} degC exceeds the {name} temperature "
                    f"{c(limit.temp_K):.0f} degC of {self.material_name}: {limit.effect}")
        depth_um = verdict.depth_m * 1e6
        if limit.kind in ("burn", "overtemper"):
            return (f"machined surface peak {c(surface_K):.0f} degC: tempering parameter {verdict.metric:.0f} K "
                    f"exceeds {verdict.limit:.0f} K ({c(limit.temp_K):.0f} degC for {limit.hold_s / 60.0:.0f} min) of "
                    f"{self.material_name} to {depth_um:.1f} um: {limit.effect}")
        rate = verdict.details.get("heating_rate_K_s")
        extra = f" at {rate:.2g} K/s" if rate else ""
        at_um = verdict.details.get("depth_evaluated_m", 0.0) * 1e6
        return (f"machined surface {c(verdict.metric):.0f} degC at {at_um:.1f} um depth exceeds the {name} temperature "
                f"{c(verdict.limit):.0f} degC{extra} of {self.material_name} to {depth_um:.1f} um (surface peak "
                f"{c(surface_K):.0f} degC): {limit.effect}")

    def to_meta(self) -> dict:
        return {"seen": sorted([list(k) for k in self.seen]), "counts": dict(sorted(self.counts.items())),
                "max_K": dict(sorted(self.max_K.items())), "material_name": self.material_name,
                "limits": [[t.kind, t.temp_K, t.severity.value, t.effect, t.hold_s, t.fast_K, t.fast_rate]
                           for t in self.limits], "hj_C": self.hj_C}

    @classmethod
    def from_meta(cls, meta: dict) -> "DamageMonitor":
        limits = []
        for row in meta["limits"]:
            k, t, s, e = row[:4]
            rest = [float(x) for x in row[4:7]] + [0.0] * (3 - len(row[4:7]))
            limits.append(Threshold(k, float(t), Severity(s), e, *rest))
        obj = cls(meta["material_name"], limits, meta.get("hj_C"))
        obj.seen = {(int(a), int(b), str(c), int(d)) for a, b, c, d in meta["seen"]}
        obj.counts = {str(k): int(v) for k, v in meta["counts"].items()}
        obj.max_K = {str(k): float(v) for k, v in meta["max_K"].items()}
        return obj


__all__ = ["KINDS", "Threshold", "Verdict", "DamageMonitor", "thresholds", "hj_constant", "judge",
           "tempering_parameter", "reference_parameter", "heating_rate", "ac1_at_rate", "damage_params"]
