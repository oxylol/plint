"""Tool-life criteria, finish estimate and remaining life (docs/models/wear.md section 11).

End of life is the first of (ISO 3685 / ISO 8688 and the part's own specifications):

* ``vb``        mean flank land of the most worn tooth VB_B >= VB limit (the Taylor entry's, else ISO 3685);
* ``vb_max``    largest local land VB_Bmax >= 0.6 mm;
* ``crater``    crater depth KT >= 0.06 mm + 0.3 f;
* ``notch``     notch land VB_N >= 1 mm;
* ``dimension`` the largest edge recession of the finishing elements >= the share of the tightest
  tolerance's allowed surface displacement (protocols.ToolTolerances);
* ``finish``    estimated Ra (or Rt) of the finished surface >= its specification;
* ``fatigue``   Miner sum of a critical section >= 1 (the tool breaks: also ``breakage``);
* ``breakage``  the tool broke.

Finish (Brammertz 1961 as revised by Grzesik 1996): R_t = f^2 / (8 r) + (h_min / 2)(1 + r h_min / f^2),
h_min = minimum chip thickness ratio x the worn hone radius, plus the depth of a chip on a finishing
element; Ra = R_t 8 / (18 sqrt 3) (the exact ratio of a parabolic feed-mark profile). f is the feed per
revolution (turning) or per tooth (milling: Martellotti 1941, the radius being the corner radius, or
the tool radius for peripheral finishing), r the radius of the finishing edge.

Remaining life: each criterion's quantity grows at a rate per second of cutting, averaged with an
exponential window of ``rate_window_s`` of cutting time (deterministic); the remaining cutting time is
the smallest (limit - value) / rate, the predicted replacement time the current time plus the remaining
cutting time over the recent in-cut duty.
"""

from __future__ import annotations

import math

CRITERIA = ("vb", "vb_max", "crater", "notch", "dimension", "finish", "fatigue", "breakage")
RA_PER_RT = 8.0 / (18.0 * math.sqrt(3.0))


def roughness_m(feed_m: float, radius_m: float, h_min_m: float, chip_m: float = 0.0) -> tuple[float, float]:
    """(Ra, R_t) of the feed-mark profile left by an edge of radius r at feed f (see the module doc)."""
    if feed_m <= 0.0:
        return 0.0, chip_m
    if radius_m <= 0.0:
        rt = feed_m + chip_m
        return RA_PER_RT * rt, rt
    rt = feed_m * feed_m / (8.0 * radius_m)
    if h_min_m > 0.0:
        rt += 0.5 * h_min_m * (1.0 + radius_m * h_min_m / (feed_m * feed_m))
    rt += chip_m
    return RA_PER_RT * rt, rt


def ewma(pair, sample: float, weight_s: float, window_s: float) -> list:
    """Moving average [numerator, denominator] with exponential weighting by the sample's weight (cutting
    or elapsed time); the denominator removes the start-up bias (the mean of the samples seen so far)."""
    num, den = (float(pair[0]), float(pair[1])) if pair else (0.0, 0.0)
    if weight_s <= 0.0:
        return [num, den]
    keep = math.exp(-weight_s / window_s)
    return [keep * num + (1.0 - keep) * sample, keep * den + (1.0 - keep)]


def mean(pair) -> float:
    """Value of an :func:`ewma` pair (0 before any sample)."""
    if not pair or pair[1] <= 0.0:
        return 0.0
    return float(pair[0]) / float(pair[1])


def remaining_s(values: dict, limits: dict, rates: dict) -> tuple[float, str]:
    """(cutting time left, criterion) at the recent rates (plain numbers); inf when nothing is growing."""
    best, cause = math.inf, ""
    for name in CRITERIA:
        limit = limits.get(name)
        if limit is None or not math.isfinite(limit) or limit <= 0.0:
            continue
        value = values.get(name, 0.0)
        if value >= limit:
            return 0.0, name
        rate = rates.get(name, 0.0)
        if rate > 0.0:
            left = (limit - value) / rate
            if left < best:
                best, cause = left, name
    return best, cause


def first_reached(before: dict, after: dict, limits: dict) -> tuple[str, float]:
    """(criterion, fraction of the step) of the first limit crossed during a step, the quantities growing
    linearly from ``before`` to ``after``; ties go to the earlier name in CRITERIA. ('', 1) when none."""
    best, when = "", math.inf
    for name in CRITERIA:
        limit = limits.get(name)
        if limit is None or not math.isfinite(limit) or limit <= 0.0:
            continue
        end = after.get(name, 0.0)
        if end < limit:
            continue
        start = before.get(name, 0.0)
        fraction = 0.0 if start >= limit or end <= start else (limit - start) / (end - start)
        if fraction < when:
            best, when = name, fraction
    return best, (when if best else 1.0)


def used_fraction(values: dict, limits: dict) -> float:
    best = 0.0
    for name in CRITERIA:
        limit = limits.get(name)
        if limit is None or not math.isfinite(limit) or limit <= 0.0:
            continue
        best = max(best, values.get(name, 0.0) / limit)
    return best


__all__ = ["CRITERIA", "RA_PER_RT", "roughness_m", "ewma", "mean", "remaining_s", "first_reached", "used_fraction"]
