"""Thread limits against the ISO 965 tables (6g / 6H coarse threads, M8 x 1.25 and M10 x 1.5 in particular), ASME B1.1
limits from Machinery's Handbook, the virtual pitch diameter and the gauge judgement; ISO 286 limits."""

import math

import pytest

from cnctwin.inspect import fits286
from cnctwin.inspect import threads as T

# mm: 6g major max/min, pitch max/min; 6H minor min/max, pitch min/max (ISO 965-2 values)
ISO_TABLE = {
    "M3x0.5": (2.980, 2.874, 2.655, 2.580, 2.459, 2.599, 2.675, 2.775),
    "M4x0.7": (3.978, 3.838, 3.523, 3.433, 3.242, 3.422, 3.545, 3.663),
    "M5x0.8": (4.976, 4.826, 4.456, 4.361, 4.134, 4.334, 4.480, 4.605),
    "M6x1": (5.974, 5.794, 5.324, 5.212, 4.917, 5.153, 5.350, 5.500),
    "M8x1.25": (7.972, 7.760, 7.160, 7.042, 6.647, 6.912, 7.188, 7.348),
    "M10x1.5": (9.968, 9.732, 8.994, 8.862, 8.376, 8.676, 9.026, 9.206),
    "M12x1.75": (11.966, 11.701, 10.829, 10.679, 10.106, 10.441, 10.863, 11.063),
    "M16x2": (15.962, 15.682, 14.663, 14.503, 13.835, 14.210, 14.701, 14.913),
}


def _mm(v):
    return round(v * 1e3, 3)


def test_m8_and_m10_6g_6h():
    e = T.limits(T.parse("M8x1.25-6g"))
    assert (_mm(e.major_max_m), _mm(e.major_min_m), _mm(e.d2_max_m), _mm(e.d2_min_m)) == (7.972, 7.760, 7.160, 7.042)
    assert e.deviation_m == pytest.approx(-28e-6) and e.tolerances_m["T_d2_m"] == pytest.approx(118e-6)
    assert e.tolerances_m["T_d_m"] == pytest.approx(212e-6)
    i = T.limits(T.parse("M8x1.25-6H"))
    assert (_mm(i.minor_min_m), _mm(i.minor_max_m), _mm(i.d2_min_m), _mm(i.d2_max_m)) == (6.647, 6.912, 7.188, 7.348)
    assert i.major_min_m == pytest.approx(0.008) and i.major_max_m is None
    e10 = T.limits(T.parse("M10-6g"))                                  # coarse pitch from ISO 261
    assert e10.spec.pitch_m == pytest.approx(1.5e-3)
    assert (_mm(e10.major_max_m), _mm(e10.major_min_m), _mm(e10.d2_max_m), _mm(e10.d2_min_m)) == \
        (9.968, 9.732, 8.994, 8.862)
    i10 = T.limits(T.parse("M10x1.5-6H"))
    assert (_mm(i10.minor_min_m), _mm(i10.minor_max_m), _mm(i10.d2_min_m), _mm(i10.d2_max_m)) == \
        (8.376, 8.676, 9.026, 9.206)


@pytest.mark.parametrize("name", sorted(ISO_TABLE))
def test_iso_coarse_table(name):
    e = T.limits(T.parse(name + "-6g"))
    i = T.limits(T.parse(name + "-6H"))
    got = tuple(_mm(v) for v in (e.major_max_m, e.major_min_m, e.d2_max_m, e.d2_min_m, i.minor_min_m, i.minor_max_m,
                                 i.d2_min_m, i.d2_max_m))
    assert got == ISO_TABLE[name]


def test_iso_grades_and_positions():
    # 4h6h: pitch diameter grade 4 (0.63 T_d2(6)), position h (es = 0)
    l = T.limits(T.parse("M10x1.5-4h6h"))
    assert l.deviation_m == 0.0
    assert l.tolerances_m["T_d2_m"] == pytest.approx(T.round_r40_um(0.63 * 90 * 1.5 ** 0.4 * math.sqrt(5.6 * 11.2) ** 0.1)
                                                     * 1e-6)
    assert T.limits(T.parse("M10x1.5-6e")).deviation_m == pytest.approx(-67e-6)   # -(50 + 11 x 1.5) -> R40 67
    assert T.limits(T.parse("M10x1.5-6G")).d2_min_m == pytest.approx(T.limits(T.parse("M10x1.5-6H")).d2_min_m + 32e-6)
    with pytest.raises(T.ThreadSpecError):
        T.parse("M10x1.5-6h", internal=True)
    with pytest.raises(T.ThreadSpecError):
        T.limits(T.parse("M10x1.5-2g"))                                  # no grade 2


def test_asme_unified_limits():
    def inch(v):
        return None if v is None else round(v / 0.0254, 4)

    a = T.limits(T.parse("1/4-20 UNC-2A"))
    assert (inch(a.major_max_m), inch(a.major_min_m), inch(a.d2_max_m), inch(a.d2_min_m)) == (0.2489, 0.2408, 0.2164,
                                                                                              0.2127)
    b = T.limits(T.parse("1/4-20 UNC-2B"))
    assert (inch(b.d2_min_m), inch(b.d2_max_m), inch(b.minor_min_m), inch(b.minor_max_m)) == (0.2175, 0.2224, 0.196,
                                                                                              0.207)
    a2 = T.limits(T.parse("1/2-13 UNC-2A"))
    assert (inch(a2.major_max_m), inch(a2.major_min_m), inch(a2.d2_max_m), inch(a2.d2_min_m)) == (0.4985, 0.4876,
                                                                                                  0.4485, 0.4435)
    b2 = T.limits(T.parse("1/2-13 UNC-2B"))
    assert (inch(b2.d2_min_m), inch(b2.d2_max_m), inch(b2.minor_min_m), inch(b2.minor_max_m)) == (0.45, 0.4565, 0.417,
                                                                                                  0.434)
    a3 = T.limits(T.parse("1/4-20 UNC-3A"))
    assert inch(a3.major_max_m) == 0.25 and a3.deviation_m == 0.0                 # class 3A: no allowance


def test_virtual_pitch_diameter_and_gauges():
    spec = T.parse("M10x1.5-6g")
    lim = T.limits(spec)
    alpha = math.pi / 3
    d2 = 8.94e-3
    # a pitch error of 2 um per pitch over the normal engagement and flank errors of +-10 arc minutes
    err = math.radians(10 / 60)
    v = T.virtual_pitch_diameter(spec, d2, 2e-6, [err, -err])
    P = 1.5
    le = 6.7 * P * math.sqrt(5.6 * 11.2) ** 0.2 * 1e-3
    assert v["engagement_m"] == pytest.approx(le)
    assert v["f_pitch_m"] == pytest.approx(2e-6 * le / 1.5e-3 * math.sqrt(3), rel=1e-12)
    # ISO 1502 form: 0.073 P (K1 |da1| + K2 |da2|) um with K = 2 (positive, external) and 3 (negative), minutes;
    # the exact geometry gives 0.0727 (= (H/4) 2 / sin 60 deg per arc minute / 2), ISO rounds it to 0.073
    assert v["f_flank_m"] == pytest.approx(0.073 * P * (2 * 10 + 3 * 10) * 1e-6, rel=5e-3)
    assert v["f_flank_m"] == pytest.approx(0.25 * P * math.radians(1 / 60) * (2 * 10 + 3 * 10) * 1e-3, rel=1e-12)
    good = T.judge(lim, {"simple_pitch_diameter_m": d2, "pitch_error_m": 0.0, "flank_half_angles_rad": [alpha / 2] * 2,
                         "major_diameter_m": 9.9e-3})
    assert good["pass"] and good["gauges"] == {"GO": True, "NOT GO": True}
    big = T.judge(lim, {"simple_pitch_diameter_m": 8.99e-3, "pitch_error_m": 3e-6,
                        "flank_half_angles_rad": [alpha / 2, alpha / 2], "major_diameter_m": 9.9e-3})
    assert big["gauges"]["GO"] is False and big["gauges"]["NOT GO"] is True and big["pass"] is False


def test_iso286_limits():
    def um(lo_hi, nominal):
        return tuple(round((v - nominal) * 1e6, 3) for v in lo_hi[:2])

    n = 0.020                                                    # 18-30 mm step
    assert um(fits286.limits_m(n, "H7"), n) == (0.0, 21.0)
    assert um(fits286.limits_m(n, "g6"), n) == (-20.0, -7.0)
    assert um(fits286.limits_m(n, "f7"), n) == (-41.0, -20.0)
    assert um(fits286.limits_m(n, "k6"), n) == (2.0, 15.0)
    assert um(fits286.limits_m(n, "p6"), n) == (22.0, 35.0)
    assert um(fits286.limits_m(n, "s6"), n) == (35.0, 48.0)
    assert um(fits286.limits_m(n, "js6"), n) == (-6.5, 6.5)
    assert um(fits286.limits_m(n, "K7"), n) == (-15.0, 6.0)
    assert um(fits286.limits_m(n, "M7"), n) == (-21.0, 0.0)
    assert um(fits286.limits_m(n, "N7"), n) == (-28.0, -7.0)
    assert um(fits286.limits_m(n, "N9"), n) == (-52.0, 0.0)
    assert um(fits286.limits_m(n, "P7"), n) == (-35.0, -14.0)
    assert um(fits286.limits_m(n, "P9"), n) == (-74.0, -22.0)
    assert um(fits286.limits_m(n, "S7"), n) == (-48.0, -27.0)
    assert um(fits286.limits_m(n, "G7"), n) == (7.0, 28.0)
    assert um(fits286.limits_m(n, "F8"), n) == (20.0, 53.0)
    assert um(fits286.limits_m(n, "M8"), n) == (-29.0, 4.0)
    assert um(fits286.limits_m(0.008, "H7"), 0.008) == (0.0, 15.0)
    assert um(fits286.limits_m(0.008, "h6"), 0.008) == (-9.0, 0.0)
    assert um(fits286.limits_m(0.040, "e8"), 0.040) == (-89.0, -50.0)
    assert um(fits286.limits_m(0.020, "c11"), 0.020) == (-240.0, -110.0)
    assert um(fits286.limits_m(0.002, "N7"), 0.002) == (-14.0, -4.0)            # up to 3 mm: no delta
    with pytest.raises(fits286.FitDesignationError):
        fits286.limits_m(0.6, "H7")                                             # above 500 mm
    with pytest.raises(fits286.FitDesignationError):
        fits286.limits_m(0.02, "t6")                                            # position not encoded
