import math

import numpy as np
import pytest

from cnctwin.controller import abi, kernels


def _pocket():
    points = [(0.0, 0.0), (0.030, 0.0), (0.030, 0.020), (0.018, 0.020), (0.018, 0.008), (0.012, 0.008),
              (0.012, 0.020), (0.0, 0.020)]
    return abi.boundary(points)


def test_raster_intervals_match_the_python_reference():
    seg = _pocket()
    lo = np.empty(16)
    hi = np.empty(16)
    for y in np.linspace(0.0005, 0.0195, 41):
        for d in (0.0, 0.001, 0.003):
            n = kernels.raster_intervals(seg, float(y), d, lo, hi)
            expected = kernels.raster_intervals_ref(seg, float(y), d)
            assert n == len(expected)
            for k, (a, b) in enumerate(expected):
                assert lo[k] == pytest.approx(a, abs=1e-15) and hi[k] == pytest.approx(b, abs=1e-15)


def test_raster_intervals_geometry():
    seg = _pocket()
    lo = np.empty(8)
    hi = np.empty(8)
    n = kernels.raster_intervals(seg, 0.015, 0.002, lo, hi)
    assert n == 2
    assert (lo[0], hi[0]) == pytest.approx((0.002, 0.010)) and (lo[1], hi[1]) == pytest.approx((0.020, 0.028))
    n = kernels.raster_intervals(seg, 0.004, 0.002, lo, hi)
    assert n == 1 and (lo[0], hi[0]) == pytest.approx((0.002, 0.028))
    below_island = kernels.raster_intervals(seg, 0.0095, 0.002, lo, hi)
    assert below_island == 2
    assert (lo[0], hi[0]) == pytest.approx((0.002, 0.010)) and (lo[1], hi[1]) == pytest.approx((0.020, 0.028))
    clear = kernels.raster_intervals(seg, 0.0055, 0.002, lo, hi)
    assert clear == 1 and (lo[0], hi[0]) == pytest.approx((0.002, 0.028))
    corner = kernels.raster_intervals(seg, 0.0065, 0.002, lo, hi)
    reach = math.sqrt(0.002 ** 2 - 0.0015 ** 2)
    assert corner == 2 and hi[0] == pytest.approx(0.012 - reach) and lo[1] == pytest.approx(0.018 + reach)
    small = np.empty(1)
    assert kernels.raster_intervals(seg, 0.015, 0.002, small, small.copy()) == -1


def test_segment_clearance_matches_reference():
    seg = _pocket()
    cases = [(0.005, 0.005, 0.025, 0.005), (0.005, 0.015, 0.025, 0.015), (0.002, 0.002, 0.004, 0.003),
             (0.015, 0.002, 0.015, 0.006)]
    for case in cases:
        assert kernels.segment_clearance(seg, *case) == pytest.approx(kernels.segment_clearance_ref(seg, *case),
                                                                         abs=1e-15)
    assert kernels.segment_clearance(seg, 0.005, 0.015, 0.025, 0.015) == 0.0
    assert kernels.segment_clearance(seg, 0.005, 0.005, 0.025, 0.005) == pytest.approx(0.003)


def test_kernels_compile_once():
    for kernel in kernels.KERNELS + (kernels._point_segment,):
        assert len(kernel.signatures) == 1


def test_arc_chords_respect_the_sagitta():
    points = abi.arc_chords((0.0, 0.0), (0.01, 0.0), math.pi / 2, 1e-6)
    assert points[-1] == pytest.approx((0.0, 0.01), abs=1e-15)
    step = math.atan2(points[0][1], points[0][0])
    assert 0.01 * (1 - math.cos(step / 2)) <= 1e-6
