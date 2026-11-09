"""Pure helpers of the harnesses that need no physics kernels (fast)."""

from __future__ import annotations

import math
import re

import pytest

from cnctwin.db import Database
from cnctwin.validation.cases import load_cases
from cnctwin.validation.harnesses import common as H
from cnctwin.validation.harnesses import cycletime as CT


@pytest.fixture(scope="module")
def cycle_cases():
    return {c.id: c for c in load_cases(Database.load(), ["cycle_time"])}


def _length(blocks: list[str]) -> float:
    pos = (0.0, 0.0)
    total = 0.0
    for b in blocks:
        x = float(re.search(r"X(-?[0-9.]+)", b).group(1)) * 1e-3
        y = float(re.search(r"Y(-?[0-9.]+)", b).group(1)) * 1e-3
        if b.startswith("G03"):
            r = float(re.search(r"J(-?[0-9.]+)", b).group(1)) * 1e-3
            total += 2.0 * math.pi * r
        else:
            total += math.hypot(x - pos[0], y - pos[1])
        pos = (x, y)
    return total


def test_trochoid_programs_have_the_stated_path_length(cycle_cases):
    case = cycle_cases["cycletime-wilkinson2025-trochoid-840d-tnc640"]
    xs, end, cy = CT._path(case, 10e-3)
    assert xs == pytest.approx([0.0149, 0.0302, 0.0456, 0.0609, 0.0763]) and end == pytest.approx(0.0789)
    arcs = CT.trochoid_blocks(xs, 10e-3, end, cy, False, 5e-6)
    assert len(arcs) == 11
    assert _length(arcs) == pytest.approx(5 * 2 * math.pi * 0.010 + 0.0789, rel=1e-6)    # "about 393 mm"
    poly = CT.trochoid_blocks(xs, 10e-3, end, cy, True, 5e-6)
    assert _length(poly) == pytest.approx(_length(arcs), rel=2e-3)     # chords slightly shorter than the arcs
    case2 = cycle_cases["cycletime-wilkinson2025-trochoid-tnc640-r5-r10"]
    xs2, end2, cy2 = CT._path(case2, 5e-3)
    assert xs2 == pytest.approx([0.0, 0.0025, 0.005, 0.0075, 0.01]) and cy2 == pytest.approx(0.005)
    assert _length(CT.trochoid_blocks(xs2, 5e-3, end2, cy2, False, 5e-6)) == pytest.approx(0.1671, abs=1e-4)


def test_cycle_time_tests_read_the_stated_controller_limits(cycle_cases):
    t1 = CT._tests(cycle_cases["cycletime-wilkinson2025-trochoid-840d-tnc640"])
    assert [(t["controller"], t["ipo_ms"], t["jerk"]) for t in t1[::4]] == [("840d", 2.0, 145.0), ("tnc640", 3.0, 50.0)]
    assert [t["feed_mm_min"] for t in t1[:4]] == [3000, 3000, 6000, 6000]
    t2 = CT._tests(cycle_cases["cycletime-wilkinson2025-trochoid-tnc640-r5-r10"])
    assert {(round(t["radius_m"] * 1e3), round(t["tol_m"] * 1e6)) for t in t2} == {(5, 10), (5, 100), (10, 10), (10, 100)}
    assert all(t["ipo_ms"] == pytest.approx(1.0) and t["accel"] == 3.1 for t in t2)
    text, first, last = CT.program(["G01 X1. Y0.", "G01 X2. Y0."], 3000)
    lines = text.splitlines()
    assert lines[first - 1] == "G01 X1. Y0." and lines[last - 1] == "G01 X2. Y0." and "G05.1 Q1" in text


def test_collector_keeps_unique_assumptions_and_sources():
    col = H.Collector()
    col.assume("edge radius", 3.5e-5, "not stated", "est")
    col.assume("edge radius", 3.5e-5, "not stated", "est")
    col.used("flow stress", "paper", "note")
    col.used("flow stress", "paper", "note")
    assert len(col.assumptions) == 1 and col.assumptions[0].value == "3.5e-05"
    assert len(col.provenance) == 1


def test_coolant_turning_reads_the_stated_rake_land():
    """Struzikiewicz states the SGF profile's 15 deg rake land: on the negative holder's -6 deg tilt the working
    rake is +9 deg (the chip-form helper's holder rake alone would treat the insert top as flat)."""
    from types import SimpleNamespace

    from cnctwin.validation.harnesses import coolant as CO

    case = load_cases(Database.load(), ["coolant_reach"], ["coolant-struzikiewicz2023-ti64-jet-direction"])[0]
    col = H.Collector()
    col.assume("rake and inclination", "-6 / -6 deg", "not stated: a negative ISO holder for a negative insert", "est")
    chips = SimpleNamespace(tool=SimpleNamespace(chip_breaker=None), rake=math.radians(-6.0))
    CO._stated_rake(case, chips, col)
    assert math.degrees(chips.rake) == pytest.approx(9.0)
    inputs = {a.input: a.value for a in col.assumptions}
    assert "rake and inclination" not in inputs and inputs["rake"] == "+9 deg working" and inputs["inclination"] == "-6 deg"
    listed = SimpleNamespace(tool=SimpleNamespace(chip_breaker=object()), rake=0.1)      # a database insert: kept
    CO._stated_rake(case, listed, H.Collector())
    assert listed.rake == 0.1
