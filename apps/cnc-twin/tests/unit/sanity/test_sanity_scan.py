"""The own modal scan: positions, compensation, canned cycles, arcs, subprograms, offsets."""

import math

import numpy as np
import pytest

from cnctwin.controller.ir import Kind
from cnctwin.sanity.scan import scan, spindle_omega

from sanity_testkit import context_with, program


def _ends(result, kind=None):
    return [tuple(np.round(result.tip_end[i] * 1e3, 6)) for i, c in enumerate(result.commands)
            if kind is None or c.kind == kind]


def test_absolute_incremental_and_length_compensation(ctx):
    text = program("T1 M06", "G90 G54 G00 X10. Y20.", "G43 H01 Z50.", "G91 G00 X5. Z-10.", "G90 G00 Z5.")
    result = scan(text, "p.nc", ctx)
    rapids = _ends(result, Kind.RAPID)
    assert rapids[0][:2] == (10.0, 20.0)
    assert rapids[1] == (10.0, 20.0, 50.0)
    assert rapids[2] == (15.0, 20.0, 40.0)
    assert rapids[3] == (15.0, 20.0, 5.0)
    first = next(c for c in result.commands if c.kind == Kind.RAPID)
    assert first.target["X"] == pytest.approx(0.010 - 0.431)
    move = next(c for c in result.commands if c.kind == Kind.RAPID and c.modal["length_comp"])
    assert move.target["Z"] == pytest.approx(0.050 - 0.400 + 0.120)
    # at the start the tool stands at machine home: program Z = 0 - (-400) with G49
    assert result.tip_start[0][2] == pytest.approx(0.400)


def test_machine_coordinates_and_reference_return(ctx):
    text = program("T1 M06", "G90 G54 G00 X10. Y10.", "G43 H01 Z25.", "G53 Z0.", "G91 G28 Z0.", "G90 G28 Z0.")
    result = scan(text, "p.nc", ctx)
    rapids = _ends(result, Kind.RAPID)
    assert rapids[2][2] == pytest.approx(400.0 - 120.0)
    assert [e["distance"] for e in result.g28] == ["incremental", "absolute"]
    assert result.g28[1]["intermediate_W_m"][2] == pytest.approx(0.0)


def test_canned_cycle_holes_and_retract_levels(ctx):
    text = program("T3 M06", "G90 G54 G00 X10. Y10.", "G43 H03 Z25.", "G99 G81 Z-12. R2. F240.", "X20.",
                   "G98 G83 X30. Z-16. Q5.", "G80")
    result = scan(text, "p.nc", ctx)
    holes = result.holes
    assert [(round(h.x_m * 1e3, 6), round(h.z_bottom_m * 1e3, 6)) for h in holes] == [(10, -12), (20, -12), (30, -16)]
    assert [round(h.retract_z_m * 1e3, 6) for h in holes] == [2.0, 2.0, 25.0]
    assert all(h.z_r_m == pytest.approx(0.002) for h in holes)
    feeds = [c for c in result.commands if c.kind == Kind.LINEAR]
    assert all(c.modal["cycle_info"]["phase"] == "feed" for c in feeds)


def test_incremental_cycle_with_repeats(ctx):
    text = program("T3 M06", "G90 G54 G00 X0. Y0.", "G43 H03 Z10.", "G91 G99 G81 X10. Z-5. R-8. K3 F200.", "G80 G90")
    result = scan(text, "p.nc", ctx)
    assert [round(h.x_m * 1e3, 6) for h in result.holes] == [10.0, 20.0, 30.0]
    assert all(h.z_r_m == pytest.approx(0.002) and h.z_bottom_m == pytest.approx(-0.003) for h in result.holes)


def test_arcs_centre_and_radius(ctx):
    text = program("T1 M06", "G90 G54 G00 X0. Y0.", "G43 H01 Z5.", "G01 X10. F500.", "G03 X20. Y10. I0. J10.",
                   "G02 X30. Y0. R10.")
    result = scan(text, "p.nc", ctx)
    arcs = [c.arc for c in result.commands if c.kind == Kind.ARC]
    assert arcs[0].centre[:2] == pytest.approx((0.010, 0.010))
    assert arcs[0].radius_start_m == pytest.approx(0.010) and arcs[0].radius_end_m == pytest.approx(0.010)
    assert arcs[0].sweep_rad == pytest.approx(math.pi / 2)
    assert arcs[1].sweep_rad < 0 and arcs[1].radius_end_m == pytest.approx(0.010)


def test_fanuc_integers_are_increments():
    ctx = context_with(lambda d: d["machine"]["controller"].update({"dialect": "fanuc"}))
    result = scan(program("G90 G54 G00 X1000 Y2.", "Z5."), "p.nc", ctx)
    assert result.tip_end[0][0] == pytest.approx(0.001)
    assert result.tip_end[0][1] == pytest.approx(0.002)


def test_subprograms_in_the_same_file(ctx):
    text = program("O1000", "T1 M06", "G90 G54 G00 X0. Y0.", "G43 H01 Z5.", "M98 P2000 L2", "M97 P100", "M30",
                   "N100 G00 X50.", "M99", "O2000", "G91 G00 X1.", "G90", "M99")
    result = scan(text, "p.nc", ctx)
    xs = [round(p[0] * 1e3, 6) for p in result.tip_end[[i for i, c in enumerate(result.commands)
                                                         if c.kind == Kind.RAPID]]]
    assert xs[-3:] == [1.0, 2.0, 50.0]
    assert result.end_code == "M30"


def test_missing_subprogram_and_macros_are_noted(ctx):
    result = scan(program("T1 M06", "M98 P9999", "#100 = 2", "G00 X#100", "G65 P9810 X1."), "p.nc", ctx)
    kinds = sorted({n.kind for n in result.notes})
    assert kinds == ["call", "flow", "macro"]


def test_g10_sets_offsets_and_g37_measures(ctx):
    text = program("G90 G10 L2 P3 X-400. Y-200. Z-350.", "G10 L10 P5 R118.", "G56", "T1 M06", "G37 H01")
    result = scan(text, "p.nc", ctx)
    assert result.offsets_set["G56"][0] == 1
    assert result.length_measured[5][0] == 2 and result.length_measured[1][0] == 5


def test_spindle_sign_convention():
    assert spindle_omega("M3", 1000.0) < 0 < spindle_omega("M4", 1000.0)
    assert spindle_omega("M5", 1000.0) == 0.0


def test_lathe_tool_words_and_diameters():
    def lathe(d):
        d["machine"]["type"] = "lathe"
        d["stock"] = {"kind": "cylinder", "size_mm": [50, 100]}
    ctx = context_with(lathe)
    text = program("G18 G21 G99", "T0101", "G97 S1500 M03", "G00 X52. Z2.", "G01 X40. F0.2", "W-20.", "U10.",
                   "T0202", "G28 U0. W0.", "M30")
    result = scan(text, "p.nc", ctx)
    ends = [tuple(np.round(result.tip_end[i] * 1e3, 6)) for i, c in enumerate(result.commands)
            if c.kind in (Kind.RAPID, Kind.LINEAR)]
    assert ends[:4] == [(26.0, 0.0, 2.0), (20.0, 0.0, 2.0), (20.0, 0.0, -18.0), (25.0, 0.0, -18.0)]
    changes = [c.tool["number"] for c in result.commands if c.kind == Kind.TOOL_CHANGE]
    assert changes == [1, 2]
    assert result.commands[-1].modal["feed_mode"] == "per_rev"
    assert result.g28[0]["distance"] == "incremental"
