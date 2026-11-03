"""Canned cycles for drilling (FANUC M series, Haas NGC): exact command sequences through the state machine."""

import copy
import io
import json
import math

import numpy as np
import pytest

from cnctwin.controller import Controller, ControllerState, dry_run
from cnctwin.controller.ir import FeedMode, Kind
from harness import HAAS_MILL, MILL, alarm, controller, run, wrap


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import comp, cycles, hooks
    hooks.load_plugins()
    cycles.register()
    comp.register()


def seq(result, path: int = 0):
    """(kind, X, Y, Z in mm | payload) for every command except the program end."""
    out = []
    for c in result.commands[path]:
        if c.kind in (Kind.RAPID, Kind.LINEAR):
            out.append((c.kind.value, *(round(c.target[a] * 1000.0, 4) for a in ("X", "Y", "Z"))))
        elif c.kind == Kind.DWELL:
            out.append(("dwell", round(c.dwell_s, 6)))
        elif c.kind == Kind.SPINDLE:
            out.append(("spindle", round(c.spindle["omega_rad_s"] * 60 / (2 * math.pi), 3),
                        None if c.spindle.get("orient_rad") is None else round(c.spindle["orient_rad"], 6)))
        elif c.kind == Kind.STOP and c.stop not in ("M30",):
            out.append(("stop", c.stop))
        elif c.kind == Kind.MODE and "rigid_tap" in c.modal:
            out.append(("rigid", c.modal["rigid_tap"]))
    return out


START = "G90 G00 X0 Y0 Z20.\nS1000 M03\n"


def test_g81_g98_g99_r_plane_and_return_levels():
    r = run(wrap(START + "G99 G81 X10. Y5. Z-8. R2. F120.\nG98 Y15.\nG80"))
    assert alarm(r) is None
    assert seq(r)[2:] == [("rapid", 10.0, 5.0, 20.0), ("rapid", 10.0, 5.0, 2.0), ("linear", 10.0, 5.0, -8.0),
                          ("rapid", 10.0, 5.0, 2.0),
                          ("rapid", 10.0, 15.0, 2.0), ("linear", 10.0, 15.0, -8.0), ("rapid", 10.0, 15.0, 20.0)]
    feed = [c for c in r.commands[0] if c.kind == Kind.LINEAR][0].feed
    assert feed.mode == FeedMode.PER_MIN and feed.value == pytest.approx(0.12 / 60.0)


def test_g82_dwell_and_g91_incremental_repeats_k():
    r = run(wrap(START + "G99 G82 X10. Y0 Z-5. R2. P300 F100.\nG91 X10. K3\nG90 G80"))
    s = seq(r)
    bottoms = [step for step in s if step[0] == "linear"]
    assert [b[1] for b in bottoms] == [10.0, 20.0, 30.0, 40.0] and all(b[3] == -5.0 for b in bottoms)
    assert s.count(("dwell", 0.3)) == 4


def test_g91_r_and_z_are_incremental_from_the_initial_level_and_r():
    r = run(wrap("G90 G00 X0 Y0 Z20.\nG91 G98 G81 X10. Z-15. R-18. F100.\nG90 G80"))
    assert seq(r)[1:] == [("rapid", 10.0, 0.0, 20.0), ("rapid", 10.0, 0.0, 2.0), ("linear", 10.0, 0.0, -13.0),
                          ("rapid", 10.0, 0.0, 20.0)]


def test_g73_high_speed_peck_retracts_by_parameter_5114():
    r = run(wrap(START + "G99 G73 X0 Y0 Z-10. R2. Q4. F100.\nG80"))
    z = [step[3] for step in seq(r)[2:]]
    assert z == [2.0, -2.0, -1.5, -6.0, -5.5, -10.0, 2.0]


def test_g83_peck_returns_to_r_and_rapids_back_to_clearance():
    r = run(wrap(START + "G98 G83 X0 Y0 Z-10. R2. Q4. F100.\nG80"))
    s = seq(r)[2:]
    assert [(k, z) for k, _, _, z in s] == [("rapid", 2.0), ("linear", -2.0), ("rapid", 2.0), ("rapid", -1.5),
                                            ("linear", -6.0), ("rapid", 2.0), ("rapid", -5.5), ("linear", -10.0),
                                            ("rapid", 20.0)]
    assert alarm(run(wrap(START + "G83 X0 Y0 Z-10. R2. F100.")))["code"] == "PS0045"


def test_g84_floating_tap_reverses_with_m04_and_no_override():
    r = run(wrap("G90 G00 X0 Y0 Z20.\nS200 M03\nG99 G84 X5. Y0 Z-10. R3. P100 F250.\nG80"))
    s = seq(r)
    i = s.index(("linear", 5.0, 0.0, -10.0))
    assert s[i + 1:i + 6] == [("dwell", 0.1), ("spindle", 200.0, None), ("linear", 5.0, 0.0, 3.0), ("dwell", 0.1),
                              ("spindle", -200.0, None)]
    taps = [c for c in r.commands[0] if c.kind == Kind.LINEAR]
    assert all(not c.feed.override_allowed for c in taps)
    assert taps[0].modal["tap"]["rigid"] is False


def test_rigid_g84_after_m29_is_synchronised_and_g80_releases_it():
    r = run(wrap("G90 G94 G00 X0 Y0 Z20.\nM29 S500\nG98 G84 X10. Z-10. R3. F500. P200\nG80\nG00 Z30."))
    s = seq(r)
    assert ("rigid", True) in s and s[-2:] == [("rigid", False), ("rapid", 10.0, 0.0, 30.0)]
    taps = [c for c in r.commands[0] if c.kind == Kind.LINEAR]
    assert [c.modal["tap"]["phase"] for c in taps] == ["in", "out"]
    assert all(c.feed.mode == FeedMode.PER_REV and c.feed.value == pytest.approx(0.001) for c in taps)
    assert all(c.modal["rigid_tap"] and not c.feed.override_allowed for c in taps)
    assert s.count(("dwell", 0.2)) == 2
    assert r.controller.state.paths[0].spindle.s_value == 0.0


def test_rigid_tapping_alarms():
    assert alarm(run(wrap("G90 G00 X0 Y0 Z20.\nM29 S500\nG00 X5.\nG84 X10. Z-10. R3. F500.")))["code"] == "PS0203"
    assert alarm(run(wrap("G90 G00 X0 Y0 Z20.\nM29 S50000\nG84 X10. Z-10. R3. F500.")))["code"] == "PS0200"
    assert alarm(run(wrap("G90 G00 X0 Y0 Z20.\nS500 M03\nG84.2 X10. Z-10. R3.\nG80")))["code"] == "PS0201"


def test_g84_2_fs15_rigid_and_peck_tapping():
    r = run(wrap("G90 G95 G00 X0 Y0 Z20.\nS600 M03\nG84.2 X0 Y0 Z-6. R2. Q3. F1.25\nG80"))
    assert alarm(r) is None
    taps = [(c.modal["tap"]["phase"], round(c.target["Z"] * 1000, 4)) for c in r.commands[0] if c.kind == Kind.LINEAR]
    assert taps == [("in", -1.0), ("out", 0.0), ("in", -4.0), ("out", -3.0), ("in", -6.0), ("out", 2.0)]


def test_g74_left_hand_tap_and_g85_g89_bore_out_at_feed():
    r = run(wrap("G90 G00 X0 Y0 Z20.\nS200 M04\nG98 G74 X5. Z-5. R2. F200.\nG80"))
    s = seq(r)
    assert ("spindle", -200.0, None) in s and s[-2] == ("spindle", 200.0, None)
    r = run(wrap(START + "G99 G89 X0 Y0 Z-5. R2. P500 F80.\nG85 X5.\nG80"))
    s = seq(r)[2:]
    assert s == [("rapid", 0.0, 0.0, 2.0), ("linear", 0.0, 0.0, -5.0), ("dwell", 0.5), ("linear", 0.0, 0.0, 2.0),
                 ("rapid", 5.0, 0.0, 2.0), ("linear", 5.0, 0.0, -5.0), ("linear", 5.0, 0.0, 2.0)]


def test_g76_orients_shifts_and_restarts_the_spindle():
    r = run(wrap(START + "G98 G76 X10. Y0 Z-10. R3. Q0.5 P100 F60.\nG80"))
    s = seq(r)[2:]
    assert s == [("rapid", 10.0, 0.0, 20.0), ("rapid", 10.0, 0.0, 3.0), ("linear", 10.0, 0.0, -10.0), ("dwell", 0.1),
                 ("spindle", 0.0, None), ("spindle", 0.0, 0.0), ("rapid", 10.5, 0.0, -10.0),
                 ("rapid", 10.5, 0.0, 20.0), ("rapid", 10.0, 0.0, 20.0), ("spindle", -1000.0, None)]


def test_g87_back_boring_from_below():
    r = run(wrap(START + "G98 G87 X0 Y0 Z-10. R-30. Q1. F50.\nG80"))
    s = seq(r)[2:]
    zs = [(step[0], step[1], step[3]) for step in s if step[0] in ("rapid", "linear")]
    assert zs == [("rapid", 1.0, 20.0), ("rapid", 1.0, -30.0), ("rapid", 0.0, -30.0), ("linear", 0.0, -10.0),
                  ("rapid", 1.0, -10.0), ("rapid", 1.0, 20.0), ("rapid", 0.0, 20.0)]


def test_g86_and_g88_manual_retract_waits_for_cycle_start():
    r = run(wrap(START + "G99 G86 X0 Y0 Z-5. R2. F80.\nG80"))
    assert seq(r)[2:] == [("rapid", 0.0, 0.0, 2.0), ("linear", 0.0, 0.0, -5.0), ("spindle", 0.0, None),
                          ("rapid", 0.0, 0.0, 2.0), ("spindle", -1000.0, None)]
    r = run(wrap(START + "G98 G88 X0 Y0 Z-5. R2. P200 F80.\nG80"))
    s = seq(r)[2:]
    assert s[2:6] == [("dwell", 0.2), ("spindle", 0.0, None), ("stop", "G88"), ("rapid", 0.0, 0.0, 20.0)]


def test_canned_cycle_modal_data_cancel_and_plane_change():
    r = run(wrap(START + "G99 G81 X0 Y0 Z-5. R2. F100.\nX5.\nG01 X10. F200.\nX12."))
    lin = [c for c in r.commands[0] if c.kind == Kind.LINEAR]
    assert [round(c.target["X"] * 1000, 3) for c in lin] == [0.0, 5.0, 10.0, 12.0]
    assert r.controller.state.paths[0].cycle is None
    assert alarm(run(wrap(START + "G81 X0 Y0 Z-5. R2. F100.\nG18 X5.")))["code"] == "PS0549"


def test_haas_peck_with_i_j_k_and_g87_manual_retract():
    r = run(wrap(START + "G99 G83 X0 Y0 Z-10. R0 I4. J1. K2. F100.\nG80"), HAAS_MILL)
    depths = [round(c.target["Z"] * 1000, 4) for c in r.commands[0] if c.kind == Kind.LINEAR]
    assert depths == [-4.0, -7.0, -9.0, -10.0]
    r = run(wrap(START + "G98 G87 X0 Y0 Z-5. R2. F80.\nG80"), HAAS_MILL)
    assert ("stop", "G87") in seq(r)


def test_haas_g84_is_rigid_with_j_retract_multiple():
    r = run(wrap("G90 G00 X0 Y0 Z20.\nS400 M03\nG99 G84 X0 Y0 Z-8. R2. F400. J2\nG80"), HAAS_MILL)
    assert alarm(r) is None
    spindles = [c.spindle["omega_rad_s"] * 60 / (2 * math.pi) for c in r.commands[0] if c.kind == Kind.SPINDLE]
    assert spindles[-2] == pytest.approx(-800.0) and spindles[-1] == pytest.approx(-400.0)
    assert any(c.kind == Kind.MODE and c.modal.get("rigid_tap") for c in r.commands[0])


def _drive(ctl, stop_after=None):
    out, tick = [], 0
    while True:
        out.extend(ctl.produce(0, tick))
        path = ctl.state.paths[0]
        if path.alarm is not None or (path.waiting is not None and path.waiting["kind"] == "end"):
            return out, tick
        if path.buffer:
            tick += 2000
            ctl.retire(0, 1, tick)
        else:
            tick += 5000
        if stop_after is not None and len(out) >= stop_after:
            return out, tick


def test_checkpoint_resume_inside_a_peck_cycle_is_bit_identical():
    small = copy.deepcopy(MILL)
    small["controller"]["lookahead_blocks"] = 3
    program = wrap(START + "G99 G83 X0 Y0 Z-20. R2. Q2. F100.\nG91 X10. K3\nG90 G80\nG00 Z50.")
    reference, _ = _drive(controller(program, small))
    first = controller(program, small)
    part, tick = _drive(first, stop_after=9)
    assert first.state.paths[0].pending, "the checkpoint must fall inside the cycle expansion"
    arrays, meta = first.state.to_arrays()
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    restored_state = ControllerState.from_arrays(dict(np.load(io.BytesIO(buffer.getvalue()))),
                                                 json.loads(json.dumps(meta)))
    restored = Controller(first.setup, restored_state)
    rest = []
    while True:
        rest.extend(restored.produce(0, tick))
        path = restored.state.paths[0]
        if path.alarm is not None or (path.waiting is not None and path.waiting["kind"] == "end"):
            break
        if path.buffer:
            tick += 2000
            restored.retire(0, 1, tick)
        else:
            tick += 5000
    assert part + rest == reference


def test_dry_run_cycle_time_of_a_drilled_pattern_is_exact():
    r = run(wrap("G90 G00 X0 Y0 Z10.\nS1000 M03\nG99 G81 X0 Y0 Z-6. R2. F480.\nG91 X10. K2\nG90 G80"))
    linear = [(s, e) for c, s, e in zip(r.commands[0], r.start_ticks[0], r.end_ticks[0]) if c.kind == Kind.LINEAR]
    assert all(e - s == pytest.approx(1e6, abs=2) for s, e in linear)      # 8 mm at 480 mm/min = 1 s each
    assert dry_run is not None
