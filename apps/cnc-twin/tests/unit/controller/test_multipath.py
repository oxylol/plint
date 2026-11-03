import pytest

from cnctwin.controller.ir import Kind
from harness import SWISS, controller, mm, moves, run


P1 = "%\nO0001\nG98 G00 X10. Z5.\nG01 Z50. F300.\nM100\nG00 X20.\nM30\n%\n"
P2 = "%\nO0002\nG98 G00 X10.\nM100\nG01 Z10. F300.\nM30\n%\n"


def test_two_path_wait_synchronises_both_paths():
    r = run("", SWISS, programs=[P1, P2])
    assert r.alarms == [None, None]
    syncs = [[(i, c) for i, c in enumerate(cmds) if c.kind == Kind.SYNC] for cmds in r.commands]
    (i1, s1), (i2, s2) = syncs[0][0], syncs[1][0]
    assert s1.sync["code"] == "M100" and s1.sync["paths"] == [0, 1] and s2.sync["code"] == "M100"
    assert s1.sync["id"] != s2.sync["id"]
    arrive1, arrive2 = r.end_ticks[0][i1], r.end_ticks[1][i2]
    release = max(arrive1, arrive2)
    assert arrive1 > arrive2
    after1 = r.commands[0][i1 + 1]
    after2 = r.commands[1][i2 + 1]
    assert after1.ready_tick >= release and after2.ready_tick >= release
    assert r.start_ticks[1][i2 + 1] >= release
    assert after1.path == 0 and after2.path == 1
    assert "X1" in after1.target and "X2" in after2.target


def test_paths_use_their_own_axes_and_spindles():
    r = run("", SWISS, programs=["%\nO0001\nS1000 M03\nG00 X10.\nM30\n%\n", "%\nO0002\nS500 M03\nG00 X4.\nM30\n%\n"])
    s1 = [c for c in r.commands[0] if c.kind == Kind.SPINDLE][0]
    s2 = [c for c in r.commands[1] if c.kind == Kind.SPINDLE][0]
    assert s1.spindle["id"] == "main" and s2.spindle["id"] == "sub"
    assert mm(moves(r, 0)[0], "X1") == 5.0 and mm(moves(r, 1)[0], "X2") == 2.0


def test_mismatched_wait_codes_alarm_on_both_paths():
    r = run("", SWISS, programs=["%\nO0001\nM100\nM30\n%\n", "%\nO0002\nM101\nM30\n%\n"])
    assert r.alarms[0]["code"] == "PS0160" and r.alarms[1]["code"] == "PS0160"


def test_wait_for_an_ended_path_is_reported():
    r = run("", SWISS, programs=["%\nO0001\nG00 X1.\nM100\nM30\n%\n", "%\nO0002\nM30\n%\n"])
    assert r.alarms[0]["code"] == "TW0008"


def test_wait_with_p_names_the_paths():
    ctl = controller("", SWISS, programs=["%\nO0001\nM100 P12\nM30\n%\n", "%\nO0002\nM100 P3\nM30\n%\n"])
    ctl.produce(0, 0)
    assert ctl.state.paths[0].waiting["paths"] == [0, 1]
    ctl.produce(1, 0)
    assert ctl.state.paths[1].alarm["code"] == "PS0161"
    assert ctl.state.paths[0].alarm["code"] == "TW0008"


def test_feed_per_revolution_with_the_spindle_stopped_is_reported_by_the_dry_run():
    r = run("", SWISS, programs=["%\nO0001\nG99 G01 Z10. F0.1\nM30\n%\n", "%\nO0002\nM30\n%\n"])
    assert "spindle stopped" in r.stopped
    assert pytest.approx(1.0) == 1.0
