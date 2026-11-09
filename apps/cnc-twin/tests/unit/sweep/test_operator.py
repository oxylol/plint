"""Operator overrides (cnctwin/sweep/operator.py, runner.py): the commands the controller hands to motion are
changed as the panel's keys change them on the machine; the runner installs the hook on every motion domain the
loop's Run creates (a fake Run here: the real loop is tested in the integration tests)."""

from __future__ import annotations

import math
import types

import pytest

from cnctwin.controller.ir import Command, Feed, FeedMode, Kind, LineRef
from cnctwin.core.errors import TwinError
from cnctwin.sweep import operator as O

REF = LineRef("op.nc", 1, "G01 X1")


def project(*tools, feed_range=(0.0, 2.0), spindle_range=None, rapid_steps=(0.05, 0.25, 0.5)):
    entries = [types.SimpleNamespace(number=n, path=p, tool=types.SimpleNamespace(type=t)) for n, p, t in tools]
    controller = types.SimpleNamespace(feed_override_range_frac=list(feed_range) if feed_range else None,
                                       spindle_override_range_frac=list(spindle_range) if spindle_range else None,
                                       rapid_override_steps_frac=list(rapid_steps) if rapid_steps else None)
    machine = types.SimpleNamespace(controller=controller,
                                    spindles=[types.SimpleNamespace(id="main", max_rad_s=1000.0)])
    return types.SimpleNamespace(tools=types.SimpleNamespace(list=entries), machine=machine)


def feed_cmd(value, tool=1, allowed=True, kind=Kind.LINEAR, mode=FeedMode.PER_MIN):
    return Command(kind, REF, feed=Feed(mode, value, allowed), modal={"tool": tool})


def spindle_cmd(omega, tool=1, css=None, clamp=None):
    return Command(Kind.SPINDLE, REF, spindle={"id": "main", "omega_rad_s": omega, "css_m_s": css, "max_rad_s": clamp},
                   modal={"tool": tool})


def test_normalise_drops_what_is_at_100_percent():
    assert O.normalise(None) is None and O.normalise({}) is None
    assert O.normalise({"feed_override": {"*": 1.0, "T1": 1.0}, "rapid_override": 1.0}) is None
    assert O.normalise({"feed_override": 1.2}) == {"feed_override": {"*": 1.2}}
    assert O.normalise({"feed_override": {"*": 1.2, "T1": 1.2, "T2": 1.0}}) == {"feed_override": {"*": 1.2, "T2": 1.0}}
    assert O.normalise({"spindle_override": {"T3@2": 0.9}}) == {"spindle_override": {"T3@2": 0.9}}
    for bad in ({"torque_override": 1}, {"feed_override": {"X1": 1.1}}, {"feed_override": -1}, {"feed_override": True}):
        with pytest.raises(ValueError):
            O.normalise(bad)


def test_problems_follow_the_machine_and_the_tools():
    p = project((1, 1, "square_endmill"), (5, 1, "tap_cutting"), spindle_range=(0.5, 1.2))
    assert O.problems({"feed_override": {"*": 1.5}, "spindle_override": {"T1": 0.8}}, p) == []
    found = O.problems({"feed_override": {"*": 2.5, "T9": 1.1}, "spindle_override": {"T5": 0.9, "*": 1.3},
                        "rapid_override": 0.3}, p)
    text = " | ".join(found)
    assert "2.5 is outside the machine's override range 0-200 %" in text and "no cutting tool 9" in text
    assert "tool 5 is a tap_cutting" in text and "50-120 %" in text and "not a rapid key" in text
    assert O.problems({"rapid_override": 0.25}, p) == [] and O.problems({"feed_override": "x"}, p)


class FakeMotion:
    def __init__(self):
        self.batches = []

    def accept(self, ctx, state, path, commands, panel=None):
        self.batches.append((path, list(commands), panel))


def test_the_hook_scales_feeds_and_spindle_commands_as_the_panel_does():
    p = project((1, 1, "square_endmill"), (2, 1, "ball_endmill"), (5, 1, "tap_cutting"), (1, 2, "turning_insert"))
    overrides = O.Overrides.from_options({"operator": {"feed_override": {"*": 1.5, "T2": 0.5, "T1@2": 2.0},
                                                       "spindle_override": {"*": 1.2},
                                                       "rapid_override": 0.25}})
    hook = O.OperatorHook(overrides, p)
    motion = FakeMotion()
    hook.install(motion)
    hook.install(motion)                                             # idempotent
    panel = types.SimpleNamespace(rapid_override=1.0, feed_override=1.0)
    commands = [feed_cmd(0.02), feed_cmd(0.02, tool=2), feed_cmd(0.001, tool=5, allowed=False),
                Command(Kind.RAPID, REF, feed=Feed(FeedMode.RAPID, 0.0, True), modal={"tool": 1}),
                feed_cmd(1e-4, mode=FeedMode.PER_REV), spindle_cmd(-500.0), spindle_cmd(-900.0),
                spindle_cmd(80.0, tool=5), spindle_cmd(-300.0, css=200.0, clamp=330.0), spindle_cmd(0.0),
                feed_cmd(0.02, tool=None)]
    originals = [c.feed.value if c.feed else None for c in commands]
    motion.accept(None, None, 0, commands, panel)
    path, seen, got_panel = motion.batches[0]
    assert path == 0 and got_panel is panel and panel.rapid_override == 0.25 and panel.feed_override == 1.0
    assert seen[0].feed.value == pytest.approx(0.03) and seen[1].feed.value == pytest.approx(0.01)
    assert seen[2].feed.value == 0.001 and seen[3].feed.value == 0.0          # tapping and rapids keep theirs
    assert seen[4].feed.value == pytest.approx(1.5e-4)                        # per revolution alike
    assert seen[5].spindle["omega_rad_s"] == pytest.approx(-600.0)
    assert seen[6].spindle["omega_rad_s"] == -1000.0                          # held at the spindle's maximum
    assert seen[7].spindle["omega_rad_s"] == 80.0                             # a tap stays at 100 %
    assert seen[8].spindle["css_m_s"] == pytest.approx(240.0) and seen[8].spindle["omega_rad_s"] == -330.0
    assert seen[9].spindle["omega_rad_s"] == 0.0 and seen[10].feed.value == pytest.approx(0.03)
    assert [c.feed.value if c.feed else None for c in commands] == originals  # the controller's objects untouched
    motion.accept(None, None, 1, [feed_cmd(0.02, tool=1)], None)
    assert motion.batches[1][1][0].feed.value == pytest.approx(0.04)          # T1 of path 2
    summary = hook.summary()
    assert summary["batches"] == 2 and summary["scaled"] == {"feed": 5, "spindle": 3, "locked_spindle": 1}
    with pytest.raises(TypeError):
        hook.install(object())


def test_the_runner_hooks_every_motion_the_run_creates(monkeypatch):
    from cnctwin.core import loop
    from cnctwin.sweep import runner
    calls = {}

    class FakeRun:
        def __init__(self, project_path, run_dir, options=None, progress_cb=None, *, replay=None):
            self.project = project((1, 1, "square_endmill"))
            self.options = options
            self.replay = replay

        def _new_motion(self, op_index):
            return FakeMotion()

        def execute(self):
            for op in range(2):
                motion = self._new_motion(op)
                motion.accept(None, None, 0, [feed_cmd(0.01)], None)
                calls.setdefault("feeds", []).append(motion.batches[0][1][0].feed.value)
            return {"complete": True}

    def plain_run_project(project_path, run_dir, options=None, progress_cb=None):
        calls["plain"] = dict(options or {})
        if (options or {}).get("replay"):
            return loop.Run(project_path, run_dir, options, progress_cb, replay=(0, 1)).execute()
        return {"plain": True}

    monkeypatch.setattr(loop, "Run", FakeRun)
    monkeypatch.setattr(loop, "run_project", plain_run_project)
    assert runner.run_project("p.yaml", "/tmp/x", {"operator": {"feed_override": 1.0}, "a": 1}) == {"plain": True}
    assert calls["plain"] == {"a": 1}
    result = runner.run_project("p.yaml", "/tmp/twin-sweep-runner-test", {"operator": {"feed_override": {"*": 2.0}}})
    assert calls["feeds"] == [pytest.approx(0.02), pytest.approx(0.02)] and result["complete"]
    assert result["operator"]["batches"] == 2 and result["operator"]["feed_override"] == {"*": 2.0}
    calls.clear()
    replayed = runner.run_project("p.yaml", "/tmp/twin-sweep-runner-test",
                                  {"operator": {"feed_override": {"*": 0.5}}, "replay": {"tick0": 0, "tick1": 5}})
    assert calls["feeds"] == [pytest.approx(0.005)] * 2 and replayed["operator"]["batches"] == 2
    assert loop.Run is FakeRun                                                  # restored after the replay

    class Unhooked(FakeRun):
        def execute(self):
            return {"complete": True}

    monkeypatch.setattr(loop, "Run", Unhooked)
    with pytest.raises(TwinError, match="no command reached the hook"):
        runner.run_project("p.yaml", "/tmp/twin-sweep-runner-test", {"operator": {"feed_override": 2.0}})


def test_the_runner_refuses_overrides_the_project_cannot_take(monkeypatch, tmp_path):
    """twin run, the API and the sweeps reach the overrides through this runner: an override outside the machine's
    range (or for a tool the project does not have) is refused before the run starts, naming each problem."""
    from cnctwin.core import loop
    from cnctwin.core.errors import InputError
    from cnctwin.sweep import runner
    started = []

    class FakeRun:
        def __init__(self, project_path, run_dir, options=None, progress_cb=None, *, replay=None):
            self.project = project((1, 1, "square_endmill"), feed_range=(0.0, 1.5))
            self.options = dict(options or {})

        def _new_motion(self, op_index):
            return FakeMotion()

        def execute(self):
            started.append(True)
            return {"complete": True}

    monkeypatch.setattr(loop, "Run", FakeRun)
    with pytest.raises(InputError, match=r"options.operator cannot be applied to this project: feed_override\.\*: "
                                         r"2 is outside the machine's override range 0-150 %; feed_override.T7: the "
                                         r"project has no cutting tool 7 on path 1"):
        runner.run_project("p.yaml", tmp_path / "run", {"operator": {"feed_override": {"*": 2.0, "T7": 1.1}}})
    assert started == []


def test_point_scratch_files_carry_their_format(tmp_path):
    """progress.json and outcome.json of a point process name their format (cnctwin.sweep.point); the sweep process
    reads them with or without it."""
    import json
    from cnctwin.sweep import point
    progress = point.Progress(tmp_path)
    progress(sim_time_s=0.5, est_total_s=2.0, op="OP10", phase="done")
    document = json.loads((tmp_path / "progress.json").read_text())
    assert document["format"] == "cnctwin-sweep-progress" and document["version"] == 1
    assert document["sim_time_s"] == 0.5 and document["op"] == "OP10"
