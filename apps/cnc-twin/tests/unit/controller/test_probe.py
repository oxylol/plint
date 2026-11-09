"""Probing through the state machine: G31 P1-P4, the built-in probe macros O9810-O9814 updating work and
tool offsets from a feedback stub, the tool-setter cycles (G37, O9851, O9853) and their alarms."""

import copy
import math

import pytest

from cnctwin.controller import ToolOffsetInit
from cnctwin.controller.ir import Kind, ProbeResult, SetterResult
from cnctwin.controller.schema import ControllerSettings
from harness import HAAS_MILL, MILL, alarm, moves, run, wrap

PROBE_MILL = copy.deepcopy(MILL)
PROBE_MILL["probes"] = {"spindle_probe": {"model": "test probe"},
                        "tool_setter": {"model": "test setter", "position_mm": [-50, -20, -400]}}
HAAS_PROBE = copy.deepcopy(HAAS_MILL)
HAAS_PROBE["probes"] = copy.deepcopy(PROBE_MILL["probes"])
G54 = [-0.4, -0.25, -0.3]
BALL = 0.003                                  # stylus ball radius = D1 (tool 1 is the probe)
TOOLS = [ToolOffsetInit(1, radius_m=BALL), ToolOffsetInit(2, length_m=0.1)]


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import cycles, hooks, probing
    hooks.load_plugins()
    cycles.register()
    probing.register()


class Box:
    """Probe stub: the stylus centre moves freely inside an axis-aligned box (machine coordinates, the part
    shrunk by the ball radius) or a cylinder; an axis-aligned skip move that ends outside stops on its wall."""

    def __init__(self, lo=(-1.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0), disc=None):
        self.lo, self.hi, self.disc, self.calls = list(lo), list(hi), disc, []

    def __call__(self, path, cmd):
        self.calls.append(cmd)
        t = [cmd.target[a] for a in "XYZ"]
        if self.disc is not None:
            (cx, cy), radius = self.disc
            if math.hypot(t[0] - cx, t[1] - cy) <= radius:
                return None
            if abs(t[0] - cx) >= abs(t[1] - cy):
                t[0] = cx + math.copysign(math.sqrt(radius ** 2 - (t[1] - cy) ** 2), t[0] - cx)
            else:
                t[1] = cy + math.copysign(math.sqrt(radius ** 2 - (t[0] - cx) ** 2), t[1] - cy)
            return ProbeResult(True, dict(zip("XYZ", t)), 0.0)
        if all(lo <= v <= hi for v, lo, hi in zip(t, self.lo, self.hi)):
            return None
        return ProbeResult(True, {a: min(max(v, lo), hi) for a, v, lo, hi in zip("XYZ", t, self.lo, self.hi)}, 0.0)


def probe_run(body, stub, machine=PROBE_MILL, **kw):
    return run(wrap("T1 M06\nG90 G54 G00 X0 Y0 Z20.\n" + body), machine, tools=TOOLS, offsets={"G54": G54},
               probe=stub, **kw)


def var(result, n):
    return result.controller.state.paths[0].variables["c100"][n - 100]


def g54(result):
    ex = result.controller.executors[0]
    return [round(c * a.quantum * 1e3, 4) for c, a in zip(result.controller.state.paths[0].offsets["G54"], ex.axes)]


def test_single_surface_macro_sets_g54_z_and_the_length_wear():
    top = G54[2] + 0.00005                                   # the part top is 0.05 mm above Z0
    stub = Box(lo=(-1, -1, top + BALL))
    r = probe_run("G65 P9811 Z0 S1", stub)
    assert alarm(r) is None
    assert var(r, 137) == pytest.approx(0.05) and var(r, 140) == pytest.approx(0.05)
    assert g54(r)[2] == pytest.approx(-299.95)
    skip = [c for c in stub.calls if c.kind == Kind.PROBE][0]
    assert skip.feed.override_allowed is False and skip.probe["stylus"] == "spindle_probe"
    haas = probe_run("G65 P9811 Z0 S1", Box(lo=(-1, -1, top + BALL)), machine=HAAS_PROBE)
    assert alarm(haas) is None and g54(haas)[2] == pytest.approx(-299.95), "Haas variable numbers"
    r = probe_run("G65 P9811 Z0 T2", Box(lo=(-1, -1, top + BALL)))
    wear = r.controller.state.paths[0].tool_offsets["2"][1]
    assert alarm(r) is None and wear == -50, "the tool cut 0.05 mm high: its length wear drops by 0.05 mm"


def test_pocket_and_bore_macros_centre_the_work_offset():
    # pocket 40.02 wide in X whose centre is 0.03 mm right of the nominal centre
    lo, hi = G54[0] + 0.00003 - 0.02001 + BALL, G54[0] + 0.00003 + 0.02001 - BALL
    r = probe_run("G00 Z-5.\nG65 P9812 X40. S1", Box(lo=(lo, -1, -1), hi=(hi, 1, 1)))
    assert alarm(r) is None
    assert var(r, 135) == pytest.approx(0.03) and var(r, 138) == pytest.approx(40.02) and var(r, 139) == pytest.approx(0.02)
    assert g54(r)[0] == pytest.approx(-399.97)
    # bore of 30.04 diameter centred 0.02 / -0.01 mm off the nominal centre
    centre = (G54[0] + 0.00002, G54[1] - 0.00001)
    r = probe_run("G00 Z-5.\nG65 P9814 D30. S1 H0.1", Box(disc=(centre, 0.01502 - BALL)))
    assert alarm(r) is None
    assert (var(r, 135), var(r, 136)) == (pytest.approx(0.02, abs=1e-3), pytest.approx(-0.01, abs=1e-3))
    assert var(r, 138) == pytest.approx(30.04, abs=1e-3) and var(r, 139) == pytest.approx(0.04, abs=1e-3)
    assert g54(r)[:2] == [pytest.approx(-399.98, abs=1e-3), pytest.approx(-250.01, abs=1e-3)]
    assert alarm(probe_run("G00 Z-5.\nG65 P9814 D30. H0.01", Box(disc=(centre, 0.01502 - BALL))))["text"] \
        .endswith("OUT OF TOLERANCE")


def test_renishaw_like_alarms_obstructed_open_calibration_format():
    obstructed = probe_run("G65 P9810 X50.", Box(hi=(G54[0] + 0.02, 1, 1)))
    a = alarm(obstructed)
    assert a["text"].endswith("PATH OBSTRUCTED") and "86" in a["code"]
    hit = [c for c in moves(obstructed) if c.kind == Kind.PROBE][-1]
    assert hit.target["X"] == pytest.approx(G54[0] + 0.05)
    assert alarm(probe_run("G65 P9811 Z0", Box()))["text"].endswith("PROBE OPEN")
    r = run(wrap("T2 M06\nG90 G00 X0 Y0 Z20.\nG65 P9811 Z0"), PROBE_MILL, tools=TOOLS, probe=Box())
    assert alarm(r)["text"].endswith("PROBE NOT CALIBRATED")
    assert alarm(probe_run("G65 P9811 X1. Y1.", Box()))["text"].endswith("FORMAT ERROR")


def test_g31_p_numbers_skip_variables_and_alarms():
    r = probe_run("G31 P3 X10. F100.\n#120 = #5061", Box(hi=(G54[0] + 0.004, 1, 1)))
    cmd = [c for c in moves(r) if c.kind == Kind.PROBE][0]
    payload = dict(cmd.probe)
    start = payload.pop("start_m")                       # additive: where the skip move starts (machine)
    latched = payload.pop("latched_m")                   # added by the dry run: where the probe triggered
    assert alarm(r) is None and payload == {"skip": 3, "stylus": "spindle_probe", "expect": "trigger"}
    assert set(start) == {"X", "Y", "Z"} and start["X"] < latched["X"] < cmd.target["X"]
    assert var(r, 120) == pytest.approx(4.0)
    assert alarm(probe_run("G31 P5 X10. F100.", Box()))["code"] == "TW0034"
    assert alarm(probe_run("G31 P99 X10. F100.", Box()))["code"] == "TW0012"
    assert alarm(probe_run("G41 D1 G01 X1. F100.\nG31 X10. F100.", Box()))["code"] == "PS0035"
    assert alarm(probe_run("G31 F100.", Box()))["code"] == "PS0005"
    free = probe_run("G31 X10. F100.\n#120 = #5061", Box(), settings=ControllerSettings(
        cycle_parameters={"skip_feed_override": True}))
    assert [c for c in moves(free) if c.kind == Kind.PROBE][0].feed.override_allowed is True
    assert var(free, 120) == pytest.approx(10.0), "no signal: #5061 is the end point"


def setter_stub(length_m):
    def answer(path, cmd):
        return SetterResult(cmd.setter["tool"], length_m=length_m)
    return answer


def setter_run(body, length_m, tools=None, machine=PROBE_MILL):
    tools = tools if tools is not None else [ToolOffsetInit(1, radius_m=BALL), ToolOffsetInit(2)]
    return run(wrap("T2 M06\n" + body), machine, tools=tools, setter=setter_stub(length_m))


def test_tool_length_macro_finds_a_new_tool_and_g37_sets_the_offset_from_the_setter_result():
    r = setter_run("G65 P9851\n#120 = #2202", 0.100013)
    assert alarm(r) is None
    assert var(r, 120) == pytest.approx(100.013)
    kinds = [c.setter["measure"] for c in r.commands[0] if c.kind == Kind.TOOL_SETTER]
    assert kinds == ["skip", "length"], "a zero offset is searched on the setter signal, then G37 measures"
    search = [c for c in r.commands[0] if c.kind == Kind.TOOL_SETTER][0]
    assert search.target["Z"] == pytest.approx(-0.41) and search.setter["skip"] == 2
    assert r.controller.state.paths[0].sys["setter_result"]["length_m"] == pytest.approx(0.100013)
    known = setter_run("G65 P9851", 0.1004, tools=[ToolOffsetInit(1, radius_m=BALL), ToolOffsetInit(2, length_m=0.1)])
    assert alarm(known) is None and known.controller.state.paths[0].tool_offsets["2"][0] == 100400
    assert [c.setter["measure"] for c in known.commands[0] if c.kind == Kind.TOOL_SETTER] == ["length"]
    far = setter_run("G65 P9851", 0.104, tools=[ToolOffsetInit(1, radius_m=BALL), ToolOffsetInit(2, length_m=0.1)])
    assert alarm(far)["code"] == "PS0080", "a length outside the G37 window"


def test_broken_tool_check_and_haas_setter():
    tools = [ToolOffsetInit(1, radius_m=BALL), ToolOffsetInit(2, length_m=0.1)]
    assert alarm(setter_run("G65 P9853", 0.10005, tools=tools)) is None
    assert alarm(setter_run("G65 P9853", 0.098, tools=tools))["text"].endswith("BROKEN TOOL")
    assert alarm(setter_run("G65 P9853", 0.1005, tools=tools))["text"].endswith("OUT OF TOLERANCE")
    assert alarm(setter_run("G65 P9853", 0.103, tools=tools))["text"].endswith("OUT OF TOLERANCE")
    r = setter_run("G65 P9853", 0.10005, tools=tools, machine=HAAS_PROBE)
    assert alarm(r) is None and r.controller.state.paths[0].tool_offsets["2"][0] == 100000
    r = setter_run("G65 P9853 H0.1", 0.10005, tools=tools)
    assert alarm(r) is None and var(r, 135) == pytest.approx(0.05)
    assert alarm(setter_run("G65 P9853 H0.01", 0.10005, tools=tools))["text"].endswith("OUT OF TOLERANCE")
    r = setter_run("G37 Z0", None, tools=tools)
    assert alarm(r)["code"] == "PS0081"


def test_library_selection_and_project_programs_win():
    from cnctwin.controller import Controller, ControllerSetup, ProgramSource
    from harness import FANUC_DB
    mine = "%\nO9811 (MY OWN)\n#100 = 7.\nM99\n%\n"
    setup = ControllerSetup(PROBE_MILL, [ProgramSource("t.nc", wrap("G65 P9811 Z0")), ProgramSource("m.nc", mine, -1, False)],
                            ControllerSettings(), controller_db=FANUC_DB, db_id="fixture", work_offsets_m=[{}],
                            tools=TOOLS)
    ctl = Controller(setup)
    names = [p.file for p in ctl.store.programs]
    assert "cnctwin-builtin/O9810.nc" in names and "cnctwin-builtin/O9811.nc" not in names and not ctl.load_problems
    plain = Controller(ControllerSetup(MILL, [ProgramSource("t.nc", wrap("M00"))], ControllerSettings(),
                                       controller_db=FANUC_DB, db_id="fixture", work_offsets_m=[{}], tools=TOOLS))
    assert not any(p.file.startswith("cnctwin-builtin") for p in plain.store.programs)
    forced = Controller(ControllerSetup(MILL, [ProgramSource("t.nc", wrap("M00"))], ControllerSettings(builtin_macros=True),
                                        controller_db=FANUC_DB, db_id="fixture", work_offsets_m=[{}], tools=TOOLS))
    assert sorted(p.file for p in forced.store.programs if p.file.startswith("cnctwin")) == \
        [f"cnctwin-builtin/O98{n}.nc" for n in ("10", "11", "12", "14")], "no setter position: no setter macros"
