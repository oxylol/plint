"""Tool life management through the state machine: group calls, sister tools, use and time counting,
G10 L3 data, H99/D99, expiry from the physics, lathe T(group)99 / T(group)88, Haas groups."""

import pytest

from cnctwin.controller import ToolOffsetInit
from cnctwin.controller.ir import Feedback, Kind
from cnctwin.controller.schema import ControllerSettings
from harness import HAAS_MILL, LATHE, MILL, alarm, controller, run, wrap
from cnctwin.controller.dryrun import dry_run

TOOLS = [ToolOffsetInit(n, length_m=0.1, radius_m=0.005) for n in (5, 10, 11, 12, 20, 21, 31)]


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import hooks, toollife
    hooks.load_plugins()
    toollife.register()


def life(groups, **extra):
    return ControllerSettings(tool_life={"groups": groups}, cycle_parameters=extra or {})


def changes(result):
    return [c.tool["number"] for c in result.commands[0] if c.kind == Kind.TOOL_CHANGE]


PART = "T9901 M06\nG00 X0 Y0 Z10.\nT9901\nM06"


def test_sister_tools_are_taken_when_the_use_count_expires_and_all_expired_alarms():
    settings = life([{"group": 1, "life_uses": 1, "tools": [10, 11, 12]}])
    r = run(wrap(PART), MILL, tools=TOOLS, settings=settings, parts=3)
    assert alarm(r) is None and changes(r) == [10, 10, 11, 11, 12, 12], "one use per run; the same tool inside a run"
    r = run(wrap(PART), MILL, tools=TOOLS, settings=settings, parts=4)
    a = alarm(r)
    assert a["code"] == "PS5317" and changes(r) == [10, 10, 11, 11, 12, 12]
    r = run(wrap(PART), MILL, tools=TOOLS, parts=4,
            settings=life([{"group": 1, "life_uses": 1, "tools": [10, 11]}], tool_life_all_expired="last"))
    assert alarm(r) is None and changes(r) == [10, 10, 11, 11, 11, 11, 11, 11]     # the last tool is kept
    assert any("every tool expired" in e.get("message", "") for e in r.controller.state.paths[0].events)


def test_ordinary_t_codes_and_unknown_groups_are_not_managed():
    settings = life([{"group": 1, "life_uses": 5, "tools": [10]}])
    r = run(wrap("T5 M06\nT9901 M06\nG43 H99 Z50.\n#100 = #4111"), MILL, tools=TOOLS, settings=settings)
    assert alarm(r) is None and changes(r) == [5, 10]
    assert r.controller.state.paths[0].variables["c100"][0] == 10.0, "H99 is the H registered for the tool"
    assert alarm(run(wrap("T9907 M06"), MILL, tools=TOOLS, settings=settings))["code"] == "TW0013", \
        "an unknown group is an ordinary T code (then too big for the magazine)"
    assert alarm(run(wrap("T5 M06\nG43 H99 Z50."), MILL, tools=TOOLS, settings=settings))["code"] == "PS0154"
    assert alarm(run(wrap("T10099 M06"), MILL, tools=TOOLS, settings=settings))["code"] == "PS0150"


def test_g10_l3_registers_groups_with_offsets_and_count_types():
    body = ("G10 L3\nP1 L1\nT10 H10 D10\nT11 H31 D32\nP2 L30 Q2\nT20\nG11\n"
            "T9901 M06\nG43 H99 Z50.\nG41 D99 G01 X10. F500.\nG40 G00 X0\n#100 = #4111\n#101 = #4107")
    r = run(wrap(body), MILL, tools=TOOLS, settings=ControllerSettings())
    assert alarm(r) is None
    st = r.controller.state.paths[0].sys["tool_life"]
    assert st["groups"]["1"]["tools"][1] == {"t": 11, "h": 31, "d": 32, "used": 0.0, "expired": False}
    assert st["groups"]["2"]["count"] == "time" and st["groups"]["2"]["life"] == 1800.0
    assert r.controller.state.paths[0].variables["c100"][:2] == [10.0, 10.0]
    assert alarm(run(wrap("G10 L3\nT10\nG11"), MILL, tools=TOOLS))["code"] == "PS0156"
    assert alarm(run(wrap("G10 L3\nP1 L1\nH10\nG11"), MILL, tools=TOOLS))["code"] == "PS0153"
    assert alarm(run(wrap("G10 L3\nP1 L1\nT10 X1.\nG11"), MILL, tools=TOOLS))["code"] == "PS0149"
    assert alarm(run(wrap("G10 L3\nP200 L1\nG11"), MILL, tools=TOOLS))["code"] == "PS0150"
    r = run(wrap("G10 L3\nP1 L1\nT10\nP2 L1\nT11\nG11\nG10 L3 P2\nP1\nG11\nT9901 M06"), MILL, tools=TOOLS)
    assert alarm(r)["code"] == "TW0013", "group 1 deleted: T9901 is an ordinary (too big) T code"


def test_time_counting_from_retired_cutting_moves():
    settings = life([{"group": 2, "life_min": 0.05, "tools": [20, 21]}])
    body = "T9902 M06\nG00 X0 Y0 Z10.\nG01 X-100. F1000.\nG00 X0"
    r = run(wrap(body), MILL, tools=TOOLS, settings=settings, parts=2)
    assert alarm(r) is None and changes(r) == [20, 21]
    tools = r.controller.state.paths[0].sys["tool_life"]["groups"]["2"]["tools"]
    assert tools[0]["expired"] and tools[0]["used"] == pytest.approx(6.0, abs=0.01)
    assert tools[1]["used"] == pytest.approx(6.0, abs=0.01), "rapids are not counted"


def test_expired_tools_from_the_physics_feedback():
    ctl = controller(wrap("T9901 M06"), MILL, tools=TOOLS,
                     settings=life([{"group": 1, "life_uses": 10, "tools": [10, 11]}]))
    ctl.feedback(0, Feedback(0.0, tool_life_expired=[10]), 0)
    r = dry_run(ctl)
    assert alarm(r) is None and changes(r) == [11]
    assert 10 in ctl.state.paths[0].expired_tools


def test_lathe_group_calls_offsets_and_cancel():
    settings = life([{"group": 1, "life_uses": 1, "tools": [101, 102, 301]}])
    body = "T199\nG00 X50. Z5.\nT199\nG00 X40.\nT188\nG00 X30."
    lathe_tools = [ToolOffsetInit(n) for n in (1, 2, 3)]
    r = run(wrap(body), LATHE, tools=lathe_tools, settings=settings, parts=2)
    assert alarm(r) is None
    tc = [c.tool for c in r.commands[0] if c.kind == Kind.TOOL_CHANGE]
    assert [(t["number"], t["offset"]) for t in tc] == [(1, 1), (3, 1)]
    assert r.controller.state.paths[0].offset_number == 0, "T188 cancels the offset of the tool counted"
    r = run(wrap("T199\nT0202\nG00 X10."), LATHE, tools=lathe_tools, settings=settings)
    assert alarm(r) is None and r.controller.state.paths[0].sys["tool_life"]["current"] is None
    assert alarm(run(wrap("T188"), LATHE, tools=lathe_tools, settings=settings))["code"] == "PS0155"


def test_haas_tool_groups():
    settings = life([{"group": 1000, "life_uses": 1, "tools": [10, 11]}])
    body = "T1000 M06\nG43 H#3026 Z50.\n#100 = #4108\nT5 M06\nT1000 M06"
    r = run(wrap(body), HAAS_MILL, tools=TOOLS, settings=settings)
    assert alarm(r) is None and changes(r) == [10, 5, 11], "Haas counts every tool change"
    assert r.controller.state.paths[0].variables["c10000"][100] == 10.0
    assert alarm(run(wrap("T1001 M06"), HAAS_MILL, tools=TOOLS, settings=settings))["code"] == "TW0033"
    assert alarm(run(wrap("G10 L3\nG11"), HAAS_MILL, tools=TOOLS, settings=settings))["code"] == "TW0012"
