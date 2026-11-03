"""The hook interfaces as the cycles / compensation / TCP / probing agents will use them."""

import math

import pytest

from cnctwin.controller import hooks
from cnctwin.controller.hooks import PathItem, Step
from cnctwin.controller.ir import Command, Kind, Space
from harness import MILL, alarm, controller, mm, moves, run, wrap


def _restore(table: dict, key: str, saved) -> None:
    """Put back the provider a test replaced (the cycles/comp packages register real ones)."""
    if saved is None:
        table.pop(key, None)
    else:
        table[key] = saved


class ShiftFilter:
    """A minimal look-ahead compensation: holds one item, offsets every motion by +D in Y."""

    def __init__(self, held=None):
        self.held = held or []

    def push(self, ex, item: PathItem):
        if item.kind in ("linear", "rapid", "arc"):
            item.end[1] += item.comp_radius_m if item.comp != "G40" else 0.0
        out, self.held = self.held, [item]
        if item.comp == "G40" and item.kind != "command":
            out, self.held = out + self.held, []
            ex.end_comp()
        return out

    def flush(self, ex, reason):
        out, self.held = self.held, []
        return out

    def to_data(self):
        from cnctwin.controller import serial
        return {"held": serial.dump(self.held)}


class ShiftFactory:
    def create(self, ex):
        return ShiftFilter()

    def from_data(self, ex, data):
        from cnctwin.controller import serial
        return ShiftFilter([serial.load(PathItem, d) for d in data["held"]])


def test_compensation_filter_hook_with_look_ahead_keeps_program_order():
    saved = hooks.COMPENSATION.get("cutter")        # the providers of the cycles/comp packages are restored
    hooks.register_compensation("cutter", ShiftFactory(), replace=True)
    try:
        r = run(wrap("G90 G00 X0 Y0\nG41 G01 X10. Y0 D1 F100.\nM08\nX20.\nG40 X30."))
        assert alarm(r) is None
        kinds = [c.kind for c in r.commands[0]]
        assert kinds.index(Kind.COOLANT) > kinds.index(Kind.LINEAR)
        m = moves(r)
        assert mm(m[1], "Y") == pytest.approx(5.0) and mm(m[2], "Y") == pytest.approx(5.0)
        assert mm(m[3], "Y") == 0.0
        assert r.controller.state.paths[0].comp_state is None
    finally:
        _restore(hooks.COMPENSATION, "cutter", saved)


class FlatTransform:
    def to_command(self, ex, item: PathItem) -> Command:
        cmd = ex.default_command(item)
        cmd.space = Space.TIP
        cmd.tip_target_m = tuple(item.end[:3])
        cmd.tip_axis = (0.0, 0.0, 1.0)
        return cmd


def test_transform_hook_turns_items_into_tip_space_commands():
    saved = hooks.TRANSFORMS.get("fanuc:G43.4")        # the providers of the cycles/comp packages are restored
    hooks.register_transform("fanuc:G43.4", FlatTransform(), replace=True)
    try:
        r = run(wrap("G43.4 H1 G00 X5. Y6. Z7.\nG49 G00 X0"))
        m = moves(r)
        assert alarm(r) is None and m[0].space == Space.TIP and m[0].tip_target_m == pytest.approx((0.005, 0.006, 0.007))
        assert mm(m[0], "Z") == pytest.approx(107.0)
        assert m[1].space == Space.MACHINE
    finally:
        _restore(hooks.TRANSFORMS, "fanuc:G43.4", saved)


class TwoHoles:
    """A canned cycle provider: rapid to the hole, feed to Z, rapid back to R (G99) or the initial level (G98)."""

    def expand(self, ex, call):
        r_level = call.data["R"]
        bottom = call.data["Z"]
        back = call.initial_level if call.return_to_initial else r_level
        x, y = call.position["X"], call.position["Y"]
        return [Step("rapid", {"X": x, "Y": y}), Step("rapid", {"Z": r_level}),
                Step("linear", {"Z": bottom}, feed=call.data["F"], exact_stop=True),
                Step("dwell", dwell_s=call.data.get("P", 0.0)), Step("rapid", {"Z": back})]


def test_canned_cycle_hook_receives_modal_cycle_data_and_repeats():
    saved = hooks.CYCLES.get("G82")        # the providers of the cycles/comp packages are restored
    hooks.register_cycle("G82", TwoHoles(), replace=True)
    try:
        r = run(wrap("G90 G00 X0 Y0 Z20.\nG98 G82 X10. Y10. Z-5. R2. P500 F120.\nX20.\nG91 G99 X5. K2\nG80 G90 G00 Z50."))
        assert alarm(r) is None
        m = moves(r)
        bottoms = [c for c in m if c.kind == Kind.LINEAR]
        assert [mm(c, "X") for c in bottoms] == [10.0, 20.0, 25.0, 30.0]
        assert all(mm(c, "Z") == -5.0 for c in bottoms)
        dwells = [c.dwell_s for c in r.commands[0] if c.kind == Kind.DWELL]
        assert dwells == pytest.approx([0.5] * 4)
        retracts = [mm(m[i + 1], "Z") for i, c in enumerate(m) if c.kind == Kind.LINEAR]
        assert retracts == [20.0, 20.0, 2.0, 2.0]
        assert bottoms[0].smoothing.corner == "exact_stop"
    finally:
        _restore(hooks.CYCLES, "G82", saved)


class ProbeCycle:
    def start(self, ex, call):
        return [Step("probe", {"X": call.data["X"]}, feed=0.001, provider="test:probe", data={"n": 1},
                     probe={"skip": 1})]

    def resume(self, ex, data, feedback):
        hit = ex.si("X", ex.s.skip["X"])
        return [Step("rapid", {"X": hit - 0.001})]


def test_probe_hook_buffer_stops_and_continues_from_data():
    from cnctwin.controller.ir import ProbeResult
    saved = hooks.PROBES.get("G37")
    hooks.register_probe("test:probe", ProbeCycle(), replace=True)
    hooks.register_probe("G37", ProbeCycle(), replace=True)
    try:
        r = run(wrap("G90 G00 X0\nG37 X30.\nG00 Y1."),
                probe=lambda p, cmd: ProbeResult(True, {"X": 0.0123, "Y": 0.0, "Z": 0.0}, 0.0))
        assert alarm(r) is None
        m = moves(r)
        assert m[1].kind == Kind.PROBE and mm(m[2], "X") == pytest.approx(11.3)
    finally:
        del hooks.PROBES["test:probe"]
        _restore(hooks.PROBES, "G37", saved)


def test_default_g37_measures_and_updates_the_length_offset():
    from cnctwin.controller.ir import ProbeResult
    from cnctwin.controller import ToolOffsetInit
    body = "G90 G54 G43 H1 G00 X0 Y0 Z50.\nG37 Z0\n#100 = #2201\nG00 Z50."
    r = run(wrap(body), tools=[ToolOffsetInit(1, length_m=0.100)], offsets={"G54": [0.0, 0.0, -0.3]},
            setter=lambda p, cmd: ProbeResult(True, {"X": 0.0, "Y": 0.0, "Z": -0.3 + 0.1 + 0.0004}, 0.0))
    assert alarm(r) is None
    assert r.controller.state.paths[0].variables["c100"][0] == pytest.approx(100.4)
    setters = [c for c in r.commands[0] if c.kind == Kind.TOOL_SETTER]
    assert setters and setters[0].setter["offset"] == 1
    assert alarm(run(wrap("G37 Z0")))["code"] == "PS0081"


def test_tool_life_hook_selects_sister_tools():
    class Sisters:
        def select(self, ex, number):
            return 11 if number == 1 else number

        def expired(self, ex, numbers):
            ex.s.sys["expired_seen"] = list(numbers)

    saved = hooks.TOOL_LIFE.get("default")        # the providers of the cycles/comp packages are restored
    hooks.register_tool_life(Sisters(), replace=True)
    try:
        r = run(wrap("T1 M06"))
        changes = [c for c in r.commands[0] if c.kind == Kind.TOOL_CHANGE]
        assert changes[0].tool["number"] == 11
    finally:
        _restore(hooks.TOOL_LIFE, "default", saved)


def test_restart_search_keeps_modal_state_without_output():
    ctl = controller(wrap("G90 G54 S800 M03\nM08\nG00 X1.\nN50 G01 X2. F100.\nG00 X3."))
    ex = ctl.executors[0]
    count = ex.search(lambda e: e.s.last_n == 50)
    assert count > 0 and ex.s.buffer == []
    summary = ex.s.restart_summary
    assert summary["spindle"].spindle["omega_rad_s"] < 0 and summary["coolant"].coolant["pumps"]["flood"]
    assert ex.s.pos["X"] == 2000
    rest = ctl.produce(0, 0)
    assert [c.kind for c in rest][:1] == [Kind.RAPID]


def test_duplicate_registration_is_refused():
    class A:
        def expand(self, ex, call):
            return []

    hooks.register_cycle("G89", A(), dialect="fanuc", replace=True)
    try:
        with pytest.raises(ValueError):
            hooks.register_cycle("G89", A(), dialect="fanuc")
    finally:
        del hooks.CYCLES["fanuc:G89"]
    assert math.isfinite(1.0) and MILL["type"] == "vertical_mill"


def test_chamfer_and_corner_words_route_to_the_corner_provider():
    from cnctwin.controller.lexer import parse_block
    from cnctwin.controller.dialects import fanuc
    block = parse_block("G01 X10. ,R2. F100.", 1, fanuc.build_fanuc.__globals__["LexerRules"](
        letters=frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), comma_letters=frozenset("CRA")))
    assert [w.letter for w in block.words] == ["G", "X", ",R", "F"]
    corner_provider = hooks.COMPENSATION.pop("corner", None)
    try:
        assert alarm(run(wrap("G01 X10. ,R2. F100.\nY10.")))["code"] == "TW0002"
    finally:
        _restore(hooks.COMPENSATION, "corner", corner_provider)

    class CornerFilter:
        seen = []

        def __init__(self, held=None):
            self.held = held or []

        def push(self, ex, item):
            if item.corner:
                CornerFilter.seen.append(item.corner)
                self.held.append(item)
                return []
            out = self.held + [item]
            self.held = []
            if item.kind != "command":
                ex.end_comp()
            return out

        def flush(self, ex, reason):
            out, self.held = self.held, []
            return out

        def to_data(self):
            from cnctwin.controller import serial
            return {"held": serial.dump(self.held)}

    class Corners:
        def create(self, ex):
            return CornerFilter()

        def from_data(self, ex, data):
            from cnctwin.controller import serial
            return CornerFilter([serial.load(PathItem, d) for d in data["held"]])

    saved = hooks.COMPENSATION.get("corner")        # the providers of the cycles/comp packages are restored
    hooks.register_compensation("corner", Corners(), replace=True)
    try:
        r = run(wrap("G01 X10. ,C1.5 F100.\nY10.\nX0"))
        m = moves(r)
        assert alarm(r) is None and CornerFilter.seen == [{",C": pytest.approx(0.0015)}]
        assert [round(c.target["X"] * 1000, 3) for c in m] == [10.0, 10.0, 0.0]
        assert r.controller.state.paths[0].comp_state is None
    finally:
        _restore(hooks.COMPENSATION, "corner", saved)
