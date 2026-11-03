import io
import json

import numpy as np

from cnctwin.controller import Controller, ControllerState, dry_run
from cnctwin.controller.ir import Kind, ProbeResult
import copy

from harness import HAAS_MILL, MILL, controller, wrap

SMALL = copy.deepcopy(MILL)
SMALL["controller"]["lookahead_blocks"] = 3
SMALL_HAAS = copy.deepcopy(HAAS_MILL)
SMALL_HAAS["controller"]["lookahead_blocks"] = 3

PROGRAM = wrap("""G90 G54 G00 X0 Y0 Z5.
S1200 M03
#1 = 0
#2 = 0
WHILE [#1 LT 12] DO 1
#3 = 10. * COS[#1 * 30.]
#4 = 10. * SIN[#1 * 30.]
IF [#1 EQ 5] GOTO 100
G01 X#3 Y#4 Z-1. F300.
GOTO 200
N100 G02 X#3 Y#4 R10.
N200 #1 = #1 + 1
#2 = #2 + #3
END 1
G65 P9001 A#2
#500 = #2
G00 Z50.""")
MACRO = "O9001\nG91 G00 X#1\nG90\nM99\n"


def _drive(ctl, stop_after: int | None = None):
    """Produce/retire in a fixed rhythm (one command retired per call at a fixed tick step)."""
    out = []
    tick = 0
    while True:
        new = ctl.produce(0, tick)
        out.extend(new)
        path = ctl.state.paths[0]
        if path.waiting is not None and path.waiting["kind"] == "end":
            break
        if path.alarm is not None:
            break
        if path.buffer:
            tick += 2000
            ctl.retire(0, 1, tick)
        else:
            tick += 5000
        if stop_after is not None and len(out) >= stop_after:
            return out, tick
    return out, tick


def _save(ctl):
    arrays, meta = ctl.state.to_arrays()
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue(), json.dumps(meta)


def _load(blob, meta_text) -> ControllerState:
    arrays = dict(np.load(io.BytesIO(blob)))
    return ControllerState.from_arrays(arrays, json.loads(meta_text))


def test_checkpoint_resume_mid_while_is_bit_identical():
    reference, _ = _drive(controller(PROGRAM, SMALL, extra=[("lib.nc", MACRO)]))
    for k in (4, 8, 12):
        first = controller(PROGRAM, SMALL, extra=[("lib.nc", MACRO)])
        part, tick = _drive(first, stop_after=k)
        frames = first.state.paths[0].frames
        assert frames[0].loops and frames[0].loops[0].m == 1, "checkpoint must fall inside the WHILE loop"
        blob, meta = _save(first)
        restored = Controller(first.setup, _load(blob, meta))
        rest = []
        while True:
            new = restored.produce(0, tick)
            rest.extend(new)
            path = restored.state.paths[0]
            if path.waiting is not None and path.waiting["kind"] == "end" or path.alarm is not None:
                break
            if path.buffer:
                tick += 2000
                restored.retire(0, 1, tick)
            else:
                tick += 5000
        assert part + rest == reference
        assert restored.state.paths[0].variables["c500"][0] is not None


def test_state_round_trip_is_complete():
    ctl = controller(PROGRAM, SMALL, extra=[("lib.nc", MACRO)])
    _drive(ctl, stop_after=9)
    blob, meta = _save(ctl)
    again = _load(blob, meta)
    blob2, meta2 = _save(Controller(ctl.setup, again))
    assert json.loads(meta) == json.loads(meta2)
    assert ctl.state.paths[0] == again.paths[0]


def test_resume_while_waiting_for_a_probe_feedback():
    program = wrap("G90 G00 X0 Y0 Z0\nG31 X50. F100.\n#100 = #5061\nG00 X#100 Y1.")
    ctl = controller(program)
    produced = ctl.produce(0, 0)
    assert produced[-1].kind == Kind.PROBE and ctl.state.paths[0].waiting["kind"] == "probe"
    blob, meta = _save(ctl)
    restored = Controller(ctl.setup, _load(blob, meta))
    result = dry_run(restored, probe=lambda p, cmd: ProbeResult(True, {"X": 0.02, "Y": 0.0, "Z": 0.0}, 0.0))
    last = [c for c in result.commands[0] if c.kind == Kind.RAPID][-1]
    assert round(last.target["X"] * 1000, 6) == 20.0


def test_digest_mismatch_is_refused():
    ctl = controller(wrap("G00 X1."))
    state = ctl.state
    other = controller(wrap("G00 X2."))
    try:
        Controller(other.setup, state)
    except ValueError as error:
        assert "program texts" in str(error)
    else:
        raise AssertionError("a state of another program was accepted")


def test_haas_state_round_trip_with_pending_cycle_steps():
    program = wrap("G90 G00 X50. Y50. Z1.\nG13 I10. K20. Q5. D1 Z-2. F200.\nG00 Z10.")
    from cnctwin.controller import ToolOffsetInit
    ctl = controller(program, SMALL_HAAS, tools=[ToolOffsetInit(1, radius_m=0.005)])
    ctl.produce(0, 0)
    assert ctl.state.paths[0].pending, "the cycle expansion must still be pending (buffer full)"
    blob, meta = _save(ctl)
    restored = Controller(ctl.setup, _load(blob, meta))
    a = dry_run(ctl).commands[0]
    b = dry_run(restored).commands[0]
    assert a == b
