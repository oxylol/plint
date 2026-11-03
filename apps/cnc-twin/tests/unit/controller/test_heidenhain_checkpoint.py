"""HEIDENHAIN Klartext: checkpoint / resume inside a CALL LBL REP section repeat (and inside a label
subprogram with Q, QL and QR parameters) continues bit-identically."""

from __future__ import annotations

import io
import json

import numpy as np

from cnctwin.controller import Controller, ControllerState
from test_heidenhain import controller, pgm

PROGRAM = pgm("""TOOL CALL 1 Z S2000 F400
L X+0 Y+0 Z+5 R0 FMAX M3
Q1 = 0
QL1 = 0
LBL 1
Q1 = Q1 + 1
QL1 = QL1 + Q1
L X+10*Q1 FMAX
L IY+2 F400
CALL LBL 2
CALL LBL 1 REP 5
QR3 = QL1
L Z+50 FMAX
M30
LBL 2
L IX+0.5 F400
LBL 0""".replace("L X+10*Q1 FMAX", "Q2 = 10 * Q1\nL X+Q2 FMAX"))


def _drive(ctl, stop_after: int | None = None):
    out = []
    tick = 0
    while True:
        new = ctl.produce(0, tick)
        out.extend(new)
        path = ctl.state.paths[0]
        if (path.waiting is not None and path.waiting["kind"] == "end") or path.alarm is not None:
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


def _load(blob, meta) -> ControllerState:
    return ControllerState.from_arrays(dict(np.load(io.BytesIO(blob))), json.loads(meta))


def make():
    return controller(PROGRAM)


def test_resume_inside_a_label_repeat_is_bit_identical():
    reference, _ = _drive(make())
    assert reference and len([c for c in reference if c.kind.value == "linear"]) == 12
    checked = 0
    for k in (5, 8, 11, 15):
        first = make()
        part, tick = _drive(first, stop_after=k)
        state = first.state.paths[0]
        if not state.sys.get("rep"):
            continue
        checked += 1
        blob, meta = _save(first)
        restored = Controller(first.setup, _load(blob, meta))
        rest = []
        while True:
            new = restored.produce(0, tick)
            rest.extend(new)
            path = restored.state.paths[0]
            if (path.waiting is not None and path.waiting["kind"] == "end") or path.alarm is not None:
                break
            if path.buffer:
                tick += 2000
                restored.retire(0, 1, tick)
            else:
                tick += 5000
        assert part + rest == reference
        assert restored.state.paths[0].variables["QR"][3] == 21.0
    assert checked >= 2, "checkpoints must fall inside the section repeat"


def test_round_trip_keeps_q_ql_qr_and_the_repeat_counter():
    ctl = make()
    _drive(ctl, stop_after=9)
    blob, meta = _save(ctl)
    again = _load(blob, meta)
    assert ctl.state.paths[0] == again.paths[0]
    assert "QL1" in again.paths[0].variables and again.paths[0].sys["rep"]
