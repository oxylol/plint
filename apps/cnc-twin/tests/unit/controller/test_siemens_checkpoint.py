"""SINUMERIK 840D sl: checkpoint / resume inside a FOR loop (with local user data, R parameters and a
subprogram level) continues bit-identically; the dialect state round-trips completely."""

from __future__ import annotations

import io
import json

import numpy as np

from cnctwin.controller import Controller, ControllerState, dry_run
from test_siemens import controller

PROGRAM = """DEF INT K
DEF REAL ACC=0
G90 G17 G71 G94
T1 M6
S2000 M3 F400
G0 X0 Y0 Z5
FOR K=1 TO 12
ACC = ACC + SIN(K * 30)
G1 X=10*COS(K*30) Y=10*SIN(K*30) Z=-1
IF K==6
SUB3(K)
ENDIF
ENDFOR
R10 = ACC
CYCLE83(10, 0, 2, -12, , -5, , 2, 0, 0, 1, 0)
G0 Z=50+R10
M30
"""
SUB3 = """PROC SUB3(INT N)
DEF REAL W=N*2
G91 G1 X=W
G90
M17
"""


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
        elif path.waiting is not None:
            tick += 1000
            ctl.feedback(0, None, tick)
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


def small():
    return controller(PROGRAM, extra=[("SUB3.SPF", SUB3)])


def test_resume_inside_a_for_loop_is_bit_identical():
    reference, _ = _drive(small())
    assert reference and not small().state.paths[0].alarm
    checked = 0
    for k in (6, 9, 14, 17):
        first = small()
        part, tick = _drive(first, stop_after=k)
        state = first.state.paths[0]
        if "for" not in state.sys or not state.sys["for"]:
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
            elif path.waiting is not None:
                tick += 1000
                restored.feedback(0, None, tick)
            else:
                tick += 5000
        assert part + rest == reference
    assert checked >= 2, "checkpoints must fall inside the FOR loop"


def test_dialect_state_round_trip_is_complete():
    ctl = small()
    _drive(ctl, stop_after=10)
    blob, meta = _save(ctl)
    again = _load(blob, meta)
    assert ctl.state.paths[0] == again.paths[0]
    assert again.paths[0].sys["lud"][0]["vars"]["K"][0] == "INT"


def test_resume_with_a_pending_cycle_expansion():
    ctl = small()
    out, tick = _drive(ctl, stop_after=40)
    blob, meta = _save(ctl)
    restored = Controller(ctl.setup, _load(blob, meta))
    a = dry_run(ctl).commands[0]
    b = dry_run(restored).commands[0]
    assert a == b
