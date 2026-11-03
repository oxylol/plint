import pytest

from cnctwin.controller.ir import Kind
from harness import MILL, controller, run, wrap


def test_ready_ticks_follow_the_block_processing_time():
    ctl = controller(wrap("G01 F6000.\n" + "".join(f"X{0.01 * k:.2f}\n" for k in range(1, 31))))
    early = ctl.produce(0, 0)
    assert 0 < len(early) < 20 and early[-1].ready_tick <= ctl.params["horizon_ticks"] + 1000
    produced = early + ctl.produce(0, 1_000_000)
    linear = [c for c in produced if c.kind == Kind.LINEAR]
    assert len(produced) == 20
    gaps = {b.ready_tick - a.ready_tick for a, b in zip(linear, linear[1:])}
    assert gaps == {1000}


def test_full_buffer_stops_the_interpreter_until_motion_retires():
    ctl = controller(wrap("G01 F6000.\n" + "".join(f"X{0.01 * k:.2f}\n" for k in range(1, 41))))
    produced = ctl.produce(0, 0) + ctl.produce(0, 1_000_000)
    assert len(produced) == 20 and ctl.state.paths[0].blocked
    last = produced[-1].ready_tick
    assert ctl.produce(0, 10_000_000) == []
    ctl.retire(0, 1, 50_000)
    more = ctl.produce(0, 50_000)
    assert len(more) == 1 and more[0].ready_tick == 50_000 + 1000
    assert last < 50_000


def test_data_starvation_short_segments_at_high_feed():
    body = "G01 F10000.\n" + "".join(f"X{0.05 * k:.2f}\n" for k in range(1, 201))
    r = run(wrap(body))
    linear = [(c, s, e) for c, s, e in zip(r.commands[0], r.start_ticks[0], r.end_ticks[0]) if c.kind == Kind.LINEAR]
    starved = sum(1 for c, s, e in linear[1:] if s == c.ready_tick and s > 0)
    assert starved > 150
    move_ticks = round(0.05e-3 / (10000e-3 / 60.0) / 1e-6)
    assert move_ticks < 1000
    duration = linear[-1][2] - linear[0][1]
    assert duration >= 199 * 1000


def test_macro_statements_cost_time():
    fast = controller(wrap("G00 X1.\nG00 X2."))
    slow = controller(wrap("G00 X1.\n" + "#100 = SIN[30] * COS[30] + SQRT[2] * #100\n" * 20 + "G00 X2."))
    a = [c for c in fast.produce(0, 0) if c.kind == Kind.RAPID]
    b = [c for c in slow.produce(0, 0) if c.kind == Kind.RAPID]
    gap_fast = a[1].ready_tick - a[0].ready_tick
    gap_slow = b[1].ready_tick - b[0].ready_tick
    p = slow.params
    statement = p["macro_statement_s"] + 10 * p["macro_term_s"]
    assert gap_slow - gap_fast == pytest.approx(20 * statement / 1e-6, rel=0.2)


def test_macro_loop_without_motion_advances_time_and_respects_the_horizon():
    ctl = controller(wrap("#100 = 0\nWHILE [#100 LT 100000] DO 1\n#100 = #100 + 1\nEND 1\nG00 X1."))
    assert ctl.produce(0, 0) == []
    interp = ctl.state.paths[0].interp_tick
    assert 0 < interp <= ctl.params["horizon_ticks"] + 100
    assert ctl.produce(0, interp) == []
    assert ctl.state.paths[0].interp_tick > interp


def test_buffer_stopping_blocks_resume_at_the_feedback_tick():
    r = run(wrap("G00 X10.\nM00\nG00 X20."))
    stop_index = next(i for i, c in enumerate(r.commands[0]) if c.kind == Kind.STOP)
    after = r.commands[0][stop_index + 1]
    assert after.ready_tick >= r.end_ticks[0][stop_index]


def test_cycle_time_of_a_simple_program():
    r = run(wrap("G00 X-300.\nG01 X-200. F6000."))
    assert r.end_tick == pytest.approx(round(0.3 / 0.5 / 1e-6) + round(0.1 / 0.1 / 1e-6), abs=3000)
    assert MILL["controller"]["block_processing_ms"] == 1.0
