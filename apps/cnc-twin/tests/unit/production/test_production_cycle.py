"""Exact cycle-time breakdown per block, line, tool and operation."""

import pytest

from cnctwin.core.errors import InputError
from cnctwin.production import cycle_time, cycle_to_dict, with_tools

from production_testkit import block


def test_categories_sum_exactly_and_gaps_are_path_waits():
    blocks = with_tools([
        block(0, 0, 1000, "tool_change", text="T1 M06"),
        block(1, 1000, 5000, ticks={"cutting": 3000, "acceleration": 500}),
        block(2, 5200, 6000, "dwell", text="G04 P800"),
        block(3, 6000, 6100, "tool_change", text="T2 M06"),
        block(4, 6100, 9100, "linear", in_cut_ticks=1000),
        block(5, 9100, 9200, "stop", text="M00"),
        block(6, 9200, 9300, "linear"),
    ])
    cycle = cycle_time(blocks)
    ticks = cycle.breakdown.ticks
    assert cycle.machine_ticks == 9300 == cycle.breakdown.total
    assert ticks == {"tool_change": 1100, "cutting": 4000, "acceleration": 500, "unclassified": 500,
                     "path_wait": 200, "dwell": 800, "air_cutting": 2000, "program_stop": 100, "feed": 100}
    per_tool = {tool: b.ticks for tool, b in cycle.per_tool.items()}
    assert per_tool[1] == {"tool_change": 1000, "cutting": 3000, "acceleration": 500, "unclassified": 500,
                           "path_wait": 200, "dwell": 800}
    assert sum(b.total for b in cycle.per_tool.values()) == cycle.machine_ticks
    assert cycle.stops == {"M00": 1}
    document = cycle_to_dict(cycle, 1e-6)
    assert document["machine_s"] == pytest.approx(9.3e-3)
    assert list(document["breakdown"]["ticks"])[:3] == ["cutting", "air_cutting", "feed"]


def test_paths_and_operations():
    blocks = [block(0, 0, 4000, "rapid", path=0), block(1, 4000, 6000, "rapid", path=0),
              block(2, 0, 3000, "linear", path=1, ticks={"cutting": 3000}),
              block(3, 3000, 4500, "sync", path=1),
              block(4, 10000, 12000, "rapid", op="OP20")]
    cycle = cycle_time(blocks)
    first, second = cycle.operations
    assert first.critical_path == 0 and first.span_ticks == 6000 and second.span_ticks == 2000
    assert cycle.machine_ticks == 8000 == cycle.breakdown.total
    assert first.paths[1].breakdown.ticks == {"cutting": 3000, "path_wait": 1500}


def test_lines_executed_several_times():
    blocks = [block(0, 0, 100, line=10, ticks={"cutting": 100}), block(1, 100, 300, line=10, ticks={"cutting": 200}),
              block(2, 300, 350, "rapid", line=11)]
    lines = cycle_time(blocks).lines
    assert [(t.line, t.executions, t.breakdown.total) for t in lines] == [(10, 2, 300), (11, 1, 50)]


def test_broken_timelines_are_rejected():
    with pytest.raises(InputError):
        cycle_time([block(0, 0, 100, "rapid"), block(1, 50, 120, "rapid")])
    with pytest.raises(InputError):
        block(0, 0, 100, ticks={"cutting": 150})
    with pytest.raises(InputError):
        cycle_time([])
