"""Cycle time of one part: per block, per tool, per program line and per operation, exactly.

Everything is summed in integer base ticks (architecture 5.1, 12 "Time"): the ticks of a block's
categories sum to its duration; per path, the block durations plus the gaps between consecutive
blocks (booked ``path_wait``: the path waits for another path, or the timeline has a hole) sum to
the path's span from its first block start to its last block end; an operation's machine time is
its longest path's span; the part's machine time is the sum of its operations. Seconds are only
formed at the output (ticks x tick_s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.errors import InputError
from .timeline import TimelineBlock, ordered


@dataclass
class Breakdown:
    """Ticks by category; :attr:`total` is their exact sum."""

    ticks: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.ticks.values())

    def add(self, category: str, count: int) -> None:
        if count:
            self.ticks[category] = self.ticks.get(category, 0) + int(count)

    def merge(self, other: "Breakdown") -> "Breakdown":
        for name, count in other.ticks.items():
            self.add(name, count)
        return self

    def to_dict(self, tick_s: float) -> dict:
        ticks = ordered(self.ticks)
        return {"total_ticks": self.total, "total_s": self.total * tick_s, "ticks": ticks,
                "s": {name: count * tick_s for name, count in ticks.items()}}


def block_breakdown(block: TimelineBlock) -> Breakdown:
    return Breakdown(dict(block.ticks))


@dataclass
class PathTime:
    path: int
    tick0: int
    tick1: int
    breakdown: Breakdown
    blocks: int

    @property
    def span_ticks(self) -> int:
        return self.tick1 - self.tick0


@dataclass
class OperationTime:
    op: str
    paths: dict[int, PathTime]
    critical_path: int

    @property
    def span_ticks(self) -> int:
        return self.paths[self.critical_path].span_ticks

    @property
    def breakdown(self) -> Breakdown:
        return self.paths[self.critical_path].breakdown


@dataclass
class LineTime:
    op: str
    program: str
    line: int
    block: str
    executions: int
    breakdown: Breakdown


@dataclass
class CycleTime:
    """Machine time of one part (all operations), broken down."""

    part: int
    operations: list[OperationTime]
    breakdown: Breakdown
    per_tool: dict[int | None, Breakdown]
    lines: list[LineTime]
    stops: dict[str, int]

    @property
    def machine_ticks(self) -> int:
        return sum(op.span_ticks for op in self.operations)


def path_time(blocks: list[TimelineBlock], path: int, per_tool: dict | None = None) -> PathTime:
    ordered_blocks = sorted(blocks, key=lambda b: (b.tick0, b.index))
    breakdown = Breakdown()
    previous_end = ordered_blocks[0].tick0
    previous_tool = ordered_blocks[0].tool
    for block in ordered_blocks:
        gap = block.tick0 - previous_end
        if gap < 0:
            raise InputError(f"timeline blocks overlap on path {path}: block {block.index} (line {block.line}) "
                             f"starts at tick {block.tick0}, before the previous block ends at {previous_end}")
        breakdown.add("path_wait", gap)
        if per_tool is not None:
            per_tool.setdefault(previous_tool, Breakdown()).add("path_wait", gap)
        for name, count in block.ticks:
            breakdown.add(name, count)
            if per_tool is not None:
                per_tool.setdefault(block.tool, Breakdown()).add(name, count)
        previous_end = block.tick1
        previous_tool = block.tool
    return PathTime(path, ordered_blocks[0].tick0, previous_end, breakdown, len(ordered_blocks))


def cycle_time(blocks: list[TimelineBlock], part: int = 0) -> CycleTime:
    """The breakdown of one part's executed blocks (any number of operations and paths)."""
    if not blocks:
        raise InputError(f"part {part}: the timeline has no blocks")
    operations: list[OperationTime] = []
    per_tool: dict[int | None, Breakdown] = {}
    order: list[str] = []
    by_op: dict[str, list[TimelineBlock]] = {}
    for block in blocks:
        if block.op not in by_op:
            order.append(block.op)
            by_op[block.op] = []
        by_op[block.op].append(block)
    total = Breakdown()
    for op in order:
        op_blocks = by_op[op]
        paths: dict[int, PathTime] = {}
        for path in sorted({b.path for b in op_blocks}):
            path_blocks = [b for b in op_blocks if b.path == path]
            paths[path] = path_time(path_blocks, path, per_tool)
        critical = max(sorted(paths), key=lambda p: (paths[p].span_ticks, -p))
        operation = OperationTime(op, paths, critical)
        operations.append(operation)
        total.merge(operation.breakdown)
    lines: dict[tuple, LineTime] = {}
    stops: dict[str, int] = {}
    for block in blocks:
        key = (block.op, block.program, block.line)
        if key not in lines:
            lines[key] = LineTime(block.op, block.program, block.line, block.block, 0, Breakdown())
        lines[key].executions += 1
        lines[key].breakdown.merge(block_breakdown(block))
        if block.stop:
            stops[block.stop] = stops.get(block.stop, 0) + 1
    line_list = sorted(lines.values(), key=lambda t: (order.index(t.op), t.program, t.line))
    return CycleTime(part, operations, total, per_tool, line_list, stops)


def cycle_to_dict(cycle: CycleTime, tick_s: float) -> dict:
    """JSON form (deterministic order) of a part's cycle time."""
    return {
        "part": cycle.part,
        "machine_ticks": cycle.machine_ticks,
        "machine_s": cycle.machine_ticks * tick_s,
        "breakdown": cycle.breakdown.to_dict(tick_s),
        "operations": [{"op": op.op, "span_ticks": op.span_ticks, "span_s": op.span_ticks * tick_s,
                        "critical_path": op.critical_path, "breakdown": op.breakdown.to_dict(tick_s),
                        "paths": [{"path": p.path, "tick0": p.tick0, "tick1": p.tick1, "blocks": p.blocks,
                                   "breakdown": p.breakdown.to_dict(tick_s)} for _, p in sorted(op.paths.items())]}
                       for op in cycle.operations],
        "tools": [{"tool": tool, "breakdown": breakdown.to_dict(tick_s)}
                  for tool, breakdown in sorted(cycle.per_tool.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))],
        "stops": dict(sorted(cycle.stops.items())),
    }


def lines_to_dict(cycle: CycleTime, tick_s: float) -> list[dict]:
    return [{"op": t.op, "program": t.program, "line": t.line, "block": t.block, "executions": t.executions,
             "breakdown": t.breakdown.to_dict(tick_s)} for t in cycle.lines]
