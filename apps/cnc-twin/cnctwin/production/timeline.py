"""Executed blocks of a run (``timeline.json``) and the time category of every tick of each block.

A timeline block is ``{index, op, program, line, block, path, kind, tick0, tick1}`` (docs/models/
production.md, "Inputs"), the block running ``[tick0, tick1)`` in base ticks. Optional fields the
writer may add: ``part`` (0-based part of a batch), ``tool`` (T number active), ``ticks``
(``{category: ticks}`` summing to the block's duration), ``in_cut_ticks`` (feed ticks in contact),
``pumps_on`` (coolant pump ids running), ``stop`` (M00/M01/M30 ...).

Categories (:data:`CATEGORIES`): the writer's ``ticks`` are used as given (an unknown category name
is kept under its own name); without them the whole block goes to the category of its ``kind``
(:data:`KIND_CATEGORY`), a feed block to ``cutting``/``air_cutting`` when ``in_cut_ticks`` is given
and to ``feed`` (contact not recorded) otherwise. Whatever the writer's categories do not cover
of the block's duration is booked as ``unclassified``, so every breakdown sums exactly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import InputError
from ..sanity.program import lex_line

CATEGORIES = ("cutting", "air_cutting", "feed", "rapid", "acceleration", "dwell", "tool_change", "spindle_ramp",
              "coolant_delay", "probing", "program_stop", "auxiliary", "path_wait", "unclassified")
KIND_CATEGORY = {
    "rapid": "rapid", "dwell": "dwell", "tool_change": "tool_change", "turret_index": "tool_change",
    "spindle": "spindle_ramp", "coolant": "coolant_delay", "probe": "probing", "tool_setter": "probing",
    "stop": "program_stop", "pallet": "auxiliary", "door": "auxiliary", "conveyor": "auxiliary",
    "chuck": "auxiliary", "message": "auxiliary", "alarm": "auxiliary", "sync": "path_wait", "mode": "auxiliary",
}
FEED_KINDS = ("linear", "arc")


@dataclass(frozen=True)
class TimelineBlock:
    """One executed block: ``[tick0, tick1)`` and its ticks by category (summing to the duration)."""

    index: int
    tick0: int
    tick1: int
    op: str = ""
    program: str = ""
    line: int = 0
    block: str = ""
    path: int = 0
    kind: str = "mode"
    part: int = 0
    tool: int | None = None
    ticks: tuple[tuple[str, int], ...] = ()
    pumps_on: tuple[str, ...] | None = None
    stop: str = ""

    @property
    def duration_ticks(self) -> int:
        return self.tick1 - self.tick0

    def categories(self) -> dict[str, int]:
        return dict(self.ticks)

    def codes(self, letter: str) -> list[str]:
        return lex_line(self.block, self.line).codes(letter)


def categorise(record: Mapping, duration: int) -> tuple[tuple[str, int], ...]:
    """Ticks by category of one timeline record (exact: they sum to ``duration``)."""
    given = record.get("ticks")
    booked: dict[str, int] = {}
    if isinstance(given, Mapping):
        for name, value in given.items():
            count = int(value)
            if count < 0:
                raise InputError(f"timeline block {record.get('index')}: negative ticks for {name}")
            if count:
                booked[str(name)] = booked.get(str(name), 0) + count
    else:
        kind = str(record.get("kind", "mode"))
        if kind in FEED_KINDS:
            in_cut = record.get("in_cut_ticks")
            if in_cut is not None:
                cut = max(0, min(int(in_cut), duration))
                booked = {"cutting": cut, "air_cutting": duration - cut}
            else:
                booked = {"feed": duration}
        else:
            booked = {KIND_CATEGORY.get(kind, "auxiliary"): duration}
    total = sum(booked.values())
    if total > duration:
        raise InputError(f"timeline block {record.get('index')} (line {record.get('line')}): categories book "
                         f"{total} ticks, the block lasts {duration}")
    if total < duration:
        booked["unclassified"] = booked.get("unclassified", 0) + duration - total
    return tuple((name, count) for name, count in ordered(booked).items() if count)


def ordered(values: Mapping[str, int]) -> dict[str, int]:
    """Categories in the order of :data:`CATEGORIES`, unknown names after them (sorted)."""
    known = [c for c in CATEGORIES if c in values]
    extra = sorted(c for c in values if c not in CATEGORIES)
    return {name: values[name] for name in known + extra}


def block_from_record(record: Mapping) -> TimelineBlock:
    try:
        tick0 = int(record["tick0"])
        tick1 = int(record["tick1"])
    except KeyError as error:
        raise InputError(f"timeline block {record.get('index')} has no {error.args[0]}") from None
    if tick1 < tick0:
        raise InputError(f"timeline block {record.get('index')}: tick1 {tick1} before tick0 {tick0}")
    pumps = record.get("pumps_on")
    tool = record.get("tool")
    return TimelineBlock(index=int(record.get("index", 0)), tick0=tick0, tick1=tick1, op=str(record.get("op", "")),
                         program=str(record.get("program", "")), line=int(record.get("line", 0)),
                         block=str(record.get("block", "")), path=int(record.get("path", 0)),
                         kind=str(record.get("kind", "mode")), part=int(record.get("part", 0)),
                         tool=None if tool is None else int(tool), ticks=categorise(record, tick1 - tick0),
                         pumps_on=None if pumps is None else tuple(str(p) for p in pumps),
                         stop=str(record.get("stop", "")))


def with_tools(blocks: list[TimelineBlock], initial_tool: int | None = None) -> list[TimelineBlock]:
    """Blocks with ``tool`` filled where the writer did not give it: the T number of the last tool
    change (M06 or a lathe T word on a tool-change block) on the same path."""
    current: dict[int, int | None] = {}
    result = []
    for block in blocks:
        if block.kind in ("tool_change", "turret_index"):
            words = lex_line(block.block, block.line).all("T")
            if words and words[0].value is not None:
                value = int(words[0].value)
                current[block.path] = value // 100 if value >= 100 and block.kind == "turret_index" else value
        tool = block.tool if block.tool is not None else current.get(block.path, initial_tool)
        if block.tool is not None:
            current[block.path] = block.tool
        stop = block.stop
        if block.kind == "stop" and not stop:
            codes = [c for c in lex_line(block.block, block.line).codes("M") if c in ("M0", "M1", "M2", "M30")]
            stop = {"M0": "M00", "M1": "M01", "M2": "M02", "M30": "M30"}.get(codes[0], "") if codes else ""
        result.append(TimelineBlock(block.index, block.tick0, block.tick1, block.op, block.program, block.line,
                                    block.block, block.path, block.kind, block.part, tool, block.ticks,
                                    block.pumps_on, stop))
    return result


@dataclass
class Timeline:
    blocks: list[TimelineBlock] = field(default_factory=list)
    tick_s: float = 1e-6

    def parts(self) -> list[int]:
        return sorted({b.part for b in self.blocks})

    def of_part(self, part: int) -> list[TimelineBlock]:
        return [b for b in self.blocks if b.part == part]


def read_timeline(run_dir, *, initial_tool: int | None = None) -> Timeline:
    """``timeline.json`` of a run directory (blocks in file order, tools filled in)."""
    path = Path(run_dir) / "timeline.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InputError(f"cannot read timeline: {error.strerror}", str(path)) from None
    records = document.get("blocks", []) if isinstance(document, Mapping) else document
    tick_s = float(document.get("tick_s", 1e-6)) if isinstance(document, Mapping) else 1e-6
    blocks = [block_from_record(r) for r in records]
    return Timeline(with_tools(blocks, initial_tool), tick_s)
