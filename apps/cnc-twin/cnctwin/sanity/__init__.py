"""Program sanity checks: the machinist's checks a control would not catch (docs/models/sanity.md).

Use :func:`check_operation` for one operation of a resolved project, or :func:`check_program` with
an explicit :class:`SanityContext`. Each problem becomes a :class:`cnctwin.core.findings.Finding`
of category ``sanity`` with the program line, the block, the tool-tip position and the chain of
cause (trace events for the lines it rests on). The checks read the controller's command stream
when one is given (:class:`cnctwin.controller.ir.Command` with modal snapshots), and otherwise
their own static modal scan of the program text (Fanuc and Haas dialects).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..controller.ir import Command, Kind, Space
from ..core.errors import InputError
from ..core.findings import Category, Finding, FindingList, Severity
from ..core.trace import TraceLog
from .assist import Assist, controller_assist
from .context import Box, CuttingData, SanityContext, ToolFacts, normalise_offset
from .geometry import Engagement, estimate
from .program import Line, Program, ToolComment, lex, parse_tool_comment
from .rules import RULE_IDS, Issue, run_rules
from .scan import SUPPORTED_DIALECTS, ScanResult, scan
from .si import get, si_view, text

__all__ = ["check_operation", "check_program", "build_program", "to_findings", "SanityReport", "SanityContext",
           "ToolFacts", "CuttingData", "Box", "Program", "Line", "ToolComment", "Issue", "RULE_IDS",
           "SUPPORTED_DIALECTS", "parse_tool_comment", "scan", "lex"]


@dataclass
class SanityReport:
    """Result of one program's checks."""

    issues: list[Issue]
    findings: list[Finding]
    program: Program
    engagement: Engagement
    trace: TraceLog
    context: SanityContext = None
    assist: Assist | None = None        # the controller's dry run the scan consulted (note: why there is none)

    def add_to(self, findings: FindingList) -> list[Finding]:
        """Append the findings to a run's list (ids follow the list's numbering)."""
        return [findings.add(f) for f in self.findings]

    def safe(self) -> bool:
        return all(f.severity.rank < Severity.ERROR.rank for f in self.findings)

    def by_rule(self) -> dict[str, list[Issue]]:
        grouped: dict[str, list[Issue]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.rule, []).append(issue)
        return grouped


def build_program(text: str, name: str, ctx: SanityContext, *, commands: list[Command] | None = None,
                  subprograms: dict[str, str] | None = None, assist: Assist | None = None) -> Program:
    """The program view: lines, commands (controller's or the own scan's) and tool-tip positions."""
    lines = lex(text)
    own: ScanResult | None = None
    if ctx.dialect in SUPPORTED_DIALECTS:
        own = scan(text, name, ctx, subprograms=subprograms, assist=assist if commands is None else None)
    if commands is None:
        if own is None:
            return Program(name, ctx.dialect, _kind(ctx), text, lines, [], np.zeros((0, 3)), np.zeros((0, 3)),
                           "none", None, [], {name: lines})
        machine = "lathe" if own.end_state.get("machine") == "lathe" else "mill"
        return Program(name, ctx.dialect, machine, text, lines, own.commands, own.tip_start, own.tip_end, "scan",
                       own, [], dict(own.lines))
    merged = _merge(list(commands), own)
    starts, ends = _tips(merged, ctx)
    machine = "lathe" if (own is not None and own.end_state.get("machine") == "lathe") else _kind(ctx)
    remapped = _remap(own, merged) if own is not None else None
    texts = dict(own.lines) if own is not None else {name: lines}
    return Program(name, ctx.dialect, machine, text, lines, merged, starts, ends, "controller", remapped, [], texts)


def check_program(text: str, name: str, ctx: SanityContext, *, commands: list[Command] | None = None,
                  subprograms: dict[str, str] | None = None, restart_analysis=None, trace: TraceLog | None = None,
                  timeline=None, tick_s: float = 1e-6, assist: Assist | None = None) -> SanityReport:
    """Run every rule on one program; ``restart_analysis`` is the controller's restart analysis
    (records with ``line``, ``program``, ``issues: [{group, message, severity}]``) when available;
    ``assist`` the controller's dry run of the program (cnctwin.sanity.assist) for what the scan cannot
    evaluate."""
    program = build_program(text, name, ctx, commands=commands, subprograms=subprograms, assist=assist)
    engagement = estimate(program, ctx)
    issues = run_rules(program, ctx, engagement, restart_analysis=restart_analysis)
    findings, trace = to_findings(issues, ctx, program, trace=trace, timeline=timeline, tick_s=tick_s)
    report = SanityReport(issues, findings, program, engagement, trace, ctx)
    report.assist = assist
    return report


def check_operation(project, op_index: int = 0, *, program_text: str | None = None,
                    commands: list[Command] | None = None, subprograms: dict[str, str] | None = None,
                    restart_analysis=None, trace: TraceLog | None = None, timeline=None, project_dir=None,
                    stock_box_M: Box | None = None, clamps_M: list[Box] | None = None,
                    use_controller: bool = True) -> SanityReport:
    """Checks of operation ``op_index`` of a resolved project (the program text from
    ``operations[i].program_text``, else read from ``project_dir/operations[i].program``). With
    ``use_controller`` (and no ``commands``) the controller's dry run of the operation evaluates what the
    static scan cannot (cnctwin.sanity.assist)."""
    ctx = SanityContext.from_project(project, op_index, stock_box_M=stock_box_M, clamps_M=clamps_M)
    operation = si_view(get(si_view(project), "operations", op_index))
    name = text(get(operation, "program"), f"op{op_index}")
    if program_text is None:
        program_text = text(get(operation, "program_text"))
    if program_text is None:
        if project_dir is None:
            raise InputError(f"operation {ctx.op_name or op_index}: no program text (program_text) and no project "
                             "directory to read the program from")
        path = Path(project_dir) / name
        try:
            program_text = path.read_text(encoding="utf-8-sig")
        except OSError as error:
            raise InputError(f"cannot read program: {error.strerror}", str(path)) from None
    tick_s = float(get(si_view(project), "physics", "tick_s", default=1e-6) or 1e-6)
    assist = None
    if use_controller and commands is None and ctx.dialect in SUPPORTED_DIALECTS:
        assist = controller_assist(project, op_index, program_text=program_text)
    return check_program(program_text, name, ctx, commands=commands, subprograms=subprograms,
                         restart_analysis=restart_analysis, trace=trace, timeline=timeline, tick_s=tick_s,
                         assist=assist)


def to_findings(issues: list[Issue], ctx: SanityContext, program: Program | None = None, *,
                trace: TraceLog | None = None, timeline=None, tick_s: float = 1e-6) -> tuple[list[Finding], TraceLog]:
    """Findings of category ``sanity`` for the issues (trace events record the lines they rest on).

    With a ``timeline`` (blocks with ``program``, ``line``, ``start_tick``) each finding carries the
    tick at which its line first runs; static findings otherwise carry tick 0."""
    trace = trace if trace is not None else TraceLog()
    first_tick = _first_ticks(timeline)
    findings: list[Finding] = []
    for issue in issues:
        tick = first_tick.get((issue.program, issue.line), 0)
        time_s = tick * tick_s
        parents = []
        for program_name, line, why in issue.causes:
            if line:
                cause_tick = first_tick.get((program_name, line), 0)
                parents.append(trace.record(cause_tick * tick_s, cause_tick, "sanity", f"sanity.cause:{why}",
                                            {}, [], line))
        event = trace.record(time_s, tick, "sanity", f"finding:sanity:{issue.rule}", issue.quantities, parents,
                             issue.line)
        findings.append(Finding(category=Category.SANITY, severity=issue.severity,
                                message=f"{issue.rule}: {issue.message}", time_s=time_s, tick=tick, domain="sanity",
                                op=ctx.op_name, program=issue.program, line=issue.line, block=issue.block,
                                pos_W_m=issue.pos_W_m, pos_M_m=_machine_position(issue, ctx, program),
                                quantities=dict(issue.quantities), cause_chain=trace.chain_ids(event)))
    return findings, trace


def _first_ticks(timeline) -> dict[tuple[str, int], int]:
    ticks: dict[tuple[str, int], int] = {}
    blocks = timeline.get("blocks", []) if isinstance(timeline, dict) else (timeline or [])
    for block in blocks:
        key = (str(_field(block, "program", "")), int(_field(block, "line", 0)))
        start = int(_field(block, "start_tick", 0))
        if key not in ticks or start < ticks[key]:
            ticks[key] = start
    return ticks


def _field(record, name, default):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _machine_position(issue: Issue, ctx: SanityContext, program: Program | None):
    if issue.pos_W_m is None or issue.command is None or program is None or program.machine_kind != "mill":
        return None
    modal = program.commands[issue.command].modal or {}
    offset = ctx.work_offsets_m.get(normalise_offset(modal.get("work_offset", "G54")))
    if offset is None:
        return None
    return tuple(float(p + o) for p, o in zip(issue.pos_W_m, offset))


def _kind(ctx: SanityContext) -> str:
    return "lathe" if ctx.machine_kind == "lathe" else "mill"


# ----------------------------------------------------------------------------- controller command stream

def _merge(commands: list[Command], own: ScanResult | None) -> list[Command]:
    """Controller commands with modal snapshots completed from the own scan of the same line (the
    controller's values win; keys it does not give come from the scan)."""
    if own is None:
        return commands
    by_line: dict[tuple[str, int], dict] = {}
    for command in own.commands:
        by_line[(command.ref.program, command.ref.line)] = command.modal or {}
    merged = []
    for command in commands:
        scan_modal = by_line.get((command.ref.program, command.ref.line))
        if scan_modal is None:
            merged.append(command)
            continue
        clone = copy.copy(command)
        clone.modal = {**scan_modal, **(command.modal or {})}
        merged.append(clone)
    return merged


def _tips(commands: list[Command], ctx: SanityContext) -> tuple[np.ndarray, np.ndarray]:
    """Tool-tip positions of controller commands (program coordinates of each command's work offset).

    As in the own scan, the program starts with the tool at the machine home and a tool change
    without targets leaves Z at the home (tool-change) height."""
    n = len(commands)
    starts = np.full((n, 3), np.nan)
    ends = np.full((n, 3), np.nan)
    position = np.full(3, np.nan)
    started = False
    for i, command in enumerate(commands):
        if not started and command.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE):
            home = {axis: ctx.home_m[axis] for axis in "XYZ" if axis in ctx.home_m}
            position = _tip_of(_with_target(command, home), ctx, position)
            started = True
        starts[i] = position
        if command.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE):
            position = _tip_of(command, ctx, position)
        elif command.kind == Kind.TOOL_CHANGE and started:
            target = command.target or ({"Z": ctx.home_m["Z"]} if "Z" in ctx.home_m else {})
            position = _tip_of(_with_target(command, target), ctx, position)
        ends[i] = position
    return starts, ends


def _with_target(command: Command, target: dict) -> Command:
    clone = copy.copy(command)
    clone.space = Space.MACHINE
    clone.tip_target_m = None
    clone.target = dict(target)
    return clone


def _tip_of(command: Command, ctx: SanityContext, previous: np.ndarray) -> np.ndarray:
    if command.space == Space.TIP and command.tip_target_m is not None:
        return np.asarray(command.tip_target_m, dtype=float)
    modal = command.modal or {}
    offset = ctx.work_offsets_m.get(normalise_offset(modal.get("work_offset", "G54")))
    tip = previous.copy()
    if offset is None:
        return np.full(3, np.nan)
    length = 0.0
    if modal.get("length_comp") and modal.get("length_offset"):
        register = ctx.tools.get(modal.get("length_offset"))
        length = register.length_offset_m if register is not None and register.length_offset_m is not None else math.nan
        if modal.get("length_comp_code") == "G44":
            length = -length
    for k, axis in enumerate("XYZ"):
        if axis in command.target:
            tip[k] = command.target[axis] - offset[k] - (length if axis == "Z" else 0.0)
    return tip


def _remap(own: ScanResult, commands: list[Command]) -> ScanResult:
    """The own scan's structures (holes, sections, G28 events, offsets set) re-indexed to the
    controller's command list by program line and kind (k-th occurrence)."""
    slots: dict[tuple[str, int, str], list[int]] = {}
    for index, command in enumerate(commands):
        slots.setdefault((command.ref.program, command.ref.line, command.kind.value), []).append(index)
    used: dict[tuple[str, int, str], int] = {}
    mapping: dict[int, int] = {}
    for index, command in enumerate(own.commands):
        key = (command.ref.program, command.ref.line, command.kind.value)
        candidates = slots.get(key, [])
        k = used.get(key, 0)
        if k < len(candidates):
            mapping[index] = candidates[k]
            used[key] = k + 1

    def remap(index: int) -> int:
        return mapping.get(index, -1)

    def remap_after(index: int) -> int:
        for k in range(index, len(own.commands)):
            if k in mapping:
                return mapping[k]
        return len(commands)

    result = copy.copy(own)
    result.commands = commands
    result.holes = [copy.copy(h) for h in own.holes if remap(h.command) >= 0]
    for hole, original in zip(result.holes, [h for h in own.holes if remap(h.command) >= 0]):
        hole.command = remap(original.command)
    result.sections = []
    for section in own.sections:
        moved = copy.copy(section)
        moved.command = remap(section.command) if section.command >= 0 else -1
        if section.command < 0 or moved.command >= 0:
            result.sections.append(moved)
    result.g28 = [dict(event, command=remap_after(event["command"])) for event in own.g28]
    result.offsets_set = {k: (line, remap_after(c)) for k, (line, c) in own.offsets_set.items()}
    result.length_measured = {k: (line, remap_after(c)) for k, (line, c) in own.length_measured.items()}
    result.radius_set = {k: (line, remap_after(c)) for k, (line, c) in own.radius_set.items()}
    result.macro_calls = [(remap_after(c), line) for c, line in own.macro_calls]
    return result
