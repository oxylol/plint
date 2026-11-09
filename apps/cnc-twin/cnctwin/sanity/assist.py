"""The controller's dry run as an aid to the static scan (docs/models/sanity.md "Controller assist").

The scan reads the program as written; what it cannot evaluate it takes from the controller's own
execution of the program, line by line:

* address values with macro expressions (``G43 H#3026``, ``G41 D#7``, ``Z[#1 - 2]``): the values the
  executor evaluated for that execution of the line;
* the motion of lines the scan does not expand: tool centre point control (G43.4/G43.5, Haas G234),
  dynamic work offsets (G254) and tilted planes (G68.2), whose commands carry the tool tip and axis in the
  work frame; the lathe cycles (G70-G76, G90/G92/G94) and the Haas lathe drilling cycles through their
  cycle providers; the Haas mill pattern cycles (G12/G13/G150, G47, the G70-G72 bolt hole patterns of the
  active canned cycle); macro calls (G65) and subprograms the scan cannot read, with the motion of every
  block they run.

The dry run is the controller's ideal machine (cnctwin.controller.dry_run) with an ideal probe against the
stock as set up (cnctwin.controller.domain.stock_probe). A line's k-th visit by the scan is matched with the
k-th execution of that program line by the controller: without macro flow (GOTO, WHILE) the two orders are
the same; with it, the scan's single pass through a loop takes the first execution. When the controller
stops with an alarm, the lines after it have no execution and the scan reads them alone. When the
controller cannot be built or run for the operation, the scan runs alone and ``Assist.note`` says why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from ..controller.ir import Command, Kind, Space

MOTION_KINDS = (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE, Kind.TOOL_SETTER)


@dataclass
class Execution:
    """One execution of one program line by the controller."""

    seq: int
    depth: int
    words: dict[str, float]                      # evaluated address values (program units)
    commands: list[Command] = field(default_factory=list)   # motion of the block and of the blocks it called


@dataclass
class Assist:
    executions: dict[tuple[str, int], list[Execution]] = field(default_factory=dict)
    letters: tuple[str, ...] = ()                # the path's program axis letters (order of the snapshot lists)
    names: tuple[str, ...] = ()                  # their machine axis names (command target keys)
    lathe: bool = False
    alarm: dict | None = None                    # the alarm the dry run stopped at, if any
    note: str = ""                               # why there is no assist (empty when there is one)

    def execution(self, program: str, line: int, occurrence: int) -> Execution | None:
        runs = self.executions.get((program, line))
        if runs is None or occurrence >= len(runs):
            return None
        return runs[occurrence]

    @property
    def available(self) -> bool:
        return bool(self.executions)


def build_assist(log: list[dict], commands: list[Command], *, letters=(), names=(), lathe=False,
                 alarm=None) -> Assist:
    """Index the executor's block log and the dry run's commands by program line and execution."""
    by_seq: dict[int, list[Command]] = {}
    for command in commands:
        if command.kind in MOTION_KINDS:
            by_seq.setdefault(int(command.block_seq), []).append(command)
    executions: dict[tuple[str, int], list[Execution]] = {}
    n = len(log)
    for j, record in enumerate(log):
        moves = list(by_seq.get(record["seq"], []))
        k = j + 1
        while k < n and log[k]["depth"] > record["depth"]:
            moves.extend(by_seq.get(log[k]["seq"], []))
            k += 1
        executions.setdefault((str(record["program"]), int(record["line"])), []).append(
            Execution(int(record["seq"]), int(record["depth"]), dict(record.get("words") or {}), moves))
    return Assist(executions, tuple(letters), tuple(names), bool(lathe), alarm)


def controller_point(assist: Assist, command: Command, point, shift) -> np.ndarray:
    """Program coordinates (of the active work offset, with the G52/G92 shift the scan keeps) of a point in
    the space of ``command``: a TIP-space command is already in the work frame; a machine-space one is
    taken back by the offsets of its modal snapshot (mills: ``work_offset_m`` and the length on the tool
    axis; lathes: ``program_offset_m``, X as a radius). ``point`` is a mapping of machine axis names or an
    (x, y, z) triple (arc centres)."""
    out = np.full(3, np.nan)
    if command.space == Space.TIP and command.tip_target_m is not None:
        values = point if not isinstance(point, dict) else [point.get(n, np.nan) for n in ("X", "Y", "Z")]
        return np.asarray(values, dtype=float) + np.asarray(shift, dtype=float)
    modal = command.modal or {}
    offsets = modal.get("program_offset_m" if assist.lathe else "work_offset_m") or [0.0] * len(assist.letters)
    length = 0.0 if assist.lathe else float(modal.get("length_offset_m") or 0.0)
    axis_letter = str(modal.get("tool_axis_letter", "Z"))
    for k, letter in enumerate("XYZ"):
        if letter not in assist.letters:
            if assist.lathe and letter == "Y":
                out[k] = 0.0
            continue
        i = assist.letters.index(letter)
        value = point.get(assist.names[i]) if isinstance(point, dict) else point[k]
        if value is None:
            continue
        offset = float(offsets[i]) if i < len(offsets) else 0.0
        out[k] = float(value) - offset - (length if letter == axis_letter else 0.0) + float(shift[k])
    return out


def controller_assist(project, op_index: int = 0, *, program_text: str | None = None) -> Assist:
    """The assist of one operation of a (resolved or raw) project: its controller's dry run (of
    ``program_text`` when given, else the operation's program)."""
    from ..controller import Controller, ProgramSource, dry_run
    from ..controller.domain import setup_from_project, stock_probe
    from .si import si_view

    view = si_view(project)
    try:
        ctx = SimpleNamespace(project=view, tick_s=1e-6)
        state = SimpleNamespace(tools=None, clock=SimpleNamespace(op_index=op_index))
        setup = setup_from_project(ctx, state, op_index)
        if program_text is not None:
            setup.programs = [ProgramSource(p.name, program_text, p.path, p.main) if (p.main and p.path == 0) else p
                              for p in setup.programs]
        controller = Controller(setup)
    except Exception as error:      # noqa: BLE001  (an aid: the scan runs alone, the note says why)
        return Assist(note=f"the controller could not be built for this operation: {type(error).__name__}: {error}")
    executor = controller.executors[0]
    executor.block_log = []
    try:
        probe = stock_probe(view, op_index)
    except Exception as error:      # noqa: BLE001
        probe = None
        note = f"no ideal probe ({type(error).__name__}: {error})"
    else:
        note = ""
    try:
        result = dry_run(controller, probe=probe)
    except Exception as error:      # noqa: BLE001
        return Assist(note=f"the controller's dry run failed: {type(error).__name__}: {error}")
    assist = build_assist(executor.block_log, result.commands[0], letters=[a.letter for a in executor.axes],
                          names=[a.name for a in executor.axes], lathe=bool(getattr(executor.handler, "T", False)),
                          alarm=result.alarms[0])
    assist.note = note
    return assist


__all__ = ["Assist", "Execution", "build_assist", "controller_assist", "controller_point"]
