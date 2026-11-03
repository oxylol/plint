"""Program restart for the controller core (docs/dialects.md "Cycles and compensation", program restart).

* :func:`restart` - FANUC program restart (fanuc-30ibplus-om-2019 4.9, pp. 1206-1208) of one path at
  a sequence number (``n``, the ``frequency``-th time it is reached) or at a line of the main program
  (``line``): the program is interpreted from its head without output (``ex.search``; modal G codes,
  offsets, coordinate systems, tool, spindle and coolant state are rebuilt), then the restart restores
  the M/S/T state per ``restart_output`` (parameter 7300 MOU/MOA: ``none`` - the operator does it in MDI,
  nothing is output; ``last`` - the last tool change, spindle and coolant commands; ``all`` - every one
  met during the search, in order) and moves to the restart point axis by axis in the order of parameter
  7310 (``restart_axis_order``) at the dry-run feed (``restart_feed_m_s``, parameter 1410), in machine
  coordinates; then the program goes on at the target block. P type: the approach starts where the tool
  is (the operator retracted it); Q type: the path is first RESET (rewound, compensation and cycles
  cancelled; offsets and tool data kept) and the approach starts at the reference position, the twin's
  "program starting point".
  PS0060 when the block is not reached before the program end; TW0024 when the restart point lies in
  rigid tapping (the spindle-axis synchronisation cannot be rebuilt by positioning).
* :func:`restart_analysis` - dangerous restarts for the sanity domain: for every candidate block (each
  tool change of the main program, or the lines given) the modal state the program reaches it with is
  compared with a fresh power-on start at that block (what an operator gets by starting there without
  the restart function); a group that differs and is not restated by the section before its first move
  is an issue (``{"group", "message", "severity", "value", "fresh", "first_move_line"}``).

Assumptions: restarts are counted in the main program (subprogram blocks are reached through it); the
restated-before-first-move test is lexical (G codes of the group, M06/T, M03/M04/M05, M07/M08/M09 in the
blocks up to and including the first block with an axis word); macro statements are not evaluated by it.
No randomness; no bus fields.
"""

from __future__ import annotations

import dataclasses

from . import hooks
from .alarms import ControllerAlarm, register_alarm_codes
from .cycles.common import params as cycle_params
from .hooks import Step
from .ir import Kind

FANUC_ALARMS = {"restart_not_found": "PS0060", "restart_in_cycle": "TW0024"}
HAAS_ALARMS = {"restart_not_found": "TW0036", "restart_in_cycle": "TW0024"}
TEXTS = {"PS0060": "SEQUENCE NUMBER NOT FOUND", "TW0024": "PROGRAM RESTART INSIDE RIGID TAPPING",
         "TW0036": "RESTART BLOCK NOT FOUND"}
RESTORED_KINDS = (Kind.TOOL_CHANGE, Kind.SPINDLE, Kind.COOLANT)

M_GROUPS = {1: "motion", 2: "plane", 3: "distance", 5: "feed_mode", 6: "units", 7: "radius_comp", 8: "length_comp",
            9: "cycle", 10: "return_level", 11: "scaling", 12: "macro_modal_call", 13: "css", 14: "work_offset",
            15: "exact_stop", 16: "rotation", 17: "polar", 22: "mirror"}
T_GROUPS = {1: "motion", 2: "css", 5: "feed_mode", 6: "units", 7: "radius_comp", 10: "cycle", 12: "macro_modal_call",
            14: "work_offset", 16: "plane", 21: "polar"}
WARNING_GROUPS = ("return_level", "css", "macro_modal_call", "exact_stop", "coolant")
SPINDLE_M = (3, 4, 5)
COOLANT_M = (7, 8, 9)


class _Recorder(dict):
    """``restart_summary`` stand-in that also keeps every command met during the search (``all``)."""

    def __init__(self):
        super().__init__()
        self.sequence: list = []

    def __setitem__(self, key, value):
        self.sequence.append(value)
        super().__setitem__(key, value)


def _target_predicate(ex, target: dict):
    """True when the next block to interpret is the target (or the program has ended)."""
    s = ex.s
    seen = {"count": 0, "seq": None}
    main = ex.store.mains[ex.ctx.path]

    def predicate(_ex) -> bool:
        if s.waiting is not None and s.waiting.get("kind") == "end":
            target["ended"] = True
            return True
        if s.pending or s.pending_call is not None or not s.frames:
            return False
        frame = s.frames[-1]
        program = ex.store.programs[frame.program]
        if frame.pc >= len(program.blocks):
            return False
        block = program.blocks[frame.pc]
        if target.get("line") is not None:
            return frame.program == main and block.line == int(target["line"])
        if block.n != int(target["n"]):
            return False
        if seen["seq"] != s.block_seq:
            seen["seq"] = s.block_seq
            seen["count"] += 1
        return seen["count"] >= int(target.get("frequency", 1))

    return predicate


class Restart:
    """The ``RESTART`` provider: ``reconstruct(ex, target)`` scans to the target and returns the steps that
    restore the state and approach the restart point."""

    def reconstruct(self, ex, target: dict) -> list[Step]:
        s = ex.s
        p = cycle_params(ex)
        start = dict(s.mach)
        recorder = _Recorder()
        s.restart_summary = recorder
        target = dict(target)
        try:
            ex.search(_target_predicate(ex, target))
            if s.alarm is not None:
                return []
            if target.get("ended"):
                what = f"line {target['line']}" if target.get("line") is not None else f"N{target['n']}"
                raise ControllerAlarm("restart_not_found", f"program restart: {what} not reached")
            if s.sys.get("rigid_tap"):
                raise ControllerAlarm("restart_in_cycle", "the restart block lies in rigid tapping (M29 ... G80)")
            s.suppress = True
            try:
                ex.flush_comp("program restart")       # look-ahead held by compensation stays unreleased
            finally:
                s.suppress = False
        finally:
            sequence = list(recorder.sequence)
            last = dict(recorder)
            s.restart_summary = dict(recorder)
        restore = self._restore(str(p["restart_output"]), sequence, last)
        goal = dict(s.mach)
        s.mach = start
        s.pos = ex.program_from_machine(start)
        steps = [Step("command", command=dataclasses.replace(cmd)) for cmd in restore]
        order = [l for l in str(p["restart_axis_order"]) if l in ex.by_letter]
        order += [a.letter for a in ex.axes if a.letter not in order]
        feed = float(p["restart_feed_m_s"])
        position = dict(start)
        for letter in order:
            if goal.get(letter, 0) == position.get(letter, 0):
                continue
            position[letter] = goal[letter]
            steps.append(Step("linear", {letter: ex.si(letter, goal[letter])}, machine=True, feed=feed,
                              feed_mode="per_min", exact_stop=True, modal={"program_restart": True}))
        return steps

    def _restore(self, mode: str, sequence: list, last: dict) -> list:
        if mode == "none":
            return []
        if mode == "all":
            return [cmd for cmd in sequence if cmd.kind in RESTORED_KINDS]
        if mode == "last":
            return [last[kind.value] for kind in RESTORED_KINDS if kind.value in last]
        raise ControllerAlarm("unsupported_function", f"restart_output {mode!r} (none, last or all)")


RESTART = Restart()


def restart(controller, *, n: int | None = None, line: int | None = None, frequency: int = 1, kind: str = "P",
            path: int = 0) -> dict:
    """Prepare a program restart of ``path`` at sequence number ``n`` (its ``frequency``-th occurrence) or at
    ``line`` of the main program; P or Q type. The restore and approach steps are queued, so the next
    ``produce`` (or a dry run) continues from the restart. Returns ``{"blocks_scanned", "line", "n",
    "steps"}``; an alarm (PS0060, TW0024 ...) is left on the path like any other."""
    if (n is None) == (line is None):
        raise ValueError("restart needs exactly one of n= or line=")
    if kind not in ("P", "Q"):
        raise ValueError("restart kind is 'P' or 'Q'")
    ex = controller.executors[path]
    s = ex.s
    provider = hooks.RESTART.get("default")
    if provider is None:
        raise ControllerAlarm("unsupported_function", "program restart")
    if kind == "Q":                                 # RESET, the machine at its reference position
        ex.reset(s.interp_tick)
        s.mach = {a.letter: int(a.home) for a in ex.axes}
        s.pos = ex.program_from_machine(s.mach)
    before = s.block_seq
    try:
        steps = provider.reconstruct(ex, {"n": n, "line": line, "frequency": int(frequency), "kind": kind})
    except ControllerAlarm as alarm:
        ex._stop_alarm(alarm)
        steps = []
    if s.alarm is None:
        ex.queue_steps(steps, front=True)
    frame = s.frames[-1] if s.frames else None
    block = ex.store.programs[frame.program].blocks[frame.pc] if frame is not None and s.alarm is None else None
    return {"blocks_scanned": s.block_seq - before, "line": block.line if block else None,
            "n": block.n if block else None, "steps": len(steps)}


# --------------------------------------------------------------------------- dangerous-restart analysis

def _lathe(ex) -> bool:
    return bool(getattr(ex.handler, "T", False))


def _gcode(word) -> str | None:
    if word.letter != "G" or word.value is None:
        return None
    value = float(word.value)
    return f"G{int(value):02d}" if value.is_integer() else f"G{word.text.lstrip('0') or word.text}"


def modal_groups(ex) -> dict[str, str]:
    """The restart-relevant state of a path, group name -> value text."""
    s = ex.s
    lathe = bool(getattr(ex.handler, "T", False))
    out = {name: str(s.modal.get(str(group), "")) for group, name in (T_GROUPS if lathe else M_GROUPS).items()}
    out["work_offset"] = s.wcs
    if not lathe:
        code = s.modal.get("8", "G49")
        out["length_comp"] = code if code == "G49" else f"{code} H{s.h_number}"
    out["cycle"] = (s.cycle or {}).get("code") or out.get("cycle") or "G80"
    out["tcp"] = s.tcp or ""
    out["local_shift"] = " ".join(f"{k}{v}" for k, v in sorted(s.g52.items()) if v) or ""
    direction = s.spindle.direction
    out["spindle"] = "M05" if direction == 0 else f"{'M03' if direction > 0 else 'M04'} S{s.spindle.s_value:g}"
    out["coolant"] = " ".join(sorted(k for k, on in s.pumps.items() if on)) or "off"
    out["tool"] = f"T{s.tool}"
    return out


def _restated(ex, program, pc: int) -> tuple[set[str], int | None]:
    """Groups the section from ``pc`` restates up to and including its first block with an axis word."""
    gcodes = ex.dialect.gcodes
    names = T_GROUPS if _lathe(ex) else M_GROUPS
    axes = {a.letter for a in ex.axes} | ({"U", "W"} if _lathe(ex) else set())
    restated: set[str] = set()
    for block in program.blocks[pc:]:
        moved = False
        for word in block.words:
            code = _gcode(word)
            if code is not None:
                spec = gcodes.get(code)
                if spec is not None and spec.group in names:
                    restated.add(names[spec.group])
                    if spec.group == 8:
                        restated.add("tcp")
                if code == "G52":
                    restated.add("local_shift")
                if code in ("G54.1", "G154") or (spec is not None and spec.kind == "wcs"):
                    restated.add("work_offset")
            elif word.letter == "M" and word.value is not None:
                m = int(float(word.value))
                if m == 6 and not _lathe(ex):
                    restated.add("tool")
                if m in SPINDLE_M:
                    restated.add("spindle")
                if m in COOLANT_M:
                    restated.add("coolant")
            elif word.letter == "T" and _lathe(ex):
                restated.add("tool")
            elif word.letter in axes:
                moved = True
        if moved:
            return restated, block.line
    return restated, None


def _candidates(ex, program, lines) -> list[int]:
    if lines is not None:
        wanted = {int(v) for v in lines}
        return [pc for pc, block in enumerate(program.blocks) if block.line in wanted]
    out = []
    for pc, block in enumerate(program.blocks):
        for word in block.words:
            if (word.letter == "M" and word.value is not None and int(float(word.value)) == 6 and not _lathe(ex)) \
                    or (word.letter == "T" and _lathe(ex)):
                out.append(pc)
                break
    return out


def restart_analysis(setup, *, path: int = 0, lines=None) -> list[dict]:
    """Dangerous-restart analysis of one path of a :class:`~cnctwin.controller.ControllerSetup` (records as
    ``cnctwin.sanity.rules._restart_from_controller`` reads them)."""
    from .controller import Controller

    fresh = modal_groups(Controller(setup).executors[path])
    ctl = Controller(setup)
    ex = ctl.executors[path]
    s = ex.s
    main = ex.store.mains[ex.ctx.path]
    program = ex.store.programs[main]
    records = []
    for pc in _candidates(ex, program, lines):
        block = program.blocks[pc]
        target = {"line": block.line}
        ex.search(_target_predicate(ex, target))
        if s.alarm is not None or target.get("ended"):
            break
        reached = modal_groups(ex)
        restated, first_move = _restated(ex, program, pc)
        issues = []
        for group, value in reached.items():
            if value == fresh.get(group) or group in restated:
                continue
            severity = "warning" if group in WARNING_GROUPS else "error"
            where = f"before its first move at line {first_move}" if first_move is not None else "in the section"
            issues.append({"group": group, "value": value, "fresh": fresh.get(group), "severity": severity,
                           "first_move_line": first_move,
                           "message": f"{group} is '{value}' when the program reaches line {block.line} but a start "
                                      f"there runs with the power-on '{fresh.get(group)}', not restated {where}"})
        records.append({"line": block.line, "program": program.file, "n": block.n, "issues": issues})
        # step over the candidate block so the next search does not stop on it again
        ex.search(lambda _e, seq=s.block_seq: s.block_seq > seq and not s.pending and s.pending_call is None
                  or (s.waiting is not None and s.waiting.get("kind") == "end"))
    return records


def register(replace: bool = True) -> None:
    """(Re-)register the restart provider and the alarm numbers."""
    hooks.register_restart(RESTART, replace=replace)
    register_alarm_codes("fanuc", FANUC_ALARMS, TEXTS)
    register_alarm_codes("haas", HAAS_ALARMS, TEXTS)


register()

__all__ = ["Restart", "RESTART", "restart", "restart_analysis", "modal_groups", "register"]
