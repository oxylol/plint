"""The machinist's checks: each rule reads the program view, the context and the engagement
estimate and reports :class:`Issue` objects (turned into findings of category ``sanity``).

Every rule is a function ``rule(check)`` registered in :data:`RULES` with its id; the ids,
conditions, severities and sources are tabulated in docs/models/sanity.md. Rules that repeat
within one tool section (a whole pocket without coolant) report once per section, at the first
line, with the count and the worst value in the quantities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..controller.ir import Kind
from ..core.findings import Severity
from ..units import INCH_M
from .context import Box, SanityContext, ToolFacts, normalise_offset
from .geometry import Engagement
from .program import COMMENT_KINDS, Program, ToolComment, parse_tool_comment
from .scan import TAP_CYCLES, SUPPORTED_DIALECTS

MOTION = (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE)
FEED = (Kind.LINEAR, Kind.ARC, Kind.PROBE)
CUT = (Kind.LINEAR, Kind.ARC)
WATER_BASED = ("emulsion", "semi_synthetic", "synthetic", "water")


@dataclass
class Issue:
    """One rule violation, tied to a program line and the lines it rests on (``causes``)."""

    rule: str
    severity: Severity
    message: str
    line: int
    program: str
    block: str = ""
    command: int | None = None
    quantities: dict[str, float] = field(default_factory=dict)
    causes: list[tuple[str, int, str]] = field(default_factory=list)   # (program, line, why)
    pos_W_m: tuple[float, float, float] | None = None


class Check:
    """Everything one run of the rules reads, and the issues it collects."""

    def __init__(self, program: Program, ctx: SanityContext, engagement: Engagement, restart_analysis=None):
        self.program = program
        self.ctx = ctx
        self.eng = engagement
        self.restart_analysis = restart_analysis
        self.issues: list[Issue] = []
        self.commands = program.commands
        self.scan = program.scan

    # --------------------------------------------------------------- helpers
    def modal(self, i: int) -> dict:
        return self.commands[i].modal or {}

    def tool(self, i: int) -> ToolFacts | None:
        number = self.modal(i).get("tool")
        return self.ctx.tools.get(number) if number is not None else None

    def section(self, i: int) -> int:
        return int(self.modal(i).get("section", 0))

    def tip(self, i: int, end: bool = True) -> np.ndarray:
        return (self.program.tip_end_W_m if end else self.program.tip_start_W_m)[i]

    def ref(self, i: int):
        return self.commands[i].ref

    def indices(self, kinds) -> list[int]:
        return [i for i, c in enumerate(self.commands) if c.kind in kinds]

    def issue(self, rule: str, severity: Severity, message: str, *, command: int | None = None,
              line: int | None = None, program: str | None = None, block: str | None = None,
              quantities: dict | None = None, causes: list | None = None, pos=None) -> Issue:
        if command is not None:
            ref = self.ref(command)
            line = ref.line if line is None else line
            program = ref.program if program is None else program
            block = ref.block if block is None else block
            if pos is None:
                tip = self.tip(command)
                pos = tuple(float(v) for v in tip) if np.all(np.isfinite(tip)) else None
        program = program or self.program.name
        if block is None:
            text_line = self.program.line(line or 0, program)
            block = text_line.text.strip() if text_line is not None else ""
        found = Issue(rule, severity, message, int(line or 0), program, block, command,
                      {k: float(v) for k, v in sorted((quantities or {}).items())}, list(causes or []), pos)
        self.issues.append(found)
        return found

    def stock_top(self, offset: str) -> float | None:
        box = self.ctx.stock_box_W(offset)
        return None if box is None else box.top

    def clamp_boxes(self, offset: str) -> list[Box]:
        return self.ctx.clamps_W(offset) or []

    def is_mill(self) -> bool:
        return self.program.machine_kind == "mill"

    def lines_between(self, program: str, first: int, last: int):
        lines = self.program.texts.get(program, self.program.lines)
        return [line for line in lines if first <= line.number <= last]

    def unit_m(self, i: int | None = None) -> float:
        units = self.modal(i).get("units", "mm") if i is not None and self.commands else "mm"
        return 1e-3 if units == "mm" else INCH_M


def _mm(value_m: float) -> str:
    return f"{value_m * 1e3:.3f} mm"


def _first_by_section(check: Check, indices: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for i in indices:
        grouped.setdefault(check.section(i), []).append(i)
    return grouped


# =============================================================================== rules

def rule_feed_missing(check: Check) -> None:
    bad = [i for i in check.indices(FEED) if not check.modal(i).get("feed")]
    for section, group in _first_by_section(check, bad).items():
        first = group[0]
        modal = check.modal(first)
        cause = "no F word has been programmed" if not modal.get("feed_line") else \
            f"F0 programmed at line {modal.get('feed_line')}"
        check.issue("feed-missing", Severity.ERROR,
                    f"feed move without a feed rate ({cause}); {len(group)} feed move(s) in this tool section",
                    command=first, quantities={"count": len(group)},
                    causes=[(check.ref(first).program, int(modal.get("feed_line") or 0), "feed rate")]
                    if modal.get("feed_line") else [])


def rule_feed_inherited(check: Check) -> None:
    first_feed: dict[int, int] = {}
    for i in check.indices(CUT):
        first_feed.setdefault(check.section(i), i)
    for section, i in sorted(first_feed.items()):
        if section == 0:
            continue
        modal = check.modal(i)
        feed = modal.get("feed")
        if not feed or modal.get("feed_section", section) >= section:
            continue
        feed_line = int(modal.get("feed_line") or 0)
        unit = check.unit_m(i)
        per_min = modal.get("feed_mode") == "per_min"
        value = feed * 60.0 / unit if per_min else feed / unit
        check.issue("feed-inherited", Severity.WARNING,
                    f"first feed move of T{modal.get('tool')} has no F of its own: it runs at F{value:g} programmed "
                    f"at line {feed_line} for T{modal.get('feed_tool')}",
                    command=i, quantities={"feed_m_s" if per_min else "feed_m_per_rev": feed},
                    causes=[(check.ref(i).program, feed_line, "feed rate of the previous tool")])


def rule_spindle_stopped(check: Check) -> None:
    bad = [i for i in check.indices(CUT) if check.eng.cut[i] and not _spindle_on(check.modal(i))]
    for section, group in _first_by_section(check, bad).items():
        tool = check.tool(group[0])
        if tool is not None and not tool.rotating and check.is_mill():
            continue
        modal = check.modal(group[0])
        check.issue("spindle-stopped", Severity.ERROR,
                    f"cutting with the spindle stopped (T{modal.get('tool')}: {len(group)} cutting move(s) without "
                    f"M03/M04 since the tool change at line {modal.get('tool_line')})",
                    command=group[0], quantities={"count": len(group)},
                    causes=[(check.ref(group[0]).program, int(modal.get("spindle_line") or 0), "spindle state")])


def _spindle_on(modal: dict) -> bool:
    if "spindle_on" in modal:
        return bool(modal["spindle_on"])
    return bool(modal.get("spindle_rad_s"))


def rule_spindle_direction(check: Check) -> None:
    if not check.is_mill():
        return
    reported: set[int] = set()
    for i in check.indices(CUT):
        if not check.eng.cut[i]:
            continue
        modal = check.modal(i)
        omega = modal.get("spindle_rad_s") or 0.0
        tool = check.tool(i)
        if omega == 0.0 or tool is None or tool.hand not in ("right", "left") or not tool.rotating:
            continue
        section = check.section(i)
        if section in reported:
            continue
        expected = "M03" if tool.hand == "right" else "M04"
        turning = "M03" if omega < 0 else "M04"
        cycle = (modal.get("cycle_info") or {}).get("code")
        message = None
        if cycle in TAP_CYCLES:
            cycle_hand = "left" if cycle == "G74" else "right"
            if cycle_hand != tool.hand:
                message = f"{cycle} taps a {cycle_hand}-hand thread but T{modal.get('tool')} is a {tool.hand}-hand tap"
            elif turning != expected:
                message = (f"{cycle} with the spindle in {turning}: a {tool.hand}-hand tap needs {expected} before the "
                           "cycle")
        elif turning != expected:
            message = (f"spindle turns {turning} but T{modal.get('tool')} is a {tool.hand}-hand tool (needs {expected})")
        if message is not None:
            reported.add(section)
            check.issue("spindle-direction", Severity.ERROR, message, command=i,
                        quantities={"spindle_rad_s": omega},
                        causes=[(check.ref(i).program, int(modal.get("spindle_line") or 0), "spindle direction")])


def rule_tool_unknown(check: Check) -> None:
    seen: set[tuple[int | None, int]] = set()
    for i, command in enumerate(check.commands):
        if command.kind not in (Kind.TOOL_CHANGE, Kind.MODE) or not command.tool:
            continue
        number = command.tool.get("number")
        if command.kind == Kind.MODE and not command.tool.get("preselect"):
            continue
        key = (number, command.ref.line)
        if key in seen:
            continue
        seen.add(key)
        if number is None and command.kind == Kind.TOOL_CHANGE:
            check.issue("tool-unknown", Severity.ERROR, "tool change without a tool number (no T word before M06)",
                        command=i)
        elif number is not None and number not in check.ctx.tools and number != 0:
            check.issue("tool-unknown", Severity.ERROR, f"T{number} is not in the tool table "
                        f"(tools: {', '.join(f'T{n}' for n in sorted(check.ctx.tools)) or 'none'})", command=i,
                        quantities={"tool_number": number})


def rule_tool_comment(check: Check) -> None:
    program = check.program
    ctx = check.ctx
    unit = 1e-3 if str(ctx.p("default_units")) == "mm" else INCH_M
    changes = [i for i, c in enumerate(check.commands) if c.kind == Kind.TOOL_CHANGE and c.ref.program == program.name]
    first_change_line = check.ref(changes[0]).line if changes else len(program.lines) + 1
    if check.commands:
        unit = check.unit_m(0)
    for line in program.lines:
        if line.number >= first_change_line:
            break
        for text in line.comments:
            comment = parse_tool_comment(text, line.number, unit)
            if comment.number is not None and comment.describes_tool:
                _compare_comment(check, comment, comment.number, None)
    motion_lines = sorted({c.ref.line for c in check.commands if c.kind in MOTION and c.ref.program == program.name})
    for i in changes:
        number = (check.commands[i].tool or {}).get("number")
        if number is None:
            continue
        line_no = check.ref(i).line
        before = max([m for m in motion_lines if m < line_no], default=first_change_line - 1)
        after = min([m for m in motion_lines if m > line_no], default=len(program.lines) + 1)
        candidates: list[ToolComment] = []
        for line in program.lines[max(before, 0):after - 1]:
            if before < line.number < first_change_line and line.number < line_no:
                continue
            for text in line.comments:
                comment = parse_tool_comment(text, line.number, check.unit_m(i))
                if comment.describes_tool and (comment.number is not None or comment.diameter_m is not None
                                               or comment.thread_major_m is not None or line.number == line_no):
                    candidates.append(comment)
        if candidates:
            nearest = min(candidates, key=lambda c: (abs(c.line - line_no), c.line))
            _compare_comment(check, nearest, number, i)


def _differences(a: ToolComment, b: ToolComment, tolerance: float) -> list[str]:
    problems = []
    if a.diameter_m is not None and b.diameter_m is not None and abs(a.diameter_m - b.diameter_m) > tolerance:
        problems.append(f"diameter {_mm(a.diameter_m)} vs {_mm(b.diameter_m)}")
    if a.kind and b.kind and not (set(COMMENT_KINDS[a.kind]) & set(COMMENT_KINDS[b.kind])):
        problems.append(f"kind {a.kind} vs {b.kind}")
    if a.thread_major_m and b.thread_major_m and abs(a.thread_major_m - b.thread_major_m) > tolerance:
        problems.append(f"thread M{a.thread_major_m * 1e3:g} vs M{b.thread_major_m * 1e3:g}")
    return problems


def _compare_comment(check: Check, comment: ToolComment, number: int, command: int | None) -> None:
    ctx = check.ctx
    problems = []
    if command is not None and comment.number is not None and comment.number != number:
        problems.append(f"the comment names T{comment.number} but T{number} is loaded")
    tool = ctx.tools.get(comment.number if command is None else number)
    if tool is not None and tool.resolved:
        tolerance = float(ctx.p("diameter_match_tolerance_m"))
        measured = comment.thread_major_m if tool.is_tap and comment.thread_major_m else comment.diameter_m
        if measured is not None and tool.diameter_m is not None and abs(measured - tool.diameter_m) > tolerance:
            problems.append(f"comment diameter {_mm(measured)}, tool table {_mm(tool.diameter_m)}")
        if comment.kind is not None and tool.type is not None and tool.type not in COMMENT_KINDS[comment.kind]:
            problems.append(f"comment says {comment.kind}, the tool is a {tool.type}")
        if (comment.thread_pitch_m and tool.thread_pitch_m
                and abs(comment.thread_pitch_m - tool.thread_pitch_m) > tolerance):
            problems.append(f"comment pitch {_mm(comment.thread_pitch_m)}, tool {_mm(tool.thread_pitch_m)}")
        if not problems and tool.comment:
            stated = parse_tool_comment(tool.comment, 0, 1e-3)
            differences = _differences(stated, comment, tolerance)
            if differences:
                problems.append(f"the project's tool comment is '{tool.comment}' ({'; '.join(differences)})")
    if problems:
        label = f"T{number}" if command is not None else f"T{comment.number}"
        check.issue("tool-comment", Severity.WARNING, f"{label}: tool comment '{comment.text}' does not match: "
                    + "; ".join(problems), line=comment.line,
                    causes=[(check.program.name, check.ref(command).line, "tool change")] if command is not None else [])


def rule_offset_length(check: Check) -> None:
    ctx = check.ctx
    measured = check.scan.length_measured if check.scan is not None else {}
    macro_calls = check.scan.macro_calls if check.scan is not None else []
    reported: set[tuple[int, str]] = set()
    for i in check.indices(MOTION):
        modal = check.modal(i)
        section = check.section(i)
        if check.program.machine_kind == "lathe":
            if check.eng.cut[i] and modal.get("length_offset") == 0 and (section, "zero") not in reported:
                reported.add((section, "zero"))
                check.issue("offset-length", Severity.ERROR,
                            f"T{modal.get('tool')} cuts with offset 00 (no tool geometry offset active)", command=i)
            continue
        if not modal.get("length_comp") or (section > 0 and _length_section(check, i) != section):
            continue
        h = modal.get("length_offset")
        tool_number = modal.get("tool")
        length_line = int(modal.get("length_line") or 0)
        cause = [(check.ref(i).program, length_line, "length compensation")]
        if not h:
            if (section, "zero") not in reported:
                reported.add((section, "zero"))
                check.issue("offset-length", Severity.ERROR, f"G43 with H{h or 0}: no length offset is applied to "
                            f"T{tool_number}", command=i, causes=cause)
            continue
        register = ctx.tools.get(h)
        set_by_program = h in measured and measured[h][1] <= i
        if register is None and not set_by_program:
            if (section, "missing") not in reported:
                reported.add((section, "missing"))
                check.issue("offset-length", Severity.ERROR, f"H{h} has no length offset in the tool table",
                            command=i, quantities={"h_register": h}, causes=cause)
            continue
        if register is not None and register.length_offset_m is None and not set_by_program:
            if (section, "unset") not in reported:
                reported.add((section, "unset"))
                by_macro = any(call_index <= i and check.section(min(call_index, len(check.commands) - 1)) == section
                               for call_index, _ in macro_calls)
                check.issue("offset-length", Severity.WARNING if by_macro else Severity.ERROR,
                            f"length offset H{h} is not set (null in the project"
                            + (", a macro call in this section may measure it)" if by_macro else
                               " and no G37/G10 in the program sets it)"),
                            command=i, quantities={"h_register": h}, causes=cause)
            continue
        if tool_number is not None and h != tool_number and (section, "mismatch") not in reported:
            reported.add((section, "mismatch"))
            own = ctx.tools.get(tool_number)
            own_length = own.length_offset_m if own is not None else None
            other_length = register.length_offset_m if register is not None else None
            differs = own_length is not None and other_length is not None and \
                abs(own_length - other_length) > float(ctx.p("diameter_match_tolerance_m"))
            quantities = {"h_register": h}
            if own_length is not None and other_length is not None:
                quantities["length_error_m"] = other_length - own_length
            check.issue("offset-length", Severity.ERROR if differs else Severity.WARNING,
                        f"G43 H{h} with T{tool_number} in the spindle: the length offset of another tool is applied"
                        + (f" ({_mm(other_length - own_length)} off)" if differs else ""),
                        command=i, quantities=quantities, causes=cause)


def rule_offset_radius(check: Check) -> None:
    if not check.is_mill():
        return
    ctx = check.ctx
    radius_set = check.scan.radius_set if check.scan is not None else {}
    reported: set[tuple[int, str]] = set()
    for i in check.indices(MOTION):
        modal = check.modal(i)
        if modal.get("radius_comp") not in ("G41", "G42"):
            continue
        section = check.section(i)
        d = modal.get("radius_offset")
        tool_number = modal.get("tool")
        if not d:
            key = (section, "zero")
            if key not in reported:
                reported.add(key)
                check.issue("offset-radius", Severity.ERROR, f"{modal.get('radius_comp')} with D{d or 0}: cutter "
                            "compensation without a radius offset", command=i)
            continue
        register = ctx.tools.get(d)
        set_by_program = d in radius_set and radius_set[d][1] <= i
        if (register is None or register.radius_offset_m is None) and not set_by_program:
            key = (section, "unset")
            if key not in reported:
                reported.add(key)
                check.issue("offset-radius", Severity.ERROR, f"radius offset D{d} is not set in the tool table",
                            command=i, quantities={"d_register": d})
            continue
        if tool_number is not None and d != tool_number and (section, "mismatch") not in reported:
            reported.add((section, "mismatch"))
            check.issue("offset-radius", Severity.WARNING, f"{modal.get('radius_comp')} D{d} with T{tool_number}: "
                        "the radius offset of another tool is applied", command=i, quantities={"d_register": d})


def rule_work_offset(check: Check) -> None:
    ctx = check.ctx
    set_by_program = check.scan.offsets_set if check.scan is not None else {}
    motions = check.indices(MOTION)
    if not motions:
        return
    first = motions[0]
    if "work_offset_line" in check.modal(first) and not check.modal(first).get("work_offset_line"):
        check.issue("work-offset-unset", Severity.WARNING,
                    f"no work offset is selected before the first move: the control's power-on "
                    f"{check.modal(first).get('work_offset', 'G54')} is used", command=first)
    reported: set[str] = set()
    for i in motions:
        code = normalise_offset(check.modal(i).get("work_offset", "G54"))
        if code in reported or code in ctx.work_offsets_m:
            continue
        if code in set_by_program and set_by_program[code][1] <= i:
            continue
        reported.add(code)
        check.issue("work-offset-unset", Severity.ERROR, f"work offset {code} has no value in the setup "
                    "(work_offsets_mm) and the program does not set it", command=i,
                    causes=[(check.ref(i).program, int(check.modal(i).get("work_offset_line") or 0), "work offset")])


def rule_missing_g43(check: Check) -> None:
    if not check.is_mill() or check.ctx.dialect not in SUPPORTED_DIALECTS:
        return
    first_motion: dict[int, int] = {}
    for i in check.indices(MOTION):
        start, end = check.tip(i, False), check.tip(i)
        z_move = not (math.isfinite(start[2]) and math.isfinite(end[2]) and abs(end[2] - start[2]) < 1e-12)
        if z_move or check.commands[i].kind in FEED:
            first_motion.setdefault(check.section(i), i)
    for section, i in sorted(first_motion.items()):
        if section == 0:
            continue
        modal = check.modal(i)
        programmed_here = modal.get("length_comp") and _length_section(check, i) == section
        if programmed_here:
            continue
        tool = check.tool(i)
        quantities = {}
        if tool is not None and tool.length_offset_m is not None:
            quantities["tool_length_m"] = tool.length_offset_m
        active = (f"G43 H{modal.get('length_offset')} of the previous tool stays active" if modal.get("length_comp")
                  else "no length compensation is active")
        check.issue("missing-g43", Severity.ERROR,
                    f"T{modal.get('tool')} moves in Z before G43 H{modal.get('tool')} is programmed after its tool "
                    f"change at line {modal.get('tool_line')} ({active})", command=i, quantities=quantities,
                    causes=[(check.ref(i).program, int(modal.get("tool_line") or 0), "tool change")])


def _length_section(check: Check, i: int) -> int:
    modal = check.modal(i)
    if "length_section" in modal:
        return int(modal["length_section"])
    tool_line = int(modal.get("tool_line") or 0)
    return check.section(i) if int(modal.get("length_line") or 0) >= tool_line else -1


def rule_tap_depth(check: Check) -> None:
    if not check.is_mill() or check.scan is None:
        return
    ctx = check.ctx
    tolerance = float(ctx.p("hole_match_tolerance_m"))
    drilled = [h for h in check.scan.holes if (ctx.tools.get(h.tool) is not None and ctx.tools[h.tool].is_drill)]
    drilled += _plunge_holes(check)
    for hole in check.scan.holes:
        tap = ctx.tools.get(hole.tool)
        if not (hole.code in TAP_CYCLES or (tap is not None and tap.is_tap)):
            continue
        if not all(math.isfinite(v) for v in (hole.x_m, hole.y_m, hole.z_bottom_m)):
            continue
        diameter = tap.diameter_m if tap is not None else None
        matches = [d for d in drilled if d.command < hole.command and d.work_offset == hole.work_offset
                   and math.hypot(d.x_m - hole.x_m, d.y_m - hole.y_m) <= tolerance
                   and (diameter is None or ctx.tools[d.tool].diameter_m is None
                        or ctx.tools[d.tool].diameter_m < diameter)]
        box = ctx.stock_box_W(hole.work_offset)
        if not matches:
            check.issue("tap-depth", Severity.ERROR, f"tapping at X{hole.x_m * 1e3:g} Y{hole.y_m * 1e3:g} where no "
                        "smaller hole was drilled before", command=hole.command)
            continue
        deepest = min(matches, key=lambda d: d.z_bottom_m)
        drill = ctx.tools[deepest.tool]
        cone = 0.0
        if drill.point_angle_rad and drill.diameter_m:
            cone = 0.5 * drill.diameter_m / math.tan(0.5 * drill.point_angle_rad)
        full_bottom = deepest.z_bottom_m + cone
        if box is not None and full_bottom < box.lo[2]:
            continue
        pitch = tap.thread_pitch_m if tap is not None and tap.thread_pitch_m else 0.0
        margin = float(ctx.p("tap_bottom_margin_pitches")) * pitch
        causes = [(deepest.program, deepest.line, f"hole drilled by T{deepest.tool}")]
        quantities = {"tap_end_z_m": hole.z_bottom_m, "drill_tip_z_m": deepest.z_bottom_m,
                      "drill_full_diameter_z_m": full_bottom}
        if hole.z_bottom_m < deepest.z_bottom_m:
            check.issue("tap-depth", Severity.ERROR, f"tap goes {_mm(deepest.z_bottom_m - hole.z_bottom_m)} deeper "
                        f"than the hole drilled at line {deepest.line}", command=hole.command,
                        quantities=quantities, causes=causes)
        elif hole.z_bottom_m < full_bottom + margin:
            check.issue("tap-depth", Severity.WARNING, f"tap end reaches the drill point cone of the hole drilled at "
                        f"line {deepest.line} (needs Z >= {(full_bottom + margin) * 1e3:.3f} mm)",
                        command=hole.command, quantities=quantities, causes=causes)


def _plunge_holes(check: Check) -> list:
    from .scan import Hole
    holes = []
    for i in check.indices((Kind.LINEAR,)):
        tool = check.tool(i)
        modal = check.modal(i)
        if tool is None or not tool.is_drill or modal.get("cycle_info"):
            continue
        start, end = check.tip(i, False), check.tip(i)
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            continue
        if math.hypot(end[0] - start[0], end[1] - start[1]) > 1e-9 or end[2] >= start[2]:
            continue
        ref = check.ref(i)
        holes.append(Hole(i, ref.line, ref.program, modal.get("tool"), "G1", float(end[0]), float(end[1]),
                          float(end[2]), float(start[2]), float(start[2]), float(start[2]),
                          modal.get("work_offset", "G54"), check.section(i)))
    return holes


def rule_flute_length(check: Check) -> None:
    if not check.is_mill():
        return
    worst: dict[int, tuple[float, int, str]] = {}
    for i in check.indices(CUT):
        if not check.eng.cut[i]:
            continue
        tool = check.tool(i)
        if tool is None:
            continue
        depth = check.eng.ap_m[i]
        if not math.isfinite(depth):
            continue
        section = check.section(i)
        for kind, limit in (("flutes", tool.usable_length_m), ("stickout", tool.stickout_m)):
            if limit is None or depth <= limit + 1e-9:
                continue
            key = (section, kind)
            if key not in worst or depth > worst[key][0]:
                worst[key] = (depth, i if key not in worst else worst[key][1], kind)
    for (section, kind), (depth, first, _) in sorted(worst.items()):
        tool = check.tool(first)
        limit = tool.usable_length_m if kind == "flutes" else tool.stickout_m
        what = "flute length" if kind == "flutes" else "stickout (the holder reaches the part)"
        check.issue("flute-length", Severity.ERROR,
                    f"T{check.modal(first).get('tool')} cuts {_mm(depth)} deep, more than its {what} of {_mm(limit)}",
                    command=first, quantities={"depth_m": depth, "limit_m": limit})


def rule_coolant_unavailable(check: Check) -> None:
    ctx = check.ctx
    for i, command in enumerate(check.commands):
        if command.kind != Kind.COOLANT or not command.coolant or command.coolant.get("on") is False:
            continue
        needs = command.coolant.get("needs")
        code = command.coolant.get("code")
        if needs is None:
            continue
        available = {
            "flood": "flood" in ctx.pump_kinds,
            "mist": bool({"mist", "mql"} & ctx.pump_kinds),
            "shower": any("shower" in n for n in ctx.nozzle_ids | ctx.pump_ids),
            "through_spindle": bool({"through_spindle", "high_pressure", "through_tool"} & ctx.pump_kinds),
            "through_tool_air": bool({"through_tool_air", "air"} & ctx.pump_kinds),
            "air_blast": ctx.has_air_blast,
            "programmable_nozzle": ctx.has_programmable_nozzle,
        }.get(needs, False)
        if not available:
            check.issue("coolant-unavailable", Severity.ERROR, f"{code} asks for {needs.replace('_', ' ')} coolant, "
                        "which the machine does not have", command=i)
            continue
        if needs == "through_spindle":
            tool = check.tool(i)
            if tool is not None and tool.resolved and not tool.coolant_holes:
                check.issue("coolant-unavailable", Severity.WARNING, f"{code} through-spindle coolant with "
                            f"T{check.modal(i).get('tool')}, which has no coolant holes", command=i)


def _cutting_entry(tool: ToolFacts, ctx: SanityContext, operation: str):
    group = ctx.material_iso_group
    entries = [c for c in tool.cutting_data if c.iso_group == group]
    if not entries:
        return None
    exact = [c for c in entries if c.material == ctx.material_id]
    entries = exact or entries
    for entry in entries:
        if not entry.operation or operation in entry.operation:
            return entry
    return entries[0]


def rule_cutting_data(check: Check) -> None:
    ctx = check.ctx
    factor = float(ctx.p("cutting_data_error_factor"))
    stats: dict[tuple[int, str], dict] = {}
    no_data: dict[int, int] = {}
    for i in check.indices(CUT):
        if not check.eng.cut[i]:
            continue
        tool = check.tool(i)
        modal = check.modal(i)
        omega = abs(modal.get("spindle_rad_s") or 0.0)
        if tool is None or not tool.resolved or (omega == 0.0 and modal.get("css_m_s") is None) \
                or not modal.get("feed"):
            continue
        info = modal.get("cycle_info") or {}
        operation = _operation(tool, check, i, info)
        if operation == "tapping":
            _tap_feed(check, i, tool, stats)
            continue
        entry = _cutting_entry(tool, ctx, operation)
        if entry is None:
            no_data.setdefault(check.section(i), i)
            continue
        for name, value, limits in _cutting_values(check, i, tool, entry, omega):
            if value is None or not math.isfinite(value):
                continue
            low, high = limits
            key = (check.section(i), name)
            record = stats.setdefault(key, {"first": None, "worst": None, "count": 0, "entry": entry})
            out = (high is not None and value > high * (1 + 1e-9)) or (low is not None and value < low * (1 - 1e-9))
            if not out:
                continue
            ratio = value / high if high is not None and value > high else (low / value if value > 0 else math.inf)
            record["count"] += 1
            if record["first"] is None:
                record["first"] = i
            if record["worst"] is None or ratio > record["worst"][0]:
                record["worst"] = (ratio, value, low, high, i)
    for (section, name), record in sorted(stats.items()):
        if record["worst"] is None:
            continue
        if name == "tap_feed":
            continue
        ratio, value, low, high, worst_i = record["worst"]
        severity = Severity.ERROR if ratio > factor else Severity.WARNING
        tool_number = check.modal(record["first"]).get("tool")
        units, scale = _CUT_UNITS[name]
        bounds = f"[{'-' if low is None else f'{low * scale:g}'}, {'-' if high is None else f'{high * scale:g}'}]"
        entry = record["entry"]
        source = f" ({entry.page})" if entry.page else ""
        check.issue("cutting-data", severity,
                    f"T{tool_number}: {name} {value * scale:.4g} {units} is outside the maker's range {bounds} {units} "
                    f"for ISO {ctx.material_iso_group}{source}; {record['count']} cutting move(s)",
                    command=record["first"], quantities={f"{name}_value": value, "count": record["count"],
                                                         "ratio": ratio})
    for section, i in sorted(no_data.items()):
        check.issue("cutting-data", Severity.INFO, f"T{check.modal(i).get('tool')} has no maker cutting data for ISO "
                    f"group {ctx.material_iso_group}: speeds and feeds are not checked", command=i)


_CUT_UNITS = {"vc": ("m/min", 60.0), "fz": ("mm", 1e3), "f": ("mm/rev", 1e3), "ap": ("mm", 1e3), "ae": ("mm", 1e3)}


def _operation(tool: ToolFacts, check: Check, i: int, info: dict) -> str:
    if tool.is_tap or info.get("code") in TAP_CYCLES:
        return "tapping"
    if tool.is_drill or info.get("phase") == "feed":
        return "drilling"
    if check.program.machine_kind == "lathe":
        return "turning"
    ae = check.eng.ae_m[i]
    if tool.diameter_m and math.isfinite(ae) and ae >= float(check.ctx.p("slotting_ae_fraction")) * tool.diameter_m:
        return "slotting"
    return "side_milling"


def _cutting_values(check: Check, i: int, tool: ToolFacts, entry, omega: float):
    modal = check.modal(i)
    css = modal.get("css_m_s")
    feed = modal.get("feed") or 0.0
    per_rev = modal.get("feed_mode") == "per_rev"
    rev_s = omega / (2.0 * math.pi)
    if check.program.machine_kind == "lathe":
        radius = abs(check.tip(i)[0])
        vc = css if css is not None else omega * radius
        f = feed if per_rev else (feed / rev_s if rev_s > 0 else None)
        return [("vc", vc, entry.vc_m_s[0::2]), ("f", f, entry.f_m[0::2])]
    diameter = tool.diameter_m or 0.0
    ap = check.eng.ap_m[i]
    ae = check.eng.ae_m[i]
    effective = diameter
    corner = tool.corner_radius_m or 0.0
    if corner > 0 and math.isfinite(ap) and ap < corner:
        effective = diameter - 2.0 * corner + 2.0 * math.sqrt(max(corner * corner - (corner - ap) ** 2, 0.0))
    vc = omega * 0.5 * effective
    f_rev = feed if per_rev else (feed / rev_s if rev_s > 0 else None)
    values = [("vc", vc, entry.vc_m_s[0::2])]
    start, end = check.tip(i, False), check.tip(i)
    plunge = math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-12 and end[2] < start[2]
    if tool.is_drill or entry.f_m != (None, None, None):
        values.append(("f", f_rev, entry.f_m[0::2]))
    if not tool.is_drill and tool.flutes and f_rev is not None:
        low, high = entry.fz_m[0::2]
        values.append(("fz", f_rev / tool.flutes, (None if plunge else low, high)))
    ap_max = entry.ap_max_m if entry.ap_max_m is not None else (
        entry.ap_max_xD * diameter if entry.ap_max_xD is not None else None)
    ae_max = entry.ae_max_m if entry.ae_max_m is not None else (
        entry.ae_max_xD * diameter if entry.ae_max_xD is not None else None)
    if not tool.is_drill:
        values.append(("ap", ap, (None, ap_max)))
        values.append(("ae", ae, (None, ae_max)))
    return values


def _tap_feed(check: Check, i: int, tool: ToolFacts, stats: dict) -> None:
    modal = check.modal(i)
    omega = abs(modal.get("spindle_rad_s") or 0.0)
    if not tool.thread_pitch_m or omega == 0.0:
        return
    key = (check.section(i), "tap_feed")
    if key in stats:
        return
    feed = modal.get("feed") or 0.0
    rev_s = omega / (2.0 * math.pi)
    per_rev = feed if modal.get("feed_mode") == "per_rev" else feed / rev_s
    stats[key] = {"worst": None}
    tolerance = float(check.ctx.p("tapping_feed_tolerance_frac"))
    if abs(per_rev - tool.thread_pitch_m) > tolerance * tool.thread_pitch_m:
        check.issue("cutting-data", Severity.ERROR,
                    f"T{modal.get('tool')}: tapping feed {per_rev * 1e3:.4g} mm/rev does not match the tap pitch "
                    f"{tool.thread_pitch_m * 1e3:g} mm (feed must be pitch x rpm)", command=i,
                    quantities={"feed_m_per_rev": per_rev, "pitch_m": tool.thread_pitch_m},
                    causes=[(check.ref(i).program, int(modal.get("feed_line") or 0), "feed rate")])


def rule_coolant_required(check: Check) -> None:
    ctx = check.ctx
    need_by_group = dict(ctx.p("coolant_need_by_iso_group"))
    required_types = set(ctx.p("coolant_required_tool_types"))
    water_free = set(ctx.p("water_free_families"))
    dry_classes = set(ctx.p("dry_tool_classes"))
    deep_ratio = float(ctx.p("deep_hole_depth_ratio"))
    by_section: dict[int, list[int]] = {}
    for i in check.indices(CUT):
        if check.eng.cut[i]:
            by_section.setdefault(check.section(i), []).append(i)
    for section, group in sorted(by_section.items()):
        tool = check.tool(group[0])
        if tool is None:
            continue
        dry = [i for i in group if not check.modal(i).get("coolant")]
        wet = [i for i in group if check.modal(i).get("coolant")]
        entry = _cutting_entry(tool, ctx, "drilling" if tool.is_drill else "side_milling") if tool.resolved else None
        said = (entry.coolant or "").lower() if entry is not None and entry.coolant else ""
        number = check.modal(group[0]).get("tool")
        if wet and ctx.material_family in water_free and (ctx.coolant_kind in WATER_BASED):
            check.issue("coolant-required", Severity.ERROR, f"T{number}: water-based {ctx.coolant_kind} on "
                        f"{ctx.material_family} (machine it dry or with oil)", command=wet[0])
            continue
        if wet and tool.tool_class in dry_classes and said not in ("wet", "flood", "required", "emulsion"):
            check.issue("coolant-required", Severity.WARNING, f"T{number}: coolant on a {tool.tool_class} tool "
                        "(run dry: intermittent cooling cracks the edge)", command=wet[0])
            continue
        if not dry:
            continue
        reason = None
        if said in ("wet", "flood", "required", "emulsion", "oil", "mql", "through_tool", "through_spindle"):
            reason = f"the maker's data for ISO {ctx.material_iso_group} say {said}"
        elif said in ("dry", "any", "optional"):
            reason = None
        elif tool.type in required_types:
            reason = f"a {tool.type.replace('_', ' ')} needs lubrication"
        elif tool.is_drill and tool.diameter_m and _deepest_hole(check, group, tool) > deep_ratio * tool.diameter_m:
            reason = f"drilling deeper than {deep_ratio:g} x D needs coolant for chip evacuation"
        elif need_by_group.get(ctx.material_iso_group) == "required":
            reason = f"ISO {ctx.material_iso_group} ({ctx.material_family}) is cut with coolant"
        if reason is not None:
            check.issue("coolant-required", Severity.WARNING, f"T{number} cuts without coolant: {reason}; "
                        f"{len(dry)} cutting move(s)", command=dry[0], quantities={"count": len(dry)})


def _deepest_hole(check: Check, group: list[int], tool: ToolFacts) -> float:
    depths = [check.eng.ap_m[i] for i in group if math.isfinite(check.eng.ap_m[i])]
    return max(depths, default=0.0)


def rule_retract_clamps(check: Check) -> None:
    if not check.is_mill():
        return
    ctx = check.ctx
    if ctx.clamps_unknown and check.indices(MOTION):
        check.issue("retract-below-clamps", Severity.INFO, "clamp geometry unknown for "
                    f"{', '.join(ctx.clamps_unknown)} (give outline_mm in the setup): moves are not checked against it",
                    line=0)
    if not ctx.clamps_M:
        return
    reported_planes: set[tuple[int, str]] = set()
    crossing_sections: set[int] = set()
    plane_candidates: list[tuple[int, dict, float]] = []
    for i in check.indices((Kind.RAPID,)):
        modal = check.modal(i)
        offset = modal.get("work_offset", "G54")
        clamps = check.clamp_boxes(offset)
        start, end = check.tip(i, False), check.tip(i)
        if not clamps or not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            continue
        tool = check.tool(i)
        radius = tool.radius_m if tool is not None and tool.radius_m else 0.0
        info = modal.get("cycle_info") or {}
        for clamp in clamps:
            low = _segment_min_z_over_box(start, end, clamp, radius)
            if low is not None and low < clamp.top:
                what = ("the move between holes" if info.get("phase") == "position" else "rapid move")
                check.issue("retract-below-clamps", Severity.ERROR, f"{what} passes over {clamp.name} at "
                            f"Z{low * 1e3:.3f}, below its top Z{clamp.top * 1e3:.3f}", command=i,
                            quantities={"clearance_m": low - clamp.top})
                crossing_sections.add(check.section(i))
                break
        if info and info.get("retract_W_m") is not None:
            plane_candidates.append((i, info, max(c.top for c in clamps)))
    for i, info, highest in plane_candidates:
        plane = info["retract_W_m"]
        key = (check.section(i), info.get("retract_mode", ""))
        if check.section(i) in crossing_sections or key in reported_planes:
            continue
        if math.isfinite(plane) and plane < highest:
            reported_planes.add(key)
            check.issue("retract-below-clamps", Severity.WARNING,
                        f"{info.get('code')} retract plane ({info.get('retract_mode')}) Z{plane * 1e3:.3f} is below "
                        f"the top of the clamps Z{highest * 1e3:.3f}", command=i,
                        quantities={"retract_z_m": plane, "clamp_top_z_m": highest})


def _segment_min_z_over_box(start: np.ndarray, end: np.ndarray, box: Box, radius: float) -> float | None:
    """Lowest tip height of a straight move over the XY footprint of ``box`` grown by ``radius``."""
    lo = (box.lo[0] - radius, box.lo[1] - radius)
    hi = (box.hi[0] + radius, box.hi[1] + radius)
    t0, t1 = 0.0, 1.0
    for axis in (0, 1):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-15:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                return None
            continue
        a = (lo[axis] - start[axis]) / delta
        b = (hi[axis] - start[axis]) / delta
        a, b = min(a, b), max(a, b)
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    z0 = start[2] + (end[2] - start[2]) * t0
    z1 = start[2] + (end[2] - start[2]) * t1
    return float(min(z0, z1))


def rule_travel(check: Check) -> None:
    if not check.is_mill():
        return
    ctx = check.ctx
    if not ctx.travel_m:
        return
    reported: set[tuple[int, str]] = set()
    for i in check.indices(MOTION):
        modal = check.modal(i)
        offset = ctx.work_offsets_m.get(normalise_offset(modal.get("work_offset", "G54")))
        if offset is None:
            continue
        tip = check.tip(i)
        if not np.all(np.isfinite(tip)):
            continue
        length = _active_register_length(check, modal)
        machine = {"X": tip[0] + offset[0], "Y": tip[1] + offset[1]}
        if length is not None:
            machine["Z"] = tip[2] + offset[2] + length
        for axis, value in machine.items():
            travel = ctx.travel_m.get(axis)
            if travel is None:
                continue
            lo, hi = travel
            if lo - 1e-9 <= value <= hi + 1e-9:
                continue
            key = (check.section(i), axis)
            if key in reported:
                continue
            reported.add(key)
            excess = value - hi if value > hi else value - lo
            check.issue("part-outside-travel", Severity.ERROR,
                        f"with {normalise_offset(modal.get('work_offset', 'G54'))} the tool needs {axis}"
                        f"{value * 1e3:.3f} mm in machine coordinates, outside the travel "
                        f"[{lo * 1e3:g}, {hi * 1e3:g}] mm ({_mm(abs(excess))} beyond)", command=i,
                        quantities={"machine_pos_m": value, "excess_m": excess},
                        causes=[(check.ref(i).program, int(modal.get("work_offset_line") or 0), "work offset")])


def _active_register_length(check: Check, modal: dict) -> float | None:
    """Length the control adds to Z: the active H register (0 with G49 or H00), None if not known."""
    if not modal.get("length_comp"):
        return 0.0
    register = modal.get("length_offset")
    if not register:
        return 0.0
    tool = check.ctx.tools.get(register)
    if tool is None or tool.length_offset_m is None:
        return None
    return tool.length_offset_m if modal.get("length_comp_code", "G43") != "G44" else -tool.length_offset_m


def rule_arc(check: Check) -> None:
    ctx = check.ctx
    alarm = ctx.arc_alarm_tolerance_m or float(ctx.p("arc_alarm_tolerance_m"))
    for i in check.indices((Kind.ARC,)):
        command = check.commands[i]
        arc = command.arc
        modal = check.modal(i)
        if arc is None:
            continue
        if modal.get("arc_chord_too_long"):
            check.issue("arc-endpoint", Severity.ERROR, f"arc radius R{arc.radius_start_m * 1e3:g} mm is smaller "
                        "than half the distance to its end point: the arc cannot be drawn", command=i)
            continue
        difference = abs(arc.radius_end_m - arc.radius_start_m)
        if not math.isfinite(difference):
            continue
        increment = ctx.least_increment_m if modal.get("units", "mm") == "mm" else float(ctx.p("fanuc_inch_increment_m"))
        accept = float(ctx.p("arc_accept_increments")) * increment
        quantities = {"radius_start_m": arc.radius_start_m, "radius_end_m": arc.radius_end_m,
                      "difference_m": difference}
        if difference > alarm:
            check.issue("arc-endpoint", Severity.ERROR, f"arc end point is {_mm(difference)} off its radius: more "
                        f"than the control's tolerance of {_mm(alarm)}, the control alarms", command=i,
                        quantities=quantities)
        elif difference > accept:
            check.issue("arc-endpoint", Severity.WARNING, f"arc end point is {_mm(difference)} off its radius "
                        f"(start radius {_mm(arc.radius_start_m)}, end {_mm(arc.radius_end_m)}): the control blends "
                        "it into a spiral without an alarm", command=i, quantities=quantities)


def rule_program_end(check: Check) -> None:
    program = check.program
    end_code = check.scan.end_code if check.scan is not None and program.source == "scan" else None
    stops = [i for i, c in enumerate(check.commands) if c.kind == Kind.STOP and c.stop in ("M30", "M02", "M99-end")]
    if program.source == "controller":
        end_code = check.commands[stops[-1]].stop if stops else None
    if end_code is None:
        last = max((line.number for line in program.lines if line.words), default=0)
        check.issue("program-end", Severity.ERROR, "the program does not end with M30, M02 or M99", line=last)
        return
    if end_code == "M99" or not stops:
        return
    i = stops[-1]
    modal = check.modal(i)
    left_on = []
    if modal.get("spindle_rad_s"):
        left_on.append(f"spindle ({_m_code(modal.get('spindle_dir'))} from line {modal.get('spindle_line')})")
    if modal.get("coolant"):
        left_on.append(f"coolant ({' '.join(_m_code(c) for c in modal.get('coolant'))})")
    if left_on:
        check.issue("program-end", Severity.WARNING, f"{check.commands[i].stop} with {' and '.join(left_on)} still on "
                    "(stop them with M05/M09 before the end)", command=i)


def _m_code(code) -> str:
    text = str(code or "")
    return f"M{int(text[1:]):02d}" if text[:1] == "M" and text[1:].isdigit() else text


RESTART_GROUPS = {
    "distance": (("G90", "G91"), "absolute"),
    "work_offset": (("G54", "G55", "G56", "G57", "G58", "G59", "G54.1", "G154") + tuple(f"G{n}" for n in range(110, 130)),
                    "G54"),
    "plane": (("G17", "G18", "G19"), "XY"),
    "units": (("G20", "G21"), None),
    "radius_comp": (("G40", "G41", "G42"), "G40"),
    "cycle": (("G80", "G73", "G74", "G76", "G81", "G82", "G83", "G84", "G85", "G86", "G87", "G88", "G89"), None),
    "feed_mode": (("G93", "G94", "G95"), "per_min"),
}
ERROR_GROUPS = ("distance", "work_offset", "plane", "units", "radius_comp", "cycle")


def rule_restart(check: Check) -> None:
    if check.restart_analysis is not None:
        _restart_from_controller(check)
        return
    if check.scan is None:
        return
    defaults = {name: default for name, (_, default) in RESTART_GROUPS.items()}
    defaults["units"] = str(check.ctx.p("default_units"))
    if check.program.machine_kind == "lathe":
        defaults["plane"] = "ZX"
        defaults["feed_mode"] = "per_rev"
    sections = [s for s in check.scan.sections if s.command >= 0]
    for section in sections:
        inherited = check.modal(section.command)
        uses = _first_uses(check, section)
        for group, (codes, _) in RESTART_GROUPS.items():
            if group not in uses:
                continue
            written = {code for line in check.lines_between(section.program, section.restart_line, uses[group])
                       for code in line.codes("G")}
            if any(code in written for code in codes):
                continue
            value = inherited.get(group)
            default = defaults[group]
            if group == "cycle" and value is None:
                continue
            if group == "distance" and check.program.machine_kind == "lathe":
                continue
            differs = value != default
            if group == "distance" and not differs:
                check.issue("restart-modal", Severity.WARNING,
                            f"restart at line {section.restart_line} (T{section.tool}): the section does not state "
                            "G90/G91 before its first move; after an interruption in a G91 block it would move "
                            "incrementally", line=section.restart_line, program=section.program)
                continue
            if not differs:
                continue
            severity = Severity.ERROR if group in ERROR_GROUPS else Severity.WARNING
            check.issue("restart-modal", severity,
                        f"restart at line {section.restart_line} (T{section.tool}) runs with the power-on {group} "
                        f"'{default}' although the program reaches it with '{value}' (set before the section, used at "
                        f"line {uses[group]})", line=section.restart_line, program=section.program,
                        causes=[(section.program, uses[group], f"first use of {group}")])


def _first_uses(check: Check, section) -> dict[str, int]:
    uses: dict[str, int] = {}
    end = len(check.commands)
    for later in check.scan.sections:
        if later.command > section.command:
            end = later.command
            break
    for i in range(section.command, end):
        command = check.commands[i]
        line = command.ref.line
        if command.kind in MOTION:
            for group in ("distance", "work_offset", "units", "radius_comp"):
                uses.setdefault(group, line)
            if command.kind in (Kind.LINEAR, Kind.ARC):
                uses.setdefault("feed_mode", line)
            if command.kind == Kind.ARC or (command.modal or {}).get("cycle_info"):
                uses.setdefault("plane", line)
            if command.kind == Kind.RAPID and not (command.modal or {}).get("cycle_info"):
                pass
            uses.setdefault("cycle", line)
    return uses


def _restart_from_controller(check: Check) -> None:
    for record in check.restart_analysis:
        line = int(_field(record, "line", 0))
        program = _field(record, "program", None) or check.program.name
        for issue in _field(record, "issues", []) or []:
            message = _field(issue, "message", "") or f"{_field(issue, 'group', 'modal state')} not restored"
            severity = _field(issue, "severity", "error")
            check.issue("restart-modal", Severity(severity) if severity in [s.value for s in Severity]
                        else Severity.ERROR, f"restart at line {line}: {message}", line=line, program=program)


def _field(record, name, default):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def rule_g28(check: Check) -> None:
    if check.scan is None:
        return
    ctx = check.ctx
    for event in check.scan.g28:
        if event["distance"] != "absolute" or not event["axes"]:
            continue
        index = min(event["command"], len(check.commands) - 1)
        offset = check.modal(index).get("work_offset", "G54") if check.commands else "G54"
        intermediate = event["intermediate_W_m"]
        if check.program.machine_kind == "lathe":
            radius = ctx.stock_radius_m
            danger = "X" in event["axes"] and radius is not None and math.isfinite(intermediate[0]) and \
                abs(intermediate[0]) < radius
            check.issue("g28-absolute", Severity.ERROR if danger else Severity.WARNING,
                        f"G28 with absolute X/Z: the tool first goes to X{intermediate[0] * 2e3:g} Z"
                        f"{intermediate[2] * 1e3:g} (work coordinates)" + (", inside the stock" if danger else "")
                        + "; write G28 U0 W0", line=event["line"], program=event["program"])
            continue
        top = check.stock_top(offset)
        clamp_tops = [c.top for c in check.clamp_boxes(offset)]
        highest = max([top] + clamp_tops) if top is not None else (max(clamp_tops) if clamp_tops else None)
        z = intermediate[2]
        danger = "Z" in event["axes"] and highest is not None and math.isfinite(z) and z < highest
        check.issue("g28-absolute", Severity.ERROR if danger else Severity.WARNING,
                    f"{event['code']} in G90: the intermediate point is in work coordinates"
                    + (f" (Z{z * 1e3:.3f}, below the part/clamp top Z{highest * 1e3:.3f}: the tool moves into it before "
                       "going home)" if danger else "") + f"; write G91 {event['code']} Z0",
                    line=event["line"], program=event["program"], quantities={"intermediate_z_m": z}
                    if math.isfinite(z) else {})


def rule_toolchange_clearance(check: Check) -> None:
    if not check.is_mill():
        return
    ctx = check.ctx
    clearance = float(ctx.p("toolchange_clearance_m"))
    for i, command in enumerate(check.commands):
        if command.kind != Kind.TOOL_CHANGE:
            continue
        tip = check.tip(i, False)
        if not np.all(np.isfinite(tip)):
            continue
        offset = check.modal(i).get("work_offset", "G54")
        top = check.stock_top(offset)
        if top is None:
            continue
        box = ctx.stock_box_W(offset)
        tops = [top] + [c.top for c in check.clamp_boxes(offset)]
        needed = max(tops) + clearance
        if tip[2] >= needed:
            continue
        over = box is not None and box.lo[0] <= tip[0] <= box.hi[0] and box.lo[1] <= tip[1] <= box.hi[1]
        height = tip[2] - max(tops)
        where = f"{_mm(-height)} below" if height < 0 else f"only {_mm(height)} above"
        check.issue("toolchange-clearance", Severity.WARNING,
                    f"M06 with the tool tip at Z{tip[2] * 1e3:.3f}, {where} the highest of stock and clamps"
                    f"{' and over the part' if over else ''}: retract Z first (G91 G28 Z0 or G53 Z0)",
                    command=i, quantities={"tip_z_m": float(tip[2]), "needed_z_m": needed})


def rule_rapid_below_stock(check: Check) -> None:
    if not check.is_mill():
        return
    ctx = check.ctx
    minimum = float(ctx.p("rapid_min_clearance_m"))
    groups: dict[tuple[int, str], list[int]] = {}
    for i in check.indices((Kind.RAPID,)):
        if not check.eng.evaluated[i] or not check.eng.in_footprint[i]:
            continue
        offset = check.modal(i).get("work_offset", "G54")
        top = check.stock_top(offset)
        end = check.tip(i)
        start = check.tip(i, False)
        if top is None:
            continue
        gap = check.eng.gap_m[i]
        if end[2] > start[2] + 1e-12:
            if gap < -1e-6:
                groups.setdefault((check.section(i), "into"), []).append(i)
            continue
        if end[2] >= top - 1e-9:
            continue
        kind = "into" if gap < minimum else "below"
        groups.setdefault((check.section(i), kind), []).append(i)
    for (section, kind), group in sorted(groups.items()):
        first = group[0]
        top = check.stock_top(check.modal(first).get("work_offset", "G54"))
        gap = float(np.min(check.eng.gap_m[group]))
        z = float(min(check.tip(first)[2], check.tip(first, False)[2]))
        if kind == "into":
            message = (f"rapid {'into the material' if gap < 0 else 'to within ' + _mm(gap) + ' of the material'} "
                       f"at Z{z * 1e3:.3f} (stock top Z{top * 1e3:.3f}); {len(group)} rapid(s)")
            severity = Severity.ERROR
        else:
            message = (f"rapid below the stock top to Z{z * 1e3:.3f} (top Z{top * 1e3:.3f}), "
                       f"{_mm(gap)} above the material left by earlier cuts; {len(group)} rapid(s)")
            severity = Severity.WARNING
        check.issue("rapid-below-stock", severity, message, command=first,
                    quantities={"gap_m": gap, "depth_below_top_m": top - z, "count": len(group)})


def rule_scan_limits(check: Check) -> None:
    program = check.program
    if program.source == "none":
        check.issue("scan-limits", Severity.INFO, f"dialect {check.ctx.dialect}: the static scan reads Fanuc and Haas "
                    "programs; give the controller's command stream to check this program", line=0)
        return
    if program.scan is None:
        return
    grouped: dict[str, list] = {}
    for note in program.scan.notes:
        grouped.setdefault(note.kind, []).append(note)
    for kind, notes in sorted(grouped.items()):
        first = notes[0]
        lines = sorted({n.line for n in notes if n.line})
        listed = ", ".join(str(n) for n in lines[:8]) + (" ..." if len(lines) > 8 else "")
        check.issue("scan-limits", Severity.INFO, f"{first.message}" + (f" (lines {listed})" if len(lines) > 1 else ""),
                    line=first.line, program=first.program, quantities={"count": len(notes)})
    reasons: dict[str, list[int]] = {}
    for index, reason in sorted(check.eng.reasons.items()):
        if check.commands[index].kind in MOTION and reason != "work offset not set":
            reasons.setdefault(reason, []).append(index)
    for reason, indices in sorted(reasons.items()):
        check.issue("scan-limits", Severity.INFO, f"engagement not estimated for {len(indices)} move(s): {reason}; "
                    "rules that need the cut (spindle, coolant, cutting data, depth) skip them",
                    command=indices[0], quantities={"count": len(indices)})


RULES = (
    ("feed-missing", rule_feed_missing),
    ("feed-inherited", rule_feed_inherited),
    ("spindle-stopped", rule_spindle_stopped),
    ("spindle-direction", rule_spindle_direction),
    ("tool-unknown", rule_tool_unknown),
    ("tool-comment", rule_tool_comment),
    ("offset-length", rule_offset_length),
    ("offset-radius", rule_offset_radius),
    ("work-offset-unset", rule_work_offset),
    ("missing-g43", rule_missing_g43),
    ("tap-depth", rule_tap_depth),
    ("flute-length", rule_flute_length),
    ("coolant-unavailable", rule_coolant_unavailable),
    ("cutting-data", rule_cutting_data),
    ("coolant-required", rule_coolant_required),
    ("retract-below-clamps", rule_retract_clamps),
    ("part-outside-travel", rule_travel),
    ("arc-endpoint", rule_arc),
    ("program-end", rule_program_end),
    ("restart-modal", rule_restart),
    ("g28-absolute", rule_g28),
    ("toolchange-clearance", rule_toolchange_clearance),
    ("rapid-below-stock", rule_rapid_below_stock),
    ("scan-limits", rule_scan_limits),
)
RULE_IDS = tuple(name for name, _ in RULES)


def run_rules(program: Program, ctx: SanityContext, engagement: Engagement, *, restart_analysis=None,
              only: tuple[str, ...] | None = None) -> list[Issue]:
    check = Check(program, ctx, engagement, restart_analysis)
    for name, rule in RULES:
        if only is not None and name not in only:
            continue
        if program.source == "none" and name != "scan-limits":
            continue
        rule(check)
    order = {name: k for k, (name, _) in enumerate(RULES)}
    return sorted(check.issues, key=lambda issue: (issue.program != program.name, issue.program, issue.line,
                                                   order[issue.rule], issue.message))
