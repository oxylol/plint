"""Timeline of executed blocks and the cycle-time ledger (docs/models/motion.md "Timeline and ledger").

One record per executed NC block of a path (the commands sharing ``(path, block_seq)``): ``{index, op,
program, line, block, path, kind, tick0, tick1, part, tool, ticks, stop, pumps_on}`` in base ticks,
the block running ``[tick0, tick1)`` (the format cnctwin.production reads). Records of a path are
contiguous: waits before a block (data starvation, spindle arrival, path waits, operator) belong to
the block that follows them, so the records of a path sum exactly to its span and ``ticks`` (time by
category) sums exactly to ``tick1 - tick0``: every L1 tick is booked once, in integer base ticks.

Categories (cnctwin.production.timeline.CATEGORIES, plus motion's own ``data_starvation``,
``feed_hold`` and ``in_position``): ``cutting`` / ``air_cutting`` (feed in / out of contact, split by the
hot loop's per-tick contact flag or else by removal's ToolContact of the step), ``rapid``,
``acceleration`` (time lost to acceleration, corner slow-down and the centripetal limits: the block's
motion time beyond length / programmed speed), ``in_position`` (settling at exact stops), ``dwell``,
``tool_change``, ``spindle_ramp``, ``coolant_delay``, ``probing``, ``program_stop``, ``auxiliary``,
``path_wait``, ``data_starvation`` (standing still for the next block's preparation), ``feed_hold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..controller.ir import LineRef

CAT_FEED = "feed"               # resolved later into cutting / air_cutting
CAT_RAPID = "rapid"
CAT_ACCEL = "acceleration"
CAT_INPOS = "in_position"
CAT_DWELL = "dwell"
CAT_TOOL = "tool_change"
CAT_SPINDLE = "spindle_ramp"
CAT_COOLANT = "coolant_delay"
CAT_PROBE = "probing"
CAT_STOP = "program_stop"
CAT_AUX = "auxiliary"
CAT_WAIT = "path_wait"
CAT_STARVE = "data_starvation"
CAT_HOLD = "feed_hold"
CAT_CUT = "cutting"
CAT_AIR = "air_cutting"

CATEGORIES = (CAT_CUT, CAT_AIR, CAT_FEED, CAT_RAPID, CAT_ACCEL, CAT_INPOS, CAT_DWELL, CAT_TOOL, CAT_SPINDLE,
              CAT_COOLANT, CAT_PROBE, CAT_STOP, CAT_AUX, CAT_WAIT, CAT_STARVE, CAT_HOLD)
CAT_CODE = {name: i for i, name in enumerate(CATEGORIES)}


@dataclass
class Record:
    index: int
    path: int
    block_seq: int
    op: str
    program: str
    line: int
    block: str
    kind: str
    tick0: int
    tick1: int = -1
    part: int = 0
    tool: int | None = None
    stop: str = ""
    ticks: dict = field(default_factory=dict)          # category -> base ticks
    pumps_on: list = field(default_factory=list)
    motion_ticks: int = 0                               # feed/rapid base ticks booked (for the ideal-time split)
    ideal_ticks: int = 0                                # programmed-speed time of its motion (base ticks)
    motion_cat: str = ""
    starve_ticks: int = 0                               # motion L1 ticks held back by unprepared blocks
    pending_feed: dict = field(default_factory=dict)    # macro step -> feed base ticks awaiting the contact split

    def book(self, category: str, base_ticks: int) -> None:
        if base_ticks:
            self.ticks[category] = self.ticks.get(category, 0) + int(base_ticks)

    def to_dict(self) -> dict:
        out = {"index": self.index, "op": self.op, "program": self.program, "line": self.line, "block": self.block,
               "path": self.path, "kind": self.kind, "tick0": self.tick0, "tick1": self.tick1, "part": self.part,
               "ticks": {k: int(v) for k, v in self.ticks.items() if v}}
        if self.tool is not None:
            out["tool"] = int(self.tool)
        if self.stop:
            out["stop"] = self.stop
        if self.pumps_on:
            out["pumps_on"] = list(self.pumps_on)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        rec = cls(int(d["index"]), int(d["path"]), int(d.get("block_seq", -1)), str(d.get("op", "")),
                  str(d.get("program", "")), int(d.get("line", 0)), str(d.get("block", "")), str(d.get("kind", "")),
                  int(d["tick0"]), int(d.get("tick1", -1)), int(d.get("part", 0)), d.get("tool"),
                  str(d.get("stop", "")), {k: int(v) for k, v in (d.get("ticks") or {}).items()},
                  list(d.get("pumps_on") or []), int(d.get("motion_ticks", 0)), int(d.get("ideal_ticks", 0)),
                  str(d.get("motion_cat", "")), int(d.get("starve_ticks", 0)),
                  {int(k): int(v) for k, v in (d.get("pending_feed") or {}).items()})
        return rec

    def state_dict(self) -> dict:
        d = self.to_dict()
        d.update({"block_seq": self.block_seq, "motion_ticks": self.motion_ticks, "ideal_ticks": self.ideal_ticks,
                  "motion_cat": self.motion_cat, "starve_ticks": self.starve_ticks,
                  "pending_feed": {str(k): v for k, v in self.pending_feed.items()},
                  "ticks": dict(self.ticks)})
        return d


class Timeline:
    """Every record of the run (index = position, global over paths, in order of opening)."""

    def __init__(self):
        self.records: list[Record] = []

    def open(self, path: int, block_seq: int, ref: LineRef, kind: str, tick0: int, op: str = "", part: int = 0,
             tool: int | None = None) -> Record:
        rec = Record(len(self.records), path, block_seq, op, ref.program if ref else "", ref.line if ref else 0,
                     ref.block if ref else "", kind, int(tick0), part=part, tool=tool)
        self.records.append(rec)
        return rec

    def line_ref(self, index: int):
        """LineRef-like view of record ``index`` (core.domain.RunContext.timeline protocol)."""
        if not 0 <= index < len(self.records):
            return None
        r = self.records[index]
        return _Ref(r.op, r.program, r.line, r.block)

    def closed(self) -> list[dict]:
        return [r.to_dict() for r in self.records if r.tick1 >= 0 and not r.pending_feed]

    def to_json(self, tick_s: float) -> dict:
        return {"format": "cnctwin-timeline", "version": 1, "tick_s": tick_s, "blocks": self.closed()}

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class _Ref:
    op: str
    program: str
    line: int
    block: str


def split_motion(rec: Record, ipo_ticks: int) -> None:
    """Move the motion ticks beyond the ideal (programmed speed) time of a closed record into
    ``data_starvation`` (up to the ticks the generator was held back by unprepared blocks) and
    ``acceleration`` (the rest): the feed/rapid time keeps min(motion, ideal) base ticks. Feed time is taken
    from air cutting first, then cutting, then the feed still awaiting its contact split (latest step
    first)."""
    if not rec.motion_cat or rec.motion_ticks <= 0:
        return
    excess = max(0, rec.motion_ticks - max(rec.ideal_ticks, 0))
    if excess <= 0:
        return
    starve = min(excess, max(rec.starve_ticks, 0) * ipo_ticks)
    left = excess
    order = (CAT_RAPID,) if rec.motion_cat == CAT_RAPID else (CAT_AIR, CAT_CUT, CAT_FEED)

    def move(take: int) -> None:
        nonlocal starve
        to_starve = min(take, starve)
        rec.book(CAT_STARVE, to_starve)
        rec.book(CAT_ACCEL, take - to_starve)
        starve -= to_starve

    for cat in order:
        have = rec.ticks.get(cat, 0)
        take = min(left, have)
        if take:
            rec.ticks[cat] = have - take
            move(take)
            left -= take
        if left <= 0:
            break
    if left > 0 and rec.motion_cat == CAT_FEED and rec.pending_feed:
        for step in sorted(rec.pending_feed, reverse=True):
            take = min(left, rec.pending_feed[step])
            rec.pending_feed[step] -= take
            move(take)
            left -= take
            if left <= 0:
                break
        rec.pending_feed = {k: v for k, v in rec.pending_feed.items() if v > 0}
    rec.ticks = {k: v for k, v in rec.ticks.items() if v}


def split_largest_remainder(total: int, weights: list[int]) -> list[int]:
    """Integer shares of ``total`` proportional to ``weights`` (largest remainder, ties by position)."""
    s = sum(weights)
    if total <= 0 or s <= 0:
        return [0] * len(weights)
    raw = [total * w / s for w in weights]
    out = [int(r) for r in raw]
    rest = total - sum(out)
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in order[:rest]:
        out[i] += 1
    return out
