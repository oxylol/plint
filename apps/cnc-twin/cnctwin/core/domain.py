"""The interface every physical domain implements, and the context a run passes to it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .bus import Bus
from .channels import Channel, ChannelRegistry
from .findings import Category, Finding, Severity
from .rng import stream, stream_key
from .trace import TraceLog


@dataclass
class RunContext:
    """Everything a domain may read that is not simulation state (nothing here is checkpointed
    except through SimState; the trace log is cut at checkpoints by byte offset)."""

    project: Any                    # cnctwin.project.ResolvedProject
    db: Any                         # cnctwin.db.Database
    run_dir: Path | None
    seed: int
    tick_s: float                   # base clock (1e-6 s)
    ipo_ticks: int                  # base ticks per L1 tick (from the machine's IPO period)
    macro_ipo: int                  # L1 ticks per macro step
    channels: ChannelRegistry = field(default_factory=ChannelRegistry)
    trace: TraceLog = field(default_factory=TraceLog)
    recorder: Any = None            # cnctwin.core.recorder.Recorder
    timeline: Any = None            # block index -> LineRef (cnctwin.core.rundir.Timeline)

    @property
    def ipo_s(self) -> float:
        return self.tick_s * self.ipo_ticks

    @property
    def macro_s(self) -> float:
        return self.tick_s * self.ipo_ticks * self.macro_ipo

    def init_rng(self, name: str) -> np.random.Generator:
        """A generator for initialisation only (never inside macro steps; see core.rng)."""
        return stream(self.seed, name)

    @staticmethod
    def stream_id(name: str) -> int:
        return stream_key(name)

    def finding(self, state: Any, domain: str, category: Category, severity: Severity, message: str,
                tick: int, path: int = 0, quantities: dict | None = None,
                parents: list[int] | None = None) -> Finding:
        """Raise a finding at base tick ``tick``: line, block and positions are resolved at that tick
        from the motion frame of the step (state.bus_next.motion, falling back to bus_prev)."""
        line_ref, tip_S, tip_M = _resolve_tick(self, state, tick, path)
        time_s = tick * self.tick_s
        event_id = self.trace.record(time_s, tick, domain, f"finding:{category.value}", quantities or {},
                                     parents or [], line_ref.line if line_ref else 0)
        stock_pose = _stock_pose(self)
        tip_W = (stock_pose[:3, :3] @ tip_S + stock_pose[:3, 3]) if tip_S is not None else None
        finding = Finding(category=category, severity=severity, message=message, time_s=time_s,
                          tick=tick, domain=domain,
                          op=getattr(line_ref, "op", "") if line_ref else "",
                          program=line_ref.program if line_ref else "",
                          line=line_ref.line if line_ref else 0,
                          block=line_ref.block if line_ref else "",
                          pos_W_m=tuple(float(v) for v in tip_W) if tip_W is not None else None,
                          pos_M_m=tuple(float(v) for v in tip_M) if tip_M is not None else None,
                          quantities={k: float(v) for k, v in sorted((quantities or {}).items())},
                          cause_chain=self.trace.chain_ids(event_id))
        return state.findings.add(finding)


def _resolve_tick(ctx: RunContext, state: Any, tick: int, path: int):
    for bus in (state.bus_next, state.bus_prev):
        paths = bus.motion.paths
        if path < len(paths) and len(paths[path].tick):
            frame = paths[path]
            index = int(np.searchsorted(frame.tick, tick, side="right")) - 1
            index = min(max(index, 0), len(frame.tick) - 1)
            line_ref = ctx.timeline.line_ref(int(frame.block_index[index])) if ctx.timeline else None
            return line_ref, frame.tip_S_m[index], frame.tip_M_m[index]
    return None, None, None


def _stock_pose(ctx: RunContext) -> np.ndarray:
    pose = getattr(ctx.project, "stock_pose_W_S", None) if ctx.project is not None else None
    return np.asarray(pose) if pose is not None else np.eye(4)


@runtime_checkable
class Domain(Protocol):
    name: str

    def channels(self) -> list[Channel]:
        """Time series this domain records."""

    def init(self, ctx: RunContext, state: Any) -> None:
        """Build this domain's state (``state.<name>``) from the resolved project."""

    def kernel_inputs(self, ctx: RunContext, state: Any, bus: Bus) -> Any:
        """NamedTuple of arrays this domain hands to the compiled loop for the coming macro step
        (frozen for the step), or None if it takes no part in the hot loop."""

    def absorb(self, ctx: RunContext, state: Any, outputs: Any, bus_next: Bus) -> None:
        """Turn the compiled loop's accumulators and event buffers into this domain's bus record
        and trace events (called after the hot loop, before macro_step)."""

    def macro_step(self, ctx: RunContext, state: Any, prev: Bus, nxt: Bus) -> None:
        """Advance one macro step: read ``prev`` (and only own records of ``nxt``), write own record
        into ``nxt``."""

    def end_operation(self, ctx: RunContext, state: Any) -> None:
        """Operation finished: unclamp, cool down for the time between operations, carry state."""

    def finalize(self, ctx: RunContext, state: Any) -> dict:
        """Results for the report (JSON-serialisable)."""
