"""A toy multi-domain run that drives the whole run infrastructure (tests of cnctwin/core).

Three toy domains with counter-based draws (cnctwin.core.rng.hash_uniform) and growing arrays,
fresh bus records every step, a timeline, trace events and findings, the recorder (fast ring,
tick and macro tiers), frames.bin, keyframes, the ledger and checkpoints. Run as a script::

    python tests/unit/core/toysim.py RUN_DIR --steps N [--crash-after K] [--resume]

``--crash-after K`` stops the process abruptly (os._exit, nothing flushed or closed) after step K;
``--resume`` continues RUN_DIR from its latest checkpoint. The same N steps give the same result
set however they are split (tests/unit/core/test_checkpoint.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnctwin.core.bus import (Bus, ChipFeedback, CuttingContact, DynamicsFrame, MotionFrame,  # noqa: E402
                              PathFrame, RemovalContact)
from cnctwin.core.channels import Channel, ChannelRegistry, Tier  # noqa: E402
from cnctwin.core.checkpoint import Checkpointer  # noqa: E402
from cnctwin.core.domain import RunContext  # noqa: E402
from cnctwin.core.findings import Category, Severity  # noqa: E402
from cnctwin.core.keyframes import FieldGrid, KeyframeWriter  # noqa: E402
from cnctwin.core.ledger import Ledger  # noqa: E402
from cnctwin.core.recorder import Recorder, ring_begin  # noqa: E402
from cnctwin.core.rng import hash_uniform, stream_key  # noqa: E402
from cnctwin.core.rundir import FramesWriter, RunDirectory, Timeline  # noqa: E402
from cnctwin.core.state import SimState, ToolSlot, ToolTable  # noqa: E402
from cnctwin.core.trace import TraceLog  # noqa: E402

TICK_S = 1e-6
IPO = 1000
MACRO_IPO = 10
MACRO = IPO * MACRO_IPO
L0 = 250
SEED = 4711
RHO = 2700.0
OMEGA = -1047.1975511965977          # 10 000 rpm, M03
CHECKPOINT_TICKS = 200_000
KEYFRAME_TICKS = 50_000
BLOCK_STEPS = 5
PROJECT_HASH = "toy-project-hash"
SETTINGS = {"fast_min_spacing_s": 0.02, "fast_max_windows_per_second": 6, "fast_window_min_s": 0.001,
            "fast_window_max_s": 0.03, "series_buffer_max_bytes": 20000}
RECORDING = {"fast": "windows", "fast_window_revs": 2, "fast_period_s": 0.25, "keyframe_period_s": 0.05}


def u(name: str, a: int, b: int = 0, c: int = 0) -> float:
    return float(hash_uniform(SEED, stream_key(name), a, b, c))


class Pose(NamedTuple):
    position: np.ndarray
    angle: float


@dataclass
class ToyField:
    """Toy removal state with its own to_arrays/from_arrays (as a domain state ships them)."""

    heights: np.ndarray                      # [4] top of each brick (m), cut down in place
    removed: np.ndarray                      # growing: volume removed per step

    def to_arrays(self):
        return {"heights": self.heights, "removed": self.removed}, {"kind": "toy-field"}

    @classmethod
    def from_arrays(cls, arrays, meta):
        assert meta["kind"] == "toy-field"
        return cls(arrays["heights"], arrays["removed"])


@dataclass
class ToyCut:
    """Toy cutting state through the generic codec: nested dicts, a list, a NamedTuple, an alias, a view."""

    work: dict = field(default_factory=lambda: {"shaft": 0.0, 1: [0.0, 0.0]})
    history: list = field(default_factory=list)
    pose: Pose = field(default_factory=lambda: Pose(np.zeros(3), 0.0))
    pool: np.ndarray = field(default_factory=lambda: np.zeros(8))
    window: np.ndarray | None = None         # a view into pool
    alias: np.ndarray | None = None          # the same object as pool

    def __post_init__(self):
        if self.window is None:
            self.window = self.pool[2:6]
            self.alias = self.pool


@dataclass
class ToyChips:
    in_flight_kg: float = 0.0
    settled_kg: float = 0.0
    emitted: list = field(default_factory=list)


def registry() -> ChannelRegistry:
    reg = ChannelRegistry()
    reg.register_all([
        Channel("toy.force_N", "N", Tier.FAST, "cutting", "toy force", (3,)),
        Channel("toy.h_m", "m", Tier.FAST, "cutting", "toy chip thickness per tooth", (4,)),
        Channel("toy.feed_m_s", "m/s", Tier.TICK, "motion", "toy feed"),
        Channel("toy.err_m", "m", Tier.TICK, "motion", "toy following error", (3,)),
        Channel("toy.mode", "-", Tier.TICK, "motion", "toy mode", (), "int32"),
        Channel("toy.volume_m3", "m3", Tier.MACRO, "removal", "toy removed volume"),
        Channel("toy.flag", "-", Tier.MACRO, "chips", "toy flag", (2,), "bool"),
    ])
    reg.register_all(Ledger.channels())
    return reg


class Toy:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.rundir = RunDirectory(self.run_dir)
        self.registry = registry()
        self.recorder = Recorder(self.run_dir, self.registry, l0_ticks_min=L0, ipo_ticks=IPO, macro_ipo=MACRO_IPO,
                                 recording=RECORDING, settings=SETTINGS)
        self.frames = FramesWriter(self.run_dir / "frames.bin", axes=[
            {"name": "X", "kind": "linear", "origin": -0.30}, {"name": "Y", "kind": "linear", "origin": -0.20},
            {"name": "Z", "kind": "linear", "origin": -0.40}, {"name": "C", "kind": "rotary"}],
            paths=[{"path": 0, "spindle": "main"}], ipo_ticks=IPO)
        self.keyframes = KeyframeWriter(self.run_dir, period_ticks=KEYFRAME_TICKS, full_every=4, lattice_m=0.00025,
                                        origin_S_m=[0, 0, 0], grid=[2, 2, 1])
        self.timeline = Timeline(self.run_dir)
        self.ledger = Ledger(tol_mass=1e-9, tol_energy=1e-6, density_kg_m3=RHO, domains={"removal", "chips", "cutting"},
                             max_lag_ticks=3 * MACRO)
        self.checkpointer = Checkpointer(self.run_dir, project_hash=PROJECT_HASH, build="toy-build",
                                         interval_ticks=CHECKPOINT_TICKS, allowed_modules=("cnctwin", "toysim"))
        self.ctx = RunContext(project=None, db=None, run_dir=self.run_dir, seed=SEED, tick_s=TICK_S, ipo_ticks=IPO,
                              macro_ipo=MACRO_IPO, channels=self.registry, recorder=self.recorder,
                              timeline=self.timeline)
        self.state: SimState | None = None
        self.block = -1

    def participants(self) -> dict:
        return {"recorder": self.recorder, "frames": self.frames, "keyframes": self.keyframes,
                "timeline": self.timeline, "rundir": self.rundir, "ledger": self.ledger}

    # ------------------------------------------------------------------ start or resume
    def start(self) -> None:
        self.rundir.prepare()
        self.ctx.trace = TraceLog(self.run_dir / "trace.jsonl")
        self.frames.open()
        self.recorder.write_channels()
        state = SimState()
        state.tools = ToolTable((ToolSlot(0, 1, "toy-mill", "toy-holder"),))
        state.removal = ToyField(np.full(4, 0.008), np.zeros(0))
        state.cutting = ToyCut()
        state.chips = ToyChips()
        state.rng_counters = {"toy.removal": 0}
        self.state = state
        self.keyframes.write(0, self.meshes(), full=True, fields=self.fields(0))

    def resume(self) -> None:
        marker = self.checkpointer.latest()
        restored = self.checkpointer.resume(marker, self.participants(), run_directory=self.rundir)
        self.state = restored.state
        self.ctx.trace = restored.trace
        self.block = self._open_block()

    def _open_block(self) -> int:
        open_ = [r["index"] for r in self.timeline.records if r["tick1"] < 0]
        return open_[-1] if open_ else -1

    # ------------------------------------------------------------------ the toy physics
    def meshes(self) -> dict:
        out = {}
        for brick in range(4):
            x0, y0 = 0.008 * (brick % 2), 0.008 * (brick // 2)
            top = float(self.state.removal.heights[brick])
            if top <= 1e-4:
                out[brick] = None
                continue
            v = np.array([[x0, y0, 0], [x0 + 0.008, y0, 0], [x0 + 0.008, y0 + 0.008, 0], [x0, y0 + 0.008, 0],
                          [x0, y0, top], [x0 + 0.008, y0, top], [x0 + 0.008, y0 + 0.008, top], [x0, y0 + 0.008, top]])
            f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
            out[brick] = (v, f)
        return out

    def fields(self, tick: int) -> dict:
        return {"pile_height_m": np.array([self.state.chips.settled_kg * 1e-3], np.float32),
                "part_pose": np.eye(4), "surface_temp_K": np.full((3, 3), 293.15 + tick * 1e-5)}

    def step(self) -> None:
        state, ctx = self.state, self.ctx
        n = state.clock.macro
        tick0 = n * MACRO
        tick1 = tick0 + MACRO
        state.bus_next = Bus()
        contact = (n // 7) % 2 == 0 and n > 0
        # timeline: a block every BLOCK_STEPS steps
        if n % BLOCK_STEPS == 0:
            if self.block >= 0:
                self.timeline.close(self.block, tick0, ticks={"cutting": tick0 - self.timeline.records[self.block]["tick0"]})
            self.block = self.timeline.open(path=0, op="OP10", program="toy.nc", line=10 + n // BLOCK_STEPS,
                                            block=f"G01 X{n} F600", kind="linear", tick0=tick0, tool=1)
        # motion record
        ticks = tick0 + IPO * np.arange(MACRO_IPO, dtype=np.int64)
        t = ticks * TICK_S
        tip = np.stack([0.008 + 0.006 * np.cos(3.0 * t), 0.008 + 0.006 * np.sin(3.0 * t), np.full(len(t), 0.004)], 1)
        angle = OMEGA * t
        frame = PathFrame(path=0, tick0=tick0, ipo_ticks=IPO, tick=ticks, block_index=np.full(MACRO_IPO, self.block, np.int32),
                          tip_S_m=tip, tip_M_m=tip + np.array([-0.3, -0.2, -0.4]),
                          axis_T_M=np.tile([0.0, 0.0, 1.0], (MACRO_IPO, 1)), axis_names=("X", "Y", "Z", "C"),
                          axis_actual=np.column_stack([tip + np.array([-0.3, -0.2, -0.4]), 0.5 * t]),
                          spindle_angle_rad=np.mod(angle, 2 * math.pi),
                          spindle_rev=np.floor(angle / (2 * math.pi)).astype(np.int64),
                          spindle_omega_rad_s=np.full(MACRO_IPO, OMEGA), tool_slot=0)
        state.bus_next.motion = MotionFrame(time_s=tick1 * TICK_S, tick=tick1, paths=[frame])
        # fast ring rows (the hot loop's job): one per L0 step in contact
        ring = self.recorder.ring
        f0, _ = self.recorder.fast_columns("toy.force_N")
        h0, _ = self.recorder.fast_columns("toy.h_m")
        if contact:
            for k, tick in enumerate(range(tick0, tick1, L0)):
                slot = ring_begin(ring, tick)
                for j in range(3):
                    ring.data[slot, f0 + j] = 100.0 * u("toy.force", n, k, j) - 50.0
                for j in range(4):
                    ring.data[slot, h0 + j] = 1e-4 * u("toy.h", n, k, j)
        # removal, cutting, dynamics (the hot loop's window: this step)
        field_ = state.removal
        volume = 1e-9 * u("toy.removal", n) if contact else 0.0
        field_.removed = np.append(field_.removed, volume)
        brick = n % 4
        field_.heights[brick] = max(0.0, field_.heights[brick] - (1e-3 if contact else 0.0))
        state.rng_counters["toy.removal"] = n + 1
        state.bus_next.removal.contacts = [RemovalContact(contact=0, tool_slot=0, removed_volume_m3=volume)]
        energy_in = (100.0 * u("toy.shaft", n) if contact else 0.0)
        feed = (1.0 * u("toy.feed", n) if contact else 0.0)
        total = energy_in + feed
        cut = CuttingContact(contact=0, tool_slot=0, shaft_work_J=energy_in, feed_work_J=feed,
                             heat_shear_J=0.5 * total, heat_rake_J=0.3 * total, heat_flank_J=0.1 * total,
                             ploughing_J=0.05 * total)
        state.bus_next.cutting.contacts = [cut]
        state.bus_next.dynamics = DynamicsFrame(modal_energy_change_J=0.02 * total, damping_dissipation_J=0.02 * total,
                                                process_damping_J=0.005 * total)
        toy_cut = state.cutting
        toy_cut.work["shaft"] += energy_in
        toy_cut.work[1][n % 2] += feed
        toy_cut.history.append(float(total))
        toy_cut.window += 1.0                      # in place through the view: pool and alias see it
        toy_cut.pose = Pose(tip[-1].copy(), float(angle[-1]))
        # chips (a slow domain: books the previous step's emissions and kinetic energy)
        prev = state.bus_prev
        chips = state.chips
        prev_volume = sum(c.removed_volume_m3 for c in prev.removal.contacts)
        prev_total = sum(c.shaft_work_J + c.feed_work_J for c in prev.cutting.contacts)
        emitted = RHO * prev_volume
        settle = 0.5 * chips.in_flight_kg
        chips.in_flight_kg = chips.in_flight_kg - settle + emitted
        chips.settled_kg += settle
        chips.emitted.append(emitted)
        record = ChipFeedback(booked_until_tick=tick0 if n > 0 else -1, kinetic_energy_J=0.005 * prev_total,
                              in_flight_kg=chips.in_flight_kg, settled_kg=chips.settled_kg)
        state.bus_next.chips = record
        state.ledger.chips_in_flight_kg = chips.in_flight_kg
        state.ledger.chips_settled_kg = chips.settled_kg
        # trace and findings
        event = ctx.trace.record(tick1 * TICK_S, tick1 - 1, "cutting", f"peak:{n}", {"force_N": 50.0 * u("toy.peak", n)},
                                 [n - 1] if n > 0 else [], 10 + n // BLOCK_STEPS)
        if n in (15, 31, 47):
            ctx.finding(state, "cutting", Category.CHATTER, Severity.WARNING, f"toy chatter at step {n}",
                        tick0 + 3 * IPO, quantities={"index": u("toy.chatter", n)}, parents=[event])
            self.rundir.write_findings(state.findings)
        # L2: ledger, recorder, frames, keyframes
        self.ledger.check(ctx, state, tick1)
        self.recorder.record_tick(ticks, {"toy.feed_m_s": 0.01 + 0.001 * np.sin(t),
                                          "toy.err_m": np.column_stack([1e-6 * np.cos(t), 1e-6 * np.sin(t)]),
                                          "toy.mode": np.full(MACRO_IPO, n % 3)})
        self.recorder.record_macro(tick1, {"toy.volume_m3": float(field_.removed.sum()),
                                           "toy.flag": np.array([contact, n % 2 == 0]), **self.ledger.channel_values()})
        self.recorder.end_step(tick0, tick1, OMEGA, state.findings)
        self.frames.append_motion(state.bus_next.motion)
        self.timeline.write()
        state.swap_bus()
        state.clock.macro = n + 1
        state.clock.tick = tick1
        if self.keyframes.due(tick1):
            grid = FieldGrid(np.full((5, 4, 3), 300.0 + n), np.zeros(3), 0.002)
            self.keyframes.write(tick1, self.meshes(), changed=[brick], fields=self.fields(tick1),
                                 grids={"work_temp_K": grid}, chips={"mass_kg": np.array(chips.emitted[-3:])})
        if self.checkpointer.due(tick1):
            self.checkpointer.write(state, tick=tick1, op="OP10", participants=self.participants(), trace=ctx.trace)

    def finish(self) -> None:
        self.timeline.close(self.block, self.state.clock.tick, ticks={"cutting": self.state.clock.tick
                                                                       - self.timeline.records[self.block]["tick0"]})
        self.timeline.write()
        self.recorder.finish()
        self.keyframes.write_attributes({b: {"deviation_m": np.full(8, 1e-6 * b, np.float32)} for b in range(4)})
        self.frames.close()
        self.ctx.trace.close()
        self.rundir.write_findings(self.state.findings)

    def summary(self) -> dict:
        s = self.state
        return {"macro": s.clock.macro, "removed": s.removal.removed.tolist(), "heights": s.removal.heights.tolist(),
                "work": {str(k): v for k, v in s.cutting.work.items()}, "history": s.cutting.history,
                "pool": s.cutting.pool.tolist(), "alias_is_pool": s.cutting.alias is s.cutting.pool,
                "view_shares": bool(np.shares_memory(s.cutting.window, s.cutting.pool)),
                "chips": [s.chips.in_flight_kg, s.chips.settled_kg], "ledger": s.ledger.__dict__,
                "findings": len(s.findings), "conservation": self.ledger.summary()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--crash-after", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    toy = Toy(Path(args.run_dir))
    toy.resume() if args.resume else toy.start()
    while toy.state.clock.macro < args.steps:
        toy.step()
        if args.crash_after is not None and toy.state.clock.macro >= args.crash_after:
            os._exit(17)
    toy.finish()
    (Path(args.run_dir) / "toy_summary.json").write_text(json.dumps(toy.summary(), indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import toysim  # noqa: E402  (the classes must live in module 'toysim', not '__main__')

    sys.exit(toysim.main())
