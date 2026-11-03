"""Checkpoints (cnctwin.core.checkpoint): the state codec, writing and retention, refusal, and a resume in a
fresh process that continues bit-identically (tests/unit/core/toysim.py)."""

import collections
import enum
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from cnctwin.core import checkpoint as C
from cnctwin.core.abi import event_buffer
from cnctwin.core.bus import Bus, CuttingContact, PathFrame, ProcessCoefficients
from cnctwin.core.errors import CheckpointError
from cnctwin.core.findings import Category, Finding, Severity
from cnctwin.core.rundir import result_set_digest
from cnctwin.core.state import SimState, ToolSlot, ToolTable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class Colour(enum.Enum):
    RED = "red"


class Pair(NamedTuple):
    a: np.ndarray
    b: float


@dataclass
class Inner:
    values: np.ndarray
    names: list = field(default_factory=list)


@dataclass
class Holder:
    """A domain-like state through the generic codec."""

    inner: Inner
    by_slot: dict
    pair: Pair
    seen: set
    queue: collections.deque
    colour: Colour = Colour.RED
    scalar: np.float32 = np.float32(1.5)
    view: np.ndarray | None = None
    cache: object = None
    __checkpoint_exclude__ = ("cache",)


class Own:
    """A state that ships its own to_arrays / from_arrays."""

    def __init__(self, grid, step):
        self.grid, self.step = grid, step

    def to_arrays(self):
        return {"grid": self.grid}, {"step": self.step}

    @classmethod
    def from_arrays(cls, arrays, meta):
        return cls(arrays["grid"], meta["step"])


ALLOWED = ("cnctwin", __name__)


def round_trip(obj):
    document, arrays = C.encode(obj, ALLOWED)
    text = json.dumps(document)                 # the tree is plain JSON
    return C.decode(json.loads(text), {k: v.copy() for k, v in arrays.items()}, ALLOWED), document, arrays


def test_codec_keeps_values_types_bits_and_sharing():
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    shared = [1, 2]
    nan_payload = struct.unpack("<d", struct.pack("<Q", 0x7FF800000000BEEF))[0]
    obj = {
        "holder": Holder(Inner(np.zeros(3), ["a"]), {1: shared, 2: shared, (3, 4): "t"}, Pair(base, -0.0),
                         {3, 1, 2}, collections.deque([1, 2], maxlen=5), view=base[1:3, ::2], cache=lambda: 0),
        "base": base, "again": base, "fortran": np.asfortranarray(np.ones((3, 2))),
        "floats": [0.1 * i for i in range(100)], "ints": list(range(70)), "special": [math.inf, -math.inf, nan_payload],
        "np": [np.int16(-3), np.float32(0.1), np.bool_(True), np.complex128(1 - 2j)], "bytes": b"\x00\x01",
        "path": Path("/tmp/x"), "own": Own(np.ones((2, 2)), 7), "od": collections.OrderedDict([("z", 1), ("a", 2)]),
        "dd": collections.defaultdict(list, {"k": [1]}), "frozen": frozenset({"x"}), "none": None,
        "$weird": {"$a": 3}, 5: "int key",
    }
    back, document, arrays = round_trip(obj)
    h = back["holder"]
    assert back["base"] is back["again"]
    assert np.shares_memory(h.view, back["base"]) and h.view.base is not None
    h.view[0, 0] = -1.0
    assert back["base"][1, 0] == -1.0                         # a view stays a view of its base
    assert h.by_slot[1] is h.by_slot[2] and h.by_slot[(3, 4)] == "t"
    assert isinstance(h.pair, Pair) and h.pair.a is back["base"] and math.copysign(1, h.pair.b) < 0
    assert h.seen == {1, 2, 3} and h.queue.maxlen == 5 and list(h.queue) == [1, 2] and h.colour is Colour.RED
    assert type(h.scalar) is np.float32 and h.scalar == np.float32(1.5) and h.cache is None
    assert back["floats"] == obj["floats"] and back["ints"] == obj["ints"]
    assert struct.pack("<d", back["special"][2]) == struct.pack("<d", nan_payload)
    assert back["special"][:2] == [math.inf, -math.inf]
    assert [type(v) for v in back["np"]] == [np.int16, np.float32, np.bool_, np.complex128]
    assert back["np"][1] == np.float32(0.1) and back["bytes"] == b"\x00\x01" and back["path"] == Path("/tmp/x")
    assert isinstance(back["own"], Own) and back["own"].step == 7
    assert list(back["od"]) == ["z", "a"] and back["dd"]["new"] == [] and back["frozen"] == frozenset({"x"})
    assert back["$weird"] == {"$a": 3} and back[5] == "int key"
    fortran = back["fortran"]
    assert fortran.flags.c_contiguous and fortran.flags.writeable and np.array_equal(fortran, np.ones((3, 2)))
    assert len(arrays) < len(document["arrays"])           # base, again and the view are stored once


def test_simstate_round_trip_with_bus_and_findings():
    state = SimState()
    state.tools = ToolTable((ToolSlot(0, 1, "endmill", "holder"),))
    state.rng_counters = {"wear.chipping": 12}
    state.clock.tick = 123_000
    state.bus_prev.motion.paths = [PathFrame(tick=np.arange(3, dtype=np.int64), tip_S_m=np.ones((3, 3)))]
    state.bus_next.cutting.contacts = [CuttingContact(contact=0, shaft_work_J=2.5)]
    state.bus_next.cutting.coefficients = {3: ProcessCoefficients(tool_slot=3, h_grid_m=np.geomspace(1e-7, 1e-3, 8))}
    state.findings.add(Finding(Category.CHATTER, Severity.WARNING, "c", 0.1, 100, "dynamics", pos_W_m=(1.0, 2.0, 3.0)))
    state.cutting = {"events": event_buffer(4, 3)}
    back, _, _ = round_trip(state)
    assert back.tools.slots[0].tool_id == "endmill" and back.rng_counters == {"wear.chipping": 12}
    assert back.bus_prev.motion.paths[0].tick.dtype == np.int64
    assert back.bus_next.cutting.coefficients[3].h_grid_m[-1] == pytest.approx(1e-3)
    assert back.findings.items[0].category is Category.CHATTER and back.findings.items[0].pos_W_m == (1.0, 2.0, 3.0)
    assert back.cutting["events"].rec.shape == (4, 3) and back.cutting["events"].n.dtype == np.int64
    for array in (back.bus_prev.motion.paths[0].tip_S_m, back.cutting["events"].rec):
        assert array.flags.c_contiguous and array.flags.writeable


def test_codec_refuses_what_it_cannot_restore():
    with pytest.raises(CheckpointError, match="cannot be checkpointed"):
        C.encode({"f": lambda: 1}, ALLOWED)
    with pytest.raises(CheckpointError, match="Python objects"):
        C.encode({"o": np.array([object()])}, ALLOWED)

    def local():
        @dataclass
        class Local:
            x: int = 1
        return Local()
    with pytest.raises(CheckpointError, match="local class"):
        C.encode(local(), ALLOWED)
    with pytest.raises(CheckpointError, match="outside"):
        C.encode(Inner(np.zeros(1)), ("cnctwin",))
    document, arrays = C.encode(Inner(np.zeros(1)), ALLOWED)
    with pytest.raises(CheckpointError, match="outside"):
        C.decode(document, arrays, ("cnctwin",))


def write_state(tmp_path, **kw):
    ck = C.Checkpointer(tmp_path, project_hash="p" * 64, build="b" * 64, interval_ticks=1000, allowed_modules=ALLOWED,
                        **kw)
    state = SimState()
    state.removal = Own(np.arange(6.0), 1)
    return ck, state


def test_write_markers_retention_and_restore(tmp_path):
    ck, state = write_state(tmp_path, keep_last=2)
    assert not ck.due(999) and ck.due(1000)
    for tick in (1000, 2000, 3000, 4000):
        state.clock.tick = tick
        ck.write(state, tick=tick, op="OP10", op_end=(tick == 2000))
    assert [t for t, _ in ck.markers()] == [2000, 3000, 4000]  # the last two and the operation end
    marker = json.loads((tmp_path / "checkpoints" / "ck_4000.json").read_text())
    assert marker["format"] == "cnctwin-checkpoint" and marker["npz"] == "ck_4000.npz" and marker["time_s"] == 0.004
    assert not list((tmp_path / "checkpoints").glob(".*"))
    restored = ck.load(ck.latest())
    assert restored.tick == 4000 and restored.state.clock.tick == 4000
    assert np.array_equal(restored.state.removal.grid, np.arange(6.0)) and restored.state.removal.grid.flags.writeable
    assert ck.nearest(3500).name == "ck_3000.json" and ck.nearest(10) is None
    resumed = ck.resume(tmp_path / "checkpoints" / "ck_2000.json")
    assert resumed.tick == 2000 and [t for t, _ in ck.markers()] == [2000]   # later checkpoints are gone
    assert json.loads((tmp_path / "findings.json").read_text())["findings"] == []


def test_refuses_other_project_build_or_damaged_files(tmp_path):
    ck, state = write_state(tmp_path)
    ck.write(state, tick=1000)
    path = ck.latest()
    other = C.Checkpointer(tmp_path, project_hash="q" * 64, build="b" * 64, allowed_modules=ALLOWED)
    with pytest.raises(CheckpointError, match="another project"):
        other.load(path)
    newer = C.Checkpointer(tmp_path, project_hash="p" * 64, build="c" * 64, allowed_modules=ALLOWED)
    with pytest.raises(CheckpointError, match="another build"):
        newer.load(path)
    npz = path.with_suffix(".npz")
    with open(npz, "ab") as handle:
        handle.write(b"x")
    with pytest.raises(CheckpointError, match="size"):
        ck.load(path)


def test_an_uncommitted_checkpoint_does_not_exist(tmp_path):
    ck, state = write_state(tmp_path)
    ck.write(state, tick=1000)
    (tmp_path / "checkpoints" / "ck_2000.npz").write_bytes(b"half")      # crashed before its marker
    (tmp_path / "checkpoints" / ".ck_3000.npz.tmp").write_bytes(b"half")
    assert [t for t, _ in ck.markers()] == [1000]
    ck.write(state, tick=4000)
    assert not (tmp_path / "checkpoints" / ".ck_3000.npz.tmp").exists()


def run_toy(run_dir, *args):
    env = dict(os.environ)
    return subprocess.run([sys.executable, str(HERE / "toysim.py"), str(run_dir), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=240)


def test_resume_in_a_fresh_process_is_bit_identical(tmp_path):
    whole = run_toy(tmp_path / "whole", "--steps", "60")
    assert whole.returncode == 0, whole.stderr
    crashed = run_toy(tmp_path / "split", "--steps", "60", "--crash-after", "26")
    assert crashed.returncode == 17, crashed.stderr
    assert (tmp_path / "split" / "checkpoints" / "ck_200000.json").exists()
    assert not (tmp_path / "split" / "toy_summary.json").exists()
    resumed = run_toy(tmp_path / "split", "--steps", "60", "--resume")
    assert resumed.returncode == 0, resumed.stderr
    a = result_set_digest(tmp_path / "whole")
    b = result_set_digest(tmp_path / "split")
    assert a == b
    assert any(name.startswith("series/fast/") for name in a) and any(name.startswith("keyframes/kf_") for name in a)
    assert "frames.bin" in a and "trace.jsonl" in a and "findings.json" in a
    summary_a = json.loads((tmp_path / "whole" / "toy_summary.json").read_text())
    summary_b = json.loads((tmp_path / "split" / "toy_summary.json").read_text())
    assert summary_a == summary_b
    assert summary_b["alias_is_pool"] and summary_b["view_shares"] and summary_b["findings"] == 3
    assert summary_b["conservation"]["mass"]["ok"] and summary_b["conservation"]["energy"]["ok"]
    for name in ("timeline.json", "channels.json"):
        assert (tmp_path / "whole" / name).read_bytes() == (tmp_path / "split" / name).read_bytes()


def test_mappings_and_array_subclasses():
    from cnctwin.db.simap import SIMap
    view = SIMap({"travel_mm": [0, 10], "nested": {"speed_rpm": 1000}})
    back, _, _ = round_trip({"m": view, "again": view})
    assert back["m"] is back["again"] and back["m"]["travel_m"] == [0.0, 0.01]
    assert back["m"].nested.speed_rad_s == pytest.approx(1000 * 2 * math.pi / 60)
    with pytest.raises(CheckpointError, match="ndarray subclass"):
        C.encode({"masked": np.ma.masked_array([1.0, 2.0], mask=[True, False])}, ALLOWED)
