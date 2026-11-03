"""The surface domain's SimState field (checkpointable: to_arrays / from_arrays round-trip bit-identically).

Everything a continuation needs: the patch store (patches, passes, open lines), the burr edges, the marks,
composite damage, thread features, per-tool and per-path running state (last generating event, spindle and
dwell state of the mark detectors, recent thrust), counters and the channel values of the last step. The
tool outlines, the workpiece query and the brick grid are not state: SurfaceDomain attaches them again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .schema import SurfaceSettings
from .store import Store
from .threads import ThreadFeature

BF_NAMES = ("sx", "sy", "sz", "nsx", "nsy", "nsz", "count", "h_max", "h_sum", "t_max", "hx_sum", "psi_sum", "bx_max",
            "neg_max")
BI_NAMES = ("workpiece", "brick", "kind", "slot", "op", "first_tick", "last_tick", "block", "tip_last", "removed",
            "finding")
BF = {n: i for i, n in enumerate(BF_NAMES)}
BI = {n: i for i, n in enumerate(BI_NAMES)}
BURR_KINDS = ("rollover", "breakout", "drill_uniform", "drill_crown", "fibre_pullout")


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass
class BurrStore:
    bf: np.ndarray = field(default_factory=lambda: np.zeros((0, len(BF_NAMES))))
    bi: np.ndarray = field(default_factory=lambda: np.zeros((0, len(BI_NAMES)), np.int64))
    n: int = 0
    index: dict = field(default_factory=dict)          # (workpiece, brick, facet) -> row
    dropped: int = 0

    def row(self, key: tuple, cap: int) -> int:
        if key in self.index:
            return self.index[key]
        if self.n >= cap:
            return -1
        if self.n == self.bf.shape[0]:
            size = max(16, 2 * self.bf.shape[0])
            bf = np.zeros((size, len(BF_NAMES)))
            bi = np.zeros((size, len(BI_NAMES)), np.int64)
            bf[:self.n] = self.bf[:self.n]
            bi[:self.n] = self.bi[:self.n]
            self.bf, self.bi = bf, bi
        i = self.n
        self.n += 1
        self.bf[i] = 0.0
        self.bi[i] = 0
        self.bi[i, BI["workpiece"]], self.bi[i, BI["brick"]] = key[0], key[1]
        self.bi[i, BI["first_tick"]] = -1
        self.bi[i, BI["block"]] = -1
        self.bi[i, BI["finding"]] = -1
        self.index[key] = i
        return i

    def normal(self, i: int) -> np.ndarray:
        n = self.bf[i, BF["nsx"]:BF["nsz"] + 1]
        norm = float(np.linalg.norm(n))
        return n / norm if norm > 0.0 else np.zeros(3)

    def location(self, i: int) -> np.ndarray:
        c = self.bf[i, BF["count"]]
        return self.bf[i, BF["sx"]:BF["sz"] + 1] / c if c > 0 else np.zeros(3)


@dataclass
class SurfaceState:
    settings: SurfaceSettings
    seed: int
    store: Store = field(default_factory=Store)
    burrs: BurrStore = field(default_factory=BurrStore)
    marks: list = field(default_factory=list)
    marks_dropped: int = 0
    damage: list = field(default_factory=list)          # composite delamination records
    threads: dict = field(default_factory=dict)         # (slot, op) -> ThreadFeature
    last_event: dict = field(default_factory=dict)      # slot -> last generating event
    path_marks: dict = field(default_factory=dict)      # path -> mark detector state
    thrust: dict = field(default_factory=dict)          # slot -> largest thrust of the last two steps
    counters: dict = field(default_factory=lambda: {"events": 0, "profiled": 0, "unprofiled": 0, "rejected": 0,
                                                    "lines": 0, "lines_short": 0, "exits": 0, "exits_interior": 0,
                                                    "recut_dents": 0, "macro": 0})
    findings: dict = field(default_factory=dict)        # once-only finding keys -> finding id
    values: dict = field(default_factory=dict)          # channel values of the last step
    op_index: int = 0
    estimates: list = field(default_factory=list)       # model defaults in use (surface.json)
    results: list = field(default_factory=list)         # evaluated cross profiles (bounded by patches)

    def to_arrays(self) -> tuple[dict, dict]:
        arrays, store_meta = self.store.to_arrays("store.")
        arrays["burr.bf"] = self.burrs.bf[:self.burrs.n].copy()
        arrays["burr.bi"] = self.burrs.bi[:self.burrs.n].copy()
        thread_meta = []
        for k, (key, th) in enumerate(sorted(self.threads.items())):
            arrays[f"thread.{k}.pts"] = th.pts.copy()
            arrays[f"thread.{k}.vec"] = np.concatenate([th.axis_point, th.axis_dir, th.tip_sum])
            thread_meta.append({"key": [int(v) for v in key], "slot": th.slot, "op": th.op, "pitch_m": th.pitch_m,
                                "lead_m": th.lead_m, "form_angle_rad": th.form_angle_rad, "hand": th.hand,
                                "starts": th.starts, "rotating": th.rotating, "keep_every": th.keep_every,
                                "seen": th.seen, "tip_n": th.tip_n, "block_first": th.block_first,
                                "block_last": th.block_last, "first_tick": th.first_tick, "last_tick": th.last_tick})
        meta = {"settings": dict(self.settings.__dict__), "seed": int(self.seed), "store": store_meta,
                "burr_index": [[int(a), int(b), int(c), int(i)] for (a, b, c), i in sorted(self.burrs.index.items())],
                "burr_dropped": int(self.burrs.dropped), "marks": json_safe(self.marks),
                "marks_dropped": int(self.marks_dropped), "damage": json_safe(self.damage), "threads": thread_meta,
                "last_event": json_safe({str(k): v for k, v in sorted(self.last_event.items())}),
                "path_marks": json_safe({str(k): v for k, v in sorted(self.path_marks.items())}),
                "thrust": json_safe({str(k): v for k, v in sorted(self.thrust.items())}),
                "counters": dict(self.counters), "findings": dict(sorted(self.findings.items())),
                "values": json_safe(dict(sorted(self.values.items()))), "op_index": int(self.op_index),
                "estimates": list(self.estimates), "results": json_safe(self.results)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "SurfaceState":
        obj = cls(settings=SurfaceSettings(**meta["settings"]), seed=int(meta["seed"]))
        obj.store = Store.from_arrays(arrays, meta["store"], "store.")
        bf = np.ascontiguousarray(arrays["burr.bf"], dtype=float).copy()
        bi = np.ascontiguousarray(arrays["burr.bi"], dtype=np.int64).copy()
        obj.burrs = BurrStore(bf=bf, bi=bi, n=bf.shape[0],
                              index={(int(a), int(b), int(c)): int(i) for a, b, c, i in meta["burr_index"]},
                              dropped=int(meta["burr_dropped"]))
        obj.marks = list(meta["marks"])
        obj.marks_dropped = int(meta["marks_dropped"])
        obj.damage = list(meta["damage"])
        for k, m in enumerate(meta["threads"]):
            vec = np.asarray(arrays[f"thread.{k}.vec"], dtype=float)
            th = ThreadFeature(slot=int(m["slot"]), op=int(m["op"]), pitch_m=float(m["pitch_m"]),
                               lead_m=float(m["lead_m"]), form_angle_rad=float(m["form_angle_rad"]),
                               hand=float(m["hand"]), starts=int(m["starts"]), axis_point=vec[:3].copy(),
                               axis_dir=vec[3:6].copy(), rotating=bool(m["rotating"]),
                               pts=np.ascontiguousarray(arrays[f"thread.{k}.pts"], dtype=float).reshape(-1, 7).copy(),
                               keep_every=int(m["keep_every"]), seen=int(m["seen"]), tip_sum=vec[6:9].copy(),
                               tip_n=int(m["tip_n"]), block_first=int(m["block_first"]), block_last=int(m["block_last"]),
                               first_tick=int(m["first_tick"]), last_tick=int(m["last_tick"]))
            obj.threads[tuple(int(v) for v in m["key"])] = th
        obj.last_event = {int(k): v for k, v in meta["last_event"].items()}
        obj.path_marks = {int(k): v for k, v in meta["path_marks"].items()}
        obj.thrust = {int(k): v for k, v in meta["thrust"].items()}
        obj.counters = dict(meta["counters"])
        obj.findings = dict(meta["findings"])
        obj.values = {k: (math.nan if v is None else v) for k, v in meta["values"].items()}
        obj.op_index = int(meta["op_index"])
        obj.estimates = list(meta["estimates"])
        obj.results = list(meta["results"])
        return obj


__all__ = ["SurfaceState", "BurrStore", "BF", "BI", "BF_NAMES", "BI_NAMES", "BURR_KINDS", "json_safe"]
