"""The removal domain's SimState field (``state.removal``), checkpointable as arrays + JSON metadata.

Everything that affects later results is stored: every workpiece's tri-dexel field (all levels, the pool and
the brick flags), its target, its polar field, the cumulative removed lengths (Python integers, stored as
text), the per-contact accumulators and pass statistics, the finding episodes, the air-cut runs and the tool
profiles. ``from_arrays(to_arrays())`` continues bit for bit.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EventBuffer, event_buffer
from . import abi as A
from . import field as FD
from . import polar as PO
from .schema import RemovalSettings

FORMAT_VERSION = 1


@dataclass
class Workpiece:
    index: int
    field: A.DexelField
    target: A.TargetField
    polar: A.PolarField | None = None
    rho_kg_m3: float = 0.0
    kind: str = "block"
    held_boxes: list = field(default_factory=list)          # [(lo [3], hi [3])] in S
    removed_q: list = field(default_factory=list)           # [level][family] cumulative removed length (int)
    polar_q2: int = 0                                       # cumulative polar radial sum of q^2 differences
    next_sev: int = A.SEV_FIRST_CUT                         # next surface event id the loop gives out
    polar_turned_rad: float = 0.0                           # spindle rotation since the last polar -> tri-dexel sync
    peak_blocks_step: int = 0                               # most pool blocks one macro step used
    initial_volume_m3: float = 0.0
    target_volume_m3: float = 0.0
    notes: list = field(default_factory=list)
    one_body: bool = False                                  # one component at the last connectivity check, removals
                                                            #   only since (the incremental check applies)


@dataclass
class ToolShape:
    """What removal needs of one tool slot (built from tooling's geometry at init)."""

    slot: int
    kind: int                                   # abi.TK_*
    rotating: bool
    centre_cutting: bool
    hand_sign: float
    cut: A.Profile
    whole: A.Profile
    insert: A.InsertRegion
    helix: A.HelixSweep
    slabs: A.ToothSlabs
    r_dead_m: float = 0.0
    z_dead_m: float = 0.0
    radius_m: float = 0.0
    edge_version: int = -1


@dataclass
class RemovalState:
    settings: RemovalSettings
    workpieces: list = field(default_factory=list)          # Workpiece
    tools: dict = field(default_factory=dict)               # slot -> ToolShape
    n_paths: int = 1
    accum: list = field(default_factory=list)               # RemovalAccum per contact (= path)
    contact_slot: list = field(default_factory=list)        # tool slot per contact at the last step
    contact_work: list = field(default_factory=list)        # workpiece per contact
    events: EventBuffer = None
    pass_stats: np.ndarray = None                           # [n_paths, N_PASS] f8
    prev_field_m3: np.ndarray = None                        # [n_paths] field volume of the previous step
    prev_tick: np.ndarray = None                            # [n_paths] i8 last tick of the previous step
    air: np.ndarray = None                                  # [n_paths, N_AIR] f8 open air-cut runs
    episodes: dict = field(default_factory=dict)            # "kind:contact" -> last tick of the open episode
    macro: int = 0
    last_connect_macro: int = -10 ** 9
    loose_log: list = field(default_factory=list)           # released bodies (dicts)
    counts: dict = field(default_factory=dict)              # event kind -> count (finalize)
    notes: list = field(default_factory=list)
    scratch: A.Scratch = None
    last_inputs: object = None                              # RemovalInputs of the current step (not stored)
    step_surface_ids: list = field(default_factory=list)    # transient

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict[str, np.ndarray] = {}
        meta: dict = {"format": FORMAT_VERSION, "settings": _settings_meta(self.settings), "n_paths": self.n_paths,
                      "macro": self.macro, "last_connect_macro": self.last_connect_macro,
                      "episodes": sorted([k, int(v)] for k, v in self.episodes.items()),
                      "loose_log": self.loose_log, "counts": dict(sorted(self.counts.items())),
                      "notes": list(self.notes), "contact_slot": [int(v) for v in self.contact_slot],
                      "contact_work": [int(v) for v in self.contact_work], "workpieces": [], "tools": []}
        for w, wp in enumerate(self.workpieces):
            arrays.update(FD.to_arrays(wp.field, f"w{w}.f."))
            arrays.update(FD.target_to_arrays(wp.target, f"w{w}.t."))
            if wp.polar is not None:
                arrays.update(PO.to_arrays(wp.polar, f"w{w}.p."))
            meta["workpieces"].append({
                "index": wp.index, "rho_kg_m3": wp.rho_kg_m3, "kind": wp.kind, "has_polar": wp.polar is not None,
                "held_boxes": [[list(map(float, lo)), list(map(float, hi))] for lo, hi in wp.held_boxes],
                "removed_q": [[str(int(v)) for v in row] for row in wp.removed_q], "polar_q2": str(int(wp.polar_q2)),
                "next_sev": int(wp.next_sev), "polar_turned_rad": float(wp.polar_turned_rad),
                "peak_blocks_step": int(wp.peak_blocks_step), "initial_volume_m3": float(wp.initial_volume_m3),
                "target_volume_m3": float(wp.target_volume_m3), "notes": list(wp.notes),
                "one_body": bool(wp.one_body)})
        for c, acc in enumerate(self.accum):
            for name in A.RemovalAccum._fields:
                arrays[f"c{c}.acc.{name}"] = np.array(getattr(acc, name))
        if self.events is not None:
            arrays["events.rec"] = np.array(self.events.rec)
            arrays["events.n"] = np.array(self.events.n)
        for name in ("pass_stats", "prev_field_m3", "prev_tick", "air"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = np.array(value)
        for slot, ts in sorted(self.tools.items()):
            prefix = f"tool{slot}."
            for part in ("cut", "whole", "insert", "helix", "slabs"):
                obj = getattr(ts, part)
                for name in type(obj)._fields:
                    arrays[f"{prefix}{part}.{name}"] = np.array(getattr(obj, name))
            meta["tools"].append({"slot": ts.slot, "kind": ts.kind, "rotating": ts.rotating,
                                  "centre_cutting": ts.centre_cutting, "hand_sign": ts.hand_sign,
                                  "r_dead_m": ts.r_dead_m, "z_dead_m": ts.z_dead_m, "radius_m": ts.radius_m,
                                  "edge_version": ts.edge_version})
        return arrays, json.loads(json.dumps(meta))

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "RemovalState":
        if int(meta.get("format", 0)) != FORMAT_VERSION:
            raise ValueError(f"removal state format {meta.get('format')} is not {FORMAT_VERSION}")
        settings = _settings_from_meta(meta["settings"])
        st = cls(settings=settings, n_paths=int(meta["n_paths"]), macro=int(meta["macro"]),
                 last_connect_macro=int(meta["last_connect_macro"]),
                 episodes={k: int(v) for k, v in meta["episodes"]}, loose_log=list(meta["loose_log"]),
                 counts=dict(meta["counts"]), notes=list(meta["notes"]),
                 contact_slot=[int(v) for v in meta["contact_slot"]],
                 contact_work=[int(v) for v in meta["contact_work"]])
        for w, wm in enumerate(meta["workpieces"]):
            fld = FD.from_arrays(arrays, f"w{w}.f.")
            tgt = FD.target_from_arrays(arrays, f"w{w}.t.")
            pol = PO.from_arrays(arrays, f"w{w}.p.") if wm["has_polar"] else None
            st.workpieces.append(Workpiece(
                index=int(wm["index"]), field=fld, target=tgt, polar=pol, rho_kg_m3=float(wm["rho_kg_m3"]),
                kind=str(wm["kind"]), held_boxes=[(np.array(lo), np.array(hi)) for lo, hi in wm["held_boxes"]],
                removed_q=[[int(v) for v in row] for row in wm["removed_q"]], polar_q2=int(wm["polar_q2"]),
                next_sev=int(wm["next_sev"]), polar_turned_rad=float(wm["polar_turned_rad"]),
                peak_blocks_step=int(wm["peak_blocks_step"]), initial_volume_m3=float(wm["initial_volume_m3"]),
                target_volume_m3=float(wm["target_volume_m3"]), notes=list(wm["notes"]),
                one_body=bool(wm.get("one_body", False))))
        c = 0
        while f"c{c}.acc.removed_q" in arrays:
            st.accum.append(A.RemovalAccum(*(np.ascontiguousarray(np.array(arrays[f"c{c}.acc.{name}"]))
                                             for name in A.RemovalAccum._fields)))
            c += 1
        if "events.rec" in arrays:
            st.events = EventBuffer(np.ascontiguousarray(np.array(arrays["events.rec"])),
                                    np.ascontiguousarray(np.array(arrays["events.n"])))
        for name in ("pass_stats", "prev_field_m3", "prev_tick", "air"):
            if name in arrays:
                setattr(st, name, np.ascontiguousarray(np.array(arrays[name])))
        for tm in meta["tools"]:
            slot = int(tm["slot"])
            prefix = f"tool{slot}."
            parts = {}
            for part, cls_ in (("cut", A.Profile), ("whole", A.Profile), ("insert", A.InsertRegion),
                               ("helix", A.HelixSweep), ("slabs", A.ToothSlabs)):
                parts[part] = cls_(*(np.ascontiguousarray(np.array(arrays[f"{prefix}{part}.{name}"]))
                                     for name in cls_._fields))
            st.tools[slot] = ToolShape(slot=slot, kind=int(tm["kind"]), rotating=bool(tm["rotating"]),
                                       centre_cutting=bool(tm["centre_cutting"]), hand_sign=float(tm["hand_sign"]),
                                       r_dead_m=float(tm["r_dead_m"]), z_dead_m=float(tm["z_dead_m"]),
                                       radius_m=float(tm["radius_m"]), edge_version=int(tm["edge_version"]), **parts)
        st.scratch = A.make_scratch()
        return st


N_PASS = 8          # pass_stats columns
PS_ACTIVE = 0
PS_START_TICK = 1
PS_V_FIELD = 2
PS_V_ANALYTIC = 3
PS_V_X = 4          # x family volume over the pass (field spread)
PS_V_Y = 5
PS_LAST_TICK = 6
N_AIR = 8           # air columns
AIR_ACTIVE = 0
AIR_START_TICK = 1
AIR_X0 = 2
AIR_Y0 = 3
AIR_Z0 = 4
AIR_TIME_S = 5      # length of the open run
AIR_REPORTED = 6


def new_state(settings: RemovalSettings, n_paths: int, n_levels: int) -> RemovalState:
    st = RemovalState(settings=settings, n_paths=int(n_paths))
    st.accum = [A.new_accum(n_levels) for _ in range(st.n_paths)]
    st.contact_slot = [-1] * st.n_paths
    st.contact_work = [0] * st.n_paths
    st.events = event_buffer(int(settings.event_capacity), A.EVENT_WIDTH)
    st.pass_stats = np.zeros((st.n_paths, N_PASS))
    st.prev_field_m3 = np.zeros(st.n_paths)
    st.prev_tick = np.full(st.n_paths, -1, np.int64)
    st.air = np.zeros((st.n_paths, N_AIR))
    st.scratch = A.make_scratch()
    return st


def _settings_meta(s: RemovalSettings) -> dict:
    out = {}
    for item in dataclasses.fields(s):
        value = getattr(s, item.name)
        if item.name == "refine":
            value = [[list(lo), list(hi), int(f)] for lo, hi, f in value]
        out[item.name] = value
    return out


def _settings_from_meta(meta: dict) -> RemovalSettings:
    values = dict(meta)
    values["refine"] = tuple((tuple(lo), tuple(hi), int(f)) for lo, hi, f in values.get("refine", []))
    return RemovalSettings(**values)
