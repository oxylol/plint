"""SurfaceDomain: roughness, marks, burrs, surface integrity and thread accuracy of the machined part
(docs/models/surface.md; contract: core/domain.py).

An L2 slow domain (no part in the hot loop). Per macro step it reads ``prev`` - cutting's surface events,
exit shocks, recut rows, loads and coefficients; motion's paths (tool pose, axes, following errors, spindle);
removal's contacts (path, workpiece, ap, ae); wear's per-element radial/axial wear, VB and BUE heights;
thermal's generated-surface temperatures; dynamics' chatter index - and writes **no bus record** (it is a
sink: surface.json, per-brick attribute arrays, findings, trace events and the :class:`SurfaceView` other
parts query, e.g. holding for burrs under later locators).

Geometry comes through protocols (protocols.py): the integration wires
``SurfaceDomain(tools=GeometryToolInfo(state.tooling.tools), query=RemovalDomain.geometry(state),
bricks=LatticeBricks(field.lv_origin_m[0], field.lv_cell_m[0], field.lv_dims[0]))``; without them the domain
builds the defaults documented in the model doc (tooling's geometries from state.tooling, the analytic
stock, the stock lattice).
"""

from __future__ import annotations

import math

import numpy as np

from .. import FORMAT_VERSION
from ..core.findings import Category, Severity
from ..units import REFERENCE_TEMP_K, ZERO_DEGC_K
from . import burrs as BU
from . import faces as FC
from . import integrity as IN
from . import kinematics as KIN
from . import marks as MK
from . import params as PRM
from . import threads as TH
from .channels import channels as surface_channels
from .gather import Gatherer, StepInput
from .lines import CLASSES, evaluate_line
from .protocols import GeometryToolInfo, LatticeBricks, RunDirBricks, StockQuery, parse_brick_key
from .roughness import IsoRules
from .schema import settings as read_settings
from .state import BF, BI, BURR_KINDS, SurfaceState, json_safe
from .store import PF, PI, QI

DOMAIN = "surface"
SHOCK_EXIT, SHOCK_BREAKTHROUGH, SHOCK_BREAKOUT = 2, 3, 4      # cutting.abi SHOCK_* codes (bus CuttingContact.shocks)
EDGE_END, EDGE_CORNER, EDGE_LIP, EDGE_CHISEL = 1, 2, 3, 4      # core.abi EDGE_* codes
HOLE_TYPES = ("twist_drill", "spot_drill", "centre_drill", "gun_drill", "indexable_drill", "reamer", "tap",
              "drill", "boring_head")


class SurfaceDomain:
    """Surface finish, marks, burrs, integrity and threads."""

    name = "surface"

    def __init__(self, tools=None, query=None, bricks=None, meshes=None):
        self.tools = tools
        self.query = query
        self.bricks = bricks
        self.meshes = meshes
        self._rt: dict = {}

    # ------------------------------------------------------------------ Domain interface
    def channels(self):
        return surface_channels()

    def kernel_inputs(self, ctx, state, bus):
        return None

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        return None

    def init(self, ctx, state) -> None:
        physics = PRM.field_of(ctx.project, "physics")
        st = read_settings(PRM.field_of(physics, "surface"))
        sst = SurfaceState(st, int(ctx.seed))
        sst.op_index = int(state.clock.op_index)
        state.surface = sst
        self._attach(ctx, state)
        sst.estimates = list(self._rt["material"].notes) + list(self._rt["notes"])
        ctx.trace.record(0.0, 0, DOMAIN, "surface:init", {"specs": len(self._rt["specs"]),
                                                          "min_chip_ratio": self._rt["material"].min_chip_ratio}, [])

    def _attach(self, ctx, state) -> None:
        """Runtime objects (not checkpointed): tools, query, bricks, rules, models, specs."""
        if self._rt.get("state_id") == id(state.surface) and self._rt.get("ctx_id") == id(ctx):
            return
        sst: SurfaceState = state.surface
        project = ctx.project
        notes = []
        tools = self.tools
        if tools is None:
            tooling = getattr(state, "tooling", None)
            geos = getattr(tooling, "tools", None)
            if geos:
                tools = GeometryToolInfo(geos)
        stock = PRM.field_of(project, "stock")
        kind = str(PRM.plain(PRM.field_of(stock, "kind")) or "block")
        size = PRM.field_of(stock, "size_m")
        if size is None and isinstance(stock, dict) and stock.get("size_mm") is not None:
            size = [float(v) * 1e-3 for v in stock["size_mm"]]
        size = [float(PRM.plain(v)) for v in PRM.plain(size)] if size is not None else None
        query = self.query
        if query is None and size is not None and kind in ("block", "cylinder", "tube"):
            query = StockQuery(kind, size)
            notes.append("workpiece query: the analytic stock (removal's field not wired): exit faces of burrs are "
                         "found on the stock outline only")
        elif query is None:
            notes.append("workpiece query: none wired and no analytic stock: burr exit faces cannot be resolved, exit "
                         "burrs are not predicted")
        bricks = self.bricks
        physics = PRM.field_of(project, "physics")
        lattice = PRM.num(physics, "lattice_m")
        if lattice is None:
            lat_mm = PRM.num(physics, "lattice_mm")
            lattice = lat_mm * 1e-3 if lat_mm is not None else 2.5e-4
        if bricks is None:
            if size is not None and kind in ("block", "cylinder", "tube"):
                bricks = LatticeBricks.for_stock(kind, size, float(lattice))
            else:
                bricks = LatticeBricks(np.zeros(3), float(lattice), np.full(3, 2 ** 40))
                notes.append("brick grid: removal's lattice not wired and no analytic stock: patches use a lattice "
                             "from the S origin (brick ids then differ from removal's)")
        material = PRM.material_props(PRM.field_of(project, "material"))
        env = PRM.field_of(project, "environment")
        ambient = PRM.num(env, "ambient_K")
        if ambient is None:
            amb_c = PRM.num(env, "ambient_degC")
            ambient = amb_c + ZERO_DEGC_K if amb_c is not None else REFERENCE_TEMP_K
        rules = IsoRules()
        self._rt = {"state_id": id(sst), "ctx_id": id(ctx), "tools": tools, "query": query, "bricks": bricks,
                    "material": material, "rules": rules, "rough": PRM.roughness(), "burr_p": PRM.burr(),
                    "integ_p": PRM.integrity(), "ambient_K": float(ambient), "specs": FC.read_specs(project),
                    "notes": notes, "silhouettes": {}, "wear": {}, "ctx": ctx, "state": state,
                    "burr_model": BU.BurrModel(PRM.burr(), material)}
        self._rt["gather"] = Gatherer(sst.settings, self._rt["rough"], material, sst.seed, ctx.tick_s, bricks,
                                      lambda line: self._close_line(line))

    # ------------------------------------------------------------------ macro step
    def macro_step(self, ctx, state, prev, nxt) -> None:
        sst: SurfaceState = state.surface
        self._attach(ctx, state)
        rt = self._rt
        rt["ctx"], rt["state"] = ctx, state
        sst.counters["macro"] += 1
        step_tick = int(getattr(prev.motion, "tick", 0))
        rt["wear"] = dict(getattr(prev.wear, "tools", {}) or {})
        removal_of = {int(c.contact): c for c in getattr(prev.removal, "contacts", []) or []}
        events_by_path: dict = {}
        cutting_paths: set = set()
        stats = {"events": 0, "profiled": 0, "sigma": math.nan}
        for contact in getattr(prev.cutting, "contacts", []) or []:
            slot = int(contact.tool_slot)
            if slot < 0:
                continue
            if rt["tools"] is None:
                n_ev = len(np.asarray(contact.surface_events).reshape(-1, 16))
                sst.counters["events"] += n_ev
                sst.counters["unprofiled"] += n_ev
                note = "tool geometry: no ToolInfo wired and no tooling state: surface events are counted, not profiled"
                if note not in sst.estimates:
                    sst.estimates.append(note)
                continue
            rc = removal_of.get(int(contact.contact))
            path = int(rc.path) if rc is not None else None
            frame = KIN.frame_of(prev.motion, slot, path)
            path = int(frame.path) if frame is not None else (path or 0)
            workpiece = int(rc.workpiece) if rc is not None else int(getattr(frame, "workpiece", 0) or 0)
            outline = rt["tools"].outline(slot)
            edge = rt["tools"].edge_model(slot)
            wear = rt["wear"].get(slot)
            rows = np.asarray(contact.surface_events, dtype=float).reshape(-1, 16)
            if float(contact.in_cut_s) > 0.0 or len(rows):
                cutting_paths.add(path)
            self._thrust(sst, slot, contact)
            if outline.thread is not None and len(rows):
                self._thread_events(sst, slot, outline, frame, rows, ctx)
                stats["events"] += len(rows)
                continue
            if len(rows):
                inp = StepInput(rows=rows, slot=slot, contact=int(contact.contact), frame=frame,
                                workpiece=workpiece, outline=outline, edge=edge, wear=wear,
                                recut=np.asarray(contact.recut, dtype=float).reshape(-1, 6),
                                h_max_m=float(contact.h_max_step_m), chip_ratio=self._chip_ratio(prev, slot),
                                op=sst.op_index)
                out = rt["gather"].ingest(sst.store, inp)
                for key in ("events", "profiled", "unprofiled", "rejected"):
                    sst.counters[key] += int(getattr(out, key))
                stats["events"] += out.events
                stats["profiled"] += out.profiled
                if out.last_event is not None:
                    sst.last_event[slot] = out.last_event
                ev = self._step_events(rows, out)
                if ev is not None:
                    prior = events_by_path.get(path)
                    events_by_path[path] = ev if prior is None else tuple(np.concatenate([a, b]) for a, b in zip(prior, ev))
                integ = self._integrity(prev, contact, rows, out.patches, sst)
                if integ is not None:
                    stats["sigma"] = integ.sigma_surface_Pa
                self._chatter_bue(prev, out.patches, sst, wear, rows, edge)
            self._exits(ctx, state, sst, contact, rc, frame, outline, edge, workpiece)
            self._composite(ctx, state, sst, contact, frame, outline, edge, workpiece)
        self._marks(ctx, state, sst, prev, cutting_paths, events_by_path)
        self._close_stale(sst, step_tick)
        self._values(sst, stats)

    # ------------------------------------------------------------------ helpers of the step
    @staticmethod
    def _step_events(rows, out):
        return out.arrays if out.arrays is not None and len(out.arrays[0]) else None

    @staticmethod
    def _chip_ratio(prev, slot: int) -> float:
        coeffs = getattr(prev.cutting, "coefficients", {}) or {}
        pc = coeffs.get(slot)
        if pc is None:
            return 0.0
        cr = np.asarray(getattr(pc, "chip_ratio", np.zeros(0)), dtype=float)
        cr = cr[np.isfinite(cr) & (cr > 0.0)]
        return float(np.mean(cr)) if len(cr) else 0.0

    @staticmethod
    def _thrust(sst, slot: int, contact) -> None:
        w = np.asarray(contact.wrench_tool_T0_mean, dtype=float).reshape(-1, 6)
        now = float(np.max(np.abs(w[:, 2]))) if len(w) else 0.0
        before = sst.thrust.get(slot, [0.0, 0.0])
        sst.thrust[slot] = [float(before[1]), now]

    def _thread_events(self, sst, slot, outline, frame, rows, ctx) -> None:
        ok = np.all(np.isfinite(rows), axis=1) & (np.linalg.norm(rows[:, 6:9], axis=1) > 0.0)
        r = rows[ok]
        if len(r) == 0:
            return
        key = (int(slot), int(sst.op_index))
        th = sst.threads.get(key)
        prof = outline.thread
        if th is None:
            pitch = float(np.asarray(prof.pitch_m)[0])
            lead = float(np.asarray(prof.lead_m)[0])
            mot = KIN.sample(frame, r[:1, 0], ctx.tick_s) if frame is not None else {}
            axis = mot["axis"][0] if (mot and outline.rotating) else np.array([0.0, 0.0, 1.0])
            th = TH.ThreadFeature(slot=int(slot), op=int(sst.op_index), pitch_m=pitch,
                                  lead_m=lead if lead > 0.0 else pitch * int(np.asarray(prof.starts)[0]),
                                  form_angle_rad=float(np.asarray(prof.form_angle_rad)[0]),
                                  hand=float(np.asarray(prof.thread_hand)[0]), starts=int(np.asarray(prof.starts)[0]),
                                  axis_point=np.zeros(3), axis_dir=np.asarray(axis, dtype=float),
                                  rotating=bool(outline.rotating))
            sst.threads[key] = th
        N = r[:, 6:9] / np.linalg.norm(r[:, 6:9], axis=1)[:, None]
        th.add(r[:, 3:6], N, int(sst.settings.thread_points))
        if outline.rotating and frame is not None:
            mot = KIN.sample(frame, r[:, 0], ctx.tick_s)
            if mot:
                th.tip_sum += np.sum(mot["tip"], axis=0)
                th.tip_n += len(r)
                blk = mot["block"]
                if th.block_first < 0:
                    th.block_first = int(blk[0])
                th.block_last = int(blk[-1])
        elif frame is not None:
            mot = KIN.sample(frame, r[:, 0], ctx.tick_s)
            if mot:
                if th.block_first < 0:
                    th.block_first = int(mot["block"][0])
                th.block_last = int(mot["block"][-1])
        if th.first_tick < 0:
            th.first_tick = int(r[0, 0])
        th.last_tick = int(np.max(r[:, 0]))

    def _integrity(self, prev, contact, rows, patches, sst):
        rt = self._rt
        in_cut = float(contact.in_cut_s)
        v = float(contact.mean_speed_m_s)
        L = float(contact.engaged_edge_m)
        if not patches or in_cut <= 0.0 or v <= 0.0 or L <= 0.0:
            return None
        p_t = float(contact.ploughing_J) / (in_cut * v * L)
        coeffs = (getattr(prev.cutting, "coefficients", {}) or {}).get(int(contact.tool_slot))
        ratio = None
        if coeffs is not None:
            kte = np.asarray(getattr(coeffs, "Kte_N_per_m", np.zeros(0)), dtype=float)
            kre = np.asarray(getattr(coeffs, "Kre_N_per_m", np.zeros(0)), dtype=float)
            if len(kte) and len(kre) == len(kte) and float(np.mean(kte)) > 0.0:
                ratio = float(np.mean(kre)) / float(np.mean(kte))
        mat = rt["material"]
        mu = mat.mu_sliding if mat.mu_sliding is not None else float(rt["integ_p"]["friction_default"])
        if ratio is None:
            ratio = 1.0 / max(mu, 1e-6)
        ok = np.all(np.isfinite(rows), axis=1)
        vb = float(np.mean(rows[ok, 11])) if ok.any() else 0.0
        er = float(np.mean(rows[ok, 10])) if ok.any() else 0.0
        temps = getattr(prev.thermal, "contact_surface_temp_K", {}) or {}
        T = float(temps.get(int(contact.contact), rt["ambient_K"]))
        judged = getattr(prev.thermal, "contact_damage", {}) or {}
        damage = judged.get(int(contact.contact))
        res = IN.evaluate(p_t * ratio, vb, er, v, T, rt["ambient_K"], mat, rt["integ_p"], mu, damage=damage)
        pf = sst.store.pf
        for p in patches:
            pf[p, PF["T_peak"]] = res.surface_temp_K
            pf[p, PF["sigma"]] = res.sigma_surface_Pa
            pf[p, PF["sigma_depth"]] = res.affected_depth_m
            pf[p, PF["hard_change"]] = res.hardness_change_frac
            pf[p, PF["hard_depth"]] = res.hardened_depth_m
            pf[p, PF["white"]] = float(res.white_layer)
            pf[p, PF["burn"]] = float(res.burn)
            pf[p, PF["overtemper"]] = float(res.overtemper)
            pf[p, PF["melt"]] = float(res.melt_smear)
        return res

    def _chatter_bue(self, prev, patches, sst, wear, rows, edge) -> None:
        dyn = prev.dynamics
        pf = sst.store.pf
        idx = float(getattr(dyn, "chatter_index", 0.0))
        if idx > sst.settings.chatter_index:
            for p in patches:
                pf[p, PF["chatter"]] = 1.0
                pf[p, PF["chatter_freq"]] = float(getattr(dyn, "chatter_freq_Hz", 0.0))
                pf[p, PF["chatter_index"]] = max(pf[p, PF["chatter_index"]], idx)
        if wear is not None:
            bue = np.asarray(getattr(wear, "bue_height_m", np.zeros((0, 0))), dtype=float)
            if bue.shape == np.shape(edge.elem_r_m) and len(rows):
                j = np.clip(rows[:, 1].astype(int), 0, bue.shape[0] - 1)
                k = np.clip(rows[:, 2].astype(int), 0, bue.shape[1] - 1)
                if np.any(bue[j, k] > 0.0):
                    for p in patches:
                        pf[p, PF["bue"]] = 1.0

    # ------------------------------------------------------------------ lines
    def _close_line(self, line) -> None:
        rt = self._rt
        sst: SurfaceState = rt["state"].surface
        tools = rt["tools"]
        outline = tools.outline(line.slot)
        edge = tools.edge_model(line.slot)
        wear = rt["wear"].get(line.slot)
        radial = axial = None
        if wear is not None:
            radial = np.asarray(getattr(wear, "radial_wear_m", np.zeros((0, 0))), dtype=float)
            axial = np.asarray(getattr(wear, "axial_wear_m", np.zeros((0, 0))), dtype=float)
            if radial.size == 0:
                radial = None
            if axial.size == 0:
                axial = None
        res = evaluate_line(line, outline.pts, edge, radial, axial, rt["rules"], sst.settings.half_spacings,
                            recovery=float(rt["rough"]["recovery_ratio"]), finish_radius_m=float(outline.finish_radius_m))
        sst.counters["lines"] += 1
        if res is None:
            sst.counters["lines_short"] += 1
            return
        self._aggregate(sst, res.patch, "f", res.roughness, res.length_m, res.kinematic.Rt_m)
        pf = sst.store.pf
        d = np.asarray(res.direction_S, dtype=float)
        if line.cylindrical:
            d = np.array([d[0], d[1], 0.0])
        pf[res.patch, PF["dx"]:PF["dz"] + 1] = d if len(d) == 3 else np.zeros(3)
        pf[res.patch, PF["vb"]] = res.mean_VB_m
        pf[res.patch, PF["u_n"]] = res.mean_u_n_m
        sst.values["last_Ra_m"] = res.roughness.Ra_m
        sst.values["last_Rz_m"] = res.roughness.Rz_m
        sst.values["last_Rt_m"] = res.roughness.Rt_m

    @staticmethod
    def _aggregate(sst, patch: int, tag: str, rough, length: float, pt: float) -> None:
        pf = sst.store.pf
        if rough.method == "none" or length <= 0.0:
            return
        pf[patch, PF[f"{tag}_len"]] += length
        pf[patch, PF[f"{tag}_ra"]] += rough.Ra_m * length
        pf[patch, PF[f"{tag}_rq"]] += rough.Rq_m * length
        pf[patch, PF[f"{tag}_rz"]] += rough.Rz_m * length
        pf[patch, PF[f"{tag}_rt_max"]] = max(pf[patch, PF[f"{tag}_rt_max"]], rough.Rt_m)
        pf[patch, PF[f"{tag}_rz_max"]] = max(pf[patch, PF[f"{tag}_rz_max"]], rough.Rz_m)
        pf[patch, PF[f"{tag}_ra_max"]] = max(pf[patch, PF[f"{tag}_ra_max"]], rough.Ra_m)
        pf[patch, PF[f"{tag}_lines"]] += 1.0
        pf[patch, PF[f"{tag}_cutoff"]] = rough.cutoff_m
        pf[patch, PF[f"{tag}_short"]] += 1.0 if rough.method != "R" else 0.0
        if tag == "f":
            pf[patch, PF["f_pt_max"]] = max(pf[patch, PF["f_pt_max"]], pt)
            pf[patch, PF["f_rsm"]] += rough.RSm_m * length

    def _close_stale(self, sst, now_tick: int) -> None:
        """Evaluate the lines that saw no event for longer than pass_gap, four revolutions and four of their
        own event intervals; close the passes whose patch saw none for longer than pass_gap and four
        revolutions."""
        g = self._rt["gather"]
        store = sst.store
        tick_s = self._rt["ctx"].tick_s
        base = sst.settings.pass_gap_s / tick_s
        for key in sorted(store.lines):
            line = store.lines.get(key)
            if line is None:
                continue
            gap = max(base, line.hold_ticks, 4.0 * line.mean_gap_ticks())
            if now_tick - line.last_tick > gap:
                g.close_line_key(store, key)
        for (p, slot), q in sorted(store.active.items()):
            w = abs(float((sst.last_event.get(slot) or {}).get("omega", 0.0)))
            hold = 4.0 * 2.0 * math.pi / w / tick_s if w > 0.0 else 0.0
            if now_tick - int(store.qi[q, QI["tick1"]]) > max(base, hold):
                g.close_pass(store, q)

    # ------------------------------------------------------------------ marks
    def _marks(self, ctx, state, sst, prev, cutting_paths, events_by_path) -> None:
        for frame in getattr(prev.motion, "paths", []) or []:
            p = int(frame.path)
            ps = sst.path_marks.setdefault(p, {})
            slot = int(frame.tool_slot)
            last = sst.last_event.get(slot)
            found = MK.dwell_and_stop(frame, ps, p in cutting_paths, last, sst.settings, ctx.tick_s, ctx.timeline)
            found += MK.servo_marks(frame, ps, events_by_path.get(p), sst.settings, ctx.tick_s)
            for mark in found:
                mark["slot"] = slot
                mark["op"] = sst.op_index
                self._add_mark(ctx, state, sst, mark)

    def _add_mark(self, ctx, state, sst, mark) -> None:
        p = int(mark.get("patch", -1))
        if 0 <= p < sst.store.n_patch:
            pf = sst.store.pf
            pf[p, PF["marks"]] += 1.0
            pf[p, PF["mark_max"]] = max(pf[p, PF["mark_max"]], abs(float(mark["height_m"])))
        if len(sst.marks) < sst.settings.max_marks:
            sst.marks.append(json_safe(mark))
        else:
            sst.marks_dropped += 1
        limit = self._mark_limit(sst, p)
        spindle = mark["kind"] == MK.KIND_SPINDLE_STOP
        if (limit > 0.0 and abs(float(mark["height_m"])) > limit) or spindle:
            key = f"mark:{mark['kind']}:{mark.get('block', -1)}:{p}"
            if key in sst.findings:
                return
            msg = (f"{mark['kind'].replace('_', ' ')} mark {abs(mark['height_m']) * 1e6:.2f} um deep on the finished "
                   f"surface") if not spindle else ("the spindle stopped while the tool was in the cut: a stop mark on "
                                                     "the finished surface (and a risk to the edge at the restart)")
            f = self._finding(ctx, state, Category.FINISH_OUT_OF_SPEC, Severity.WARNING, msg, int(mark["tick"]),
                              int(mark.get("block", -1)), {"height_m": float(mark["height_m"]), "limit_m": limit})
            sst.findings[key] = f.id

    def _mark_limit(self, sst, patch: int) -> float:
        """The mark depth that raises a finding on a patch: mark_limit, or the Rz (else Rt) specification of
        the roughness feature the patch lies on, the larger."""
        limit = sst.settings.mark_limit_m
        if limit <= 0.0:
            return 0.0
        if 0 <= patch < sst.store.n_patch:
            self._patch_spec(sst, patch)
            spec = sst.store.pf[patch, PF["spec_rz"]] or sst.store.pf[patch, PF["spec_rt"]]
            if spec > 0.0:
                return max(limit, float(spec))
        return limit

    def _patch_spec(self, sst, patch: int) -> None:
        """Store on the patch the limits of the first roughness specification whose selector names it
        (plane: normal within face_angle and plane offset within the tolerance; cylinder: normal perpendicular
        to the axis and radius within the tolerance; other faces: within one brick of the point)."""
        store = sst.store
        if store.pi[patch, PI["feature"]] != -1:
            return
        store.pi[patch, PI["feature"]] = -2                     # looked up: no specification
        n = store.normal(patch)
        c = store.centroid(patch)
        size = float(self._rt["bricks"].brick_size_m(int(store.pi[patch, PI["workpiece"]])))
        tol = max(sst.settings.plane_tol_m, 0.05 * size)
        for k, spec in enumerate(self._rt["specs"]):
            near = spec.get("near")
            if near is None:
                continue
            face = spec.get("face")
            if face == "plane" and spec.get("normal") is not None:
                ok = float(n @ spec["normal"]) >= math.cos(sst.settings.face_angle_rad) and \
                    abs(float((c - near) @ spec["normal"])) <= tol
            elif face == "cylinder" and spec.get("axis") is not None:
                a = spec["axis"]
                rel = c - near
                dist = float(np.linalg.norm(rel - (rel @ a) * a))
                radius = spec.get("radius_m")
                ok = abs(float(n @ a)) <= math.sin(sst.settings.face_angle_rad) and \
                    (radius is None or abs(dist - radius) <= max(tol, 0.1 * size))
            else:
                ok = float(np.linalg.norm(c - near)) <= size
            if ok:
                store.pi[patch, PI["feature"]] = k
                store.pf[patch, PF["spec_ra"]] = spec.get("Ra_m") or 0.0
                store.pf[patch, PF["spec_rz"]] = spec.get("Rz_m") or 0.0
                store.pf[patch, PF["spec_rt"]] = spec.get("Rt_m") or 0.0
                return

    # ------------------------------------------------------------------ burrs
    def _exits(self, ctx, state, sst, contact, rc, frame, outline, edge, workpiece) -> None:
        rt = self._rt
        shocks = np.asarray(getattr(contact, "shocks", np.zeros((0, 8))), dtype=float).reshape(-1, 8)
        if len(shocks) == 0 or frame is None:
            return
        kind = shocks[:, 7].astype(int)
        rows = shocks[(kind == SHOCK_EXIT) | (kind == SHOCK_BREAKTHROUGH) | (kind == SHOCK_BREAKOUT)]
        if len(rows) == 0:
            return
        query = rt["query"]
        if query is None:
            sst.counters["exits"] += len(rows)
            sst.counters["exits_interior"] += len(rows)
            return
        mot = KIN.sample(frame, rows[:, 0], ctx.tick_s)
        if not mot:
            return
        model: BU.BurrModel = rt["burr_model"]
        ap = float(rc.geometry.ap_m) if rc is not None else 0.0
        r_arr = np.asarray(edge.elem_r_m, dtype=float)
        z_arr = np.asarray(edge.elem_z_m, dtype=float)
        kap = np.asarray(edge.elem_kappa_rad, dtype=float)
        ekind = np.asarray(edge.elem_kind)
        rad = np.asarray(edge.elem_edge_radius_m, dtype=float)
        worn = rt["wear"].get(int(contact.tool_slot))
        worn_r = np.asarray(getattr(worn, "edge_radius_m", np.zeros((0, 0))), dtype=float) if worn is not None else None
        if worn_r is not None and worn_r.shape == rad.shape:
            rad = np.where(worn_r > 0.0, worn_r, rad)          # wear's rounded hone where it reports one
        R_S_T0 = np.asarray(frame.pose_S_T, dtype=float)[:3, :3]
        angle_end = float(np.asarray(getattr(frame, "spindle_angle_rad", np.zeros(1)))[-1]) \
            if len(np.asarray(getattr(frame, "spindle_angle_rad", np.zeros(0)))) else 0.0
        c, s = math.cos(-angle_end), math.sin(-angle_end)
        R_S_T0 = R_S_T0 @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for i, row in enumerate(rows):
            sst.counters["exits"] += 1
            j = int(np.clip(row[1], 0, r_arr.shape[0] - 1))
            k = int(np.clip(row[2], 0, r_arr.shape[1] - 1))
            h_x = max(float(row[3]), 0.0)
            tip = mot["tip"][i]
            a = mot["axis"][i]
            if outline.rotating:
                alpha = float(row[4])
                p_T0 = np.array([r_arr[j, k] * math.cos(alpha), r_arr[j, k] * math.sin(alpha), z_arr[j, k]])
                X = tip + R_S_T0 @ p_T0
                omega = float(mot["omega"][i])
                v_T0 = math.copysign(1.0, omega if omega != 0.0 else -1.0) * np.array([-math.sin(alpha),
                                                                                       math.cos(alpha), 0.0])
                push = R_S_T0 @ v_T0
                n_e_T0 = np.array([math.sin(kap[j, k]) * math.cos(alpha), math.sin(kap[j, k]) * math.sin(alpha),
                                   -math.cos(kap[j, k])])
                n_gen = -(R_S_T0 @ n_e_T0)
                tip_last = int(ekind[j, k]) in (EDGE_END, EDGE_CORNER, EDGE_LIP, EDGE_CHISEL)
                kappa = float(kap[j, k])
            else:
                rho = math.hypot(tip[0], tip[1])
                theta = math.atan2(tip[1], tip[0])
                X = np.array([rho * math.cos(theta), rho * math.sin(theta), tip[2]])
                vr, vz = float(mot["v_rho"][i]), float(mot["v_z"][i])
                rh = np.array([math.cos(theta), math.sin(theta), 0.0])
                push = vr * rh + vz * np.array([0.0, 0.0, 1.0])
                if np.linalg.norm(push) <= 0.0:
                    continue
                push /= np.linalg.norm(push)
                n_gen = rh if rho > 0.0 else np.array([1.0, 0.0, 0.0])      # outward normal of the turned face
                tip_last = True
                kappa = float(outline.kappa_major_rad)
            edge_r = float(rad[j, k])
            if int(row[7]) == SHOCK_BREAKTHROUGH:
                thrust = max(sst.thrust.get(int(contact.tool_slot), [0.0, 0.0]))
                res = model.drill_burr(thrust)
                start = tip + 1e-5 * a
                hit, t, P, n_x = query.ray_cast(workpiece, start, -a, 0.05, True)
                X = P if hit else tip
                n_x = n_x if hit else -a
                psi_cos = 1.0
            else:
                depth = max(edge_r, 5.0e-6)
                start = X - depth * n_gen
                hit, t, P, n_x = query.ray_cast(workpiece, start, push, max(4.0 * (h_x + edge_r), 1.0e-3), True)
                if not hit:
                    sst.counters["exits_interior"] += 1
                    continue
                psi_cos = float(push @ n_x)
                # turning: the depth of cut lies in the exit face the feed pushes through; rotating tools: the
                # axial depth on a face parallel to the tool axis, the chip thickness on a face normal to it
                b_x = BU.exit_extent(ap, h_x, n_x, a) if outline.rotating else ap
                res = model.exit_burr(h_x, b_x, kappa, psi_cos, tip_last, edge_r)
                X = P
            if res.height_m <= 0.0:
                continue
            self._burr_record(sst, workpiece, X, n_x, res, contact, mot["block"][i], int(row[0]), psi_cos, h_x,
                              tip_last)

    def _burr_record(self, sst, workpiece, X, n_x, res, contact, block, tick, psi_cos, h_x, tip_last) -> None:
        rt = self._rt
        brick = int(np.asarray(rt["bricks"].brick_of(workpiece, X[None, :]))[0])
        facet = _facet_code(n_x)
        i = sst.burrs.row((int(workpiece), brick, facet), sst.settings.max_burr_edges)
        if i < 0:
            sst.burrs.dropped += 1
            return
        bf, bi = sst.burrs.bf, sst.burrs.bi
        bf[i, BF["sx"]:BF["sz"] + 1] += X
        bf[i, BF["nsx"]:BF["nsz"] + 1] += n_x
        bf[i, BF["count"]] += 1.0
        if res.kind == BU.KIND_BREAKOUT:
            bf[i, BF["neg_max"]] = max(bf[i, BF["neg_max"]], res.height_m)
        bf[i, BF["h_max"]] = max(bf[i, BF["h_max"]], res.height_m if res.kind != BU.KIND_BREAKOUT else 0.0)
        bf[i, BF["h_sum"]] += res.height_m
        bf[i, BF["t_max"]] = max(bf[i, BF["t_max"]], res.thickness_m)
        bf[i, BF["hx_sum"]] += h_x
        bf[i, BF["psi_sum"]] += psi_cos
        bf[i, BF["bx_max"]] = max(bf[i, BF["bx_max"]], float(res.factors.get("b_x_m", 0.0)))
        bi[i, BI["kind"]] = BURR_KINDS.index(res.kind) if res.kind in BURR_KINDS else 0
        bi[i, BI["slot"]] = int(contact.tool_slot)
        bi[i, BI["op"]] = sst.op_index
        if bi[i, BI["first_tick"]] < 0:
            bi[i, BI["first_tick"]] = tick
        bi[i, BI["last_tick"]] = max(int(bi[i, BI["last_tick"]]), tick)
        bi[i, BI["block"]] = int(block)
        bi[i, BI["tip_last"]] += 1 if tip_last else 0

    def _composite(self, ctx, state, sst, contact, frame, outline, edge, workpiece) -> None:
        rt = self._rt
        mat = rt["material"]
        if not mat.composite or frame is None or rt["query"] is None or float(contact.in_cut_s) <= 0.0:
            return
        kinds = np.asarray(edge.elem_kind)
        if not (outline.kind in HOLE_TYPES or np.any((kinds == EDGE_LIP) | (kinds == EDGE_CHISEL))):
            return
        mot = KIN.sample(frame, np.array([int(frame.tick[-1])]), ctx.tick_s)
        if not mot:
            return
        tip, a = mot["tip"][0], mot["axis"][0]
        hit, t, P, n_x = rt["query"].ray_cast(workpiece, tip, -a, 0.05, True)
        if not hit:
            return
        thrust = float(sst.thrust.get(int(contact.tool_slot), [0.0, 0.0])[1])
        crit, source = BU.critical_thrust(mat, float(t))
        if crit is None:
            key = "delamination:nodata"
            if key not in sst.findings:
                sst.findings[key] = -1
                sst.estimates.append("composite delamination: the material file has neither composite.delamination."
                                     "G_IC_J_m2 nor critical_thrust_N: the exit is not checked")
            return
        if thrust <= crit:
            return
        hole = tuple(int(round(v * 1e4)) for v in tip[:2])
        key = f"delamination:{int(contact.tool_slot)}:{hole}"
        if key in sst.findings:
            return
        record = {"kind": BU.KIND_DELAMINATION, "slot": int(contact.tool_slot), "op": sst.op_index,
                  "location_S_m": [float(v) for v in P], "uncut_thickness_m": float(t), "thrust_N": thrust,
                  "critical_thrust_N": float(crit), "source": source, "tick": int(frame.tick[-1]),
                  "block": int(mot["block"][0])}
        sst.damage.append(record)
        f = self._finding(ctx, state, Category.BURR, Severity.ERROR,
                          f"push-out delamination at the drill exit: thrust {thrust:.0f} N exceeds the critical "
                          f"{crit:.0f} N of the remaining {t * 1e3:.2f} mm ({source})", int(frame.tick[-1]),
                          int(mot["block"][0]), {"thrust_N": thrust, "critical_thrust_N": float(crit),
                                                 "uncut_thickness_m": float(t)})
        sst.findings[key] = f.id

    # ------------------------------------------------------------------ findings
    def _finding(self, ctx, state, category, severity, message, tick, block, quantities):
        f = ctx.finding(state, DOMAIN, category, severity, message, tick=int(max(tick, 0)), quantities=quantities)
        ref = ctx.timeline.line_ref(int(block)) if (ctx.timeline is not None and block is not None and block >= 0) \
            else None
        if ref is not None:
            f.op = str(getattr(ref, "op", f.op) or f.op)
            f.program = str(getattr(ref, "program", f.program) or f.program)
            f.line = int(getattr(ref, "line", f.line) or f.line)
            f.block = str(getattr(ref, "block", f.block) or f.block)
        return f

    def _values(self, sst, stats) -> None:
        store = sst.store
        bmax = float(np.max(sst.burrs.bf[:sst.burrs.n, BF["h_max"]])) if sst.burrs.n else 0.0
        sst.values.update({"events": float(stats["events"]), "profiled_events": float(stats["profiled"]),
                           "patches": float(store.n_patch), "open_lines": float(len(store.lines)),
                           "burr_max_m": bmax, "marks": float(len(sst.marks) + sst.marks_dropped),
                           "residual_stress_Pa": float(stats["sigma"])})
        for key in ("last_Ra_m", "last_Rz_m", "last_Rt_m"):
            sst.values.setdefault(key, math.nan)

    def channel_values(self, ctx, state, bus=None) -> dict:
        """Values of the surface channels for the recorder (names as in channels.py)."""
        sst: SurfaceState = state.surface
        return {f"surface.{k}": float(v) for k, v in sorted(sst.values.items())}

    # ------------------------------------------------------------------ operation end
    def end_operation(self, ctx, state) -> None:
        """Close every line and pass, evaluate the step-over profiles and witness marks, check which burrs
        survived the operation and raise their findings, measure the operation's threads."""
        sst: SurfaceState = state.surface
        self._attach(ctx, state)
        self._rt["ctx"], self._rt["state"] = ctx, state
        self._flush(ctx, state, sst, sst.op_index)
        sst.op_index += 1

    def _flush(self, ctx, state, sst, op: int) -> None:
        store = sst.store
        g = self._rt["gather"]
        for q in range(store.n_pass):
            if store.qi[q, QI["closed"]] == 0:
                g.close_pass(store, q)
        for p in range(store.n_patch):
            if int(store.pi[p, PI["op"]]) != op:
                continue
            self._cross(ctx, state, sst, p)
        self._burr_survival(ctx, state, sst, op)
        self._spec_findings(ctx, state, sst, op)
        for key in sorted(sst.threads):
            if key[1] != op:
                continue
            th = sst.threads[key]
            meas = TH.measure(th)
            if meas is not None and not any(r.get("thread_key") == list(key) for r in sst.results):
                meas["thread_key"] = list(key)
                sst.results.append(json_safe(meas))

    def _cross(self, ctx, state, sst, p: int) -> None:
        rt = self._rt
        if any(r.get("cross_patch") == p for r in sst.results):
            return
        rough, marks, info = FC.cross_profile(sst.store, p, rt["tools"].outline, sst.settings, rt["rules"],
                                              rt["silhouettes"], recovery=float(rt["rough"]["recovery_ratio"]))
        for m in marks:
            m["patch"] = p
            m["location_S_m"] = [float(v) for v in sst.store.centroid(p)]
            m["normal_S"] = [float(v) for v in sst.store.normal(p)]
            m["op"] = int(sst.store.pi[p, PI["op"]])
            self._add_mark(ctx, state, sst, m)
        if rough is not None:
            self._aggregate(sst, p, "c", rough, rough.evaluation_m, float(info.get("kinematic", {}).get("Rt_m", 0.0)))
            sst.results.append(json_safe({"cross_patch": p, "Ra_m": rough.Ra_m, "Rz_m": rough.Rz_m,
                                          "Rt_m": rough.Rt_m, "cutoff_m": rough.cutoff_m, "method": rough.method,
                                          "passes": info.get("passes"), "track_spacing_m": info.get("track_spacing_m"),
                                          "direction_S": info.get("direction_S")}))

    def _burr_survival(self, ctx, state, sst, op: int) -> None:
        rt = self._rt
        query = rt["query"]
        bf, bi = sst.burrs.bf, sst.burrs.bi
        limit = sst.settings.burr_limit_m
        for i in range(sst.burrs.n):
            if int(bi[i, BI["op"]]) != op or bi[i, BI["removed"]] != 0:
                continue
            X = sst.burrs.location(i)
            n_x = sst.burrs.normal(i)
            if query is not None:
                probe = X - 2.0 * max(sst.settings.plane_tol_m, 1e-5) * n_x
                inside = bool(np.asarray(query.point_inside(int(bi[i, BI["workpiece"]]), probe[None, :]))[0])
                if not inside:
                    bi[i, BI["removed"]] = 1
                    continue
            h = float(bf[i, BF["h_max"]])
            neg = float(bf[i, BF["neg_max"]])
            if limit > 0.0 and (h > limit or neg > limit) and bi[i, BI["finding"]] < 0:
                kind = BURR_KINDS[int(bi[i, BI["kind"]])]
                what = "edge breakout" if neg > h else f"{kind.replace('_', ' ')} burr"
                f = self._finding(ctx, state, Category.BURR, Severity.WARNING,
                                  f"{what} {max(h, neg) * 1e3:.3f} mm at an exit edge exceeds the allowed "
                                  f"{limit * 1e3:.3f} mm", int(bi[i, BI["last_tick"]]), int(bi[i, BI["block"]]),
                                  {"height_m": max(h, neg), "limit_m": limit,
                                   "root_thickness_m": float(bf[i, BF["t_max"]])})
                bi[i, BI["finding"]] = f.id

    def _spec_findings(self, ctx, state, sst, op: int) -> None:
        rt = self._rt
        for spec in rt["specs"]:
            res = self._feature(sst, spec)
            if res is None or res.get("pass") is not False:
                continue
            if res.get("op_index") != op:
                continue
            key = f"spec:{spec['id']}:{op}"
            if key in sst.findings:
                continue
            msg = (f"surface finish of {spec['id']}: Ra {res['Ra_m'] * 1e6:.2f} um, Rz {res['Rz_m'] * 1e6:.2f} um "
                   f"against the specification" + (f" Ra {spec['Ra_m'] * 1e6:.2f} um" if spec.get("Ra_m") else "")
                   + (f" Rz {spec['Rz_m'] * 1e6:.2f} um" if spec.get("Rz_m") else "")
                   + (f" Rt {spec['Rt_m'] * 1e6:.2f} um" if spec.get("Rt_m") else ""))
            f = self._finding(ctx, state, Category.FINISH_OUT_OF_SPEC, Severity.WARNING, msg, res["tick"],
                              res["block"], {"Ra_m": res["Ra_m"], "Rz_m": res["Rz_m"], "Rt_m": res["Rt_m"]})
            sst.findings[key] = f.id

    # ------------------------------------------------------------------ results
    def _feature(self, sst, spec: dict, patches: list | None = None) -> dict | None:
        rt = self._rt
        store = sst.store
        if patches is None:
            patches = FC.match(store, spec, sst.settings, rt["bricks"])
        vals = []
        for p in patches:
            r = FC.patch_roughness(store.pf[p])
            if r is not None:
                vals.append((p, r))
        if not vals:
            return None
        weights = np.array([store.pf[p, PF["count"]] for p, _ in vals])
        ra = float(np.average([r["Ra_m"] for _, r in vals], weights=weights))
        rz = float(np.average([r["Rz_m"] for _, r in vals], weights=weights))
        rt_max = float(max(r["Rt_m"] for _, r in vals))
        verdicts = [FC.judge([r["Ra_m"] for _, r in vals], spec.get("Ra_m"), spec.get("max_rule", False)),
                    FC.judge([r["Rz_m"] for _, r in vals], spec.get("Rz_m"), spec.get("max_rule", False)),
                    FC.judge([r["Rt_m"] for _, r in vals], spec.get("Rt_m"), True)]
        known = [v for v in verdicts if v is not None]
        worst = max(vals, key=lambda pr: pr[1]["Ra_m"])[0]
        lay = self._lay(store, worst)
        ops = sorted({int(store.pi[p, PI["op"]]) for p, _ in vals})
        return {"Ra_m": ra, "Rz_m": rz, "Rt_m": rt_max, "pass": (all(known) if known else None),
                "patches": [int(p) for p, _ in vals], "worst_patch": int(worst), "op_index": ops[-1],
                "tick": int(store.pi[worst, PI["last_tick"]]), "block": int(store.pi[worst, PI["block_last"]]),
                "lay": lay, "slots": sorted({int(store.pi[p, PI["slot"]]) for p, _ in vals})}

    @staticmethod
    def _lay(store, p: int) -> dict:
        """Lay of a patch (ISO 1302 classes in words): turned faces circular (facing) or circumferential;
        milled faces parallel to the marks of the roughest direction, multidirectional when the feed and the
        step-over profiles are within 30 % of each other in Ra."""
        r = FC.patch_roughness(store.pf[p])
        n = store.normal(p)
        d = store.pf[p, PF["dx"]:PF["dz"] + 1].copy()
        code = int(store.pi[p, PI["cls"]])
        cls = CLASSES[code] if 0 <= code < len(CLASSES) else "wall"
        if r is None:
            return {"lay_S": None, "lay": "unknown"}
        if cls == "turn":
            return {"lay_S": None, "lay": "circular" if abs(float(n[2])) > 0.7 else "circumferential"}
        if np.linalg.norm(d) == 0.0:
            return {"lay_S": None, "lay": "unknown"}
        d = d - (d @ n) * n
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.0 else d
        lay = np.cross(n, d) if r["direction"] == "feed" else d
        dirs = r["directions"]
        multi = "f" in dirs and "c" in dirs and \
            min(dirs["f"]["Ra_m"], dirs["c"]["Ra_m"]) > 0.7 * max(dirs["f"]["Ra_m"], dirs["c"]["Ra_m"])
        return {"lay_S": [float(v) for v in lay], "lay": "multidirectional" if multi else "parallel"}

    def finalize(self, ctx, state) -> dict:
        """The surface.json document (docs/models/surface.md "Results"); written to the run directory with the
        per-brick attribute arrays when the run has one."""
        sst: SurfaceState = state.surface
        self._attach(ctx, state)
        self._rt["ctx"], self._rt["state"] = ctx, state
        for op in sorted({int(sst.store.pi[p, PI["op"]]) for p in range(sst.store.n_patch)} | {sst.op_index}):
            self._flush(ctx, state, sst, op)
        doc = self.document(ctx, state)
        if ctx.run_dir is not None:
            from pathlib import Path

            from ..core.rundir import write_json_atomic
            run_dir = Path(ctx.run_dir)
            write_json_atomic(run_dir / "surface.json", doc)
            written = self._write_attributes(ctx, state, run_dir)
            doc["attributes"] = written
        return doc

    def document(self, ctx, state) -> dict:
        sst: SurfaceState = state.surface
        rt = self._rt
        store = sst.store
        used = set()
        features = []
        for spec in rt["specs"]:
            res = self._feature(sst, spec)
            entry = {"id": spec["id"], "spec_Ra_m": spec.get("Ra_m"), "spec_Rz_m": spec.get("Rz_m"),
                     "spec_Rt_m": spec.get("Rt_m"), "face": spec.get("face")}
            if res is None:
                entry.update({"Ra_m": None, "Rz_m": None, "pass": None, "note": "no machined surface matched"})
            else:
                used.update(res["patches"])
                entry.update(self._feature_entry(ctx, sst, res))
            features.append(entry)
        faces = []
        rest = [p for p in range(store.n_patch) if p not in used and FC.patch_roughness(store.pf[p]) is not None]
        for k, group in enumerate(FC.group_faces(store, rest, sst.settings, rt["bricks"])):
            res = self._feature(sst, {"id": f"face-{k}"}, group)
            if res is None:
                continue
            entry = {"id": f"face-{k}", "spec_Ra_m": None, "spec_Rz_m": None, "spec_Rt_m": None,
                     "normal_S": [float(v) for v in store.normal(group[0])],
                     "centroid_S_m": [float(v) for v in np.mean([store.centroid(p) for p in group], axis=0)]}
            entry.update(self._feature_entry(ctx, sst, res))
            faces.append(entry)
        burrs = self._burr_entries(ctx, sst)
        threads = [self._thread_entry(ctx, r) for r in sst.results if "thread_key" in r]
        doc = {"format": "cnctwin-surface", "version": FORMAT_VERSION, "features": features, "faces": faces,
               "burrs": burrs, "marks": [self._mark_entry(ctx, m) for m in sst.marks], "damage": [
                   self._with_line(ctx, dict(d)) for d in sst.damage], "threads": threads,
               "patches": self._patch_summary(sst), "summary": {**sst.counters, "patches": store.n_patch,
                                                                "passes": store.n_pass,
                                                                "marks_dropped": sst.marks_dropped,
                                                                "burr_edges_dropped": sst.burrs.dropped},
               "estimates": list(dict.fromkeys(sst.estimates)),
               "models": {"roughness": "kinematic profile (lower envelope of the generating passes), ISO 4287/4288",
                          "burr": "roll-over/breakout exit burr scaling, drilling plastic-collapse burr, Hocheng-"
                                  "Dharan delamination", "integrity": "line-contact shakedown + thermal relief"}}
        return json_safe(doc)

    def _feature_entry(self, ctx, sst, res) -> dict:
        store = sst.store
        blocks = sorted({int(store.pi[p, PI[k]]) for p in res["patches"] for k in ("block_first", "block_last")
                         if int(store.pi[p, PI[k]]) >= 0})
        lines = sorted({int(getattr(ctx.timeline.line_ref(b), "line", 0)) for b in blocks
                        if ctx.timeline is not None and ctx.timeline.line_ref(b) is not None})
        ops = sorted({str(getattr(ctx.timeline.line_ref(b), "op", "")) for b in blocks
                      if ctx.timeline is not None and ctx.timeline.line_ref(b) is not None})
        numbers = self._tool_numbers(ctx, res["slots"])
        worst = res["worst_patch"]
        pr = FC.patch_roughness(store.pf[worst])
        pf = store.pf
        integ = {"residual_stress_Pa": float(np.mean([pf[p, PF["sigma"]] for p in res["patches"]])),
                 "affected_depth_m": float(np.max([pf[p, PF["sigma_depth"]] for p in res["patches"]])),
                 "hardness_change": float(np.mean([pf[p, PF["hard_change"]] for p in res["patches"]])),
                 "hardened_depth_m": float(np.max([pf[p, PF["hard_depth"]] for p in res["patches"]])),
                 "surface_temp_max_K": float(np.max([pf[p, PF["T_peak"]] for p in res["patches"]])),
                 "white_layer": bool(any(pf[p, PF["white"]] > 0 for p in res["patches"])),
                 "burn": bool(any(pf[p, PF["burn"]] > 0 for p in res["patches"])),
                 "overtemper": bool(any(pf[p, PF["overtemper"]] > 0 for p in res["patches"])),
                 "melt_smear": bool(any(pf[p, PF["melt"]] > 0 for p in res["patches"]))}
        flags = {"chatter_marks": bool(any(pf[p, PF["chatter"]] > 0 for p in res["patches"])),
                 "bue": bool(any(pf[p, PF["bue"]] > 0 for p in res["patches"])),
                 "recut_marks": int(sum(pf[p, PF["recut"]] for p in res["patches"])),
                 "recut_dent_max_m": float(max(pf[p, PF["recut_max"]] for p in res["patches"])),
                 "marks": int(sum(pf[p, PF["marks"]] for p in res["patches"])),
                 "mark_max_m": float(max(pf[p, PF["mark_max"]] for p in res["patches"]))}
        return {"op": ops[-1] if ops else "", "tool": numbers[0] if len(numbers) == 1 else (numbers or None),
                "lines": lines, "Ra_m": res["Ra_m"], "Rz_m": res["Rz_m"], "Rt_m": res["Rt_m"], "pass": res["pass"],
                "patches": len(res["patches"]), "lay": res["lay"]["lay"], "lay_S": res["lay"]["lay_S"],
                "direction": pr["direction"] if pr else None, "cutoff_m": pr["cutoff_m"] if pr else None,
                "short_evaluation": bool(pr and pr["short"]), "kinematic_Pt_m": float(pf[worst, PF["f_pt_max"]]),
                "integrity": integ, **flags}

    def _tool_numbers(self, ctx, slots) -> list:
        tools = getattr(self._rt["state"], "tools", None)
        out = []
        for s in slots:
            num = None
            for ts in getattr(tools, "slots", ()) or ():
                if int(ts.slot) == int(s):
                    num = int(ts.number)
            out.append(num if num is not None else int(s))
        return out

    def _with_line(self, ctx, entry: dict) -> dict:
        block = int(entry.get("block", -1))
        ref = ctx.timeline.line_ref(block) if (ctx.timeline is not None and block >= 0) else None
        entry["op"] = str(getattr(ref, "op", "")) if ref is not None else entry.get("op")
        entry["program"] = str(getattr(ref, "program", "")) if ref is not None else None
        entry["line"] = int(getattr(ref, "line", 0)) if ref is not None else None
        return entry

    def _mark_entry(self, ctx, m: dict) -> dict:
        return self._with_line(ctx, dict(m))

    def _burr_entries(self, ctx, sst) -> list:
        out = []
        bf, bi = sst.burrs.bf, sst.burrs.bi
        for i in range(sst.burrs.n):
            kind = BURR_KINDS[int(bi[i, BI["kind"]])]
            count = max(bf[i, BF["count"]], 1.0)
            neg = float(bf[i, BF["neg_max"]])
            entry = {"edge": f"w{int(bi[i, BI['workpiece']])}.b{int(bi[i, BI['brick']])}."
                             f"{_facet_name(sst.burrs.normal(i))}",
                     "kind": "breakout" if neg > bf[i, BF["h_max"]] else kind,
                     "height_m": float(bf[i, BF["h_max"]]), "breakout_depth_m": neg,
                     "mean_height_m": float(bf[i, BF["h_sum"]] / count),
                     "root_thickness_m": float(bf[i, BF["t_max"]]),
                     "location_S_m": [float(v) for v in sst.burrs.location(i)],
                     "exit_normal_S": [float(v) for v in sst.burrs.normal(i)],
                     "exit_order": "tip_last" if bi[i, BI["tip_last"]] * 2 >= count else "top_last",
                     "exit_angle_cos": float(bf[i, BF["psi_sum"]] / count), "exits": int(count),
                     "removed": bool(bi[i, BI["removed"]]), "tool": self._tool_numbers(ctx, [int(bi[i, BI["slot"]])])[0],
                     "block": int(bi[i, BI["block"]]), "finding": int(bi[i, BI["finding"]])}
            out.append(self._with_line(ctx, entry))
        return out

    def _thread_entry(self, ctx, r: dict) -> dict:
        entry = dict(r)
        entry["id"] = f"thread-T{self._tool_numbers(ctx, [int(r['slot'])])[0]}-op{int(r['op'])}"
        entry["block"] = int(r.get("block_last", -1))
        return self._with_line(ctx, entry)

    def _patch_summary(self, sst) -> list:
        out = []
        store = sst.store
        for p in range(store.n_patch):
            r = FC.patch_roughness(store.pf[p])
            out.append({"patch": p, "workpiece": int(store.pi[p, PI["workpiece"]]),
                        "brick": int(store.pi[p, PI["brick"]]), "class": CLASSES[int(store.pi[p, PI["cls"]])],
                        "normal_S": [float(v) for v in store.normal(p)],
                        "centroid_S_m": [float(v) for v in store.centroid(p)], "events": int(store.pf[p, PF["count"]]),
                        "Ra_m": r["Ra_m"] if r else None, "Rz_m": r["Rz_m"] if r else None,
                        "Rt_m": r["Rt_m"] if r else None, "direction": r["direction"] if r else None,
                        "residual_stress_Pa": float(store.pf[p, PF["sigma"]])})
        return out

    def _write_attributes(self, ctx, state, run_dir) -> list:
        rt = self._rt
        sst: SurfaceState = state.surface
        meshes_src = self.meshes
        if meshes_src is None and (run_dir / "keyframes" / "bricks.idx").is_file():
            meshes_src = RunDirBricks(run_dir)
        if meshes_src is None:
            sst.estimates.append("per-brick attributes: no final brick meshes (no keyframes, none wired): not written")
            return []
        meshes = meshes_src.final_bricks()
        live = [i for i in range(sst.burrs.n) if sst.burrs.bi[i, BI["removed"]] == 0]
        loc = np.array([sst.burrs.location(i) for i in live]).reshape(-1, 3)
        hh = np.array([sst.burrs.bf[i, BF["h_max"]] for i in live])
        attrs = FC.brick_attributes(sst.store, loc, hh, meshes, rt["bricks"], parse_brick_key, sst.settings)
        from ..core.rundir import write_npz
        folder = run_dir / "keyframes" / "attr"
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        for key in sorted(attrs, key=lambda k: parse_brick_key(k)):
            path = folder / f"brick_{key}.npz"
            merged = {}
            if path.is_file():
                with np.load(path, allow_pickle=False) as old:
                    merged = {name: np.asarray(old[name]) for name in old.files}
            merged.update(attrs[key])
            write_npz(path, merged)
            written.append(f"keyframes/attr/brick_{key}.npz")
        return written

    # ------------------------------------------------------------------ queries for other parts
    def view(self, state) -> "SurfaceView":
        return SurfaceView(lambda: state.surface)


class SurfaceView:
    """What other parts may ask the surface domain (the integration hands it to them): burr heights at a
    face (holding: burrs under locators and clamps of a later setup) and the finish of a patch."""

    def __init__(self, state_getter):
        self._state = state_getter

    def burr_height_at(self, workpiece: int, point_S_m, normal=None, radius_m: float = 1.0e-3,
                       since_tick: int | None = None) -> float:
        """Highest surviving burr within radius of the point whose exit face matches the normal (within 45 deg
        when given): the overhang a locator or clamp on that face sits on. ``since_tick``: only burrs formed at or
        after that base tick (the part in the machine: a batch's earlier parts have been unloaded with theirs)."""
        sst: SurfaceState = self._state()
        p = np.asarray(point_S_m, dtype=float)
        best = 0.0
        for i in range(sst.burrs.n):
            if sst.burrs.bi[i, BI["removed"]] or int(sst.burrs.bi[i, BI["workpiece"]]) != int(workpiece):
                continue
            if since_tick is not None and int(sst.burrs.bi[i, BI["first_tick"]]) < int(since_tick):
                continue
            if np.linalg.norm(sst.burrs.location(i) - p) > radius_m:
                continue
            if normal is not None and float(sst.burrs.normal(i) @ np.asarray(normal, dtype=float)) < math.cos(math.pi / 4):
                continue
            best = max(best, float(sst.burrs.bf[i, BF["h_max"]]))
        return best

    def finish_at(self, workpiece: int, point_S_m, radius_m: float = 5.0e-3) -> dict | None:
        """Finish of the patch nearest to a point (None when no machined patch lies within radius)."""
        sst: SurfaceState = self._state()
        store = sst.store
        p = np.asarray(point_S_m, dtype=float)
        best, dist = None, radius_m
        for i in range(store.n_patch):
            if int(store.pi[i, PI["workpiece"]]) != int(workpiece):
                continue
            d = float(np.linalg.norm(store.centroid(i) - p))
            if d <= dist:
                best, dist = i, d
        return FC.patch_roughness(store.pf[best]) if best is not None else None


def _facet_code(n: np.ndarray) -> int:
    """Exit-face class of a normal: the dominant signed axis (0..5) and an octant refinement."""
    n = np.asarray(n, dtype=float)
    k = int(np.argmax(np.abs(n)))
    sign = 0 if n[k] >= 0.0 else 1
    return 2 * k + sign


def _facet_name(n: np.ndarray) -> str:
    code = _facet_code(n)
    return ("+x", "-x", "+y", "-y", "+z", "-z")[code]


__all__ = ["SurfaceDomain", "SurfaceView"]
