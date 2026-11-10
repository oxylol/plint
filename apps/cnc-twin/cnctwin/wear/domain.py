"""WearDomain: tool wear, damage, life and breakage (docs/models/wear.md; contract: core/domain.py).

An L2 slow domain (no part in the hot loop). Per macro step it reads ``prev`` - cutting's per-element
integrals, entry/exit shocks, section moments and stresses, torque peaks and surface events; thermal's
edge and flank temperatures at the in-cut quadrature points, cycle amplitudes and body temperatures;
coolant's friction factor (BUE); chips' packing torque and force; motion's paths (active tool, spindle
speed, feed, pose) - and writes only ``nxt.wear``: per tool slot a :class:`cnctwin.core.bus.ToolWear`
(VB, crater, notch, worn hone radius, radial/axial wear, coating, chips, BUE, thermal damage, cracks,
section fatigue, broken, life used, remaining life) and the expired and broken tool numbers.

Geometry and the calibration response come through protocols (protocols.py, calibration.py): the
integration wires ``WearDomain(tools=GeometryToolInfo(state.tooling.tools), response=..., skin=...,
tolerances=...)``; without them the domain builds its own defaults (documented in the model doc).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.bus import WearFeedback
from ..core.errors import InputError
from ..core.findings import Category, Severity
from ..mechanics.materials import ToolSide, WorkMaterial, normalise
from ..units import REFERENCE_TEMP_K
from . import abi as A
from . import build as B
from . import kernels as K
from . import life as L
from . import params as P
from . import step as S
from .channels import MAX_SLOTS, MAX_TEETH
from .channels import channels as wear_channels
from .protocols import GeometryToolInfo, ProjectTolerances, StockShell
from .schema import settings as read_settings
from .state import WearState

TOOLS_FORMAT = "cnctwin-tools"         # tools.json (docs/formats.md "Results of a run"); older files have no header
TOOLS_VERSION = 1


class WearDomain:
    """Tool wear, damage, life and breakage."""

    name = "wear"

    def __init__(self, tools=None, response=None, skin=None, tolerances=None, catalog=None):
        self.tools = tools
        self.response = response
        self.skin = skin
        self.tolerances = tolerances
        self.catalog = catalog
        self._rt: dict = {}
        self._info = None

    # ------------------------------------------------------------------ Domain interface
    def channels(self):
        return wear_channels()

    def kernel_inputs(self, ctx, state, bus):
        return None

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        return None

    def init(self, ctx, state) -> None:
        project = ctx.project
        physics = P.field_of(project, "physics")
        st = read_settings(P.field_of(physics, "wear"))
        ws = WearState(st, int(ctx.seed))
        material = P.field_of(project, "material")
        work = P.work_data(material)
        ws.estimated.extend(work.notes)
        work_mech = self._work_mech(material, ws)
        ambient = self._ambient(project)
        info = self._tool_info(ctx, state)
        self._info = info
        entries = list(P.field_of(P.field_of(project, "tools"), "list") or [])
        catalog = self.catalog if self.catalog is not None else ctx.db
        from ..tooling.spec import read_instance, read_tool
        for slot in state.tools.slots:
            if slot.slot >= len(entries):
                raise InputError(f"wear: tool slot {slot.slot} (T{slot.number}) has no entry in the project tools list")
            entry = entries[slot.slot]
            instance = read_instance(entry)
            tool_item = P.field_of(entry, "tool")
            spec = read_tool(tool_item, slot.tool_id)
            tm_obj = self._lookup(catalog, "tool_material", spec.tool_material_id, slot)
            tool_mat = P.tool_material_data(tm_obj, spec.tool_material_class)
            coating_id = spec.coating_id or "none"
            coat_obj = None
            if normalise(coating_id) not in ("none", "uncoated", ""):
                coat_obj = self._lookup(catalog, "coating", coating_id, slot)
            coating = P.coating_data(coat_obj, spec.coating_thickness_m)
            response = self.response
            if response is None:
                try:
                    side = ToolSide.from_db(tool_item, tm_obj, coat_obj)
                    substrate = ToolSide.from_db(None, tm_obj, None)
                except (TypeError, ValueError, KeyError) as error:
                    side = substrate = None
                    ws.estimated.append(f"tool slot {slot.slot}: no tool-side heat data ({error}): adiabatic tool in "
                                        "the default response")
                response = B.default_response(work_mech, side, ambient, substrate)
            sw = B.build_slot(slot=slot, tool_spec=spec, instance=instance, tool_mat=tool_mat, coating=coating,
                              coating_id=coating_id, work=work, work_mech=work_mech, info=info, response=response,
                              settings=st, seed=ws.seed, estimated=ws.estimated)
            sw.props["dimensional_share"] = st.dimensional_share
            sw.props["ambient_K"] = ambient
            sw.props["interface_max_K"] = self._interface_max_K(work_mech)
            ws.slots[slot.slot] = sw
            law = sw.info["law"]
            quantities = {"slot": slot.slot, "A_per_Pa": float(sw.law.fl_A[0]), "B_K": float(sw.law.fl_B[0]),
                          "abrasive_coeff": float(sw.law.scal[A.LS_ABR])}
            calib = law.get("flank", {}).get("calibration")
            if calib:
                for i, r in enumerate(calib["residual_rel"]):
                    if r is not None:
                        quantities[f"residual_{i}"] = float(r)
            event = ctx.trace.record(0.0, 0, "wear", f"wear:law:{law.get('flank', {}).get('source', '')}", quantities, [])
            ws.trace_notes[f"law:{slot.slot}"] = int(event)
        ws.part_index = state.clock.part_index
        ws.op_index = state.clock.op_index
        state.wear = ws
        self._rt = {}
        self._attach(ctx, state)

    def macro_step(self, ctx, state, prev, nxt) -> None:
        ws: WearState = state.wear
        self._attach(ctx, state)
        dt = ctx.macro_s
        if state.clock.part_index != ws.part_index:
            self._close_part(ws)
            ws.part_index = state.clock.part_index
        slot_frames = {}
        for frame in prev.motion.paths:
            s = int(frame.tool_slot)
            if s >= 0:
                slot_frames.setdefault(s, frame)
            if s >= 0 and ws.path_slot.get(int(frame.path)) != s and s in ws.slots:
                ws.slots[s].counters["uses"] += 1
            ws.path_slot[int(frame.path)] = s
        feedback = WearFeedback()
        stresses = {}
        for slot in sorted(ws.slots):
            record, stress = self._step_slot(ctx, state, ws, slot, prev, slot_frames.get(slot), dt)
            feedback.tools[slot] = record
            stresses[slot] = stress
            sw = ws.slots[slot]
            if sw.life["expired"]:
                feedback.expired.append(int(sw.number))
            if sw.life["broken"]:
                feedback.broken.append(int(sw.number))
        feedback.expired = sorted(set(feedback.expired))
        feedback.broken = sorted(set(feedback.broken))
        ws.time_s = float(prev.motion.tick) * ctx.tick_s
        self._values(ws, prev, feedback, stresses)
        nxt.wear = feedback

    def end_operation(self, ctx, state) -> None:
        """Wear, damage, counters and the life record carry over to the next operation (the same tool
        instances continue); per-operation trace caps are reset."""
        ws: WearState = state.wear
        ws.op_index += 1
        for sw in ws.slots.values():
            sw.counters["op_events"] = 0

    def finalize(self, ctx, state) -> dict:
        """The tools.json document (docs/models/wear.md "Results"): ``{format: cnctwin-tools, version: 1, tools,
        estimated}``."""
        ws: WearState = state.wear
        self._attach(ctx, state)
        self._close_part(ws)
        tools = []
        for slot, sw in sorted(ws.slots.items()):
            rt = self._rt[slot]
            out = rt.out if rt.out is not None else self._outputs(sw)
            valid = sw.geo.valid != 0
            vb_tooth = [float(np.max(out.VB[j][valid[j]])) if valid[j].any() else 0.0 for j in range(valid.shape[0])]
            life = sw.life
            used_coat = (sw.elem.used != 0) & valid
            record = {
                "slot": slot, "number": sw.number, "tool_id": sw.tool_id, "holder_id": sw.holder_id,
                "tool_material": sw.props["tool_material"], "coating": sw.props["coating"],
                "VB_m": vb_tooth, "VB_max_m": max(vb_tooth) if vb_tooth else 0.0,
                "VB_limit_m": float(sw.props["limits"]["vb"]),
                "KT_max_m": float(np.max(sw.elem.kt[valid])) if valid.any() else 0.0,
                "notch_max_m": float(np.max(out.VB_N[valid])) if valid.any() else 0.0,
                "edge_radius_max_m": float(np.max(out.edge_r[valid])) if valid.any() else 0.0,
                "coating_min_m": float(np.min(sw.elem.coat_f[used_coat])) if used_coat.any() else None,
                "chipped_elements": int(np.count_nonzero(sw.elem.chipped)), "chip_events": sw.counters["chip_events"],
                "coating_spalls": sw.counters["spalls"], "bue_breakoffs": sw.counters["bue_breakoffs"],
                "thermal_damage_max": float(np.max(sw.elem.D_th)) if sw.elem.D_th.size else 0.0,
                "comb_crack_depth_max_m": float(np.max(sw.elem.crack_m)) if sw.elem.crack_m.size else 0.0,
                "section_fatigue": {n: float(v) for n, v in zip(sw.section_names, sw.sections["fatigue"])},
                "broken": bool(life["broken"]), "broken_tick": int(life["broken_tick"]),
                "broken_cause": life["broken_cause"], "broken_section": life["broken_section"],
                "expired": bool(life["expired"]), "life_cause": life["cause"], "expired_tick": int(life["expired_tick"]),
                "life_used_fraction": float(life["used_fraction"]),
                "remaining_life_s": life["remaining_s"] if math.isfinite(life["remaining_s"]) else None,
                "replacement_time_s": life["replacement_s"] if math.isfinite(life["replacement_s"]) else None,
                "remaining_cause": life["remaining_cause"],
                "cut_time_s": float(sw.counters["in_cut_s"] - sw.counters["carried_in_s"]),
                "in_cut_total_s": float(sw.counters["in_cut_s"]),
                "cutting_distance_m": float(sw.counters["cutting_distance_m"]),
                "cut_path_m": float(sw.counters["cut_path_m"]), "uses": int(sw.counters["uses"]),
                "parts": int(sw.counters["parts"]),
                "max_edge_temp_K": max(sw.counters["max_flank_K"], sw.counters["max_rake_K"]) or None,
                "max_section_stress_Pa": sw.counters["max_section_stress_Pa"],
                "above_service_temperature_s": sw.counters["above_service_s"],
                "max_failure_probability": sw.counters["max_failure_probability"],
                "Ra_m": life.get("Ra_m"), "dimensional_drift_m": life.get("drift_m"),
                "criteria": {"values": life["values"], "limits": life["limits"]},
                "wear_offsets_m": self._offsets(sw, out), "law": sw.info["law"],
                "per_part": list(sw.per_part),
            }
            if sw.per_part:
                last = sw.per_part[-1]
                record["life_used_part_fraction"] = last["life_used_part_fraction"]
            tools.append(record)
        return {"format": TOOLS_FORMAT, "version": TOOLS_VERSION, "tools": tools, "estimated": list(ws.estimated)}

    # ------------------------------------------------------------------ results for the recorder
    def macro_values(self, state) -> dict:
        return dict(state.wear.values)

    # ------------------------------------------------------------------ setup helpers
    @staticmethod
    def _lookup(catalog, kind: str, ident, slot):
        if ident is None:
            raise InputError(f"wear: tool slot {slot.slot} (T{slot.number}) names no {kind} id")
        if catalog is None:
            raise InputError(f"wear: no database to look up {kind} {ident!r} (tool slot {slot.slot})")
        return getattr(catalog, kind)(ident)

    @staticmethod
    def _ambient(project) -> float:
        env = P.field_of(project, "environment")
        value = P.num(env, "ambient_K") if env is not None else None
        if value is None and env is not None:
            raw = P.num(env, "ambient_degC")
            value = None if raw is None else raw + 273.15
        return float(value) if value is not None else REFERENCE_TEMP_K

    @staticmethod
    def _interface_max_K(work_mech) -> float | None:
        """The highest tool-work interface temperature: the work material's solidus (else liquidus, else its flow
        law's melting temperature); None without one (data-only materials keep thermal's temperatures)."""
        if work_mech is None:
            return None
        for value in (getattr(work_mech, "solidus_K", None), getattr(work_mech, "liquidus_K", None)):
            if value is not None and float(value) > 0.0:
                return float(value)
        jc = getattr(work_mech, "jc", None)
        if jc is not None:
            from ..mechanics.abi import MAT_TMELT_K
            if float(jc[MAT_TMELT_K]) > 0.0:
                return float(jc[MAT_TMELT_K])
        return None

    @staticmethod
    def _work_mech(material, ws):
        try:
            return WorkMaterial.from_db(P.as_map(material))
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            ws.estimated.append(f"work material: no mechanics view ({error}): no BUE window and no default response")
            return None

    def _tool_info(self, ctx, state):
        if self.tools is not None:
            return self.tools
        tooling = getattr(state, "tooling", None)
        geos = getattr(tooling, "tools", None) if tooling is not None else None
        if geos:
            return GeometryToolInfo(geos)
        from ..tooling.build import build_tool
        from ..tooling.envelope import spindle_nose_from_machine
        from ..tooling.schema import settings_from
        project = ctx.project
        entries = list(P.field_of(P.field_of(project, "tools"), "list") or [])
        machine = P.field_of(project, "machine")
        tsettings = settings_from(P.field_of(project, "physics"))
        nose = spindle_nose_from_machine(machine)
        built = []
        for slot in state.tools.slots:
            entry = entries[slot.slot]
            built.append(build_tool(P.field_of(entry, "tool"), P.field_of(entry, "holder"), entry, tsettings,
                                    slot=slot.slot, tool_id=slot.tool_id, holder_id=slot.holder_id,
                                    db=getattr(ctx, "db", None), spindle_nose=nose))
        return GeometryToolInfo(built)

    def _attach(self, ctx, state) -> None:
        """Runtime objects of every slot (after init and after a resume)."""
        ws: WearState = state.wear
        if all(slot in self._rt for slot in ws.slots):
            return
        info = self._info if self._info is not None else self._tool_info(ctx, state)
        self._info = info
        project = ctx.project
        material = P.field_of(project, "material")
        work = P.work_data(material) if material is not None else None
        work_mech = self._work_mech(material, ws) if material is not None else None
        skin = self.skin
        skin_active = work is not None and work.skin_thickness_m > 0.0
        if skin_active and skin is None:
            stock = P.field_of(project, "stock")
            stock = P.as_map(stock) if stock is not None else None
            kind = P.text(stock, "kind", "") if stock is not None else ""
            size = P.plain(P.field_of(stock, "size_m")) if stock is not None else None
            envelope = _mesh_stock_envelope(project, stock) if kind == "mesh" else None
            if kind in ("block", "cylinder", "tube") and size is not None:
                skin = StockShell(kind, [float(P.plain(v)) for v in size], work.skin_thickness_m,
                                  ws.settings.skin_samples)
            elif envelope is not None:
                # a bar or plate given as a mesh (the lathe-shaft example's d45 bar with its centre hole): its
                # as-rolled skin is on the envelope (outer diameter and ends, or the block's faces)
                skin = StockShell(envelope[0], envelope[1], work.skin_thickness_m, ws.settings.skin_samples)
                ws.estimated.append(f"stock skin ({work.skin_type}) of the mesh stock on its {envelope[0]} envelope "
                                    f"({', '.join(f'{v * 1e3:.2f}' for v in envelope[1])} mm)")
            else:
                skin_active = False
                self._note_once(ctx, ws, "skin", "wear:skin_not_evaluated", {"skin_thickness_m": work.skin_thickness_m},
                                0, [])
                ws.estimated.append(f"stock skin ({work.skin_type}) not evaluated: no skin query for {kind or 'this'} "
                                    "stock")
        tolerances = self.tolerances if self.tolerances is not None else ProjectTolerances(project)
        ratio = None
        if work_mech is not None and work_mech.min_chip_thickness_ratio is not None:
            ratio = float(work_mech.min_chip_thickness_ratio)
        if ratio is None:
            from ..mechanics import params as MP
            ratio = MP.number(MP.CONTACT, "default_min_chip_thickness_ratio")
        for slot, sw in sorted(ws.slots.items()):
            if slot in self._rt:
                continue
            edge = info.edge_model(slot)
            shape = np.asarray(edge.elem_valid).shape
            if shape != sw.geo.valid.shape:
                raise InputError(f"wear: tool slot {slot}: the edge model has shape {shape}, the wear state "
                                 f"{sw.geo.valid.shape}")
            nt, ne = shape
            valid = np.asarray(edge.elem_valid) != 0
            z = np.asarray(edge.elem_z_m, dtype=float)
            lengths = np.asarray(edge.elem_len_m, dtype=float)
            z_load = float(np.sum(z[valid] * lengths[valid]) / np.sum(lengths[valid])) \
                if valid.any() and np.sum(lengths[valid]) > 0.0 else 0.0
            rt = S.Runtime(edge=edge, keys=np.ascontiguousarray(A.element_keys(slot, nt, ne).reshape(-1)),
                           work_mech=work_mech, skin=skin, skin_active=skin_active, tolerances=tolerances,
                           h_min_ratio=ratio, z_load_m=z_load)
            rt.out = self._outputs(sw)
            self._rt[slot] = rt
            assigned = sw.props.get("finish_assignment")
            if assigned is not None and hasattr(tolerances, "assign"):
                tolerances.assign(slot, *assigned)

    @staticmethod
    def _finishing(ws, rt, sw, g, nt, ne) -> None:
        """Mark the elements whose surface events generate a toleranced feature's final surface and assign the slot
        the tightest tolerance and finish of those features (persisted in the slot's props). Before 2026-11-10 every
        element that wrote a surface event finished and every tool took the part's tightest tolerance and finish: a
        roughing end mill leaving 0.2 mm on the floor of examples/pocket-islands-al6061 was held to the floor's
        Ra 1.6 um and the islands' 0.02 mm flatness and reached its 'end of life' 0.32 s into its first cut."""
        rows = g.surface_rows
        if rows.shape[0] == 0:
            return
        tol = rt.tolerances
        if hasattr(tol, "finishing"):
            mask, allowed, ra, rt_ = tol.finishing(rows[:, 3:6], rows[:, 6:9], ws.settings.final_surface_band_m)
            if mask.any() and hasattr(tol, "assign"):
                old = sw.props.get("finish_assignment") or [None, None, None]
                new = [b if a is None else (a if b is None else min(a, b)) for a, b in zip(old, (allowed, ra, rt_))]
                sw.props["finish_assignment"] = new
                tol.assign(sw.slot, *new)
        else:
            mask = np.ones(rows.shape[0], bool)
        for j, k in rows[mask, 1:3].astype(np.int64):
            if 0 <= j < nt and 0 <= k < ne:
                sw.finishes[j, k] = 1

    @staticmethod
    def _outputs(sw) -> A.ElemOut:
        nt, ne = sw.geo.valid.shape
        out = A.new_output(nt, ne)
        K.element_outputs(sw.geo, sw.law, sw.elem, out)
        return out

    # ------------------------------------------------------------------ the step of one slot
    def _step_slot(self, ctx, state, ws, slot, prev, frame, dt):
        sw = ws.slots[slot]
        rt = self._rt[slot]
        nt, ne = sw.geo.valid.shape
        g = S.gather(prev, slot, (nt, ne), sw.sections["z_m"].shape[0])
        self._finishing(ws, rt, sw, g, nt, ne)
        tick0, tick1 = self._ticks(prev, frame)
        window = ws.settings.rate_window_s
        if frame is not None:
            sw.counters["active_s"] += dt
        rt.cut = False
        if sw.life["broken"] or not g.cutting:
            if frame is not None:
                sw.life["duty"] = L.ewma(sw.life["duty"], 0.0, dt, window)
            if rt.last is None:
                rt.summary = S.summary(sw, rt.out)
                rt.last = S.publish(sw, rt.out, rt.summary)
            return rt.last, 0.0
        heat = prev.thermal.tools.get(slot) if prev.thermal is not None else None
        notes: list = []
        pose = np.asarray(frame.pose_S_T, dtype=float) if frame is not None else None
        inp = S.kernel_input(sw, rt, g, heat, float(sw.props["ambient_K"]), dt, pose, notes)
        g.friction_factor = self._friction(prev, g)
        broke = S.bue_step(sw, rt, g, inp, ws.seed)
        out = rt.out
        K.advance_elements(sw.geo, inp, sw.law, sw.elem, out)
        omega = float(np.mean(np.abs(np.asarray(frame.spindle_omega_rad_s, dtype=float)))) \
            if frame is not None and np.size(frame.spindle_omega_rad_s) else 0.0
        period = 2.0 * math.pi / omega if sw.rotating and omega > 0.0 else 0.0
        chips = S.chip_step(sw, inp, g, ws.seed, now_s=tick1 * ctx.tick_s, period_s=period)
        K.element_outputs(sw.geo, sw.law, sw.elem, out)
        rt.summary = S.summary(sw, out)
        self._notes(ctx, ws, sw, notes, tick1)
        self._events(ctx, ws, sw, inp, out, chips, g, broke, tick1)
        before_hot = sw.counters["above_service_s"]
        self._counters(ws, sw, inp, g, frame)
        if sw.counters["above_service_s"] > before_hot:
            self._note_once(ctx, ws, f"service:{slot}", "wear:edge_above_tool_service_temperature",
                            {"slot": slot, "max_service_K": float(sw.props["max_service_K"]),
                             "edge_temp_max_K": float(max(np.max(inp.flank_T), np.max(inp.rake_T)))}, tick1, [])
        stress = self._breakage(ctx, state, ws, sw, rt, g, prev, frame, tick0, tick1)
        self._life(ctx, state, ws, sw, rt, out, g, frame, dt, tick0, tick1, float(prev.motion.tick) * ctx.tick_s)
        rt.last = S.publish(sw, out, rt.summary)
        rt.cut = True
        return rt.last, stress

    @staticmethod
    def _ticks(prev, frame) -> tuple[int, int]:
        if frame is not None:
            ticks = np.asarray(frame.tick)
            if ticks.size:
                return int(ticks[0]), int(ticks[-1])
        return int(prev.motion.tick), int(prev.motion.tick)

    @staticmethod
    def _friction(prev, g) -> float:
        coolant = prev.coolant
        if coolant is None or not coolant.on:
            return 1.0
        values = [float(coolant.friction_factor[c]) for c in g.contacts if c in coolant.friction_factor]
        return float(min(values)) if values else 1.0

    def _note_once(self, ctx, ws, key, event, quantities, tick, parents) -> int:
        if key in ws.trace_notes:
            return int(ws.trace_notes[key])
        event_id = ctx.trace.record(tick * ctx.tick_s, tick, "wear", event, quantities, parents)
        ws.trace_notes[key] = int(event_id)
        return int(event_id)

    def _notes(self, ctx, ws, sw, notes, tick) -> None:
        for note in sorted(set(notes)):
            if note == "temperature_fallback" and not sw.flags["temperature_fallback"]:
                sw.flags["temperature_fallback"] = True
                self._note_once(ctx, ws, f"tfb:{sw.slot}", "wear:temperature_from_response_table", {"slot": sw.slot},
                                tick, [])
            elif note == "temperature_body":
                self._note_once(ctx, ws, f"tbody:{sw.slot}", "wear:temperature_body_only", {"slot": sw.slot}, tick, [])
            elif note == "stress_fallback" and not sw.flags["stress_fallback"]:
                sw.flags["stress_fallback"] = True
                self._note_once(ctx, ws, f"sfb:{sw.slot}", "wear:contact_stress_from_response_table", {"slot": sw.slot},
                                tick, [])
            elif note == "stress_missing":
                self._note_once(ctx, ws, f"nostress:{sw.slot}", "wear:no_contact_stress", {"slot": sw.slot}, tick, [])
                if f"tool slot {sw.slot}: no contact stresses (cutting gives no wear integrals, no response table)" \
                        not in ws.estimated:
                    ws.estimated.append(f"tool slot {sw.slot}: no contact stresses (cutting gives no wear integrals, "
                                        "no response table)")
            elif note == "finish_unreachable":
                self._note_once(ctx, ws, f"finish:{sw.slot}", "wear:finish_spec_not_reachable_by_feed",
                                {"slot": sw.slot}, tick, [])
            elif note == "interface_above_melting":
                self._note_once(ctx, ws, f"melt:{sw.slot}", "wear:edge_temperature_above_work_melting",
                                {"slot": sw.slot, "interface_max_K": float(sw.props.get("interface_max_K") or 0.0)},
                                tick, [])

    def _events(self, ctx, ws, sw, inp, out, chips, g, broke, tick) -> None:
        valid = sw.geo.valid != 0
        law_event = ws.trace_notes.get(f"law:{sw.slot}")
        parents = [int(law_event)] if law_event is not None else []
        if out.scal[A.OS_COAT_THROUGH] > 0 and not sw.flags["coating_through"]:
            sw.flags["coating_through"] = True
            self._note_once(ctx, ws, f"coat:{sw.slot}", "wear:flank_coating_worn_through",
                            {"slot": sw.slot, "elements": float(out.scal[A.OS_COAT_THROUGH]),
                             "VB_max_m": float(np.max(out.VB[valid]))}, tick, parents)
        kt_max = self._rt[sw.slot].summary["KT_max"]
        if kt_max >= P.get(P.FLANK)["crater_onset_m"] and not sw.flags["crater_onset"]:
            sw.flags["crater_onset"] = True
            self._note_once(ctx, ws, f"crater:{sw.slot}", "wear:crater_onset",
                            {"slot": sw.slot, "KT_m": kt_max, "rake_temp_max_K": float(np.max(inp.rake_T))}, tick,
                            parents)
        if out.scal[A.OS_CRACK_INIT] > 0 and not sw.flags["comb_cracks"]:
            sw.flags["comb_cracks"] = True
            self._note_once(ctx, ws, f"cracks:{sw.slot}", "wear:comb_cracks",
                            {"slot": sw.slot, "thermal_damage_max": float(out.scal[A.OS_MAX_D]),
                             "cycle_dT_max_K": float(np.max(inp.cycle_dT)),
                             "crack_depth_max_m": float(np.max(sw.elem.crack_m))}, tick, parents)
        sw.counters["bue_breakoffs"] += int(broke)
        sw.counters["chip_trials"] += int(g.shocks.shape[0])
        cap = ws.settings.max_chip_events
        for i in range(chips.shape[0]):
            chipped = chips[i, A.CO_CHIPPED] > 0.0
            spalled = chips[i, A.CO_SPALLED] > 0.0
            if not (chipped or spalled):
                continue
            row = g.shocks[i]
            if chipped:
                sw.counters["chip_events"] += 1
                sw.flags["chipped_any"] = True
            else:
                sw.counters["spalls"] += 1
            if sw.counters["op_events"] >= cap:
                continue
            sw.counters["op_events"] += 1
            ev_tick = int(row[A.SH_TICK])
            ev = ctx.trace.record(ev_tick * ctx.tick_s, ev_tick, "wear",
                                  "wear:edge_chipped" if chipped else "wear:coating_spalled",
                                  {"slot": sw.slot, "tooth": float(row[A.SH_TOOTH]), "elem": float(row[A.SH_ELEM]),
                                   "probability": float(chips[i, A.CO_P] if chipped else chips[i, A.CO_P_COAT]),
                                   "edge_hazard": float(chips[i, A.CO_HAZARD]),
                                   "wedge_stress_Pa": float(chips[i, A.CO_SIGMA_PA]),
                                   "force_N": float(row[A.SH_FORCE]), "kind": float(row[A.SH_KIND]),
                                   "chip_depth_m": float(chips[i, A.CO_DEPTH_M])}, parents)
            ws.trace_notes[f"last_chip:{sw.slot}"] = int(ev)

    @staticmethod
    def _counters(ws, sw, inp, g, frame) -> None:
        c = sw.counters
        c["in_cut_s"] += float(g.in_cut_s)
        c["cutting_distance_m"] += float(np.max(g.elem["flank_sliding_m"])) if g.elem is not None else 0.0
        if frame is not None:
            feed = np.asarray(frame.feed_vel_M_m_s, dtype=float).reshape(-1, 3)
            speed = float(np.mean(np.linalg.norm(feed, axis=1))) if feed.size else 0.0
            c["cut_path_m"] += speed * float(g.in_cut_s)
        if c["last_part"] != ws.part_index:
            c["parts"] += 1
            c["last_part"] = ws.part_index
        cutting = inp.in_cut_s > 0.0
        flank = np.where(cutting, inp.flank_T.max(axis=2), 0.0)
        rake = np.where(cutting, inp.rake_T.max(axis=2), 0.0)
        c["max_flank_K"] = max(c["max_flank_K"], float(flank.max()))
        c["max_rake_K"] = max(c["max_rake_K"], float(rake.max()))
        hot = np.maximum(flank, rake) > float(sw.props["max_service_K"])
        if hot.any():
            c["above_service_s"] += float(inp.in_cut_s[hot].max())

    def _breakage(self, ctx, state, ws, sw, rt, g, prev, frame, tick0, tick1) -> float:
        chips = prev.chips
        pack_t = abs(float(chips.packing_torque_Nm.get(sw.slot, 0.0))) if chips is not None else 0.0
        force = np.asarray(chips.packing_force_T0_N.get(sw.slot, np.zeros(3)), dtype=float).reshape(-1) \
            if chips is not None else np.zeros(3)
        pack_f = float(np.hypot(force[0], force[1])) if force.size >= 2 else 0.0
        omega = float(np.mean(np.abs(np.asarray(frame.spindle_omega_rad_s, dtype=float)))) \
            if frame is not None and np.size(frame.spindle_omega_rad_s) else 0.0
        worst, failed = S.check_sections(sw, g, pack_t, pack_f, omega, rt.z_load_m)
        if worst is None:
            return 0.0
        c = sw.counters
        c["max_section_stress_Pa"] = max(c["max_section_stress_Pa"], worst.sigma_eq_Pa)
        c["max_failure_probability"] = max(c["max_failure_probability"], worst.probability)
        tick = tick1
        if frame is not None and g.peak_row >= 0 and g.peak_row < np.size(frame.tick):
            tick = int(np.asarray(frame.tick)[g.peak_row])
        path = int(frame.path) if frame is not None else 0
        quantities = {"slot": sw.slot, "section_z_m": worst.z_m, "bending_stress_Pa": worst.sigma_b_Pa,
                      "torsion_stress_Pa": worst.tau_Pa, "equivalent_stress_Pa": worst.sigma_eq_Pa,
                      "strength_Pa": worst.strength_Pa, "failure_probability": worst.probability,
                      "fatigue_damage": worst.fatigue, "cutting_von_mises_Pa": worst.cutting_vm_Pa,
                      "torque_peak_Nm": g.torque_peak_Nm, "packing_torque_Nm": pack_t, "packing_force_N": pack_f}
        parents = []
        for key in (f"last_chip:{sw.slot}", f"cracks:{sw.slot}", f"law:{sw.slot}"):
            if key in ws.trace_notes:
                parents.append(int(ws.trace_notes[key]))
        if failed is not None:
            q = dict(quantities, section_z_m=failed.z_m, bending_stress_Pa=failed.sigma_b_Pa,
                     torsion_stress_Pa=failed.tau_Pa, equivalent_stress_Pa=failed.sigma_eq_Pa,
                     strength_Pa=failed.strength_Pa, failure_probability=failed.probability,
                     fatigue_damage=failed.fatigue)
            event = ctx.trace.record(tick * ctx.tick_s, tick, "wear", f"wear:tool_fracture:{failed.cause}", q, parents)
            life = sw.life
            life.update(broken=True, broken_tick=int(tick), broken_cause=failed.cause, broken_section=failed.name,
                        broken_z_m=float(failed.z_m))
            what = {"bending": "bending overload", "torsion": "torsion overload", "packing": "chip packing load",
                    "fatigue": "fatigue (Miner sum reached 1)"}[failed.cause]
            S.finding(ctx, state, Category.TOOL_BREAKAGE, Severity.ERROR,
                      f"T{sw.number} breaks at its {failed.name} section ({what}: {failed.sigma_eq_Pa / 1e6:.0f} MPa "
                      f"against {failed.strength_Pa / 1e6:.0f} MPa)", tick, path, q, [event])
        elif worst.probability > ws.settings.breakage_risk_warn and sw.flags["risk_warned_op"] != ws.op_index:
            sw.flags["risk_warned_op"] = ws.op_index
            event = ctx.trace.record(tick * ctx.tick_s, tick, "wear", "wear:breakage_risk", quantities, parents)
            S.finding(ctx, state, Category.TOOL_BREAKAGE, Severity.WARNING,
                      f"T{sw.number}: fracture probability {worst.probability:.1%} at the {worst.name} section "
                      f"({worst.sigma_eq_Pa / 1e6:.0f} MPa, median strength {worst.strength_Pa / 1e6:.0f} MPa)",
                      tick, path, quantities, [event])
        return worst.sigma_eq_Pa

    def _life(self, ctx, state, ws, sw, rt, out, g, frame, dt, tick0, tick1, now_s) -> None:
        life = sw.life
        window = ws.settings.rate_window_s
        feed = self._feed(sw, frame)
        if feed > 0.0 and g.in_cut_s > 0.0:
            life["feed"] = L.ewma(life["feed"], feed, g.in_cut_s, window)
            life["feed_m"] = L.mean(life["feed"])
        notes: list = []
        values, limits, extra = S.criteria(sw, out, rt, rt.tolerances, life["feed_m"], notes)
        self._notes(ctx, ws, sw, notes, tick1)
        before = dict(life["values"])
        rates = {k: list(v) for k, v in life["rates"].items()}
        if g.in_cut_s > 0.0:
            for name, value in values.items():
                if name in before:
                    rates[name] = L.ewma(rates.get(name), max(value - before[name], 0.0) / g.in_cut_s, g.in_cut_s,
                                         window)
        life["rates"] = rates
        if frame is not None:
            life["duty"] = L.ewma(life["duty"], min(g.in_cut_s / dt, 1.0), dt, window)
        remaining, cause = L.remaining_s(values, limits, {k: L.mean(v) for k, v in rates.items()})
        duty = L.mean(life["duty"])
        life["remaining_s"] = remaining
        life["remaining_cause"] = cause
        life["replacement_s"] = (now_s + remaining / duty) if math.isfinite(remaining) and duty > 0.0 else math.inf
        if not life["expired"]:
            name, fraction = L.first_reached(before, values, limits)
            if name:
                tick = int(round(tick0 + fraction * (tick1 - tick0)))
                valid = sw.geo.valid != 0
                q = {"slot": sw.slot, "value": float(values[name]), "limit": float(limits[name]),
                     "VB_max_m": float(np.max(out.VB[valid])), "KT_max_m": float(np.max(sw.elem.kt[valid])),
                     "notch_max_m": float(np.max(out.VB_N[valid])), "in_cut_s": float(sw.counters["in_cut_s"]),
                     "cutting_distance_m": float(sw.counters["cutting_distance_m"])}
                parents = [int(ws.trace_notes[k]) for k in (f"law:{sw.slot}", f"coat:{sw.slot}", f"crater:{sw.slot}",
                                                              f"last_chip:{sw.slot}") if k in ws.trace_notes]
                event = ctx.trace.record(tick * ctx.tick_s, tick, "wear", f"wear:life_end:{name}", q, parents)
                life.update(expired=True, cause=name, expired_tick=tick, expired_time_s=tick * ctx.tick_s)
                if name in ("breakage", "fatigue"):
                    life["cause"] = sw.life["broken_cause"] or name
                else:
                    S.finding(ctx, state, Category.TOOL_END_OF_LIFE, Severity.WARNING,
                              f"T{sw.number} reaches the end of its life ({_CRITERION_TEXT[name]}: "
                              f"{_fmt(name, values[name])} of {_fmt(name, limits[name])})", tick,
                              int(frame.path) if frame is not None else 0, q, [event])
        life["values"] = values
        life["limits"] = {k: v for k, v in limits.items()}
        life["used_fraction"] = 1.0 if life["broken"] else L.used_fraction(values, limits)
        for key in ("Ra_m", "Rt_m", "drift_m"):
            if key in extra:
                life[key] = float(extra[key])

    @staticmethod
    def _feed(sw, frame) -> float:
        """Feed per tooth (rotating tools) or per revolution (turning) of the step."""
        if frame is None:
            return 0.0
        feed = np.asarray(frame.feed_vel_M_m_s, dtype=float).reshape(-1, 3)
        omega = np.asarray(frame.spindle_omega_rad_s, dtype=float)
        if not feed.size or not omega.size:
            return 0.0
        w = float(np.abs(omega).mean())
        if w <= 0.0:
            return 0.0
        per_rev = float(np.sqrt((feed * feed).sum(axis=1)).mean()) * 2.0 * math.pi / w
        teeth = sw.geo.valid.shape[0]
        return per_rev / teeth if sw.rotating and teeth > 0 else per_rev

    # ------------------------------------------------------------------ parts and results
    def _close_part(self, ws) -> None:
        for sw in ws.slots.values():
            out = self._rt[sw.slot].out if sw.slot in self._rt and self._rt[sw.slot].out is not None \
                else self._outputs(sw)
            used = float(sw.life["used_fraction"])
            valid = sw.geo.valid != 0
            sw.per_part.append({
                "part": int(ws.part_index), "life_used_fraction": used,
                "life_used_part_fraction": max(used - float(sw.part_start.get("used_fraction", 0.0)), 0.0),
                "cut_time_s": float(sw.counters["in_cut_s"] - sw.part_start.get("in_cut_s", 0.0)),
                "VB_max_m": float(np.max(out.VB[valid])) if valid.any() else 0.0,
                "wear_offsets_m": self._offsets(sw, out), "changes": 0, "broken": bool(sw.life["broken"])})
            sw.part_start = {"used_fraction": used, "in_cut_s": float(sw.counters["in_cut_s"])}

    @staticmethod
    def _offsets(sw, out) -> dict:
        """Wear offsets a machinist would enter: radius = largest radial wear of the finishing (else of the
        used) elements, length = largest axial wear."""
        valid = sw.geo.valid != 0
        sel = (sw.finishes != 0) & valid
        if not sel.any():
            sel = (sw.elem.used != 0) & valid
        if not sel.any():
            return {"radius": 0.0, "length": 0.0}
        return {"radius": float(np.max(out.radial[sel])), "length": float(np.max(out.axial[sel]))}

    def _values(self, ws, prev, feedback, stresses) -> None:
        values = {}
        active = None
        for frame in prev.motion.paths:
            if int(frame.tool_slot) in ws.slots:
                active = int(frame.tool_slot)
                break
        nan = float("nan")
        if active is not None:
            sw = ws.slots[active]
            rt = self._rt[active]
            summ = rt.summary if rt.summary is not None else S.summary(sw, rt.out)
            vb_tooth = np.full(MAX_TEETH, nan)
            n = min(summ["VB_tooth"].shape[0], MAX_TEETH)
            vb_tooth[:n] = summ["VB_tooth"][:n]
            fatigue = sw.sections["fatigue"]
            values.update({
                "wear.VB_max_m": summ["VB_max"], "wear.VB_tooth_m": vb_tooth, "wear.KT_max_m": summ["KT_max"],
                "wear.notch_max_m": summ["notch_max"], "wear.edge_radius_max_m": summ["edge_r_max"],
                "wear.radial_wear_max_m": summ["radial_max"], "wear.coating_min_m": summ["coat_min"],
                "wear.bue_height_max_m": summ["bue_max"],
                "wear.flank_rate_max_m_s": float(rt.out.scal[A.OS_MAX_RATE_M_S]) if rt.cut else 0.0,
                "wear.chipped_elements": float(summ["chipped"]), "wear.thermal_damage_max": summ["D_max"],
                "wear.section_fatigue_max": float(fatigue.max()) if fatigue.size else 0.0,
                "wear.section_stress_max_Pa": float(stresses.get(active, 0.0)),
                "wear.life_used_fraction": float(sw.life["used_fraction"]),
                "wear.remaining_life_s": float(sw.life["remaining_s"]) if math.isfinite(sw.life["remaining_s"])
                else nan})
        else:
            for name in ("wear.VB_max_m", "wear.KT_max_m", "wear.notch_max_m", "wear.edge_radius_max_m",
                         "wear.radial_wear_max_m", "wear.coating_min_m", "wear.bue_height_max_m",
                         "wear.flank_rate_max_m_s", "wear.chipped_elements", "wear.thermal_damage_max",
                         "wear.section_fatigue_max", "wear.section_stress_max_Pa", "wear.life_used_fraction",
                         "wear.remaining_life_s"):
                values[name] = nan
            values["wear.VB_tooth_m"] = np.full(MAX_TEETH, nan)
        per_slot = np.full(MAX_SLOTS, nan)
        for slot, sw in sorted(ws.slots.items()):
            if slot < MAX_SLOTS:
                per_slot[slot] = float(sw.life["used_fraction"])
        values["wear.life_used_slot"] = per_slot
        ws.values = values


_CRITERION_TEXT = {"vb": "flank wear VB_B", "vb_max": "local flank wear VB_Bmax", "crater": "crater depth KT",
                   "notch": "notch wear VB_N", "dimension": "dimensional drift of the finished surfaces",
                   "finish": "surface finish estimate / specification", "fatigue": "section fatigue",
                   "breakage": "breakage"}


def _fmt(name: str, value: float) -> str:
    if name in ("finish", "fatigue", "breakage"):
        return f"{value:.3g}"
    return f"{value * 1e3:.3f} mm"


def _mesh_stock_envelope(project, stock):
    """(kind, size) of StockShell for a mesh stock: on a lathe a cylinder about S z from z = 0 (diameter = twice the
    mesh's largest radius, length = its top), elsewhere a block from the S origin (its bounding box) when the mesh
    starts at the origin as stock frames do; None when the mesh cannot be read or does not fit either."""
    path = P.field_of(stock, "mesh") if stock is not None else None
    if not path:
        return None
    try:
        from ..removal.stock import read_mesh_file
        V, _ = read_mesh_file(str(P.plain(path)), unit="mm")
    except Exception:  # noqa: BLE001 - no skin model without the mesh; the caller notes it
        return None
    V = np.asarray(V, dtype=float)
    if V.size == 0:
        return None
    lo, hi = V.min(axis=0), V.max(axis=0)
    tol = 1e-6
    machine = P.field_of(project, "machine")
    mtype = P.text(machine, "type", "") if machine is not None else ""
    if mtype in ("lathe", "mill_turn", "swiss") and abs(lo[2]) <= tol and hi[2] > 0.0:
        return "cylinder", [2.0 * float(np.hypot(V[:, 0], V[:, 1]).max()), float(hi[2])]
    if np.all(np.abs(lo) <= tol) and np.all(hi > 0.0):
        return "block", [float(v) for v in hi]
    return None


__all__ = ["WearDomain"]
