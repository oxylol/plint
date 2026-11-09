"""The chips domain (docs/models/chips.md): what happens to chips after the cut.

Timing. The hot loop of step n writes emission rows into the chips EventBuffer; :meth:`ChipsDomain.absorb`
keeps them. :meth:`ChipsDomain.macro_step` of step n reads ``bus_prev`` (the records of step n-1) and
integrates the chips over the window of that motion record: it releases the rows of that window scaled by
the same window's removal record (mass rule, architecture 5.4), processes the recut rows of that window,
moves, cools, settles, flushes and conveys the chips, updates the flutes, and writes ``bus_next.chips``.
Chips therefore run one macro step behind the hot loop; the ledger terms written at step n cover the
removed volume up to the window of ``bus_prev`` (``ChipFeedback.booked_until_tick``).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.abi import EventBuffer
from ..core.bus import ChipFeedback, ChipFormationContact
from ..core.errors import InputError, TwinError
from ..units import REFERENCE_TEMP_K
from . import abi, emission, flutes, hazards, kernels, params, scene
from . import schema as chips_schema
from .channels import channels as chip_channels
from .formation import MechanicsFormation
from .protocols import BoxScene, FormationModel, ProjectTools, StockFaces
from .state import (FL_BULK_M3, FL_CONFINED, FL_ENTHALPY_J, FL_FILL, FL_FORCE_N, FL_FORM, FL_INFLOW_M3_S,
                    FL_LENGTH_E_M, FL_LENGTH_MASS, FL_MASS_KG, FL_POWER_W, FL_SIZE_MASS, FL_STRESS_PA, FL_THICK_MASS,
                    FL_TORQUE_NM, FL_V_EFF_M_S, FL_WIDTH_MASS, H_AIR, H_COOLANT, H_FIXTURE, H_MACHINE, H_PART,
                    LS_CURL_M, LS_FORM, LS_IN_CUT, LS_LENGTH_M, LS_TEMP_K, LS_THICK_M, LS_VALID, LS_WIDTH_M, PAN_BULK,
                    PAN_ENTHALPY, PAN_LONG, PAN_MASS, T_FORMATION, T_RECUT, T_WELD, T_WRAP, ChipsState)

DOMAIN = "chips"
FORM_ELEMENTAL = 70


class ChipsDomain:
    """Chips after the cut. Collaborators (Protocols of :mod:`cnctwin.chips.protocols`) are wired by the
    integration: ``formation`` (mechanics), ``geometry`` (removal + holding + machine), ``tools`` (tooling),
    ``locators`` (holding). Without ``geometry``/``tools``/``locators`` the domain uses the analytic stock
    box, the project tool list and the stock faces (recorded in the trace and in finalize())."""

    name = DOMAIN

    def __init__(self, formation: FormationModel | None = None, geometry=None, tools=None, locators=None):
        self.formation_model = formation
        self.geometry = geometry
        self.tools = tools
        self.locators = locators
        self.fallbacks: list[str] = []

    # ================================================================== Domain interface
    def channels(self):
        return chip_channels()

    def init(self, ctx, state) -> None:
        project = ctx.project
        self.settings = chips_schema.settings(getattr(project, "physics", None) if project is not None else None)
        self.params = params.load()
        material = getattr(project, "material", None) if project is not None else None
        self.material = material
        self.rho = params.density(material)
        self.model = params.chip_model(material, self.settings, self.params)
        self.yield_Pa = params.yield_strength(material)
        self.dust_fraction = params.dust_fraction(material, self.params.bulk)
        self.ignition_chip_K, self.ignition_fines_K = params.ignition_K(material)
        machine = getattr(project, "machine", None) if project is not None else None
        self.machine = machine
        self.gravity = scene.gravity_M(machine)
        self.spindles = scene.spindle_kinds(machine)
        self.conveyor_rate_kg_s, self.conveyor_kind = scene.conveyor_rate(machine)
        env = getattr(project, "environment", None) if project is not None else None
        self.ambient_K = params.value(getattr(env, "ambient_K", None) if env is not None else None, REFERENCE_TEMP_K)
        coolant = getattr(project, "coolant", None) if project is not None else None
        self.fluid = getattr(coolant, "fluid", None) if coolant is not None else None
        self.fluid_kind = str(getattr(self.fluid, "kind", "") or "") if self.fluid is not None else ""
        if self.formation_model is None:
            self.formation_model = MechanicsFormation(material, ambient_K=self.ambient_K,
                                                      bin_rel=self.settings.formation_bin_rel,
                                                      bin_rad=self.settings.formation_bin_rad)
        if self.geometry is None:
            self.geometry = _stock_scene(project, str(getattr(machine, "type", "") or "") in scene.LATHE_TYPES)
            self.fallbacks.append("geometry: analytic stock box (no in-process workpiece wired)")
        if self.tools is None:
            self.tools = ProjectTools(project, state.tools, float(self.params.flute["gullet_area_fraction"]))
        lo, hi = self.geometry.stock_bounds_S(0)
        if self.locators is None:
            self.locators = StockFaces(lo, hi)
            self.fallbacks.append("locators: the six stock faces (no next-setup locators wired)")
        self.operator_cleans = self._operator_cleans(project)
        slots = [s.slot for s in getattr(state.tools, "slots", ())]
        n_slots = max(slots) + 1 if slots else 1
        axis, sign, axes = scene.pile_axes(-self.gravity)
        pile = scene.new_pile(lo, hi, axes, self.settings.pile_cell_m, self.settings.pile_margin_m)
        cs = ChipsState(particles=abi.particles(256), pile=pile, n_slots=n_slots)
        cs.pile_axis, cs.pile_sign = axis, sign
        cs.recut_hist = np.zeros((n_slots, max(1, int(round(float(self.params.hazards["recut_window_s"])
                                                             / max(ctx.macro_s, 1e-12)))), 2))
        cs.emission_capacity = self.settings.emission_capacity
        state.chips = cs
        scene.refresh_base(cs.pile, self.geometry, 0, axis, sign)
        cs.pile_version = int(self.geometry.version(0))
        for note in self.fallbacks:
            ctx.trace.record(0.0, 0, DOMAIN, "chips:fallback:" + note.split(":")[0], {}, [])
        state.bus_next.chips = self._record(state, cs, None, [], [], 0.0, {})
        state.bus_prev.chips = self._record(state, cs, None, [], [], 0.0, {})

    def kernel_inputs(self, ctx, state, bus) -> abi.ChipsKernelInputs:
        cs = state.chips
        occ = abi.Occupancy(np.array(cs.occ_origin_S_m, dtype=float), np.array([float(cs.occ_cell_m)]),
                            np.array(cs.occ_mass_kg, dtype=float), np.array(cs.occ_size_m, dtype=float),
                            np.array(cs.occ_thickness_m, dtype=float))
        packing = abi.packing_loads(cs.n_slots)
        packing.torque_Nm[:cs.n_slots] = cs.flute[:, FL_TORQUE_NM]
        packing.force_T0_N[:cs.n_slots, 2] = cs.flute[:, FL_FORCE_N]
        return abi.ChipsKernelInputs(abi.emission_buffer(cs.emission_capacity), occ, packing)

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        cs = state.chips
        buffer = _emission_buffer(outputs)
        if buffer is None:
            return
        n = int(buffer.n[0])
        if buffer.n[1]:
            cs.emission_capacity *= 2
            ctx.trace.record(state.clock.tick * ctx.tick_s, state.clock.tick, DOMAIN, "chips:emission_overflow",
                             {"rows": float(n), "capacity": float(cs.emission_capacity)}, [])
        if n:
            rows = np.array(buffer.rec[:n, :abi.EMISSION_WIDTH], dtype=float)
            cs.pending = np.concatenate([cs.pending, rows]) if len(cs.pending) else rows

    def macro_step(self, ctx, state, prev, nxt) -> None:
        cs = state.chips
        frames = scene.window_frames(prev.motion, ctx.tick_s)
        if frames is None or frames.tick_end <= cs.window_end_tick:
            nxt.chips = self._record(state, cs, None, [], [], 0.0, {})
            return
        cs.macro += 1
        self._step(ctx, state, cs, prev, nxt, frames)
        cs.window_end_tick = frames.tick_end
        cs.last_R_M_S = frames.R_M_S.copy()
        cs.last_o_S_M = frames.o_S_M[-1].copy()

    def end_operation(self, ctx, state) -> None:
        """The operation ends: the last window's chips are booked, flying chips come to rest, flutes empty
        and wraps fall, the chips on the faces the next setup clamps or locates on are measured
        (locator_debris_m), then part and fixture are cleaned into the chip pan."""
        cs = state.chips
        self._flush_pending(ctx, state)
        tick = max(cs.window_end_tick, state.clock.tick)
        frames = scene.static_frames(cs.last_R_M_S, cs.last_o_S_M, tick * ctx.tick_s, ctx.macro_s, tick)
        W = scene.window_layout(frames, cs.pile, self.gravity)
        P = cs.particles
        n = int(P.n[0])
        attached = np.nonzero((P.status[:n] == abi.ON_TOOL) | (P.status[:n] == abi.WRAPPED)
                              | (P.status[:n] == abi.ON_NOZZLE))[0]
        P.status[attached] = abi.FLYING
        P.owner[attached] = -1
        self._empty_flutes(cs)
        out = np.zeros(abi.N_OUT)
        kernels.drop_particles(P, cs.pile, W, self.model, out)
        self._pan_add(cs, out)
        cs.locator_debris = self._locator_debris(ctx, state, cs, tick)
        self._clean_part(cs)
        for bus in (state.bus_prev, state.bus_next):
            bus.chips.locator_debris_m = dict(cs.locator_debris)
            terms = cs.mass_terms()
            bus.chips.in_flight_kg, bus.chips.settled_kg = terms["in_flight_kg"], terms["settled_kg"]
            bus.chips.adhered_kg, bus.chips.in_flutes_kg = terms["adhered_kg"], terms["in_flutes_kg"]
        self._write_ledger(state, cs)

    def finalize(self, ctx, state) -> dict:
        cs = state.chips
        self._flush_pending(ctx, state)
        terms = cs.mass_terms()
        return {"emitted_kg": float(cs.emitted[0]), **terms, "pan_kg": float(cs.pan[PAN_MASS]),
                "heat_to_part_J": float(cs.heat[H_PART]), "heat_to_fixture_J": float(cs.heat[H_FIXTURE]),
                "heat_to_machine_J": float(cs.heat[H_MACHINE]), "heat_to_coolant_J": float(cs.heat[H_COOLANT]),
                "heat_to_air_J": float(cs.heat[H_AIR]), "recut_missing_kg": float(cs.recut_missing_kg),
                "unscaled_kg": float(cs.unscaled_kg), "particles_created": int(cs.next_pid),
                "pending_rows": int(len(cs.pending)), "locator_debris_m": dict(cs.locator_debris),
                "fallbacks": list(self.fallbacks)}

    def _flush_pending(self, ctx, state) -> None:
        """Book the windows not integrated yet (the last window of an operation or of the run): the newest
        motion record whose window ends after the last integrated one (no concurrent writer at this point)."""
        cs = state.chips
        for bus in (state.bus_prev, state.bus_next):
            frames = scene.window_frames(bus.motion, ctx.tick_s)
            if frames is not None and frames.tick_end > cs.window_end_tick:
                cs.macro += 1
                self._step(ctx, state, cs, bus, state.bus_next, frames)
                cs.window_end_tick = frames.tick_end
                cs.last_R_M_S = frames.R_M_S.copy()
                cs.last_o_S_M = frames.o_S_M[-1].copy()

    # ================================================================== helpers for other parts
    def macro_values(self, state) -> dict:
        return dict(state.chips.channel_values)

    def keyframe_arrays(self, state) -> dict:
        """Particles in flight and the pile summary for chips/kf_<k>.npz."""
        cs = state.chips
        P = cs.particles
        n = int(P.n[0])
        live = np.nonzero(P.status[:n] != abi.FREE)[0]
        cell = float(cs.pile.cell_m[0])
        return {"pos_M_m": P.pos_M_m[live].astype(np.float32), "mass_kg": P.mass_kg[live].astype(np.float32),
                "temp_K": P.temp_K[live].astype(np.float32), "form": P.form[live].astype(np.int16),
                "status": P.status[live].astype(np.int8), "colour_rgb": cs.colour[live].copy(),
                "length_m": P.length_m[live].astype(np.float32),
                "pile_origin_S_m": cs.pile.origin_G_m.copy(), "pile_axes_S": cs.pile.axes_S.copy(),
                "pile_cell_m": np.array([cell]),
                "pile_height_m": (cs.pile.bulk_m3 / (cell * cell)).astype(np.float16)}

    # ================================================================== the step
    def _step(self, ctx, state, cs, prev, nxt, frames) -> None:
        settings = self.settings
        dt_window = frames.t_end_s - frames.t0_s
        tick_end = frames.tick_end
        workpiece = frames.paths[0].workpiece
        rotating_part = self._part_rotates(frames)
        # ---------------------------------------------------------- pile frame and base
        up_S = -(frames.R_M_S.T @ self.gravity)
        axis = int(np.argmax(np.abs(up_S)))
        sign = 1 if up_S[axis] >= 0 else -1
        batches = []
        # a spinning workpiece holds no pile: its grid is kept (its S axes turn with the spindle)
        if (not rotating_part and (axis != cs.pile_axis or sign != cs.pile_sign)) or workpiece != cs.pile_workpiece:
            _, _, axes = scene.pile_axes(up_S)
            batches.append(self._regrid(cs, frames, axes, axis, sign, workpiece))
        elif int(self.geometry.version(workpiece)) != cs.pile_version:
            box = self.geometry.changed_box_S(workpiece)
            lost = scene.refresh_base(cs.pile, self.geometry, workpiece, cs.pile_axis, cs.pile_sign, box)
            cs.pile_version = int(self.geometry.version(workpiece))
            self._spill_cells(cs, lost)
        H = cs.pile
        W = scene.window_layout(frames, H, self.gravity)
        coolant = prev.coolant
        fluid_temp = float(getattr(coolant, "fluid_temp_K", self.ambient_K) or self.ambient_K)
        fluid_props = self._fluid_props(fluid_temp)
        liquid = scene.liquid_properties(fluid_props, fluid_temp, float(self.params.contact["film_thickness_m"]))
        air = scene.air_properties(self.ambient_K)
        scene.coolant_on_pile(H, coolant, float(liquid[0]))
        scene.surface_temperatures(H, prev.thermal, workpiece, self.ambient_K)
        nozzle_r0 = dict(zip(cs.nozzle_ids, cs.nozzle_r0_m.tolist()))
        info = scene.build_scene(frames, self.tools, *self._enclosure(frames), self.gravity, coolant, air, liquid,
                                 self.params.aero, self.spindles, self.geometry, nozzle_r0,
                                 float(self.params.contact["nozzle_radius_default_m"]), settings.rotating_part_rad_s)
        self._sync_nozzles(cs, nozzle_r0, info)
        S = info.scene
        rec, n_mob = np.zeros((settings.mobilise_capacity, abi.MOB_WIDTH)), np.zeros(2, np.int64)
        # ---------------------------------------------------------- tools entering the cut: chips nested there
        entering = self._entering(cs, prev)
        nested = self._nested_at_entry(cs, frames, entering)
        # ---------------------------------------------------------- recut rows of the window
        recut_kg_slot = self._recut(ctx, cs, prev.cutting, frames, W, S, info, rec, n_mob)
        # ---------------------------------------------------------- tools sweeping through the pile, flushing
        kernels.sweep_cylinders(H, W, S, self.model, rec, n_mob)
        if rotating_part:
            self._shed_rotating_part(cs, W, frames, rec, n_mob)
        self._flush(cs, H, W, coolant, liquid, dt_window, rec, n_mob)
        if n_mob[1]:
            ctx.trace.record(frames.t_end_s, tick_end, DOMAIN, "chips:mobilise_buffer_full",
                             {"rows": float(n_mob[0])}, [])
        mob_rows = rec[:int(n_mob[0])].copy()
        batches.append(emission.batch_from_mob(mob_rows, W, self.rho, self._temperature))
        # ---------------------------------------------------------- emissions of the window
        released, kinetic = self._release_rows(ctx, state, cs, prev, frames, tick_end)
        for item in released:
            batches.append(item.batch)
        # ---------------------------------------------------------- attached chips released
        self._release_attached(cs, prev, frames, info, recut_kg_slot, entering)
        # ---------------------------------------------------------- flutes
        batches.append(self._flutes(ctx, cs, prev, frames, H, dt_window))
        new = emission.concat(batches)
        emission.insert(cs, new, self.params, self.rho, settings.max_particles, self._enthalpy, self._temperature)
        # ---------------------------------------------------------- flight
        out, ev = abi.step_out(settings.event_capacity)
        kernels.advance_particles(cs.particles, H, W, S, self.model, out, ev)
        if ev.n[1]:
            ctx.trace.record(frames.t_end_s, tick_end, DOMAIN, "chips:event_buffer_full", {"rows": float(ev.n[0])}, [])
        self._attach_events(ctx, cs, ev, frames, info)
        # ---------------------------------------------------------- piles, heat, pan, conveyor
        kernels.relax_pile(H, settings.relax_max_iter, settings.relax_tol_fraction * H.cell_m[0] ** 3, out)
        self._pan_add(cs, out)
        q_surface = np.zeros(H.base_m.shape)
        heat_out = np.zeros(2)
        hp = np.array([self.params.heat["h_bed_W_m2K"], self.params.heat["h_top_air_W_m2K"],
                       self.params.heat["h_top_wet_W_m2K"], self.ambient_K, fluid_temp])
        kernels.pile_heat(H, self.model, hp, dt_window, q_surface, heat_out)
        heat_rows, q_part, q_fixture, q_machine = self._surface_heat(H, q_surface)
        pan_coolant, pan_air = self._pan_heat(cs, dt_window, S.walls_wet[0] != 0, fluid_temp)
        self._convey(cs, dt_window)
        heat = {"coolant": float(out[abi.OUT_HEAT_COOLANT_J]) + float(heat_out[0]) + pan_coolant,
                "air": float(out[abi.OUT_HEAT_AIR_J]) + float(heat_out[1]) + pan_air,
                "part": q_part, "fixture": q_fixture, "machine": q_machine}
        cs.heat[H_PART] += q_part
        cs.heat[H_FIXTURE] += q_fixture
        cs.heat[H_MACHINE] += q_machine
        cs.heat[H_COOLANT] += heat["coolant"]
        cs.heat[H_AIR] += heat["air"]
        # ---------------------------------------------------------- occupancy grid for the next hot loop
        self._occupancy(cs, frames, ctx.macro_s)
        self._update_attached_positions(cs, frames, info)
        # ---------------------------------------------------------- findings
        extras = self._findings(ctx, state, cs, prev, frames, info, released, recut_kg_slot, nested, dt_window,
                                tick_end)
        record = self._record(state, cs, frames, released, heat_rows, dt_window, heat)
        record.recut_mass_kg = float(sum(recut_kg_slot.values()))
        record.kinetic_energy_J = kinetic
        record.blocked_nozzles = extras["blocked"]
        record.probe_blocked = extras["probe"]
        record.nest_fraction = extras["nest"]
        record.locator_debris_m = dict(cs.locator_debris)
        nxt.chips = record
        self._write_ledger(state, cs, record)
        self._channels(cs, record, out, heat, dt_window, mob_rows, frames)

    # ------------------------------------------------------------------ frames and scene
    def _part_rotates(self, frames) -> bool:
        ps = frames.paths[0]
        kind = self.spindles.get(ps.spindle_id, "milling")
        return kind in scene.TURNING_SPINDLES and float(np.max(np.abs(ps.omega))) > self.settings.rotating_part_rad_s

    def _enclosure(self, frames):
        lo, hi = self.geometry.stock_bounds_S(frames.paths[0].workpiece)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pts = frames.o_S_M[-1] + corners @ frames.R_M_S.T
        box_lo, box_hi, _ = scene.enclosure_M(self.machine, pts.min(axis=0), pts.max(axis=0),
                                              self.settings.enclosure_margin_m)
        return box_lo, box_hi

    def _fluid_props(self, temp_K: float):
        if self.fluid is None or self.fluid_kind in ("air", ""):
            return None
        cached = getattr(self, "_fluid_cache", None)
        if cached is not None and cached[0] == temp_K:
            return cached[1]
        try:
            props = params.fluid_properties(self.fluid, temp_K)
        except ValueError as error:
            raise InputError(f"chips: {error}") from None
        self._fluid_cache = (temp_K, props)
        return props

    def _sync_nozzles(self, cs, nozzle_r0: dict, info) -> None:
        ids = sorted(set(cs.nozzle_ids) | set(nozzle_r0))
        old = dict(zip(cs.nozzle_ids, cs.nozzle_blocked.tolist()))
        cs.nozzle_ids = ids
        cs.nozzle_r0_m = np.array([float(nozzle_r0.get(i, 0.0)) for i in ids])
        cs.nozzle_blocked = np.array([int(old.get(i, 0)) for i in ids], np.int64)
        self._nozzle_index = {name: info.nozzle_ids.index(name) for name in info.nozzle_ids}

    # ------------------------------------------------------------------ enthalpy
    def _enthalpy(self, temp: float) -> float:
        return kernels.enthalpy(float(temp), self.model.cp_T_K, self.model.cp_J_kgK, self.model.h_J_kg)

    def _enthalpies(self, temps) -> np.ndarray:
        return params.enthalpy_array(temps, self.model.cp_T_K, self.model.cp_J_kgK, self.model.h_J_kg)

    def _temperature(self, h: float) -> float:
        return kernels.temperature(float(h), self.model.cp_T_K, self.model.cp_J_kgK, self.model.h_J_kg)

    # ------------------------------------------------------------------ pile maintenance
    def _regrid(self, cs, frames, axes, axis, sign, workpiece):
        """The part turned or another workpiece is active: the chips on the old grid move again (at rest) and
        a new grid is built along the new up axis."""
        H = cs.pile
        W = scene.window_layout(frames, H, self.gravity)
        rec, n = np.zeros((int(np.count_nonzero(H.mass_kg > 0)) + 1, abi.MOB_WIDTH)), np.zeros(2, np.int64)
        taken = np.zeros(10)
        cell = float(H.cell_m[0])
        for iu, iv in zip(*np.nonzero(H.mass_kg > 0)):
            top = H.base_m[iu, iv] + H.bulk_m3[iu, iv] / cell ** 2
            kernels.take_fraction(H, int(iu), int(iv), 1.0, taken)
            kernels.write_mob(rec, n, abi.MOB_FLUSH, float(iu), float(iv), taken, 0.0, 0.0, 0.0, (iu + 0.5) * cell,
                              (iv + 0.5) * cell, top, -1.0, frames.t0_s)
        batch = emission.batch_from_mob(rec[:int(n[0])], W, self.rho, self._temperature)
        lo, hi = self.geometry.stock_bounds_S(workpiece)
        cs.pile = scene.new_pile(lo, hi, axes, self.settings.pile_cell_m, self.settings.pile_margin_m)
        cs.pile_axis, cs.pile_sign, cs.pile_workpiece = axis, sign, workpiece
        scene.refresh_base(cs.pile, self.geometry, workpiece, axis, sign)
        cs.pile_version = int(self.geometry.version(workpiece))
        return batch

    def _spill_cells(self, cs, flat: np.ndarray) -> None:
        """Cells that lost the surface under their chips (breakthrough): the chips fall to the pan."""
        H = cs.pile
        taken = np.zeros(10)
        for iu, iv in zip(*np.unravel_index(flat, H.base_m.shape)):
            kernels.take_fraction(H, int(iu), int(iv), 1.0, taken)
            cs.pan[PAN_MASS] += taken[1]
            cs.pan[PAN_ENTHALPY] += taken[2]
            cs.pan[PAN_BULK] += taken[0]
            cs.pan[PAN_LONG] += taken[8]

    def _pan_add(self, cs, out) -> None:
        cs.pan[PAN_MASS] += out[abi.OUT_PAN_MASS_KG]
        cs.pan[PAN_ENTHALPY] += out[abi.OUT_PAN_ENTHALPY_J]
        cs.pan[PAN_BULK] += out[abi.OUT_PAN_BULK_M3]
        cs.pan[PAN_LONG] += out[abi.OUT_PAN_LONG_KG]
        out[abi.OUT_PAN_MASS_KG] = 0.0
        out[abi.OUT_PAN_ENTHALPY_J] = 0.0
        out[abi.OUT_PAN_BULK_M3] = 0.0
        out[abi.OUT_PAN_LONG_KG] = 0.0

    def _clean_part(self, cs) -> None:
        """Between operations the operator cleans part and fixture: every resting chip goes to the pan."""
        H = cs.pile
        nz = np.nonzero(H.mass_kg > 0)
        self._spill_cells(cs, np.ravel_multi_index(nz, H.base_m.shape) if len(nz[0]) else np.zeros(0, np.int64))

    def _shed_rotating_part(self, cs, W, frames, rec, n) -> None:
        """A turning workpiece holds no chips: the chips resting on part cells leave with its surface speed."""
        H = cs.pile
        ps = frames.paths[0]
        omega = float(ps.omega[-1])
        axis_M = frames.R_M_S @ np.array([0.0, 0.0, 1.0])
        taken = np.zeros(10)
        cell = float(H.cell_m[0])
        for iu, iv in zip(*np.nonzero((H.mass_kg > 0) & (H.base_kind == abi.SURF_PART))):
            if n[0] >= rec.shape[0]:
                n[1] = 1
                break
            top = H.base_m[iu, iv] + H.bulk_m3[iu, iv] / cell ** 2
            p_S = scene.cell_points_S(H, np.array([iu]), np.array([iv]), top)[0]
            r_M = frames.R_M_S @ np.array([p_S[0], p_S[1], 0.0])
            v = omega * np.cross(axis_M, r_M)
            kernels.take_fraction(H, int(iu), int(iv), 1.0, taken)
            kernels.write_mob(rec, n, abi.MOB_SWEEP, float(iu), float(iv), taken, float(v[0]), float(v[1]),
                              float(v[2]), (iu + 0.5) * cell, (iv + 0.5) * cell, top, -1.0, frames.t0_s)

    def _flush(self, cs, H, W, coolant, liquid, dt, rec, n) -> None:
        if coolant is None or not getattr(coolant, "on", False) or not np.any(H.flush_Pa > 0):
            return
        fl = self.params.flush
        kind = str(getattr(coolant, "kind", "") or "")
        submerged = kind not in ("mql", "mist", "air", "dry")
        rho_f = float(liquid[0]) if liquid[0] > 0 else scene.air_properties(self.ambient_K)[0]
        cap = 0.0 if submerged else float(fl["capillary_bridge_factor"]) * float(self.params.contact["contact_angle_cos"])
        fp = np.array([fl["skin_friction_coefficient"], fl["critical_shields"], fl["bedload_coefficient"],
                       fl["bedload_exponent"], cap, float(liquid[6]), fl["bed_friction"], self.rho])
        kernels.flush_pile(H, fp, W.R_M_G, rho_f, float(np.linalg.norm(self.gravity)), dt, rec, n)

    # ------------------------------------------------------------------ recut
    def _recut(self, ctx, cs, cutting, frames, W, S, info, rec, n) -> dict:
        """Recut rows of the window: the chips leave the pile as fragments thrown by the flute."""
        H = cs.pile
        cell = float(H.cell_m[0])
        out: dict[int, float] = {}
        requests = []
        for contact in getattr(cutting, "contacts", []) or []:
            rows = np.asarray(getattr(contact, "recut", np.zeros((0, 6))), dtype=float)
            if rows.size == 0:
                continue
            slot = int(contact.tool_slot)
            path = next((p for p in frames.paths if p.tool_slot == slot), frames.paths[0])
            tool = self.tools.tool(slot)
            radius = tool.radius_m if tool is not None else 0.0
            for row in rows:
                tick, mass, x, y = int(row[0]), float(row[3]), float(row[4]), float(row[5])
                k = frames.index(tick)
                tip = path.tip_M[k]
                p_M = np.array([x, y, tip[2]])
                g = W.R_M_G.T @ (p_M - W.grid_origin_M[k])
                iu, iv = int(math.floor(g[0] / cell)), int(math.floor(g[1] / cell))
                a = path.axis_M[k]
                r = (p_M - tip) - np.dot(p_M - tip, a) * a
                rm = float(np.linalg.norm(r))
                v = path.omega[k] * np.cross(a, r * (radius / rm if rm > 0 and radius > 0 else 1.0))
                v = (1.0 + float(self.model.paddle_e[0])) * v + path.feed_M[k]
                inside = 0 <= iu < H.base_m.shape[0] and 0 <= iv < H.base_m.shape[1]
                top = H.base_m[iu, iv] + H.bulk_m3[iu, iv] / cell ** 2 if inside else -np.inf
                w = max(float(top), float(g[2])) + 0.5 * cell
                requests.append([iu, iv, mass, v[0], v[1], v[2], slot, tick * ctx.tick_s, g[0], g[1], w])
        if not requests:
            return out
        req = np.ascontiguousarray(np.array(requests, dtype=float))
        before = int(n[0])
        missing = kernels.recut_take(H, req, rec, n)
        cs.recut_missing_kg += missing
        rows = rec[before:int(n[0])]
        for row in rows:
            slot = int(row[abi.MOB_COLUMNS.index("owner")])
            out[slot] = out.get(slot, 0.0) + float(row[abi.MOB_COLUMNS.index("mass_kg")])
        for slot, mass in sorted(out.items()):
            if 0 <= slot < cs.n_slots:
                cs.trace_ids[slot, T_RECUT] = ctx.trace.record(frames.t_end_s, frames.tick_end, DOMAIN, "chips:recut_rows",
                                                               {"slot": slot, "mass_kg": mass}, [])
        return out

    # ------------------------------------------------------------------ emissions
    def _release_rows(self, ctx, state, cs, prev, frames, tick_end):
        pending = cs.pending
        if len(pending):
            due = pending[:, abi.EM["tick"]] < tick_end
            rows, cs.pending = pending[due], pending[~due]
        else:
            rows = pending
        removal = {int(c.contact): c for c in getattr(prev.removal, "contacts", []) or []}
        contacts = sorted(set(rows[:, abi.EM["contact"]].astype(np.int64).tolist()) | {
            k for k, c in removal.items() if float(c.removed_volume_m3) > 0.0})
        coeffs = getattr(prev.cutting, "coefficients", {}) or {}
        coolant_kind = str(getattr(prev.coolant, "kind", "dry") or "dry") if getattr(prev.coolant, "on", False) else "dry"
        out = []
        kinetic = 0.0
        for contact in contacts:
            crow = rows[rows[:, abi.EM["contact"]].astype(np.int64) == contact]
            rc = removal.get(contact)
            target = self.rho * float(rc.removed_volume_m3) if rc is not None else None
            slot = int(crow[0, abi.EM["tool_slot"]]) if len(crow) else int(rc.tool_slot)
            if target is not None and target <= 0.0 and len(crow) == 0:
                continue
            tool = self.tools.tool(slot) if slot >= 0 else None
            continuous = tool is None or (not tool.rotating) or tool.hole_making
            if len(crow) == 0:
                crow = self._fallback_row(cs, contact, slot, rc, frames, prev)
            if target is None:
                hazards.unscaled(ctx, state, cs, contact, float(np.maximum(crow[:, abi.EM["mass_kg"]], 0).sum()),
                                 frames.tick_end)
                cs.unscaled_kg += float(np.maximum(crow[:, abi.EM["mass_kg"]], 0).sum())
            query = emission.query_for(contact, slot, crow, coeffs.get(slot), coolant_kind, continuous)
            formation = self.formation_model.formation(query)
            confined = float(cs.flute[slot, FL_CONFINED]) if 0 <= slot < cs.n_slots else 0.0
            release, groups = emission.parcels_from_rows(
                contact, slot, crow, target, formation, query, self.settings.particles_per_tooth, ctx.seed,
                cs.group_counter, self.params, self.dust_fraction, confined, self.rho, self._enthalpies, ctx.tick_s)
            cs.group_counter += groups
            self._book_release(ctx, cs, release, frames)
            kinetic += release.kinetic_J
            out.append(release)
        return out, kinetic

    def _fallback_row(self, cs, contact, slot, rc, frames, prev) -> np.ndarray:
        """A contact removed material but sent no emission row (e.g. a rapid into material): one row at the
        contact point, the tool's surface velocity there, the slot's last chip (else an elemental chip)."""
        row = np.zeros((1, abi.EMISSION_WIDTH))
        geom = rc.geometry
        p = np.asarray(geom.point_M_m, dtype=float)
        path = next((ps for ps in frames.paths if ps.tool_slot == slot), frames.paths[0])
        tip, a, omega = path.tip_M[-1], path.axis_M[-1], float(path.omega[-1])
        r = (p - tip) - np.dot(p - tip, a) * a
        v = omega * np.cross(a, r)
        mid_tick = (frames.tick0 + frames.tick_end) // 2
        last = cs.last[slot] if 0 <= slot < cs.n_slots else np.zeros(8)
        size = min(max(float(geom.ae_m), 0.0) or float(cs.pile.cell_m[0]), float(cs.pile.cell_m[0]))
        h = float(last[LS_THICK_M]) if last[LS_VALID] else size
        row[0, abi.EM["tick"]] = mid_tick
        row[0, abi.EM["contact"]] = contact
        row[0, abi.EM["tool_slot"]] = slot
        row[0, abi.EM["mass_kg"]] = self.rho * float(rc.removed_volume_m3)
        row[0, abi.EM["x_M_m"]:abi.EM["z_M_m"] + 1] = p
        row[0, abi.EM["vx_M_m_s"]:abi.EM["vz_M_m_s"] + 1] = v
        row[0, abi.EM["temp_K"]] = float(last[LS_TEMP_K]) if last[LS_VALID] else self.ambient_K
        row[0, abi.EM["h_m"]] = h
        row[0, abi.EM["b_m"]] = float(last[LS_WIDTH_M]) if last[LS_VALID] else size
        row[0, abi.EM["l_m"]] = float(last[LS_LENGTH_M]) if last[LS_VALID] else size
        row[0, abi.EM["v_cut_m_s"]] = float(np.linalg.norm(v))
        row[0, abi.EM["edge_kind"]] = 0
        return row

    def _book_release(self, ctx, cs, release, frames) -> None:
        slot = release.slot
        cs.emitted[0] += release.made_kg
        cs.emitted[1] += release.particle_J + release.dust_J + release.flute_J
        cs.dust[0] += release.dust_kg
        cs.dust[1] += release.dust_J
        if 0 <= slot < cs.n_slots:
            fl = cs.flute[slot]
            fl[FL_MASS_KG] += release.flute_kg
            form = int(release.formation.form_code)
            fl[FL_BULK_M3] += release.flute_kg * float(self.model.form_R[min(max(form, 0), abi.N_FORM - 1)]) / self.rho
            fl[FL_ENTHALPY_J] += release.flute_J
            fl[FL_THICK_MASS] += release.flute_dims[0]
            fl[FL_WIDTH_MASS] += release.flute_dims[1]
            fl[FL_LENGTH_MASS] += release.flute_dims[2]
            fl[FL_SIZE_MASS] += release.flute_kg * float(np.cbrt(6.0 / math.pi * max(release.formation.thickness_m, 1e-9)
                                                                  * max(release.formation.width_m, 1e-9)
                                                                  * max(release.formation.length_m, 1e-9)))
            fl[FL_FORM] = form
            fl[FL_INFLOW_M3_S] += release.flute_kg * float(self.model.form_R[min(max(form, 0), abi.N_FORM - 1)]) \
                / self.rho / max(frames.t_end_s - frames.t0_s, 1e-12)
            hist = cs.recut_hist[slot]
            hist[cs.recut_pos % hist.shape[0], 1] += release.made_kg
            f = release.formation
            last = cs.last[slot]
            changed = int(last[LS_FORM]) != f.form_code or not last[LS_VALID]
            last[LS_FORM] = f.form_code
            last[LS_THICK_M] = f.thickness_m
            last[LS_WIDTH_M] = f.width_m if f.width_m > 0 else release.query.b_m
            last[LS_LENGTH_M] = f.length_m if f.length_m > 0 else release.query.l_m * max(f.chip_ratio, 1e-6)
            last[LS_CURL_M] = f.curl_radius_m
            last[LS_TEMP_K] = f.chip_temp_K
            last[LS_VALID] = 1
            if changed:
                cs.trace_ids[slot, T_FORMATION] = ctx.trace.record(
                    frames.t_end_s, frames.tick_end, DOMAIN, "chips:formation",
                    {"slot": slot, "form_code": f.form_code, "chip_ratio": f.chip_ratio, "shear_angle_rad": f.shear_angle_rad,
                     "thickness_m": f.thickness_m, "length_m": f.length_m, "curl_radius_m": f.curl_radius_m,
                     "chip_temp_K": f.chip_temp_K, "h_m": release.query.h_m, "v_m_s": release.query.v_m_s}, [])

    # ------------------------------------------------------------------ attached chips
    @staticmethod
    def _in_cut(prev) -> set:
        return {int(c.tool_slot) for c in getattr(prev.removal, "contacts", []) or []
                if getattr(c.geometry, "in_cut", False) or float(c.removed_volume_m3) > 0.0}

    def _entering(self, cs, prev) -> set:
        """Tool slots that start an engagement in this window (in cut now, not in the previous window)."""
        return {s for s in self._in_cut(prev) if 0 <= s < cs.n_slots and cs.last[s, LS_IN_CUT] == 0}

    def _nested_at_entry(self, cs, frames, entering: set) -> dict:
        """Chip mass and height lying in the footprint of every entering rotating tool, above its tip."""
        out = {}
        for ps in frames.paths:
            if ps.tool_slot not in entering:
                continue
            tool = self.tools.tool(ps.tool_slot)
            if tool is not None and tool.rotating:
                out[ps.tool_slot] = self._nested_mass(cs, ps, tool)
        return out

    def _release_attached(self, cs, prev, frames, info, recut_kg_slot: dict, entering: set) -> None:
        """Welded chips are recut at the slot's next engagement, wraps fall when their body stops (or the tool
        leaves the spindle), short chips on a running nozzle are blown off."""
        P = cs.particles
        n = int(P.n[0])
        in_cut = self._in_cut(prev)
        active_tools = {ps.tool_slot for ps in frames.paths}
        omega_of = {ps.tool_slot: float(ps.omega[-1]) for ps in frames.paths}
        body_omega = {-1 - ps.workpiece: float(ps.omega[-1]) for ps in frames.paths
                      if self.spindles.get(ps.spindle_id, "milling") in scene.TURNING_SPINDLES}
        thr = self.settings.rotating_part_rad_s
        t0 = frames.t0_s
        for i in range(n):
            st = P.status[i]
            if st == abi.FREE or st == abi.FLYING:
                continue
            owner = int(P.owner[i])
            release = False
            if st == abi.ON_TOOL:
                if owner in entering:
                    release = True
                    P.length_m[i] = min(P.length_m[i], P.width_m[i])
                    recut_kg_slot[owner] = recut_kg_slot.get(owner, 0.0) + float(P.mass_kg[i])
            elif st == abi.WRAPPED:
                if owner < 0:
                    release = abs(body_omega.get(owner, 0.0)) <= thr
                else:
                    release = owner not in active_tools or abs(omega_of.get(owner, 0.0)) <= thr
            elif st == abi.ON_NOZZLE:
                j = self._nozzle_index.get(cs.nozzle_ids[owner] if 0 <= owner < len(cs.nozzle_ids) else "", -1)
                running = j >= 0 and info.scene.sph_active[j] != 0
                hooked = P.length_m[i] >= float(self.model.hook_radii[0]) * (float(info.scene.sph_radius[j]) if j >= 0
                                                                             else 0.0)
                if running and not hooked:
                    release = True
                    P.vel_M_m_s[i] = info.scene.jet_speed[j] * info.scene.jet_dir_M[j]
            if release:
                P.status[i] = abi.FLYING
                P.owner[i] = -1
                P.time_s[i] = t0
                P.bounces[i] = 0
                if st == abi.WRAPPED:
                    P.vel_M_m_s[i] = 0.0
        for ps in frames.paths:
            if 0 <= ps.tool_slot < cs.n_slots:
                cs.last[ps.tool_slot, LS_IN_CUT] = 1.0 if ps.tool_slot in in_cut else 0.0
        for slot in range(cs.n_slots):
            if slot not in active_tools:
                cs.last[slot, LS_IN_CUT] = 0.0

    def _attach_events(self, ctx, cs, ev: EventBuffer, frames, info) -> None:
        """Offsets of newly attached particles from their body, trace events of welds and wraps."""
        P = cs.particles
        for row in ev.rec[:int(ev.n[0])]:
            kind, i, owner, t, mass = int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4])
            pos = row[5:8]
            k = int(min(max(round((t - frames.t0_s) / frames.dt_tick_s), 0), frames.n_t))
            if kind in (abi.EV_WELD, abi.EV_WRAP):
                ref = self._body_point(frames, owner, k)
                cs.attach_rel[i] = pos - ref if ref is not None else 0.0
                if 0 <= owner < cs.n_slots:
                    event = ctx.trace.record(t, int(round(t / ctx.tick_s)), DOMAIN,
                                             "chips:weld" if kind == abi.EV_WELD else "chips:wrap",
                                             {"slot": owner, "mass_kg": mass, "chip_temp_K": float(row[8]),
                                              "form_code": float(row[9])}, [int(cs.trace_ids[owner, T_FORMATION])])
                    cs.trace_ids[owner, T_WELD if kind == abi.EV_WELD else T_WRAP] = event
            elif kind == abi.EV_NOZZLE:
                centre = info.scene.sph_center_M[owner] if owner < len(info.scene.sph_radius) else pos
                cs.attach_rel[i] = pos - centre
                name = info.nozzle_ids[owner] if owner < len(info.nozzle_ids) else ""
                P.owner[i] = cs.nozzle_ids.index(name) if name in cs.nozzle_ids else -1
            elif kind == abi.EV_BOUNCE_LIMIT:
                ctx.trace.record(t, int(round(t / ctx.tick_s)), DOMAIN, "chips:laid_down", {"mass_kg": mass}, [])

    def _body_point(self, frames, owner: int, k: int) -> np.ndarray | None:
        """Reference point of the body a chip is attached to: the tip of the tool slot (None when the tool is
        not in a spindle in this window), the S origin for a turning workpiece."""
        if owner >= 0:
            for ps in frames.paths:
                if ps.tool_slot == owner:
                    return ps.tip_M[k]
            return None
        return frames.o_S_M[k]

    def _update_attached_positions(self, cs, frames, info) -> None:
        P = cs.particles
        n = int(P.n[0])
        for i in np.nonzero((P.status[:n] == abi.ON_TOOL) | (P.status[:n] == abi.WRAPPED))[0]:
            ref = self._body_point(frames, int(P.owner[i]), -1)
            if ref is not None:
                P.pos_M_m[i] = ref + cs.attach_rel[i]
            P.time_s[i] = frames.t_end_s
        for i in np.nonzero(P.status[:n] == abi.ON_NOZZLE)[0]:
            owner = int(P.owner[i])
            name = cs.nozzle_ids[owner] if 0 <= owner < len(cs.nozzle_ids) else ""
            j = self._nozzle_index.get(name, -1)
            if j >= 0:
                P.pos_M_m[i] = info.scene.sph_center_M[j] + cs.attach_rel[i]

    # ------------------------------------------------------------------ flutes
    def _confinement(self, cs, path, tool) -> tuple[float, float]:
        """(enclosure fraction of the ring R + ring_cells cells around the tool tip, evacuation length)."""
        H = cs.pile
        cell = float(H.cell_m[0])
        k = -1
        g_tip = (path.tip_S[k] - H.origin_G_m) @ H.axes_S
        ring_r = tool.radius_m + self.settings.ring_cells * cell
        n_ring = max(16, int(math.ceil(2 * math.pi * ring_r / cell)))
        ang = 2.0 * math.pi * np.arange(n_ring) / n_ring
        iu = np.floor((g_tip[0] + ring_r * np.cos(ang)) / cell).astype(np.int64)
        iv = np.floor((g_tip[1] + ring_r * np.sin(ang)) / cell).astype(np.int64)
        ok = (iu >= 0) & (iu < H.base_m.shape[0]) & (iv >= 0) & (iv < H.base_m.shape[1])
        tops = np.full(n_ring, -np.inf)
        tops[ok] = H.base_m[iu[ok], iv[ok]]
        above = tops > g_tip[2] + cell
        frac = float(np.count_nonzero(above)) / n_ring
        length = float(np.mean(tops[above]) - g_tip[2]) if np.any(above) else 0.0
        return frac, max(length, 0.0)

    def _flutes(self, ctx, cs, prev, frames, H, dt):
        """Flute stores of the tools in the spindles: confinement for the next window's chips, transport,
        packing, release at the hole/slot mouth."""
        fp = self.params.flute
        onset = float(fp["confinement_onset"])
        through_flow = float(getattr(prev.coolant, "through_tool_flow_m3_s", 0.0) or 0.0) \
            if getattr(prev.coolant, "on", False) else 0.0
        out_batches = []
        active = {}
        for ps in frames.paths:
            if ps.tool_slot >= 0:
                active[ps.tool_slot] = ps
        for slot in range(cs.n_slots):
            fl = cs.flute[slot]
            ps = active.get(slot)
            tool = self.tools.tool(slot)
            if tool is None or ps is None:
                length = 0.0
                frac = 0.0
            else:
                frac, length = self._confinement(cs, ps, tool)
            confined = min(max((frac - onset) / (1.0 - onset), 0.0), 1.0) if tool is not None and tool.rotating else 0.0
            if tool is not None and tool.spiral == "point":
                confined = 0.0     # spiral-point taps push the chips ahead of the tap
            fl[FL_CONFINED] = confined
            if tool is None or (fl[FL_MASS_KG] <= 0.0 and fl[FL_INFLOW_M3_S] <= 0.0 and length <= 0.0):
                fl[FL_FILL] = 0.0
                fl[FL_TORQUE_NM] = fl[FL_FORCE_N] = fl[FL_POWER_W] = fl[FL_STRESS_PA] = 0.0
                fl[FL_LENGTH_E_M] = length
                fl[FL_INFLOW_M3_S] = 0.0
                continue
            form = int(fl[FL_FORM])
            ratio = float(self.model.form_R[min(max(form, 0), abi.N_FORM - 1)])
            factor = float(dict(fp["conveying_by_form"]).get(form, 1.0))
            omega = float(ps.omega[-1]) if ps is not None else 0.0
            v_ax = float(np.dot(ps.feed_M[-1], ps.axis_M[-1])) if ps is not None else 0.0
            inflow = fl[FL_INFLOW_M3_S]
            volume0 = fl[FL_BULK_M3] - inflow * dt
            step = flutes.advance(tool, max(volume0, 0.0), inflow, length, omega, v_ax, dt, ratio, factor,
                                  through_flow, self.yield_Pa, fp)
            total = max(volume0, 0.0) + inflow * dt
            out_fraction = step.out_m3 / total if total > 0.0 else 1.0
            out_fraction = min(max(out_fraction, 0.0), 1.0)
            if fl[FL_MASS_KG] > 0.0 and out_fraction > 0.0:
                out_batches.append(self._flute_release(ctx, cs, slot, ps, tool, out_fraction, length, step.v_eff_m_s,
                                                       frames))
            fl[FL_BULK_M3] = fl[FL_BULK_M3] * (1.0 - out_fraction) if out_fraction < 1.0 else 0.0
            fl[FL_FILL] = step.fill
            fl[FL_LENGTH_E_M] = length
            fl[FL_TORQUE_NM] = step.torque_Nm
            fl[FL_FORCE_N] = step.force_N
            fl[FL_POWER_W] = step.power_W
            fl[FL_STRESS_PA] = step.stress_Pa
            fl[FL_V_EFF_M_S] = step.v_eff_m_s
            fl[FL_INFLOW_M3_S] = 0.0
        return emission.concat(out_batches)

    def _flute_release(self, ctx, cs, slot, ps, tool, fraction, length, v_eff, frames):
        """Chips leaving the flutes at the mouth of the hole (tip + L_e along the axis, on the tool radius)."""
        fl = cs.flute[slot]
        mass = fl[FL_MASS_KG] * fraction
        enth = fl[FL_ENTHALPY_J] * fraction
        m_tot = fl[FL_MASS_KG]
        thick = fl[FL_THICK_MASS] / m_tot
        width = fl[FL_WIDTH_MASS] / m_tot
        length_c = fl[FL_LENGTH_MASS] / m_tot
        for col in (FL_MASS_KG, FL_ENTHALPY_J, FL_SIZE_MASS, FL_THICK_MASS, FL_WIDTH_MASS, FL_LENGTH_MASS):
            fl[col] = fl[col] * (1.0 - fraction) if fraction < 1.0 else 0.0
        if ps is None:
            ps = frames.paths[0]
        uni, _ = emission.draws(ctx.seed, emission.FLUTE_STREAM, np.array([cs.next_pid]), slot, 1, 0)
        a = ps.axis_M[-1]
        k = int(np.argmin(np.abs(a)))
        e = np.zeros(3)
        e[k] = 1.0
        e1 = np.cross(a, e)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        ang = 2.0 * math.pi * float(uni[0, 0])
        radial = math.cos(ang) * e1 + math.sin(ang) * e2
        r = tool.radius_m if tool is not None else 0.0
        pos = ps.tip_M[-1] + length * a + (r + max(thick, 0.0)) * radial
        vel = float(ps.omega[-1]) * np.cross(a, r * radial) + v_eff * a + ps.feed_M[-1]
        temp = self._temperature(enth / mass) if mass > 0 else self.ambient_K
        form = int(fl[FL_FORM])
        return emission.make_batch(pos[None, :], vel[None, :], [frames.t_end_s - 0.5 * (frames.t_end_s - frames.t0_s)],
                                   [frames.tick_end], [mass], [temp], [thick], [width], [length_c], [0.0], [form],
                                   [slot])

    def _empty_flutes(self, cs) -> None:
        """End of operation: the tool has left the hole; chips still in its flutes are thrown off at the tool
        change and land in the chip pan."""
        for slot in range(cs.n_slots):
            fl = cs.flute[slot]
            if fl[FL_MASS_KG] <= 0.0:
                continue
            cs.pan[PAN_MASS] += fl[FL_MASS_KG]
            cs.pan[PAN_ENTHALPY] += fl[FL_ENTHALPY_J]
            cs.pan[PAN_BULK] += fl[FL_BULK_M3]
            fl[:] = 0.0

    # ------------------------------------------------------------------ heat, pan, conveyor
    def _surface_heat(self, H, q_surface):
        flat = np.flatnonzero(q_surface)
        if len(flat) == 0:
            return np.zeros((0, 4)), 0.0, 0.0, 0.0
        q = q_surface.ravel()[flat]
        kind = H.base_kind.ravel()[flat]
        part = kind == abi.SURF_PART
        iu, iv = np.unravel_index(flat[part], q_surface.shape)
        rows = np.zeros((len(iu), 4))
        rows[:, :3] = scene.cell_points_S(H, iu, iv, H.base_m[iu, iv])
        rows[:, 3] = q[part]
        return (rows, float(q[part].sum()), float(q[kind == abi.SURF_FIXTURE].sum()),
                float(q[kind == abi.SURF_TABLE].sum()))

    def _pan_heat(self, cs, dt, wet: bool, fluid_temp: float) -> tuple[float, float]:
        """The pan's chips cool toward the coolant (flood running) or the air through the top of their layer."""
        mass = cs.pan[PAN_MASS]
        if mass <= 0.0:
            return 0.0, 0.0
        area = cs.pan[PAN_BULK] / float(self.params.heat["pan_layer_m"])
        h = float(self.params.heat["h_top_wet_W_m2K"] if wet else self.params.heat["h_top_air_W_m2K"])
        sink = fluid_temp if wet else self.ambient_K
        temp = self._temperature(cs.pan[PAN_ENTHALPY] / mass)
        cp = kernels.cp_at(temp, self.model.cp_T_K, self.model.cp_J_kgK)
        t_new = sink + (temp - sink) * math.exp(-h * area * dt / (mass * cp))
        q = cs.pan[PAN_ENTHALPY] - mass * self._enthalpy(t_new)
        cs.pan[PAN_ENTHALPY] -= q
        return (q, 0.0) if wet else (0.0, q)

    def _convey(self, cs, dt) -> float:
        mass = cs.pan[PAN_MASS]
        if mass <= 0.0 or self.conveyor_rate_kg_s <= 0.0:
            return 0.0
        frac = min(self.conveyor_rate_kg_s * dt / mass, 1.0)
        moved = mass * frac if frac < 1.0 else mass
        cs.conveyed[0] += moved
        cs.conveyed[1] += cs.pan[PAN_ENTHALPY] * frac
        if frac >= 1.0:
            cs.pan[:] = 0.0
        else:
            cs.pan *= 1.0 - frac
        return moved

    # ------------------------------------------------------------------ occupancy
    def _occupancy(self, cs, frames, macro_s: float) -> None:
        """The frozen chip-occupancy grid for the next hot loop: the voxels of S (aligned with the pile cells)
        that the tools can sweep before the next grid is published (tool envelope plus the travel at the
        current feed over the lag), holding the pile columns and the slow flying particles there."""
        H = cs.pile
        cell = float(H.cell_m[0])
        P = cs.particles
        lag_s = self.settings.slow_lag_steps * max(macro_s, 1e-12)
        v_slow = cell / lag_s
        los, his = [], []
        for ps in frames.paths:
            tool = self.tools.tool(ps.tool_slot) if ps.tool_slot >= 0 else None
            if tool is None:
                continue
            reach = tool.radius_m + float(np.linalg.norm(ps.feed_M[-1])) * lag_s + cell
            axis_S = frames.R_M_S.T @ ps.axis_M[-1]
            tip = ps.tip_S[-1]
            end = tip + axis_S * (tool.flute_length_m + tool.holder_length_m)
            los.append(np.minimum(tip, end) - reach)
            his.append(np.maximum(tip, end) + reach)
        empty = True
        if los:
            lo = np.min(np.array(los), axis=0)
            hi = np.max(np.array(his), axis=0)
            lattice = H.origin_G_m.copy()
            lattice[cs.pile_axis] = 0.0
            origin = lattice + np.floor((lo - lattice) / cell) * cell
            dims = np.maximum(np.ceil((hi - origin) / cell).astype(np.int64), 1)
            occ = abi.occupancy(int(dims[0]), int(dims[1]), int(dims[2]), origin, cell)
            kernels.fill_occupancy(H, P, np.ascontiguousarray(frames.R_M_S), np.ascontiguousarray(frames.o_S_M[-1]),
                                   v_slow, occ)
            empty = not np.any(occ.mass_kg > 0)
        if empty:
            cs.occ_origin_S_m = np.zeros(3)
            cs.occ_cell_m = cell
            cs.occ_mass_kg = np.zeros((0, 0, 0))
            cs.occ_size_m = np.zeros((0, 0, 0))
            cs.occ_thickness_m = np.zeros((0, 0, 0))
            return
        cs.occ_origin_S_m = occ.origin_S_m
        cs.occ_cell_m = cell
        cs.occ_mass_kg = occ.mass_kg
        cs.occ_size_m = occ.size_m
        cs.occ_thickness_m = occ.thickness_m

    # ------------------------------------------------------------------ findings
    def _findings(self, ctx, state, cs, prev, frames, info, released, recut_kg_slot, nested, dt, tick) -> dict:
        hz = self.params.hazards
        H = cs.pile
        cell = float(H.cell_m[0])
        P = cs.particles
        n = int(P.n[0])
        extras = {"blocked": [], "probe": False, "nest": {}}
        self._has_long = bool(np.any(H.long_kg > 0.0))
        pos = cs.recut_pos % cs.recut_hist.shape[1]
        for slot, mass in recut_kg_slot.items():
            if 0 <= slot < cs.n_slots:
                cs.recut_hist[slot, pos, 0] += mass
        for ps in frames.paths:
            slot = ps.tool_slot
            if not 0 <= slot < cs.n_slots:
                continue
            tool = self.tools.tool(slot)
            fl = cs.flute[slot]
            hazards.packing(ctx, state, cs, hz, slot, tool, float(fl[FL_FILL]),
                            {"packing_torque_Nm": float(fl[FL_TORQUE_NM]), "packing_force_N": float(fl[FL_FORCE_N]),
                             "evacuation_length_m": float(fl[FL_LENGTH_E_M]), "v_eff_m_s": float(fl[FL_V_EFF_M_S])},
                            tick, ps.path)
            made = float(cs.recut_hist[slot, :, 1].sum())
            recut_mass = float(cs.recut_hist[slot, :, 0].sum())
            if made > 0.0:
                hazards.recut(ctx, state, cs, hz, slot, recut_mass / made, recut_mass, tick, ps.path)
            welded = float(P.mass_kg[:n][(P.status[:n] == abi.ON_TOOL) & (P.owner[:n] == slot)].sum())
            if welded > 0.0:
                temps = P.temp_K[:n][(P.status[:n] == abi.ON_TOOL) & (P.owner[:n] == slot)]
                hazards.weld(ctx, state, cs, hz, slot, welded, float(temps.max()), tick, ps.path)
            wrapped = float(P.mass_kg[:n][(P.status[:n] == abi.WRAPPED) & (P.owner[:n] == slot)].sum())
            hazards.wrap(ctx, state, cs, hz, slot, wrapped, tick, ps.path, body=False)
            if tool is not None and tool.rotating:
                frac = self._nest_fraction(cs, ps, tool)
                extras["nest"][slot] = frac
                hazards.nest(ctx, state, cs, hz, slot, frac, tick, ps.path)
                if slot in nested:
                    mass, height = nested[slot]
                    hazards.nested_pocket(ctx, state, cs, hz, slot, mass, height, tick, ps.path)
        body = float(P.mass_kg[:n][(P.status[:n] == abi.WRAPPED) & (P.owner[:n] < 0)].sum())
        hazards.wrap(ctx, state, cs, hz, -1, body, tick, 0, body=True)
        cs.recut_pos += 1
        cs.recut_hist[:, cs.recut_pos % cs.recut_hist.shape[1], :] = 0.0
        # nozzles
        for j, name in enumerate(cs.nozzle_ids):
            sel = (P.status[:n] == abi.ON_NOZZLE) & (P.owner[:n] == j)
            bulk = float(np.dot(P.mass_kg[:n][sel], self.model.form_R[np.clip(P.form[:n][sel], 0, abi.N_FORM - 1)])) \
                / self.rho
            r0 = float(cs.nozzle_r0_m[j]) if j < len(cs.nozzle_r0_m) else 0.0
            blocked = r0 > 0.0 and bulk >= float(hz["nozzle_block_fraction"]) * math.pi * r0 ** 3
            if blocked and not cs.nozzle_blocked[j]:
                hazards.nozzle(ctx, state, name, bulk, tick)
            cs.nozzle_blocked[j] = 1 if blocked else 0
            if blocked:
                extras["blocked"].append(name)
        # probing
        for ps in frames.paths:
            for event in ps.events:
                kind = str(event.get("kind", event.get("type", "")))
                if kind not in ("probe", "probe_trigger", "probe_hit", "tool_setter"):
                    continue
                k = frames.index(int(event.get("tick", frames.tick_end)))
                p_S = np.asarray(event.get("pos_S_m", ps.tip_S[k]), dtype=float)
                g = (p_S - H.origin_G_m) @ H.axes_S
                iu, iv = int(math.floor(g[0] / cell)), int(math.floor(g[1] / cell))
                height = 0.0
                if 0 <= iu < H.base_m.shape[0] and 0 <= iv < H.base_m.shape[1]:
                    height = float(H.bulk_m3[iu, iv] / cell ** 2)
                extras["probe"] = hazards.probe(ctx, state, cs, hz, height, int(event.get("tick", tick)), ps.path) \
                    or extras["probe"]
        # dust, fire, conveyor
        dust_rate = float(sum(r.dust_kg for r in released)) / max(dt, 1e-12)
        hazards.dust(ctx, state, cs, dust_rate, tick)
        if self.ignition_chip_K is not None:
            live = (P.status[:n] != abi.FREE) & (P.mass_kg[:n] > 0)
            t_max = float(P.temp_K[:n][live].max()) if np.any(live) else 0.0
            pile_t = self._pile_max_temp(H)
            hazards.ignition(ctx, state, cs, hz, max(t_max, pile_t), self.ignition_chip_K, tick)
            dry = not getattr(prev.coolant, "on", False) or str(getattr(prev.coolant, "kind", "dry")) in ("dry", "air")
            if dry:
                hazards.fines(ctx, state, cs, hz, float(H.fines_kg.sum()), tick)
        lo, hi = info.scene.box_lo_M, info.scene.box_hi_M
        axis = int(info.scene.floor[0])
        others = [k for k in range(3) if k != axis]
        capacity = float(np.prod(hi[others] - lo[others])) * float(hz["pan_capacity_height_m"])
        hazards.conveyor(ctx, state, cs, float(cs.pan[PAN_BULK]), capacity, tick)
        return extras

    def _column_overlap(self, H, cells_u, cells_v, w_lo, w_hi):
        """Per cell the fraction of its pile column inside [w_lo, w_hi]."""
        cell = float(H.cell_m[0])
        base = H.base_m[cells_u, cells_v]
        h = H.bulk_m3[cells_u, cells_v] / cell ** 2          # the column's height (a column without a base: -inf)
        top = base + h
        ov = np.clip(np.minimum(top, w_hi) - np.maximum(base, w_lo), 0.0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(h > 0, ov / np.where(h > 0, h, 1.0), 0.0)

    def _ring_cells(self, H, center_uv, r_in, r_out):
        cell = float(H.cell_m[0])
        nu, nv = H.base_m.shape
        i0 = max(int(math.floor((center_uv[0] - r_out) / cell)), 0)
        i1 = min(int(math.floor((center_uv[0] + r_out) / cell)), nu - 1)
        j0 = max(int(math.floor((center_uv[1] - r_out) / cell)), 0)
        j1 = min(int(math.floor((center_uv[1] + r_out) / cell)), nv - 1)
        if i1 < i0 or j1 < j0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        iu, iv = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        d = np.hypot((iu + 0.5) * cell - center_uv[0], (iv + 0.5) * cell - center_uv[1])
        sel = (d >= r_in) & (d <= r_out)
        return iu[sel], iv[sel]

    def _nest_fraction(self, cs, ps, tool) -> float:
        H = cs.pile
        if not self._has_long:
            return 0.0
        cell = float(H.cell_m[0])
        g = (ps.tip_S[-1] - H.origin_G_m) @ H.axes_S
        iu, iv = self._ring_cells(H, g[:2], tool.radius_m, 3.0 * tool.radius_m)
        if len(iu) == 0:
            return 0.0
        frac = self._column_overlap(H, iu, iv, g[2], g[2] + tool.flute_length_m)
        mass = H.mass_kg[iu, iv]
        with np.errstate(invalid="ignore", divide="ignore"):
            long_share = np.where(mass > 0, H.long_kg[iu, iv] / np.where(mass > 0, mass, 1.0), 0.0)
        vol = float(np.sum(H.bulk_m3[iu, iv] * frac * long_share))
        space = len(iu) * cell * cell * tool.flute_length_m
        return vol / space if space > 0 else 0.0

    def _nested_mass(self, cs, ps, tool) -> tuple[float, float]:
        H = cs.pile
        cell = float(H.cell_m[0])
        g = (ps.tip_S[-1] - H.origin_G_m) @ H.axes_S
        iu, iv = self._ring_cells(H, g[:2], 0.0, tool.radius_m + cell)
        if len(iu) == 0:
            return 0.0, 0.0
        frac = self._column_overlap(H, iu, iv, g[2], np.inf)
        mass = float(np.sum(H.mass_kg[iu, iv] * frac))
        height = float(np.max(H.bulk_m3[iu, iv] / cell ** 2 * frac)) if len(iu) else 0.0
        return mass, height

    def _pile_max_temp(self, H) -> float:
        """Hottest pile cell (the enthalpy is monotonic in temperature)."""
        sel = H.mass_kg > 0
        if not np.any(sel):
            return 0.0
        return float(self._temperature(float(np.max(H.enthalpy_J[sel] / H.mass_kg[sel]))))

    @staticmethod
    def _operator_cleans(project) -> bool:
        """Whether the operator cleans fixture and locating faces before loading (production.operator.clean_s,
        cnctwin.production.operator: blow chips off fixture and locators; 0 s: no cleaning)."""
        if project is None:
            return True
        from ..production.operator import OperatorModel
        try:
            return OperatorModel.from_project(project, getattr(project, "machine", None)).clean_s > 0.0
        except (TypeError, ValueError, KeyError, AttributeError):
            return True

    def _locator_debris(self, ctx, state, cs, tick) -> dict:
        """Chip (and burr) height on every patch the next setup clamps or locates on. The operator cleans the
        fixture and the locating faces before loading (production.operator.clean_s > 0): the chips there go to the
        pan and only burrs (on the part, not blown off) remain under the locators; with clean_s 0 the chips stay."""
        H = cs.pile
        cell = float(H.cell_m[0])
        hz = self.params.hazards
        out = {}
        e_w = H.axes_S[:, 2]
        burrs = [e for e in (getattr(state.bus_prev.removal, "events", []) or []) if e.get("kind") == "burr"]
        op_index = int(getattr(state.clock, "op_index", 0))
        for locator_id, centre, normal, half in self.locators.faces_S(op_index):
            normal = np.asarray(normal, dtype=float)
            centre = np.asarray(centre, dtype=float)
            height = 0.0
            if float(np.dot(normal, e_w)) > 0.99 and not self.operator_cleans:
                g = (centre - H.origin_G_m) @ H.axes_S
                axis = int(np.argmax(np.abs(normal)))
                others = [k for k in range(3) if k != axis]
                pts_u = np.arange(H.base_m.shape[0])
                pts_v = np.arange(H.base_m.shape[1])
                iu, iv = np.meshgrid(pts_u, pts_v, indexing="ij")
                p = scene.cell_points_S(H, iu.ravel(), iv.ravel())
                inside = (np.abs(p[:, others[0]] - centre[others[0]]) <= half[0]) & \
                         (np.abs(p[:, others[1]] - centre[others[1]]) <= half[1])
                on_face = np.abs(H.base_m.ravel() - g[2]) <= cell
                sel = inside & on_face
                if np.any(sel):
                    height = float(np.max((H.bulk_m3.ravel() / cell ** 2)[sel]))
            for burr in burrs:
                point = np.asarray(burr.get("point_S_m", np.zeros(3)), dtype=float)
                axis = int(np.argmax(np.abs(normal)))
                others = [k for k in range(3) if k != axis]
                if abs(point[axis] - centre[axis]) <= cell and all(abs(point[o] - centre[o]) <= h for o, h in
                                                                    zip(others, half)):
                    height = max(height, float(burr.get("height_m", 0.0)))
            out[str(locator_id)] = height
            hazards.locator(ctx, state, str(locator_id), height, float(hz["locator_debris_tolerance_m"]), tick)
        return out

    # ------------------------------------------------------------------ bus record, ledger, channels
    def _record(self, state, cs, frames, released, heat_rows, dt, heat: dict) -> ChipFeedback:
        H = cs.pile
        cell = float(H.cell_m[0])
        rec = ChipFeedback()
        for item in released:
            f = item.formation
            rec.formation.append(ChipFormationContact(
                contact=item.contact, shear_angle_rad=f.shear_angle_rad, chip_ratio=f.chip_ratio,
                shear_strain=f.shear_strain, strain_rate_1_s=f.strain_rate_1_s,
                chip_flow_angle_rad=f.chip_flow_angle_rad, chip_temp_K=f.chip_temp_K, form_code=int(f.form_code),
                bue_active=bool(f.bue_active), emissions=emission.to_emission_batch(item.batch)))
        for slot in range(cs.n_slots):
            fl = cs.flute[slot]
            if fl[FL_MASS_KG] > 0.0 or fl[FL_LENGTH_E_M] > 0.0 or fl[FL_FILL] > 0.0:
                rec.flute_fill[slot] = float(fl[FL_FILL])
                rec.flute_mass_kg[slot] = float(fl[FL_MASS_KG])
            rec.packing_torque_Nm[slot] = float(fl[FL_TORQUE_NM])
            rec.packing_force_T0_N[slot] = np.array([0.0, 0.0, float(fl[FL_FORCE_N])])
            rec.packing_power_W[slot] = float(fl[FL_POWER_W])
        rec.occupancy_origin_S_m = cs.occ_origin_S_m.copy()
        rec.occupancy_cell_m = float(cs.occ_cell_m)
        rec.occupancy_mass_kg = cs.occ_mass_kg.copy()
        rec.occupancy_size_m = cs.occ_size_m.copy()
        rec.occupancy_thickness_m = cs.occ_thickness_m.copy()
        rec.pile_origin_S_m = H.origin_G_m.copy()
        rec.pile_cell_m = cell
        rec.pile_height_m = H.bulk_m3 / (cell * cell)
        rec.pile_axes_S = H.axes_S.copy()
        rec.pile_base_m = H.base_m.copy()
        rec.pile_mass_kg = H.mass_kg.copy()
        P = cs.particles
        n = int(P.n[0])
        rec.wrap_mass_kg = float(P.mass_kg[:n][P.status[:n] == abi.WRAPPED].sum())
        rec.dust_mass_rate_kg_s = float(sum(r.dust_kg for r in released)) / dt if dt > 0 else 0.0
        rec.heat_to_part = np.asarray(heat_rows, dtype=float).reshape(-1, 4) if len(heat_rows) else np.zeros((0, 4))
        rec.heat_to_fixture_J = float(heat.get("fixture", 0.0))
        rec.heat_to_coolant_J = float(heat.get("coolant", 0.0))
        rec.heat_to_machine_J = float(heat.get("machine", 0.0))
        rec.heat_to_air_J = float(heat.get("air", 0.0))
        rec.part_chip_mass_kg = {int(cs.pile_workpiece): float(H.mass_kg[H.base_kind == abi.SURF_PART].sum())}
        terms = cs.mass_terms()
        rec.in_flight_kg = terms["in_flight_kg"]
        rec.settled_kg = terms["settled_kg"]
        rec.conveyed_kg = terms["conveyed_kg"]
        rec.adhered_kg = terms["adhered_kg"]
        rec.in_flutes_kg = terms["in_flutes_kg"]
        rec.dust_kg = terms["dust_kg"]
        rec.emitted_kg = float(cs.emitted[0])
        rec.conveyor_backlog_kg = float(cs.pan[PAN_MASS])
        rec.booked_until_tick = int(frames.tick_end if frames is not None else cs.window_end_tick)
        return rec

    def _write_ledger(self, state, cs, record=None) -> None:
        if record is not None:
            terms = {"in_flight_kg": record.in_flight_kg, "settled_kg": record.settled_kg,
                     "conveyed_kg": record.conveyed_kg, "adhered_kg": record.adhered_kg,
                     "in_flutes_kg": record.in_flutes_kg, "dust_kg": record.dust_kg}
        else:
            terms = cs.mass_terms()
        ledger = state.ledger
        ledger.chips_in_flight_kg = terms["in_flight_kg"]
        ledger.chips_settled_kg = terms["settled_kg"]
        ledger.chips_conveyed_kg = terms["conveyed_kg"]
        ledger.chips_adhered_kg = terms["adhered_kg"]
        ledger.chips_in_flutes_kg = terms["in_flutes_kg"]
        ledger.dust_kg = terms["dust_kg"]

    def _channels(self, cs, record, out, heat, dt, mob_rows, frames) -> None:
        terms = {"in_flight_kg": record.in_flight_kg, "settled_kg": record.settled_kg,
                 "conveyed_kg": record.conveyed_kg, "adhered_kg": record.adhered_kg,
                 "in_flutes_kg": record.in_flutes_kg, "dust_kg": record.dust_kg}
        P = cs.particles
        n = int(P.n[0])
        live = (P.status[:n] != abi.FREE)
        slot = frames.paths[0].tool_slot
        fl = cs.flute[slot] if 0 <= slot < cs.n_slots else np.zeros(cs.flute.shape[1])
        cell = float(cs.pile.cell_m[0])
        flushed = float(mob_rows[mob_rows[:, 0] == abi.MOB_FLUSH, 3].sum()) if len(mob_rows) else 0.0
        t_live = float(P.temp_K[:n][live].max()) if np.any(live) else 0.0
        cs.channel_values = {
            "chips.in_flight_kg": terms["in_flight_kg"], "chips.settled_kg": terms["settled_kg"],
            "chips.conveyed_kg": terms["conveyed_kg"], "chips.adhered_kg": terms["adhered_kg"],
            "chips.in_flutes_kg": terms["in_flutes_kg"], "chips.dust_kg": terms["dust_kg"],
            "chips.emitted_kg": float(cs.emitted[0]), "chips.particles": float(np.count_nonzero(live)),
            "chips.substeps": float(out[abi.OUT_SUBSTEPS]), "chips.flute_fill": float(fl[FL_FILL]),
            "chips.packing_torque_Nm": float(fl[FL_TORQUE_NM]), "chips.packing_force_N": float(fl[FL_FORCE_N]),
            "chips.packing_power_W": float(fl[FL_POWER_W]),
            "chips.part_chip_mass_kg": float(sum(record.part_chip_mass_kg.values())),
            "chips.pile_max_height_m": float((cs.pile.bulk_m3 / cell ** 2).max()) if cs.pile.bulk_m3.size else 0.0,
            "chips.recut_mass_kg": float(record.recut_mass_kg), "chips.flushed_mass_kg": flushed,
            "chips.heat_to_part_W": heat["part"] / dt if dt > 0 else 0.0,
            "chips.heat_to_coolant_W": heat["coolant"] / dt if dt > 0 else 0.0,
            "chips.heat_to_air_W": heat["air"] / dt if dt > 0 else 0.0,
            "chips.max_temp_K": max(t_live, self._pile_max_temp(cs.pile)),
            "chips.pan_kg": float(cs.pan[PAN_MASS]), "chips.dust_rate_kg_s": float(record.dust_mass_rate_kg_s)}


# ---------------------------------------------------------------------- module helpers

def _emission_buffer(outputs):
    if outputs is None:
        return None
    if isinstance(outputs, abi.ChipsKernelInputs):
        return outputs.emission
    if isinstance(outputs, EventBuffer):
        return outputs
    if isinstance(outputs, dict):
        return _emission_buffer(outputs.get(DOMAIN))
    emission_buf = getattr(outputs, "emission", None)
    if isinstance(emission_buf, EventBuffer):
        return emission_buf
    chips = getattr(outputs, DOMAIN, None)
    if chips is not None:
        return _emission_buffer(chips)
    raise TwinError(f"chips.absorb: cannot find the chips emission buffer in {type(outputs).__name__}")


def _stock_scene(project, lathe: bool) -> BoxScene:
    """The analytic stock box of the project (stock.kind block: size; cylinder/tube: its bounding box about the
    S z axis). On a lathe the bar hangs in the chuck: no table plane under it."""
    stock = getattr(project, "stock", None) if project is not None else None
    size = np.asarray(getattr(stock, "size_m", None) if stock is not None else None, dtype=float) \
        if stock is not None and getattr(stock, "size_m", None) is not None else None
    if size is None or size.size < 2:
        raise InputError("chips: the project gives no stock size and no workpiece geometry is wired")
    kind = str(getattr(stock, "kind", "block") or "block")
    if kind == "block" and size.size >= 3:
        lo, hi = np.zeros(3), size[:3].copy()
    elif kind in ("cylinder", "tube"):
        r = 0.5 * size[0]
        length = size[-1]
        lo, hi = np.array([-r, -r, 0.0]), np.array([r, r, length])
    else:
        lo, hi = np.zeros(3), np.resize(size, 3)
    return BoxScene(lo, hi, table_half_size_m=0.0 if lathe else math.inf)
