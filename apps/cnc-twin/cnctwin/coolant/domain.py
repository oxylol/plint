"""CoolantDomain: the coolant slow domain (docs/models/coolant.md; Domain contract: core/domain.py).

Per macro step it reads ``prev`` (motion: pumps, nozzle settings, poses, spindle; removal: contacts;
cutting: chip thickness and coefficient tables; thermal: wall temperatures and heat to the coolant;
chips: blocked nozzles, the pile, chip heat and mass) and writes only ``nxt.coolant``:

1. supply: the operating point of every running pump with its open outlets (supply.py; cached per
   configuration and fluid-temperature bucket), air blast, MQL, mist;
2. jets in M from the mount frames (jets.py);
3. occlusion jobs (occlusion.py) every ``every_n_macro`` steps and whenever the jet configuration or
   the set of cutting contacts changes; a job started at macro n is applied at n + ``latency_macro``
   (architecture 5.1, k = 1), never earlier;
4. per contact: reach, h_rake / h_flank / h_work with the boiling regime (heat.py), friction factor
   (lubrication.py); h_work_far, wetted cells and flush pressure on the pile grid;
5. the tank (tank.py): heat from thermal, chips and the pumps, losses, concentration, consumption;
6. coolant_shadowing findings with the cause chain jets -> occlusion -> finding.

The domain takes no part in the hot loop (kernel_inputs and absorb return None).
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import numpy as np

from ..core.bus import CoolantFeedback, Jet
from ..core.findings import Category, Severity
from ..units import REFERENCE_TEMP_K, STANDARD_GRAVITY_M_S2
from . import abi
from . import heat as H
from . import occlusion as O
from .channels import MAX_CONTACTS, channels as coolant_channels
from .fluid import air_props, fluid_from_db
from .jets import hole_jets, jet_spec, programmable_aim, through_tool_pose, unit
from .kinematics import ChainMounts, MachineChains, axis_values
from .lubrication import friction as lube_friction
from .protocols import (MachineMeshes, NoMeshes, OccupancyBricks, StockBricks, ToolShapes, TooledShapes,
                        WorkpieceBricks, merge_bricks)
from .schema import settings as read_settings
from .state import CoolantState
from .supply import AEROSOL, GAS, LIQUID, THROUGH_KINDS, SupplyModel
from .tank import Tank

DOMAIN = "coolant"
KIND_PRIORITY = ("high_pressure", "through_spindle", "through_tool", "flood", "mist", "mql", "air")
LATHE_TYPES = ("lathe", "mill_turn", "swiss")


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        try:
            value = getattr(obj, name)
        except AttributeError:
            value = default
    return default if value is None else value


def _num(obj, name, default=None):
    value = _get(obj, name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class CoolantDomain:
    """Supply, jets, occlusion, heat transfer, lubrication and the tank (see the module docstring)."""

    name = DOMAIN

    def __init__(self, shapes: ToolShapes | None = None, bricks: WorkpieceBricks | None = None, meshes=None,
                 mounts=None):
        self.shapes = shapes
        self.bricks = bricks
        self.meshes = meshes
        self.mounts = mounts
        self._supply_cache: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------ Domain interface
    def channels(self):
        return coolant_channels()

    def kernel_inputs(self, ctx, state, bus):
        return None

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        return None

    def init(self, ctx, state) -> None:
        project = ctx.project
        physics = _get(project, "physics")
        self.settings = read_settings(_get(physics, "coolant"))
        st = self.settings
        self.machine = _get(project, "machine")
        cfg = _get(project, "coolant")
        self.fluid = fluid_from_db(_get(cfg, "fluid"))
        self.supply = SupplyModel(self.machine, self.fluid, st.solver_rel_tol, st.solver_max_iter)
        self.chains = MachineChains(self.machine)
        if self.mounts is None:
            self.mounts = ChainMounts(self.chains)
        env = _get(project, "environment")
        self.ambient_K = _num(env, "ambient_K", REFERENCE_TEMP_K)
        start_K = _num(env, "coolant_K", self.ambient_K)
        self.idle_s = _num(env, "time_between_ops_s", 0.0)
        material = _get(project, "material")
        self.work_family = str(_get(material, "family", "") or "").lower()
        self.gravity_M = self._gravity()
        notes = [f"fluid {self.fluid.id}: {e}" for e in self.fluid.estimated]
        if self.shapes is None:
            self.shapes, note = _default_shapes(project, state, st)
            if note:
                notes.append(note)
        if self.bricks is None:
            self.bricks, note = _default_bricks(project, st)
            if note:
                notes.append(note)
        if self.meshes is None:
            try:
                self.meshes = MachineMeshes(self.machine)
            except (ValueError, KeyError, TypeError) as error:
                self.meshes = NoMeshes()
                notes.append(f"machine geometry not usable for occlusion ({error})")
        self.mesh_list = list(self.meshes.meshes())
        enc = _get(self.machine, "enclosure")
        box = _get(enc, "box_m")
        if box is not None and np.asarray(box, dtype=float).shape == (2, 3):
            arr = np.asarray(box, dtype=float)
            self.box_lo, self.box_hi = np.minimum(arr[0], arr[1]), np.maximum(arr[0], arr[1])
        else:
            self.box_lo, self.box_hi = np.full(3, -10.0), np.full(3, 10.0)
            notes.append("machine enclosure not given: jets traced in a 20 m box")
        fill = _num(cfg, "tank_fill_fraction", None)
        if fill is None:
            from .tank import P as tank_params
            fill = tank_params()["default_fill_fraction"]
        conc = _num(cfg, "concentration_frac", None)
        if conc is None:
            conc = self.fluid.concentration_frac
        chiller = self.supply.chiller
        capacity = _num(chiller, "capacity_W", 0.0) if chiller is not None and not isinstance(
            getattr(chiller, "value", None), str) else 0.0
        setpoint = _num(chiller, "setpoint_K", math.inf) if capacity else math.inf
        if chiller is not None and not capacity:
            notes.append("machine chiller entry gives no capacity_kW / setpoint: no tank chiller")
        tank = Tank.create(self.fluid, self.supply.tank_m3, fill, start_K, self.ambient_K, conc, capacity or 0.0,
                           setpoint)
        cs = CoolantState(tank=tank, part_index=state.clock.part_index, op_index=state.clock.op_index, notes=notes)
        cs.tank_min_K = cs.tank_max_K = tank.temp_K if tank.liquid else self.ambient_K
        state.coolant = cs
        for note in notes:
            ctx.trace.record(0.0, 0, DOMAIN, "coolant:fallback", {}, [])
        record = self._off_record(cs)
        state.bus_prev.coolant = record
        state.bus_next.coolant = copy.deepcopy(record)

    # ------------------------------------------------------------------ the step
    def macro_step(self, ctx, state, prev, nxt) -> None:
        cs: CoolantState = state.coolant
        st = self.settings
        cs.macro += 1
        dt = ctx.macro_s
        self._dt = dt
        tick = int(prev.motion.tick)
        self._part_op(state, cs)
        paths = list(prev.motion.paths)
        pumps_on, air_on, nozzle_cmd = self._switches(paths)
        spindle_path = self._spindle_path(paths)
        tool = self.shapes.shape(int(spindle_path.tool_slot)) if spindle_path is not None and \
            spindle_path.tool_slot >= 0 else None
        through_ok = tool is not None and tool.rotating
        blocked = self._blocked(prev)
        temp_K = self._supply_temp(cs)
        sup = self._solve(pumps_on, air_on, blocked, tool if through_ok else None, temp_K)
        on = bool(sup.outlets)
        self._supply_events(ctx, state, cs, sup, pumps_on, air_on, tick, tool)
        poses = self.chains.poses(axis_values(paths))
        positions = self._nozzle_positions(ctx, cs, nozzle_cmd, paths, tick)
        rows = self._tool_rows(paths)
        bus_jets = self._jets(sup, paths, poses, positions, tool, rows, None)
        contacts = self._contacts(prev, paths)
        # ---------------------------------------------------------- occlusion jobs (latency k)
        if not on:
            cs.applied = None
            cs.pending = None
            cs.config_key = ""          # switching on again starts a job at once
        if cs.pending is not None and cs.pending.apply_macro <= cs.macro:
            cs.applied, cs.pending = cs.pending, None
        job_key = repr((sorted(pumps_on.items()), air_on, sorted(positions.items()), blocked,
                        sorted((r[0], r[1]) for r in rows), sorted(c["key"] for c in contacts)))
        if on and (cs.macro % st.every_n_macro == 0 or job_key != cs.config_key):
            cs.config_key = job_key
            zones = self._zones(contacts, paths, rows)
            scene = self._scene(prev, paths, poses, rows, temp_K)
            job_jets = self._jets(sup, paths, poses, positions, tool, rows, st.through_tool_angles)
            result = O.run(job_jets, zones, scene, st, tick, cs.macro, cs.macro + st.latency_macro)
            self._job_events(ctx, cs, result, job_jets, zones, tick)
            if st.latency_macro == 0:
                cs.applied = result
            else:
                cs.pending = result
        applied = cs.applied if on else None
        # ---------------------------------------------------------- record
        rec = CoolantFeedback()
        rec.on = on
        rec.kind = self._kind(sup) if on else "dry"
        rec.fluid_temp_K = self._fluid_temp(cs, sup, temp_K)
        rec.through_tool_flow_m3_s = sup.through_flow_m3_s
        rec.through_tool_exit_speed_m_s = sup.through_speed_m_s
        rec.jets = self._bus_jets(bus_jets, applied)
        rec.pump_flow_m3_s = {k: c.flow_m3_s for k, c in sorted(sup.circuits.items())}
        rec.pump_pressure_Pa = {k: c.pressure_Pa for k, c in sorted(sup.circuits.items())}
        rec.pump_power_W = {k: c.shaft_power_W for k, c in sorted(sup.circuits.items())}
        rec.nozzle_flow_m3_s = {o.jet_id: o.flow_m3_s for o in sorted(sup.outlets, key=lambda o: o.jet_id)}
        rec.air_flow_m3_s = sup.air_free_m3_s
        rec.mql_oil_m3_s = sup.mql_oil_m3_s
        if applied is not None:
            rec.occlusion_tick = applied.tick
            rec.shadows = self._global_shadows(applied, contacts)
            if applied.grid_cell_m > 0.0:
                rec.wetted_origin_S_m = applied.grid_origin_S.copy()
                rec.wetted_axes_S = applied.grid_axes_S.copy()
                rec.wetted_cell_m = applied.grid_cell_m
                rec.wetted = applied.wetted.copy()
                rec.flush_pressure_Pa = applied.flush_Pa.copy()
        self._contact_outputs(ctx, state, cs, prev, rec, contacts, applied, sup, paths, tick)
        rec.h_work_far_W_m2K = self._far_h(prev, applied, rec.fluid_temp_K) if on else 0.0
        self._tank(cs, prev, sup, rec, dt)
        nxt.coolant = rec

    # ------------------------------------------------------------------ switches and supply
    def _switches(self, paths):
        pumps: dict[str, bool] = {}
        air = False
        nozzles: dict[str, float] = {}
        for path in paths:
            for pid, value in sorted((getattr(path, "coolant_pumps", None) or {}).items()):
                if pid in ("air", "air_blast"):
                    air = air or bool(value)
                elif pid in self.supply.pumps:
                    pumps[pid] = pumps.get(pid, False) or bool(value)
            for nid, value in sorted((getattr(path, "nozzle_settings", None) or {}).items()):
                nozzles[str(nid)] = float(value)
        return dict(sorted(pumps.items())), air, nozzles

    def _spindle_path(self, paths):
        for path in paths:
            slot = int(getattr(path, "tool_slot", -1))
            if slot >= 0:
                shape = self.shapes.shape(slot)
                if shape is not None and shape.rotating:
                    return path
        return paths[0] if paths else None

    def _blocked(self, prev) -> tuple:
        """Machine nozzles chips block (chips name the jet ids of the bus: '<nozzle>#<k>' for one orifice of
        a multi-orifice nozzle, which then blocks the nozzle; through-tool jets cannot be blocked so)."""
        names = {n.id for n in self.supply.nozzles if n.kind not in THROUGH_KINDS}
        reported = {str(b).split("#")[0] for b in (getattr(prev.chips, "blocked_nozzles", None) or [])}
        return tuple(sorted(reported & names))

    def _supply_temp(self, cs) -> float:
        tank = cs.tank
        if not tank.liquid:
            return self.ambient_K
        step = self.settings.supply_temp_step_dK
        return round(tank.temp_K / step) * step

    def _solve(self, pumps_on, air_on, blocked, tool, temp_K):
        key = (tuple(sorted(pumps_on.items())), air_on, blocked, None if tool is None else tool.slot, temp_K)
        cached = self._supply_cache.get(key)
        if cached is not None:
            self._supply_cache.move_to_end(key)
            return cached
        result = self.supply.solve(pumps_on, air_on, blocked, tool, temp_K, self.ambient_K)
        self._supply_cache[key] = result
        while len(self._supply_cache) > 64:
            self._supply_cache.popitem(last=False)
        return result

    def _fluid_temp(self, cs, sup, temp_K) -> float:
        """Supply temperature at the outlets: the tank plus the pumps' heating of the delivered flow."""
        tank = cs.tank
        if not tank.liquid or sup.pumped_m3_s <= 0.0:
            return tank.temp_K if tank.liquid else self.ambient_K
        props = self.fluid.props(tank.temp_K)
        return tank.temp_K + sup.shaft_power_W / (props.rho_kg_m3 * props.cp_J_kgK * sup.pumped_m3_s)

    def _kind(self, sup) -> str:
        kinds = {o.kind for o in sup.outlets}
        for k in KIND_PRIORITY:
            if k in kinds:
                return k
        return "dry"

    # ------------------------------------------------------------------ frames, jets
    def _gravity(self) -> np.ndarray:
        kin = _get(self.machine, "kinematics")
        direction = _get(kin, "gravity_direction")
        if direction is not None:
            d = np.asarray([float(v) for v in direction], dtype=float)
        else:
            mtype = str(_get(self.machine, "type", "") or "")
            d = np.array([-1.0, 0.0, 0.0]) if mtype in LATHE_TYPES else np.array([0.0, 0.0, -1.0])
        return STANDARD_GRAVITY_M_S2 * d / np.linalg.norm(d)

    def _nozzle_positions(self, ctx, cs, nozzle_cmd, paths, tick) -> dict:
        """Programmable nozzle positions: motion's nozzle settings, else the operation's per-tool table,
        else home (1)."""
        out = {}
        ops = _get(getattr(ctx, "project", None), "operations") or []
        op = ops[cs.op_index] if 0 <= cs.op_index < len(ops) else None
        table = _get(op, "nozzle_positions") or {}
        number = None
        sp = self._spindle_path(paths)
        if sp is not None and sp.tool_slot >= 0:
            shape = self.shapes.shape(int(sp.tool_slot))
            number = shape.number if shape is not None else None
        for nozzle in self.supply.programmable():
            if nozzle.id in nozzle_cmd:
                pos = int(round(nozzle_cmd[nozzle.id]))
            else:
                pos = 1
                for k, v in (table.items() if hasattr(table, "items") else []):
                    if number is not None and int(float(k)) == int(number):
                        pos = int(float(v))
            pos = min(max(pos, 1), max(nozzle.programmable_positions, 1))
            out[nozzle.id] = pos
            if cs.nozzle_positions.get(nozzle.id) != pos:
                cs.nozzle_positions[nozzle.id] = pos
                cs.trace_ids[f"nozzle:{nozzle.id}"] = ctx.trace.record(
                    tick * ctx.tick_s, tick, DOMAIN, "coolant:nozzle_position",
                    {"position": float(pos), "step_rad": nozzle.programmable_step_rad}, [])
        return dict(sorted(out.items()))

    def _tool_rows(self, paths) -> list:
        """(path index, tool slot, shape, tip_M, R_M_T) of every path with a tool."""
        rows = []
        for i, path in enumerate(paths):
            slot = int(getattr(path, "tool_slot", -1))
            if slot < 0 or not len(getattr(path, "tip_M_m", [])):
                continue
            shape = self.shapes.shape(slot)
            if shape is None:
                continue
            tip, R = through_tool_pose(path)
            rows.append((i, slot, shape, tip, R))
        return rows

    @staticmethod
    def _rev_rows(rows) -> list:
        """The rows that become Revolved occluders (rotating tools with an envelope), in row order."""
        return [r for r in rows if r[2].rotating and len(r[2].profile_r_m) >= 2]

    def _mount(self, link, paths, poses, rows):
        chain = self.chains.chain_of(link)
        path_index = self.chains.path_of_chain(chain) if chain is not None and "tool_chain" in chain else 0
        path = paths[path_index] if path_index < len(paths) else (paths[0] if paths else None)
        gauge = None
        for i, slot, shape, tip, R in rows:
            if i == path_index:
                gauge = shape.z_gauge_m
        return self.mounts.pose_M(link, path, gauge, poses)

    def _jets(self, sup, paths, poses, positions, tool, rows, angles) -> list:
        """JetSpecs of the outlets; ``angles``: number of spindle angles of through-tool jets (None: the
        actual end-of-step angle only)."""
        jets = []
        fluid = self.fluid
        liquid = fluid.props(self._supply_temp_from(sup)) if not (fluid.is_gas or fluid.is_mql or fluid.dry) else None
        tool_row = -1
        tip = R = None
        for r, (i, slot, shape, tip_i, R_i) in enumerate(self._rev_rows(rows)):
            if tool is not None and slot == tool.slot:
                tool_row, tip, R = r, tip_i, R_i
        for outlet in sup.outlets:
            if outlet.through_tool:
                if tool is None or tip is None:
                    continue
                n = 1 if angles is None else int(angles)
                theta = np.zeros(1) if angles is None else 2.0 * math.pi * np.arange(n) / n
                jets.extend(hole_jets(outlet, tool, tip, R, theta, liquid, tool_row))
                continue
            nozzle = self.supply.nozzle(outlet.nozzle_id)
            if nozzle is None:
                continue
            T = self._mount(nozzle.link, paths, poses, rows)
            if T is None:
                continue
            aim = np.asarray(nozzle.aim, dtype=float)
            if nozzle.programmable_positions > 0:
                aim = programmable_aim(aim, nozzle.position_m, positions.get(nozzle.id, 1),
                                       nozzle.programmable_step_rad)
            origin = T[:3, :3] @ np.asarray(nozzle.position_m, dtype=float) + T[:3, 3]
            direction = T[:3, :3] @ unit(aim)
            jets.append(jet_spec(outlet, origin, direction, liquid, nozzle.spread_rad))
        return jets

    def _supply_temp_from(self, sup) -> float:
        for o in sup.outlets:
            if o.phase == LIQUID and o.temp_K > 0.0:
                return o.temp_K
        return self.ambient_K

    def _bus_jets(self, jets, applied) -> list:
        out = []
        for jet in sorted(jets, key=lambda j: j.jet_id):
            path = applied.jet_paths.get(jet.jet_id) if applied is not None else None
            rows = np.asarray(path, dtype=float).reshape(-1, 5) if path is not None else np.zeros((0, 5))
            out.append(Jet(nozzle_id=jet.jet_id, kind=jet.kind, origin_M_m=jet.origin_M.copy(),
                           direction_M=jet.direction_M.copy(), speed_m_s=jet.speed_m_s, flow_m3_s=jet.flow_m3_s,
                           half_angle_rad=jet.half_angle_rad,
                           reach_fraction=float(applied.jet_reach.get(jet.jet_id, 0.0)) if applied else 0.0,
                           diameter_m=jet.diameter_m, breakup_m=jet.breakup_m, path_M_m=rows[:, :3].copy(),
                           path_q_Pa=self._dynamic_pressure(jet, rows[:, 3], rows[:, 4]),
                           blocked_by=applied.jet_blocked.get(jet.jet_id, "") if applied else "",
                           pump_id=self._pump_of(jet), through_tool=jet.through_tool))
        return out

    @staticmethod
    def _dynamic_pressure(jet, s, v) -> np.ndarray:
        """1/2 rho v^2 along the jet axis: the coherent liquid column up to break-up, then the spray, whose
        liquid spreads over the widening cone (mass conservation, spray half-angle); gas jets at their
        expanded density with the centreline decay already in v."""
        q = 0.5 * jet.rho_kg_m3 * v * v
        if jet.phase == LIQUID and math.isfinite(jet.breakup_m) and len(s):
            from .jets import P as jet_params
            spread = math.tan(jet_params()["spray_half_angle_rad"])
            beyond = np.maximum(s - jet.breakup_m, 0.0)
            d = max(jet.diameter_m, 1e-9)
            q = q * (d / (d + 2.0 * beyond * spread)) ** 2
        return q

    def _pump_of(self, jet) -> str:
        nozzle = self.supply.nozzle(jet.nozzle_id)
        if nozzle is not None:
            return nozzle.pump
        base = jet.jet_id.split("#")[0]
        return base if base in self.supply.pumps else ""

    # ------------------------------------------------------------------ contacts and zones
    def _contacts(self, prev, paths) -> list:
        cutting = {int(c.contact): c for c in getattr(prev.cutting, "contacts", [])}
        out = []
        for rc in sorted(getattr(prev.removal, "contacts", []), key=lambda c: int(c.contact)):
            g = rc.geometry
            cut = cutting.get(int(rc.contact))
            in_cut = bool(g.in_cut) or (cut is not None and cut.in_cut_s > 0.0)
            if not in_cut or rc.tool_slot < 0:
                continue
            out.append({"key": int(rc.contact), "slot": int(rc.tool_slot), "workpiece": int(rc.workpiece),
                        "path": int(rc.path), "geometry": g, "cutting": cut})
        return out

    def _zones(self, contacts, paths, rows) -> list:
        from .occlusion import P as occ
        p = occ()
        hp = H.P()
        zones = []
        for c in contacts:
            shape = self.shapes.shape(c["slot"])
            radius = shape.radius_m if shape is not None else 0.0
            g = c["geometry"]
            size = max(radius, 0.5 * float(g.ap_m), 0.5 * float(g.ae_m), p["zone_min_radius_m"])
            centre = np.asarray(g.point_M_m, dtype=float)
            if not np.any(centre) and shape is not None:
                centre = self._tool_zone_centre(c["slot"], rows, radius, p)
            if centre is None:
                continue
            zones.append(O.Zone(c["key"], c["slot"], c["workpiece"], centre, p["zone_radius_factor"] * size,
                                hp["heat_zone_radius_factor"] * size, True))
        for i, slot, shape, tip, R in rows:
            centre = tip + R[:, 2] * p["tool_zone_height_ratio"] * shape.radius_m
            size = max(shape.radius_m, p["zone_min_radius_m"])
            zones.append(O.Zone(-1 - slot, slot, 0, centre, p["zone_radius_factor"] * size,
                                hp["heat_zone_radius_factor"] * size, False))
        return zones

    def _tool_zone_centre(self, slot, rows, radius, p):
        for i, s, shape, tip, R in rows:
            if s == slot:
                return tip + R[:, 2] * p["tool_zone_height_ratio"] * radius
        return None

    # ------------------------------------------------------------------ scene
    def _s_pose(self, path) -> tuple[np.ndarray, np.ndarray]:
        pose_M_T = np.asarray(path.pose_M_T, dtype=float)
        pose_S_T = np.asarray(path.pose_S_T, dtype=float)
        R_M_S = pose_M_T[:3, :3] @ pose_S_T[:3, :3].T
        if len(path.tip_M_m) and len(path.tip_S_m):
            o = np.asarray(path.tip_M_m[-1], dtype=float) - R_M_S @ np.asarray(path.tip_S_m[-1], dtype=float)
        else:
            o = pose_M_T[:3, 3] - R_M_S @ pose_S_T[:3, 3]
        return R_M_S, o

    def _scene(self, prev, paths, poses, rows, temp_K) -> O.Scene:
        st = self.settings
        rev_rows = self._rev_rows(rows)
        n_max = max([len(r[2].profile_r_m) for r in rev_rows] + [2])
        nt = len(rev_rows)
        rev = abi.Revolved(np.zeros((nt, 3)), np.zeros((nt, 3, 3)), np.zeros(nt), np.ones(nt), np.zeros(nt, np.int64),
                           np.zeros((nt, n_max)), np.zeros((nt, n_max), np.int64), np.zeros(nt))
        names = []
        for r, (i, slot, shape, tip, R) in enumerate(rev_rows):
            n = len(shape.profile_r_m)
            rev.tip_M[r] = tip
            rev.R_M_T[r] = R
            rev.z0[r] = shape.profile_z0_m
            rev.dz[r] = shape.profile_dz_m
            rev.n_nodes[r] = n
            rev.radius[r, :n] = shape.profile_r_m
            rev.part[r, :n] = shape.profile_part
            rev.r_max[r] = float(np.max(shape.profile_r_m))
            names.append((slot, shape.number))
        box_rows = [(i, slot, shape, tip, R) for i, slot, shape, tip, R in rows if not shape.rotating]
        nb = len(box_rows)
        boxes = abi.Boxes(np.zeros((nb, 3)), np.zeros((nb, 3, 3)), np.zeros((nb, 3)), np.zeros(nb, np.int64))
        box_names = []
        for k, (i, slot, shape, tip, R) in enumerate(box_rows):
            boxes.centre_M[k] = tip + R @ shape.box_centre_T_m
            boxes.R_M_B[k] = R @ shape.box_axes_T
            boxes.half[k] = shape.box_half_m
            boxes.owner[k] = slot
            box_names.append(f"tool T{shape.number}")
        sets = []
        seen = set()
        pose_by_wp = {}
        for path in paths:
            if not len(getattr(path, "tip_M_m", [])):
                continue
            wp = int(getattr(path, "workpiece", 0))
            if wp in seen:
                continue
            seen.add(wp)
            pose_by_wp[wp] = self._s_pose(path)
            bs = self.bricks.bricks(wp)
            if bs is not None:
                sets.append((bs, *pose_by_wp[wp]))
        bricks = merge_bricks(sets)
        R_M_S, o_M_S = pose_by_wp.get(int(getattr(paths[0], "workpiece", 0)), (np.eye(3), np.zeros(3))) if paths \
            else (np.eye(3), np.zeros(3))
        hf, grid = self._pile(prev, R_M_S, o_M_S, paths)
        meshes = []
        for m in self.mesh_list:
            if m.link and m.link not in poses:
                continue
            T = poses.get(m.link, np.eye(4)) if m.link else np.eye(4)
            meshes.append((m.name, m.tree, T[:3, :3].copy(), T[:3, 3].copy()))
        air = air_props(self.ambient_K)
        prm = O.trace_params(self.gravity_M, air.rho_kg_m3, air.mu_Pa_s, st, self.box_lo, self.box_hi)
        liquid = self.fluid.props(temp_K) if not (self.fluid.is_gas or self.fluid.dry or self.fluid.is_mql) else None
        mist_wets = not (self.fluid.is_mql or self.fluid.is_gas or self.fluid.dry)
        return O.Scene(rev, names, boxes, box_names, bricks, hf, meshes, prm, grid, R_M_S, o_M_S,
                       liquid.rho_kg_m3 if liquid else 0.0, liquid.nu_m2_s if liquid else 1.0, air.rho_kg_m3,
                       air.nu_m2_s, mist_wets)

    def _pile(self, prev, R_M_S, o_M_S, paths):
        chips = prev.chips
        height = np.asarray(getattr(chips, "pile_height_m", np.zeros((0, 0))), dtype=float)
        base = np.asarray(getattr(chips, "pile_base_m", np.zeros((0, 0))), dtype=float)
        cell = float(getattr(chips, "pile_cell_m", 0.0) or 0.0)
        if height.ndim == 2 and height.size and cell > 0.0:
            if base.shape != height.shape:
                base = np.zeros_like(height)
            top = np.where(np.isfinite(base), base + np.maximum(height, 0.0), -np.inf)
            axes = np.asarray(getattr(chips, "pile_axes_S", np.eye(3)), dtype=float)
            origin = np.asarray(chips.pile_origin_S_m, dtype=float)
            hf = abi.HeightField(abi.contiguous(origin), abi.contiguous(axes), np.array([cell]),
                                 abi.contiguous(base), abi.contiguous(top), abi.contiguous(R_M_S), abi.contiguous(o_M_S),
                                 np.ones(1, np.int64))
            return hf, O.FilmGrid(origin.copy(), axes.copy(), cell, base.copy(), top.copy())
        grid = self._own_grid(R_M_S, paths)
        return abi.empty_heightfield()._replace(R_M_S=abi.contiguous(R_M_S), o_M_S=abi.contiguous(o_M_S)), grid

    def _own_grid(self, R_M_S, paths):
        """A grid over the stock footprint (axes like the chips pile grid: e_w the S axis closest to up),
        surface = top of the material per column from the workpiece bricks (a vertical ray per cell)."""
        wp = int(getattr(paths[0], "workpiece", 0)) if paths else 0
        up_S = -(R_M_S.T @ self.gravity_M)
        axis = int(np.argmax(np.abs(up_S)))
        sign = 1.0 if up_S[axis] >= 0 else -1.0
        version = getattr(self.bricks, "version", None)
        key = (wp, int(version(wp)) if callable(version) else -1, axis, sign)
        cached = getattr(self, "_grid_cache", None)
        if cached is not None and cached[0] == key and key[1] >= 0:
            return cached[1]
        bounds = self.bricks.bounds_S(wp)
        bs = self.bricks.bricks(wp)
        if bounds is None or bs is None:
            return None
        lo, hi = (np.asarray(b, dtype=float) for b in bounds)
        a, b = (k for k in range(3) if k != axis)
        e = np.eye(3)
        axes = np.stack([e[a], e[b], sign * e[axis]], axis=1)
        cell = self.settings.wetted_cell_m
        nu = max(int(math.ceil((hi[a] - lo[a]) / cell)), 1)
        nv = max(int(math.ceil((hi[b] - lo[b]) / cell)), 1)
        origin = np.zeros(3)
        origin[a], origin[b] = lo[a], lo[b]
        G = merge_bricks([(bs, np.eye(3), np.zeros(3))])
        base = np.full((nu, nv), -np.inf)
        scratch = np.zeros(8)
        from .kernels import brick_hit
        start = np.zeros(3)
        d = np.zeros(3)
        span = (hi[axis] - lo[axis]) + 2.0
        for iu in range(nu):
            for iv in range(nv):
                start[a] = lo[a] + (iu + 0.5) * cell
                start[b] = lo[b] + (iv + 0.5) * cell
                start[axis] = (hi[axis] + 1.0) if sign > 0 else (lo[axis] - 1.0)
                d[:] = 0.0
                d[axis] = -sign * span
                t = brick_hit(start, d, G, 0, scratch)
                if t < 2.0:
                    base[iu, iv] = sign * (start[axis] + t * d[axis])
        grid = O.FilmGrid(origin, axes, cell, base, base.copy())
        self._grid_cache = (key, grid)
        return grid

    # ------------------------------------------------------------------ per contact
    def _contact_outputs(self, ctx, state, cs, prev, rec, contacts, applied, sup, paths, tick) -> None:
        st = self.settings
        T_f = rec.fluid_temp_K
        for c in contacts:
            key = c["key"]
            slot = c["slot"]
            shape = self.shapes.shape(slot)
            zr = None
            if applied is not None:
                zr = applied.zones.get(key)
                if zr is None or zr.tool_slot != slot:      # a contact index reused after a tool change
                    zr = applied.zones.get(-1 - slot)
            reach = min(max(zr.reach * st.reach_scale, 0.0), 1.0) if zr is not None else 0.0
            if not rec.on:
                rec.friction_factor[key] = 1.0
                continue
            rec.reach_fraction[key] = reach
            rec.aimed_flow_kg_s[key] = zr.aimed_kg_s if zr is not None else 0.0
            rec.arriving_flow_kg_s[key] = zr.arrived_kg_s if zr is not None else 0.0
            if zr is not None and zr.aimed_kg_s > 0.0:
                rec.contact_shadows[key] = {k: v / zr.aimed_kg_s for k, v in sorted(zr.shadows.items())}
            walls = self._walls(prev, c, T_f)
            h_rake, h_flank, h_work, regimes, film = self._contact_heat(zr, shape, c, walls, T_f)
            rec.h_rake_W_m2K[key] = h_rake
            rec.h_flank_W_m2K[key] = h_flank
            rec.h_work_W_m2K[key] = h_work
            rec.boiling_regime[key] = np.array(regimes, np.int8)
            rec.film_boiling[key] = film
            lube = self._friction(prev, c, shape, reach, zr, sup, paths, T_f)
            rec.friction_factor[key] = lube.factor
            rec.lubricant_access[key] = lube.penetration
            if reach < cs.worst_reach.get(slot, math.inf):
                cs.worst_reach[slot] = reach
            self._shadow_finding(ctx, state, cs, c, shape, zr, reach, applied, tick)
        for note_key, text in sup.notes:
            if note_key.startswith("no_holes") and contacts and cs.flags.get(note_key, 0) == 0:
                cs.flags[note_key] = 1
                parent = cs.trace_ids.get("supply")
                ctx.finding(state, DOMAIN, Category.COOLANT_SHADOWING, Severity.WARNING, text, tick,
                            quantities={"through_tool_flow_m3_s": 0.0}, parents=[parent] if parent is not None else [])

    def _walls(self, prev, c, T_f) -> dict:
        th = prev.thermal
        slot = c["slot"]
        tool = th.tools.get(slot) if hasattr(th, "tools") else None
        cut = c["cutting"]
        mask = None
        if cut is not None and np.asarray(cut.in_cut_time_s).size:
            mask = np.asarray(cut.in_cut_time_s) > 0.0
        rake = body = None
        if tool is not None:
            arr = np.asarray(tool.rake_temp_mean_K)
            if arr.size and mask is not None and arr.shape == mask.shape and mask.any():
                rake = float(np.mean(arr[mask]))
            elif arr.size:
                rake = float(np.mean(arr))
            arr = np.asarray(tool.body_temp_at_elem_K)
            if arr.size and mask is not None and arr.shape == mask.shape and mask.any():
                body = float(np.mean(arr[mask]))
            else:
                body = float(tool.body_temp_K)
        w = c["workpiece"]
        key = c["key"]
        peak = th.contact_surface_temp_K.get(key, th.work_surface_temp_K.get(w))
        pre = th.contact_preheat_K.get(key, th.work_preheat_K.get(w))
        bulk = th.work_bulk_mean_K.get(w)
        f = H.P()["near_zone_temp_fraction"]
        if peak is not None and pre is not None:
            near = pre + f * (peak - pre)
        else:
            near = peak if peak is not None else (bulk if bulk is not None else T_f)
        return {"rake": rake if rake is not None else (body if body is not None else T_f),
                "body": body if body is not None else T_f, "work": near, "bulk": bulk if bulk is not None else T_f}

    def _contact_heat(self, zr, shape, c, walls, T_f):
        st = self.settings
        hp = H.P()
        best = {"rake": (0.0, H.SINGLE_PHASE), "flank": (0.0, H.SINGLE_PHASE), "work": (0.0, H.SINGLE_PHASE)}
        if zr is None or zr.arrived_kg_s <= 0.0:
            return 0.0, 0.0, 0.0, (0, 0, 0), False
        radius = shape.radius_m if shape is not None else 0.0
        diameter = max(2.0 * radius, 1e-4)
        g = c["geometry"]
        size = max(radius, 0.5 * float(g.ap_m), 0.5 * float(g.ae_m), 1e-4)
        heat_r = hp["heat_zone_radius_factor"] * size
        for jid, arr in sorted(zr.arrivals.items()):
            if arr.arrived_kg_s <= 0.0:
                continue
            frac = min(arr.arrived_m3_s / arr.jet_flow_m3_s, 1.0) if arr.jet_flow_m3_s > 0.0 else 0.0
            gas = arr.phase in (GAS, AEROSOL)
            if gas:
                speed = arr.exit_speed_m_s
                d_arr = arr.exit_diameter_m * math.sqrt(frac)
            else:
                speed = max(arr.speed_m_s, 1e-6)
                d_imp = arr.exit_diameter_m * math.sqrt(max(arr.exit_speed_m_s, 1e-6) / speed)
                d_arr = d_imp * math.sqrt(frac)
            if d_arr <= 0.0 or speed <= 0.0:
                continue
            targets = {"rake": (walls["rake"], max(hp["martin_rd_lo"] * d_arr, radius), H.CYLINDER, diameter),
                       "flank": (walls["body"], max(hp["martin_rd_lo"] * d_arr, radius), H.CYLINDER, diameter),
                       "work": (walls["work"], max(hp["martin_rd_lo"] * d_arr, arr.offset_m + heat_r), H.PLATE, 0.0)}
            for name, (T_w, r_avg, target, target_size) in targets.items():
                T_film = 0.5 * (T_w + T_f)
                props = air_props(T_film) if gas else self.fluid.props(T_film)
                h_fc = H.jet_h(gas, speed, d_arr, max(arr.distance_m, 0.0), r_avg, props)
                if gas:
                    if arr.phase == AEROSOL and arr.liquid_m3_s > 0.0:
                        deposit = 1.0 - self.fluid.mist_loss_fraction
                        liquid_rho = self.fluid.props(T_f).rho_kg_m3 if not (self.fluid.is_gas or self.fluid.dry) \
                            else 0.0
                        flux = liquid_rho * arr.liquid_m3_s * deposit / (math.pi * r_avg * r_avg)
                        h = H.aerosol_h(h_fc, flux, T_w, T_f, self.fluid)
                    else:
                        h = h_fc
                    regime = H.SINGLE_PHASE
                else:
                    impact = props.rho_kg_m3 * speed * speed * d_arr * d_arr / (4.0 * r_avg * r_avg)
                    b = H.boiling(self.fluid, T_w, T_f, h_fc, target, target_size if target_size > 0 else r_avg,
                                  speed, impact, st.solver_rel_tol, st.solver_max_iter)
                    h, regime = b.h_W_m2K, b.regime
                if h > best[name][0]:
                    best[name] = (h, regime)
        regimes = (best["rake"][1], best["flank"][1], best["work"][1])
        film = best["work"][1] == H.FILM or best["rake"][1] == H.FILM
        return best["rake"][0], best["flank"][0], best["work"][0], regimes, film

    def _friction(self, prev, c, shape, reach, zr, sup, paths, T_f):
        cut = c["cutting"]
        h = None
        if cut is not None:
            t = np.asarray(cut.in_cut_time_s)
            ih = np.asarray(cut.int_h_dt)
            if t.size and ih.shape == t.shape and float(t.sum()) > 0.0:
                h = float(ih.sum() / t.sum())
        ratio = None
        coeffs = getattr(prev.cutting, "coefficients", {}).get(c["slot"]) if hasattr(prev.cutting, "coefficients") \
            else None
        if coeffs is not None and np.asarray(coeffs.chip_ratio).size and np.asarray(coeffs.h_grid_m).size:
            grid = np.asarray(coeffs.h_grid_m, dtype=float)
            table = np.atleast_2d(np.asarray(coeffs.chip_ratio, dtype=float))
            at = h if h is not None else float(np.exp(np.mean(np.log(grid))))
            ratio = float(np.mean([np.interp(at, grid, row) for row in table]))
        path = paths[c["path"]] if c["path"] < len(paths) else (paths[0] if paths else None)
        omega = float(np.mean(np.abs(path.spindle_omega_rad_s))) if path is not None and \
            len(path.spindle_omega_rad_s) else 0.0
        g = c["geometry"]
        if shape is not None and shape.rotating:
            v_cut = omega * shape.radius_m
        else:
            p = np.asarray(g.point_S_m, dtype=float)
            v_cut = omega * math.hypot(p[0], p[1])
        width = float(g.engaged_length_m) if float(g.engaged_length_m) > 0.0 else float(g.ap_m)
        aerosol = all(a.phase in (GAS, AEROSOL) for a in zr.arrivals.values()) if zr is not None and zr.arrivals \
            else False
        deposited = sum(a.liquid_m3_s for a in zr.arrivals.values()) * (1.0 - self.fluid.mist_loss_fraction) \
            if zr is not None else 0.0
        props = self.fluid.props(T_f) if not self.fluid.dry else air_props(T_f)
        tool_class = shape.tool_class if shape is not None else "carbide"
        return lube_friction(self.fluid, props, reach, tool_class, self.work_family, v_cut, h, ratio, width, aerosol,
                             deposited)

    def _global_shadows(self, applied, contacts) -> dict:
        keys = [c["key"] for c in contacts if c["key"] in applied.zones]
        zones = [applied.zones[k] for k in keys] or [z for k, z in applied.zones.items() if k < 0]
        aimed = sum(z.aimed_kg_s for z in zones)
        out: dict[str, float] = {}
        if aimed <= 0.0:
            return out
        for z in zones:
            for name, value in z.shadows.items():
                out[name] = out.get(name, 0.0) + value / aimed
        return dict(sorted(out.items()))

    def _far_h(self, prev, applied, T_f) -> float:
        if applied is None or not applied.jet_surface:
            return 0.0
        hp = H.P()
        st = self.settings
        bounds = self.bricks.bounds_S(0)
        diag = float(np.linalg.norm(np.asarray(bounds[1]) - np.asarray(bounds[0]))) if bounds is not None else 0.1
        r_far = hp["far_radius_fraction"] * diag
        bulk = prev.thermal.work_bulk_mean_K.get(0, T_f) if hasattr(prev.thermal, "work_bulk_mean_K") else T_f
        best = 0.0
        for jid, s in sorted(applied.jet_surface.items()):
            gas = s["phase"] in (GAS, AEROSOL)
            frac = min(s["q_m3_s"] / s["jet_flow_m3_s"], 1.0) if s["jet_flow_m3_s"] > 0 else 0.0
            if gas:
                speed = s["exit_speed_m_s"]
                d = s["diameter_m"] * math.sqrt(frac)
            else:
                speed = max(s["speed_m_s"], 1e-6)
                d = s["diameter_m"] * math.sqrt(max(s["exit_speed_m_s"], 1e-6) / speed) * math.sqrt(frac)
            if d <= 0.0:
                continue
            props = air_props(0.5 * (bulk + T_f)) if gas else self.fluid.props(0.5 * (bulk + T_f))
            h_fc = H.jet_h(gas, speed, d, s["distance_m"], max(r_far, hp["martin_rd_lo"] * d), props)
            if not gas:
                h_fc = H.boiling(self.fluid, bulk, T_f, h_fc, H.PLATE, r_far, speed, 0.0, st.solver_rel_tol,
                                 st.solver_max_iter).h_W_m2K
            best = max(best, h_fc)
        return best

    # ------------------------------------------------------------------ findings and trace
    def _shadow_finding(self, ctx, state, cs, c, shape, zr, reach, applied, tick) -> None:
        from .occlusion import P as occ
        p = occ()
        slot = c["slot"]
        flag = f"shadow:{slot}"
        number = shape.number if shape is not None else slot
        if zr is None:
            return
        if reach >= p["shadow_recover_reach"]:
            cs.flags.pop(flag, None)
            return
        if cs.flags.get(flag, 0):
            return
        aimed = zr.aimed_m3_s > p["shadow_min_flow_m3_s"]
        if aimed and reach >= p["shadow_warning_reach"]:
            return
        parent = applied.events.get(zr.key)
        parents = [parent] if parent is not None else []
        quantities = {"reach_fraction": reach, "aimed_flow_kg_s": zr.aimed_kg_s, "arriving_flow_kg_s": zr.arrived_kg_s}
        if not aimed:
            miss = zr.nearest_miss_m
            quantities["nearest_miss_m"] = miss if math.isfinite(miss) else -1.0
            text = (f"no coolant jet is aimed at the cut of T{number}"
                    + (f": jet {zr.nearest_jet} passes {miss * 1e3:.1f} mm from the cutting zone"
                       if math.isfinite(miss) and zr.nearest_jet else ""))
        else:
            shares = sorted(((v / zr.aimed_kg_s, k) for k, v in zr.shadows.items()), reverse=True)
            for share, name in shares:
                quantities[f"blocked_frac[{name}]"] = share
            causes = ", ".join(f"{name} {share * 100:.0f} %" for share, name in shares[:3])
            text = f"only {reach * 100:.0f} % of the coolant aimed at the cut of T{number} reaches it" + (
                f" (blocked by {causes})" if causes else "")
        cs.flags[flag] = 1
        ctx.finding(state, DOMAIN, Category.COOLANT_SHADOWING, Severity.WARNING, text, tick, path=c["path"],
                    quantities=quantities, parents=parents)

    def _supply_events(self, ctx, state, cs, sup, pumps_on, air_on, tick, tool) -> None:
        key = repr((sorted(pumps_on.items()), air_on, None if tool is None else tool.slot,
                    sorted((o.jet_id, round(o.flow_m3_s, 12)) for o in sup.outlets)))
        if cs.trace_ids.get("supply_key_hash") == hash_text(key):
            return
        cs.trace_ids["supply_key_hash"] = hash_text(key)
        q = {"flow_m3_s": sum(o.flow_m3_s for o in sup.outlets if o.phase == LIQUID),
             "shaft_power_W": sup.shaft_power_W, "air_flow_m3_s": sup.air_free_m3_s,
             "through_tool_flow_m3_s": sup.through_flow_m3_s}
        for pid, circuit in sorted(sup.circuits.items()):
            q[f"pressure_Pa[{pid}]"] = circuit.pressure_Pa
            q[f"flow_m3_s[{pid}]"] = circuit.flow_m3_s
        cs.trace_ids["supply"] = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "coolant:supply", q, [])
        for note_key, text in sup.notes:
            if note_key.startswith("relief") or note_key.startswith("blocked"):
                ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "coolant:" + note_key.split(":")[0], {},
                                 [cs.trace_ids["supply"]])

    def _job_events(self, ctx, cs, result, jets, zones, tick) -> None:
        """Trace events of a job: ``coolant:jets`` and one ``coolant:occlusion`` per zone, recorded when the
        result changed materially (reach or a blocked share by more than 0.01, the set of jets or occluders);
        otherwise the previous event stands for it (steady machining does not flood the trace)."""
        parents = [v for k, v in sorted(cs.trace_ids.items()) if k == "supply" or (k.startswith("nozzle:"))]
        q = {"jets": float(len({j.jet_id for j in jets})), "rays": float(result.n_rays)}
        for jid, reach in sorted(result.jet_reach.items()):
            q[f"reach[{jid}]"] = reach
        signature = hash_text(repr((sorted((k, round(v, 2)) for k, v in q.items()), parents)))
        if cs.trace_ids.get("jets_sig") != signature or "jets" not in cs.trace_ids:
            cs.trace_ids["jets"] = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "coolant:jets", q, parents)
            cs.trace_ids["jets_sig"] = signature
        jets_id = cs.trace_ids["jets"]
        for zone in zones:
            zr = result.zones.get(zone.key)
            if zr is None:
                continue
            qz = {"reach_fraction": zr.reach, "aimed_flow_kg_s": zr.aimed_kg_s, "arriving_flow_kg_s": zr.arrived_kg_s,
                  "zone_radius_m": zone.radius_m, "tool_slot": float(zone.tool_slot)}
            shares = {name: value / zr.aimed_kg_s for name, value in zr.shadows.items()} if zr.aimed_kg_s > 0 else {}
            for name, value in sorted(zr.shadows.items()):
                qz[f"blocked_kg_s[{name}]"] = value
            sig = hash_text(repr((round(zr.reach, 2), sorted((k, round(v, 2)) for k, v in shares.items()),
                                  zr.aimed_kg_s > 0.0, jets_id)))
            key = f"occ:{zone.key}"
            if cs.trace_ids.get(key + ":sig") != sig or key not in cs.trace_ids:
                cs.trace_ids[key] = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "coolant:occlusion", qz,
                                                     [jets_id])
                cs.trace_ids[key + ":sig"] = sig
            result.events[zone.key] = cs.trace_ids[key]

    # ------------------------------------------------------------------ tank and consumption
    def _tank(self, cs, prev, sup, rec, dt) -> None:
        tank = cs.tank
        thermal_J = float(getattr(prev.thermal, "heat_to_coolant_J", 0.0) or 0.0)
        chips = prev.chips
        booked = int(getattr(chips, "booked_until_tick", -1))
        chips_J = 0.0
        if booked == -1 or booked != cs.chips_booked_tick:
            chips_J = float(getattr(chips, "heat_to_coolant_J", 0.0) or 0.0)
            cs.chips_booked_tick = booked
        emitted = float(getattr(chips, "emitted_kg", 0.0) or 0.0)
        new_chips = max(emitted - cs.chips_emitted_kg, 0.0)
        cs.chips_emitted_kg = max(emitted, cs.chips_emitted_kg)
        pump_J = sup.shaft_power_W * dt
        liquid_out = sum(o.flow_m3_s for o in sup.outlets if o.phase == LIQUID)
        drag = self.fluid.drag_out_m3_per_kg * new_chips if tank.liquid else 0.0
        mist = self.fluid.mist_loss_fraction * liquid_out * dt if tank.liquid else 0.0
        heat_in_W = (thermal_J + chips_J + pump_J) / dt if dt > 0 else 0.0
        if tank.liquid:
            step = tank.step(dt, heat_in_W, drag, mist)
            rec.heat_in_J = step.heat_in_J
            rec.heat_out_J = step.ambient_J + step.evaporation_J + step.chiller_J
            rec.tank_residual_J = step.residual_J
        else:
            rec.heat_in_J = thermal_J + chips_J + pump_J
            rec.heat_out_J = rec.heat_in_J
        t = cs.totals
        t["thermal_heat_J"] += thermal_J
        t["chips_heat_J"] += chips_J
        t["pump_energy_J"] += pump_J
        t["air_m3"] += sup.air_free_m3_s * dt
        t["mql_oil_m3"] += sup.mql_oil_m3_s * dt
        t["mist_m3"] += mist
        t["drag_out_m3"] += drag
        t["pumped_m3"] += sup.pumped_m3_s * dt
        t["chip_mass_kg"] += new_chips
        t["on_s"] += dt if sup.outlets else 0.0
        rec.tank_temp_K = tank.temp_K if tank.liquid else self.ambient_K
        rec.tank_volume_m3 = tank.volume_m3
        rec.concentration_frac = tank.concentration
        cs.tank_min_K = min(cs.tank_min_K, rec.tank_temp_K)
        cs.tank_max_K = max(cs.tank_max_K, rec.tank_temp_K)

    def _part_op(self, state, cs) -> None:
        if state.clock.part_index != cs.part_index:
            cs.per_part[cs.part_index] = {k: v for k, v in cs.totals.items()}
            cs.part_index = state.clock.part_index
        cs.op_index = state.clock.op_index

    def _off_record(self, cs) -> CoolantFeedback:
        rec = CoolantFeedback(on=False, kind="dry")
        tank = cs.tank
        rec.fluid_temp_K = tank.temp_K if tank.liquid else self.ambient_K
        rec.tank_temp_K = rec.fluid_temp_K
        rec.tank_volume_m3 = tank.volume_m3
        rec.concentration_frac = tank.concentration
        return rec

    # ------------------------------------------------------------------ operations and the end
    def end_operation(self, ctx, state) -> None:
        cs: CoolantState = state.coolant
        st = self.settings
        if cs.tank.liquid and self.idle_s > 0.0:
            n = max(int(math.ceil(self.idle_s / st.idle_step_s)), 1)
            for _ in range(n):
                cs.tank.step(self.idle_s / n, 0.0)
        row = {"op_index": cs.op_index}
        for k, v in cs.totals.items():
            row[k] = v - cs.op_start.get(k, 0.0)
        cs.per_op.append(row)
        cs.op_start = dict(cs.totals)
        cs.applied = None
        cs.pending = None
        cs.config_key = ""
        cs.flags = {}

    def finalize(self, ctx, state) -> dict:
        cs: CoolantState = state.coolant
        tank = cs.tank
        parts = dict(cs.per_part)
        parts[cs.part_index] = dict(cs.totals)
        per_part = []
        previous = None
        for idx in sorted(parts):
            row = parts[idx]
            per_part.append({"part": idx, **{k: row[k] - (previous[k] if previous else 0.0) for k in row}})
            previous = row
        conc0 = cs.tank.totals.get("concentrate_out_m3", 0.0)
        return {
            "fluid": self.fluid.id, "kind": self.fluid.kind,
            "tank": {"temp_start_K": tank.ref_temp_K, "temp_end_K": tank.temp_K, "temp_min_K": cs.tank_min_K,
                     "temp_max_K": cs.tank_max_K, "volume_end_m3": tank.volume_m3,
                     "concentration_end_frac": tank.concentration, "totals": dict(tank.totals),
                     "makeup_water_m3": tank.totals.get("evaporated_m3", 0.0),
                     "makeup_mix_m3": tank.totals.get("drag_out_m3", 0.0) + tank.totals.get("mist_m3", 0.0),
                     "concentrate_lost_m3": conc0},
            "totals": dict(cs.totals), "per_operation": list(cs.per_op), "per_part": per_part,
            "worst_reach": {str(k): v for k, v in sorted(cs.worst_reach.items())},
            "notes": list(cs.notes),
            "property_tables_clamped": self.fluid.clamped_lookups() if not self.fluid.dry else {},
        }

    # ------------------------------------------------------------------ recording
    def macro_values(self, state, record: CoolantFeedback | None = None) -> dict:
        rec = record if record is not None else state.bus_next.coolant
        keys = sorted(rec.reach_fraction)[:MAX_CONTACTS]
        values = {"coolant.on": 1.0 if rec.on else 0.0, "coolant.fluid_temp_K": rec.fluid_temp_K,
                  "coolant.tank_temp_K": rec.tank_temp_K, "coolant.tank_volume_m3": rec.tank_volume_m3,
                  "coolant.concentration_frac": rec.concentration_frac,
                  "coolant.flow_m3_s": sum(rec.pump_flow_m3_s.values()),
                  "coolant.through_tool_flow_m3_s": rec.through_tool_flow_m3_s,
                  "coolant.through_tool_exit_speed_m_s": rec.through_tool_exit_speed_m_s,
                  "coolant.pump_power_W": sum(rec.pump_power_W.values()), "coolant.air_flow_m3_s": rec.air_flow_m3_s,
                  "coolant.mql_oil_m3_s": rec.mql_oil_m3_s, "coolant.h_work_far_W_m2K": rec.h_work_far_W_m2K}
        for name, src in (("reach_fraction", rec.reach_fraction), ("h_flank_W_m2K", rec.h_flank_W_m2K),
                          ("h_rake_W_m2K", rec.h_rake_W_m2K), ("h_work_W_m2K", rec.h_work_W_m2K),
                          ("friction_factor", rec.friction_factor)):
            row = np.full(MAX_CONTACTS, np.nan)
            for i, k in enumerate(keys):
                row[i] = src.get(k, np.nan)
            values[f"coolant.{name}"] = row
        dt = self._dt if getattr(self, "_dt", 0.0) > 0.0 else 1.0
        values["coolant.heat_in_W"] = rec.heat_in_J / dt
        values["coolant.heat_out_W"] = rec.heat_out_J / dt
        wet = np.asarray(rec.wetted)
        values["coolant.wetted_fraction"] = float(wet.mean()) if wet.size else 0.0
        return values


def hash_text(text: str) -> int:
    """Stable 63-bit hash of a configuration text (zlib CRC of both halves; PYTHONHASHSEED-free)."""
    import zlib
    data = text.encode("utf-8")
    return (zlib.crc32(data) << 31) ^ zlib.crc32(data[::-1])


def _default_shapes(project, state, settings):
    tooling = getattr(state, "tooling", None)
    if tooling is not None and hasattr(tooling, "geometry"):
        return TooledShapes(tooling, settings.profile_dz_m, settings.profile_nodes), ""
    from ..tooling import build_tool
    tools = _get(project, "tools")
    entries = list(_get(tools, "list") or [])
    geos = []
    for slot in getattr(state.tools, "slots", ()):
        if slot.slot >= len(entries):
            break
        entry = entries[slot.slot]
        geos.append(build_tool(_get(entry, "tool"), _get(entry, "holder"), entry, slot=slot.slot,
                               tool_id=slot.tool_id, holder_id=slot.holder_id))
    return TooledShapes(geos, settings.profile_dz_m, settings.profile_nodes), \
        "tool shapes built from the project tool list (no tooling state wired)"


def _default_bricks(project, settings):
    stock = _get(project, "stock")
    size = _get(stock, "size_m")
    if size is None:
        return StockBricks(np.zeros(3), np.full(3, 1e-3), settings.brick_voxel_m, settings.brick_sub), \
            "no stock size: a 1 mm placeholder block stands for the workpiece"
    size = np.asarray([float(v) for v in size], dtype=float)
    kind = str(_get(stock, "kind", "block") or "block")
    note = "workpiece occlusion from the stock shape (no in-process workpiece field wired)"
    if kind == "block" and size.size >= 3:
        return StockBricks(np.zeros(3), size[:3], settings.brick_voxel_m, settings.brick_sub), note
    if kind in ("cylinder", "tube") and size.size >= 2:
        r_out = 0.5 * size[0]
        r_in = 0.5 * size[1] if kind == "tube" and size.size >= 3 else 0.0
        length = size[-1]
        lo = np.array([-r_out, -r_out, 0.0])
        hi = np.array([r_out, r_out, length])

        def inside(p):
            rho = np.hypot(p[:, 0], p[:, 1])
            return (rho <= r_out) & (rho >= r_in) & (p[:, 2] >= 0.0) & (p[:, 2] <= length)
        return OccupancyBricks(lo, hi, settings.brick_voxel_m, settings.brick_sub, inside=inside), note
    return StockBricks(np.zeros(3), np.resize(size, 3), settings.brick_voxel_m, settings.brick_sub), note
