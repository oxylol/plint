"""RemovalDomain: the in-process workpiece of every operation (architecture 5.3, 8, 19; docs/models/removal.md).

* ``init`` builds workpiece 0 from ``project.stock`` (block, cylinder, tube, mesh), its CAD target
  (``project.cad``), a polar field when the machine turns a cylindrical stock, the held regions and fixture
  boxes of the first setup, and every tool slot's shapes (revolved profiles, insert region, tap sweep, tooth
  slabs) from the tooling state.
* ``kernel_inputs`` keeps pool headroom, refreshes the tool shapes of the paths' active tools and returns
  :class:`cnctwin.removal.abi.RemovalInputs` (arrays the compiled loop updates in place).
* ``absorb`` turns the accumulators and the event buffer into ``bus_next.removal`` (per contact removed
  volume and ToolContact geometry; events) and raises the findings (rapid into material, collision, gouge,
  breakthrough, non-centre-cutting plunge) once per episode. The core Ledger books the removed volume and
  mass and the freed bodies from this record (removal writes nothing outside it).
* ``macro_step`` closes the volume-consistency ratio (lag one step: the analytic chip volume of the previous
  step is cutting's ``prev`` record), runs the connectivity pass when topology changed (loose bodies,
  ``freed_body_mass_kg``), syncs the tri-dexel from the polar field once per ``polar_sync_revolutions``,
  reports long air cutting and publishes ``field_version``.
* ``end_operation`` rebuilds the tri-dexel exactly from the polar field, checks the remaining material after
  the last operation and carries the workpiece to the next operation (``from_previous``).

Bus read (prev): ``motion.paths[*]`` (tool_slot, workpiece, spindle_omega_rad_s, ipo_ticks, pose_M_T,
pose_S_T), ``cutting.contacts[*]`` (contact, analytic_chip_volume_m3); state read: ``state.tooling`` (tool
geometry, edge model, version per slot). Bus written: ``removal`` only.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.abi import event_buffer
from ..core.bus import RemovalContact, RemovalFrame, ToolContact
from ..core.errors import InputError
from ..core.findings import Category, Severity
from ..tooling.abi import PART_NAMES
from . import abi as A
from . import field as FD
from . import polar as PO
from . import profile as PR
from . import stock as SK
from .channels import MAX_PATHS, channels as _channels
from .schema import RemovalSettings, settings as _settings
from .state import (AIR_ACTIVE, AIR_REPORTED, AIR_START_TICK, AIR_TIME_S, AIR_X0, PS_ACTIVE, PS_LAST_TICK,
                    PS_START_TICK, PS_V_ANALYTIC, PS_V_FIELD, PS_V_X, PS_V_Y, RemovalState, ToolShape, Workpiece,
                    new_state)
from .views import FieldGeometry

DOMAIN = "removal"
LATHES = ("lathe", "mill_turn", "swiss")
TAPS = ("tap_cutting", "tap_forming")
NON_ROTATING = ("turning_insert", "grooving_insert", "parting_insert", "threading_insert", "form_tool", "boring_bar")

_FINDING = {
    A.EV_RAPID_INTO_MATERIAL: (Category.RAPID_INTO_MATERIAL, Severity.ERROR, "rapid move into material"),
    A.EV_COLLISION: (Category.COLLISION, Severity.ERROR, "non-cutting part of the tool in material"),
    A.EV_GOUGE: (Category.GOUGE, Severity.ERROR, "the cut removed part material (gouge)"),
    A.EV_BREAKTHROUGH: (Category.BREAKTHROUGH, Severity.ERROR, "the cut went through the part wall (breakthrough)"),
    A.EV_NON_CENTRE_PLUNGE: (Category.NON_CENTRE_CUTTING_PLUNGE, Severity.ERROR,
                             "axial feed into material with a tool that does not cut at its centre"),
    A.EV_FIXTURE_CONTACT: (Category.COLLISION, Severity.ERROR, "tool or holder in the fixture"),
}


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        value = getattr(obj, key)
    except (AttributeError, KeyError):
        return default
    return default if value is None else value


def _float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except TypeError:
        return default


def density_20C(material) -> float:
    """Density at 20 degC (kg/m^3) of a material object (number, sourced value or temperature table).
    Database materials keep it under ``physical`` (db/materials/*.yaml); the top-level key serves fixtures.
    (Integration fix, additive: core.hotloop / docs/models/hotloop.md "Changes in domain packages".)"""
    rho = _get(material, "density_kg_m3")
    if rho is None:
        rho = _get(_get(material, "physical"), "density_kg_m3")
    if rho is None:
        raise InputError("removal: the material has no density_kg_m3 (mass of removed material)")
    if callable(rho) and not isinstance(rho, (int, float)):
        try:
            return float(rho(293.15))
        except TypeError:
            pass
    return float(rho)


class RemovalDomain:
    name = DOMAIN

    def __init__(self, tool_shapes: dict | None = None, held_boxes=None, n_paths: int | None = None):
        """``tool_shapes`` (slot -> ToolShape) and ``held_boxes`` override what init derives from the tooling
        state and the setup (tests, analytic scenes); ``n_paths`` overrides the path count."""
        self._tool_shapes = tool_shapes
        self._held = held_boxes
        self._n_paths = n_paths

    # ------------------------------------------------------------------ interface
    def channels(self):
        return _channels()

    def init(self, ctx, state) -> None:
        project = ctx.project
        physics = _get(project, "physics")
        st_set = _settings(physics)
        n_paths = self._n_paths or self._path_count(state)
        wp = self._build_workpiece(ctx, project, st_set)
        rs = new_state(st_set, n_paths, FD.n_levels(wp.field))
        rs.workpieces.append(wp)
        rs.tools = dict(self._tool_shapes) if self._tool_shapes is not None else self._tools_from_state(state, st_set)
        state.removal = rs
        for note in wp.notes:
            if note.startswith("polar: "):
                ctx.finding(state, DOMAIN, Category.SANITY, Severity.WARNING, note[7:], 0)
        ctx.trace.record(0.0, 0, DOMAIN, "removal:stock",
                         {"volume_m3": wp.initial_volume_m3, "target_volume_m3": wp.target_volume_m3,
                          "rays": float(wp.field.info[A.FI_NRAYS]), "levels": float(FD.n_levels(wp.field)),
                          "memory_bytes": float(FD.memory_bytes(wp.field, wp.target)),
                          "polar": 1.0 if wp.polar is not None else 0.0}, [])

    def kernel_inputs(self, ctx, state, bus) -> A.RemovalInputs:
        rs: RemovalState = state.removal
        s = rs.settings
        for wp in rs.workpieces:
            need = max(int(s.pool_min_free), int(math.ceil(s.pool_headroom * wp.peak_blocks_step)))
            wp.field = FD.ensure_free(wp.field, need)
        paths = list(getattr(bus.motion, "paths", []) or []) if bus is not None else []
        cut, whole, inserts, helix, slabs = [], [], [], [], []
        ip = np.zeros(A.RI_PATH0 + A.RI_PSTRIDE * rs.n_paths + len(rs.workpieces), np.int64)
        fp = np.zeros(A.RF_PATH0 + A.RF_PSTRIDE * rs.n_paths)
        ip[A.RI_NPATHS] = rs.n_paths
        ip[A.RI_NWORK] = len(rs.workpieces)
        a = float(rs.workpieces[0].field.lv_cell_m[0])
        fp[A.RF_GOUGE_TOL] = s.gouge_tolerance_m
        fp[A.RF_COLLISION_TOL] = s.collision_tolerance_m
        fp[A.RF_GATE_SAMPLE] = s.gate_sample_cells * a
        fp[A.RF_GATE_TOL] = s.gate_tolerance_m
        fp[A.RF_GATE_MARGIN] = s.gate_margin_m
        fp[A.RF_SECTOR_MARGIN] = s.sector_margin_cells * a
        fp[A.RF_SWEEP_STEP] = s.sweep_step_cells * a
        for p in range(rs.n_paths):
            frame = paths[p] if p < len(paths) else None
            slot = int(_get(frame, "tool_slot", -1)) if frame is not None else -1
            work = int(_get(frame, "workpiece", 0)) if frame is not None else 0
            work = min(max(work, 0), len(rs.workpieces) - 1)
            ts = self._shape(state, rs, slot)
            base = A.RI_PATH0 + A.RI_PSTRIDE * p
            ip[base + A.RP_WORKPIECE] = work
            ip[base + A.RP_SLOT] = slot
            ip[base + A.RP_KIND] = ts.kind if ts is not None else A.TK_NONE
            ip[base + A.RP_CENTRE_CUTTING] = 1 if (ts is None or ts.centre_cutting) else 0
            ip[base + A.RP_ROTATING] = 1 if (ts is not None and ts.rotating) else 0
            fb = A.RF_PATH0 + A.RF_PSTRIDE * p
            if ts is not None:
                fp[fb + A.RQ_R_DEAD] = ts.r_dead_m
                fp[fb + A.RQ_Z_DEAD] = ts.z_dead_m
                fp[fb + A.RQ_TAN_RAMP_MAX] = _tan_ramp_max()
                fp[fb + A.RQ_SLAB_H] = _slab_height(state, slot)
            cut.append(ts.cut if ts is not None else A.empty_profile())
            whole.append(ts.whole if ts is not None else A.empty_profile())
            inserts.append(ts.insert if ts is not None else A.empty_insert())
            helix.append(ts.helix if ts is not None else A.empty_helix())
            slabs.append(ts.slabs if ts is not None else A.empty_slabs())
            rs.contact_slot[p] = slot
            rs.contact_work[p] = work
            A.reset_accum(rs.accum[p])
        for w, wp in enumerate(rs.workpieces):
            ip[A.RI_PATH0 + A.RI_PSTRIDE * rs.n_paths + w] = wp.next_sev
            wp.field.info[A.FI_OVERFLOW] = 0
        rs.events.n[:] = 0
        polars = tuple(wp.polar if wp.polar is not None else _EMPTY_POLAR for wp in rs.workpieces)
        inputs = A.RemovalInputs(tuple(wp.field for wp in rs.workpieces), tuple(wp.target for wp in rs.workpieces),
                                 polars, tuple(cut), tuple(whole), tuple(inserts), tuple(helix), tuple(slabs),
                                 tuple(rs.accum), rs.events, rs.scratch, ip, fp)
        rs.last_inputs = inputs
        return inputs

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        rs: RemovalState = state.removal
        inputs = rs.last_inputs
        frame = RemovalFrame()
        tick_now = int(state.clock.tick)
        first_sev = [wp.next_sev for wp in rs.workpieces]
        for w, wp in enumerate(rs.workpieces):
            if inputs is not None and w < len(inputs.fields):
                wp.field = inputs.fields[w]
                ip = inputs.iparams
                wp.next_sev = int(ip[A.RI_PATH0 + A.RI_PSTRIDE * rs.n_paths + w])
            used = int(wp.field.info[A.FI_N_BLOCKS] - wp.field.info[A.FI_N_FREE])
            wp.peak_blocks_step = max(wp.peak_blocks_step, used)
            if wp.field.info[A.FI_SCRATCH_OVF]:
                wp.field.info[A.FI_SCRATCH_OVF] = 0
                ctx.trace.record(tick_now * ctx.tick_s, tick_now, DOMAIN, "removal:scratch_overflow",
                                 {"workpiece": float(w)}, [])
                ctx.finding(state, DOMAIN, Category.SANITY, Severity.ERROR,
                            f"a ray of workpiece {w} holds more than {A.MAX_SEG} material segments: the field cannot "
                            f"represent the cut there", tick_now)
        for p in range(rs.n_paths):
            acc = rs.accum[p]
            w = rs.contact_work[p]
            wp = rs.workpieces[w]
            vz = float(acc.removed_q[0, 2]) * FD.volume_factor_m3(wp.field, 0)
            vx = float(acc.removed_q[0, 0]) * FD.volume_factor_m3(wp.field, 0)
            vy = float(acc.removed_q[0, 1]) * FD.volume_factor_m3(wp.field, 0)
            q2 = int(acc.ivals[A.IA_POLAR_Q2])
            turning = wp.polar is not None and q2 > 0
            vol = float(q2) * PO.q2_volume_factor(wp.polar) if turning else vz
            for lvl in range(acc.removed_q.shape[0]):
                if lvl < len(wp.removed_q):
                    for fam in range(3):
                        wp.removed_q[lvl][fam] += int(acc.removed_q[lvl, fam])
            wp.polar_q2 += q2
            geom = self._contact_geometry(ctx, state, acc, p)
            contact = RemovalContact(contact=p, tool_slot=rs.contact_slot[p], workpiece=w, path=p,
                                     removed_volume_m3=vol, displaced_volume_m3=0.0, geometry=geom,
                                     volume_consistency=1.0)
            contact.removed_volume_xyz_m3 = np.array([vx, vy, vz])
            contact.polar = bool(turning)
            contact.surface_ids = (int(first_sev[w]), int(wp.next_sev))
            frame.contacts.append(contact)
            self._pass_update(rs, p, acc, vol, vx, vy, geom.in_cut, tick_now)
            self._air_update(ctx, state, rs, p, acc)
        # events
        n = int(rs.events.n[0])
        if rs.events.n[1]:
            rs.settings = _replace(rs.settings, event_capacity=int(rs.settings.event_capacity) * 2)
            rs.events = event_buffer(int(rs.settings.event_capacity), A.EVENT_WIDTH)
            ctx.trace.record(tick_now * ctx.tick_s, tick_now, DOMAIN, "removal:event_overflow",
                             {"rows": float(n), "capacity": float(rs.settings.event_capacity)}, [])
        seen_kinds = set()
        for i in range(n):
            row = np.array(rs.events.rec[i]) if i < rs.events.rec.shape[0] else None
            if row is None:
                continue
            kind = int(row[A.EC_KIND])
            contact = int(row[A.EC_CONTACT])
            ev = {"kind": A.EV_NAMES.get(kind, str(kind)), "tick": int(row[A.EC_TICK]), "contact": contact,
                  "point_S_m": [float(row[A.EC_X]), float(row[A.EC_Y]), float(row[A.EC_Z])],
                  "depth_m": float(row[A.EC_DEPTH]), "volume_m3": float(row[A.EC_VOLUME]),
                  "surface_event": int(row[A.EC_SEV_ID]), "level": int(row[A.EC_LEVEL])}
            part = int(row[A.EC_PART])
            if part >= 0:
                ev["part"] = PART_NAMES[part] if part < len(PART_NAMES) else str(part)
            frame.events.append(ev)
            rs.counts[ev["kind"]] = rs.counts.get(ev["kind"], 0) + 1
            self._finding_for(ctx, state, rs, kind, contact, ev)
            seen_kinds.add((kind, contact))
        for key in list(rs.episodes):
            kind_s, contact_s = key.split(":")
            if (int(kind_s), int(contact_s)) not in seen_kinds:
                del rs.episodes[key]
        frame.field_version = {w: int(wp.field.info[A.FI_VERSION]) for w, wp in enumerate(rs.workpieces)}
        bus_next.removal = frame

    def macro_step(self, ctx, state, prev, nxt) -> None:
        rs: RemovalState = state.removal
        s = rs.settings
        rs.macro += 1
        frame = nxt.removal if isinstance(nxt.removal, RemovalFrame) else RemovalFrame()
        nxt.removal = frame
        tick = int(state.clock.tick)
        # ---- volume consistency (previous step: cutting's analytic chip volume is in prev)
        analytic = {}
        for cc in getattr(prev.cutting, "contacts", []) or []:
            analytic[int(cc.contact)] = float(getattr(cc, "analytic_chip_volume_m3", 0.0))
        for c in frame.contacts:
            p = int(c.contact)
            va = analytic.get(p, 0.0)
            vf = float(rs.prev_field_m3[p]) if p < len(rs.prev_field_m3) else 0.0
            c.volume_consistency = vf / va if va > 0.0 else 1.0
            if p < rs.pass_stats.shape[0] and rs.pass_stats[p, PS_ACTIVE]:
                rs.pass_stats[p, PS_V_ANALYTIC] += va
        for c in frame.contacts:
            p = int(c.contact)
            if p < len(rs.prev_field_m3):
                rs.prev_field_m3[p] = float(c.removed_volume_m3)
        self._pass_close(ctx, state, rs, frame, tick)
        # ---- connectivity / loose bodies
        if rs.macro - rs.last_connect_macro >= s.connect_every_n_macro:
            for w, wp in enumerate(rs.workpieces):
                if wp.field.info[A.FI_TOPO] > 0:
                    rs.last_connect_macro = rs.macro
                    self._loose(ctx, state, rs, w, frame, tick)
        # ---- polar -> tri-dexel sync
        turned = self._spindle_turned(prev, float(ctx.tick_s), int(ctx.ipo_ticks))
        for w, wp in enumerate(rs.workpieces):
            if wp.polar is None:
                continue
            wp.polar_turned_rad += turned
            if wp.polar.info[A.PI_DIRTY_Z0] >= 0 and wp.polar_turned_rad >= 2.0 * math.pi * s.polar_sync_revolutions:
                self._sync_polar(ctx, state, w, dirty_only=True)
        frame.field_version = {w: int(wp.field.info[A.FI_VERSION]) for w, wp in enumerate(rs.workpieces)}

    def end_operation(self, ctx, state) -> None:
        rs: RemovalState = state.removal
        tick = int(state.clock.tick)
        for w, wp in enumerate(rs.workpieces):
            if wp.polar is not None and wp.polar.info[A.PI_DIRTY_Z0] >= 0:
                self._sync_polar(ctx, state, w, dirty_only=False)
            frame = RemovalFrame()
            self._loose(ctx, state, rs, w, frame, tick, force=True)
            if frame.loose_bodies and isinstance(state.bus_next.removal, RemovalFrame):
                state.bus_next.removal.loose_bodies.extend(frame.loose_bodies)
        op_index = int(getattr(state.clock, "op_index", 0))
        ops = list(_get(ctx.project, "operations", []) or []) if ctx.project is not None else []
        last = op_index >= len(ops) - 1
        if last:
            for w, wp in enumerate(rs.workpieces):
                self._remaining(ctx, state, w, tick)
        else:
            self._transfer(ctx, state, ops[op_index + 1])

    def finalize(self, ctx, state) -> dict:
        rs: RemovalState = state.removal
        out = {"workpieces": [], "events": dict(sorted(rs.counts.items())), "loose_bodies": list(rs.loose_log),
               "notes": list(rs.notes)}
        for w, wp in enumerate(rs.workpieces):
            removed_z = wp.removed_q[0][2] if wp.removed_q else 0
            out["workpieces"].append({
                "workpiece": w, "kind": wp.kind, "initial_volume_m3": wp.initial_volume_m3,
                "volume_m3": FD.volume_m3(wp.field, 0, 2), "removed_volume_m3": float(removed_z) * FD.volume_factor_m3(
                    wp.field, 0), "polar_removed_m3": float(wp.polar_q2) * PO.q2_volume_factor(wp.polar)
                if wp.polar is not None else 0.0,
                "target_volume_m3": wp.target_volume_m3, "memory_bytes": FD.memory_bytes(wp.field, wp.target),
                "pool_peak_blocks": int(wp.field.info[A.FI_PEAK_USED]),
                "max_segments": int(wp.field.info[A.FI_MAX_SEG]),
                "notes": list(wp.notes)})
        return out

    # ------------------------------------------------------------------ integration helpers
    def needs_replay(self, state) -> bool:
        """A subtraction of the step was refused for lack of pool blocks, or a polar cut for lack of inline
        segments (the loop replays the step; both kernels are idempotent)."""
        return any(int(wp.field.info[A.FI_OVERFLOW]) or (wp.polar is not None and int(wp.polar.info[A.PI_OVERFLOW]))
                   for wp in state.removal.workpieces)

    def prepare_replay(self, ctx, state) -> None:
        """Grow every pool that overflowed (doubling) and every polar family that overflowed (inline segments
        doubled, polar.grow_kp); kernel_inputs of the replay passes the new arrays."""
        for wp in state.removal.workpieces:
            if wp.field.info[A.FI_OVERFLOW]:
                wp.field = FD.grow_pool(wp.field, max(int(wp.field.info[A.FI_N_BLOCKS]), 1024))
                wp.peak_blocks_step = max(wp.peak_blocks_step, int(wp.field.info[A.FI_N_BLOCKS]))
            if wp.polar is not None and wp.polar.info[A.PI_OVERFLOW]:
                wp.polar = PO.grow_kp(wp.polar)
                wp.notes.append(f"polar inline segments grown to {int(wp.polar.info[A.PI_KP])} radial, "
                                f"{int(wp.polar.info[A.PI_KPA])} axial")

    def geometry(self, state, caller: str = "default") -> FieldGeometry:
        return FieldGeometry(lambda: state.removal, caller)

    def channel_values(self, ctx, state, bus) -> dict:
        rs: RemovalState = state.removal
        rec = bus.removal
        contacts = list(getattr(rec, "contacts", []) or [])
        vol = sum(float(c.removed_volume_m3) for c in contacts)
        wp = rs.workpieces[0]
        out = {"removal.removed_volume_m3": vol, "removal.removal_rate_m3_s": vol / ctx.macro_s,
               "removal.field_volume_m3": FD.volume_m3(wp.field, 0, 2),
               "removal.freed_mass_kg": float(sum(float(r["mass_kg"]) for r in rs.loose_log)),
               "removal.pool_used_blocks": float(wp.field.info[A.FI_N_BLOCKS] - wp.field.info[A.FI_N_FREE]),
               "removal.events": float(len(getattr(rec, "events", []) or [])),
               "removal.field_version": float(wp.field.info[A.FI_VERSION]),
               "removal.polar_volume_m3": PO.volume_m3(wp.polar) if wp.polar is not None else 0.0}
        per = {k: np.full(MAX_PATHS, np.nan) for k in ("removal.volume_consistency", "removal.ae_m", "removal.ap_m",
                                                         "removal.engaged_angle_rad", "removal.in_cut_fraction",
                                                         "removal.air_cut_time_s")}
        for c in contacts:
            p = int(c.contact)
            if p >= MAX_PATHS:
                continue
            g = c.geometry
            per["removal.volume_consistency"][p] = c.volume_consistency
            per["removal.ae_m"][p] = g.ae_m
            per["removal.ap_m"][p] = g.ap_m
            per["removal.engaged_angle_rad"][p] = g.engaged_angle_rad
            per["removal.in_cut_fraction"][p] = g.in_cut_fraction
            per["removal.air_cut_time_s"][p] = g.air_cut_time_s
        out.update(per)
        return out

    # ------------------------------------------------------------------ init helpers
    @staticmethod
    def _path_count(state) -> int:
        tools = getattr(state, "tools", None)
        slots = getattr(tools, "slots", ()) if tools is not None else ()
        n = max([int(getattr(s, "path", 0)) for s in slots] + [0]) + 1
        return max(1, n)

    def _build_workpiece(self, ctx, project, st: RemovalSettings) -> Workpiece:
        stock = _get(project, "stock")
        kind = str(_get(stock, "kind", "block"))
        size = [float(v) for v in (_get(stock, "size_m") or [])]
        refine = st.refine
        if kind == "block":
            if len(size) != 3:
                raise InputError("removal: a block stock needs size_mm [x, y, z]")
            fld = SK.block_field(size, st.lattice_m, refine, margin_nodes=st.margin_nodes)
        elif kind in ("cylinder", "tube"):
            if kind == "cylinder":
                od, idia, length = size[0], 0.0, size[1]
            else:
                od, idia, length = size[0], size[1], size[2]
            fld = SK.tube_field(od, length, st.lattice_m, idia, refine, margin_nodes=st.margin_nodes)
        elif kind == "mesh":
            path = _get(stock, "mesh")
            if not path:
                raise InputError("removal: a mesh stock needs stock.mesh")
            V, Fc = SK.read_mesh_file(path, unit="mm")
            fld, _ = SK.mesh_field(V, Fc, st.lattice_m, refine, margin_nodes=st.margin_nodes)
        else:
            raise InputError(f"removal: unknown stock kind {kind!r}")
        rho = density_20C(_get(project, "material"))
        target = A.empty_target(int(fld.info[A.FI_NRAYS]))
        target_volume = 0.0
        cad = _get(project, "cad")
        if cad is not None and _get(cad, "path"):
            units = str(_get(cad, "units", "mm"))
            V, Fc = SK.read_mesh_file(_get(cad, "path"), unit=units)
            pose = np.asarray(_get(cad, "cad_pose_S_C", np.eye(4)), dtype=float)
            target, info = SK.target_from_mesh(fld, SK.transform_points(V, pose), Fc)
            target_volume = float(info["cad_volume_m3"])
        wp = Workpiece(index=0, field=fld, target=target, rho_kg_m3=rho, kind=kind,
                       removed_q=[[0, 0, 0] for _ in range(FD.n_levels(fld))],
                       initial_volume_m3=FD.volume_m3(fld, 0, 2), target_volume_m3=target_volume)
        wp.held_boxes = list(self._held) if self._held is not None else self._held_from_setup(project, 0)
        machine_type = str(_get(_get(project, "machine"), "type", ""))
        if machine_type in LATHES and kind in ("cylinder", "tube"):
            wp.polar, note = self._make_polar(ctx, project, st, fld, kind, size)
            if note:
                wp.notes.append("polar: " + note)
        return wp

    def _make_polar(self, ctx, project, st, fld, kind, size):
        if kind == "cylinder":
            od, idia, length = size[0], 0.0, size[1]
        else:
            od, idia, length = size[0], size[1], size[2]
        r_nose = self._min_nose_radius(project)
        n_th, dz, nz, dr, nr, note = PO.resolution(length, st.polar_theta_bins, st.polar_min_feed_m_per_rev, r_nose,
                                                   0.5 * od, st.polar_kp, st.polar_max_bytes,
                                                   st.polar_axial_spacing_m or None, st.lattice_m)
        pf = PO.polar_tube(0.5 * od, 0.5 * idia, 0.0, length, n_th, dz, dr, kp=st.polar_kp)
        return pf, note

    @staticmethod
    def _min_nose_radius(project) -> float:
        best = 0.0
        tools = _get(_get(project, "tools"), "list", []) or []
        for entry in tools:
            tool = _get(entry, "tool")
            geometry = _get(tool, "geometry")
            insert = _get(geometry, "insert")
            r = _float(_get(insert, "nose_radius_m"), 0.0)
            if r > 0.0 and (best == 0.0 or r < best):
                best = r
        return best

    @staticmethod
    def _held_from_setup(project, op_index: int) -> list:
        """Workholding device bodies of a setup as boxes in S (the material they touch is held)."""
        ops = list(_get(project, "operations", []) or [])
        if op_index >= len(ops):
            return []
        op = ops[op_index]
        setup = _get(op, "setup")
        S_from_W = np.linalg.inv(np.asarray(_get(op, "stock_pose_W_S", np.eye(4)), dtype=float))
        boxes = []
        for item in _get(setup, "workholding", []) or []:
            device = _get(item, "device")
            from .collision import device_body_box
            body = device_body_box(_get(device, "body_m"))
            if body is None:
                continue
            pose = _get(item, "pose")
            T = pose.matrix() if hasattr(pose, "matrix") else np.eye(4)
            corners = np.array([[x, y, z] for x in body[:, 0] for y in body[:, 1] for z in body[:, 2]])
            M = S_from_W @ T
            pts = corners @ M[:3, :3].T + M[:3, 3]
            boxes.append((pts.min(axis=0), pts.max(axis=0)))
        return boxes

    def _tools_from_state(self, state, st: RemovalSettings) -> dict:
        tooling = getattr(state, "tooling", None)
        out = {}
        if tooling is None:
            return out
        for slot in range(len(tooling)):
            out[slot] = build_shape(tooling.geometry(slot), tooling.edge_model(slot), slot,
                                    _slab_h(st), int(tooling.version[slot]) if len(tooling.version) > slot else 0)
        return out

    def _shape(self, state, rs: RemovalState, slot: int) -> ToolShape | None:
        if slot < 0:
            return None
        ts = rs.tools.get(slot)
        tooling = getattr(state, "tooling", None)
        if tooling is not None and slot < len(tooling):
            version = int(tooling.version[slot]) if len(tooling.version) > slot else 0
            if ts is None:
                ts = build_shape(tooling.geometry(slot), tooling.edge_model(slot), slot, _slab_h(rs.settings), version)
                rs.tools[slot] = ts
            elif ts.edge_version != version and ts.kind in (A.TK_ENVELOPE,):
                ts.slabs = slabs_for(tooling.edge_model(slot), ts.cut, _slab_h(rs.settings))
                ts.edge_version = version
        return ts

    # ------------------------------------------------------------------ absorb helpers
    def _contact_geometry(self, ctx, state, acc, p) -> ToolContact:
        fv = acc.fvals
        iv = acc.ivals
        g = ToolContact()
        g.in_cut = bool(iv[A.IA_IN_CUT_CALLS] > 0 or iv[A.IA_POLAR_Q2] > 0)
        W = float(fv[A.FA_W])
        if W > 0.0:
            g.point_S_m = np.array([fv[A.FA_CX], fv[A.FA_CY], fv[A.FA_CZ]]) / W
            nvec = np.array([fv[A.FA_NX], fv[A.FA_NY], fv[A.FA_NZ]])
            nn = float(np.linalg.norm(nvec))
            normal_S = nvec / nn if nn > 0 else np.zeros(3)
        else:
            normal_S = np.zeros(3)
        R = np.asarray(fv[A.FA_M_FROM_S:A.FA_M_FROM_S + 9]).reshape(3, 3)
        t = np.asarray(fv[A.FA_M_FROM_S + 9:A.FA_M_FROM_S + 12])
        if np.allclose(R, np.eye(3)) and not np.any(t):
            R, t = _m_from_s(state, p)
        g.point_M_m = R @ g.point_S_m + t if W > 0.0 else np.zeros(3)
        g.normal_M = R @ normal_S
        g.ae_m = float(fv[A.FA_AE_MAX])
        zmin, zmax = float(fv[A.FA_ZREL_MIN]), float(fv[A.FA_ZREL_MAX])
        g.ap_m = zmax - zmin if math.isfinite(zmin) and math.isfinite(zmax) and zmax > zmin else 0.0
        if g.ae_m == 0.0 and g.in_cut:
            g.ae_m = float(fv[A.FA_RREL_MAX])
        g.engaged_length_m = float(fv[A.FA_ENGAGED_LEN])
        g.engaged_angle_rad = float(fv[A.FA_ANGLE_MAX])
        calls = int(iv[A.IA_SUBTRACTIONS]) + int(iv[A.IA_POLAR_STEPS])
        g.in_cut_fraction = float(iv[A.IA_IN_CUT_CALLS]) / calls if calls else 0.0
        g.air_cut_time_s = float(iv[A.IA_AIR_TICKS]) * ctx.ipo_s
        return g

    def _pass_update(self, rs, p, acc, vol, vx, vy, in_cut, tick) -> None:
        ps = rs.pass_stats[p]
        if in_cut:
            if not ps[PS_ACTIVE]:
                ps[:] = 0.0
                ps[PS_ACTIVE] = 1.0
                ps[PS_START_TICK] = float(acc.ivals[A.IA_FIRST_TICK]) if acc.ivals[A.IA_FIRST_TICK] >= 0 else tick
            ps[PS_V_FIELD] += vol
            ps[PS_V_X] += vx
            ps[PS_V_Y] += vy
            ps[PS_LAST_TICK] = tick

    def _pass_close(self, ctx, state, rs, frame, tick) -> None:
        s = rs.settings
        in_cut = {int(c.contact): bool(c.geometry.in_cut) for c in frame.contacts}
        for p in range(rs.pass_stats.shape[0]):
            ps = rs.pass_stats[p]
            if not ps[PS_ACTIVE] or in_cut.get(p, False):
                continue
            ps[PS_ACTIVE] = 0.0
            vf, va = float(ps[PS_V_FIELD]), float(ps[PS_V_ANALYTIC])
            if va <= 0.0 or vf <= 0.0:
                continue
            ratio = vf / va
            fams = [v for v in (vf, float(ps[PS_V_X]), float(ps[PS_V_Y])) if v > 0.0]
            spread = (max(fams) - min(fams)) / vf if len(fams) > 1 else 0.0
            allowed = s.volume_consistency_tol + spread
            ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "removal:pass",
                             {"contact": float(p), "field_m3": vf, "analytic_m3": va, "ratio": ratio,
                              "family_spread": spread, "start_tick": float(ps[PS_START_TICK])}, [])
            if abs(ratio - 1.0) > allowed:
                ctx.finding(state, DOMAIN, Category.MASS_BALANCE, Severity.WARNING,
                            f"removed field volume and analytic chip volume of contact {p} differ by "
                            f"{(ratio - 1.0) * 100:.1f} % over the pass (allowed {allowed * 100:.1f} %)",
                            int(ps[PS_LAST_TICK]), path=p,
                            quantities={"field_volume_m3": vf, "analytic_volume_m3": va, "ratio": ratio,
                                        "allowed": allowed})

    def _air_update(self, ctx, state, rs, p, acc) -> None:
        air = rs.air[p]
        ticks = int(acc.ivals[A.IA_AIR_TICKS])
        feed = int(acc.ivals[A.IA_FEED_TICKS])
        if ticks == 0:
            if air[AIR_ACTIVE] and (feed > 0 or acc.ivals[A.IA_IN_CUT_CALLS] > 0):
                air[:] = 0.0
            return
        if not air[AIR_ACTIVE]:
            air[:] = 0.0
            air[AIR_ACTIVE] = 1.0
            air[AIR_START_TICK] = float(state.clock.tick)
            air[AIR_X0:AIR_X0 + 3] = acc.fvals[A.FA_AIR_X0:A.FA_AIR_X0 + 3]
        air[AIR_TIME_S] += ticks * ctx.ipo_s
        if ticks < feed:
            # the run ended inside the step (contact again)
            self._air_report(ctx, state, rs, p, acc)
            air[:] = 0.0
            return
        self._air_report(ctx, state, rs, p, acc)

    def _air_report(self, ctx, state, rs, p, acc) -> None:
        air = rs.air[p]
        if air[AIR_REPORTED] or air[AIR_TIME_S] < rs.settings.air_cut_report_s or rs.settings.air_cut_report_s <= 0:
            return
        air[AIR_REPORTED] = 1.0
        x1 = acc.fvals[A.FA_AIR_X1:A.FA_AIR_X1 + 3]
        ctx.finding(state, DOMAIN, Category.AIR_CUTTING, Severity.INFO,
                    f"feed motion without material for {air[AIR_TIME_S]:.1f} s on path {p}", int(state.clock.tick),
                    path=p, quantities={"air_cut_time_s": float(air[AIR_TIME_S]),
                                        "x0_S_m": float(air[AIR_X0]), "y0_S_m": float(air[AIR_X0 + 1]),
                                        "z0_S_m": float(air[AIR_X0 + 2]), "x1_S_m": float(x1[0]),
                                        "y1_S_m": float(x1[1]), "z1_S_m": float(x1[2])})

    def _finding_for(self, ctx, state, rs, kind, contact, ev) -> None:
        if kind not in _FINDING:
            ctx.trace.record(ev["tick"] * ctx.tick_s, ev["tick"], DOMAIN, f"removal:{ev['kind']}",
                             {"contact": float(contact)}, [])
            return
        key = f"{kind}:{contact}"
        if key in rs.episodes:
            rs.episodes[key] = ev["tick"]
            return
        rs.episodes[key] = ev["tick"]
        category, severity, text = _FINDING[kind]
        q = {"depth_m": ev["depth_m"], "volume_m3": ev["volume_m3"], "x_S_m": ev["point_S_m"][0],
             "y_S_m": ev["point_S_m"][1], "z_S_m": ev["point_S_m"][2]}
        msg = text
        if "part" in ev:
            msg = f"{ev['part']} of the tool in material (collision)"
        parent = ctx.trace.record(ev["tick"] * ctx.tick_s, ev["tick"], DOMAIN, f"removal:{ev['kind']}",
                                  dict(q, contact=float(contact)), [])
        ctx.finding(state, DOMAIN, category, severity, f"{msg}: {ev['depth_m'] * 1e3:.3f} mm deep, "
                    f"{ev['volume_m3'] * 1e9:.3f} mm^3", ev["tick"], path=max(contact, 0), quantities=q,
                    parents=[parent])

    # ------------------------------------------------------------------ macro helpers
    def _loose(self, ctx, state, rs, w, frame, tick, force=False) -> None:
        from .connect import release_loose_incremental
        from ..db.models import params

        wp = rs.workpieces[w]
        if not force and wp.field.info[A.FI_TOPO] <= 0:
            return
        # the operation's end labels the whole field (force); every macro step checks the changed region only while
        # the material is known to be one component (connect.release_loose_incremental)
        released, held, removed_q, wp.one_body, _local = release_loose_incremental(
            wp.field, wp.held_boxes, wp.one_body and not force)
        if not released:
            return
        pr = params("removal")
        slug = float(pr["slug_max_cube_edge_m"]) ** 3
        swarf = float(pr["swarf_cube_edge_m"]) ** 3
        for b in released:
            mass = b.volume_m3 * wp.rho_kg_m3
            kind = "part_free" if b.volume_m3 > slug else "chip_hazard"
            rec = {"workpiece": w, "volume_m3": b.volume_m3, "mass_kg": mass,
                   "centroid_S_m": [float(v) for v in b.centroid_S_m], "tick": tick, "kind": kind}
            frame.loose_bodies.append(rec)
            rs.loose_log.append(rec)
            if b.volume_m3 < swarf:
                continue
            category = Category.PART_FREE if kind == "part_free" else Category.CHIP_HAZARD
            ctx.finding(state, DOMAIN, category, Severity.WARNING,
                        f"a body of {b.volume_m3 * 1e9:.0f} mm^3 ({mass * 1e3:.1f} g) came loose from the workpiece",
                        tick, quantities={"volume_m3": b.volume_m3, "mass_kg": mass,
                                          "x_S_m": float(b.centroid_S_m[0]), "y_S_m": float(b.centroid_S_m[1]),
                                          "z_S_m": float(b.centroid_S_m[2])})

    def _sync_polar(self, ctx, state, w, dirty_only: bool) -> None:
        from .k_polar import refresh_zmax, rebuild_tridexel

        rs = state.removal
        wp = rs.workpieces[w]
        pf = wp.polar
        if dirty_only:
            rng = PO.dirty_z_range(pf)
            if rng is None:
                return
            zlo, zhi = rng
        else:
            zlo = float(pf.geom[A.PG_Z0])
            zhi = zlo + float(pf.geom[A.PG_NZ]) * float(pf.geom[A.PG_DZ])
        ev = event_buffer(16, A.EVENT_WIDTH)
        tick = int(state.clock.tick)
        wp.one_body = False             # the rebuild may add material: the next connectivity check labels everything
        while True:
            status = rebuild_tridexel(wp.field, pf, 0, zlo, zhi, wp.target, rs.settings.gouge_tolerance_m, ev, tick,
                                      wp.next_sev, rs.scratch)
            if status == 0:
                break
            wp.field = FD.grow_pool(wp.field, max(int(wp.field.info[A.FI_N_BLOCKS]), 1024))
        for lvl in range(1, FD.n_levels(wp.field)):
            rebuild_tridexel(wp.field, pf, lvl, zlo, zhi, A.empty_target(int(wp.field.info[A.FI_NRAYS])),
                             rs.settings.gouge_tolerance_m, ev, tick, wp.next_sev, rs.scratch)
        PO.clear_dirty(pf)
        refresh_zmax(pf, 0)
        wp.polar_turned_rad = 0.0
        for i in range(int(ev.n[0])):
            row = ev.rec[i]
            evd = {"kind": A.EV_NAMES[int(row[A.EC_KIND])], "tick": tick, "contact": -1,
                   "point_S_m": [float(row[A.EC_X]), float(row[A.EC_Y]), float(row[A.EC_Z])],
                   "depth_m": float(row[A.EC_DEPTH]), "volume_m3": float(row[A.EC_VOLUME]),
                   "surface_event": int(row[A.EC_SEV_ID]), "level": int(row[A.EC_LEVEL])}
            if isinstance(state.bus_next.removal, RemovalFrame):
                state.bus_next.removal.events.append(evd)
            self._finding_for(ctx, state, rs, int(row[A.EC_KIND]), 0, evd)

    @staticmethod
    def _spindle_turned(prev, tick_s: float, ipo_ticks: int) -> float:
        """Largest spindle angle any path turned in the previous step (rad): sum of |omega| per L1 tick x the
        L1 period (architecture 5.1: ipo_ticks base ticks of tick_s)."""
        total = 0.0
        for frame in getattr(prev.motion, "paths", []) or []:
            omega = np.asarray(getattr(frame, "spindle_omega_rad_s", []), dtype=float)
            if omega.size:
                period = float(getattr(frame, "ipo_ticks", ipo_ticks)) * tick_s
                total = max(total, float(np.sum(np.abs(omega))) * period)
        return total

    def _remaining(self, ctx, state, w, tick) -> None:
        from .k_drive import excess_material

        rs = state.removal
        wp = rs.workpieces[w]
        if not wp.target.info[0]:
            return
        out = np.zeros(6)
        excess_material(wp.field, wp.target, rs.settings.gouge_tolerance_m, out)
        ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "removal:remaining_material",
                         {"volume_m3": float(out[0]), "max_m": float(out[1]), "rays": float(out[5])}, [])
        if out[1] > rs.settings.gouge_tolerance_m:
            ctx.finding(state, DOMAIN, Category.REMAINING_MATERIAL, Severity.WARNING,
                        f"material outside the CAD part remains: {out[0] * 1e9:.1f} mm^3, up to {out[1] * 1e3:.3f} mm "
                        f"thick", tick, quantities={"volume_m3": float(out[0]), "max_thickness_m": float(out[1]),
                                                    "x_S_m": float(out[2]), "y_S_m": float(out[3]),
                                                    "z_S_m": float(out[4])})

    def _transfer(self, ctx, state, next_op) -> None:
        from . import transfer as TR

        rs = state.removal
        from_previous = _get(next_op, "from_previous")
        for w, wp in enumerate(rs.workpieces):
            R = TR.transform_of(from_previous, wp.kind)
            if R is None:
                continue
            length = float(wp.field.lv_box_hi_m[0, 2] - wp.field.lv_box_lo_m[0, 2])
            new, T, note = TR.apply(wp.field, R, wp.kind, rs.settings.lattice_m, rs.settings.margin_nodes)
            if wp.polar is not None:
                wp.polar = TR.polar_transfer(wp.polar, R, length)
            wp.field = new
            wp.one_body = False
            wp.removed_q = [[0, 0, 0]] + [[0, 0, 0] for _ in range(FD.n_levels(new) - 1)]
            wp.held_boxes = self._held_from_setup(ctx.project, int(getattr(state.clock, "op_index", 0)) + 1)
            wp.target = self._target_after(ctx, new, T, wp)
            if note:
                wp.notes.append(note)
            ctx.trace.record(state.clock.tick * ctx.tick_s, state.clock.tick, DOMAIN, "removal:transfer",
                             {"workpiece": float(w), "volume_m3": FD.volume_m3(new, 0, 2)}, [])

    def _target_after(self, ctx, new_field, T, wp) -> A.TargetField:
        cad = _get(ctx.project, "cad")
        if cad is None or not _get(cad, "path"):
            return A.empty_target(int(new_field.info[A.FI_NRAYS]))
        V, Fc = SK.read_mesh_file(_get(cad, "path"), unit=str(_get(cad, "units", "mm")))
        pose = np.asarray(_get(cad, "cad_pose_S_C", np.eye(4)), dtype=float)
        wp.__dict__.setdefault("T_S_S0", np.eye(4))
        wp.T_S_S0 = T @ wp.T_S_S0
        target, _ = SK.target_from_mesh(new_field, SK.transform_points(V, wp.T_S_S0 @ pose), Fc)
        return target


# --------------------------------------------------------------------------- tool shapes
def _slab_h(settings: RemovalSettings) -> float:
    return max(settings.lattice_m, 0.5e-3)


def _slab_height(state, slot) -> float:
    rs = state.removal
    return _slab_h(rs.settings)


def _tan_ramp_max() -> float:
    from ..db.models import params

    return math.tan(float(params("tooling")["end_dish_rad"]))


def slabs_for(edge, prof: A.Profile, slab_height_m: float) -> A.ToothSlabs:
    nt = int(edge.elem_r_m.shape[0])
    rows = [PR.tooth_slabs(edge, prof, j, slab_height_m) for j in range(nt)]
    ns = max(len(dr) for _, dr in rows) if rows else 1
    z = np.zeros((max(nt, 1), ns + 1))
    dr = np.full((max(nt, 1), ns), -1e9)
    n = np.zeros(max(nt, 1), np.int32)
    for j, (zz, dd) in enumerate(rows):
        z[j, :len(zz)] = zz
        z[j, len(zz):] = zz[-1]
        dr[j, :len(dd)] = dd
        n[j] = len(dd)
    return A.ToothSlabs(np.ascontiguousarray(z), np.ascontiguousarray(dr), n)


def build_shape(geo, edge, slot: int, slab_height_m: float, version: int = 0) -> ToolShape:
    """ToolShape of one tool slot from tooling's ToolGeometry and its effective EdgeModel."""
    kind_name = str(getattr(geo, "type", ""))
    rotating = bool(getattr(geo, "rotating", True))
    hand = float(getattr(geo, "hand_sign", -1.0))
    cut = PR.profile_from_envelope(geo.envelope, "cutting") if rotating else A.empty_profile()
    if rotating and kind_name == "thread_mill" and getattr(geo, "thread", None) is not None:
        cut = PR.profile_from_thread(geo.thread, 0)
    whole = PR.profile_from_envelope(geo.envelope, "all") if rotating else A.empty_profile()
    insert = PO.insert_region(geo.insert) if (not rotating and getattr(geo, "insert", None) is not None) \
        else A.empty_insert()
    helix = PR.helix_from_thread(geo.thread, hand) if (rotating and kind_name in TAPS and
                                                       getattr(geo, "thread", None) is not None) else A.empty_helix()
    if not rotating:
        kind = A.TK_TURNING
    elif kind_name in TAPS and len(helix.n):
        kind = A.TK_HELIX
    else:
        kind = A.TK_ENVELOPE
    spec = getattr(geo, "spec", None)
    centre = True
    geometry = getattr(spec, "geometry", None) if spec is not None else None
    flag = _get(geometry, "centre_cutting") if geometry is not None else None
    r_dead = float(cut.info[A.PI_RDEAD]) if len(cut.z0_m) else 0.0
    if flag is not None:
        centre = bool(flag)
    else:
        centre = r_dead <= 0.0
    if not centre and r_dead <= 0.0:
        props = getattr(geo, "props", None)
        r_dead = 0.5 * float(getattr(props, "core_diameter_m", 0.0) or 0.0) if props is not None else 0.0
    radius = float(cut.info[A.PI_RMAX]) if len(cut.z0_m) else 0.0
    z_dead = radius * _tan_ramp_max() if not centre else 0.0
    slabs = slabs_for(edge, cut, slab_height_m) if (kind == A.TK_ENVELOPE and len(cut.z0_m)) else A.empty_slabs()
    return ToolShape(slot=slot, kind=kind, rotating=rotating, centre_cutting=centre, hand_sign=hand, cut=cut,
                     whole=whole, insert=insert, helix=helix, slabs=slabs, r_dead_m=r_dead if not centre else 0.0,
                     z_dead_m=z_dead, radius_m=radius, edge_version=version)


def _m_from_s(state, p) -> tuple[np.ndarray, np.ndarray]:
    """M <- S of path p at the end of the previous step (motion's record), identity without it."""
    paths = list(getattr(state.bus_prev.motion, "paths", []) or [])
    if p < len(paths):
        f = paths[p]
        M_T = np.asarray(getattr(f, "pose_M_T", np.eye(4)), dtype=float)
        S_T = np.asarray(getattr(f, "pose_S_T", np.eye(4)), dtype=float)
        M_S = M_T @ np.linalg.inv(S_T)
        return M_S[:3, :3], M_S[:3, 3]
    return np.eye(3), np.zeros(3)


def _replace(settings: RemovalSettings, **changes) -> RemovalSettings:
    import dataclasses

    return dataclasses.replace(settings, **changes)


_EMPTY_POLAR = PO.empty_polar()
