"""Small inline fixtures for the surface tests: tools (edge models with consistent outlines), motion frames,
synthetic surface events that follow cutting's conventions (bus SURFACE_EVENT_COLUMNS), projects, contexts
and buses (the real databases are still being researched)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from cnctwin.core.abi import EDGE_CORNER, EDGE_END, EDGE_NOSE, EDGE_PERIPHERAL, EdgeModel
from cnctwin.core.bus import (Bus, CuttingContact, CuttingFrame, MotionFrame, PathFrame, RemovalContact, RemovalFrame,
                              ToolContact)
from cnctwin.core.domain import RunContext
from cnctwin.core.rundir import Timeline
from cnctwin.core.state import SimState, ToolSlot, ToolTable
from cnctwin.surface.protocols import (StaticToolInfo, ToolOutline, ball_outline, endmill_outline,
                                       turning_insert_outline)

TICK_S = 1e-6
IPO = 1000

MATERIAL = {
    "id": "test-steel", "name": "test steel", "family": "steel",
    "physical": {"density_kg_m3": {"value": 7850}, "youngs_modulus_GPa": {"value": 210}, "poisson": {"value": 0.29},
                 "expansion_1_K": {"value": 11.5e-6}, "conductivity_W_mK": {"value": 49.8},
                 "specific_heat_J_kgK": {"value": 486}, "yield_MPa": {"value": 505}, "ultimate_MPa": {"value": 655},
                 "elongation_pct": {"value": 16}, "hardness": {"HB": 200}},
    "cutting": {"min_chip_thickness_ratio": {"value": 0.3},
                "work_hardening": {"layer_depth_um": {"value": 60}, "hardness_increase_pct": {"value": 25}}},
    "thermal_damage": {"burn_degC": {"value": 400}, "white_layer_degC": {"value": 727}},
    "friction": [{"tool_class": "carbide", "coating": "uncoated", "mu_sliding": {"value": 0.4}}],
}


# ------------------------------------------------------------------------------------------- tools
def edge_from(nt, r, z, kappa, kind, station=None, runout=None, edge_radius=10e-6, phase=None):
    """EdgeModel with every tooth sharing the element layout (r, z, kappa per element)."""
    ne = len(r)
    full = lambda v: np.tile(np.asarray(v, dtype=float), (nt, 1))  # noqa: E731
    st = np.arange(ne, dtype=np.int32) if station is None else np.asarray(station, dtype=np.int32)
    ph = np.arange(nt) * 2.0 * math.pi / nt if phase is None else np.asarray(phase, dtype=float)
    return EdgeModel(ph, full(z), full(r), np.zeros((nt, ne)), np.full((nt, ne), 1e-4), np.full((nt, ne), 1e-4),
                     full(kappa), np.full((nt, ne), 0.1), np.zeros((nt, ne)), np.full((nt, ne), 0.1),
                     np.full((nt, ne), float(edge_radius)), np.tile(np.asarray(kind, dtype=np.int32), (nt, 1)),
                     np.zeros((nt, ne), np.int32), np.tile(st, (nt, 1)), np.ones((nt, ne), np.int8),
                     np.zeros(3) if runout is None else np.asarray(runout, dtype=float), np.zeros(2),
                     np.asarray(z, dtype=float), np.asarray(r, dtype=float), np.array([-1.0]))


def endmill(R=5e-3, nt=4, corner=0.5e-3, dish=math.radians(1.0), n_side=8, n_end=8, length=0.02, edge_radius=10e-6):
    """End mill: n_end end elements from the core to the corner start (dished), a corner element at the corner
    bottom, n_side peripheral elements; the outline and the elements agree."""
    r_end = R - corner
    core = 0.4 * R
    r_e = list(np.linspace(core, r_end, n_end + 1)[:-1] + 0.5 * (r_end - core) / n_end)
    z_e = [(r_end - r) * math.tan(dish) for r in r_e]
    k_e = [-dish] * n_end
    kinds = [EDGE_END] * n_end
    # one corner element at the bottom of the corner arc (kappa = dish)
    r_e.append(r_end + corner * math.sin(dish))
    z_e.append(corner * (1.0 - math.cos(dish)))
    k_e.append(dish)
    kinds.append(EDGE_CORNER)
    z_s = list(np.linspace(corner, length, n_side + 1)[:-1] + 0.5 * (length - corner) / n_side)
    r_e += [R] * n_side
    z_e += z_s
    k_e += [math.pi / 2.0] * n_side
    kinds += [EDGE_PERIPHERAL] * n_side
    edge = edge_from(nt, r_e, z_e, k_e, kinds, edge_radius=edge_radius)
    outline = ToolOutline(True, endmill_outline(R, corner, dish, length, core), nt, corner, R, "square_endmill")
    return edge, outline


def ballmill(R=5e-3, nt=2, n_ball=24, length=0.02, edge_radius=8e-6):
    k = (np.arange(n_ball) + 0.5) / n_ball * (math.pi / 2.0)
    r = R * np.sin(k)
    z = R * (1.0 - np.cos(k))
    edge = edge_from(nt, r, z, k, [EDGE_CORNER] * n_ball, edge_radius=edge_radius)
    outline = ToolOutline(True, ball_outline(R, length), nt, R, R, "ball_endmill")
    return edge, outline


def insert(nose=0.8e-3, kr=math.radians(93.0), kr2=math.radians(27.0), edge_radius=25e-6):
    """Turning insert: one nose element at the deepest point (x_T = 0, z_T = nose)."""
    edge = edge_from(1, [0.0], [nose], [math.pi / 2.0], [EDGE_NOSE], edge_radius=edge_radius, phase=[0.0])
    outline = ToolOutline(False, turning_insert_outline(nose, kr, kr2), 1, nose, 0.0, "turning_insert",
                          kappa_major_rad=kr)
    return edge, outline


def tool_info(parts: dict) -> StaticToolInfo:
    return StaticToolInfo({s: p[0] for s, p in parts.items()}, {s: p[1] for s, p in parts.items()})


# ------------------------------------------------------------------------------------------- motion
def frame(ticks, tip, omega, axis=(0.0, 0.0, 1.0), pose_W0_T=None, blocks=None, path=0, slot=0, workpiece=0,
          axis_names=("X", "Y", "Z"), axis_cmd=None, ferr=None, feed=None):
    ticks = np.asarray(ticks, dtype=np.int64)
    n = len(ticks)
    tip = np.asarray(tip, dtype=float).reshape(n, 3)
    pose = np.eye(4)
    pose[:3, 3] = tip[-1]
    fr = PathFrame(path=path, tick0=int(ticks[0]) - IPO, ipo_ticks=IPO, tick=ticks.copy(),
                   block_index=np.zeros(n, np.int32) if blocks is None else np.asarray(blocks, np.int32),
                   tip_S_m=tip.copy(), tip_M_m=tip.copy(), axis_T_M=np.tile(np.asarray(axis, dtype=float), (n, 1)),
                   axis_names=tuple(axis_names),
                   axis_actual=np.zeros((n, len(axis_names))) if axis_cmd is None else np.asarray(axis_cmd) - (
                       0.0 if ferr is None else np.asarray(ferr)),
                   axis_cmd=np.zeros((n, len(axis_names))) if axis_cmd is None else np.asarray(axis_cmd, dtype=float),
                   following_error=np.zeros((n, len(axis_names))) if ferr is None else np.asarray(ferr, dtype=float),
                   spindle_angle_rad=np.zeros(n), spindle_rev=np.zeros(n, np.int64),
                   spindle_omega_rad_s=np.broadcast_to(np.asarray(omega, dtype=float), (n,)).copy(),
                   feed_vel_M_m_s=np.zeros((n, 3)) if feed is None else np.asarray(feed, dtype=float).reshape(n, 3),
                   rapid=np.zeros(n, bool), tool_slot=slot, workpiece=workpiece, pose_M_T=pose.copy(),
                   pose_S_T=pose.copy(), pose_W0_T=np.eye(4) if pose_W0_T is None else np.asarray(pose_W0_T))
    return fr


def bus(frames, contacts, removal=None, wear=None, thermal=None, dynamics=None) -> Bus:
    b = Bus()
    b.motion = MotionFrame(time_s=float(frames[0].tick[-1]) * TICK_S, tick=int(frames[0].tick[-1]), paths=list(frames))
    b.cutting = CuttingFrame(contacts=list(contacts))
    if removal is not None:
        b.removal = RemovalFrame(contacts=list(removal))
    if wear is not None:
        b.wear.tools = dict(wear)
    if thermal is not None:
        b.thermal.contact_surface_temp_K = dict(thermal)
    if dynamics is not None:
        for k, v in dynamics.items():
            setattr(b.dynamics, k, v)
    return b


def contact(events, slot=0, contact_id=0, shocks=None, recut=None, in_cut_s=0.005, ploughing_J=0.0, speed=2.0,
            edge_len=2e-3, h_max=5e-5, wrench=None):
    c = CuttingContact(contact=contact_id, tool_slot=slot, in_cut_s=in_cut_s)
    c.surface_events = np.asarray(events, dtype=float).reshape(-1, 16)
    c.shocks = np.zeros((0, 8)) if shocks is None else np.asarray(shocks, dtype=float).reshape(-1, 8)
    c.recut = np.zeros((0, 6)) if recut is None else np.asarray(recut, dtype=float).reshape(-1, 6)
    c.ploughing_J = ploughing_J
    c.mean_speed_m_s = speed
    c.engaged_edge_m = edge_len
    c.h_max_step_m = h_max
    if wrench is not None:
        c.wrench_tool_T0_mean = np.asarray(wrench, dtype=float).reshape(-1, 6)
    return c


def removal_contact(contact_id=0, slot=0, path=0, workpiece=0, ap=1e-3, ae=1e-3):
    return RemovalContact(contact=contact_id, tool_slot=slot, workpiece=workpiece, path=path,
                          geometry=ToolContact(in_cut=True, ap_m=ap, ae_m=ae))


def event_row(tick, tooth, elem, X, n, h=0.0, edge_r=10e-6, VB=0.0, u_rel=(0.0, 0.0, 0.0), slot=0):
    return [float(tick), float(tooth), float(elem), *map(float, X), *map(float, n), float(h), float(edge_r), float(VB),
            *map(float, u_rel), float(slot)]


# ------------------------------------------------------------------------------------------- synthetic cuts
def side_wall(R, nt, fz, omega, n_teeth_passes, stations, edge, runout_e=0.0, runout_phase=0.0, t0_tick=1000, edge_r=10e-6,
              u_rel=None, y_offset=0.0):
    """Peripheral finishing of the wall y = R (material at y > R) by a tool on the x axis feeding +x, M03
    (omega < 0): tooth j generates its envelope point when it points along +y; runout adds
    e cos(phase_j - phase_r) to its radius. Returns (events, ticks, tips)."""
    w = abs(omega)
    period = 2.0 * math.pi / (nt * w)
    v = fz / period
    events = []
    for q in range(n_teeth_passes):
        j = q % nt
        t = q * period
        tick = t0_tick + int(round(t / TICK_S))
        x_c = v * t
        r_j = R + runout_e * math.cos(2.0 * math.pi * j / nt - runout_phase)
        for k in stations:
            z = edge.elem_z_m[j, k]
            ur = (0.0, 0.0, 0.0) if u_rel is None else u_rel(q, j, k)
            X = (x_c, r_j + y_offset - ur[1], z)
            events.append(event_row(tick, j, k, X, (0.0, -1.0, 0.0), u_rel=ur, edge_r=edge_r))
    return np.array(events), v


def floor_trailing(R_list, z_list, nt, fz, omega, n_teeth_passes, elems, axial_runout=None, t0_tick=1000,
                   tilt_y=0.0, z0=0.0, y=0.0):
    """End elements writing their floor events at the trailing side (feed +x: the element at -x of the axis);
    a tool tilt about y (the axis leaning toward +x by tilt_y) raises the trailing side by 2 r tilt."""
    w = abs(omega)
    period = 2.0 * math.pi / (nt * w)
    v = fz / period
    events = []
    for q in range(n_teeth_passes):
        j = q % nt
        t = q * period
        tick = t0_tick + int(round(t / TICK_S))
        x_c = v * t
        dz = 0.0 if axial_runout is None else axial_runout[j]
        for k in elems:
            r = R_list[k]
            X = (x_c - r, y, z0 + z_list[k] + dz + r * tilt_y)
            events.append(event_row(tick, j, k, X, (0.0, 0.0, 1.0)))
    return np.array(events), v


def turning(nose, f, omega_work, revolutions, radius, z_start=0.0, t0_tick=1000, per_rev=36, slot=0):
    """Longitudinal turning toward -z at radius ``radius`` (S rotates with the work): one event per 10 deg
    of work rotation at the deepest nose point; the tool tip path in S per event."""
    w = abs(omega_work)
    rev_t = 2.0 * math.pi / w
    v = f / rev_t
    events, ticks, tips = [], [], []
    for q in range(revolutions * per_rev):
        t = q * rev_t / per_rev
        tick = t0_tick + int(round(t / TICK_S))
        theta = -w * t                                     # the tool moves the other way round in S
        z_tip = z_start - v * t
        X = (radius * math.cos(theta), radius * math.sin(theta), z_tip + nose)
        n = (math.cos(theta), math.sin(theta), 0.0)
        events.append(event_row(tick, 0, 0, X, n, edge_r=0.0, slot=slot))
        ticks.append(tick)
        tips.append((radius * math.cos(theta), radius * math.sin(theta), z_tip))
    return np.array(events), np.array(ticks), np.array(tips), v


def turning_pose():
    """pose_W0_T of an external turning tool at +x of the work axis: x_T = -x (into the material), z_T = +z."""
    pose = np.eye(4)
    pose[:3, 0] = [-1.0, 0.0, 0.0]
    pose[:3, 1] = [0.0, -1.0, 0.0]
    pose[:3, 2] = [0.0, 0.0, 1.0]
    pose[:3, 3] = [0.02, 0.0, 0.0]
    return pose


# ------------------------------------------------------------------------------------------- run objects
def project(material=None, physics=None, stock=None, tolerances=None):
    return SimpleNamespace(material=material or MATERIAL, physics={"surface": physics or {}, "lattice_mm": 0.25},
                           stock=stock or {"kind": "block", "size_mm": [60, 40, 30]},
                           cad={"tolerances": tolerances or []}, environment={"ambient_degC": 20.0},
                           tools={"list": []}, operations=[])


def timeline(kinds=("linear",), lines=None) -> Timeline:
    tl = Timeline()
    for i, kind in enumerate(kinds):
        tl.open(path=0, op="OP10", program="op10.nc", line=(lines[i] if lines else 10 + i), block=f"N{i}", kind=kind,
                tick0=0)
    return tl


def context(proj, seed=11, tl=None) -> RunContext:
    ctx = RunContext(project=proj, db=None, run_dir=None, seed=seed, tick_s=TICK_S, ipo_ticks=IPO, macro_ipo=10)
    ctx.timeline = tl if tl is not None else timeline()
    return ctx


def sim_state(n_slots=1) -> SimState:
    state = SimState()
    state.tools = ToolTable(tuple(ToolSlot(slot=s, number=s + 1, tool_id=f"t{s}", holder_id="") for s in range(n_slots)))
    return state


def chunks(events: np.ndarray, ticks_per_step: int):
    """[(t0, rows)] of the macro steps (t0, t0 + ticks_per_step] that hold events, in order."""
    if len(events) == 0:
        return []
    t = events[:, 0]
    k = np.floor((t - 1.0) / ticks_per_step).astype(int)
    return [(int(i) * ticks_per_step, events[k == i]) for i in range(int(k.min()), int(k.max()) + 1)
            if np.any(k == i)]


def step_ticks(t0: int, ticks_per_step: int) -> np.ndarray:
    """The L1 tick ends of the step starting at t0."""
    return np.arange(t0 + IPO, t0 + ticks_per_step + 1, IPO, dtype=np.int64)


# ------------------------------------------------------------------------------------------- scenario drivers
def run_turning(f, nose=0.8e-3, kr=math.radians(93.0), kr2=math.radians(27.0), edge_r=0.0, revolutions=40,
                rpm=1000.0, radius=0.02, per_rev=1, wear=None, physics=None, material=None, seed=11, edge=None,
                outline=None, steps_per_call=10, elem=0):
    """Longitudinal turning through SurfaceDomain (one event per revolution by default: one profile line).
    Returns (domain, ctx, state)."""
    from cnctwin.surface.domain import SurfaceDomain
    omega = -2.0 * math.pi * rpm / 60.0
    if edge is None or outline is None:
        edge, outline = insert(nose, kr, kr2, edge_radius=edge_r)
    ev, ticks, tips, v = turning(nose, f, omega, revolutions, radius, per_rev=per_rev)
    ev[:, 10] = edge_r
    ev[:, 2] = elem
    proj = project(stock={"kind": "cylinder", "size_mm": [50, 60]}, physics=physics, material=material)
    ctx = context(proj, seed=seed)
    state = sim_state()
    dom = SurfaceDomain(tools=tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    step = steps_per_call * IPO
    w = abs(omega)
    for t0, rows in chunks(ev, step):
        tk = step_ticks(t0, step)
        tt = (tk - 1000) * TICK_S
        theta = -w * tt
        tip = np.stack([radius * np.cos(theta), radius * np.sin(theta), -v * tt], axis=1)
        fr = frame(tk, tip, omega, pose_W0_T=turning_pose())
        b = bus([fr], [contact(rows, speed=w * radius)], removal=[removal_contact(ap=1e-3)], wear=wear)
        dom.macro_step(ctx, state, b, bus([fr], []))
    dom.end_operation(ctx, state)
    return dom, ctx, state


def run_wall(fz=0.05e-3, R=5e-3, nt=4, rpm=10000.0, passes=200, runout=0.0, edge_r=0.0, stations=(12,), u_rel=None,
             physics=None, seed=11):
    from cnctwin.surface.domain import SurfaceDomain
    omega = -2.0 * math.pi * rpm / 60.0
    edge, outline = endmill(R, nt)
    ev, v = side_wall(R, nt, fz, omega, passes, list(stations), edge, runout_e=runout, edge_r=edge_r, u_rel=u_rel)
    ctx = context(project(physics=physics), seed=seed)
    state = sim_state()
    dom = SurfaceDomain(tools=tool_info({0: (edge, outline)}))
    dom.init(ctx, state)
    step = 10 * IPO
    for t0, rows in chunks(ev, step):
        tk = step_ticks(t0, step)
        tt = (tk - 1000) * TICK_S
        tip = np.stack([v * tt, 0 * tt, 0 * tt], axis=1)
        fr = frame(tk, tip, omega)
        b = bus([fr], [contact(rows)], removal=[removal_contact(ap=10e-3)])
        dom.macro_step(ctx, state, b, bus([fr], []))
    dom.end_operation(ctx, state)
    return dom, ctx, state, v


def patch_values(state, patch=0):
    from cnctwin.surface.faces import patch_roughness
    from cnctwin.surface.store import PF
    pf = state.surface.store.pf[patch]
    out = patch_roughness(pf) or {}
    out["Pt_m"] = float(pf[PF["f_pt_max"]])
    return out


def run_floor(edge, outline, elems, fz=0.05e-3, rpm=10000.0, nt=None, passes=((0.0, 0.0, 0.0),), teeth_passes=120,
              tilt_y=0.0, physics=None, seed=11, gap_s=0.2, x_shift=0.0, edge_r=0.0, tolerances=None, run_dir=None,
              meshes=None, stop_after=None, state=None, ctx=None, dom=None, finish=True, axial_runout=None):
    """Floor finishing passes along +x (one per (y, z offset, u_rel . n) entry, separated by gap_s): the
    axial-normal elements write their events at the trailing side. Returns (domain, ctx, state)."""
    from cnctwin.surface.domain import SurfaceDomain
    nt = nt or int(np.asarray(edge.tooth_phase_rad).shape[0])
    omega = -2.0 * math.pi * rpm / 60.0
    r_list = np.asarray(edge.elem_r_m)[0]
    z_list = np.asarray(edge.elem_z_m)[0]
    if dom is None:
        ctx = context(project(physics=physics, tolerances=tolerances), seed=seed)
        ctx.run_dir = run_dir
        state = sim_state()
        dom = SurfaceDomain(tools=tool_info({0: (edge, outline)}), meshes=meshes)
        dom.init(ctx, state)
    step = 10 * IPO
    t_start = 1000
    count = 0
    for (y, dz, un) in passes:
        ev, v = floor_trailing(r_list, z_list, nt, fz, omega, teeth_passes, list(elems), t0_tick=t_start,
                               tilt_y=tilt_y, z0=dz, y=y, axial_runout=axial_runout)
        ev[:, 3] += x_shift
        ev[:, 10] = edge_r
        ev[:, 12:15] = [0.0, 0.0, un]               # u_rel along the floor normal (+z)
        for t0, rows in chunks(ev, step):
            count += 1
            if stop_after is not None and count <= stop_after[0]:
                continue
            if stop_after is not None and count > stop_after[1]:
                return dom, ctx, state
            tk = step_ticks(t0, step)
            tt = (tk - t_start) * TICK_S
            tip = np.stack([x_shift + v * tt, np.full(len(tt), y), np.full(len(tt), dz)], axis=1)
            fr = frame(tk, tip, omega, axis=(math.sin(tilt_y), 0.0, math.cos(tilt_y)))
            b = bus([fr], [contact(rows)], removal=[removal_contact(ap=0.5e-3)])
            dom.macro_step(ctx, state, b, bus([fr], []))
        t_start = int(ev[-1, 0]) + int(gap_s / TICK_S)
        # an empty step after the gap closes the pass
        count += 1
        if stop_after is None or stop_after[0] < count <= stop_after[1]:
            tk = step_ticks((t_start // step) * step, step)
            fr = frame(tk, np.zeros((len(tk), 3)), omega)
            dom.macro_step(ctx, state, bus([fr], []), bus([fr], []))
        if stop_after is not None and count > stop_after[1]:
            return dom, ctx, state
    if finish:
        dom.end_operation(ctx, state)
    return dom, ctx, state


def envelope_pt(template_s, template_w, positions, heights=None, lo=None, hi=None, du=2.5e-7):
    """Brute-force lower envelope (independent of the surface code): copies of a profile template at the given
    positions (and heights); peak-to-valley over [lo, hi]."""
    order = np.argsort(template_s)
    s, w = np.asarray(template_s)[order], np.asarray(template_w)[order]
    heights = np.zeros(len(positions)) if heights is None else np.asarray(heights)
    grid = np.arange(lo, hi, du)
    best = np.full(len(grid), np.inf)
    for p, h in zip(positions, heights):
        best = np.minimum(best, h + np.interp(grid - p, s, w, left=np.inf, right=np.inf))
    return float(np.max(best) - np.min(best)), grid, best
