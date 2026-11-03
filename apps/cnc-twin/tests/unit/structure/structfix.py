"""Small inline fixtures for the structure tests: materials, a project namespace, holding-style supports of a
vise, a thermal grid and bus frames (no database files: the real databases are still being researched)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from cnctwin.core.bus import Bus, HoldingFeedback, MotionFrame, PartHolding, PathFrame
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState, ToolSlot, ToolTable
from cnctwin.db.simap import SIMap

ALU = {"id": "test-al", "family": "aluminium",
       "physical": {"density_kg_m3": {"value": 2700, "src": "est"}, "youngs_modulus_GPa": {"value": 70, "src": "est"},
                    "poisson": {"value": 0.33, "src": "est"},
                    "expansion_1_K": {"table": [[20, 23.0e-6], [200, 23.0e-6]], "src": "est"}},
       "residual_stress": []}
STEEL = {"id": "test-steel", "family": "steel",
         "physical": {"density_kg_m3": {"value": 7850, "src": "est"}, "youngs_modulus_GPa": {"value": 210, "src": "est"},
                      "poisson": {"value": 0.3, "src": "est"},
                      "expansion_1_K": {"table": [[20, 11.5e-6], [500, 11.5e-6]], "src": "est"}}}

MACHINE = {"spindles": [{"id": "main", "frf_modes": [
    {"fn_Hz": 950, "zeta": 0.04, "k_N_um": 70, "direction": "x", "src": "est", "note": "test"},
    {"fn_Hz": 950, "zeta": 0.04, "k_N_um": 70, "direction": "y", "src": "est", "note": "test"}]}]}


def material(base=ALU, profile=None, form="rolled_plate", thickness_mm=20.0):
    m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    if profile is not None:
        m["residual_stress"] = [{"form": form, "thickness_mm": thickness_mm, "direction": "L",
                                 "profile_MPa": [list(r) for r in profile], "src": "est"}]
    return SIMap(m)


def project(stock=None, mat=None, physics=None, machine=None, ops=1, form="rolled_plate"):
    stock = stock or {"kind": "block", "size_mm": [40, 30, 12], "form": form}
    operations = [{"name": f"OP{10 * (i + 1)}", "setup": {"workpiece": 0}} for i in range(ops)]
    return SimpleNamespace(stock=SIMap(stock), material=mat if mat is not None else material(),
                           operations=operations, physics={"structure": physics or {}},
                           machine=SIMap(machine) if machine is not None else None, stock_pose_W_S=np.eye(4))


def context(proj, seed=7, macro_ipo=10) -> RunContext:
    return RunContext(project=proj, db=None, run_dir=None, seed=seed, tick_s=1e-6, ipo_ticks=1000, macro_ipo=macro_ipo)


def sim_state(slots=0) -> SimState:
    state = SimState()
    state.tools = ToolTable(tuple(ToolSlot(slot=i, number=i + 1, tool_id=f"t{i}", holder_id="h") for i in range(slots)))
    return state


def _patch(pid, law, pts, normal, k, kt, preload=None, role="base"):
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    n = np.tile(np.asarray(normal, dtype=float), (len(pts), 1))
    t1 = np.tile(np.cross(normal, [0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0]),
                 (len(pts), 1))
    pre = np.zeros((len(pts), 3)) if preload is None else np.tile(np.asarray(preload, float), (len(pts), 1))
    return {"id": pid, "role": role, "law": law, "device": "vise", "group": "main", "pos_S_m": pts, "normal_S": n,
            "t1_S": t1, "force_S_N": pre.copy(), "preload_S_N": pre, "stiffness_N_per_m": np.full(len(pts), k),
            "tangential_stiffness_N_per_m": np.full((len(pts), 2), kt), "friction": 0.2,
            "area_m2": np.full(len(pts), 1e-5), "state": np.zeros(len(pts), np.int8), "rotating": False}


def vise_supports(size_m=(0.04, 0.03, 0.012), clamp_N=2000.0, k=2e8, grip_z=(0.002, 0.008)):
    """Base on the bottom (z = 0), fixed jaw on y = 0 (locator), moving jaw on y = sy (locked, clamp force)."""
    sx, sy, sz = size_m
    xs = np.linspace(0.1 * sx, 0.9 * sx, 3)
    base = [[x, y, 0.0] for x in xs for y in np.linspace(0.2 * sy, 0.8 * sy, 2)]
    jaw0 = [[x, 0.0, z] for x in xs for z in grip_z]
    jaw1 = [[x, sy, z] for x in xs for z in grip_z]
    per = clamp_N / len(jaw1)
    return [_patch("base", "locator", base, [0.0, 0.0, 1.0], k, 0.8 * k),
            _patch("fixed_jaw", "locator", jaw0, [0.0, 1.0, 0.0], k, 0.8 * k, role="fixed_jaw"),
            _patch("moving_jaw", "locked", jaw1, [0.0, -1.0, 0.0], k, 0.8 * k, preload=[0.0, -per, 0.0],
                   role="moving_jaw")]


def support_stiffness(supports) -> tuple[np.ndarray, np.ndarray]:
    """6 x 6 rigid-body stiffness about the supports' centroid (as holding publishes it)."""
    pts = np.concatenate([p["pos_S_m"] for p in supports])
    c = pts.mean(axis=0)
    K = np.zeros((6, 6))
    for p in supports:
        for i in range(len(p["pos_S_m"])):
            n = p["normal_S"][i]
            Kp = p["stiffness_N_per_m"][i] * np.outer(n, n)
            r = p["pos_S_m"][i] - c
            A = np.hstack([np.eye(3), -np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])])
            K += A.T @ Kp @ A
    return K, c


def bus_with(supports=None, version=1, free=False, thermal=None, tick=10_000, slot=-1, lathe=None) -> Bus:
    bus = Bus()
    frame = PathFrame(path=0, tick0=tick - 10_000, tick=np.arange(1, 11, dtype=np.int64) * 1000 + tick - 10_000,
                      block_index=np.zeros(10, np.int32), tool_slot=slot)
    bus.motion = MotionFrame(time_s=tick * 1e-6, tick=tick, paths=[frame])
    if supports is not None:
        K6, c = support_stiffness(supports) if supports else (np.zeros((6, 6)), np.zeros(3))
        bus.holding = HoldingFeedback(parts={0: PartHolding(supports=supports, supports_version=version, free=free,
                                                            support_stiffness_S=K6, centroid_S_m=c,
                                                            lathe=lathe or {})})
    if thermal is not None:
        origin, cell, T = thermal
        bus.thermal.work_temp_origin_S_m[0] = np.asarray(origin, float)
        bus.thermal.work_temp_cell_m[0] = float(cell)
        bus.thermal.work_temp_K[0] = np.asarray(T, float)
    return bus


def thermal_grid(lo, hi, cell, T_of_xyz):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    shape = tuple(int(math.ceil((hi[a] - lo[a]) / cell - 1e-9)) for a in range(3))
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
    centres = lo + (idx + 0.5) * cell
    return lo, cell, T_of_xyz(centres)


def run(dom, ctx, state, bus, steps=1):
    record = None
    for _ in range(steps):
        state.bus_prev = bus
        state.bus_next = Bus()
        dom.macro_step(ctx, state, bus, state.bus_next)
        record = state.bus_next.structure
        state.clock.macro += 1
        state.clock.tick += 10_000
    return record
