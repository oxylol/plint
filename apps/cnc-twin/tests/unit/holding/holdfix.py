"""Small inline fixtures for the holding tests: workholding items, a project with one operation, tool data,
bus frames with cutting wrenches (no database files: the real databases are still being researched)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from cnctwin.core.bus import Bus, CuttingContact, MotionFrame, PathFrame, RemovalContact
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState, ToolSlot, ToolTable
from cnctwin.db.simap import SIMap, pose_matrix
from cnctwin.holding.toolhold import ToolData

N_TICKS = 10
IPO_S = 1e-3

STEEL = {"id": "test-steel", "family": "steel",
         "physical": {"density_kg_m3": {"value": 7850, "src": "est"}, "ultimate_MPa": {"value": 600, "src": "est"},
                      "yield_MPa": {"value": 350, "src": "est"}, "youngs_modulus_GPa": {"value": 210, "src": "est"},
                      "poisson": {"value": 0.3, "src": "est"}}}
ALUMINIUM = {"id": "test-al", "family": "aluminium",
             "physical": {"density_kg_m3": {"value": 2700, "src": "est"}, "ultimate_MPa": {"value": 310, "src": "est"},
                          "youngs_modulus_GPa": {"value": 69, "src": "est"}, "poisson": {"value": 0.33, "src": "est"}}}

VISE = {"id": "test-vise", "type": "vise", "jaw_width_mm": 150, "jaw_height_mm": 40, "max_opening_mm": 200,
        "clamping_force_kN": {"value": 20.0, "at_handle_torque_Nm": 80, "src": "est"},
        "jaw_friction": {"value": 0.2, "src": "est"}, "jaw_stiffness_N_um": {"value": 200, "src": "est"}}
CHUCK = {"id": "test-chuck", "type": "chuck_3jaw", "jaws": 3, "max_rpm": 5000,
         "gripping_force_kN": {"value": 60, "src": "est"}, "jaw_friction": {"value": 0.1, "src": "est"},
         "centrifugal_loss_model": {"method": "quadratic_in_speed", "loss_kN": 38, "at_rpm": 3500, "src": "est"}}
PLATE = {"id": "test-plate", "type": "fixture_plate", "plate_mm": [300, 300, 25]}
CLAMP = {"id": "test-strap", "type": "clamp", "stud": "M12", "recommended_torque_Nm": 47.5,
         "clamping_force_kN": {"value": 9.79, "at_handle_torque_Nm": 47.5, "src": "est"}, "stud_preload_kN": 19.57,
         "lever_ratio": 2.0, "jaw_friction": {"value": 0.19, "src": "est"}}
TABS = {"id": "test-tab", "type": "tabs", "dims_mm": {"width": 4.0, "height": 2.0, "length": 4.0}}


def pose(translate_mm=(0, 0, 0), rotate_deg=(0, 0, 0)) -> np.ndarray:
    return pose_matrix([v * 1e-3 for v in translate_mm], [math.radians(v) for v in rotate_deg])


def entry(item: dict, translate_mm=(0, 0, 0), rotate_deg=(0, 0, 0), **extra) -> dict:
    out = {"device": SIMap(item), "pose": pose(translate_mm, rotate_deg)}
    for key, value in extra.items():
        out.update(SIMap({key: value})._data)
    return out


def project(entries: list, stock=None, material=None, stock_pose=None, physics=None, machine=None, ops=1,
            tolerances=None) -> SimpleNamespace:
    stock = stock or {"kind": "block", "size_mm": [100, 60, 30]}
    operations = []
    nested = bool(entries) and isinstance(entries[0], list)
    for i in range(ops):
        operations.append({"name": f"OP{10 * (i + 1)}",
                           "setup": {"workholding": entries[i] if nested else entries, "workpiece": 0},
                           "stock_pose_W_S": stock_pose if stock_pose is not None else np.eye(4)})
    return SimpleNamespace(stock=SIMap(stock), material=SIMap(material or STEEL), operations=operations,
                           physics={"holding": physics or {}}, machine=machine, cad={"tolerances": tolerances or []})


def context(proj, seed=11) -> RunContext:
    return RunContext(project=proj, db=None, run_dir=None, seed=seed, tick_s=1e-6, ipo_ticks=1000, macro_ipo=N_TICKS)


def sim_state(slots=1) -> SimState:
    state = SimState()
    state.tools = ToolTable(tuple(ToolSlot(slot=i, number=i + 1, tool_id=f"t{i}", holder_id="h") for i in range(slots)))
    return state


def tool_data(slot=0, holder_type="er_collet", shank_d_mm=10.0, flutes=4, **kw) -> ToolData:
    td = ToolData(slot=slot, number=slot + 1, holder_type=holder_type, shank_d_m=shank_d_mm * 1e-3,
                  tool_mass_kg=0.05, tool_com_z_m=0.03, tool_I_axial_kgm2=6.3e-7, tool_I_trans_kgm2=1.6e-5,
                  holder_mass_kg=1.4, holder_com_z_m=0.12, holder_grade=(0.0025, 25000 * math.pi / 30),
                  max_rad_s_tool=40000 * math.pi / 30, max_rad_s_holder=25000 * math.pi / 30, flutes=flutes,
                  collet_size="ER32", bore_max_m=0.020)
    for key, value in kw.items():
        setattr(td, key, value)
    return td


def path_frame(tick0=0, omega=-1047.2, slot=0, workpiece=0, n=N_TICKS) -> PathFrame:
    ticks = tick0 + (np.arange(n) + 1) * 1000
    theta = (np.abs(omega) * ticks * 1e-6) % (2 * math.pi)
    return PathFrame(path=0, tick0=tick0, tick=ticks.astype(np.int64), block_index=np.zeros(n, np.int32),
                     tip_S_m=np.zeros((n, 3)), tip_M_m=np.zeros((n, 3)), spindle_angle_rad=theta,
                     spindle_omega_rad_s=np.full(n, float(omega)), tool_slot=slot, workpiece=workpiece)


def bus_with(work_mean=None, work_peak=None, tool_mean=None, tool_peak=None, omega=-1047.2, tick0=0, slot=0,
             n=N_TICKS, contact=True) -> Bus:
    """A bus_prev with one path and (optionally) one cutting contact carrying constant per-tick wrenches."""
    bus = Bus()
    frame = path_frame(tick0, omega, slot, 0, n)
    bus.motion = MotionFrame(time_s=(tick0 + n * 1000) * 1e-6, tick=tick0 + n * 1000, paths=[frame])
    if contact:
        def rows(w):
            return np.tile(np.asarray(w if w is not None else np.zeros(6), dtype=float), (n, 1))
        cc = CuttingContact(contact=0, tool_slot=slot, wrench_work_W_mean=rows(work_mean),
                            wrench_work_W_peak=rows(work_peak if work_peak is not None else work_mean),
                            wrench_tool_T0_mean=rows(tool_mean),
                            wrench_tool_T0_peak=rows(tool_peak if tool_peak is not None else tool_mean))
        bus.cutting.contacts = [cc]
        bus.removal.contacts = [RemovalContact(contact=0, tool_slot=slot, workpiece=0)]
    return bus


def run(dom, ctx, state, bus, steps=1):
    """``steps`` macro steps with the same bus_prev; returns the last HoldingFeedback."""
    record = None
    for _ in range(steps):
        state.bus_prev = bus
        state.bus_next = Bus()
        dom.macro_step(ctx, state, bus, state.bus_next)
        record = state.bus_next.holding
        state.clock.tick = int(bus.motion.tick)
        state.clock.macro += 1
    return record
