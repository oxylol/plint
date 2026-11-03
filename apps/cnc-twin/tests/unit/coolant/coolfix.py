"""Small inline fixtures for the coolant tests: a vertical mill with flood and through-spindle coolant, an
end mill and a drill in shrink-fit holders, a block stock, motion / removal / cutting / thermal records."""

from __future__ import annotations

import functools
import math
from types import SimpleNamespace

import numpy as np

from cnctwin.core.bus import (Bus, ChipFeedback, CuttingContact, CuttingFrame, MotionFrame, PathFrame, RemovalContact,
                              RemovalFrame, ThermalFeedback, ToolContact, ToolHeat)
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState, ToolSlot, ToolTable
from cnctwin.db.simap import SIMap

ENDMILL = {"id": "t-em10", "type": "square_endmill",
           "tool_material": {"class": "carbide", "id": "carbide-submicron-10co"},
           "geometry": {"diameter_mm": 10, "flutes": 4, "cutting_length_mm": 22, "helix_deg": 35, "rake_deg": 8,
                        "clearance_deg": 10, "shank": {"diameter_mm": 10, "length_mm": 40}, "overall_length_mm": 72,
                        "core_diameter_mm": 6.0, "gullet_area_mm2": 9.5}}
DRILL = {"id": "t-dr8", "type": "twist_drill", "tool_material": {"class": "carbide", "id": "carbide-submicron-10co"},
         "geometry": {"diameter_mm": 8, "flutes": 2, "point_angle_deg": 140, "cutting_length_mm": 40,
                      "flute_length_mm": 45, "helix_deg": 30, "rake_deg": 8, "clearance_deg": 10,
                      "shank": {"diameter_mm": 8, "length_mm": 36}, "overall_length_mm": 91,
                      "coolant_holes": {"count": 2, "diameter_mm": 1.0, "exit": "tip"}}}
HOLDER = {"id": "t-sk40", "type": "shrink_fit", "taper": "SK40", "bore_mm": 10, "gauge_length_mm": 80,
          "nose_diameter_mm": 24, "profile_mm": [[0, 24], [30, 27], [80, 44]], "mass_kg": 1.6, "clamp_length_mm": 40}

STOCK_MM = (100.0, 80.0, 30.0)
TABLE_TOP_M = np.array([0.0, 0.0, 0.0])        # S origin in M (stock corner on the table)


def machine_raw(nozzles=None, pumps=None, chiller=None, air=True):
    pumps = pumps if pumps is not None else [
        {"id": "flood", "kind": "flood", "pressure_bar": 2.0, "flow_l_min": 45, "power_kW": 0.75},
        {"id": "tsc", "kind": "through_spindle", "pressure_bar": 21, "flow_l_min": 11, "power_kW": 1.1}]
    nozzles = nozzles if nozzles is not None else [
        {"id": "n1", "kind": "flood", "attached_to": "head", "position_mm": [70, 0, -60], "aim": [-70, 0, -50],
         "diameter_mm": 4, "pump": "flood"},
        {"id": "tsc", "kind": "through_spindle", "attached_to": "spindle_nose", "position_mm": [0, 0, 0],
         "aim": [0, 0, -1], "pump": "tsc"}]
    coolant = {"tank_l": 200, "pumps": pumps, "nozzles": nozzles, "filtration_um": 40}
    if air:
        coolant["air_blast"] = {"flow_l_min": 170, "pressure_bar": 6.9}
    if chiller is not None:
        coolant["chiller"] = chiller
    return {"id": "t-vmc", "type": "vertical_mill",
            "axes": {"X": {"type": "linear"}, "Y": {"type": "linear"}, "Z": {"type": "linear"}},
            "kinematics": {"workpiece_chain": [{"link": "saddle", "axis": "Y", "direction": [0, -1, 0]},
                                               {"link": "table", "axis": "X", "direction": [-1, 0, 0]},
                                               {"link": "table_top", "fixed": {"translate_mm": [0, 0, 0]}}],
                           "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]},
                                          {"link": "spindle_nose", "fixed": {"translate_mm": [0, 0, 500]}}],
                           "tool_axis": [0, 0, 1]},
            "coolant": coolant,
            "enclosure": {"box_mm": [[-600, -500, -300], [600, 500, 900]]},
            "geometry": [{"name": "table", "attached_to": "table", "collide": True,
                          "primitives": [{"box": {"min_mm": [-300, -200, -60], "max_mm": [300, 200, -0.5]}}]}]}


@functools.lru_cache(maxsize=1)
def database():
    from cnctwin.db import Database
    return Database.load()


def fluid_db(fluid_id="emulsion-8pct"):
    return database().coolant(fluid_id)


def project(fluid="emulsion-8pct", machine=None, physics=None, stock_mm=STOCK_MM, ambient_degC=20.0,
            coolant_degC=22.0, nozzle_positions=None, wait_min=0.0, family="steel"):
    fluid_obj = fluid_db(fluid) if isinstance(fluid, str) else fluid
    return SimpleNamespace(
        machine=SIMap(machine if machine is not None else machine_raw()),
        coolant=SimpleNamespace(fluid=fluid_obj, concentration_frac=None, tank_fill_fraction=0.9),
        environment=SimpleNamespace(ambient_K=ambient_degC + 273.15, coolant_K=coolant_degC + 273.15,
                                    time_between_ops_s=wait_min * 60.0),
        material=SimpleNamespace(family=family),
        stock=SimpleNamespace(kind="block", size_m=[v * 1e-3 for v in stock_mm]),
        physics={"coolant": physics or {"every_n_macro": 1, "latency_macro": 1, "rays_per_jet": 32}},
        tools={"list": []}, operations=[SimpleNamespace(nozzle_positions=nozzle_positions or {})],
        stock_pose_W_S=None)


def context(proj) -> RunContext:
    return RunContext(project=proj, db=None, run_dir=None, seed=11, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10)


def geometries(raws=(ENDMILL,), stickout_mm=35.0):
    from cnctwin.tooling import build_tool
    return [build_tool(SIMap(raw), SIMap(HOLDER), {"number": slot + 1, "stickout_mm": stickout_mm}, slot=slot)
            for slot, raw in enumerate(raws)]


def sim_state(n_tools=1) -> SimState:
    state = SimState()
    state.tools = ToolTable(tuple(ToolSlot(slot=i, number=i + 1, tool_id=f"t{i}", holder_id="t-sk40")
                                  for i in range(n_tools)))
    return state


def path_frame(n, tip_S=(0.05, 0.04, 0.025), slot=0, rpm=10000.0, pumps=None, nozzles=None, ticks=10, z_axis=0.0):
    """Motion record of macro step n: the tool tip at tip_S (S = M here: stock on the table at the origin)."""
    tick = (n * ticks + np.arange(1, ticks + 1)) * 1000
    tip = np.tile(np.asarray(tip_S, dtype=float) + TABLE_TOP_M, (ticks, 1))
    omega = -rpm * 2 * math.pi / 60.0
    pose = np.eye(4)
    pose[:3, 3] = tip[-1]
    pose_S = np.eye(4)
    pose_S[:3, 3] = np.asarray(tip_S, dtype=float)
    axes = np.tile([0.0, 0.0, z_axis], (ticks, 1))
    return PathFrame(path=0, tick0=int(tick[0]), tick=tick.astype(np.int64), block_index=np.zeros(ticks, np.int32),
                     tip_S_m=np.tile(np.asarray(tip_S, dtype=float), (ticks, 1)), tip_M_m=tip,
                     axis_T_M=np.tile([0.0, 0.0, 1.0], (ticks, 1)), axis_names=("X", "Y", "Z"), axis_actual=axes,
                     axis_cmd=axes, spindle_angle_rad=np.zeros(ticks), spindle_rev=np.zeros(ticks, np.int64),
                     spindle_omega_rad_s=np.full(ticks, omega), feed_vel_M_m_s=np.zeros((ticks, 3)),
                     rapid=np.zeros(ticks, bool), tool_slot=slot, spindle_id="main", workpiece=0, pose_M_T=pose,
                     pose_S_T=pose_S, coolant_pumps=dict(pumps if pumps is not None else {"flood": True}),
                     nozzle_settings=dict(nozzles or {}))


def bus_step(n, tip_S=(0.05, 0.04, 0.025), pumps=None, nozzles=None, cutting=True, contact_point_S=None,
             ap=5e-3, ae=2e-3, rake_K=650.0, body_K=330.0, surface_K=520.0, preheat_K=310.0, slot=0,
             chips_heat_J=0.0, thermal_heat_J=0.0, emitted_kg=0.0, blocked=(), pile=None, booked=None):
    bus = Bus()
    frame = path_frame(n, tip_S, slot, pumps=pumps, nozzles=nozzles)
    bus.motion = MotionFrame(time_s=float(frame.tick[-1]) * 1e-6, tick=int(frame.tick[-1]), paths=[frame])
    bus.chips = ChipFeedback(blocked_nozzles=list(blocked), heat_to_coolant_J=chips_heat_J, emitted_kg=emitted_kg,
                             booked_until_tick=int(frame.tick[-1]) if booked is None else booked)
    if pile is not None:
        bus.chips.pile_origin_S_m, bus.chips.pile_axes_S, bus.chips.pile_cell_m, bus.chips.pile_base_m, \
            bus.chips.pile_height_m = pile
    bus.thermal = ThermalFeedback(heat_to_coolant_J=thermal_heat_J)
    if not cutting:
        return bus
    point = np.asarray(contact_point_S if contact_point_S is not None else
                       (tip_S[0] + 5e-3, tip_S[1], tip_S[2] + 0.5 * ap), dtype=float)
    geo = ToolContact(in_cut=True, point_M_m=point + TABLE_TOP_M, normal_M=np.array([1.0, 0.0, 0.0]), point_S_m=point,
                      ae_m=ae, ap_m=ap, engaged_length_m=ap, engaged_angle_rad=0.9, in_cut_fraction=0.3)
    bus.removal = RemovalFrame(contacts=[RemovalContact(contact=0, tool_slot=slot, workpiece=0, path=0, geometry=geo)])
    t_in = np.full((4, 6), 2e-3)
    bus.cutting = CuttingFrame(contacts=[CuttingContact(contact=0, tool_slot=slot, in_cut_s=3e-3, in_cut_time_s=t_in,
                                                        int_h_dt=t_in * 40e-6)])
    heat = ToolHeat(rake_temp_mean_K=np.full((4, 6), rake_K), body_temp_K=body_K,
                    body_temp_at_elem_K=np.full((4, 6), body_K))
    bus.thermal.tools[slot] = heat
    bus.thermal.contact_surface_temp_K[0] = surface_K
    bus.thermal.contact_preheat_K[0] = preheat_K
    bus.thermal.work_bulk_mean_K[0] = 300.0
    return bus


def make_domain(proj=None, raws=(ENDMILL,), **kwargs):
    from cnctwin.coolant.domain import CoolantDomain
    from cnctwin.coolant.protocols import TooledShapes
    proj = proj or project(**kwargs)
    ctx = context(proj)
    state = sim_state(len(raws))
    dom = CoolantDomain(shapes=TooledShapes(geometries(raws)))
    dom.init(ctx, state)
    return dom, ctx, state


def run(dom, ctx, state, steps, start=0, **bus_kwargs):
    outs = []
    for n in range(start, start + steps):
        prev = bus_step(n, **bus_kwargs)
        nxt = Bus()
        state.clock.tick = int(prev.motion.tick)
        state.bus_prev = prev
        state.bus_next = nxt
        dom.macro_step(ctx, state, prev, nxt)
        outs.append(nxt)
    return outs
