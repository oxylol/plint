"""Small inline fixtures for the thermal tests: a fake project, tool table, edge model, bus frames."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from cnctwin.core.abi import EdgeModel
from cnctwin.core.bus import (Bus, CoolantFeedback, CuttingContact, CuttingFrame, MotionFrame, PathFrame,
                              ProcessCoefficients, RemovalContact, RemovalFrame, ToolContact)
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState, ToolSlot, ToolTable
from cnctwin.db.simap import SIMap
from cnctwin.thermal.protocols import BoxStock

TOOL_RAW = {"id": "test-em10", "type": "square_endmill",
            "tool_material": {"class": "carbide", "id": "carbide-submicron-10co"},
            "geometry": {"diameter_mm": 10, "flutes": 4, "cutting_length_mm": 22,
                         "shank": {"diameter_mm": 10, "length_mm": 40}, "overall_length_mm": 72,
                         "core_diameter_mm": 6.0, "gullet_area_mm2": 9.5}}
HOLDER_RAW = {"id": "test-sk40", "type": "shrink_fit", "taper": "SK40", "bore_mm": 10, "gauge_length_mm": 80,
              "nose_diameter_mm": 24, "profile_mm": [[0, 24], [30, 27], [80, 44]], "mass_kg": 1.6,
              "clamp_length_mm": 40}
MACHINE_RAW = {
    "spindles": [{"id": "main", "kind": "milling", "max_rpm": 12000,
                  "thermal": {"growth_um_at_max_rpm": {"value": 25, "src": "est", "note": "t"},
                              "time_constant_min": 25, "bearing_heat_W_at_max_rpm": 450,
                              "thermal_comp": True, "comp_residual_fraction": 0.3}}],
    "axes": {"X": {"type": "linear", "travel_mm": [-762, 0], "ballscrew": {"diameter_mm": 40, "length_mm": 1100}},
             "Y": {"type": "linear", "travel_mm": [-406, 0]},
             "Z": {"type": "linear", "travel_mm": [-508, 0], "ballscrew": {"diameter_mm": 40, "length_mm": 800}}},
    "kinematics": {"workpiece_chain": [{"link": "saddle", "axis": "Y", "direction": [0, -1, 0]},
                                       {"link": "table", "axis": "X", "direction": [-1, 0, 0]}],
                   "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]}]},
    "thermal": {"axis_drift": {"expansion_1_K": 11.5e-6, "heat_per_m_travel_J": 60, "time_constant_min": 40,
                               "electronic_comp": True},
                "structure_sensitivity_um_per_K": 3, "warmup": {"cold_to_warm_min": 60}},
    "power": {"machine_mass_kg": 3600},
}
MATERIAL_RAW = {
    "id": "test-steel", "name": "test steel", "family": "steel",
    "physical": {"density_kg_m3": {"value": 7850},
                 "specific_heat_J_kgK": {"table": [[20, 460], [500, 615]]},
                 "conductivity_W_mK": {"table": [[20, 49.8], [500, 36.0]]},
                 "expansion_1_K": {"table": [[20, 11.6e-6], [500, 14.2e-6]]}},
    "thermal_damage": {"burn_degC": {"value": 200}, "white_layer_degC": {"value": 725}, "melt_smear_degC": None},
}


def project(machine_state="cold", every_n_macro=2, cell_mm=4.0, wait_min=5.0, ambient_degC=20.0):
    return SimpleNamespace(
        environment={"ambient_degC": ambient_degC, "machine_state": machine_state, "time_between_ops_min": wait_min},
        machine=MACHINE_RAW, material=SIMap(MATERIAL_RAW),
        physics={"thermal": {"every_n_macro": every_n_macro, "cell_mm": cell_mm, "tool_cell_mm": 2.0}},
        tools={"list": [{"number": 1, "tool": SIMap(TOOL_RAW), "holder": SIMap(HOLDER_RAW), "stickout_mm": 35}]},
        operations=[], coolant=None, stock_pose_W_S=None)


def context(proj) -> RunContext:
    return RunContext(project=proj, db=None, run_dir=None, seed=7, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10)


def sim_state() -> SimState:
    state = SimState()
    state.tools = ToolTable((ToolSlot(slot=0, number=1, tool_id="test-em10", holder_id="test-sk40"),))
    return state


def edge_model(nt=4, ne=10, length_m=5e-3):
    z = np.tile((np.arange(ne) + 0.5) * length_m / ne, (nt, 1))
    full = lambda v: np.full((nt, ne), float(v))
    return EdgeModel(np.arange(nt) * 2 * math.pi / nt, z, full(0.005), np.zeros((nt, ne)), full(length_m / ne),
                     full(length_m / ne), full(math.pi / 2), full(0.14), full(0.6), full(0.17), full(8e-6),
                     np.zeros((nt, ne), np.int32), np.zeros((nt, ne), np.int32),
                     np.tile(np.arange(ne, dtype=np.int32), (nt, 1)), np.ones((nt, ne), np.int8), np.zeros(3),
                     np.zeros(2), np.zeros(ne), np.full(ne, 0.005))


class Edges:
    def __init__(self, model):
        self.model = model

    def edge_model(self, tool_slot):
        return self.model


def coefficients(slot=0, sc=0.8, rt=0.2, ft=0.45):
    h = np.geomspace(1e-6, 1e-3, 8)
    tab = lambda v: np.full((1, h.size), float(v))
    return ProcessCoefficients(tool_slot=slot, h_grid_m=h, frac_shear_to_chip=tab(sc), frac_rake_to_tool=tab(rt),
                               frac_flank_to_tool=tab(ft), chip_ratio=tab(0.4), shear_angle_rad=tab(0.5), version=1)


def bus_step(n, nt=4, ne=10, rpm=8000.0, power_W=(300.0, 100.0, 30.0, 5.0), point=(0.02, 0.015, 0.02),
             cutting=True, ticks=10, x_speed_m_s=0.02, coolant_on=False):
    """A bus with motion, removal and cutting records of macro step n (10 ticks of 1 ms)."""
    dt = ticks * 1e-3
    omega = -rpm * 2 * math.pi / 60.0
    tick = (n * ticks + np.arange(1, ticks + 1)) * 1000
    x = -0.4 + x_speed_m_s * (n * dt + np.arange(1, ticks + 1) * 1e-3)
    axes = np.stack([x, np.full(ticks, -0.2), np.full(ticks, -0.3)], axis=1)
    frame = PathFrame(path=0, tick0=int(tick[0]), tick=tick.astype(np.int64), block_index=np.full(ticks, 3, np.int32),
                      tip_S_m=np.tile(point, (ticks, 1)), tip_M_m=np.zeros((ticks, 3)),
                      axis_T_M=np.tile([0.0, 0.0, 1.0], (ticks, 1)), axis_names=("X", "Y", "Z"), axis_actual=axes,
                      axis_cmd=axes, spindle_angle_rad=np.zeros(ticks), spindle_rev=np.zeros(ticks, np.int64),
                      spindle_omega_rad_s=np.full(ticks, omega),
                      feed_vel_M_m_s=np.tile([x_speed_m_s, 0.0, 0.0], (ticks, 1)), rapid=np.zeros(ticks, bool),
                      tool_slot=0, spindle_id="main", workpiece=0)
    bus = Bus()
    bus.motion = MotionFrame(time_s=float(tick[-1]) * 1e-6, tick=int(tick[-1]), paths=[frame])
    bus.coolant = CoolantFeedback(on=coolant_on, fluid_temp_K=295.0, h_work_far_W_m2K=3000.0 if coolant_on else 0.0)
    if coolant_on:
        bus.coolant.h_flank_W_m2K[0] = 8000.0
        bus.coolant.h_work_W_m2K[0] = 8000.0
        bus.coolant.reach_fraction[0] = 1.0
    if not cutting:
        return bus
    engaged = 0.3
    in_cut = np.full((nt, ne), dt * engaged)
    entries = np.full((nt, ne), int(round(dt * abs(omega) / (2 * math.pi))), np.int32)
    h = 0.05e-3
    shear, rake, flank, plough = (p * dt for p in power_W)
    cut = CuttingContact(contact=0, tool_slot=0, in_cut_s=dt * engaged, heat_shear_J=shear, heat_rake_J=rake,
                         heat_flank_J=flank, ploughing_J=plough, in_cut_time_s=in_cut,
                         rake_sliding_m=in_cut * 2.0, flank_sliding_m=in_cut * 5.0,
                         int_sigma_v_rake=in_cut * 1e9, int_sigma_v_flank=in_cut * 3e8, int_h_dt=in_cut * h,
                         h_max_m=np.full((nt, ne), 2 * h), entries=entries,
                         work_heat_S=np.array([[point[0], point[1], point[2], 0, 0, 1, 1.0],
                                               [point[0] - 0.002, point[1], point[2], 0, 0, 1, 1.0]]))
    bus.cutting = CuttingFrame(contacts=[cut], coefficients={0: coefficients()})
    geo = ToolContact(in_cut=True, point_S_m=np.array(point), normal_M=np.array([0.0, 0.0, 1.0]),
                      point_M_m=np.zeros(3), ae_m=0.002, ap_m=0.005, engaged_angle_rad=0.93)
    bus.removal = RemovalFrame(contacts=[RemovalContact(contact=0, tool_slot=0, workpiece=0, path=0, geometry=geo)])
    return bus


def stock(size=(0.04, 0.03, 0.02)):
    return BoxStock([0.0, 0.0, 0.0], list(size))


def SampledStockCylinder(radius, length):
    """A bar along z_S with its axis on the S origin (turning convention)."""
    from cnctwin.thermal.protocols import SampledStock
    return SampledStock(lambda p: p[:, 0] ** 2 + p[:, 1] ** 2 < radius ** 2, [-radius, -radius, 0.0],
                        [radius, radius, length], samples=3)
