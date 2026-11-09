"""Small inline fixtures for the wear tests: materials, a tool table, edge models, bus frames, and an
analytic Taylor response (the real databases are still being researched)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from cnctwin.core.abi import EDGE_PERIPHERAL, EdgeModel
from cnctwin.core.bus import (Bus, ChipFeedback, CoolantFeedback, CuttingContact, CuttingFrame, MotionFrame, PathFrame,
                              ThermalFeedback, ToolHeat)
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState, ToolSlot, ToolTable
from cnctwin.db.simap import SIMap
from cnctwin.wear import calibration as CAL
from cnctwin.wear.protocols import SectionInfo, StaticToolInfo

TOOL_MATERIAL = {
    "id": "test-carbide", "class": "carbide", "density_kg_m3": {"value": 14500},
    "youngs_modulus_GPa": {"value": 580}, "poisson": {"value": 0.22}, "hardness_HV": {"value": 1600},
    "transverse_rupture_MPa": {"value": 4000}, "fracture_toughness_MPa_sqrt_m": {"value": 10.0},
    "conductivity_W_mK": {"table": [[20, 80], [400, 65], [800, 55]]},
    "specific_heat_J_kgK": {"table": [[20, 200], [400, 240], [800, 270]]},
    "expansion_1_K": {"value": 5.5e-6}, "max_service_temp_degC": {"value": 1000}, "weibull_modulus": {"value": 15},
    "grain_size_um": {"value": 0.8}, "usui": [],
}
COATING = {"id": "test-tialn", "process": "PVD", "hardness_HV": {"value": 3300}, "max_service_temp_degC": {"value": 800},
           "typical_thickness_um": [2, 4], "wear_rate_factor": {"value": 0.5}, "conductivity_W_mK": {"value": 4.5},
           "thermal_barrier_factor": {"value": 0.9}}
MATERIAL = {
    "id": "test-steel", "name": "test steel", "family": "steel", "iso_group": "P",
    "physical": {"density_kg_m3": {"value": 7850}, "specific_heat_J_kgK": {"table": [[20, 460], [500, 615]]},
                 "conductivity_W_mK": {"table": [[20, 49.8], [500, 36.0]]}, "youngs_modulus_GPa": {"value": 210},
                 "yield_MPa": {"value": 505}, "ultimate_MPa": {"value": 655}},
    "flow_stress": {"model": "johnson_cook", "A_MPa": 553.1, "B_MPa": 600.8, "n": 0.234, "C": 0.0134, "m": 1.0,
                    "ref_strain_rate_1_s": 1.0, "T_ref_degC": 20, "T_melt_degC": 1460},
    "cutting": {"min_chip_thickness_ratio": {"value": 0.27},
                "work_hardening": {"layer_depth_um": {"value": 100}, "hardness_increase_pct": {"value": 48}}},
    "friction": [{"tool_class": "carbide", "coating": "uncoated", "mu_sliding": {"value": 0.5}}],
    "chip": {"tendency": "continuous", "bue": {"vc_m_min": [5, 60], "interface_temp_degC": [150, 450]}},
    "tool_life": [{"tool_class": "carbide", "coating": "uncoated", "C_m_min": 290, "n": 0.22, "a": 0.3, "b": 0.12,
                   "feed_ref_mm": 0.3, "ap_ref_mm": 2, "VB_limit_mm": 0.3, "T_ref_min": 1}],
    "skin": {"type": "none"},
}
TOOL = {"id": "test-em10", "type": "square_endmill", "hand": "right",
        "tool_material": {"class": "carbide", "id": "test-carbide"}, "coating": {"id": "none"},
        "geometry": {"diameter_mm": 10, "flutes": 4, "cutting_length_mm": 22, "corner_radius_mm": 0.5}}


class Catalog:
    """tool_material(id) / coating(id) over inline mappings (the database interface the domain uses)."""

    def __init__(self, tool_materials=None, coatings=None):
        self.tm = {TOOL_MATERIAL["id"]: TOOL_MATERIAL, **(tool_materials or {})}
        self.co = {COATING["id"]: COATING, **(coatings or {})}

    def tool_material(self, ident):
        return SIMap(self.tm[ident])

    def coating(self, ident):
        return SIMap(self.co[ident])


def edge_model(nt=4, ne=10, length_m=5e-3, radius_m=5e-3, kappa=math.pi / 2, rake=0.14, clear=0.17,
               edge_radius_m=8e-6) -> EdgeModel:
    z = np.tile((np.arange(ne) + 0.5) * length_m / ne, (nt, 1))
    full = lambda v: np.full((nt, ne), float(v))
    return EdgeModel(np.arange(nt) * 2 * math.pi / nt, z, full(radius_m), np.zeros((nt, ne)), full(length_m / ne),
                     full(length_m / ne), full(kappa), full(rake), full(0.6), full(clear), full(edge_radius_m),
                     np.full((nt, ne), EDGE_PERIPHERAL, np.int32), np.zeros((nt, ne), np.int32),
                     np.tile(np.arange(ne, dtype=np.int32), (nt, 1)), np.ones((nt, ne), np.int8), np.zeros(3),
                     np.zeros(2), (np.arange(ne) + 0.5) * length_m / ne, np.full(ne, radius_m), np.array([-1.0]))


def sections(d_m=8e-3, z_m=0.022):
    z_b = math.pi * d_m ** 3 / 32.0
    return [SectionInfo("flute_runout", z_m, z_b, 2.0 * z_b, d_m),
            SectionInfo("shank_at_holder", 0.035, math.pi * 0.01 ** 3 / 32.0, math.pi * 0.01 ** 3 / 16.0, 0.01)]


def tool_info(edge=None, secs=None, finish_radius_m=5e-4, rotating=True):
    edge = edge if edge is not None else edge_model()
    return StaticToolInfo({0: edge}, {0: secs if secs is not None else sections()}, {0: finish_radius_m},
                          {0: 5e-3}, {0: rotating})


def project(material=None, tool=None, physics=None, initial_wear=None, tolerances=None, stock=None, life=None):
    entry = {"number": 1, "tool": SIMap(tool or TOOL)}
    if initial_wear is not None:
        entry["initial_wear"] = initial_wear
    if life is not None:
        entry["life"] = life
    return SimpleNamespace(environment={"ambient_degC": 20.0}, material=SIMap(material or MATERIAL),
                           physics={"wear": physics or {}}, tools={"list": [entry]},
                           cad={"tolerances": tolerances or []}, stock=stock or {"kind": "block",
                                                                                   "size_mm": [100, 80, 30]},
                           machine=None, operations=[])


def context(proj, seed=7, macro_ipo=10) -> RunContext:
    return RunContext(project=proj, db=None, run_dir=None, seed=seed, tick_s=1e-6, ipo_ticks=1000, macro_ipo=macro_ipo)


def sim_state() -> SimState:
    state = SimState()
    state.tools = ToolTable((ToolSlot(slot=0, number=1, tool_id="test-em10", holder_id=""),))
    return state


class AnalyticResponse:
    """A smooth response: flank temperature rising with speed and land, three quadrature points."""

    cache_key = ("analytic", 1)

    def __init__(self, t0=500.0, rise=500.0, v_ref=3.0, power=0.45, land_rise=5e5, spread=60.0, sigma=1.2e9):
        self.t0, self.rise, self.v_ref, self.power = t0, rise, v_ref, power
        self.land_rise, self.spread, self.sigma = land_rise, spread, sigma

    def temps(self, v, vb):
        t = self.t0 + self.rise * (v / self.v_ref) ** self.power + self.land_rise * vb
        return np.array([t - self.spread, t, t + self.spread])

    def response(self, slot, cond):
        temps = self.temps(cond.speed_m_s, cond.VB_m)
        return CAL.Response(self.sigma * cond.speed_m_s, temps, 0.6e9 * cond.speed_m_s * 0.4, temps + 150.0,
                            np.array([0.25, 0.5, 0.25]), self.sigma, 0.6e9, 0.4)


# ------------------------------------------------------------------------------------------ bus frames
def path_frame(tick0=0, n_ticks=10, omega=-1047.2, feed_m_s=0.0167, slot=0, path=0):
    ticks = tick0 + 1000 * np.arange(n_ticks, dtype=np.int64)
    return PathFrame(path=path, tick0=tick0, tick=ticks, block_index=np.zeros(n_ticks, np.int32),
                     tip_S_m=np.tile([0.05, 0.04, 0.03], (n_ticks, 1)), tip_M_m=np.zeros((n_ticks, 3)),
                     spindle_omega_rad_s=np.full(n_ticks, omega),
                     feed_vel_M_m_s=np.tile([feed_m_s, 0.0, 0.0], (n_ticks, 1)), tool_slot=slot,
                     pose_S_T=np.eye(4))


def cutting(edge, speed=3.0, h=1e-4, duty=0.25, dt=0.01, sigma_f=1.2e9, sigma_r=0.6e9, chip_ratio=0.4, entries=4,
            engaged=None, shocks=None, section_moment=None, torque=0.0, slot=0, contact=0, surface=None,
            edge_fraction=None):
    """One contact record: the tool is in cut ``duty`` of the step; each engaged element is in contact
    ``edge_fraction`` of it (its engaged arc / 2 pi of the cutting time; default ``duty``)."""
    nt, ne = np.shape(edge.elem_z_m)
    mask = np.ones((nt, ne), bool) if engaged is None else np.asarray(engaged, bool)
    t_in = np.where(mask, (duty if edge_fraction is None else edge_fraction) * dt, 0.0)
    contact_rec = CuttingContact(
        contact=contact, tool_slot=slot, in_cut_s=duty * dt, in_cut_time_s=t_in, flank_sliding_m=speed * t_in,
        rake_sliding_m=speed * chip_ratio * t_in, int_sigma_v_flank=sigma_f * speed * t_in,
        int_sigma_v_rake=sigma_r * speed * chip_ratio * t_in, int_h_dt=h * t_in, h_max_m=np.where(mask, h, 0.0),
        entries=np.where(mask, entries, 0).astype(np.int32),
        shocks=np.zeros((0, 8)) if shocks is None else np.asarray(shocks, float),
        section_moment_peak_Nm=np.zeros(0) if section_moment is None else np.asarray(section_moment, float),
        torque_peak_Nm=float(torque),
        surface_events=np.zeros((0, 16)) if surface is None else np.asarray(surface, float))
    return contact_rec


def heat(edge, flank_K=1000.0, rake_K=1150.0, weights=(0.25, 0.5, 0.25), spread=60.0, cycle_dK=0.0, body_K=400.0):
    nt, ne = np.shape(edge.elem_z_m)
    w = np.asarray(weights, float)
    offsets = np.linspace(-spread, spread, w.size) if w.size > 1 else np.zeros(1)
    flank = np.broadcast_to(flank_K, (nt, ne))[:, :, None] + offsets[None, None, :]
    rake = np.broadcast_to(rake_K, (nt, ne))[:, :, None] + offsets[None, None, :]
    return ToolHeat(edge_temp_q_K=np.ascontiguousarray(rake), flank_temp_q_K=np.ascontiguousarray(flank), q_weights=w,
                    edge_cycle_amp_K=np.full((nt, ne), float(cycle_dK)), body_temp_at_elem_K=np.full((nt, ne), body_K),
                    body_temp_K=body_K)


def bus(edge=None, contacts=None, heat_rec=None, frame=None, coolant=None, chips=None, tick=10000) -> Bus:
    b = Bus()
    b.motion = MotionFrame(time_s=tick * 1e-6, tick=tick, paths=[frame] if frame is not None else [])
    b.cutting = CuttingFrame(contacts=list(contacts or []))
    b.thermal = ThermalFeedback(tools={0: heat_rec} if heat_rec is not None else {})
    b.coolant = coolant if coolant is not None else CoolantFeedback()
    b.chips = chips if chips is not None else ChipFeedback()
    return b


def step(domain, ctx, state, prev):
    """One macro step: macro_step(prev, nxt) then swap (nxt becomes bus_prev of the next step)."""
    state.bus_prev = prev
    state.bus_next = Bus()
    domain.macro_step(ctx, state, prev, state.bus_next)
    record = state.bus_next.wear
    state.clock.tick += ctx.ipo_ticks * ctx.macro_ipo
    state.clock.macro += 1
    return record


def surface_rows(edge, tooth=0, slot=0, tick=0):
    """Surface events of every element of one tooth (the tooth finishes the final surface)."""
    ne = np.shape(edge.elem_z_m)[1]
    rows = np.zeros((ne, 16))
    rows[:, 0] = tick
    rows[:, 1] = tooth
    rows[:, 2] = np.arange(ne)
    rows[:, 15] = slot
    return rows


def run(domain, ctx, state, edge, steps, start=0, heat_rec=None, **cut):
    """``steps`` macro steps of steady cutting (frames sized to the context's macro step)."""
    n_ticks = ctx.macro_ipo
    dt = ctx.macro_s
    record = None
    for i in range(start, start + steps):
        prev = bus(edge, [cutting(edge, dt=dt, **cut)], heat_rec if heat_rec is not None else heat(edge),
                   path_frame(tick0=i * ctx.ipo_ticks * n_ticks, n_ticks=n_ticks), tick=(i + 1) * ctx.ipo_ticks * n_ticks)
        record = step(domain, ctx, state, prev)
    return record
