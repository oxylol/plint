"""Small inline fixtures for the chips tests (no database files: material, machine and coolant as SI views)."""

from __future__ import annotations

import numpy as np

from cnctwin.core.bus import MotionFrame, PathFrame, RemovalContact, ToolContact
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState, ToolSlot, ToolTable
from cnctwin.db.simap import wrap
from cnctwin.chips import abi
from cnctwin.chips.protocols import BoxScene, Formation, FormationQuery

TICK_S = 1e-6
IPO = 1000
MACRO_IPO = 10


def material_raw(family: str = "aluminium", density: float = 2700.0) -> dict:
    return {"id": "test-alloy", "family": family,
            "physical": {"density_kg_m3": {"value": density, "src": "est"},
                         "specific_heat_J_kgK": {"table": [[20, 896], [100, 921], [200, 963], [300, 1005]]},
                         "solidus_degC": {"value": 582}, "yield_MPa": {"value": 276}},
            "chip": {"tendency": "continuous"}, "ignition": None, "composite": None}


def material_jc_raw() -> dict:
    """An aluminium with a Johnson-Cook flow stress (6061-T6 constants), enough for cnctwin.mechanics."""
    raw = material_raw()
    raw["physical"]["conductivity_W_mK"] = {"table": [[20, 167], [100, 172], [200, 177]]}
    raw["physical"]["youngs_modulus_GPa"] = {"table": [[20, 68.9], [200, 61.8]]}
    raw["physical"]["poisson"] = {"value": 0.33, "src": "est"}
    raw["physical"]["liquidus_degC"] = {"value": 652, "src": "est"}
    raw["flow_stress"] = {"model": "johnson_cook", "A_MPa": 324, "B_MPa": 114, "n": 0.42, "C": 0.002, "m": 1.34,
                          "ref_strain_rate_1_s": 1.0, "T_ref_degC": 20, "T_melt_degC": 652, "src": "est"}
    raw["friction"] = [{"tool_class": "carbide", "mu_sliding": {"value": 0.5, "src": "est"}}]
    raw["cutting"] = {"min_chip_thickness_ratio": {"value": 0.3, "src": "est"}}
    raw["chip"] = {"tendency": "continuous", "breakability": {"value": "poor", "src": "est"}}
    return raw


def machine_raw() -> dict:
    return {"type": "vertical_mill",
            "kinematics": {"gravity_direction": [0, 0, -1]},
            "spindles": [{"id": "main", "kind": "milling"}],
            "enclosure": {"box_mm": [[-400, -300, -200], [500, 400, 600]], "floor_slope_deg": 30},
            "chip_conveyor": {"type": "auger", "rate_kg_h": 60}}


def emulsion_raw() -> dict:
    return {"id": "emulsion", "kind": "emulsion",
            "density_kg_m3": {"table": [[20, 992.0], [60, 976.0]]},
            "viscosity_mPa_s": {"table": [[20, 1.2], [60, 0.56]]},
            "specific_heat_J_kgK": {"table": [[20, 4022]]},
            "conductivity_W_mK": {"table": [[20, 0.55]]},
            "surface_tension_mN_m": {"value": 35, "src": "est"}}


def tool_raw(number=1, diameter_mm=10.0, kind="square_endmill", flutes=4, helix=30.0, cutting_length_mm=22.0,
             gullet_area_mm2=None, holes=None) -> dict:
    geometry = {"diameter_mm": diameter_mm, "flutes": flutes, "helix_deg": helix,
                "cutting_length_mm": cutting_length_mm}
    if gullet_area_mm2 is not None:
        geometry["gullet_area_mm2"] = gullet_area_mm2
    if holes is not None:
        geometry["coolant_holes"] = holes
    return {"number": number, "tool": {"type": kind, "geometry": geometry},
            "holder": {"nose_diameter_mm": 32.0, "gauge_length_mm": 60.0}, "stickout_mm": 35.0}


def project(family="aluminium", tools=None, physics=None, stock_mm=(100.0, 80.0, 30.0), coolant=True, material=None):
    raw = {"material": material or material_raw(family), "machine": machine_raw(),
           "stock": {"kind": "block", "size_mm": list(stock_mm)},
           "environment": {"ambient_degC": 20},
           "physics": physics or {"particles_per_tooth": 0.5, "chips": {"pile_cell_mm": 2.0, "pile_margin_mm": 20.0}},
           "tools": {"list": tools or [tool_raw()]}}
    if coolant:
        raw["coolant"] = {"fluid": emulsion_raw()}
    return wrap(raw)


class FakeFormation:
    """A fixed chip (the mechanics library's job in a run)."""

    def __init__(self, form=62, thickness_m=1.5e-4, width_m=2e-3, length_m=4e-3, ratio=0.5, curl_m=2e-3):
        self.form = form
        self.thickness = thickness_m
        self.width = width_m
        self.length = length_m
        self.ratio = ratio
        self.curl = curl_m
        self.queries: list[FormationQuery] = []

    def formation(self, query: FormationQuery) -> Formation:
        self.queries.append(query)
        return Formation(shear_angle_rad=0.4, chip_ratio=self.ratio, shear_strain=2.5, strain_rate_1_s=1e4,
                         chip_flow_angle_rad=query.incl_rad, chip_temp_K=query.temp_K if query.temp_K > 0 else 400.0,
                         form_code=self.form, bue_active=False, segmented=False, thickness_m=self.thickness,
                         width_m=self.width, length_m=self.length, curl_radius_m=self.curl, colour_rgb=(200, 200, 210))


def context(proj, seed: int = 7) -> RunContext:
    return RunContext(project=proj, db=None, run_dir=None, seed=seed, tick_s=TICK_S, ipo_ticks=IPO, macro_ipo=MACRO_IPO)


def sim_state(n_slots: int = 1) -> SimState:
    state = SimState()
    state.tools = ToolTable(tuple(ToolSlot(slot=i, number=i + 1, tool_id=f"t{i}", holder_id="h") for i in range(n_slots)))
    return state


def path_frame(step: int, tip_M, axis=(0.0, 0.0, 1.0), omega=-1047.0, slot=0, R_M_S=None, o_S_M=(0.0, 0.0, 0.0),
               feed=(0.0, 0.0, 0.0), events=None) -> PathFrame:
    """One macro step's path record: MACRO_IPO ticks; tip_M a 3-vector or a function of the tick index."""
    tick0 = step * MACRO_IPO * IPO
    ticks = tick0 + IPO * (np.arange(MACRO_IPO) + 1)
    R = np.eye(3) if R_M_S is None else np.asarray(R_M_S, dtype=float)
    o = np.asarray(o_S_M, dtype=float)
    if callable(tip_M):
        tips = np.array([tip_M(k) for k in range(MACRO_IPO)], dtype=float)
    else:
        tips = np.tile(np.asarray(tip_M, dtype=float), (MACRO_IPO, 1))
    tip_S = (tips - o) @ R
    pose_M_T = np.eye(4)
    pose_M_T[:3, 3] = tips[-1]
    pose_S_T = np.eye(4)
    pose_S_T[:3, :3] = R.T
    pose_S_T[:3, 3] = tip_S[-1]
    return PathFrame(path=0, tick0=tick0, ipo_ticks=IPO, tick=ticks.astype(np.int64),
                     block_index=np.zeros(MACRO_IPO, np.int32), tip_S_m=tip_S, tip_M_m=tips,
                     axis_T_M=np.tile(np.asarray(axis, dtype=float), (MACRO_IPO, 1)),
                     spindle_omega_rad_s=np.full(MACRO_IPO, omega), feed_vel_M_m_s=np.tile(np.asarray(feed), (MACRO_IPO, 1)),
                     rapid=np.zeros(MACRO_IPO, bool), tool_slot=slot, spindle_id="main", workpiece=0,
                     pose_M_T=pose_M_T, pose_S_T=pose_S_T, events=list(events or []))


def motion(step: int, **kwargs) -> MotionFrame:
    frame = path_frame(step, **kwargs)
    return MotionFrame(time_s=(step + 1) * MACRO_IPO * IPO * TICK_S, tick=(step + 1) * MACRO_IPO * IPO, paths=[frame])


def emission_rows(step: int, n: int, mass_each: float, pos, vel, temp=450.0, contact=0, slot=0, h=1e-4, b=2e-3,
                  l=6e-3) -> np.ndarray:
    rows = np.zeros((n, abi.EMISSION_WIDTH))
    tick0 = step * MACRO_IPO * IPO
    rows[:, abi.EM["tick"]] = tick0 + (np.arange(n) + 0.5) * (MACRO_IPO * IPO / n)
    rows[:, abi.EM["contact"]] = contact
    rows[:, abi.EM["tool_slot"]] = slot
    rows[:, abi.EM["tooth"]] = np.arange(n) % 4
    rows[:, abi.EM["mass_kg"]] = mass_each
    rows[:, abi.EM["x_M_m"]:abi.EM["z_M_m"] + 1] = np.asarray(pos, dtype=float).reshape(-1, 3)
    rows[:, abi.EM["vx_M_m_s"]:abi.EM["vz_M_m_s"] + 1] = np.asarray(vel, dtype=float).reshape(-1, 3)
    rows[:, abi.EM["temp_K"]] = temp
    rows[:, abi.EM["h_m"]] = h
    rows[:, abi.EM["b_m"]] = b
    rows[:, abi.EM["l_m"]] = l
    rows[:, abi.EM["v_cut_m_s"]] = 50.0
    return rows


def removal(volume_m3: float, contact=0, slot=0, point=(0.0, 0.0, 0.0), ae=5e-3, ap=5e-3):
    from cnctwin.core.bus import RemovalFrame
    rc = RemovalContact(contact=contact, tool_slot=slot, workpiece=0, path=0, removed_volume_m3=volume_m3,
                        geometry=ToolContact(in_cut=volume_m3 > 0, point_M_m=np.asarray(point, dtype=float), ae_m=ae,
                                             ap_m=ap))
    return RemovalFrame(contacts=[rc])


def scene_box(lo=(0.0, 0.0, 0.0), hi=(0.1, 0.08, 0.03)) -> BoxScene:
    return BoxScene(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
