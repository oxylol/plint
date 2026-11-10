"""The cutting contact of one edge, through the mechanics coefficient pipeline the run uses.

:class:`ContactModel` builds one process map (``cnctwin.mechanics.coefficients.build_process_map``) for
the case's material, tool, rake and friction, on the case's own speeds and uncut thicknesses (so the
map is exact at the tested conditions), and evaluates ``group_coefficients`` for one edge group at a
chip thickness, speed and tool temperature. The heat of the contact is split as thermal's partition
does (thermal/partition.py): tangential power P = F_t V, rake heat to the tool P x power_frac_rake x
frac_rake_to_tool, edge and wear-land heat to the tool (P x power_frac_edge + tau_land VB b V) x
frac_flank_to_tool, deposited over the flank contact the share was computed for (wear land + the edge arc below
the stagnation point, GroupCoefficients.flank_contact_m), as thermal deposits it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...db.models import params
from ...mechanics import coefficients as CO
from ...mechanics import contact as C
from ...mechanics import heat as HT
from ...mechanics.coefficients import EdgeGroup, SlowState


@dataclass
class ContactState:
    h_m: float
    speed_m_s: float
    width_m: float
    tool_temp_K: float
    T_interface_K: float
    T_chip_K: float
    T_shear_K: float
    contact_length_m: float
    chip_ratio: float                 # t1 / t2 (Oxley, shearing layer)
    regime: int
    force_t_N: float
    force_r_N: float
    power_W: float
    rake_to_tool_W: float
    flank_to_tool_W: float
    chip_W: float
    h_min_m: float
    edge_radius_m: float
    flank_contact_m: float            # wear land + edge arc below the stagnation point (thermal's flank contact)
    shear_angle_rad: float

    @property
    def tool_W(self) -> float:
        return self.rake_to_tool_W + self.flank_to_tool_W

    flank_temp_K: float = 0.0         # mean tool temperature over the flank (edge) contact (Blok, tool side)
    # the heat by zone and the shares that stay in the work (thermal's partition; docs/models/thermal.md 3.4)
    shear_W: float = 0.0              # primary-zone heat P x power_frac_shear
    edge_W: float = 0.0               # edge (ploughing) heat P x power_frac_edge
    land_W: float = 0.0               # wear-land friction tau_mean VB b V
    frac_shear_to_chip: float = 0.0
    frac_flank_to_tool: float = 0.0
    VB_m: float = 0.0

    @property
    def forms_chip(self) -> bool:
        return self.chip_ratio > 0.0 and self.contact_length_m > 0.0

    @property
    def contact_mean_K(self) -> float:
        """Mean temperature over the whole tool-chip and tool-work contact (rake and flank contact
        areas weighted by their lengths): what a tool-work thermocouple averages."""
        l_c = self.contact_length_m if self.forms_chip else 0.0
        l_f = self.flank_contact_m
        if l_c + l_f <= 0.0:
            return self.tool_temp_K
        return (l_c * self.T_interface_K + l_f * self.flank_temp_K) / (l_c + l_f)


class ContactModel:
    def __init__(self, work, tool, *, rake_rad: float, clearance_rad: float, edge_radius_m: float, work_temp_K: float,
                 speeds_m_s, h_values_m, incl_rad: float = 0.0, friction_factor: float = 1.0,
                 tool_temps_K=None):
        self.work = work
        self.tool = tool
        self.rake = float(rake_rad)
        self.clearance = float(clearance_rad)
        self.edge_radius = float(edge_radius_m)
        self.incl = float(incl_rad)
        self.work_temp = float(work_temp_K)
        self.friction_factor = float(friction_factor)
        self.h_min = C.min_chip_thickness(work, self.edge_radius)
        t1 = [h - self.h_min for h in h_values_m if h - self.h_min > 0.0]
        grid = sorted(set(np.round(np.concatenate([CO.default_t1_grid(), np.array(t1, dtype=float)]), 15)))
        speeds = sorted(set(float(v) for v in speeds_m_s))
        temps = CO.default_tool_temp_grid(self.work_temp) if tool_temps_K is None else np.array(sorted(tool_temps_K))
        self.pmap = CO.build_process_map(work, tool, self.rake, speed_grid_m_s=np.array(speeds),
                                         work_temp_K=self.work_temp, friction_factor=self.friction_factor,
                                         t1_grid_m=np.array(grid), tool_temp_grid_K=np.asarray(temps, dtype=float))
        self.group = EdgeGroup(self.rake, self.incl, self.clearance, self.edge_radius)

    def state(self, h_m: float, speed_m_s: float, tool_temp_K: float, width_m: float,
              contact_time_s: float | None = None, VB_m: float = 0.0) -> ContactState:
        slow = SlowState(float(speed_m_s), tool_temp_K=float(tool_temp_K), tool_flank_temp_K=float(tool_temp_K),
                         work_temp_K=self.work_temp, friction_factor=self.friction_factor, VB_m=float(VB_m),
                         contact_width_m=float(width_m), contact_time_s=contact_time_s)
        gc = CO.group_coefficients(self.work, self.tool, self.group, slow, np.array([float(h_m)]), self.pmap)
        ft = float(gc.Kt_Pa[0]) * h_m + gc.Kte_N_per_m
        fr = float(gc.Kr_Pa[0]) * h_m + gc.Kre_N_per_m
        power = ft * width_m * speed_m_s
        p_rake = float(gc.power_frac_rake[0])
        p_edge = float(gc.power_frac_edge[0])
        p_shear = float(gc.power_frac_shear[0])
        # the wear land's friction (the run's flank force, tau_mean VB per unit edge length) is flank heat too
        land = float(gc.flank_tau_mean_Pa) * float(VB_m) * width_m * speed_m_s if VB_m > 0.0 else 0.0
        rake_tool = power * p_rake * float(gc.frac_rake_to_tool[0])
        # thermal's partition: the flank share applies to the flank (land) and ploughing (edge) heat together and the
        # tool's part is deposited over the contact the share was computed for (mechanics: wear land + edge arc)
        flank_tool = (power * p_edge + land) * float(gc.frac_flank_to_tool[0])
        chip = power * (p_shear * float(gc.frac_shear_to_chip[0]) + p_rake * (1.0 - float(gc.frac_rake_to_tool[0])))
        if gc.flank_contact_m.size:
            l_f = float(gc.flank_contact_m[0])
        else:
            l_f = float(VB_m) + float(params("thermal-tool")["flank_contact_edge_factor"]) * self.edge_radius
        power += land
        ft += land / (width_m * speed_m_s) if width_m * speed_m_s > 0.0 else 0.0
        t_flank = float(tool_temp_K)
        if flank_tool > 0.0 and l_f > 0.0 and self.tool is not None:
            q_f = flank_tool / (l_f * width_m)
            t_flank += q_f * HT.tool_side_resistance(l_f, float(tool_temp_K), self.tool, width_m, contact_time_s)
        return ContactState(h_m=float(h_m), speed_m_s=float(speed_m_s), width_m=float(width_m),
                            tool_temp_K=float(tool_temp_K), T_interface_K=float(gc.T_interface_K[0]),
                            T_chip_K=float(gc.T_chip_K[0]), T_shear_K=float(gc.T_shear_K[0]),
                            contact_length_m=float(gc.contact_length_m[0]), chip_ratio=float(gc.chip_ratio[0]),
                            regime=int(gc.regime[0]), force_t_N=ft * width_m, force_r_N=fr * width_m, power_W=power,
                            rake_to_tool_W=rake_tool, flank_to_tool_W=flank_tool, chip_W=chip, h_min_m=self.h_min,
                            edge_radius_m=self.edge_radius, flank_contact_m=l_f,
                            shear_angle_rad=float(gc.shear_angle_rad[0]), flank_temp_K=t_flank,
                            shear_W=(power - land) * p_shear, edge_W=(power - land) * p_edge, land_W=land,
                            frac_shear_to_chip=float(gc.frac_shear_to_chip[0]),
                            frac_flank_to_tool=float(gc.frac_flank_to_tool[0]), VB_m=float(VB_m))

    def edge_sources(self, st: ContactState, duration_s: float = 1.0):
        """The machined surface's sources of this contact as thermal builds them (thermal.domain._edge_sources):
        the work's shares of the shear-zone, wear-land and edge heat, the edge heat split into friction (surface)
        and pressure (layer to the ploughed-zone depth) by contact.contact_power_parts."""
        from ...thermal import surfacetemp as ST
        from ...thermal.domain import _edge_split_cached
        from ...mechanics import params as MP
        keep_f = 1.0 - st.frac_flank_to_tool
        friction, arc = _edge_split_cached(*(float(f"{x:.3g}") for x in (self.edge_radius, self.h_min, self.rake,
                                                                          self.incl)))
        arc_f = max(st.flank_contact_m - st.VB_m, 0.0) or arc
        delta = float(MP.number(MP.CONTACT, "ploughing_zone_depth_ratio")) * arc_f
        tan_phi = np.tan(st.shear_angle_rad) if st.shear_angle_rad > 0.0 else 0.0
        l_s = st.h_m / tan_phi if tan_phi > 0.0 else 0.0
        return ST.EdgeSources(speed_m_s=st.speed_m_s, width_m=st.width_m, duration_s=float(duration_s),
                              shear_W=st.shear_W * (1.0 - st.frac_shear_to_chip) if l_s > 0.0 else 0.0,
                              shear_length_m=float(l_s), arc_W=friction * st.edge_W * keep_f,
                              plough_W=(1.0 - friction) * st.edge_W * keep_f, arc_length_m=arc_f,
                              plough_depth_m=delta, land_W=st.land_W * keep_f, land_length_m=st.VB_m)


__all__ = ["ContactModel", "ContactState"]
