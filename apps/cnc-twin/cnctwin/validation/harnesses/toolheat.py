"""Tool temperatures where the experiments measured them: thermal's tool model at a point.

The body is thermal's own tool ladder (``cnctwin.thermal.toolmodel.ToolThermal``, a turning insert on
its holder bar: 1-D finite volumes along the insert, seat contact, the clamped bar to the turret,
air convection and radiation) with its 3-D cutting-region field (docs/models/thermal.md 2.4, built at the
first contact): the contact's rake and flank heat from :mod:`contact` is deposited as two contact patches
(``ToolThermal.deposit_contact``, the wedge angle beta = pi/2 - rake - clearance and the edge radius of the
case) and the ladder and field are stepped together; the coefficient pipeline's partition takes the body
temperature of the first insert node. The temperature at a point near the edge is the field's
(``ToolThermal.temperature_at`` in the edge frame: x along the rake face from the edge, y along the edge with
the contact on [0, width], z the rake face's outward normal; points outside the field take the nearest cell).
``local_rise`` keeps the analytic quarter-space composition (docs/models/thermal.md 2.2) for reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special

from ...db.models import params
from ...project import physics_defaults
from ...thermal import kernels as TK
from ...thermal import props
from ...thermal.response import gauss_legendre
from ...thermal.schema import settings as thermal_settings
from ...thermal.toolmodel import ToolEnvironment, ToolSpec, ToolThermal
from .contact import ContactModel, ContactState


def _tp() -> dict:
    return dict(params("thermal-tool"))


def edge_element_m() -> float:
    phys = physics_defaults()
    value = phys.get("edge_element_mm") if isinstance(phys, dict) else getattr(phys, "edge_element_mm", None)
    return float(value) * 1e-3 if value else 0.5e-3


@dataclass
class InsertSetup:
    tool_material_id: str
    ic_m: float
    thickness_m: float
    bar_h_m: float
    bar_b_m: float
    overhang_m: float


def insert_setup(case, tool_material_id: str, col) -> InsertSetup:
    """Insert and holder-bar geometry of a turning case: stated values, else the thermal model's defaults."""
    p = _tp()
    ic = case.si("tool", "ic_mm", "inscribed_circle_mm")
    if not isinstance(ic, (int, float)):
        ic = float(p["default_insert_ic_m"])
        col.assume("insert inscribed circle", f"{ic * 1e3:.2f} mm", "not stated: thermal model default "
                   "(default_insert_ic_m)", "db/models/thermal-tool.yaml")
    thick = case.si("tool", "insert_thickness_mm", "thickness_mm", "edge_height_mm")
    if not isinstance(thick, (int, float)):
        thick = float(p["default_insert_thickness_m"])
        col.assume("insert thickness", f"{thick * 1e3:.2f} mm", "not stated: thermal model default "
                   "(default_insert_thickness_m)", "db/models/thermal-tool.yaml")
    section = case.value("tool", "holder_section_mm")
    if isinstance(section, (list, tuple)) and len(section) == 2:
        bar_h, bar_b = float(section[0]) * 1e-3, float(section[1]) * 1e-3
    else:
        bar_h = bar_b = float(p["default_bar_section_m"])
        col.assume("holder bar section", f"{bar_h * 1e3:.0f} x {bar_b * 1e3:.0f} mm", "not stated: thermal model "
                   "default (default_bar_section_m)", "db/models/thermal-tool.yaml")
    overhang = float(p["default_bar_overhang_m"])
    col.assume("holder bar overhang", f"{overhang * 1e3:.0f} mm", "not stated: thermal model default "
               "(default_bar_overhang_m)", "db/models/thermal-tool.yaml")
    return InsertSetup(tool_material_id, float(ic), float(thick), bar_h, bar_b, overhang)


class InsertThermal:
    """Thermal's tool ladder of a turning insert, stepped by the harness."""

    def __init__(self, setup: InsertSetup, ambient_K: float):
        p = _tp()
        spec = ToolSpec(kind="turning", tool_material_id=setup.tool_material_id, insert_ic_m=setup.ic_m,
                        insert_thickness_m=setup.thickness_m, bar_h_m=setup.bar_h_m, bar_b_m=setup.bar_b_m,
                        bar_overhang_m=setup.overhang_m, holder_material_id=str(p["holder_material"]))
        st = thermal_settings({})
        self.energy_tol = float(st.energy_tol)
        self.mat_tool = props.tool_material_table_by_id(setup.tool_material_id)
        self.mat_holder = props.material_table_by_id(str(p["holder_material"]))
        self.model = ToolThermal(0, spec, self.mat_tool, self.mat_holder, float(st.tool_cell_m), ambient_K, 0.0)
        self.model.enable_field(st)        # thermal's 3-D cutting-region field, built at the first contact
        self.env = ToolEnvironment(in_spindle=False, omega_rad_s=0.0, T_air_K=ambient_K, T_end_K=ambient_K)
        self.time_s = 0.0
        self.heat_J = 0.0

    def _insert_nodes(self) -> np.ndarray:
        nodes = self.model.layout.tool_nodes
        return nodes[np.argsort(self.model.layout.z_m[nodes])]

    def tip_temperature(self) -> float:
        return float(self.model.ladder.T[self._insert_nodes()[0]])

    def temperature_at(self, distance_m: float) -> float:
        """Body temperature at a distance from the cutting corner along the insert (ladder nodes)."""
        nodes = self._insert_nodes()
        return float(np.interp(distance_m, self.model.layout.z_m[nodes], self.model.ladder.T[nodes]))

    def step(self, dt_s: float, heat_W: float, st: ContactState | None = None, wedge: float | None = None) -> None:
        if st is None:
            self.model.deposit(np.array([0.0]), np.array([heat_W * dt_s]))
        else:
            self.model.deposit_contact(st.rake_to_tool_W * dt_s, st.flank_to_tool_W * dt_s,
                                       max(st.contact_length_m, st.edge_radius_m), st.flank_contact_m, st.width_m,
                                       wedge_angle_rad=0.5 * math.pi / wedge, edge_radius_m=st.edge_radius_m)
        self.model.step(dt_s, self.env, self.energy_tol)
        self.time_s += dt_s
        self.heat_J += heat_W * dt_s

    def k_alpha(self, temp_K: float) -> tuple[float, float]:
        tab = self.mat_tool
        k = float(np.interp(temp_K, tab.T_K, tab.k_W_mK))
        cp = float(np.interp(temp_K, tab.T_K, tab.cp_J_kgK))
        return k, k / (float(tab.rho_kg_m3[0]) * cp)


def run_cut(contact: ContactModel, insert: InsertThermal, h_m: float, speed_m_s: float, width_m: float,
            duration_s: float, dt_s: float, contact_time_s: float | None = None) -> tuple[ContactState, list]:
    """Cut for ``duration_s`` (tool heat fed back through the body temperature at the edge); returns the
    contact state at the end and the history [(t, tip body temperature, interface temperature)]."""
    history = []
    t = 0.0
    while t < duration_s - 1e-12:
        step = min(dt_s, duration_s - t)
        st = contact.state(h_m, speed_m_s, insert.tip_temperature(), width_m, contact_time_s)
        insert.step(step, st.tool_W)
        t += step
        history.append((t, insert.tip_temperature(), st.T_interface_K))
    return contact.state(h_m, speed_m_s, insert.tip_temperature(), width_m, contact_time_s), history


def wedge_factor(rake_rad: float, clearance_rad: float) -> float:
    p = _tp()
    beta = min(max(0.5 * math.pi - rake_rad - clearance_rad, float(p["min_wedge_angle_rad"])),
               float(p["max_wedge_angle_rad"]))
    return 0.5 * math.pi / beta


def local_rise(st: ContactState, x_m: float, y_m: float, z_m: float, *, k: float, alpha: float, time_s: float,
               wedge: float, side_adiabatic: bool = False, gl_points: int = 16) -> float:
    """Local field above the body temperature at (x along the rake face from the edge, y across the
    width from the contact centre, z depth below the rake face). ``side_adiabatic``: the point lies on
    a free side face at the lateral end of the contact (mirror across it)."""
    glx, glw = gauss_legendre(gl_points)
    b = st.width_m
    l_c = max(st.contact_length_m, st.edge_radius_m, 1e-6)
    l_f = max(st.flank_contact_m, 1e-6)
    half_w = b if side_adiabatic else 0.5 * b
    y_rel = 0.0 if side_adiabatic else y_m
    q_r = st.rake_to_tool_W / (l_c * b) if st.rake_to_tool_W > 0.0 else 0.0
    q_f = st.flank_to_tool_W / (l_f * b) if st.flank_to_tool_W > 0.0 else 0.0
    rise = 0.0
    if q_r > 0.0:
        rise += float(TK.moving_rect_rise(q_r, l_c, half_w, 0.0, k, alpha, time_s, x_m, y_rel, z_m, glx, glw))
    if q_f > 0.0:
        rise += float(TK.moving_rect_rise(q_f, l_f, half_w, 0.0, k, alpha, time_s, z_m, y_rel, x_m, glx, glw))
    power = st.tool_W * (2.0 if side_adiabatic else 1.0)
    r_far = 2.0 * max(l_c, l_f, edge_element_m())
    far = power / (math.pi * k * r_far) * float(special.erfc(r_far / (2.0 * math.sqrt(alpha * time_s))))
    return max(wedge * (rise - far), 0.0)


def temperature_at_point(insert: InsertThermal, st: ContactState, x_m: float, y_m: float, z_m: float, *,
                         time_s: float, wedge: float, side_adiabatic: bool = False) -> float:
    y = 0.0 if side_adiabatic else 0.5 * st.width_m + y_m        # field frame: the contact on [0, width]
    return float(insert.model.temperature_at(np.array([[x_m, y, -z_m]]), outside="nearest")[0])


def window_mean(insert: InsertThermal, st: ContactState, xs: tuple[float, float], zs: tuple[float, float],
                y_m: float, *, time_s: float, wedge: float, side_adiabatic: bool, n: int = 4) -> float:
    """Mean temperature over a rectangle x in xs, z in zs at lateral position y (Gauss-Legendre)."""
    gx, gw = gauss_legendre(n)
    total = 0.0
    for a, wa in zip(gx, gw):
        x = 0.5 * (xs[1] - xs[0]) * a + 0.5 * (xs[1] + xs[0])
        for c, wc in zip(gx, gw):
            z = 0.5 * (zs[1] - zs[0]) * c + 0.5 * (zs[1] + zs[0])
            total += 0.25 * wa * wc * temperature_at_point(insert, st, x, y_m, z, time_s=time_s, wedge=wedge,
                                                           side_adiabatic=side_adiabatic)
    return total


def disc_mean_on_rake(insert: InsertThermal, st: ContactState, x0: float, y0: float, radius: float, *, time_s: float,
                      wedge: float, n: int = 4) -> float:
    """Mean rake-face temperature over a disc (an IR spot) centred at (x0, y0), z = 0."""
    gx, gw = gauss_legendre(n)
    total = 0.0
    weight = 0.0
    for a, wa in zip(gx, gw):
        r = 0.5 * radius * (a + 1.0)
        for c, wc in zip(gx, gw):
            th = math.pi * (c + 1.0)
            w = wa * wc * r
            x = max(x0 + r * math.cos(th), 0.0)
            total += w * temperature_at_point(insert, st, x, y0 + r * math.sin(th), 0.0, time_s=time_s, wedge=wedge)
            weight += w
    return total / weight


__all__ = ["InsertSetup", "insert_setup", "InsertThermal", "run_cut", "wedge_factor", "local_rise",
           "temperature_at_point", "window_mean", "disc_mean_on_rake", "edge_element_m"]
