"""Exit burrs, edge breakout and composite exit damage (docs/models/surface.md section 5).

Exit burrs form where a cutting edge leaves the workpiece through a free surface (the exit face) at the
edge with the surface it is machining (Gillespie & Blotter 1976: the material ahead of the edge bends over
the exit edge instead of being sheared off, the roll-over burr; Pekelharing 1978: foot formation; Ko &
Dornfeld 1991: initiation by a negative deformation plane from the edge to the exit face). The model:

    H = C_H * b_x * (sin kappa / sin 45 deg)^e_k * (cos psi)^e_psi * F_eos * F_duct
    T = C_T * (h_x + c_r * r_beta)

* b_x - extent of the cut on the exit face perpendicular to the exit edge: the depth of cut ap for an exit
  face parallel to the tool axis (side face of a milled part, end face in turning), the uncut chip thickness
  for an exit face normal to the tool axis (top burr), interpolated with the face orientation;
* kappa - entering angle of the exiting edge (C_H, e_k from Machado et al. 2003: H = ap at kappa_r 45 deg);
* psi - angle between the pushing direction (cutting velocity for rotating tools, feed for turning) and the
  outward normal of the exit face (a push-out exit psi = 0 gives the full burr, a tangential exit none);
* F_eos - exit order (Hashimura et al. 1999): 1 when the tool tip leaves the exit face last, else C_eos;
* F_duct = min(1, eps_f / eps_ref); below eps_brittle the edge breaks out instead (Ko & Dornfeld 1996), a
  negative burr of depth C_bo * H;
* h_x - uncut chip thickness at the exit, r_beta the hone radius (thickness grows with feed and edge radius:
  Hashimura, Chang & Dornfeld 1999).

Drilling exit burrs (Kim & Dornfeld 2002): the remaining disc under the drill yields when the thrust reaches
its limit load F = k pi sigma_f t^2 (Hopkins & Prager 1953, simply supported uniform load: k = 1.5), so the
yield-onset thickness is t_c = sqrt(F / (k pi sigma_f)); H = C_D t_c F_duct, T = C_TD t_c; a crown burr
(petals) when eps_f < eps_crown (Min, Kim & Dornfeld 2001), else uniform.

Composites (material.composite): push-out delamination at the drill exit when the thrust exceeds the
Hocheng & Dharan (1990) critical thrust F* = pi sqrt(8 G_IC E h^3 / (3 (1 - nu^2))) of the uncut thickness h
under the drill (or the material file's critical_thrust_N); every exit through a composite edge leaves
uncut / pulled-out fibres of depth C_po * r_beta instead of a metal burr.

Error band: burr heights within a factor of 2 (-50 % / +100 %) of measured exit burrs for ductile metals
(the coefficients are estimates except C_H); breakout depth and pull-out depth are indications only;
the delamination onset is the published closed form (its own error vs drilling tests about +-20 % in
Hocheng & Dharan's comparison).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KIND_ROLLOVER = "rollover"
KIND_BREAKOUT = "breakout"
KIND_UNIFORM = "drill_uniform"
KIND_CROWN = "drill_crown"
KIND_PULLOUT = "fibre_pullout"
KIND_DELAMINATION = "delamination"


@dataclass(frozen=True)
class BurrResult:
    height_m: float
    thickness_m: float
    kind: str
    factors: dict


class BurrModel:
    def __init__(self, params: dict, mat):
        self.p = params
        self.mat = mat

    def ductility(self) -> float:
        return min(1.0, self.mat.fracture_strain / float(self.p["ductile_reference_strain"]))

    def exit_burr(self, h_x: float, b_x: float, kappa_rad: float, cos_psi: float, tip_last: bool,
                  edge_r: float) -> BurrResult:
        """Roll-over (or breakout, or fibre pull-out) at one exit (see the module doc)."""
        p = self.p
        h_x = max(h_x, 0.0)
        b_x = max(b_x, 0.0)
        thickness = float(p["rollover_thickness_ratio"]) * (h_x + float(p["edge_radius_ratio"]) * max(edge_r, 0.0))
        if self.mat.composite:
            depth = float(p["pullout_depth_ratio"]) * max(edge_r, 0.0)
            return BurrResult(depth, 0.0, KIND_PULLOUT, {"edge_radius_m": edge_r})
        f_k = (max(math.sin(kappa_rad), 0.0) / math.sin(math.radians(45.0))) ** float(p["rollover_kappa_exponent"])
        f_psi = max(cos_psi, 0.0) ** float(p["exit_angle_exponent"])
        f_eos = 1.0 if tip_last else float(p["eos_top_last_factor"])
        f_d = self.ductility()
        height = float(p["rollover_height_ratio"]) * b_x * f_k * f_psi * f_eos
        factors = {"b_x_m": b_x, "kappa": f_k, "exit_angle": f_psi, "exit_order": f_eos, "ductility": f_d}
        if self.mat.fracture_strain < float(p["brittle_fracture_strain"]):
            return BurrResult(float(p["breakout_depth_ratio"]) * height, 0.0, KIND_BREAKOUT, factors)
        return BurrResult(height * f_d, thickness if height > 0.0 else 0.0, KIND_ROLLOVER, factors)

    def flow_stress(self) -> float:
        """Mean flow stress of the yielding disc: (yield + ultimate) / 2 (the plastic collapse of a strain-
        hardening plate lies between the two limit loads)."""
        return 0.5 * (self.mat.yield_Pa + self.mat.ultimate_Pa)

    def onset_thickness(self, thrust_N: float) -> float:
        k = float(self.p["drill_plate_load_factor"])
        sigma = self.flow_stress()
        if thrust_N <= 0.0 or sigma <= 0.0:
            return 0.0
        return math.sqrt(thrust_N / (k * math.pi * sigma))

    def drill_burr(self, thrust_N: float) -> BurrResult:
        t_c = self.onset_thickness(thrust_N)
        p = self.p
        kind = KIND_CROWN if self.mat.fracture_strain < float(p["crown_fracture_strain"]) else KIND_UNIFORM
        height = float(p["drill_burr_height_ratio"]) * t_c * self.ductility()
        return BurrResult(height, float(p["drill_burr_thickness_ratio"]) * t_c, kind,
                          {"onset_thickness_m": t_c, "thrust_N": thrust_N, "flow_stress_Pa": self.flow_stress()})


def hocheng_dharan_N(G_IC_J_m2: float, E_Pa: float, nu: float, h_m: float) -> float:
    """Critical thrust of push-out delamination (Hocheng & Dharan 1990)."""
    if h_m <= 0.0:
        return 0.0
    return math.pi * math.sqrt(8.0 * G_IC_J_m2 * E_Pa * h_m ** 3 / (3.0 * (1.0 - nu * nu)))


def critical_thrust(mat, h_m: float) -> tuple[float | None, str]:
    """(critical thrust, source) of a composite at uncut thickness h (None: not enough data)."""
    comp = mat.composite
    if not comp:
        return None, ""
    if "G_IC_J_m2" in comp:
        E = float(comp.get("E_Pa", mat.youngs_Pa))
        nu = float(comp.get("poisson", mat.poisson))
        return hocheng_dharan_N(float(comp["G_IC_J_m2"]), E, nu, h_m), "hocheng-dharan"
    if "critical_thrust_N" in comp:
        return float(comp["critical_thrust_N"]), "material file"
    return None, ""


def exit_extent(ap_m: float, h_m: float, exit_normal: np.ndarray, axis: np.ndarray) -> float:
    """b_x: the cut's extent on the exit face (ap for a face parallel to the axis, h for one normal to it)."""
    c = abs(float(np.asarray(exit_normal) @ np.asarray(axis)))
    s = math.sqrt(max(0.0, 1.0 - c * c))
    return ap_m * s + h_m * c


__all__ = ["BurrModel", "BurrResult", "hocheng_dharan_N", "critical_thrust", "exit_extent", "KIND_ROLLOVER",
           "KIND_BREAKOUT", "KIND_UNIFORM", "KIND_CROWN", "KIND_PULLOUT", "KIND_DELAMINATION"]
