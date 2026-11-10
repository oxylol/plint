"""Surface integrity of the machined layer: residual stress, hardness, white layer and burn
(docs/models/surface.md section 6).

Mechanical load (the edge-radius and flank-wear contact that ploughs the new surface, a sliding line contact):
the normal force per unit edge length P_n over the contact length l_c = VB + r_beta gives the peak pressure
p0 = (4 / pi) P_n / l_c (Hertzian line-contact shape). With the shear yield stress k = sigma_y / sqrt(3)
the layer yields beyond p_y = 3.1 k (first subsurface yield, Johnson 1985) and saturates at the shakedown
limit p_s = min(4 k, k / mu) (Merwin & Johnson 1963; Johnson 1985 ch. 9): the compressive residual stress
in the cutting direction is

    sigma_mech = -sigma_y * clip((p0 - p_y) / (p_s - p_y), 0, 1),  depth d_m = C_m * l_c / 2.

Thermal load (the two-scale peak temperature of the generated surface from thermal): a surface layer
heated by dT under biaxial constraint carries -E alpha dT / (1 - nu) while hot (Timoshenko & Goodier 1970);
the part beyond the yield stress is relieved plastically, so on cooling it is left in tension

    sigma_th = min(sigma_y, max(0, E alpha dT / (1 - nu) - sigma_y)),  depth d_th = sqrt(kappa l_c / v)

(the heat-penetration depth of a surface point under the moving contact for its contact time l_c / v;
Carslaw & Jaeger 1959). sigma(z) = sigma_mech exp(-z / d_m) + sigma_th exp(-z / d_th), the surface value
clipped to +-sigma_y.

Hardness: the material file's work_hardening (layer depth and hardness increase at the reference edge radius
of db/models/surface-integrity.yaml) scaled by the load p0 / p_s (at most 1) and the contact length; a
rehardened (white) layer above white_layer_degC adds white_layer_hardness_ratio, an over-tempered layer above
overtemper_degC loses overtemper_softening_ratio; burn and melt/smear thresholds set flags.

Error band: residual stress +-150 MPa or +-50 % (the larger) against X-ray measurements of turned and milled
steels; affected depth within a factor of 2; hardness change +-50 % of the change.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Integrity:
    p0_Pa: float = 0.0
    sigma_mech_Pa: float = 0.0
    sigma_thermal_Pa: float = 0.0
    sigma_surface_Pa: float = 0.0
    depth_mech_m: float = 0.0
    depth_thermal_m: float = 0.0
    affected_depth_m: float = 0.0
    hardness_change_frac: float = 0.0
    hardened_depth_m: float = 0.0
    white_layer: bool = False
    burn: bool = False
    overtemper: bool = False
    melt_smear: bool = False
    surface_temp_K: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def stress_at(self, depth_m) -> np.ndarray:
        z = np.asarray(depth_m, dtype=float)
        m = self.sigma_mech_Pa * np.exp(-z / self.depth_mech_m) if self.depth_mech_m > 0.0 else 0.0 * z
        t = self.sigma_thermal_Pa * np.exp(-z / self.depth_thermal_m) if self.depth_thermal_m > 0.0 else 0.0 * z
        return m + t


def evaluate(force_n_per_m: float, VB_m: float, edge_r_m: float, speed_m_s: float, surface_K: float,
             ambient_K: float, mat, params: dict, mu: float | None = None, damage: dict | None = None) -> Integrity:
    """Integrity of a layer generated under the given loads (see the module doc). ``damage``: thermal's judgement
    of the machined surface's temperature history ({kind: affected depth m}, docs/models/thermal.md 3.4); when given
    it sets the white layer / over-temper / burn / melt flags (and the white layer's depth) instead of the threshold
    crossings of ``surface_K``."""
    out = Integrity(surface_temp_K=float(surface_K))
    sigma_y = max(mat.yield_Pa, 1.0)
    k = sigma_y / math.sqrt(3.0)
    l_c = max(VB_m, 0.0) + max(edge_r_m, 0.0)
    if mu is None:
        mu = mat.mu_sliding if mat.mu_sliding is not None else float(params["friction_default"])
    if l_c > 0.0 and force_n_per_m > 0.0:
        p0 = (4.0 / math.pi) * force_n_per_m / l_c
        p_y = float(params["yield_onset_ratio"]) * k
        p_s = min(float(params["shakedown_ratio"]) * k, k / max(mu, 1e-6))
        p_s = max(p_s, p_y * (1.0 + 1e-9))
        frac = min(max((p0 - p_y) / (p_s - p_y), 0.0), 1.0)
        out.p0_Pa = p0
        out.sigma_mech_Pa = -sigma_y * frac
        out.depth_mech_m = float(params["mech_depth_ratio"]) * 0.5 * l_c if frac > 0.0 else 0.0
        if mat.wh_depth_m > 0.0 and mat.wh_increase_frac > 0.0:
            ref = float(params["wh_reference_edge_radius_m"])
            load = min(p0 / p_s, 1.0)
            out.hardness_change_frac = mat.wh_increase_frac * load
            out.hardened_depth_m = mat.wh_depth_m * (l_c / ref) * load if ref > 0.0 else mat.wh_depth_m * load
    dT = float(surface_K) - float(ambient_K)
    if dT > 0.0:
        sigma_heat = mat.youngs_Pa * mat.expansion_1_K * dT / (1.0 - mat.poisson)
        out.sigma_thermal_Pa = min(sigma_y, max(0.0, sigma_heat - sigma_y))
        kappa = mat.conductivity_W_mK / (mat.density_kg_m3 * mat.specific_heat_J_kgK)
        if speed_m_s > 0.0 and l_c > 0.0:
            out.depth_thermal_m = math.sqrt(kappa * l_c / speed_m_s)
    out.sigma_surface_Pa = min(max(out.sigma_mech_Pa + out.sigma_thermal_Pa, -sigma_y), sigma_y)
    out.affected_depth_m = max(out.depth_mech_m if out.sigma_mech_Pa != 0.0 else 0.0,
                               out.depth_thermal_m if out.sigma_thermal_Pa != 0.0 else 0.0) * math.log(10.0)
    if damage is not None:
        white = float(damage.get("white_layer", 0.0) or 0.0)
        is_white = white > 0.0
        flags = {kind: float(damage.get(kind, 0.0) or 0.0) > 0.0 for kind in ("overtemper", "burn", "melt_smear")}
    else:
        is_white = mat.white_layer_K is not None and surface_K > mat.white_layer_K
        white = out.depth_thermal_m
        flags = {"overtemper": mat.overtemper_K is not None and surface_K > mat.overtemper_K,
                 "burn": mat.burn_K is not None and surface_K > mat.burn_K,
                 "melt_smear": mat.melt_smear_K is not None and surface_K > mat.melt_smear_K}
    if is_white:
        out.white_layer = True
        out.hardness_change_frac += float(params["white_layer_hardness_ratio"])
        out.hardened_depth_m = max(out.hardened_depth_m, white)
    elif flags["overtemper"]:
        out.overtemper = True
        out.hardness_change_frac -= float(params["overtemper_softening_ratio"])
    out.burn = bool(flags["burn"])
    out.melt_smear = bool(flags["melt_smear"])
    return out


__all__ = ["Integrity", "evaluate"]
