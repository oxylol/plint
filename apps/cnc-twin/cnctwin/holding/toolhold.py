"""Tool-holder clamping: transmissible torque and axial retention per holder type (docs/models/holding.md
section 2).

* **ER collet** (ISO 15488, 8 deg cone): nut torque T_n (recommended REGO-FIX torque for the collet size
  and shank diameter) -> axial force on the collet through the nut thread and the nut-collet collar
  (power screw with collar friction, Budynas & Nisbett 2015 eq. 8-5/8-6):
  ``T_n = F_a [r_2 tan(lambda + rho') + mu_c r_c]``, ``tan rho' = mu_t sec 30 deg``,
  ``tan lambda = P / (pi d_2)``; the collet wedged into the cone: ``N = F_a / tan(alpha + rho_k)``
  (total radial force on the shank, rho_k = atan mu_k); friction at the shank:
  ``T_cap = mu_s N d/2``, ``F_ret = mu_s N``.
* **Hydraulic and shrink-fit chucks, milling chucks** (friction fits): contact pressure p over the clamped
  length L: from the catalogue torque (``p = 2 T_cat / (mu_s pi d^2 L)``), else from the interference
  (Lame, Budynas & Nisbett eq. 3-56, solid shank in a thick hub of outer diameter D):
  ``p = delta_d / (d [(D^2 + d^2) / (D^2 - d^2) + nu_h] / E_h + d (1 - nu_s) / E_s)``, or for hydraulic
  bores the default pressure; ``N = p pi d L``. Shrink fits and milling chucks lose interference when the
  holder runs warmer than the shank: ``delta_th = d (alpha_h dT_h - alpha_s dT_s)``,
  ``p = p0 max(0, 1 - delta_th / delta_0)``.
* **Side-lock (Weldon)**: positive torque drive; axial retention by the set screw's friction on the flat
  and the opposite bore wall, ``F_ret = 2 mu_s F_screw`` with ``F_screw = sigma_pre A_s``; the shank may
  walk out up to the play of the flat (``side_lock_walk_limit_m``), then the screw bears positively.
* **Tapping holders** (floating / synchro): positive square drive and ball-lock retention; the axial float
  is the cutting domain's (``CuttingContact.float_u_max_m``); holding checks it against the float travel.
* **Positive interfaces** (modular, arbor, boring head) and **non-rotating holders** (turret, lathe
  holder): no frictional slip mode (their failure is structural).

Under combined torque and axial load the friction is a circle: ``(T / T_cap)^2 + (F / F_ret)^2 <= 1``
(isotropic Coulomb friction of the shear vector at the bore).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from . import params as P

FRICTION_TYPES = ("er_collet", "hydraulic", "shrink_fit", "milling_chuck")
POSITIVE_TYPES = ("modular", "arbor", "boring_head")
TAPPING_TYPES = ("tapping_floating", "tapping_synchro")
NON_ROTATING_TYPES = ("turret_vdi", "turret_bmt", "lathe_holder")
INTERFERENCE_TYPES = ("shrink_fit", "milling_chuck")


@dataclass
class ToolData:
    """What the holding models need of one tool slot (built from cnctwin.tooling's ToolGeometry, or by hand)."""

    slot: int
    number: int
    tool_id: str = ""
    holder_id: str = ""
    rotating: bool = True
    holder_type: str = ""
    shank_d_m: float | None = None
    clamp_length_m: float | None = None
    bore_max_m: float | None = None
    nose_d_m: float | None = None
    collet_size: str = ""
    clamping_torque_Nm: float | None = None
    axial_retention_N: float | None = None
    float_travel_m: float = 0.0
    float_stiffness_N_per_m: float = 0.0
    tool_mass_kg: float = 0.0
    tool_com_z_m: float = 0.0
    tool_I_axial_kgm2: float = 0.0
    tool_I_trans_kgm2: float = 0.0
    holder_mass_kg: float = 0.0
    holder_com_z_m: float = 0.0
    tool_grade: tuple | None = None          # (G m/s, speed rad/s or None)
    holder_grade: tuple | None = None
    max_rad_s_tool: float | None = None
    max_rad_s_holder: float | None = None
    runout_T_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tilt_rad: np.ndarray = field(default_factory=lambda: np.zeros(2))
    flutes: int = 1
    tool_class: str = "carbide"
    shank_E_Pa: float | None = None
    shank_poisson: float | None = None
    shank_expansion_1_K: float | None = None
    hand_sign: float = -1.0
    stickout_m: float | None = None


@dataclass
class ToolClamp:
    """The holding of one tool slot: static capacities and what the slip integration needs."""

    slot: int
    number: int
    holder_type: str
    model: str                               # friction | side_lock | tapping | positive | non_rotating | none
    radius_m: float = 0.0                    # shank radius (lever of the circumferential friction)
    torque_cap_Nm: float = math.inf          # static transmissible torque (inf: positive drive)
    retention_N: float = math.inf            # static axial retention (inf: positive)
    normal_force_N: float = 0.0              # total normal force on the shank (friction fits)
    pressure_Pa: float = 0.0
    clamp_length_m: float = 0.0
    friction: float = 0.0
    kinetic_ratio: float = 1.0
    walk_limit_m: float = math.inf
    back_stop: bool = True
    tool_mass_kg: float = 0.0
    tool_I_axial_kgm2: float = 0.0
    float_travel_m: float = 0.0
    thermal: dict = field(default_factory=dict)   # interference fits: {d, delta0, alpha_h, alpha_s}
    source: str = ""
    notes: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def positive_torque(self) -> bool:
        return not math.isfinite(self.torque_cap_Nm)

    @property
    def slips(self) -> bool:
        """True when the slip and walk integration applies."""
        return self.model in ("friction", "side_lock") and self.radius_m > 0.0 and self.tool_mass_kg > 0.0

    def to_dict(self) -> dict:
        def f(x):
            return float(x) if math.isfinite(float(x)) else None
        return {"slot": self.slot, "number": self.number, "holder_type": self.holder_type, "model": self.model,
                "radius_m": self.radius_m, "torque_cap_Nm": f(self.torque_cap_Nm), "retention_N": f(self.retention_N),
                "normal_force_N": self.normal_force_N, "pressure_Pa": self.pressure_Pa,
                "clamp_length_m": self.clamp_length_m, "friction": self.friction, "kinetic_ratio": self.kinetic_ratio,
                "walk_limit_m": f(self.walk_limit_m), "back_stop": self.back_stop, "tool_mass_kg": self.tool_mass_kg,
                "tool_I_axial_kgm2": self.tool_I_axial_kgm2, "float_travel_m": self.float_travel_m,
                "thermal": dict(self.thermal), "source": self.source, "notes": list(self.notes),
                "detail": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in self.detail.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ToolClamp":
        def g(x):
            return math.inf if x is None else float(x)
        return cls(slot=int(d["slot"]), number=int(d["number"]), holder_type=str(d["holder_type"]), model=str(d["model"]),
                   radius_m=float(d["radius_m"]), torque_cap_Nm=g(d["torque_cap_Nm"]), retention_N=g(d["retention_N"]),
                   normal_force_N=float(d["normal_force_N"]), pressure_Pa=float(d["pressure_Pa"]),
                   clamp_length_m=float(d["clamp_length_m"]), friction=float(d["friction"]),
                   kinetic_ratio=float(d["kinetic_ratio"]), walk_limit_m=g(d["walk_limit_m"]),
                   back_stop=bool(d["back_stop"]), tool_mass_kg=float(d["tool_mass_kg"]),
                   tool_I_axial_kgm2=float(d["tool_I_axial_kgm2"]), float_travel_m=float(d["float_travel_m"]),
                   thermal=dict(d["thermal"]), source=str(d["source"]), notes=list(d["notes"]),
                   detail=dict(d["detail"]))


# --------------------------------------------------------------------------- ER collet chain
def er_index(collet_size: str, bore_max_m: float | None, shank_d_m: float) -> int:
    """Index into the er_* tables: the named size, else the smallest size whose range holds the bore."""
    sizes = [int(round(s)) for s in P.plist(P.TOOL, "er_sizes")]
    name = (collet_size or "").upper().replace(" ", "").replace("-", "")
    if name.startswith("ER"):
        digits = "".join(ch for ch in name[2:] if ch.isdigit())
        if digits and int(digits) in sizes:
            return sizes.index(int(digits))
    max_bore = P.plist(P.TOOL, "er_max_bore_m")
    need = max(float(bore_max_m or 0.0), float(shank_d_m))
    for i, bore in enumerate(max_bore):
        if need <= bore * (1.0 + 1e-9):
            return i
    raise InputError(f"ER collet: no size in the table holds a {need * 1e3:.2f} mm shank (largest "
                     f"{max_bore[-1] * 1e3:.0f} mm)")


def er_nut_torque_Nm(index: int, shank_d_m: float) -> float:
    """Recommended nut torque (REGO-FIX table) of the row covering the shank diameter (nearest row)."""
    size = int(round(P.plist(P.TOOL, "er_sizes")[index]))
    rows_size = np.round(P.plist(P.TOOL, "er_torque_size")).astype(int)
    lo = P.plist(P.TOOL, "er_torque_d_lo_m")
    hi = P.plist(P.TOOL, "er_torque_d_hi_m")
    torque = P.plist(P.TOOL, "er_torque_Nm")
    best, best_dist = None, math.inf
    for i in np.flatnonzero(rows_size == size):
        dist = 0.0 if lo[i] <= shank_d_m <= hi[i] else min(abs(shank_d_m - lo[i]), abs(shank_d_m - hi[i]))
        if dist < best_dist - 1e-15:
            best, best_dist = int(i), dist
    if best is None:
        raise InputError(f"ER {size}: no recommended nut torque in db/models/holding-tool.yaml")
    return float(torque[best])


def er_collet_clamp(shank_d_m: float, nut_torque_Nm: float, index: int) -> dict:
    """The ER chain: axial nut force, total radial force, torque capacity and axial retention."""
    d_thread = float(P.plist(P.TOOL, "er_thread_d_m")[index])
    pitch = float(P.plist(P.TOOL, "er_thread_pitch_m")[index])
    front = float(P.plist(P.TOOL, "er_front_d_m")[index])
    d2 = d_thread - P.p(P.TOOL, "pitch_diameter_ratio") * pitch
    lead = math.atan(pitch / (math.pi * d2))
    rho_t = math.atan(P.p(P.TOOL, "er_thread_friction") / math.cos(P.p(P.TOOL, "thread_flank_half_angle_rad")))
    r_collar = P.p(P.TOOL, "er_collar_radius_ratio") * front
    lever = 0.5 * d2 * math.tan(lead + rho_t) + P.p(P.TOOL, "er_collar_friction") * r_collar
    axial = nut_torque_Nm / lever
    alpha = P.p(P.TOOL, "er_cone_half_angle_rad")
    rho_k = math.atan(P.p(P.TOOL, "er_cone_friction"))
    radial = axial / math.tan(alpha + rho_k)
    mu_s = P.p(P.TOOL, "shank_friction")
    return {"nut_torque_Nm": nut_torque_Nm, "axial_force_N": axial, "radial_force_N": radial,
            "torque_cap_Nm": mu_s * radial * 0.5 * shank_d_m, "retention_N": mu_s * radial,
            "thread_pitch_d_m": d2, "lead_angle_rad": lead, "collar_radius_m": r_collar, "lever_m": lever,
            "friction": mu_s, "collet_length_m": float(P.plist(P.TOOL, "er_length_m")[index]),
            "size": int(round(P.plist(P.TOOL, "er_sizes")[index]))}


# --------------------------------------------------------------------------- interference fits
def lame_compliance_m_per_Pa(d_m: float, outer_m: float, E_h: float, nu_h: float, E_s: float, nu_s: float) -> float:
    """Diametral interference per unit contact pressure of a solid shaft in a hub of outer diameter D."""
    if outer_m <= d_m:
        raise InputError(f"interference fit: hub outer diameter {outer_m * 1e3:.2f} mm is not above the bore "
                         f"{d_m * 1e3:.2f} mm")
    hub = (outer_m ** 2 + d_m ** 2) / (outer_m ** 2 - d_m ** 2) + nu_h
    return d_m * hub / E_h + d_m * (1.0 - nu_s) / E_s


def friction_capacity(pressure_Pa: float, mu: float, d_m: float, length_m: float) -> tuple[float, float, float]:
    """(T_cap, F_ret, N) of a cylindrical friction fit: N = p pi d L, T = mu N d/2, F = mu N."""
    normal = pressure_Pa * math.pi * d_m * length_m
    return mu * normal * 0.5 * d_m, mu * normal, normal


def shank_elastic(td: ToolData) -> tuple[float, float, float]:
    """(E, nu, alpha) of the shank: the tool material's, else the class defaults."""
    cls = "hss" if td.tool_class == "hss" else "carbide"
    E = td.shank_E_Pa if td.shank_E_Pa else P.p(P.TOOL, f"{cls}_E_Pa")
    nu = td.shank_poisson if td.shank_poisson is not None else P.p(P.TOOL, f"{cls}_poisson")
    alpha = td.shank_expansion_1_K if td.shank_expansion_1_K else P.p(P.TOOL, f"{cls}_expansion_1_K")
    return float(E), float(nu), float(alpha)


def pressure_factor(clamp: ToolClamp, holder_temp_K: float | None, tool_temp_K: float | None,
                    reference_K: float) -> float:
    """Share of the clamping pressure left when the holder and shank temperatures differ from the
    clamping temperature (interference fits only; 1 otherwise)."""
    th = clamp.thermal
    if not th or holder_temp_K is None:
        return 1.0
    t_tool = tool_temp_K if tool_temp_K is not None else reference_K
    loss = th["d"] * (th["alpha_h"] * (holder_temp_K - reference_K) - th["alpha_s"] * (t_tool - reference_K))
    return max(0.0, 1.0 - loss / th["delta0"]) if th["delta0"] > 0.0 else 1.0


# --------------------------------------------------------------------------- the clamp of one slot
def build_clamp(td: ToolData) -> ToolClamp:
    """ToolClamp of one slot from its holder and tool data (see the module doc)."""
    kind = td.holder_type or ""
    clamp = ToolClamp(slot=td.slot, number=td.number, holder_type=kind, model="none",
                      tool_mass_kg=float(td.tool_mass_kg), tool_I_axial_kgm2=float(td.tool_I_axial_kgm2),
                      kinetic_ratio=P.p(P.TOOL, "kinetic_to_static_friction"))
    if not td.rotating or kind in NON_ROTATING_TYPES:
        clamp.model = "non_rotating"
        clamp.source = "no frictional slip mode for non-rotating tool holders"
        return clamp
    if not kind:
        clamp.model = "none"
        clamp.notes.append(f"T{td.number}: no holder: the tool is taken as held rigidly (no slip or pull-out check)")
        return clamp
    if kind in POSITIVE_TYPES:
        clamp.model = "positive"
        clamp.source = f"{kind}: positive (form-fit) interface"
        return clamp
    if kind in TAPPING_TYPES:
        clamp.model = "tapping"
        clamp.float_travel_m = float(td.float_travel_m)
        clamp.source = f"{kind}: square drive and ball lock; axial float {td.float_travel_m * 1e3:.2f} mm"
        if td.float_travel_m <= 0.0:
            clamp.notes.append(f"T{td.number}: the {kind} holder gives no axial float travel: the float is not checked")
        return clamp
    d = td.shank_d_m
    if d is None or d <= 0.0:
        raise InputError(f"T{td.number}: the {kind} holder needs the tool's shank diameter")
    clamp.radius_m = 0.5 * d
    mu = P.p(P.TOOL, "shank_friction")
    clamp.friction = mu
    if kind == "side_lock":
        ratio = P.p(P.TOOL, "side_lock_screw_ratio")
        d_screw = ratio * d
        area = P.p(P.TOOL, "side_lock_stress_area_ratio") * math.pi * d_screw ** 2 / 4.0
        preload = P.p(P.TOOL, "side_lock_preload_stress_Pa") * area
        clamp.model = "side_lock"
        clamp.retention_N = 2.0 * mu * preload if td.axial_retention_N is None else float(td.axial_retention_N)
        clamp.walk_limit_m = P.p(P.TOOL, "side_lock_walk_limit_m")
        clamp.source = "side-lock: positive drive, set-screw friction retention" if td.axial_retention_N is None \
            else "side-lock: positive drive, catalogue axial retention"
        clamp.detail = {"screw_d_m": d_screw, "screw_preload_N": preload}
        clamp.notes.append(f"T{td.number}: side-lock set screw M{d_screw * 1e3:.0f} preload {preload / 1e3:.1f} kN "
                           f"(estimated from the shank size)")
        return clamp
    if kind not in FRICTION_TYPES:
        raise InputError(f"T{td.number}: holder type {kind!r} has no holding model")
    clamp.model = "friction"
    length = td.clamp_length_m
    if kind == "er_collet":
        index = er_index(td.collet_size, td.bore_max_m, d)
        torque_n = er_nut_torque_Nm(index, d)
        er = er_collet_clamp(d, torque_n, index)
        collet_len = er["collet_length_m"]
        clamp.clamp_length_m = float(min(length, collet_len)) if length else collet_len
        clamp.normal_force_N = er["radial_force_N"]
        clamp.pressure_Pa = er["radial_force_N"] / (math.pi * d * clamp.clamp_length_m)
        clamp.torque_cap_Nm = er["torque_cap_Nm"]
        clamp.retention_N = er["retention_N"]
        clamp.detail = {k: v for k, v in er.items() if isinstance(v, (int, float))}
        clamp.source = f"ER {er['size']} collet chain at the recommended nut torque {torque_n:.0f} Nm"
        clamp.notes.append(f"T{td.number}: ER {er['size']} nut torque {torque_n:.0f} Nm (REGO-FIX recommendation; "
                           f"the project gives no nut torque)")
        if td.clamping_torque_Nm is not None:
            scale = float(td.clamping_torque_Nm) / clamp.torque_cap_Nm
            clamp.torque_cap_Nm = float(td.clamping_torque_Nm)
            clamp.normal_force_N *= scale
            clamp.pressure_Pa *= scale
            clamp.retention_N = mu * clamp.normal_force_N
            clamp.source = "catalogue clamping torque (ER chain for the pressure distribution)"
        if td.axial_retention_N is not None:
            clamp.retention_N = float(td.axial_retention_N)
        return clamp
    if length is None or length <= 0.0:
        length = P.p(P.TOOL, "clamp_length_ratio") * d
        clamp.notes.append(f"T{td.number}: clamped length {length * 1e3:.1f} mm estimated (clamp_length_ratio x d)")
    clamp.clamp_length_m = float(length)
    E_h, nu_h = P.p(P.TOOL, "holder_E_Pa"), P.p(P.TOOL, "holder_poisson")
    E_s, nu_s, alpha_s = shank_elastic(td)
    outer = td.nose_d_m if td.nose_d_m and td.nose_d_m > d else P.p(P.TOOL, "nose_ratio") * d
    if td.clamping_torque_Nm is not None:
        torque = float(td.clamping_torque_Nm)
        normal = torque / (mu * 0.5 * d)
        pressure = normal / (math.pi * d * length)
        clamp.source = "catalogue clamping torque"
    elif kind == "hydraulic":
        pressure = P.p(P.TOOL, "hydraulic_pressure_Pa")
        torque, _, normal = friction_capacity(pressure, mu, d, length)
        clamp.source = "hydraulic bore at the default contact pressure"
        clamp.notes.append(f"T{td.number}: hydraulic chuck without a catalogue torque: contact pressure "
                           f"{pressure / 1e6:.0f} MPa estimated")
    else:
        ratio = P.p(P.TOOL, "shrink_interference_ratio" if kind == "shrink_fit" else "milling_chuck_interference_ratio")
        delta0 = ratio * d
        pressure = delta0 / lame_compliance_m_per_Pa(d, outer, E_h, nu_h, E_s, nu_s)
        torque, _, normal = friction_capacity(pressure, mu, d, length)
        clamp.source = f"{kind}: Lame pressure of the estimated interference"
        clamp.notes.append(f"T{td.number}: {kind} without a catalogue torque: interference "
                           f"{delta0 * 1e6:.1f} um estimated (Lame pressure {pressure / 1e6:.0f} MPa)")
    clamp.pressure_Pa = float(pressure)
    clamp.normal_force_N = float(normal)
    clamp.torque_cap_Nm = float(torque)
    clamp.retention_N = float(td.axial_retention_N) if td.axial_retention_N is not None else mu * normal
    if kind in INTERFERENCE_TYPES:
        compliance = lame_compliance_m_per_Pa(d, outer, E_h, nu_h, E_s, nu_s)
        clamp.thermal = {"d": float(d), "delta0": float(pressure * compliance),
                         "alpha_h": P.p(P.TOOL, "holder_expansion_1_K"), "alpha_s": float(alpha_s),
                         "outer_m": float(outer)}
    return clamp


# --------------------------------------------------------------------------- from cnctwin.tooling
def tool_data(geometry, db=None) -> ToolData:
    """ToolData of a cnctwin.tooling ToolGeometry (its props, holder spec, instance and edge)."""
    props = geometry.props
    holder = getattr(geometry, "holder", None)
    spec = getattr(geometry, "spec", None)
    layout = getattr(geometry, "layout", {}) or {}
    raw = getattr(holder, "raw", None) if holder is not None else None
    td = ToolData(slot=int(geometry.slot), number=int(geometry.number), tool_id=str(geometry.tool_id),
                  holder_id=str(geometry.holder_id), rotating=bool(geometry.rotating),
                  holder_type=str(getattr(holder, "type", "") or "") if holder is not None else "",
                  hand_sign=float(getattr(geometry, "hand_sign", -1.0)), flutes=int(max(props.flutes, 1)))
    d_shank = layout.get("d_shank")
    if d_shank is None and spec is not None:
        d_shank = P.num(P.field_of(spec.geometry, "shank"), "diameter_m") or P.num(spec.geometry, "diameter_m")
    td.shank_d_m = float(d_shank) if d_shank else None
    if holder is not None:
        td.clamp_length_m = holder.clamp_length_m
        td.nose_d_m = holder.nose_diameter_m
        td.clamping_torque_Nm = holder.clamping_torque_Nm
        bores = P.numlist(raw, "bore_m") if raw is not None else None
        td.bore_max_m = max(bores) if bores else holder.bore_m
        td.collet_size = P.text(raw, "collet_size", "") if raw is not None else ""
        td.axial_retention_N = P.num(raw, "axial_retention_N") if raw is not None else None
        travel = P.numlist(raw, "axial_float_m") if raw is not None else None
        if travel:
            td.float_travel_m = float(min(abs(v) for v in travel)) if len(travel) > 1 else abs(float(travel[0]))
        td.float_stiffness_N_per_m = float(P.num(raw, "axial_stiffness_N_per_m", 0.0) or 0.0) if raw is not None \
            else 0.0
    tm = props.tool_mass
    if tm is not None:
        td.tool_mass_kg, td.tool_com_z_m = float(tm.mass_kg), float(tm.com_z_m)
        td.tool_I_axial_kgm2, td.tool_I_trans_kgm2 = float(tm.I_axial_kgm2), float(tm.I_trans_kgm2)
    hm = props.holder_mass
    if hm is not None:
        td.holder_mass_kg, td.holder_com_z_m = float(hm.mass_kg), float(hm.com_z_m)
    if props.balance_tool is not None:
        td.tool_grade = (float(props.balance_tool.grade_m_s), float(props.balance_tool.speed_rad_s))
    if props.balance_holder is not None:
        td.holder_grade = (float(props.balance_holder.grade_m_s), float(props.balance_holder.speed_rad_s))
    td.max_rad_s_tool = props.max_rad_s_tool
    td.max_rad_s_holder = props.max_rad_s_holder
    edge = geometry.edge
    td.runout_T_m = np.array(edge.runout_T_m, dtype=float).reshape(3)
    td.tilt_rad = np.array(edge.tilt_rad, dtype=float).reshape(2)
    td.stickout_m = props.stickout_m
    if spec is not None:
        td.tool_class = str(spec.tool_material_class or "carbide")
        _shank_material(td, spec, db)
    return td


def _shank_material(td: ToolData, spec, db) -> None:
    if not spec.tool_material_id or db is None or not hasattr(db, "get"):
        return
    try:
        item = db.get("tool_material", spec.tool_material_id)
    except Exception:  # noqa: BLE001 - a tool material missing from the database falls back to the class values
        return
    td.shank_E_Pa = P.num(item, "youngs_modulus_Pa")
    td.shank_poisson = P.num(item, "poisson")
    td.shank_expansion_1_K = P.num(item, "expansion_1_K")


__all__ = ["ToolData", "ToolClamp", "build_clamp", "tool_data", "er_index", "er_nut_torque_Nm", "er_collet_clamp",
           "lame_compliance_m_per_Pa", "friction_capacity", "pressure_factor", "shank_elastic", "FRICTION_TYPES",
           "POSITIVE_TYPES", "TAPPING_TYPES", "NON_ROTATING_TYPES"]
