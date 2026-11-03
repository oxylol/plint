"""Tool properties other domains read (docs/models/tooling.md "Properties").

* chip space: gullet area per flute (given, or the flute void fraction of the annulus between
  core and D), gullet volume per unit length (all flutes) and chip space of the flute length;
* fluted-section second moments: N teeth as annular sectors of angular width 2 pi f_s / N at their
  phases on a solid core (Young & Budynas 2012, App. A), principal values I1 >= I2 and the
  equivalent diameter D_eq = (64 I2 / pi)^(1/4); without the tool's core and gullet data the
  equivalent diameter of Kops & Vo (1990), 0.8 D, with the sector model's anisotropy;
* critical sections: flute run-out, neck end, shank at the holder nose (section moduli of the
  equivalent circular section, pi d^3 / 32 and pi d^3 / 16); turning shanks and boring bars;
* mass, centre of mass and inertia of the tool (revolved envelope x fill factor x density, scaled
  to the catalogue mass when it is given), of the holder (revolved profile minus the bore, scaled to
  its catalogue mass) and of both together;
* balance (ISO 21940-11: e_per = G / Omega, U_per = m e_per), maximum speed of tool and holder,
  coolant holes (count, diameter, exits in T), hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from ..paths import DB
from . import params as P
from .abi import CHAIN_HOLDER, CHAIN_TOOL, PART_BODY, Envelope
from .spec import HolderSpec, InstanceSpec, ToolSpec, num, parse_balance_grade, plain, sub, text

Z_SAMPLES = 2000     # slices for the mass integrals (relative error of the COM < 1e-6 of the length)


@dataclass
class MassProps:
    mass_kg: float
    com_z_m: float                 # along +z_T from the tip
    I_axial_kgm2: float            # about the tool axis
    I_trans_kgm2: float            # about a transverse axis through the centre of mass
    source: str                    # "catalogue" (scaled to the stated mass) or "geometry"


@dataclass
class Section:
    name: str
    z_m: float                     # from the tip along +z_T
    Z_bending_m3: float            # section modulus (weakest direction)
    Z_polar_m3: float
    I_min_m4: float
    I_max_m4: float
    area_m2: float
    d_eq_m: float


@dataclass
class Balance:
    grade_m_s: float               # G (e_per * Omega)
    speed_rad_s: float             # speed the grade refers to
    e_per_m: float                 # permissible specific unbalance
    U_per_kg_m: float              # permissible residual unbalance of the body


@dataclass
class CoolantHoles:
    count: int
    diameter_m: float
    area_m2: float
    exit: str
    exit_pos_T_m: np.ndarray       # [count, 3] at spindle angle 0
    exit_dir_T: np.ndarray         # [count, 3] unit


@dataclass
class ToolProps:
    hand: str
    hand_sign: float
    rotating: bool
    diameter_m: float
    flutes: int
    core_d_m: float
    gullet_area_per_flute_m2: float
    gullet_volume_per_length_m3_per_m: float
    flute_length_m: float
    chip_space_m3: float
    fluted: Section | None
    sections: list[Section]
    tool_mass: MassProps | None
    holder_mass: MassProps | None
    assembly_mass: MassProps | None
    max_rad_s: float | None
    max_rad_s_tool: float | None
    max_rad_s_holder: float | None
    balance_tool: Balance | None
    balance_holder: Balance | None
    coolant: CoolantHoles | None
    stickout_m: float | None
    gauge_to_tip_m: float | None
    shank_dir_T: tuple[float, float, float]
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- envelope radius (Python)
def envelope_radius(env: Envelope, z, chain: int = -1, part_max: int | None = None) -> np.ndarray:
    """Radius of the envelope (union of chains, or one chain) at heights z (0 outside)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.zeros_like(z)
    for i in range(len(env.kind)):
        if chain >= 0 and env.chain[i] != chain:
            continue
        if part_max is not None and env.part[i] > part_max:
            continue
        inside = (z >= env.z_lo_m[i] - 1e-15) & (z <= env.z_hi_m[i] + 1e-15)
        if not np.any(inside):
            continue
        zi = z[inside]
        if env.kind[i] == 1:
            cos_a = np.clip((env.zc_m[i] - zi) / env.rho_m[i], -1.0, 1.0)
            r = env.rc_m[i] + env.rho_m[i] * np.sqrt(1.0 - cos_a * cos_a)
        elif env.z1_m[i] == env.z0_m[i]:
            r = np.full_like(zi, max(env.r0_m[i], env.r1_m[i]))
        else:
            t = (zi - env.z0_m[i]) / (env.z1_m[i] - env.z0_m[i])
            r = env.r0_m[i] + t * (env.r1_m[i] - env.r0_m[i])
        out[inside] = np.maximum(out[inside], r)
    return out


# --------------------------------------------------------------------------- chip space and sections
def gullet(spec: ToolSpec, info: dict) -> tuple[float, float]:
    """(gullet area per flute, fluted cross-section area)."""
    R = info["R"]
    N = max(int(info.get("flutes", 1)), 1)
    r_c = 0.5 * info.get("core_d", 0.0)
    given = num(spec.geometry, "gullet_area_m2")
    if info.get("drill") and given is None:
        total = P.pf("drill_flute_area_fraction") * math.pi * R * R
        return total / N, math.pi * R * R - total
    if given is not None:
        per = given
    else:
        per = P.pf("flute_void_fraction") * math.pi * (R * R - r_c * r_c) / N
    area = math.pi * R * R - N * per
    if area <= 0.0:
        raise InputError(f"tool {spec.tool_id}: gullet area {per * 1e6:.2f} mm2 x {N} exceeds the cross-section")
    return per, area


def sector_moments(R: float, r_core: float, solid_fraction: float, phases) -> tuple[float, float, float]:
    """(I_xx, I_yy, I_xy) of a solid core of radius r_core plus len(phases) annular sectors (r_core..R)
    of angular width 2 pi f_s / N centred at the given phases."""
    phases = list(phases)
    N = max(len(phases), 1)
    alpha = 2.0 * math.pi * solid_fraction / N
    ring = (R ** 4 - r_core ** 4) / 4.0
    core = math.pi * r_core ** 4 / 4.0
    ixx = iyy = core
    ixy = 0.0
    for phi in phases:
        ixx += ring * (0.5 * alpha - 0.5 * math.cos(2.0 * phi) * math.sin(alpha))
        iyy += ring * (0.5 * alpha + 0.5 * math.cos(2.0 * phi) * math.sin(alpha))
        ixy += ring * 0.5 * math.sin(2.0 * phi) * math.sin(alpha)
    return ixx, iyy, ixy


def principal(ixx: float, iyy: float, ixy: float) -> tuple[float, float]:
    mean = 0.5 * (ixx + iyy)
    half = math.hypot(0.5 * (ixx - iyy), ixy)
    return mean + half, mean - half


def fluted_section(spec: ToolSpec, info: dict, phases) -> Section | None:
    R = info.get("R")
    if R is None or not info.get("flutes"):
        return None
    D = 2.0 * R
    r_c = 0.5 * info.get("core_d", 0.0)
    per, area = gullet(spec, info)
    annulus = math.pi * (R * R - r_c * r_c)
    f_s = min(max((area - math.pi * r_c * r_c) / annulus, 0.0), 1.0) if annulus > 0 else 1.0
    I1, I2 = principal(*sector_moments(R, r_c, f_s, phases))
    have_data = num(spec.geometry, "core_diameter_m") is not None and num(spec.geometry, "gullet_area_m2") is not None
    if not have_data and not info.get("drill") and not info.get("indexable"):
        # Kops & Vo equivalent diameter for the mean, the sector model for the anisotropy
        mean = math.pi * (P.pf("kops_vo_ratio") * D) ** 4 / 64.0
        split = (I1 - I2) / (I1 + I2) if I1 + I2 > 0 else 0.0
        I1, I2 = mean * (1.0 + split), mean * (1.0 - split)
    d_eq = (64.0 * I2 / math.pi) ** 0.25
    z = info.get("cutting_length", 0.0)
    return Section("flute_runout", z, math.pi * d_eq ** 3 / 32.0, math.pi * d_eq ** 3 / 16.0, I2, I1, area, d_eq)


def _round_section(name: str, z: float, d: float) -> Section:
    I = math.pi * d ** 4 / 64.0
    return Section(name, z, math.pi * d ** 3 / 32.0, math.pi * d ** 3 / 16.0, I, I, math.pi * d * d / 4.0, d)


# --------------------------------------------------------------------------- masses
def tool_density(spec: ToolSpec, db=None) -> tuple[float, str]:
    """Density of the tool material: the item's tool_material id (database), else the class's
    representative grade (tool_material_by_class)."""
    candidates = []
    if spec.tool_material_id:
        candidates.append(spec.tool_material_id)
    by_class = P.p("tool_material_by_class")
    if spec.tool_material_class in by_class:
        candidates.append(by_class[spec.tool_material_class])
    for material_id in candidates:
        value = _material_density(material_id, db)
        if value is not None:
            return value, material_id
    raise InputError(f"tool {spec.tool_id}: no density for tool material {spec.tool_material_id!r} "
                     f"(class {spec.tool_material_class})")


def _material_density(material_id: str, db) -> float | None:
    if db is not None and hasattr(db, "get"):
        try:
            item = db.get("tool_material", material_id)
            value = num(item, "density_kg_m3")
            if value is not None:
                return value
        except Exception:  # noqa: BLE001 - a database without this item falls back to the file
            pass
    path = DB / "tool_materials" / f"{material_id}.yaml"
    if not path.is_file():
        return None
    from ..db.yamlio import load_file
    data = load_file(path).data
    value = plain(data.get("density_kg_m3"))
    if isinstance(value, dict):
        value = value.get("value")
    return float(value) if value is not None else None


def _revolved(z: np.ndarray, r: np.ndarray, fill: np.ndarray, density: float) -> tuple[float, float, float, float]:
    """(mass, com_z, I_axial, I_trans about the COM) of slices of radius r and fill factor."""
    dz = np.gradient(z)
    area = math.pi * r * r * fill
    dm = density * area * dz
    mass = float(dm.sum())
    if mass <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    com = float((dm * z).sum() / mass)
    i_ax = float((0.5 * dm * r * r).sum())
    i_tr = float((dm * (0.25 * r * r + (z - com) ** 2)).sum())
    return mass, com, i_ax, i_tr


def tool_mass(spec: ToolSpec, info: dict, env: Envelope, layout: dict, fluted_area: float, db=None) -> MassProps:
    overall = layout["overall"]
    z = np.linspace(0.0, overall, Z_SAMPLES)
    r = envelope_radius(env, z, chain=CHAIN_TOOL)
    fill = np.ones_like(z)
    z_flute = layout.get("z_flute_end", info.get("cutting_length", 0.0))
    R = info.get("R", float(r.max()))
    fluted_fill = fluted_area / (math.pi * R * R) if R > 0 else 1.0
    fill[z <= z_flute] = fluted_fill
    density, _ = tool_density(spec, db)
    mass, com, i_ax, i_tr = _revolved(z, r, fill, density)
    if spec.mass_kg is not None and mass > 0.0:
        scale = spec.mass_kg / mass
        return MassProps(spec.mass_kg, com, i_ax * scale, i_tr * scale, "catalogue")
    return MassProps(mass, com, i_ax, i_tr, "geometry")


def holder_mass(holder: HolderSpec | None, env: Envelope, layout: dict) -> MassProps | None:
    if holder is None or layout.get("z_gauge") is None:
        return None
    z0, z1 = layout["z_holder_nose"], layout["z_gauge"]
    z = np.linspace(z0, z1, Z_SAMPLES)
    r = envelope_radius(env, z, chain=CHAIN_HOLDER)
    bore = holder.bore_m or 0.0
    clamp = holder.clamp_length_m if holder.clamp_length_m is not None else (z1 - z0)
    fill = np.ones_like(z)
    inside = (z - z0) <= clamp
    fill[inside] = np.clip(1.0 - (0.5 * bore) ** 2 / np.maximum(r[inside], 1e-12) ** 2, 0.0, 1.0)
    density = P.pf("holder_density_kg_m3")
    mass, com, i_ax, i_tr = _revolved(z, r, fill, density)
    if holder.mass_kg is not None and mass > 0.0:
        scale = holder.mass_kg / mass
        return MassProps(holder.mass_kg, com, i_ax * scale, i_tr * scale, "catalogue")
    return MassProps(mass, com, i_ax, i_tr, "geometry")


def combine(a: MassProps | None, b: MassProps | None) -> MassProps | None:
    if a is None:
        return b
    if b is None:
        return a
    mass = a.mass_kg + b.mass_kg
    com = (a.mass_kg * a.com_z_m + b.mass_kg * b.com_z_m) / mass
    i_tr = (a.I_trans_kgm2 + a.mass_kg * (a.com_z_m - com) ** 2 + b.I_trans_kgm2 + b.mass_kg * (b.com_z_m - com) ** 2)
    source = "catalogue" if a.source == b.source == "catalogue" else "geometry"
    return MassProps(mass, com, a.I_axial_kgm2 + b.I_axial_kgm2, i_tr, source)


# --------------------------------------------------------------------------- balance, speed, coolant
def balance(grade, mass: MassProps | None, max_rad_s: float | None) -> Balance | None:
    parsed = parse_balance_grade(grade)
    if parsed is None or mass is None:
        return None
    g, speed = parsed
    if speed is None:
        speed = max_rad_s
    if speed is None or speed <= 0.0:
        return None
    e = g / speed
    return Balance(g, speed, e, mass.mass_kg * e)


def coolant_holes(spec: ToolSpec, info: dict, phases, hand_sign: float, env: Envelope | None) -> CoolantHoles | None:
    holes = sub(spec.geometry, "coolant_holes")
    if holes is None:
        return None
    count = num(holes, "count")
    diameter = num(holes, "diameter_m")
    if count is None or diameter is None:
        return None
    count = int(count)
    exit_kind = text(holes, "exit", "tip") or "tip"
    R = info.get("R", 0.0)
    pos = np.zeros((count, 3))
    direction = np.zeros((count, 3))
    N = max(len(phases), 1)
    ratio = P.pf("coolant_exit_radius_ratio")
    for i in range(count):
        if info.get("turning"):
            pos[i] = (0.0, 0.0, 0.0)
            direction[i] = (0.0, -hand_sign, 0.0)
            continue
        # exits trail each flute by half a pitch (on the flank / in the flute)
        psi = phases[i % N] - math.pi / N
        alpha = hand_sign * psi
        if exit_kind in ("tip", "face"):
            r = ratio * R
            z = float(envelope_radius_inverse_z(env, r)) if env is not None else 0.0
            d = (0.0, 0.0, -1.0)
        elif exit_kind == "flute":
            r = 0.5 * (0.5 * info.get("core_d", 0.0) + R)
            z = info.get("cutting_length", 0.0)
            d = (0.0, 0.0, -1.0)
        else:
            r = R
            z = 0.5 * info.get("cutting_length", 0.0)
            d = (math.cos(alpha), math.sin(alpha), 0.0)
        pos[i] = (r * math.cos(alpha), r * math.sin(alpha), z)
        direction[i] = d
    return CoolantHoles(count, diameter, count * math.pi * diameter * diameter / 4.0, exit_kind, pos, direction)


def envelope_radius_inverse_z(env: Envelope, r: float) -> float:
    """Lowest height of the tool chain where its radius reaches r (the tip surface at radius r)."""
    z = np.linspace(float(env.z_lo_m.min()), float(env.z_hi_m.max()), 4001)
    radius = envelope_radius(env, z, chain=CHAIN_TOOL, part_max=PART_BODY)
    hit = np.nonzero(radius >= r)[0]
    return float(z[hit[0]]) if len(hit) else 0.0


def max_speed(spec: ToolSpec, holder: HolderSpec | None) -> tuple[float | None, float | None, float | None]:
    tool = spec.max_rad_s
    hold = holder.max_rad_s if holder is not None else None
    known = [v for v in (tool, hold) if v is not None]
    return (min(known) if known else None), tool, hold


# --------------------------------------------------------------------------- assembly
def compute(spec: ToolSpec, info: dict, instance: InstanceSpec, holder: HolderSpec | None, env: Envelope | None,
            layout: dict, phases, hand_sign: float, db=None) -> ToolProps:
    rotating = env is not None and len(env.kind) > 0
    N = int(info.get("flutes", 1))
    sections: list[Section] = []
    tool_m = holder_m = None
    fluted = None
    per = total = flute_len = chip = 0.0
    core_d = info.get("core_d", 0.0)
    D = 2.0 * info.get("R", 0.0)
    if rotating:
        per, area = gullet(spec, info)
        total = N * per
        flute_len = max(info.get("cutting_length", 0.0), num(spec.geometry, "flute_length_m", 0.0) or 0.0)
        chip = total * flute_len
        # disc heads (keyseat, T-slot, saw) and indexable bodies are not fluted sections: their
        # critical sections are the neck / shank / arbor below
        fluted = None if (info.get("disc") or info.get("indexable")) else fluted_section(spec, info, phases)
        if fluted is not None:
            fluted.z_m = layout.get("z_flute_end", flute_len)
            sections.append(fluted)
        if layout.get("d_neck") is not None and layout.get("z_neck_end") is not None:
            sections.append(_round_section("neck", layout["z_neck_end"], layout["d_neck"]))
        if layout.get("stickout") is not None:
            sections.append(_round_section("shank_at_holder", layout["stickout"], layout["d_shank"]))
        tool_m = tool_mass(spec, info, env, layout, area, db)
        holder_m = holder_mass(holder, env, layout)
    else:
        sections.extend(_turning_sections(info))
    assembly = combine(tool_m, holder_m)
    vmax, vtool, vholder = max_speed(spec, holder)
    bal_tool = balance(spec.balance_grade, tool_m, vtool if vtool is not None else vmax)
    bal_holder = balance(holder.balance_grade if holder is not None else None, holder_m,
                         vholder if vholder is not None else vmax)
    coolant = coolant_holes(spec, info, phases, hand_sign, env if rotating else None)
    stick = layout.get("stickout")
    gauge = layout.get("z_gauge")
    shank_dir = tuple(info.get("shank_dir_T", (0.0, 0.0, 1.0)))
    extra = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool)) and not isinstance(v, np.ndarray)}
    return ToolProps(
        hand=spec.hand, hand_sign=hand_sign, rotating=rotating, diameter_m=D, flutes=N, core_d_m=core_d,
        gullet_area_per_flute_m2=per, gullet_volume_per_length_m3_per_m=total, flute_length_m=flute_len,
        chip_space_m3=chip, fluted=fluted, sections=sections, tool_mass=tool_m, holder_mass=holder_m,
        assembly_mass=assembly, max_rad_s=vmax, max_rad_s_tool=vtool, max_rad_s_holder=vholder,
        balance_tool=bal_tool, balance_holder=bal_holder, coolant=coolant, stickout_m=stick, gauge_to_tip_m=gauge,
        shank_dir_T=shank_dir, extra=extra)


def _turning_sections(info: dict) -> list[Section]:
    out = []
    if info.get("bar_d"):
        d = info["bar_d"]
        out.append(_round_section("bar_at_clamp", info.get("overhang") or 0.0, d))
    hb = info.get("shank_hb")
    if hb and len(hb) == 2:
        h, b = float(hb[0]), float(hb[1])
        I_h = b * h ** 3 / 12.0          # bending in the h direction (cutting force along the rake normal)
        I_b = h * b ** 3 / 12.0
        Z_min = min(b * h * h / 6.0, h * b * b / 6.0)
        # torsion of a solid rectangle 2a x 2b, a >= b (Young & Budynas 2012 Table 10.7 case 4):
        # tau_max = T (3a + 1.8b) / (8 a^2 b^2) at the middle of the long sides
        a_, b_ = max(h, b) / 2.0, min(h, b) / 2.0
        Z_t = 8.0 * a_ * a_ * b_ * b_ / (3.0 * a_ + 1.8 * b_)
        d_eq = (64.0 * min(I_h, I_b) / math.pi) ** 0.25
        out.append(Section("shank_at_clamp", info.get("overhang") or 0.0, Z_min, Z_t, min(I_h, I_b), max(I_h, I_b),
                           h * b, d_eq))
    return out
