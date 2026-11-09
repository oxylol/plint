"""Tool-holder-spindle model of one tool slot -> :class:`cnctwin.core.bus.ToolModes` (docs/models/structure.md 2).

Rotating tools: the tool is a stepped Timoshenko beam from its tip (z_T = 0, +z_T toward the spindle) to the top
of its shank: fluted section with the reduced second moment of area from cnctwin.tooling (the sector model with
the core and gullet, or Kops & Vo's equivalent diameter 0.8 D when they are not known; the mean of the two
principal values in the non-rotating frame T0), neck and shank circles. The holder is a second stepped beam from
its nose (the stick-out, plus holding's pull-out) to the gauge line, its outline from the holder file, bored to
the shank diameter where the shank is inside. Over the clamped length the shank and the bore are joined by
distributed contact springs (Winkler foundation: lateral ``(pi d / 2)(k_n + k_t)``, axial ``pi d k_t``, torsional
``pi d (d/2)^2 k_t`` per length; k_n per holder type, k_t = 2 (1 - nu) / (2 - nu) k_n, Mindlin). The gauge-line
node is rigidly coupled to the spindle-holder base modes of the machine file (``spindles[].frf_modes``): each
mode is a modal coordinate of unit mass with stiffness omega^2 whose flange translation is d / sqrt(m) and whose
flange rotation is that of a rocking about a pivot ``spindle_pivot_length`` behind the gauge line. This is the
receptance coupling of the tool-holder receptances with the base receptances (RCSA) written as one finite
system; its undamped modes are the poles and residues of the coupled FRF, so the modal parameters are extracted
exactly (no curve fit), damping by the modal strain energy of the hysteretic parts plus the base modes' viscous
damping (Johnson & Kienholz 1982). The coupled FRF at the tool tip is evaluated by RCSA (rcsa.py) at the
dominant modes to measure the error of that damping model (``ToolModes.fit_error``).

Non-rotating tools (turning, boring bars, form tools): the shank or bar is a beam along tooling's shank
direction from the tip to its clamp, clamped over ``shank_clamp_length_ratio`` x height (bars:
``bar_clamp_length_ratio`` x diameter) by the same contact springs to a rigid tool block; the block moves with
the machine's turret modes (directions in M, turned into T0 with the pose of the path when the tool is active).

Outputs: frequencies, damping ratios, mass-normalised 6-component shapes at the stations (T0), the static
residual flexibility between station translations (static compliance minus the published modes), the critical
sections and their section moduli (tooling's), the static compliance at the tip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from ..core.bus import ToolModes
from ..core.errors import InputError
from . import beam as B
from . import modal as MO
from . import params as P
from . import rcsa
from .materials import Elastic, holder_steel, tool_substrate

AXIS_LETTERS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
QUASI_STATIC = 0.05      # fit check frequency below the first mode (fraction of omega_1)


# ------------------------------------------------------------------------------- inputs
@dataclass
class BaseMode:
    """One mode of the base (spindle flange or turret) in the beam's frame: unit modal mass coordinate."""

    omega_rad_s: float
    zeta: float
    psi: np.ndarray                 # [6] flange translation and rotation per unit modal coordinate
    label: str = ""


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        try:
            value = obj[key]
        except (KeyError, TypeError, IndexError):
            value = getattr(obj, key, default)
    return default if value is None else value


def _num(obj, key, default=None):
    value = _get(obj, key)
    if value is None:
        return default
    value = getattr(value, "value", value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _direction(value) -> np.ndarray:
    if value is None:
        return np.array(AXIS_LETTERS["x"])
    value = getattr(value, "value", value)
    if isinstance(value, str):
        if value.lower() not in AXIS_LETTERS:
            raise InputError(f"mode direction {value!r} is not x, y, z or a vector")
        return np.array(AXIS_LETTERS[value.lower()])
    v = np.array([float(getattr(c, "value", c)) for c in value], dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise InputError("mode direction is the zero vector")
    return v / n


def _modes_of(entries, label: str) -> list[tuple[float, float, float, np.ndarray]]:
    out = []
    for i, md in enumerate(entries or []):
        fn = _num(md, "fn_Hz")
        zeta = _num(md, "zeta")
        k = _num(md, "k_N_per_m")
        if fn is None or zeta is None or k is None:
            raise InputError(f"{label} mode {i}: needs fn_Hz, zeta and k_N_um")
        if fn <= 0.0 or k <= 0.0 or not 0.0 < zeta < 1.0:
            raise InputError(f"{label} mode {i}: fn_Hz and k must be positive and 0 < zeta < 1")
        out.append((2.0 * math.pi * fn, zeta, k, _direction(_get(md, "direction", "x"))))
    return out


def spindle_base_modes(machine, spindle_id: str, pivot_m: float) -> list[BaseMode]:
    """The flange modes of a spindle as base modes in T0 (directions in the spindle-nose frame = T0): translation
    d / sqrt(m), rotation -(e_z x d) / (a sqrt(m)) (rocking about a point a behind the gauge line)."""
    spindles = list(_get(machine, "spindles") or [])
    entries = None
    for sp in spindles:
        if str(_get(sp, "id", "")) == str(spindle_id):
            entries = _get(sp, "frf_modes")
            break
    if entries is None and spindles:
        entries = _get(spindles[0], "frf_modes")
    out = []
    ez = np.array([0.0, 0.0, 1.0])
    for j, (w, zeta, k, d) in enumerate(_modes_of(entries, "spindle frf")):
        m = k / (w * w)
        rot = -np.cross(ez, d) / pivot_m
        out.append(BaseMode(w, zeta, np.concatenate([d, rot]) / math.sqrt(m), f"spindle {j + 1}"))
    return out


def turret_base_modes(machine, R_M_T0: np.ndarray) -> list[BaseMode]:
    """Turret modes of the machine (directions in M) as pure translations of the tool block in T0."""
    structure = _get(machine, "structure")
    out = []
    for j, (w, zeta, k, d) in enumerate(_modes_of(_get(structure, "turret_modes"), "turret")):
        m = k / (w * w)
        d_T0 = np.asarray(R_M_T0, float).T @ d
        out.append(BaseMode(w, zeta, np.concatenate([d_T0, np.zeros(3)]) / math.sqrt(m), f"turret {j + 1}"))
    return out


# ------------------------------------------------------------------------------- the assembled system
@dataclass
class ToolSystem:
    """The coupled tool-holder-base system (reduced: base-node DOFs replaced by the base coordinates)."""

    K: np.ndarray
    M: np.ndarray
    C: np.ndarray                         # viscous damping of the base coordinates
    parts: list                           # [(name, eta, K_part reduced)]
    T: np.ndarray                         # full FE DOFs = T @ reduced DOFs
    K_full: np.ndarray                    # FE matrices before coupling (RCSA substructure, free at the base)
    M_full: np.ndarray
    parts_full: list                      # [(name, eta, K_part full)]
    base_dofs: np.ndarray                 # full DOFs of the base node (empty: no base node)
    base: list                            # BaseMode
    node_z_m: np.ndarray                  # node positions along the beam axis
    node_beam: np.ndarray                 # 0 tool, 1 holder / block
    R_T0_b: np.ndarray                    # columns: beam axes in T0
    station_nodes: np.ndarray
    station_z_m: np.ndarray               # along +z_T in T0
    tip_node: int
    rotating: bool
    notes: list = field(default_factory=list)

    def full(self, x: np.ndarray) -> np.ndarray:
        return self.T @ x


def _rigid_link(r: np.ndarray) -> np.ndarray:
    """6 x 6 map of a master node's (u, theta) to a point offset by r rigidly attached: u + theta x r."""
    A = np.eye(6)
    A[0, 4], A[0, 5] = r[2], -r[1]
    A[1, 3], A[1, 5] = -r[2], r[0]
    A[2, 3], A[2, 4] = r[1], -r[0]
    return A


def _reduce(asm: B.BeamAssembly, parts_full: list, slaves: dict, base: list) -> tuple:
    """Transformation full -> reduced: nodes in ``slaves`` (node -> 6 x n_q map of the base coordinates) are
    removed; reduced DOFs = remaining node DOFs, then the base coordinates."""
    n_full = asm.K.shape[0]
    nq = len(base)
    keep = [n for n in range(asm.n_nodes) if n not in slaves]
    n_red = 6 * len(keep) + nq
    T = np.zeros((n_full, n_red))
    for i, n in enumerate(keep):
        T[6 * n:6 * n + 6, 6 * i:6 * i + 6] = np.eye(6)
    for n, S in slaves.items():
        if nq:
            T[6 * n:6 * n + 6, 6 * len(keep):] = S
    K = T.T @ asm.K @ T
    M = T.T @ asm.M @ T
    C = np.zeros_like(K)
    for j, bm in enumerate(base):
        q = 6 * len(keep) + j
        K[q, q] += bm.omega_rad_s ** 2
        M[q, q] += 1.0
        C[q, q] += 2.0 * bm.zeta * bm.omega_rad_s
    parts = [(name, eta, T.T @ Kp @ T) for name, eta, Kp in parts_full]
    return 0.5 * (K + K.T), 0.5 * (M + M.T), C, parts, T


def _interp_profile(rows, z_from_nose: float) -> float:
    zs = np.array([r[0] for r in rows], dtype=float)
    ds = np.array([r[1] for r in rows], dtype=float)
    return float(np.interp(z_from_nose, zs, ds))


def _envelope_radius(geo, z: float) -> float:
    from ..tooling.props import envelope_radius
    return float(envelope_radius(geo.envelope, np.array([z]), chain=0)[0])


def _lateral_contact(d_m: float, k_n: float, k_t: float) -> tuple[float, float, float, float]:
    """(lateral, axial, torsional, bending-rotation stiffness) per length of a shank of diameter d gripped all
    round with normal / tangential contact stiffness k_n / k_t per area: lateral (pi d / 2)(k_n + k_t) (the
    normal and tangential components of a rigid translation), axial pi d k_t, torsional pi d (d/2)^2 k_t,
    rotation pi (d/2)^3 k_t (a section rotating by theta moves its surface axially by r cos(phi) theta)."""
    r = 0.5 * d_m
    return 0.5 * math.pi * d_m * (k_n + k_t), math.pi * d_m * k_t, math.pi * d_m * r * r * k_t, math.pi * r ** 3 * k_t


def _tool_section(geo, zm: float, mat: Elastic, z_flute: float, z_neck: float | None) -> B.Section:
    props = geo.props
    lay = geo.layout
    R = 0.5 * props.diameter_m if props.diameter_m > 0.0 else _envelope_radius(geo, zm)
    if zm < z_flute:
        r_env = _envelope_radius(geo, zm)
        s = min(max(r_env / R, 1e-3), 1.0) if R > 0.0 else 1.0
        fl = props.fluted
        if fl is not None:
            I_mean = 0.5 * (fl.I_min_m4 + fl.I_max_m4) * s ** 4
            A = fl.area_m2 * s * s
            d_eq = fl.d_eq_m * s
            k = B.cowper_circle(mat.nu)
            return B.Section(A, I_mean, I_mean, 2.0 * I_mean, k, k, mat.E_Pa, B.shear_modulus(mat.E_Pa, mat.nu),
                             mat.rho_kg_m3, mass_A_m2=A, mass_Ix_m4=I_mean, mass_Iy_m4=I_mean) \
                if d_eq > 0.0 else B.circle(2.0 * r_env, mat.E_Pa, mat.nu, mat.rho_kg_m3)
        return B.circle(max(2.0 * r_env, 1e-4), mat.E_Pa, mat.nu, mat.rho_kg_m3)
    if z_neck is not None and zm < z_neck and lay.get("d_neck"):
        return B.circle(float(lay["d_neck"]), mat.E_Pa, mat.nu, mat.rho_kg_m3)
    return B.circle(float(lay["d_shank"]), mat.E_Pa, mat.nu, mat.rho_kg_m3)


def _pick_stations(z: np.ndarray, keep: list[float], n: int) -> np.ndarray:
    """Indices of up to n nodes: the nodes nearest to the ``keep`` positions, then uniform in z."""
    chosen = set()
    for zk in keep:
        chosen.add(int(np.argmin(np.abs(z - zk))))
    if len(z) <= n:
        return np.arange(len(z))
    targets = np.linspace(z[0], z[-1], n)
    for t in targets:
        if len(chosen) >= n:
            break
        chosen.add(int(np.argmin(np.abs(z - t))))
    return np.array(sorted(chosen), dtype=np.int64)


def build_rotating(geo, machine, settings, *, pullout_m: float = 0.0, length_loss_m: float = 0.0, db=None,
                   spindle_id: str = "main") -> ToolSystem:
    """The coupled system of a rotating tool (module doc). ``geo``: tooling's ToolGeometry."""
    lay = geo.layout
    props = geo.props
    notes: list[str] = []
    spec = geo.spec
    tool_mat = tool_substrate(getattr(spec, "tool_material_id", None), getattr(spec, "tool_material_class", None),
                              db)
    steel = holder_steel()
    holder = geo.holder
    overall = float(lay["overall"])
    z0 = float(length_loss_m)
    if z0 >= overall:
        raise InputError(f"tool slot {geo.slot}: the length lost ({z0:g} m) is the whole tool")
    d_shank = float(lay["d_shank"])
    z_flute = float(lay.get("z_flute_end") or 0.0)
    z_neck = float(lay["z_neck_end"]) if lay.get("z_neck_end") is not None else None
    stick = lay.get("stickout")
    if stick is None:
        stick = max(overall - P.p(P.TOOL, "clamp_length_ratio") * d_shank, z_flute)
        notes.append(f"no stick-out: the top {P.p(P.TOOL, 'clamp_length_ratio'):g} d of the shank are clamped")
    z_nose = float(stick) + float(pullout_m)
    inserted = overall - z_nose
    if inserted <= 0.0:
        raise InputError(f"tool slot {geo.slot}: the tool is not inside its holder (stick-out {z_nose:g} m)")
    clamp = holder.clamp_length_m if holder is not None and holder.clamp_length_m else \
        P.p(P.TOOL, "clamp_length_ratio") * d_shank
    L_c = min(float(clamp), inserted)
    z_clamp = z_nose + L_c
    htype = holder.type if holder is not None else None
    k_n = P.contact_stiffness_N_per_m3(htype)
    k_t = P.mindlin_ratio(P.p(P.TOOL, "contact_poisson")) * k_n
    eta_c = P.contact_loss_factor(htype)
    if holder is not None and P.family(htype) != htype and htype not in P.HOLDER_FAMILY:
        notes.append(f"holder type {htype!r}: collet contact values used")
    # node positions
    tool_bp = [z0, overall, z_nose, z_clamp] + [v for v in (z_flute, z_neck) if v is not None and z0 < v < overall]
    if holder is not None:
        z_gauge = z_nose + holder.gauge_length_m
        hold_bp = [z_nose, z_gauge, z_clamp] + [z_nose + r[0] for r in holder.profile_m
                                               if 0.0 < r[0] < holder.gauge_length_m]
        if overall < z_gauge:
            hold_bp.append(overall)
    else:
        z_gauge = z_clamp
        hold_bp = [z_nose, z_clamp]
    z_all = B.subdivide(tool_bp + hold_bp, settings.tool_element_m)
    zt = z_all[(z_all >= z0 - 1e-12) & (z_all <= overall + 1e-12)]
    zh = z_all[(z_all >= z_nose - 1e-12) & (z_all <= z_gauge + 1e-12)]
    nt, nh = len(zt), len(zh)
    asm = B.BeamAssembly(nt + nh)
    n = 6 * (nt + nh)
    K_tool, K_hold, K_int = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i in range(nt - 1):
        zm = 0.5 * (zt[i] + zt[i + 1])
        asm.add_element(i, i + 1, zt[i + 1] - zt[i], _tool_section(geo, zm, tool_mat, z_flute, z_neck),
                        stiffness_into=K_tool)
    if holder is not None:
        for i in range(nh - 1):
            zm = 0.5 * (zh[i] + zh[i + 1])
            d_out = _interp_profile(holder.profile_m, zm - z_nose)
            d_in = d_shank if zm < overall else 0.0
            if d_out <= d_in:
                d_out = d_in * 1.05
                notes.append(f"holder outline at {zm * 1e3:.1f} mm is not wider than the shank: 5 % wall used")
            sec = B.circle(d_out, steel.E_Pa, steel.nu, steel.rho_kg_m3, d_in)
            asm.add_element(nt + i, nt + i + 1, zh[i + 1] - zh[i], sec, stiffness_into=K_hold)
    else:
        # no holder: the block is a rigid body (all block nodes slaved to the base master below)
        notes.append("no holder: the shank is clamped in a rigid block on the spindle")
    k_lat, k_ax, k_tor, k_rot = _lateral_contact(d_shank, k_n, k_t)
    for i in range(nt - 1):
        a, b = zt[i], zt[i + 1]
        if a < z_nose - 1e-12 or b > z_clamp + 1e-12:
            continue
        ja = int(np.argmin(np.abs(zh - a)))
        jb = int(np.argmin(np.abs(zh - b)))
        asm.add_coupling(i, i + 1, nt + ja, nt + jb, B.foundation(b - a, k_lat, k_ax, k_tor, k_rot_N=k_rot),
                         stiffness_into=K_int)
    base = spindle_base_modes(machine, spindle_id, P.p(P.TOOL, "spindle_pivot_length_m")) if machine is not None \
        else []
    if not base:
        notes.append("no spindle flange modes in the machine file: the spindle is rigid at the gauge line")
    master = nt + nh - 1
    slaves = {}
    Psi = np.stack([bm.psi for bm in base], axis=1) if base else np.zeros((6, 0))
    if holder is None:
        for j in range(nh):
            slaves[nt + j] = _rigid_link(np.array([0.0, 0.0, zh[j] - zh[-1]])) @ Psi
    else:
        slaves[master] = Psi
    parts_full = [("tool", P.p(P.TOOL, "tool_loss_factor"), K_tool), ("holder", P.p(P.TOOL, "holder_loss_factor"),
                                                                       K_hold), ("interface", eta_c, K_int)]
    K, M, C, parts, T = _reduce(asm, parts_full, slaves, base)
    node_z = np.concatenate([zt, zh])
    node_beam = np.concatenate([np.zeros(nt, np.int64), np.ones(nh, np.int64)])
    # stations: tool nodes up to the nose, holder nodes above it (a continuous axis line tip -> gauge line)
    line = [i for i in range(nt) if zt[i] <= z_nose + 1e-12] + [nt + j for j in range(nh) if zh[j] > z_nose + 1e-12]
    line_z = node_z[np.array(line)]
    keep_z = [z0, z_flute, z_nose, z_gauge] + ([z_neck] if z_neck is not None else [])
    pick = _pick_stations(line_z, keep_z, settings.tool_stations)
    stations = np.array(line, dtype=np.int64)[pick]
    return ToolSystem(K, M, C, parts, T, asm.K.copy(), asm.M.copy(), parts_full,
                      np.arange(6 * master, 6 * master + 6) if holder is not None else np.zeros(0, np.int64), base,
                      node_z, node_beam, np.eye(3), stations, node_z[stations], 0, True, notes)


def _beam_frame(shank_dir_T) -> np.ndarray:
    """Columns: beam axes x_b, y_b, z_b (z_b along the shank from the tip) in T0."""
    zb = np.asarray(shank_dir_T, dtype=float)
    zb = zb / np.linalg.norm(zb)
    ref = np.array([0.0, 1.0, 0.0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    xb = np.cross(ref, zb)
    xb /= np.linalg.norm(xb)
    yb = np.cross(zb, xb)
    return np.stack([xb, yb, zb], axis=1)


def build_static(geo, machine, settings, *, R_M_T0: np.ndarray | None = None, db=None) -> ToolSystem:
    """The coupled system of a non-rotating tool (module doc)."""
    info = geo.info
    notes: list[str] = []
    steel = holder_steel()
    R = _beam_frame(geo.props.shank_dir_T)
    bar_d = info.get("bar_d")
    hb = info.get("shank_hb")
    overhang = info.get("overhang")
    eta_mat = P.p(P.TOOL, "holder_loss_factor")
    if bar_d:
        material = str(info.get("bar_material") or "steel")
        if material == "carbide":
            mat = tool_substrate(getattr(geo.spec, "tool_material_id", None), "carbide", db)
        else:
            mat = steel
        if material == "damped":
            eta_mat = P.p(P.TOOL, "damped_bar_loss_factor")
        sec = B.circle(float(bar_d), mat.E_Pa, mat.nu, mat.rho_kg_m3)
        L_c = P.p(P.TOOL, "bar_clamp_length_ratio") * float(bar_d)
        k_n = P.contact_stiffness_N_per_m3("turret_vdi")
        k_t = P.mindlin_ratio(P.p(P.TOOL, "contact_poisson")) * k_n
        k_x, k_ax, k_tor, k_about_y = _lateral_contact(float(bar_d), k_n, k_t)
        k_y, k_about_x = k_x, k_about_y
    elif hb and len(hb) == 2:
        h, b = float(hb[0]), float(hb[1])
        # x_b is along z_T (width b), y_b along y_T (height h, the cutting direction) for external holders
        bx, hy = (b, h) if abs(R[:, 1] @ np.array([0.0, 1.0, 0.0])) > 0.5 else (h, b)
        sec = B.rectangle(bx, hy, steel.E_Pa, steel.nu, steel.rho_kg_m3)
        L_c = P.p(P.TOOL, "shank_clamp_length_ratio") * hy
        k_n = P.contact_stiffness_N_per_m3("turret_vdi")
        k_t = P.mindlin_ratio(P.p(P.TOOL, "contact_poisson")) * k_n
        # lateral along x: the two faces normal to x in compression, the two others in shear (per length);
        # rotation about y (ry): the surface moves axially by x theta, resisted by k_t over the perimeter
        k_x = 2.0 * (hy * k_n + bx * k_t)
        k_y = 2.0 * (bx * k_n + hy * k_t)
        k_ax = 2.0 * (bx + hy) * k_t
        k_tor = 2.0 * (bx + hy) * k_t * (0.25 * (bx * bx + hy * hy))
        k_about_y = k_t * (hy * bx * bx / 2.0 + bx ** 3 / 6.0)
        k_about_x = k_t * (bx * hy * hy / 2.0 + hy ** 3 / 6.0)
    else:
        raise InputError(f"tool slot {geo.slot}: a non-rotating tool needs its bar diameter or shank size")
    if not overhang:
        overhang = 4.0 * (float(bar_d) if bar_d else float(hb[0]))
        notes.append(f"no overhang given: {overhang * 1e3:.1f} mm (4 x section) assumed")
    overhang = float(overhang)
    z = B.subdivide([0.0, overhang, overhang + L_c], settings.tool_element_m)
    nb = len(z)
    clamp_nodes = [i for i in range(nb) if z[i] >= overhang - 1e-12]
    asm = B.BeamAssembly(nb + len(clamp_nodes))
    n = 6 * asm.n_nodes
    K_tool, K_int = np.zeros((n, n)), np.zeros((n, n))
    for i in range(nb - 1):
        asm.add_element(i, i + 1, z[i + 1] - z[i], sec, R=R, stiffness_into=K_tool)
    block = {node: nb + k for k, node in enumerate(clamp_nodes)}
    for i in range(nb - 1):
        if i in block and (i + 1) in block:
            Kf = B.foundation(z[i + 1] - z[i], k_x, k_ax, k_tor, k_lat_y_N_per_m2=k_y, k_rot_N=k_about_y,
                              k_rot_y_N=k_about_x)
            asm.add_coupling(i, i + 1, block[i], block[i + 1], Kf, R=R, stiffness_into=K_int)
    R_MT = np.eye(3) if R_M_T0 is None else np.asarray(R_M_T0, dtype=float)
    base = turret_base_modes(machine, R_MT) if machine is not None else []
    Psi = np.stack([bm.psi for bm in base], axis=1) if base else np.zeros((6, 0))
    master_z = z[clamp_nodes[-1]]
    slaves = {}
    for node, bnode in block.items():
        r_T0 = R @ np.array([0.0, 0.0, z[node] - master_z])
        slaves[bnode] = _rigid_link(r_T0) @ Psi if base else np.zeros((6, 0))
    if not base:
        notes.append("no turret modes in the machine file: the tool block is rigid")
    parts_full = [("tool", eta_mat, K_tool), ("interface", P.p(P.TOOL, "contact_loss_factor_turret"), K_int)]
    K, M, C, parts, T = _reduce(asm, parts_full, slaves, base)
    node_z = np.concatenate([z, z[np.array(clamp_nodes)]])
    node_beam = np.concatenate([np.zeros(nb, np.int64), np.ones(len(clamp_nodes), np.int64)])
    # the element and coupling matrices were rotated into T0: the DOFs are T0 components (R_T0_b = I)
    return ToolSystem(K, M, C, parts, T, asm.K.copy(), asm.M.copy(), parts_full, np.zeros(0, np.int64), base,
                      node_z, node_beam, np.eye(3), np.array([0], np.int64), np.zeros(1), 0, False, notes)


# ------------------------------------------------------------------------------- modes and ToolModes
@dataclass
class ToolResult:
    modes: ToolModes
    tip_compliance_m_per_N: np.ndarray    # [3, 3] static, T0
    fit_error: float
    omega_rad_s: np.ndarray
    notes: list
    system: ToolSystem | None = None


def _axes_projector(sys: ToolSystem) -> np.ndarray:
    """[n_red, 3]: reduced vectors with unit translation along T0 x, y, z at every free FE node."""
    n_red = sys.K.shape[0]
    n_free_nodes = (n_red - len(sys.base)) // 6
    A = np.zeros((n_red, 3))
    for i in range(n_free_nodes):
        for a in range(3):
            A[6 * i:6 * i + 3, a] = sys.R_T0_b[a, :]
    return A


def _node_vectors_T0(sys: ToolSystem, x_full: np.ndarray, node: int) -> np.ndarray:
    u = x_full[6 * node:6 * node + 6]
    R = sys.R_T0_b
    return np.concatenate([R @ u[:3], R @ u[3:]])


def _station_loads(sys: ToolSystem) -> np.ndarray:
    """[n_red, 3 n_st]: generalized loads of unit forces along T0 x, y, z at every station."""
    n_full = sys.T.shape[0]
    F = np.zeros((n_full, 3 * len(sys.station_nodes)))
    for s, node in enumerate(sys.station_nodes):
        for a in range(3):
            F[6 * node:6 * node + 3, 3 * s + a] = sys.R_T0_b[a, :]      # local components of e_a
    return sys.T.T @ F


def modal_damping(sys: ToolSystem, lam: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Modal strain energy of the hysteretic parts plus the base modes' viscous damping (module doc)."""
    zeta = np.zeros(len(lam))
    for k in range(len(lam)):
        v = V[:, k]
        w = math.sqrt(max(lam[k], 0.0))
        z = 0.0
        for _name, eta, Kp in sys.parts:
            z += 0.5 * eta * float(v @ Kp @ v) / lam[k]
        z += 0.5 * float(v @ sys.C @ v) / w if w > 0.0 else 0.0
        zeta[k] = z
    return zeta


def solve_modes(sys: ToolSystem, settings) -> ToolResult:
    lam, V = MO.dense_modes(sys.K, sys.M, _axes_projector(sys))
    pos = lam > 0.0
    lam, V = lam[pos], V[:, pos]
    f = MO.freq_Hz(lam)
    keep = np.nonzero(f <= settings.tool_max_freq_Hz)[0][: settings.tool_max_modes]
    if len(keep) == 0:
        keep = np.arange(min(1, len(lam)))
    lam_k, V_k = lam[keep], V[:, keep]
    zeta = modal_damping(sys, lam_k, V_k)
    nst = len(sys.station_nodes)
    shapes = np.zeros((len(keep), nst, 6))
    for k in range(len(keep)):
        x = sys.full(V_k[:, k])
        for s, node in enumerate(sys.station_nodes):
            shapes[k, s] = _node_vectors_T0(sys, x, int(node))
    F = _station_loads(sys)
    X = sla.solve(sys.K, F, assume_a="pos", check_finite=False)
    C = F.T @ X
    C = 0.5 * (C + C.T)
    phi_t = shapes[:, :, :3].reshape(len(keep), 3 * nst)
    residual = C - (phi_t.T / lam_k[None, :]) @ phi_t
    residual = 0.5 * (residual + residual.T)
    tip = int(np.argmin(sys.station_z_m)) if sys.rotating else 0
    tip_c = C[3 * tip:3 * tip + 3, 3 * tip:3 * tip + 3].copy()
    fit = fit_error(sys, lam_k, zeta, phi_t, residual, tip)
    modes = ToolModes(freq_Hz=MO.freq_Hz(lam_k), zeta=zeta, shape=shapes, station_z_m=sys.station_z_m.copy(),
                      static_residual_m_per_N=residual, sections_z_m=np.zeros(0), sections_modulus_m3=np.zeros(0))
    return ToolResult(modes, tip_c, fit, np.sqrt(lam_k), list(sys.notes), sys)


def fit_error(sys: ToolSystem, lam: np.ndarray, zeta: np.ndarray, phi_t: np.ndarray, residual: np.ndarray,
              tip: int) -> float:
    """Error of the modal model against the RCSA receptance G of the tip along T0 x, relative to max |G|: the
    largest difference of |G| at omega_k and of |G| and Re G at omega_k (1 -+ zeta_k) of every mode carrying at
    least 1 % of the static tip compliance (the half-power band; the extremes of Re G, which set the stability
    lobes, lie at its ends), and of |G| at the quasi-static point (5 % of the first mode; there the hysteretic
    parts add a phase the viscous modal model does not carry). G_RCSA couples the free tool-holder FE to
    the base receptances (rcsa.couple_fe_to_modal_base) with the hysteretic parts as complex stiffness."""
    col = 3 * tip
    contrib = phi_t[:, col] ** 2 / lam
    total = float(np.sum(contrib) + residual[col, col])
    if total <= 0.0 or len(lam) == 0:
        return 0.0
    picks = [(QUASI_STATIC * float(math.sqrt(lam[0])), True)]
    for k in range(len(lam)):
        if contrib[k] >= 0.01 * total:
            w = float(math.sqrt(lam[k]))
            # at omega_k itself Re G crosses zero steeply: a 1e-4 shift of the damped peak (non-proportional
            # damping) would dominate, so only |G| is compared there
            picks.extend([(w * (1.0 - zeta[k]), False), (w, True), (w * (1.0 + zeta[k]), False)])
    node = int(sys.station_nodes[tip])
    dofs_local = np.arange(6 * node, 6 * node + 3)
    e_local = sys.R_T0_b[0, :]
    w_b = np.array([b.omega_rad_s for b in sys.base])
    z_b = np.array([b.zeta for b in sys.base])
    Psi = np.stack([b.psi for b in sys.base], axis=1) if sys.base else np.zeros((6, 0))
    hyst = [(eta, Kp) for _n, eta, Kp in sys.parts_full]
    pairs = []
    for w, magnitude_only in picks:
        if len(sys.base_dofs):
            H = rcsa.couple_fe_to_modal_base(sys.K_full, sys.M_full, w, dofs_local, sys.base_dofs, w_b, z_b, Psi,
                                             hyst)
            g_exact = complex(e_local @ H @ e_local)
        else:
            Z = rcsa.dynamic_stiffness(sys.K, sys.M, w, [(eta, Kp) for _n, eta, Kp in sys.parts], sys.C)
            f_full = np.zeros(sys.T.shape[0])
            f_full[6 * node:6 * node + 3] = e_local
            f_red = sys.T.T @ f_full
            x = np.linalg.solve(Z, f_red.astype(complex))
            g_exact = complex(f_red @ x)
        den = lam - w * w + 2j * zeta * np.sqrt(lam) * w
        g_modal = complex(np.sum(phi_t[:, col] ** 2 / den) + residual[col, col])
        pairs.append((magnitude_only, g_exact, g_modal))
    peak = max(abs(g) for _q, g, _m in pairs)
    worst = 0.0
    for magnitude_only, g_exact, g_modal in pairs:
        err = abs(abs(g_modal) - abs(g_exact))
        if not magnitude_only:
            err = max(err, abs(g_modal.real - g_exact.real))
        worst = max(worst, err)
    return worst / peak if peak > 0.0 else 0.0


def tool_modes(geo, machine, settings, *, pullout_m: float = 0.0, length_loss_m: float = 0.0, R_M_T0=None,
               db=None, spindle_id: str = "main") -> ToolResult:
    """ToolModes of one tool (with tooling's critical sections)."""
    if geo.rotating:
        sys = build_rotating(geo, machine, settings, pullout_m=pullout_m, length_loss_m=length_loss_m, db=db,
                             spindle_id=spindle_id)
    else:
        sys = build_static(geo, machine, settings, R_M_T0=R_M_T0, db=db)
    res = solve_modes(sys, settings)
    secs = sorted(getattr(geo.props, "sections", []) or [], key=lambda s: s.z_m)
    res.modes.sections_z_m = np.array([s.z_m for s in secs], dtype=float)
    res.modes.sections_modulus_m3 = np.array([s.Z_bending_m3 for s in secs], dtype=float)
    res.modes.tip_compliance_m_per_N = res.tip_compliance_m_per_N.copy()
    res.modes.fit_error = float(res.fit_error)
    return res


# ------------------------------------------------------------------------------- static helpers
def static_deflection(res: ToolResult, force_T0_N, at_station: int, to_stations=None) -> np.ndarray:
    """Static deflection [n, 3] (T0) at the stations under a force at station ``at_station`` from the published
    modes and residual (exact static compliance of the model at the stations)."""
    tm = res.modes
    lam = res.omega_rad_s ** 2
    nst = tm.station_z_m.shape[0]
    phi = tm.shape[:, :, :3].reshape(len(lam), 3 * nst)
    C = (phi.T / lam[None, :]) @ phi + tm.static_residual_m_per_N
    f = np.zeros(3 * nst)
    f[3 * at_station:3 * at_station + 3] = np.asarray(force_T0_N, dtype=float)
    u = (C @ f).reshape(nst, 3)
    return u if to_stations is None else u[np.asarray(to_stations)]


def section_stresses(sections, force_T0_N, z_force_m: float, torque_Nm: float = 0.0) -> list[dict]:
    """Bending moment, bending, axial, shear and von Mises stress at each critical section (tooling's
    ``Section``: z_m, Z_bending_m3, Z_polar_m3, area_m2) of a cantilevered tool under a force at z_force (T0) and
    a torque about the axis: M = |F_lateral| (z_s - z_F) above the force (beam statics), sigma = M / Z_b +
    F_z / A, tau = T / Z_p, sigma_vM = sqrt(sigma^2 + 3 tau^2)."""
    F = np.asarray(force_T0_N, dtype=float)
    lateral = math.hypot(F[0], F[1])
    out = []
    for s in sorted(sections, key=lambda s: s.z_m):
        arm = max(float(s.z_m) - float(z_force_m), 0.0)
        M = lateral * arm
        sigma = M / s.Z_bending_m3 + (abs(F[2]) / s.area_m2 if s.area_m2 > 0.0 else 0.0)
        tau = abs(torque_Nm) / s.Z_polar_m3 if s.Z_polar_m3 > 0.0 else 0.0
        out.append({"name": s.name, "z_m": float(s.z_m), "moment_Nm": M, "sigma_bending_Pa": M / s.Z_bending_m3,
                    "sigma_Pa": sigma, "tau_Pa": tau, "sigma_vm_Pa": math.sqrt(sigma * sigma + 3.0 * tau * tau)})
    return out


def tip_frf(res: ToolResult, omega_rad_s, direction: int = 0) -> np.ndarray:
    """Tip receptance (m/N, T0) along one axis from the published modes and residual."""
    tm = res.modes
    tip = int(np.argmin(tm.station_z_m))
    lam = res.omega_rad_s ** 2
    phi = tm.shape[:, tip, direction]
    w = np.atleast_1d(np.asarray(omega_rad_s, dtype=float))
    den = lam[None, :] - (w * w)[:, None] + 2j * (tm.zeta * res.omega_rad_s)[None, :] * w[:, None]
    nst = tm.station_z_m.shape[0]
    r = tm.static_residual_m_per_N[3 * tip + direction, 3 * tip + direction] if tm.static_residual_m_per_N.shape[0] \
        == 3 * nst else 0.0
    return np.sum(phi[None, :] ** 2 / den, axis=1) + r


__all__ = ["BaseMode", "ToolSystem", "ToolResult", "spindle_base_modes", "turret_base_modes", "build_rotating",
           "build_static", "solve_modes", "modal_damping", "fit_error", "tool_modes", "static_deflection",
           "section_stresses", "tip_frf"]
