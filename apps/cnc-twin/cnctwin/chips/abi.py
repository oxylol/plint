"""Kernel layouts of the chips domain (docs/models/chips.md, architecture 5.5).

Everything a chips kernel reads or writes is a NamedTuple of writable, C-contiguous NumPy arrays
(float64 or int64). The layouts the hot loop shares with chips are:

* :data:`EMISSION_COLUMNS` - the rows the hot loop writes (one per tooth pass, through
  :func:`cnctwin.chips.kernels.write_emission`) into the chips EventBuffer of
  :class:`ChipsKernelInputs`;
* :class:`Occupancy` - the frozen chip-occupancy grid in S the hot loop applies recut impulses
  with (published once per macro step);
* :class:`PackingLoads` - the additive chip-packing torque and force per tool slot.

The other layouts (particles, pile height field, window, scene, model) are internal to the chips
L2 kernels. Frames: particles, cylinders, spheres and jets are in M; the pile grid G is a frame
fixed to S whose axes are a signed permutation of the S axes (``axes_S``: columns e_u, e_v, e_w in
S, e_w the S axis closest to "up"); positions in G are (u, v, w) from ``origin_G_m``.
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

from ..core.abi import EventBuffer, event_buffer
from ..units import REFERENCE_TEMP_K, STANDARD_GRAVITY_M_S2

# The constants below are compiled into the chips kernels as globals (@numba frozen constants), so this
# file is part of the kernel cache key of bin/tsafe and bin/twin.

# ------------------------------------------------------------------ particle status
FREE = 0            # unused slot
FLYING = 1          # moving (in air, in a jet, in the film, bouncing)
ON_TOOL = 2         # welded to a tool's cutting edge (owner = tool slot)
WRAPPED = 3         # wrapped around a rotating body (owner = tool slot, or -1 - workpiece)
ON_NOZZLE = 4       # hooked/stuck on a coolant nozzle (owner = nozzle index of the step)

# ------------------------------------------------------------------ surface kinds (pile base, contact pairs)
SURF_NONE = 0       # no surface below (open: chips fall through to the chip pan)
SURF_PART = 1
SURF_FIXTURE = 2
SURF_TABLE = 3
SURF_PILE = 4       # chip bed
SURF_ENCLOSURE = 5  # enclosure walls, roof, chip pan
SURF_TOOL = 6       # cutting part of a tool (flutes)
SURF_HOLDER = 7     # shank and holder
SURF_NOZZLE = 8
SURF_BODY = 9       # rotating workpiece (lathe part, bar)
N_SURF = 10
SURF_NAMES = ("none", "part", "fixture", "table", "pile", "enclosure", "tool", "holder", "nozzle", "body")

# ------------------------------------------------------------------ cylinder kinds
CYL_PADDLE = 0      # fluted cutting part of a rotating tool: flute faces sweep chips (tangential impulse)
CYL_SMOOTH = 1      # smooth rotating cylinder: shank, holder, lathe workpiece (friction only)

# ------------------------------------------------------------------ K1 event rows
EV_WELD = 1
EV_WRAP = 2
EV_NOZZLE = 3
EV_PAN = 4
EV_BOUNCE_LIMIT = 5
EVENT_COLUMNS = ("kind", "particle", "owner", "time_s", "mass_kg", "x_M_m", "y_M_m", "z_M_m", "temp_K", "form")
EVENT_WIDTH = len(EVENT_COLUMNS)

# ------------------------------------------------------------------ K1 scalar accumulators
OUT_HEAT_COOLANT_J = 0      # convective cooling of flying chips inside liquid (jets, film)
OUT_HEAT_AIR_J = 1          # convection to air + radiation to the enclosure
OUT_PAN_MASS_KG = 2         # mass that reached the chip pan (floor, wet walls, spill over edges)
OUT_PAN_ENTHALPY_J = 3
OUT_PAN_BULK_M3 = 4
OUT_DEPOSIT_KG = 5          # mass settled into the pile height field
OUT_SUBSTEPS = 6            # number of sub-steps taken (cost diagnostics)
OUT_HITS = 7                # number of contacts resolved
OUT_PAN_LONG_KG = 8         # long-form share of the pan mass (bird-nesting in the pan, conveyor)
N_OUT = 12

# ------------------------------------------------------------------ mobilised rows (flush, tool sweep, recut)
MOB_FLUSH = 0
MOB_SWEEP = 1
MOB_RECUT = 2
MOB_COLUMNS = ("kind", "iu", "iv", "mass_kg", "enthalpy_J", "size_mass", "thick_mass", "width_mass",
               "length_mass", "repose_mass", "long_kg", "fines_kg", "vx_M_m_s", "vy_M_m_s", "vz_M_m_s",
               "u_m", "v_m", "w_m", "owner", "time_s")   # release point (u, v, w) in G
MOB_WIDTH = len(MOB_COLUMNS)

# ------------------------------------------------------------------ emission rows (written by the hot loop)
EMISSION_COLUMNS = (
    "tick",          # base tick of chip separation (tooth leaves the cut, or end of the tooth period)
    "contact",       # contact index (RemovalContact.contact / CuttingContact.contact)
    "tool_slot",
    "tooth",
    "mass_kg",       # rho(20 degC) x analytic chip volume of this tooth pass (integral of h db r dphi)
    "x_M_m", "y_M_m", "z_M_m",          # chip centre at separation, M
    "vx_M_m_s", "vy_M_m_s", "vz_M_m_s",  # chip centre velocity at separation, M (edge velocity + chip flow)
    "temp_K",        # mean chip temperature at separation (shear-zone value)
    "h_m",           # mass-weighted mean uncut chip thickness
    "b_m",           # chip width (engaged edge length normal to the flow)
    "l_m",           # uncut chip length (arc length swept in the cut)
    "v_cut_m_s",     # mean cutting speed
    "rake_rad",      # mean normal rake
    "incl_rad",      # mean inclination (chip flow angle by Stabler's rule)
    "edge_kind",     # dominant cnctwin.core.abi EDGE_* kind
    "group",         # dominant coefficient group (ProcessCoefficients row)
)
EMISSION_WIDTH = len(EMISSION_COLUMNS)
EM = {name: i for i, name in enumerate(EMISSION_COLUMNS)}

N_FORM = 100        # form tables are indexed by the ISO 3685 code (core.bus.CHIP_FORMS), 0..99


class Particles(NamedTuple):
    """Representative particles (parcels of identical real chips), M frame. Per-chip shape fields
    describe one real chip; ``mass_kg`` is the parcel."""

    pos_M_m: np.ndarray         # [cap, 3]
    vel_M_m_s: np.ndarray       # [cap, 3]
    time_s: np.ndarray          # [cap] time the state refers to
    mass_kg: np.ndarray         # [cap] parcel mass
    temp_K: np.ndarray          # [cap]
    thickness_m: np.ndarray     # [cap] one chip
    width_m: np.ndarray         # [cap]
    length_m: np.ndarray        # [cap]
    curl_radius_m: np.ndarray   # [cap]
    dv_m: np.ndarray            # [cap] volume-equivalent sphere diameter of one chip
    chip_mass_kg: np.ndarray    # [cap] one chip
    area_m2: np.ndarray         # [cap] drag reference area of one chip: pi dv^2 / 4 (Haider-Levenspiel)
    surface_m2: np.ndarray      # [cap] surface area of one chip
    hl_a: np.ndarray            # [cap] Haider-Levenspiel A(psi)
    hl_b: np.ndarray            # [cap] B(psi)
    hl_c: np.ndarray            # [cap] C(psi)
    hl_d: np.ndarray            # [cap] D(psi)
    form: np.ndarray            # [cap] int64 ISO 3685 code
    status: np.ndarray          # [cap] int64 FREE / FLYING / ON_TOOL / WRAPPED / ON_NOZZLE
    owner: np.ndarray           # [cap] int64 body it is attached to (-1 none)
    slot: np.ndarray            # [cap] int64 tool slot that made it
    pid: np.ndarray             # [cap] int64 unique particle id (key of counter-based draws)
    bounces: np.ndarray         # [cap] int64
    wet: np.ndarray             # [cap] int64 0/1
    n: np.ndarray               # [1] int64 high-water mark of used slots


class Pile(NamedTuple):
    """Settled chips as a height field over the table/part region (grid G fixed to S).

    Extensive fields (bulk volume, mass, enthalpy and the mass-weighted sums) mix and move by
    simple addition; means are sums / mass. ``base_m`` is the w of the surface the pile rests on
    (in-process workpiece, fixture or table top), ``-inf`` where nothing is below.
    """

    origin_G_m: np.ndarray      # [3] in S: corner of cell (0, 0) at w = 0
    axes_S: np.ndarray          # [3, 3] columns e_u, e_v, e_w in S
    cell_m: np.ndarray          # [1]
    base_m: np.ndarray          # [nu, nv]
    base_kind: np.ndarray       # [nu, nv] int64 SURF_*
    bulk_m3: np.ndarray         # [nu, nv] bulk (loose) volume
    mass_kg: np.ndarray         # [nu, nv]
    enthalpy_J: np.ndarray      # [nu, nv] relative to 20 degC
    size_mass: np.ndarray       # [nu, nv] sum m dv
    thick_mass: np.ndarray      # [nu, nv] sum m thickness
    width_mass: np.ndarray      # [nu, nv] sum m width
    length_mass: np.ndarray     # [nu, nv] sum m length
    repose_mass: np.ndarray     # [nu, nv] sum m tan(angle of repose)
    long_kg: np.ndarray         # [nu, nv] long-form chips (ribbon, tubular, snarled, long helical)
    fines_kg: np.ndarray        # [nu, nv] chips thinner than the fines threshold
    wet: np.ndarray             # [nu, nv] int64 wetted by coolant this step
    flush_Pa: np.ndarray        # [nu, nv] coolant dynamic pressure over the cell
    film_uv_m_s: np.ndarray     # [nu, nv, 2] coolant film velocity in G (u, v)
    surf_temp_K: np.ndarray     # [nu, nv] temperature of the surface below


class Window(NamedTuple):
    """The macro-step window chips integrate over (the window of the motion record they read)."""

    t0_s: np.ndarray            # [1]
    t_end_s: np.ndarray         # [1]
    dt_tick_s: np.ndarray       # [1] spacing of the samples below
    grid_origin_M: np.ndarray   # [n_t + 1, 3] position in M of the G origin at t0 + k dt_tick
    R_M_G: np.ndarray           # [3, 3] G -> M rotation (constant over the window)
    gravity_M: np.ndarray       # [3] m/s2


class Scene(NamedTuple):
    """Obstacles and fluids of the window, M frame."""

    box_lo_M: np.ndarray        # [3] enclosure interior
    box_hi_M: np.ndarray        # [3]
    floor: np.ndarray           # [2] int64 (axis, side 0 = lo / 1 = hi) of the chip-pan face
    walls_wet: np.ndarray       # [1] int64 flood coolant washes the walls down to the pan
    cyl_base_M: np.ndarray      # [nc, n_t + 1, 3] axis point at z = 0 (tool tip, or chuck face)
    cyl_axis_M: np.ndarray      # [nc, n_t + 1, 3] unit axis (+z_T for tools)
    cyl_omega: np.ndarray       # [nc, n_t + 1] rad/s right-handed about the axis
    cyl_radius: np.ndarray      # [nc]
    cyl_length: np.ndarray      # [nc]
    cyl_kind: np.ndarray        # [nc] int64 CYL_*
    cyl_surf: np.ndarray        # [nc] int64 SURF_TOOL / SURF_HOLDER / SURF_BODY (contact pair)
    cyl_owner: np.ndarray       # [nc] int64 tool slot, or -1 - workpiece
    cyl_wrap: np.ndarray        # [nc] int64 long chips can wrap on it
    cyl_weld: np.ndarray        # [nc] int64 hot chips can weld on it (cutting edges)
    sph_center_M: np.ndarray    # [ns, 3] coolant nozzles
    sph_radius: np.ndarray      # [ns]
    sph_active: np.ndarray      # [ns] int64 jet running
    jet_origin_M: np.ndarray    # [nj, 3]
    jet_dir_M: np.ndarray       # [nj, 3]
    jet_speed: np.ndarray       # [nj] m/s
    jet_r0: np.ndarray          # [nj] exit radius
    jet_tan: np.ndarray         # [nj] tan(spray half angle) (liquid); unused for gas jets
    jet_gas: np.ndarray         # [nj] int64 1 = gas jet (air, mist, MQL): turbulent round-jet decay
    jet_fluid: np.ndarray       # [nj, 4] rho, mu, k, Pr of the jet fluid
    air: np.ndarray             # [5] rho, mu, k, Pr, T of the enclosure air
    liquid: np.ndarray          # [7] rho, mu, k, Pr, T, film thickness, surface tension of the coolant
    jet_const: np.ndarray       # [2] round-jet centreline decay constant B, spreading rate S (Pope 2000)


class ChipModel(NamedTuple):
    """Chip material, model coefficients and numerical settings of the K1-K5 kernels."""

    rho_kg_m3: np.ndarray       # [1] chip material density at 20 degC
    cp_T_K: np.ndarray          # [n] specific heat table
    cp_J_kgK: np.ndarray        # [n]
    h_J_kg: np.ndarray          # [n] specific enthalpy at cp_T_K (0 at 20 degC)
    emissivity: np.ndarray      # [1]
    weld_temp_K: np.ndarray     # [1] chips hotter than this weld to a cutting edge they touch
    hl: np.ndarray              # [14] Haider-Levenspiel constants (24, A: 3, B: 2, C: 4, D: 4)
    rm: np.ndarray              # [4] Ranz-Marshall Nu = a + b Re^m Pr^n
    e_pair: np.ndarray          # [N_SURF] normal restitution chip vs surface kind
    mu_pair: np.ndarray         # [N_SURF] Coulomb friction chip vs surface kind
    paddle_e: np.ndarray        # [1] restitution of a chip struck by a flute face
    wet: np.ndarray             # [3] ln(film thickness / asperity height) (Ennis), cos(contact angle),
                                #     Wilhelmy perimeter-force factor
    form_R: np.ndarray          # [N_FORM] bulk volume ratio (bulk / solid)
    form_repose_tan: np.ndarray  # [N_FORM] tan(angle of repose)
    form_long: np.ndarray       # [N_FORM] int64 long-form flag
    form_wrap: np.ndarray       # [N_FORM] int64 can wrap on a rotating body
    fines_thickness_m: np.ndarray  # [1]
    settle_bounce: np.ndarray   # [1] settle when the rebound apex is below this many chip sizes
    wrap_turns: np.ndarray      # [1] chip length / body circumference needed to wrap
    hook_radii: np.ndarray      # [1] chip length / nozzle radius needed to hook on a running nozzle
    numerics: np.ndarray        # [5] dt_max_s, drag_dt_factor (dt <= factor / k_drag), contact_offset_m,
                                #     max_bounces, max_zero_advance_hits


class Occupancy(NamedTuple):
    """Frozen chip-occupancy grid in S for recutting (hot loop input)."""

    origin_S_m: np.ndarray      # [3]
    cell_m: np.ndarray          # [1]
    mass_kg: np.ndarray         # [nx, ny, nz]
    size_m: np.ndarray          # [nx, ny, nz] mass-weighted volume-equivalent chip diameter
    thickness_m: np.ndarray     # [nx, ny, nz] mass-weighted chip thickness


class PackingLoads(NamedTuple):
    """Additive chip-packing loads per tool slot (hot loop input). The torque opposes the spindle
    rotation (the hot loop applies -sign(omega) * torque about +z_T); the force is in T0."""

    torque_Nm: np.ndarray       # [n_slots]
    force_T0_N: np.ndarray      # [n_slots, 3]


class ChipsKernelInputs(NamedTuple):
    emission: EventBuffer       # rows EMISSION_COLUMNS, written by the hot loop (write_emission)
    occupancy: Occupancy
    packing: PackingLoads


# ------------------------------------------------------------------ constructors

def _f(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _i(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.int64)


def particles(capacity: int) -> Particles:
    cap = max(int(capacity), 1)
    return Particles(_f(cap, 3), _f(cap, 3), _f(cap), _f(cap), _f(cap), _f(cap), _f(cap), _f(cap), _f(cap),
                     _f(cap), _f(cap), _f(cap), _f(cap), _f(cap), _f(cap), _f(cap), _f(cap),
                     _i(cap), _i(cap), np.full(cap, -1, np.int64), np.full(cap, -1, np.int64), _i(cap), _i(cap),
                     _i(cap), _i(1))


def grow_particles(p: Particles, capacity: int) -> Particles:
    """A copy with ``capacity`` slots (>= the current one), contents kept."""
    new = particles(max(capacity, len(p.mass_kg)))
    n = len(p.mass_kg)
    for name in Particles._fields:
        if name == "n":
            continue
        getattr(new, name)[:n] = getattr(p, name)
    new.n[0] = p.n[0]
    return new


def pile(nu: int, nv: int, origin_G_m=(0.0, 0.0, 0.0), axes_S=None, cell_m: float = 1e-3) -> Pile:
    nu, nv = max(int(nu), 1), max(int(nv), 1)
    axes = np.eye(3) if axes_S is None else np.ascontiguousarray(axes_S, dtype=np.float64)
    base = np.full((nu, nv), -np.inf)
    return Pile(np.ascontiguousarray(origin_G_m, dtype=np.float64), axes.copy(), np.array([float(cell_m)]),
                base, _i(nu, nv), _f(nu, nv), _f(nu, nv), _f(nu, nv), _f(nu, nv), _f(nu, nv), _f(nu, nv),
                _f(nu, nv), _f(nu, nv), _f(nu, nv), _f(nu, nv), _i(nu, nv), _f(nu, nv), _f(nu, nv, 2),
                np.full((nu, nv), REFERENCE_TEMP_K))


def window(n_t: int) -> Window:
    n = max(int(n_t), 1)
    return Window(_f(1), _f(1), np.ones(1), _f(n + 1, 3), np.eye(3), np.array([0.0, 0.0, -STANDARD_GRAVITY_M_S2]))


def scene(n_cyl: int, n_t: int, n_sph: int, n_jet: int) -> Scene:
    n = max(int(n_t), 1)
    return Scene(np.full(3, -1.0), np.full(3, 1.0), np.array([2, 0], np.int64), _i(1),
                 _f(n_cyl, n + 1, 3), _f(n_cyl, n + 1, 3), _f(n_cyl, n + 1), _f(n_cyl), _f(n_cyl),
                 _i(n_cyl), _i(n_cyl), _i(n_cyl), _i(n_cyl), _i(n_cyl),
                 _f(n_sph, 3), _f(n_sph), _i(n_sph),
                 _f(n_jet, 3), _f(n_jet, 3), _f(n_jet), _f(n_jet), _f(n_jet), _i(n_jet), _f(n_jet, 4),
                 _f(5), _f(7), _f(2))


def chip_model(n_cp: int = 2) -> ChipModel:
    return ChipModel(_f(1), _f(n_cp), _f(n_cp), _f(n_cp), _f(1), _f(1), _f(14), _f(4), _f(N_SURF), _f(N_SURF),
                     _f(1), _f(3), np.ones(N_FORM), _f(N_FORM), _i(N_FORM), _i(N_FORM), _f(1), _f(1), _f(1), _f(1),
                     _f(5))


def step_out(event_capacity: int) -> tuple[np.ndarray, EventBuffer]:
    return _f(N_OUT), event_buffer(max(int(event_capacity), 1), EVENT_WIDTH)


def occupancy(nx: int = 1, ny: int = 1, nz: int = 1, origin_S_m=(0.0, 0.0, 0.0), cell_m: float = 0.0) -> Occupancy:
    return Occupancy(np.ascontiguousarray(origin_S_m, dtype=np.float64), np.array([float(cell_m)]),
                     _f(nx, ny, nz), _f(nx, ny, nz), _f(nx, ny, nz))


def packing_loads(n_slots: int) -> PackingLoads:
    n = max(int(n_slots), 1)
    return PackingLoads(_f(n), _f(n, 3))


def emission_buffer(capacity: int) -> EventBuffer:
    return event_buffer(max(int(capacity), 1), EMISSION_WIDTH)


# ------------------------------------------------------------------ Numba types of the layouts

PARTICLES_T = numba.typeof(particles(1))
PILE_T = numba.typeof(pile(1, 1))
WINDOW_T = numba.typeof(window(1))
SCENE_T = numba.typeof(scene(1, 1, 1, 1))
MODEL_T = numba.typeof(chip_model())
EVENTS_T = numba.typeof(event_buffer(1, EVENT_WIDTH))
OCCUPANCY_T = numba.typeof(occupancy())


def is_contiguous_layout(nt) -> bool:
    """True when every array of a layout is writable and C-contiguous (kernel argument rule)."""
    return all(isinstance(a, np.ndarray) and a.flags.c_contiguous and a.flags.writeable for a in nt)
