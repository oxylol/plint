"""A setup of one operation: devices -> contact groups, the clamping solve, location error and seating
(docs/models/holding.md sections 4-6).

**Groups.** All devices except tabs hold the workpiece (group ``main``); tabs hold the part they connect to
the frame (group ``tabs``, checked against the same loads: the cutting wrench is assigned to the tab-held
part in full, which is conservative).

**Clamping solve** (Li & Melkote 1999: rigid part on elastic contacts with Coulomb friction; tangential
springs at the Mindlin ratio of the normal stiffness, Coulomb limit everywhere), three stages:

1. *locating*: the clamps press the part into its locators with their clamping force (force-controlled), no
   friction anywhere (the part is pushed home against every locator before the clamps lock it), with gravity
   (build orientation) and vacuum / magnetic hold-down forces; chips and burrs under a locating patch are an
   interference ``gap`` at its worst node (the node whose lift moves the part's reference point most). When a
   load component has no frictionless locator (gravity on a tilted table without side locators) the part is
   seated under gravity with friction instead, clamps not yet applied; a part that cannot rest without its
   clamps (gripped only) is taken as held in place until they close;
2. *clamping*: from the located state (re-anchored: its forces become preloads), friction acts at every
   contact while the clamps hold their force (a tailstock centre slides along its cone meanwhile);
3. *locked*: the clamped forces become the preloads of every contact and the clamps stiff springs; open
   nodes keep their remaining clearance. Loading solves start from this reference configuration.

**Location error** (Asada & By 1985 in its elastic form): the rigid-body twist of seating and clamping
with the debris minus that without, about the part's centre of mass O; the part's ``location_error_S_m`` is the
displacement of O in S, the pose delta its rigid transform. **Seated**: every node of the primary datum is
in contact or within ``seat_tolerance_m`` of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from . import abi as A
from . import devices as D
from . import kernels as K
from . import params as P
from .protocols import PartShape


@dataclass
class Body:
    mass_kg: float
    centroid_H: np.ndarray
    inertia_H: np.ndarray            # about the centroid, frame H
    l_c_m: float                     # half the diagonal of the part's bounding box

    def mass_matrix(self) -> np.ndarray:
        M = np.zeros((6, 6))
        M[:3, :3] = self.mass_kg * np.eye(3)
        M[3:, 3:] = self.inertia_H
        return M


@dataclass
class Group:
    name: str
    patches: list
    holddown: list
    body: Body
    rotating: bool = False
    axis_H: tuple | None = None
    node_patch: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    node_local: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    contacts: A.Contacts | None = None
    clamp_forces_H: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    w_ref: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gravity_H: np.ndarray = field(default_factory=lambda: np.zeros(3))
    location: np.ndarray = field(default_factory=lambda: np.zeros(6))
    seated: bool = True
    seat_gap_m: float = 0.0
    held: bool = True
    cfg: np.ndarray = field(default_factory=lambda: np.zeros(A.N_CFG))
    k_support_H: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    seal: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    bonded: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    debris_node: dict = field(default_factory=dict)       # patch id -> (node, height)

    @property
    def n(self) -> int:
        return int(self.node_patch.shape[0])


@dataclass
class Setup:
    op_index: int
    op_name: str
    workpiece: int
    T_H_S: np.ndarray
    shape: PartShape
    groups: list
    devices: list
    notes: list
    rotating: bool
    lathe: dict
    speed_limits: list                       # [(device, kind, max_rad_s)]
    no_grip: list                            # devices that do not reach the part
    empty: bool = False                      # the setup names no workholding
    tab_fraction: dict = field(default_factory=dict)   # tab patch id -> remaining section share
    gravity_H: np.ndarray = field(default_factory=lambda: np.zeros(3))   # gravity at the build (H)
    omega_built: float = 0.0                 # work-spindle speed the chuck forces were set for
    volume_built: float = 0.0                # part volume at the build (mass scaling of gravity)

    def group(self, name: str) -> Group | None:
        for g in self.groups:
            if g.name == name:
                return g
        return None


# --------------------------------------------------------------------------- helpers
def adjoint_wrench(force_H, point_H, origin_H) -> np.ndarray:
    w = np.zeros(6)
    w[:3] = force_H
    w[3:] = np.cross(np.asarray(point_H) - origin_H, force_H)
    return w


def twist_matrix(twist: np.ndarray, origin_H: np.ndarray) -> np.ndarray:
    """4x4 rigid transform of a twist (u, phi) about the point O (Rodrigues for the rotation)."""
    u, phi = np.asarray(twist[:3], float), np.asarray(twist[3:], float)
    angle = float(np.linalg.norm(phi))
    if angle > 0.0:
        k = phi / angle
        Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        R = np.eye(3) + math.sin(angle) * Kx + (1.0 - math.cos(angle)) * Kx @ Kx
    else:
        R = np.eye(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin_H + u - R @ origin_H
    return T


def material_props(material) -> dict:
    physical = P.field_of(material, "physical")
    youngs = P.field_of(physical, "youngs_modulus_Pa") if physical is not None else None
    if youngs is not None and hasattr(youngs, "x_unit"):
        youngs = float(youngs)
    else:
        youngs = P.num(physical, "youngs_modulus_Pa")
    return {"family": P.text(material, "family", ""), "density": P.num(physical, "density_kg_m3"),
            "ultimate": P.num(physical, "ultimate_Pa"), "youngs": youngs, "poisson": P.num(physical, "poisson")}


def operation(project, op_index: int):
    ops = list(P.field_of(project, "operations") or [])
    if not 0 <= op_index < len(ops):
        return None
    return ops[op_index]


def setup_entries(op) -> list:
    setup = P.field_of(op, "setup")
    return list(P.field_of(setup, "workholding") or []) if setup is not None else []


def stock_pose(op) -> np.ndarray:
    pose = P.field_of(op, "stock_pose_W_S")
    if pose is not None:
        return np.asarray(pose, dtype=float)
    setup = P.field_of(op, "setup")
    m = P.matrix(P.field_of(setup, "stock_pose")) if setup is not None else None
    return m if m is not None else np.eye(4)


def op_workpiece(op) -> int:
    setup = P.field_of(op, "setup")
    w = P.num(setup, "workpiece") if setup is not None else None
    return int(w) if w is not None else 0


def device_type(entry) -> tuple:
    item = P.field_of(entry, "device")
    item = P.as_map(item) if isinstance(item, dict) else item
    return item, P.text(item, "type", ""), P.text(entry, "label", "")


# --------------------------------------------------------------------------- building
def build_patches(op_name: str, entries: list, bc: D.BuildCtx, debris: dict) -> tuple[list, list, list, list, list]:
    """(patches, holddown, devices (summaries), notes, no_grip) of a setup's entries."""
    patches, holddown, devices, notes, no_grip = [], [], [], [], []
    soft = soft_entry = None
    for entry in entries:
        item, kind, _ = device_type(entry)
        if kind == "soft_jaws" and soft is None:
            soft, soft_entry = item, entry
    bc.soft_jaws = soft
    bc.soft_entry = soft_entry
    for j, entry in enumerate(entries):
        item, kind, label = device_type(entry)
        T_H_D = P.matrix(P.field_of(entry, "pose"))
        T_H_D = np.eye(4) if T_H_D is None else T_H_D
        if label.lower().startswith(D.LOCATOR_LABELS + D.SUPPORT_LABELS):
            out = D.build_point_locator(bc, j, entry, item, T_H_D, label)
        elif kind == "soft_jaws":
            devices.append({"entry": j, "device": P.text(item, "id"), "type": kind, "role": "jaw set"})
            continue
        elif kind == "magnetic":
            plate_id = f"{op_name}/{j}:{P.text(item, 'id', f'magnet{j}')}/magnet_plate"
            out = D.build_magnetic(bc, j, entry, item, T_H_D, label, float(debris.get(plate_id, 0.0)))
        elif kind in D.BUILDERS:
            out = D.BUILDERS[kind](bc, j, entry, item, T_H_D, label)
        else:
            raise InputError(f"{op_name}: workholding type {kind!r} has no holding model")
        if out.info.get("no_grip"):
            no_grip.append(P.text(item, "id", kind))
        patches.extend(out.patches)
        holddown.extend(out.holddown)
        notes.extend(out.notes)
        summary = {"entry": j, "device": P.text(item, "id", kind), "type": kind, "label": label}
        summary.update({k: v for k, v in out.info.items() if isinstance(v, (int, float, bool, str))})
        if "axis_H" in out.info:
            summary["_axis_H"] = out.info["axis_H"]
        if "point_H" in out.info:
            summary["_point_H"] = out.info["point_H"]
        if "push_H" in out.info:
            summary["_push_H"] = out.info["push_H"]
        devices.append(summary)
    return patches, holddown, devices, notes, no_grip


def auto_base(op_name: str, bc: D.BuildCtx, devices: list, patches: list) -> list:
    """The face a set of clamps presses the part against (table or fixture), when no device supports it."""
    if any(p.datum == "primary" for p in patches):
        return []
    pushes = [d["_push_H"] for d in devices if "_push_H" in d]
    if not pushes or bc.shape.kind != "box":
        return []
    push = np.sum(pushes, axis=0)
    if np.linalg.norm(push) <= 0.0:
        return []
    push = push / np.linalg.norm(push)
    lo, hi = np.asarray(bc.shape.lo_S_m, float), np.asarray(bc.shape.hi_S_m, float)
    R = bc.T_H_S[:3, :3]
    best, best_dot = None, -2.0
    for axis in range(3):
        for sign in (1, -1):
            n_S = np.zeros(3)
            n_S[axis] = sign
            dot = float(np.dot(R @ n_S, push))
            if dot > best_dot + 1e-12:
                best, best_dot = (axis, sign), dot
    axis, sign = best
    centre_S = 0.5 * (lo + hi)
    centre_S[axis] = hi[axis] if sign > 0 else lo[axis]
    others = [k for k in range(3) if k != axis]
    a1_S = np.zeros(3)
    a1_S[others[0]] = 1.0
    n_S = np.zeros(3)
    n_S[axis] = -sign
    half = 0.5 * (hi - lo)
    area = 4.0 * half[others[0]] * half[others[1]]
    T = bc.T_H_S
    return [D.rect_patch(f"{op_name}/table", "base", D.LAW_LOCATOR, "table", -1, T, centre_S, n_S, a1_S, None,
                         half[others[0]], half[others[1]], bc.n_side, D._k_joint(area), D._kt(),
                         P.p(P.WORK, "base_friction"), datum="primary", locating=True)]


def body_of(part_geometry, workpiece: int, T_H_S: np.ndarray, density: float) -> Body:
    volume, centroid_S, J_S = part_geometry.mass_properties_S(workpiece)
    R = T_H_S[:3, :3]
    c_H = R @ np.asarray(centroid_S, float) + T_H_S[:3, 3]
    shape = part_geometry.shape_S(workpiece)
    l_c = 0.5 * float(np.linalg.norm(shape.size_m)) if shape.kind == "box" else \
        0.5 * math.hypot(2.0 * shape.radius_m, shape.z1_m - shape.z0_m)
    return Body(float(density * volume), c_H, density * R @ np.asarray(J_S, float) @ R.T, max(l_c, 1e-6))


def assemble(group: Group, bond_failed: dict, tab_fraction: dict, vacuum_lost: set) -> None:
    """Node arrays of the group's patches (phase-1 law), with failed bond nodes and tab areas applied."""
    rows = []
    for pi, patch in enumerate(group.patches):
        failed = bond_failed.get(patch.id)
        frac = float(tab_fraction.get(patch.id, 1.0)) if patch.role == "tab" else 1.0
        for k in range(patch.n_nodes):
            rows.append((pi, k, failed is not None and k < len(failed) and bool(failed[k]), frac))
    n = len(rows)
    cs = A.empty_contacts(n)
    seal = np.zeros(n, bool)
    bonded = np.zeros(n, bool)
    node_patch = np.zeros(n, np.int64)
    node_local = np.zeros(n, np.int64)
    O = group.body.centroid_H
    for i, (pi, k, failed, frac) in enumerate(rows):
        patch = group.patches[pi]
        w = float(patch.weight[k])
        node_patch[i], node_local[i] = pi, k
        cs.r_m[i] = patch.pos_H[k] - O
        cs.n[i] = patch.n_H[k]
        cs.t1[i] = patch.t1_H[k]
        cs.t2[i] = patch.t2_H[k]
        k_node = patch.k_N_per_m * w * frac
        cs.kn_N_per_m[i] = k_node
        cs.kt_N_per_m[i] = (patch.kt_ratio[0] * k_node, patch.kt_ratio[1] * k_node)
        cs.mu[i] = patch.mu
        cs.gap_m[i] = patch.gap_m
        if patch.law == D.LAW_BOND:
            cs.kind[i] = A.KIND_OFF if (failed or frac <= 1e-9) else A.KIND_BONDED
            area = patch.area_m2 * w * frac
            cs.cap_N[i] = (patch.cap_t_Pa * area, patch.cap_s_Pa * area)
            bonded[i] = True
        else:
            cs.kind[i] = A.KIND_FRICTION
        if patch.law == D.LAW_LOCKED:
            cs.pre_N[i, 0] = patch.force_N * w
        if patch.seal is not None and k < len(patch.seal) and bool(patch.seal[k]) and patch.device not in vacuum_lost:
            seal[i] = True
    group.contacts = cs
    group.node_patch, group.node_local, group.seal, group.bonded = node_patch, node_local, seal, bonded


def _locked_nodes(group: Group) -> np.ndarray:
    return np.array([group.patches[int(i)].law == D.LAW_LOCKED for i in group.node_patch], dtype=bool)


def seat_contacts(group: Group, debris_gap: np.ndarray) -> A.Contacts:
    """Stage a (seating): the part rests on its locators and bonds under gravity and hold-down forces, the
    clamps not yet applied; chips and burrs raise their locator nodes (``debris_gap``)."""
    cs = group.contacts
    kind = cs.kind.copy()
    kind[_locked_nodes(group)] = A.KIND_OFF
    pre = cs.pre_N.copy()
    pre[_locked_nodes(group)] = 0.0
    return A.contiguous_contacts(cs._replace(kind=kind, pre_N=pre, gap_m=cs.gap_m + debris_gap))


def anchored(cs_x: A.Contacts, delta: np.ndarray, f: np.ndarray, st: np.ndarray) -> A.Contacts:
    """The contact set re-anchored at a solved state: its forces become preloads, open nodes keep their
    remaining clearance (positions unchanged: small displacements)."""
    n = cs_x.r_m.shape[0]
    pre = np.zeros((n, 3))
    gap = np.zeros(n)
    for i in range(n):
        d = delta[:3] + np.cross(delta[3:], cs_x.r_m[i])
        if st[i] in (A.ST_STICK, A.ST_SLIP, A.ST_BONDED):
            pre[i] = (f[i] @ cs_x.n[i], f[i] @ cs_x.t1[i], f[i] @ cs_x.t2[i])
        elif st[i] == A.ST_OPEN:
            gap[i] = cs_x.gap_m[i] - float(cs_x.n[i] @ d)
    return A.contiguous_contacts(cs_x._replace(pre_N=pre, gap_m=gap))


def clamp_contacts(group: Group, seated: A.Contacts) -> A.Contacts:
    """Stage b (clamping): from the seated state, the clamps apply their force (force-controlled: no normal
    stiffness yet), friction acts everywhere."""
    cs = group.contacts
    locked = _locked_nodes(group)
    kind = seated.kind.copy()
    kind[locked] = cs.kind[locked]
    kn = seated.kn_N_per_m.copy()
    kn[locked] = 0.0
    kt = seated.kt_N_per_m.copy()
    sliding = np.array([group.patches[int(i)].slides_while_clamping for i in group.node_patch], dtype=bool)
    kt[locked & sliding] = 0.0           # a centre slides along its cone while the thrust is applied
    pre = seated.pre_N.copy()
    pre[locked] = 0.0
    pre[locked, 0] = cs.pre_N[locked, 0]
    gap = seated.gap_m.copy()
    gap[locked] = 0.0
    return A.contiguous_contacts(seated._replace(kind=kind, kn_N_per_m=kn, kt_N_per_m=kt, pre_N=pre, gap_m=gap))


def external_ref(group: Group, gravity_H: np.ndarray, vacuum_lost: set) -> np.ndarray:
    O = group.body.centroid_H
    w = adjoint_wrench(group.body.mass_kg * np.asarray(gravity_H, float), O, O)
    for hd in group.holddown:
        if hd["kind"] == "vacuum" and hd["device"] in vacuum_lost:
            continue
        w += adjoint_wrench(hd["force_H"], hd["point_H"], O)
    return w


def solve_set(cs: A.Contacts, w_ext: np.ndarray, cfg: np.ndarray):
    n = cs.r_m.shape[0]
    delta = np.zeros(6)
    f = np.zeros((n, 3))
    st = np.zeros(n, np.int32)
    info = np.zeros(A.N_INFO)
    status = K.solve_equilibrium(cs, np.ascontiguousarray(w_ext, dtype=float), cfg, delta, f, st, info)
    return status, delta, f, st


def evaluate_set(cs: A.Contacts, cfg: np.ndarray):
    """Forces and states of a contact set at zero displacement (its preloads)."""
    n = cs.r_m.shape[0]
    f = np.zeros((n, 3))
    st = np.zeros(n, np.int32)
    K.contact_eval(cs, np.zeros(6), np.zeros(6), 0.0, float(cfg[A.CFG_L_C]), np.zeros(n, np.int32), f, st,
                   np.zeros(6), np.zeros((6, 6)))
    return f, st


def locate_contacts(group: Group, debris_gap: np.ndarray) -> A.Contacts:
    """Locating stage: the clamps press the part into its locators with their clamping force (force-controlled),
    no friction anywhere (the part is pushed home against every locator before the clamps lock it); chips and
    burrs raise their locator nodes."""
    cs = group.contacts
    locked = _locked_nodes(group)
    kn = cs.kn_N_per_m.copy()
    kn[locked] = 0.0
    bonded = cs.kind == A.KIND_BONDED
    kt = cs.kt_N_per_m.copy()
    kt[~bonded] = 0.0
    mu = cs.mu.copy()
    mu[~bonded] = 0.0
    return A.contiguous_contacts(cs._replace(kn_N_per_m=kn, kt_N_per_m=kt, mu=mu, gap_m=cs.gap_m + debris_gap))


def clamp_sequence(group: Group, debris_gap: np.ndarray, w_ext: np.ndarray, cfg: np.ndarray):
    """Locate (or seat), then clamp: (status, total twist, clamped forces, states, clamped contact set, twist
    of the clamping stage)."""
    cs_a = locate_contacts(group, debris_gap)
    status_a, delta_a, f_a, st_a = solve_set(cs_a, w_ext, cfg)
    if status_a != A.STATUS_HELD:
        # a load component no frictionless locator takes (gravity on a tilted table, no side locators): the part
        # is seated under gravity with friction, the clamps not yet applied
        cs_a = seat_contacts(group, debris_gap)
        status_a, delta_a, f_a, st_a = solve_set(cs_a, w_ext, cfg)
        if status_a != A.STATUS_HELD:
            # the part cannot rest without its clamps (gripped only, or nothing under it): it is held in place
            # by hand until the clamps close, the seat displacement is zero
            delta_a = np.zeros(6)
            f_a, st_a = evaluate_set(cs_a, cfg)
        seated = anchored(cs_a, delta_a, f_a, st_a)
    else:
        seated = anchored(cs_a, delta_a, f_a, st_a)
        seated = A.contiguous_contacts(seated._replace(kt_N_per_m=group.contacts.kt_N_per_m.copy(),
                                                       mu=group.contacts.mu.copy()))
    cs_b = clamp_contacts(group, seated)
    status_b, delta_b, f_b, st_b = solve_set(cs_b, w_ext, cfg)
    return status_b, delta_a + delta_b, f_b, st_b, cs_b, delta_b


def lock(group: Group, cs_b: A.Contacts, delta: np.ndarray, f: np.ndarray, st: np.ndarray) -> None:
    """Locked contact set: the clamped forces as preloads, the clamps now stiff springs."""
    locked = anchored(cs_b, delta, f, st)
    group.contacts = A.contiguous_contacts(locked._replace(kn_N_per_m=group.contacts.kn_N_per_m.copy(),
                                                           kt_N_per_m=group.contacts.kt_N_per_m.copy()))
    group.clamp_forces_H = f.copy()


def support_stiffness(group: Group) -> np.ndarray:
    cs = group.contacts
    n = group.n
    R = np.zeros(6)
    J = np.zeros((6, 6))
    K.contact_eval(cs, np.zeros(6), np.zeros(6), 0.0, group.cfg[A.CFG_L_C], np.zeros(n, np.int32), np.zeros((n, 3)),
                   np.zeros(n, np.int32), R, J)
    return -J


def clamp_group(group: Group, settings, gravity_H: np.ndarray, debris: dict, vacuum_lost: set,
                seat_tol_m: float) -> None:
    """Seat, clamp and lock (module doc), the location twist from the debris, seating."""
    cs = group.contacts
    n = group.n
    k_max = float(cs.kn_N_per_m.max()) if n else 1.0
    group.cfg = settings.cfg(k_max, group.body.l_c_m)
    cfg1 = settings.cfg(k_max, group.body.l_c_m, increments=4 * settings.load_increments)
    group.gravity_H = np.asarray(gravity_H, float).copy()
    w_ext = external_ref(group, gravity_H, vacuum_lost)
    group.w_ref = w_ext.copy()
    if n == 0:
        group.held = False
        return
    zero_gap = np.zeros(n)
    status0, total0, _, _, _, _ = clamp_sequence(group, zero_gap, w_ext, cfg1)
    debris_gap = np.zeros(n)
    group.debris_node = {}
    for pi, patch in enumerate(group.patches):
        h = float(debris.get(patch.id, 0.0))
        if h <= 0.0 or not patch.locating:
            continue
        nodes = np.flatnonzero(group.node_patch == pi)
        best, best_err = int(nodes[0]), -1.0
        for i in nodes:
            trial = debris_gap.copy()
            trial[i] += h
            status, total, _, _, _, _ = clamp_sequence(group, trial, w_ext, cfg1)
            if status != A.STATUS_HELD:
                continue
            dd = total - total0
            err = float(np.linalg.norm(dd[:3]) + group.body.l_c_m * np.linalg.norm(dd[3:]))
            if err > best_err + 1e-15:
                best, best_err = int(i), err
        debris_gap[best] += h
        group.debris_node[patch.id] = (int(group.node_local[best]), h)
    status, total, f, st, cs_b, delta_b = clamp_sequence(group, debris_gap, w_ext, cfg1)
    group.held = status == A.STATUS_HELD and status0 == A.STATUS_HELD
    if not group.held:
        return
    group.location = total - total0
    gap_max = 0.0
    for i in range(n):
        patch = group.patches[int(group.node_patch[i])]
        if patch.datum != "primary" or patch.law == D.LAW_BOND:
            continue
        if st[i] == A.ST_OPEN:
            d = delta_b[:3] + np.cross(delta_b[3:], cs_b.r_m[i])
            clearance = float(cs_b.n[i] @ d) - float(cs_b.gap_m[i])
            gap_max = max(gap_max, clearance)
    group.seat_gap_m = gap_max
    group.seated = gap_max <= seat_tol_m
    lock(group, cs_b, delta_b, f, st)
    group.k_support_H = support_stiffness(group)


def build_setup(project, op_index: int, part_geometry, tabs, settings, *, debris=None, gravity_H=None,
                slip_pose=None, bond_failed=None, vacuum_lost=None, omega_rad_s: float = 0.0,
                machine=None) -> Setup:
    """Devices, groups and the clamped state of operation ``op_index`` (module doc)."""
    op = operation(project, op_index)
    if op is None:
        raise InputError(f"holding: the project has no operation {op_index}")
    op_name = P.text(op, "name", f"op{op_index}")
    w = op_workpiece(op)
    entries = setup_entries(op)
    T_W_S = stock_pose(op)
    slip_pose = np.eye(4) if slip_pose is None else np.asarray(slip_pose, float)
    T_H_S = slip_pose @ T_W_S
    shape = part_geometry.shape_S(w)
    mat = material_props(P.field_of(project, "material"))
    if mat["density"] is None:
        raise InputError("holding: the work material needs physical.density_kg_m3")
    debris = dict(debris or {})
    vacuum_lost = set(vacuum_lost or ())
    bc = D.BuildCtx(op_name=op_name, workpiece=w, shape=shape, T_H_S=T_H_S, n_side=settings.patch_nodes,
                    centre_nodes=settings.centre_nodes, material_family=mat["family"], ultimate_Pa=mat["ultimate"],
                    youngs_Pa=mat["youngs"], poisson_work=mat["poisson"],
                    machine=machine if machine is not None else P.field_of(project, "machine"),
                    omega_rad_s=float(omega_rad_s), strict=not np.allclose(slip_pose, np.eye(4), rtol=0.0, atol=1e-15),
                    T_H_S0=T_W_S)
    patches, holddown, devices, notes, no_grip = build_patches(op_name, entries, bc, debris)
    patches.extend(auto_base(op_name, bc, devices, patches))
    rotating = any(d["type"] in D.ROTATING_TYPES for d in devices)
    gravity_H = np.array([0.0, 0.0, -9.80665]) if gravity_H is None else np.asarray(gravity_H, float)
    groups = []
    main = [p for p in patches if p.role != "tab"]
    tab_patches = [p for p in patches if p.role == "tab"]
    if tab_patches:
        # the tab-held part rests on the setup's primary datum like the workpiece (its weight goes there, the tabs
        # carry the cutting load); the datum patches are shared, not copied
        tab_patches = tab_patches + [p for p in main if p.datum == "primary" and p.law == D.LAW_LOCATOR]
    tab_fraction = {}
    for p in tab_patches:
        T_S_H = np.linalg.inv(T_H_S)
        c_S = T_S_H[:3, :3] @ p.centre_H + T_S_H[:3, 3]
        n_S = T_S_H[:3, :3] @ (-p.normal_H)
        a_S = p.axes_H @ T_S_H[:3, :3].T
        area, _ = tabs.section_S(w, c_S, n_S, a_S[0], a_S[1], float(p.half_m[0]), float(p.half_m[1]))
        nominal = 4.0 * float(p.half_m[0]) * float(p.half_m[1])
        tab_fraction[p.id] = min(max(area / nominal, 0.0), 1.0) if nominal > 0.0 else 0.0
    body = body_of(part_geometry, w, T_H_S, float(mat["density"]))
    for name, plist in (("main", main), ("tabs", tab_patches)):
        if name == "tabs" and not any(p.role == "tab" for p in plist):
            continue
        group = Group(name, plist, holddown if name == "main" else [], body, rotating)
        axes = [d["_axis_H"] for d in devices if "_axis_H" in d]
        group.axis_H = axes[0] if axes else None
        assemble(group, bond_failed or {}, tab_fraction, vacuum_lost)
        clamp_group(group, settings, gravity_H, debris, vacuum_lost, P.p(P.WORK, "seat_tolerance_m"))
        groups.append(group)
    limits = [(d["device"], d["type"], float(d["max_rad_s"])) for d in devices
              if isinstance(d.get("max_rad_s"), (int, float)) and d.get("max_rad_s")]
    setup = Setup(op_index, op_name, w, T_H_S, shape, groups, devices, notes, rotating, {}, limits, no_grip,
                  empty=not entries, tab_fraction=tab_fraction, gravity_H=gravity_H.copy(),
                  omega_built=float(omega_rad_s), volume_built=float(part_geometry.mass_properties_S(w)[0]))
    setup.lathe = lathe_support(setup)
    return setup


# --------------------------------------------------------------------------- lathe support summary
def lathe_support(setup: Setup) -> dict:
    """Support conditions of a turned part for structure and dynamics, positions along S z (the stock axis)."""
    if setup.shape.kind != "cylinder":
        return {}
    T_S_H = np.linalg.inv(setup.T_H_S)

    def z_S(point_H):
        return float((T_S_H[:3, :3] @ np.asarray(point_H, float) + T_S_H[:3, 3])[2])

    out: dict = {}
    main = setup.group("main")
    for d in setup.devices:
        kind = d["type"]
        if kind in D.ROTATING_TYPES and "_axis_H" in d:
            point_H, dir_H = d["_axis_H"]
            jaws = [p for p in (main.patches if main else []) if p.role == "jaw" and p.entry == d["entry"]]
            zs = [z_S(p.centre_H) for p in jaws]
            k_rad = float(d.get("jaw_stiffness_N_per_m", 0.0)) * max(len(jaws), 1) / 2.0
            tilt = 0.0
            if jaws:
                z0 = float(np.mean(zs))
                for p in jaws:
                    for k in range(p.n_nodes):
                        zk = z_S(p.pos_H[k])
                        tilt += 0.5 * p.k_N_per_m * float(p.weight[k]) * (zk - z0) ** 2
            out["chuck"] = {"device": d["device"], "type": kind, "z_face_S_m": z_S(point_H),
                            "grip_centre_S_m": float(np.mean(zs)) if zs else z_S(point_H),
                            "grip_length_m": float(d.get("grip_length_m", 0.0)), "jaws": int(d.get("jaws", 0)),
                            "gripping_force_N": float(d.get("gripping_force_N", 0.0)),
                            "radial_stiffness_N_per_m": k_rad, "tilt_stiffness_Nm_per_rad": tilt,
                            "inside": bool(d.get("inside", False))}
        elif kind == "tailstock_centre" and "_point_H" in d:
            out["tailstock"] = {"device": d["device"], "z_S_m": z_S(d["_point_H"]),
                                "thrust_N": float(d.get("thrust_N", 0.0)),
                                "radial_stiffness_N_per_m": float(d.get("radial_stiffness_N_per_m", 0.0)),
                                "live": bool(d.get("live", False))}
        elif kind == "steady_rest" and "_point_H" in d:
            out.setdefault("steady", []).append({"device": d["device"], "z_S_m": z_S(d["_point_H"]),
                                                 "rollers": int(d.get("rollers", 0)),
                                                 "force_per_roller_N": float(d.get("roller_force_N", 0.0)),
                                                 "stiffness_N_per_m": float(d.get("roller_stiffness_N_per_m", 0.0))})
    return out


# --------------------------------------------------------------------------- patches in S
def patch_face_S(patch, T_H_S: np.ndarray) -> tuple:
    """(id, centre_S, outward normal_S, half extents along the two other S axes in increasing order)."""
    T_S_H = np.linalg.inv(T_H_S)
    R = T_S_H[:3, :3]
    c = R @ patch.centre_H + T_S_H[:3, 3]
    n = R @ patch.normal_H
    a = patch.axes_H @ R.T
    axis = int(np.argmax(np.abs(n)))
    others = [k for k in range(3) if k != axis]
    half = np.array([abs(a[0, o]) * patch.half_m[0] + abs(a[1, o]) * patch.half_m[1] for o in others])
    return patch.id, c, n, half


__all__ = ["Body", "Group", "Setup", "build_setup", "build_patches", "auto_base", "assemble", "clamp_group",
           "external_ref", "adjoint_wrench", "twist_matrix", "lathe_support", "patch_face_S", "operation",
           "setup_entries", "stock_pose", "op_workpiece", "material_props", "body_of", "support_stiffness"]
