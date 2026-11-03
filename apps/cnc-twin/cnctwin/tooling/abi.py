"""Kernel layouts owned by tooling (docs/models/tooling.md, "Layouts").

The shared edge layout is :class:`cnctwin.core.abi.EdgeModel`; this module adds the layouts
other domains read from tooling: the revolved tool envelope (removal, collision broad phase),
the thread tooth profile (swept-profile subtraction for taps, thread mills and threading
inserts), the 2-D insert outline of turning tools (polar field, revolved swept profile), the
coefficient-group table (cutting's coefficient pipeline), the per-element angle tables of the
hot loop (sin/cos by angle addition) and the per-macro-step composition inputs.

Every array is writable and C-contiguous; all values are SI. Angles in the edge layouts are in
the tool's *cutting sense*: psi increases in the direction the cutting edges move when the tool
runs in its cutting direction. With ``hand_sign`` h (-1 right-hand: cuts with omega < 0, M03;
+1 left-hand) the right-handed angle about +z_T is alpha = h * psi, and the element midpoint in
T at spindle angle 0 is (r cos psi, h r sin psi, z).

Envelope profile convention (rotating tools): the outline of the revolved tool in the meridian
half-plane (z along +z_T from the tip, r >= 0 from the axis), as a chain of pieces traversed
counter-clockwise around the solid (from the axis at the bottom, outward along the bottom, up
the side, inward along tops), so the outward normal of every piece is its direction turned by
-90 degrees. A piece is a straight frustum/disc between (z0, r0) and (z1, r1) or a circular arc
(centre (zc, rc), radius rho > 0, from angle a0 to a1, point(a) = (zc - rho cos a, rc + rho sin a),
outward normal (sin a e_r - cos a e_z): a = 0 is the bottom of the arc, pi/2 its outermost point).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------- envelope piece kinds
ENV_FRUSTUM = 0     # straight piece between (z0, r0) and (z1, r1): cylinder, cone, or disc when z0 == z1
ENV_ARC = 1         # circular arc in the meridian plane: torus section (rc > 0) or sphere section (rc = 0)

# ---------------------------------------------------------------- envelope parts
PART_EDGE = 0       # swept by cutting edges (cutting flag 1)
PART_BODY = 1       # fluted body without cutting edges (flute run-out, relieved flutes)
PART_NECK = 2
PART_SHANK = 3
PART_HOLDER = 4
PART_SPINDLE = 5
PART_INSERT = 6     # insert / head body of indexable and turning tools outside its cutting edges
PART_NAMES = ("edge", "body", "neck", "shank", "holder", "spindle", "insert")


class Envelope(NamedTuple):
    """Revolved outline of a rotating tool from the tip to the spindle nose (convention above).

    ``np`` pieces in profile order, in one closed chain per solid (``chain``: 0 tool, 1 holder,
    2 spindle nose; each chain starts and ends on the axis). ``z_lo_m``/``z_hi_m``/``r_max_m``
    bound each piece. The envelope is the union of the chains: the solid at height z is the disc
    of radius max over pieces covering z (tools are filled to the axis; coolant holes are not part
    of the envelope).
    """

    kind: np.ndarray        # [np] int32 ENV_*
    part: np.ndarray        # [np] int32 PART_*
    cutting: np.ndarray     # [np] int8: 1 when the piece is swept by cutting edges
    z0_m: np.ndarray        # [np] start point (along the profile direction)
    r0_m: np.ndarray        # [np]
    z1_m: np.ndarray        # [np] end point
    r1_m: np.ndarray        # [np]
    zc_m: np.ndarray        # [np] arc centre (ENV_ARC; 0 otherwise)
    rc_m: np.ndarray        # [np]
    rho_m: np.ndarray       # [np] arc radius (ENV_ARC; 0 otherwise)
    a0_rad: np.ndarray      # [np] arc start angle (point(a) above)
    a1_rad: np.ndarray      # [np] arc end angle
    z_lo_m: np.ndarray      # [np] bounding range of the piece
    z_hi_m: np.ndarray      # [np]
    r_max_m: np.ndarray     # [np]
    chain: np.ndarray       # [np] int32: 0 tool, 1 holder, 2 spindle nose


class ThreadProfile(NamedTuple):
    """Tooth profiles of thread-generating tools for swept-profile subtraction.

    One closed polygon per (tooth, row): ``pts_zr_m[j, m, :n_pts[j, m]]`` in the meridian plane of
    the tool frame (rotating tools: (z_T, r); threading inserts: (z_T, x_T) with x_T into the
    material, like the EdgeModel convention for non-rotating tools). Taps include the chamfer
    truncation of every chamfer tooth and the helical lead shift of every land; thread mills have
    annular rows (no lead shift: the helix comes from the tool path); a threading insert has one
    tooth and one row. Polygons are counter-clockwise in (z, r).
    """

    pts_zr_m: np.ndarray    # [nt, n_rows, n_max, 2]
    n_pts: np.ndarray       # [nt, n_rows] int32
    pitch_m: np.ndarray     # [1] thread pitch P
    lead_m: np.ndarray      # [1] lead (starts * P); 0 for annular thread mills
    starts: np.ndarray      # [1] int32
    thread_hand: np.ndarray  # [1] float64: +1 right-hand thread, -1 left-hand
    form_angle_rad: np.ndarray  # [1] included flank angle
    major_r_m: np.ndarray   # [1] crest radius of the tool profile (rotating) / 0 for inserts
    depth_m: np.ndarray     # [1] tooth depth from crest to root
    row_z_m: np.ndarray     # [nt, n_rows] axial position of each tooth's crest centre in T
    tooth_phase_rad: np.ndarray  # [nt] cutting-sense phase of each land / flute (as EdgeModel)


class InsertProfile(NamedTuple):
    """2-D outline of a non-rotating tool's insert in its reference plane ((z_T, x_T), x_T into
    the material), closed polygon counter-clockwise; ``cutting[i]`` flags segment i -> i+1."""

    pts_zx_m: np.ndarray    # [n, 2]
    cutting: np.ndarray     # [n] int8 (segment i joins point i and point (i + 1) % n)
    nose_centre_zx_m: np.ndarray  # [2] centre of the active nose radius
    nose_radius_m: np.ndarray     # [1]


class GroupTable(NamedTuple):
    """Coefficient groups of one EdgeModel (``elem_group`` indexes the rows).

    Groups are formed by edge kind, rake class and inclination class (physics.tooling); the
    values are the means over the member elements of the nominal tool.
    """

    kind: np.ndarray        # [ng] int32 EDGE_*
    rake_rad: np.ndarray    # [ng] mean normal rake
    incl_rad: np.ndarray    # [ng] mean inclination
    clear_rad: np.ndarray   # [ng] mean normal clearance
    edge_radius_m: np.ndarray  # [ng] mean hone radius (nominal)
    chamfer_width_m: np.ndarray  # [ng] T-land width (0 without a chamfer preparation)
    chamfer_angle_rad: np.ndarray  # [ng] T-land angle (negative rake of the land)
    forming: np.ndarray     # [ng] int8: 1 for forming (chipless) elements: thread-forming lobes
    count: np.ndarray       # [ng] int32 member elements


class EdgeTrig(NamedTuple):
    """cos/sin of every element's right-handed angle at spindle angle 0 (alpha = h * psi), so the
    hot loop evaluates positions by angle addition (architecture 5.5)."""

    cos_a: np.ndarray       # [nt, ne]
    sin_a: np.ndarray       # [nt, ne]


class EdgeFrameT(NamedTuple):
    """Unit vectors of every element in T at spindle angle 0 (rotate with the spindle about z_T):
    edge tangent (along increasing profile arc length), chip-thickness normal n (into the
    material), cutting direction (unit cutting velocity of the edge when running in its cutting
    direction) and rake-face normal (outward, toward the oncoming chip)."""

    tangent: np.ndarray     # [nt, ne, 3]
    normal: np.ndarray      # [nt, ne, 3]
    cut_dir: np.ndarray     # [nt, ne, 3]
    rake_normal: np.ndarray  # [nt, ne, 3]


class EdgeState(NamedTuple):
    """Per-macro-step composition inputs of one tool (architecture 4.1), all in the tool frame:
    wear from WearFeedback, thermal growth from ThermalFeedback, pull-out and slip from
    HoldingFeedback. Arrays are [nt, ne] of the tool's EdgeModel."""

    radial_wear_m: np.ndarray      # [nt, ne] recession of the edge away from the material along e_r
    axial_wear_m: np.ndarray       # [nt, ne] recession along +z_T
    edge_radius_m: np.ndarray      # [nt, ne] worn hone radius (<= 0: keep the nominal)
    radial_growth_m: np.ndarray    # [nt, ne] thermal growth along e_r (toward the material)
    scalars: np.ndarray            # [N_STATE] float64, indices STATE_*
    broken: np.ndarray             # [1] int8


# indices into EdgeState.scalars
STATE_LENGTH_GROWTH_M = 0   # tool thermal elongation along its shank direction (tip moves along -shank_dir)
STATE_HOLDER_GROWTH_M = 1   # holder axial growth (rigid shift along -shank_dir)
STATE_PULLOUT_M = 2         # tool pulled out of the holder (rigid shift along -shank_dir)
STATE_SLIP_RAD = 3          # tool rotated relative to the spindle, against the cutting direction (rotating tools)
STATE_FREE_LENGTH_M = 4     # clamp face to tip (thermal strain assumed uniform over it)
STATE_SHANK_DIR_R = 5       # shank direction in (e_r at psi = 0, z_T): rotating tools (0, 1)
STATE_SHANK_DIR_Z = 6
STATE_ROTATING = 7          # 1.0 for rotating tools (slip applies), 0.0 otherwise
N_STATE = 8


class ToolingInputs(NamedTuple):
    """What tooling hands to the compiled loop for a macro step: per controller path, the
    effective EdgeModel of the tool active on that path (frozen for the step), its angle table
    and edge frame, and the slot (-1: no tool, then the tables are empty)."""

    edges: tuple            # (EdgeModel, ...) one per path
    trig: tuple             # (EdgeTrig, ...)
    frames: tuple           # (EdgeFrameT, ...)
    slot: np.ndarray        # [n_paths] int32


CHAIN_TOOL = 0
CHAIN_HOLDER = 1
CHAIN_SPINDLE = 2


def empty_envelope() -> Envelope:
    zero_i = np.zeros(0, np.int32)
    zero_f = np.zeros(0)
    return Envelope(zero_i.copy(), zero_i.copy(), np.zeros(0, np.int8), *(zero_f.copy() for _ in range(12)),
                    chain=zero_i.copy())
