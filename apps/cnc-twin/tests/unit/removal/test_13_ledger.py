"""The ledger volume of the tri-dexel field (docs/models/removal.md 1.1 "Volume", 13; docs/convergence.md 9 item 4):
the kernel against its reference, exactness for planes along the lattice axes, second-order convergence of a thin
layer's removed volume where the z family is first order, the booking of a side cut through the domain (field change
= booked removal, exactly), and the contact geometry of a side cut from step to step (centroid every step, a
steady depth of cut)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.removal import fixtures as FX
from cnctwin.core.abi import EngagedIntervals
from cnctwin.core.bus import PathFrame
from cnctwin.core.channels import ChannelRegistry
from cnctwin.core.domain import RunContext
from cnctwin.core.state import SimState
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_drive as KD
from cnctwin.removal import ledger as LG
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import k_volume as KV
from cnctwin.removal.domain import RemovalDomain
from cnctwin.removal.state import RemovalState, ToolShape


def _layer(lattice, normal, d_hi, thick, lo=(0.0, 0.0, 0.0), hi=(10e-3, 8e-3, 6e-3)):
    """Fields of the block before and after a layer ``thick`` normal to ``normal`` is taken off (n . x <= d)."""
    before = FX.build(lo, hi, lattice, FX.halfspace_box(lo, hi, normal, d_hi))
    after = FX.build(lo, hi, lattice, FX.halfspace_box(lo, hi, normal, d_hi - thick))
    return before, after


def _exact_layer(normal, d_hi, thick, lo=(0.0, 0.0, 0.0), hi=(10e-3, 8e-3, 6e-3)):
    """Removed volume of the layer by dense midpoint integration along the normal's largest component (planes:
    the chord along that axis is linear between the box faces, so 400 x 400 samples are exact to 1e-6)."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    f = int(np.argmax(np.abs(n)))
    u, v = [k for k in range(3) if k != f]
    m = 400
    gu = lo[u] + (np.arange(m) + 0.5) * (hi[u] - lo[u]) / m
    gv = lo[v] + (np.arange(m) + 0.5) * (hi[v] - lo[v]) / m
    U, V = np.meshgrid(gu, gv, indexing="ij")
    total = 0.0
    for d, sgn in ((d_hi, 1.0), (d_hi - thick, -1.0)):
        t = (d - n[u] * U - n[v] * V) / n[f]
        if n[f] > 0:
            length = np.clip(t, lo[f], hi[f]) - lo[f]
        else:
            length = hi[f] - np.clip(t, lo[f], hi[f])
        total += sgn * float(length.sum())
    return total * (hi[u] - lo[u]) * (hi[v] - lo[v]) / (m * m)


def test_kernel_matches_reference_and_reads_a_block_exactly():
    lo, hi = (0.0, 0.0, 0.0), (3e-3, 2.5e-3, 2e-3)
    for lattice, normal, d in ((0.25e-3, (0.3, 0.5, 0.81), 2.1e-3), (0.2e-3, (1.0, -0.4, 0.2), 1.3e-3),
                               (0.25e-3, (0.0, 1.0, 0.0), 1.437e-3)):
        f = FX.build(lo, hi, lattice, FX.halfspace_box(lo, hi, normal, d))
        assert LG.field_volume_dq(f) == R.ledger_volume_ref(f, 0)
    # a block stock: its faces lie on cell faces, the ledger volume is the block's exactly
    from cnctwin.removal.stock import block_field

    f = block_field((3e-3, 2.5e-3, 2e-3), 0.25e-3)
    assert LG.field_volume_m3(f) == pytest.approx(3e-3 * 2.5e-3 * 2e-3, rel=1e-12)
    assert LG.field_volume_dq(f) == R.ledger_volume_ref(f, 0)


def test_one_signature_each():
    for kernel in KV.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_a_plane_along_the_lattice_axes_is_exact_where_the_z_family_is_first_order():
    """A layer 0.3 mm thick off a face normal to y (the finishing pass of docs/convergence.md 2.1): the ledger
    volume is exact at every lattice (to the quantum the face is rounded to), the z family holds whole node rows."""
    exact = 10e-3 * 6e-3 * 0.3e-3
    for lattice in (0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3):
        before, after = _layer(lattice, (0.0, 1.0, 0.0), 7.4137e-3, 0.3e-3)
        removed = LG.field_volume_m3(before) - LG.field_volume_m3(after)
        assert removed == pytest.approx(exact, rel=2e-6), lattice
    # the z family at 0.25 mm: one row of columns (0.25 mm) instead of 0.3 mm
    before, after = _layer(0.25e-3, (0.0, 1.0, 0.0), 7.4137e-3, 0.3e-3)
    z = F.volume_m3(before, 0, 2) - F.volume_m3(after, 0, 2)
    assert abs(z / exact - 1.0) > 0.1


@pytest.mark.parametrize("normal", [(math.sin(0.2), math.cos(0.2), 0.0), (0.3, 0.5, 0.81)])
def test_a_thin_oblique_layer_converges_at_second_order(normal):
    errors = []
    for lattice in (0.5e-3, 0.25e-3, 0.125e-3):
        n = np.asarray(normal) / np.linalg.norm(normal)
        d = float(n @ np.array([5e-3, 4e-3, 3e-3])) + 1.0e-3
        before, after = _layer(lattice, normal, d, 0.3e-3)
        exact = _exact_layer(normal, d, 0.3e-3)
        removed = LG.field_volume_m3(before) - LG.field_volume_m3(after)
        errors.append(abs(removed / exact - 1.0))
    # below 1 % from 0.25 mm (the z family: 2 % at 11 deg, docs/models/removal.md 1.1), and falling with the lattice
    assert errors[1] < 0.01 and errors[2] < 0.005 and errors[2] < errors[0]


# ------------------------------------------------------------------------------------------- the domain, a side cut
R0 = 5e-3
TOP = 6e-3                     # the block's top (z); the tip 1 mm below it


def _project(lattice):
    op = SimpleNamespace(name="OP10", setup=SimpleNamespace(workholding=[]), stock_pose_W_S=np.eye(4),
                         from_previous=None)
    return SimpleNamespace(stock=SimpleNamespace(kind="block", size_m=[12e-3, 8e-3, TOP], mesh=None),
                           material=SimpleNamespace(density_kg_m3=2700.0),
                           physics=SimpleNamespace(lattice_m=lattice, refine=[], polar_theta_bins=720, removal=None),
                           operations=[op], machine=SimpleNamespace(type="vertical_mill"), cad=None, tools=None,
                           stock_pose_W_S=np.eye(4))


def _shape():
    prof = P.profile_from_points([0.0, 20e-3], [R0, R0])
    slabs = A.ToothSlabs(np.array([[0.0, 10e-3, 20e-3]]), np.array([[0.0, 0.0]]), np.array([2], np.int32))
    return ToolShape(slot=0, kind=A.TK_ENVELOPE, rotating=True, centre_cutting=True, hand_sign=-1.0, cut=prof,
                     whole=prof, insert=A.empty_insert(), helix=A.empty_helix(), slabs=slabs, radius_m=R0)


def _passes(inputs, xs, y_axis, tick0):
    """Tooth passes of the D10 cylinder along x at axis y = y_axis, tip 1 mm below the top."""
    f = inputs.fields[0]
    st_z = np.array([0.25e-3, 0.75e-3])
    st_r = np.array([R0, R0])
    eng = EngagedIntervals(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros(2, np.int32), np.zeros(1, np.int64))
    spec = A.subtract_spec([0, 0, 1], np.zeros((2, 3)), [0.0, 10e-3, 20e-3], [0.0, 0.0], [0] * 3, [0] * 3)
    gate = np.array([0.0, 1e-9, 0.0, 0.25e-3])
    stats = np.zeros(4)
    w = np.array([0.0, 0.0, 1.0])
    for k, x in enumerate(xs):
        tip = np.array([x, y_axis, TOP - 1e-3])
        spec.ivals[A.SI_TICK] = tick0 + 100 * k
        spec.ivals[A.SI_EVENT_ID] = inputs.iparams[A.RI_PATH0 + A.RI_PSTRIDE * 1]
        inputs.iparams[A.RI_PATH0 + A.RI_PSTRIDE * 1] += 1
        slab_c = np.ascontiguousarray(np.stack([tip, tip]))
        assert KD.tooth_pass(f, inputs.cut[0], inputs.slabs[0], 0, slab_c, tip, np.array([1.0, 0, 0]),
                             np.array([0, 1.0, 0]), w, st_z, st_r, np.ones(2, np.int8), eng, np.full(2, 0.5e-3), 1.0,
                             0.0, gate, spec, inputs.targets[0], inputs.accum[0], inputs.events, inputs.scratch,
                             stats) == 0


def _side_cut(lattice, ae=0.3e-3, step=0.04e-3, per_macro=3, x0=-6e-3, x1=18e-3):
    """A side step ae wide along the y = 8 mm face, ap 1 mm, the macro steps of ``per_macro`` tooth passes;
    returns (domain, state, per-step contacts, booked volume, ledger change, z-family change)."""
    ctx = RunContext(project=_project(lattice), db=None, run_dir=None, seed=1, tick_s=1e-6, ipo_ticks=1000,
                     macro_ipo=10, channels=ChannelRegistry())
    state = SimState()
    dom = RemovalDomain(tool_shapes={0: _shape()}, n_paths=1)
    dom.init(ctx, state)
    wp = state.removal.workpieces[0]
    v0, z0 = LG.volume_m3(wp), F.volume_m3(wp.field, 0, 2)
    xs = np.arange(x0, x1, step)
    contacts = []
    booked = 0.0
    for m in range(0, len(xs), per_macro):
        state.bus_prev.motion.paths = [PathFrame(tool_slot=0, workpiece=0)]
        inputs = dom.kernel_inputs(ctx, state, state.bus_prev)
        _passes(inputs, xs[m:m + per_macro], 8e-3 + R0 - ae, 1000 * m)
        dom.absorb(ctx, state, None, state.bus_next)
        c = state.bus_next.removal.contacts[0]
        booked += c.removed_volume_m3
        contacts.append((float(xs[min(m + per_macro, len(xs)) - 1]), c))
        dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
        state.swap_bus()
    return dom, state, contacts, booked, v0 - LG.volume_m3(wp), z0 - F.volume_m3(wp.field, 0, 2)


def test_a_side_cut_books_the_field_change_exactly_and_converges():
    exact = 12e-3 * 0.3e-3 * 1e-3            # the path runs a radius beyond both ends of the block
    for lattice in (0.25e-3, 0.125e-3):
        _, state, _, booked, change, zchange = _side_cut(lattice, step=0.2e-3, per_macro=10)
        # the booked removal is the field's own ledger change (integer doubled quanta: exact)
        assert booked == pytest.approx(change, rel=1e-12, abs=1e-18)
        wp = state.removal.workpieces[0]
        assert wp.removed_dq * LG.factor_m3(wp.field) == pytest.approx(change, rel=1e-12)
        # the per-brick cache, updated from the bricks the writers marked, equals the volume from scratch (at 0.125
        # mm the layer crosses the brick boundary at node 64 in y and the path every x boundary)
        assert LG.volume_dq(wp) == LG.field_volume_dq(wp.field)
        # against the swept volume (scallops of 1 um between passes 0.2 mm apart: 1e-3 relative)
        assert booked == pytest.approx(exact, rel=3e-3), lattice
        # the z family held whole rows: 0.25 mm of the 0.3 mm layer at 0.25 mm (-17 %)
        if lattice == 0.25e-3:
            assert abs(zchange / exact - 1.0) > 0.1


def test_side_cut_contact_geometry_is_steady_from_step_to_step():
    """A side step 0.3 mm x 1 mm on a 0.25 mm lattice in macro steps of three tooth passes (0.12 mm of feed): the z
    columns are reached once every two steps. The centroid lies on the removed sliver in every step (it was the S
    origin in the steps without a column) and the depth of cut is 1 mm in every step (0.75 mm, the rows' extent,
    in the steps without a column)."""
    _, _, contacts, _, _, _ = _side_cut(0.25e-3, step=0.04e-3, per_macro=3, x0=-6e-3, x1=6e-3)
    steady = [(x, c) for x, c in contacts if 1e-3 < x < 5e-3]
    assert len(steady) > 20
    for x, c in steady:
        g = c.geometry
        assert g.in_cut
        assert g.ap_m == pytest.approx(1e-3, abs=1e-9), x
        p = np.asarray(g.point_S_m)
        # the sliver: y within the layer (7.7 .. 8 mm), z within the cut (5 .. 6 mm), x ahead of the axis
        assert 7.7e-3 - 0.13e-3 <= p[1] <= 8e-3 + 0.13e-3 and TOP - 1e-3 <= p[2] <= TOP, (x, p)
        assert x - 0.5e-3 < p[0] < x + R0, (x, p)
        n = np.asarray(g.normal_M)
        assert n[1] > 0.5       # the machined surface faces the tool axis (+y)
    ys = np.array([c.geometry.point_S_m[1] for _, c in steady])
    zs = np.array([c.geometry.point_S_m[2] for _, c in steady])
    assert np.ptp(zs) < 0.05e-3 and np.ptp(ys) < 0.15e-3
