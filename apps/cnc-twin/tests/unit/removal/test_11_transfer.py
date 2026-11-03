"""Transfers between operations: exact lattice remaps (flips by multiples of 90 deg, sub-spindle), resampling
for other angles, the polar field's sub-spindle transfer."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_query as KQ
from cnctwin.removal import k_subtract as KS
from cnctwin.removal import k_transfer as KT
from cnctwin.removal import polar as PO
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S
from cnctwin.removal import transfer as TR

LAT = 0.25e-3


def _pocket(f, c, radius, depth_to):
    prof = P.profile_from_points([0.0, 20e-3], [radius, radius])
    KS.subtract_envelope(f, prof, A.subtract_spec([0, 0, 1], [[c[0], c[1], depth_to]], [0.0, 20e-3], [0.0],
                                                  [-1] * 3, [1] * 3),
                         A.empty_target(int(f.info[A.FI_NRAYS])), A.new_accum(1), event_buffer(4, A.EVENT_WIDTH),
                         A.make_scratch())


@pytest.mark.parametrize("axis,deg", [("X", 180), ("Y", 90), ("Z", -90), ("Y", 180)])
def test_lattice_flips_are_exact(axis, deg):
    f = S.block_field([12e-3, 8e-3, 6e-3], LAT)
    _pocket(f, (4e-3, 3e-3), 2e-3, 3e-3)
    R_ = TR.transform_of({"flip": {"axis": axis, "angle_deg": deg}}, "block")
    perm_sign = TR.signed_permutation(R_)
    assert perm_sign is not None
    new, T, note = TR.apply(f, R_, "block", LAT)
    # exact: every new family's integer length is the old family's it was mapped from
    from cnctwin.removal.k_ray import family_length

    perm, sign = perm_sign
    for fam_n in range(3):
        assert family_length(new, 0, fam_n) == family_length(f, 0, int(perm[fam_n]))
    assert F.volume_m3(new, 0, 2) == F.volume_m3(f, 0, int(perm[2]))
    ref = R.remap_ref(f, new.lv_dims[0], perm, sign)
    for (fam_n, iu, iv), segs in list(ref.items())[::23]:
        got = [(a, b) for a, b, _, _ in R.ray_segments(new, R.ray_index(new, 0, fam_n, iu, iv))]
        assert got == segs
    lo, hi = new.lv_box_lo_m[0], new.lv_box_hi_m[0]
    assert np.allclose(lo, 0.0, atol=1e-12)
    # the pocket moved with the material: a point of the pocket maps through T
    p_old = np.array([4e-3, 3e-3, 4.5e-3])
    p_new = T[:3, :3] @ p_old + T[:3, 3]
    out_old = np.zeros(1, np.int8)
    out_new = np.zeros(1, np.int8)
    KQ.points_inside(f, np.ascontiguousarray(p_old[None]), out_old)
    KQ.points_inside(new, np.ascontiguousarray(p_new[None]), out_new)
    assert out_old[0] == 0 and out_new[0] == 0


def test_sub_spindle_transfer_of_tube_and_polar():
    f = S.tube_field(10e-3, 12e-3, LAT, inner_diameter_m=4e-3)
    R_ = TR.transform_of({"transfer": "sub_spindle"}, "tube")
    new, T, note = TR.apply(f, R_, "tube", LAT)
    assert F.volume_m3(new, 0, 2) == pytest.approx(F.volume_m3(f, 0, 2), rel=1e-12)
    assert np.allclose(new.lv_box_lo_m[0], [-5e-3, -5e-3, 0.0], atol=1e-12)
    n_th, dz, nz, dr, nr, _ = PO.resolution(12e-3, 360, 0.2e-3, 0.4e-3, 5e-3)
    pf = PO.polar_tube(5e-3, 2e-3, 0.0, 12e-3, n_th, dz, dr)
    # mark one row of one bin so the mirror can be checked
    nz_ = int(pf.geom[A.PG_NZ])
    pf.rseg[5 * nz_ + 10, 0, 1] = int(round(4.5e-3 / A.QUANTUM_M))
    out = TR.polar_transfer(pf, R_, 12e-3)
    b_new = (n_th // 2 - 1 - 5) % n_th
    assert out.rseg[b_new * nz_ + (nz_ - 1 - 10), 0, 1] == int(round(4.5e-3 / A.QUANTUM_M))
    assert PO.volume_m3(out) == pytest.approx(PO.volume_m3(pf), rel=1e-12)


def test_other_angles_resample_through_the_mesh():
    f = S.block_field([8e-3, 6e-3, 4e-3], LAT)
    R_ = TR.rotation(2, math.radians(30))
    new, T, note = TR.apply(f, R_, "block", LAT)
    assert "resampled" in note
    assert F.volume_m3(new, 0, 2) == pytest.approx(F.volume_m3(f, 0, 2), rel=0.03)


def test_identity_keeps_the_field():
    f = S.block_field([4e-3, 4e-3, 4e-3], LAT)
    new, T, note = TR.apply(f, None, "block", LAT)
    assert new is f and np.allclose(T, np.eye(4))
    assert TR.transform_of({"transfer": "pallet"}, "block") is None


def test_one_signature_each():
    for kernel in KT.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
