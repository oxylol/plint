"""Thin walls: the Ritz plate against plate solutions (Leissa's cantilever plate, simply supported and clamped
squares, cylindrical bending), wall detection on a pocketed block, and the blend into the published modes and
compliance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.structure import structfix as fx
from cnctwin.structure import thinwall as TW
from cnctwin.structure import work as WK
from cnctwin.structure.lathe import LatheSupports
from cnctwin.structure.materials import work_material
from cnctwin.structure.protocols import BoxGeometry
from cnctwin.structure.schema import settings
from cnctwin.structure.voxel import VoxelGrid, supports_from_holding

E, NU, RHO, T, A = 70e9, 0.3, 2700.0, 0.002, 0.05


def lam_of(plate, omega):
    return omega * A * A * math.sqrt(RHO * T / plate.D_Nm)


def test_plate_frequencies_against_plate_solutions():
    cfff = TW.Plate(A, A, T, E, NU, RHO, ("C", "F", "F", "F"), 8)
    w, _ = cfff.modes()
    # Leissa 1969: 3.492 (a Ritz value, an upper bound); the converged value is 3.471
    assert 3.46 < lam_of(cfff, w[0]) < 3.4917
    assert lam_of(cfff, w[1]) == pytest.approx(8.525, rel=0.01)
    ssss = TW.Plate(A, A, T, E, NU, RHO, ("S", "S", "S", "S"), 6)
    w, _ = ssss.modes()
    assert lam_of(ssss, w[0]) == pytest.approx(2 * math.pi ** 2, rel=1e-6)
    cccc = TW.Plate(A, A, T, E, NU, RHO, ("C", "C", "C", "C"), 8)
    w, Y = cccc.modes()
    assert lam_of(cccc, w[0]) == pytest.approx(35.985, rel=1e-4)
    # mass normalisation: rho t int w^2 dA = 1
    assert np.allclose(np.einsum("ij,ij->j", Y, cccc.M @ Y), 1.0)


def test_cylindrical_bending_compliance():
    """A wide cantilever plate under a line load along its free edge bends as a strip of stiffness D (within the
    free side edges' anticlastic effect): mean deflection a^3 / (3 D b), above it by less than 2 %."""
    b = 4 * A
    plate = TW.Plate(A, b, T, E, NU, RHO, ("C", "F", "F", "F"), 8)
    v = np.linspace(0.0, b, 81)
    u = np.full_like(v, A)
    d = plate.deflection(u, v, u, v, np.full(len(v), 1.0 / len(v)))
    cyl = A ** 3 / (3 * plate.D_Nm * b)
    assert cyl < d.mean() < 1.02 * cyl
    # point compliance: symmetric in the plate and largest at the free corner
    c = plate.compliance([A, A, 0.5 * A], [0.0, 0.5 * b, 0.5 * b])
    assert c[0] > c[1] > c[2] > 0


def test_ritz_functions_are_orthonormal_and_satisfy_the_edges():
    for left, right in (("C", "F"), ("S", "S"), ("F", "F"), ("C", "C")):
        fs = TW.ritz_functions(left, right, 6)
        x, w = np.polynomial.legendre.leggauss(40)
        x = 0.5 * (x + 1)
        w = 0.5 * w
        G = np.array([[np.sum(w * f(x) * g(x)) for g in fs] for f in fs])
        assert np.allclose(G, np.eye(6), atol=1e-10)
        for f in fs:
            if left in "CS":
                assert abs(f(0.0)) < 1e-12
            if left == "C":
                assert abs(f.deriv(1)(0.0)) < 1e-10


def pocketed(wall_mm=1.2):
    geo = BoxGeometry([0, 0, 0], [0.03, 0.02, 0.012])
    w = wall_mm * 1e-3
    geo.remove_box([w, 0.004, 0.004], [0.03 - w, 0.016, 0.013])
    return geo


def test_wall_detection_on_a_pocketed_block():
    geo = pocketed()
    st = settings({})
    grid = VoxelGrid((0.0, 0.0, 0.0), 0.002, (15, 10, 6))
    fill = geo.fill_fraction(0, grid.origin, grid.cell_m, grid.shape)
    walls = TW.detect_walls(fill, fill > 0.5, grid.origin, grid.cell_m, st, geo, 0)
    x_walls = [w for w in walls if w.axis == 0]
    assert len(x_walls) == 2
    for wall in x_walls:
        assert wall.thickness_m == pytest.approx(1.2e-3, rel=1e-3)      # removal's wall thickness query
        assert np.allclose(wall.weight, 1.0)                            # 0.6 cell thick: fully analytic
        u_ax, v_ax = wall.plate_axes
        assert (u_ax, v_ax) == (1, 2)
        assert wall.edges == ("C", "C", "C", "F")                       # ends and floor clamped, top free
        assert wall.v_range_m[0] == pytest.approx(0.004) and wall.v_range_m[1] == pytest.approx(0.012)
        assert wall.u_range_m == pytest.approx((0.004, 0.016))
        TW.solve_wall(wall, grid.origin, grid.cell_m, E, NU, RHO, st)
        plate = TW.Plate(0.012, 0.008, 1.2e-3, E, NU, RHO, ("C", "C", "C", "F"), st.wall_terms)
        assert wall.omega_rad_s[0] == pytest.approx(plate.modes()[0][0], rel=1e-12)
        top = wall.cells[:, 2] == wall.cells[:, 2].max()
        assert wall.compliance_m_per_N[top].max() > wall.compliance_m_per_N[~top].max()


def test_blend_into_the_workpiece_modes():
    geo = pocketed()
    st = settings({"max_dof": 4000, "min_cell_mm": 2.0})
    proj = fx.project(stock={"kind": "block", "size_mm": [30, 20, 12], "form": "rolled_plate"})
    mat = work_material(proj.material)
    sup = supports_from_holding(fx.vise_supports((0.03, 0.02, 0.012), grip_z=(0.001, 0.003)))
    grid = VoxelGrid((0.0, 0.0, 0.0), 0.002, (15, 10, 6))
    inputs = WK.WorkInputs(0, grid, mat, st, geo, BoxGeometry([0, 0, 0], [0.03, 0.02, 0.012]), "rolled_plate",
                           np.zeros(3), np.array([0.03, 0.02, 0.012]), "voxel", sup, LatheSupports(), True, None)
    sol = WK.compute(inputs, None, None)
    wm = sol.modes
    assert wm.n_wall_modes == 2 * st.wall_modes
    n_fe = len(wm.freq_Hz) - wm.n_wall_modes
    wall_shapes = wm.shape[n_fe:]
    # wall modes act along the wall normal (x) and only on the wall cells (x index 0 or 14)
    assert np.all(wall_shapes[..., 1:] == 0.0)
    nonzero = np.argwhere(np.any(wall_shapes[..., 0] != 0.0, axis=0))
    assert set(nonzero[:, 0].tolist()) <= {0, 14}
    # the thin wall's top is far more compliant than the solid end of the block
    top_wall = wm.static_compliance[0, 5, 5, 0, 0]
    solid = wm.static_compliance[7, 1, 2, 0, 0]
    assert top_wall > 10 * solid
    assert np.all(wm.freq_Hz[n_fe:] > 0) and np.all(wm.zeta > 0)


def test_clamped_edge_inside_a_cell():
    """A pocket floor at 5 mm with 2 mm cells: the wall's root lies inside the transition cell; its fill
    f = h + (1 - h) T / cell gives the root at 5 mm exactly."""
    geo = BoxGeometry([0, 0, 0], [0.03, 0.02, 0.012])
    geo.remove_box([1.2e-3, 0.004, 0.005], [0.03 - 1.2e-3, 0.016, 0.013])
    grid = VoxelGrid((0.0, 0.0, 0.0), 0.002, (15, 10, 6))
    fill = geo.fill_fraction(0, grid.origin, grid.cell_m, grid.shape)
    walls = [w for w in TW.detect_walls(fill, fill > 0.5, grid.origin, grid.cell_m, settings({}), geo, 0)
             if w.axis == 0]
    assert walls and all(w.v_range_m[0] == pytest.approx(0.005, abs=1e-12) for w in walls)
    assert all(w.v_range_m[1] == pytest.approx(0.012) for w in walls)
