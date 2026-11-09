"""Long turned parts as beams: a shaft in a stiff chuck against the Euler-Bernoulli cantilever, with a tailstock
against the clamped-pinned beam, turned steps, the support selection, and the thermal growth on the grid."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.structure import structfix as fx
from cnctwin.structure import lathe as LA
from cnctwin.structure import work as WK
from cnctwin.structure.materials import work_material
from cnctwin.structure.protocols import CylinderGeometry
from cnctwin.structure.schema import settings
from cnctwin.structure.voxel import Supports, choose_grid

D, L = 0.02, 0.2
CHUCK = {"z_face_S_m": 0.02, "grip_centre_S_m": 0.01, "grip_length_m": 0.02, "radial_stiffness_N_per_m": 0.0,
         "tilt_stiffness_Nm_per_rad": 0.0, "jaws": 3}            # no stiffness given: a rigid grip


def _shear_drop(beta, Lf, mat):
    """First-order Timoshenko drop of a bending frequency (shear and rotary inertia, r = D / 4)."""
    from cnctwin.structure import beam as B
    kG = B.cowper_circle(mat.nu) * B.shear_modulus(mat.E_Pa, mat.nu)
    return 0.5 * (beta * D / 4.0 / Lf) ** 2 * (1.0 + mat.E_Pa / kG)


def shaft(lathe):
    geo = CylinderGeometry(0.5 * D, L, samples=4)
    mat = work_material(fx.material(fx.STEEL))
    model = LA.build_lathe(geo, 0, *geo.stock_bounds_S(0), mat, LA.LatheSupports.from_holding(lathe),
                           settings({"lathe_radial_cells": 12}))
    return geo, mat, model


def test_chucked_shaft_against_the_cantilever():
    geo, mat, model = shaft({"chuck": CHUCK})
    assert any("rigid" in n for n in model.notes)
    lam, V = LA.lathe_modes(model, 6)
    f = np.sqrt(lam) / (2 * math.pi)
    I = math.pi * D ** 4 / 64
    A = math.pi * D * D / 4
    Lf = L - 0.02
    f_eb = 1.875104069 ** 2 / (2 * math.pi * Lf * Lf) * math.sqrt(mat.E_Pa * I / (mat.rho_kg_m3 * A))
    # Timoshenko lowers it by 0.5-1.3 % (first-order estimate c below), the sampled section raises I by 0.2 %
    c = _shear_drop(1.875104069, Lf, mat)
    assert 1.0 - 2.5 * c < f[0] / f_eb < 1.0 - 0.5 * c + 0.002
    assert f[1] == pytest.approx(f[0], rel=1e-8)
    # the sampled section: area and second moment within the cell sampling error
    assert np.median(model.slices.A_m2) == pytest.approx(A, rel=5e-3)
    assert np.median(0.5 * (model.slices.Ixx_m4 + model.slices.Iyy_m4)) == pytest.approx(I, rel=1e-2)


def test_tailstock_makes_a_clamped_pinned_beam():
    lathe = {"chuck": CHUCK, "tailstock": {"z_S_m": L, "radial_stiffness_N_per_m": 0.0}}
    geo, mat, model = shaft(lathe)
    lam, _ = LA.lathe_modes(model, 2)
    I = math.pi * D ** 4 / 64
    A = math.pi * D * D / 4
    Lf = L - 0.02
    f_cp = 3.926602 ** 2 / (2 * math.pi * Lf * Lf) * math.sqrt(mat.E_Pa * I / (mat.rho_kg_m3 * A))
    c = _shear_drop(3.926602, Lf, mat)
    assert 1.0 - 2.5 * c < math.sqrt(lam[0]) / (2 * math.pi) / f_cp < 1.0 - 0.5 * c + 0.002
    assert any("rigid pin" in n for n in model.notes)


def test_chuck_stiffness_softens_the_shaft():
    def f1(k_rad, k_tilt):
        ch = dict(CHUCK, radial_stiffness_N_per_m=k_rad, tilt_stiffness_Nm_per_rad=k_tilt)
        return LA.lathe_modes(shaft({"chuck": ch})[2], 1)[0][0]
    rigid = f1(0.0, 0.0)
    stiff = f1(1e10, 1e6)
    soft = f1(4.5e8, 1e4)
    assert soft < stiff < rigid
    # a steady rest at mid-length (three rollers) and a tailstock raise it
    steady = {"chuck": CHUCK, "steady": [{"z_S_m": 0.11, "rollers": 3, "stiffness_N_per_m": 2e8}]}
    assert LA.lathe_modes(shaft(steady)[2], 1)[0][0] > rigid


def test_turned_groove_lowers_the_frequency():
    """A groove near the root (where the bending moment is largest) lowers f1; thinning the outer half would
    raise it (less mass at the tip)."""
    _, _, full = shaft({"chuck": CHUCK})
    geo = CylinderGeometry(0.5 * D, L, samples=4)
    geo.turn(lambda z: np.where((z > 0.03) & (z < 0.06), 0.006, 0.01))
    mat = work_material(fx.material(fx.STEEL))
    turned = LA.build_lathe(geo, 0, *geo.stock_bounds_S(0), mat, LA.LatheSupports.from_holding({"chuck": CHUCK}),
                            settings({}))
    assert LA.lathe_modes(turned, 1)[0][0] < LA.lathe_modes(full, 1)[0][0]
    L_free, Dm = LA.free_length(*geo.stock_bounds_S(0), LA.LatheSupports.from_holding({"chuck": CHUCK}))
    assert L_free == pytest.approx(0.18) and Dm == pytest.approx(D)


def test_beam_solution_on_the_grid():
    geo = CylinderGeometry(0.5 * D, L, samples=4)
    mat = work_material(fx.material(fx.STEEL))
    lo, hi = geo.stock_bounds_S(0)
    st = settings({"max_dof": 6000})
    grid = choose_grid(lo, hi, st.max_dof, st.min_cell_m)
    dT = 25.0
    th = fx.thermal_grid(lo, hi, 2e-3, lambda c: 293.15 + dT + 0.0 * c[..., 0])
    inputs = WK.WorkInputs(0, grid, mat, st, geo, geo, "cold_drawn_bar", lo, hi, "beam", Supports.empty(),
                           LA.LatheSupports.from_holding({"chuck": CHUCK}), True, th, cylinder=True)
    sol = WK.compute(inputs, None, None)
    assert sol.kind == "beam" and sol.modes.model == "beam" and sol.modes.shape.shape[1:4] == grid.shape
    d = sol.displacement.thermal_m
    alpha = 11.5e-6
    # axial growth from the chuck's axial stop and radial expansion of the section
    k_end = grid.node_shape[2] - 1
    z_end = min(grid.origin[2] + k_end * grid.cell_m, L)       # the grid may reach past the part: held there
    centre = (grid.node_shape[0] // 2, grid.node_shape[1] // 2)
    assert d[centre[0], centre[1], k_end, 2] == pytest.approx(alpha * dT * (z_end - 0.02), rel=0.02)
    x_edge = grid.origin[0] + (grid.node_shape[0] - 1) * grid.cell_m
    assert d[-1, centre[1], k_end, 0] == pytest.approx(alpha * dT * x_edge, rel=0.05)
    # modes: the first pair bends along x then y (canonical), residual compliance positive at the free end
    s0 = sol.modes.shape[0]
    assert abs(s0[..., 0]).max() > 1e3 * abs(s0[..., 1]).max()
    assert sol.modes.static_compliance[centre[0], centre[1], -1, 0, 0] > 0
