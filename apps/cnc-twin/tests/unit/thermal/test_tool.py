"""Tool and holder ladder: fin solution, growth, layout, edge outputs."""

import math

import numpy as np
import pytest

from cnctwin.db.simap import SIMap
from cnctwin.thermal import kernels as K
from cnctwin.thermal import props
from cnctwin.thermal.abi import N_LD, Ladder
from cnctwin.thermal.partition import ContactHeat
from cnctwin.thermal.schema import ThermalSettings
from cnctwin.thermal.toolmodel import ToolEnvironment, ToolThermal, taper_dims, tool_spec

import fixtures as F


def test_ladder_steady_state_is_the_fin_solution():
    """Rod with tip heat input Q, lateral h, adiabatic end: theta(x) = Q cosh(m(L-x)) / (k A m sinh(mL))."""
    k, rho, cp = 50.0, 7800.0, 500.0
    mat = props.mat_table({"value": rho}, {"value": k}, {"value": cp}, {"value": 1e-5}, "rod")
    n, length, diameter, h, q = 200, 0.1, 0.01, 40.0, 5.0
    dz = length / n
    area = math.pi * diameter ** 2 / 4
    perim = math.pi * diameter
    lad = Ladder(m=np.full(n, rho * area * dz), h=np.zeros(n), T=np.full(n, 293.15), mat=np.zeros(n, np.int8),
                 area=np.full(n, area), dz=np.full(n, dz), link_a=np.arange(n - 1, dtype=np.int32),
                 link_b=np.arange(1, n, dtype=np.int32), link_kind=np.zeros(n - 1, np.int8), link_g=np.zeros(n - 1),
                 g_air=np.full(n, h * perim * dz), g_cool=np.zeros(n), g_sp=np.zeros(n), src=np.zeros(n),
                 band=np.zeros((n, 3)), rhs=np.zeros(n), scal=np.array([293.15, 293.15, 293.15, 0.0]))
    out = np.zeros(N_LD)
    for _ in range(40):
        lad.src[:] = 0.0
        lad.src[0] = q * 1e4
        K.ladder_step(lad, mat, mat, 1e4, out)
        assert out[6] - out[5] == pytest.approx(out[0] - out[1] - out[2] - out[3], rel=1e-9, abs=1e-6)
    m = math.sqrt(h * perim / (k * area))
    x = (np.arange(n) + 0.5) * dz
    analytic = q * np.cosh(m * (length - x)) / (k * area * m * math.sinh(m * length))
    assert np.allclose(lad.T - 293.15, analytic, rtol=0.02)


def make_tool():
    spec = tool_spec({"stickout_m": 0.035}, SIMap(F.TOOL_RAW), SIMap(F.HOLDER_RAW))
    mt = props.tool_material_table_by_id(spec.tool_material_id)
    mh = props.material_table_by_id(spec.holder_material_id)
    return ToolThermal(0, spec, mt, mh, 1e-3, 293.15, 0.0), spec


def test_layout_masses_and_taper():
    tool, spec = make_tool()
    assert spec.stickout_m == pytest.approx(0.035)
    assert spec.flutes == 4 and spec.fluted_area_m2 == pytest.approx(math.pi * 0.01 ** 2 / 4 - 4 * 9.5e-6)
    lad = tool.ladder
    assert lad.m[tool.layout.holder_nodes].sum() == pytest.approx(1.6, rel=1e-12)   # catalogue holder mass
    assert np.max(np.abs(lad.link_b - lad.link_a)) <= 2
    area, volume, mean = taper_dims("SK40")
    assert mean == pytest.approx(0.5 * (0.04445 + 0.04445 - 7 / 24 * 0.0684))
    assert area == pytest.approx(math.pi * mean * 0.0684)


def test_uniform_heating_gives_free_expansion():
    tool, spec = make_tool()
    mat = tool.mat_tool
    for p in tool.layout.tool_nodes:
        tool.ladder.T[p] = 393.15
    length, holder = tool.growths()
    alpha = float(np.interp(393.15, mat.T_K, mat.alpha_1_K)) * 100.0 - \
        float(np.interp(293.15, mat.T_K, mat.alpha_1_K)) * 0.0
    assert length == pytest.approx(alpha * spec.stickout_m, rel=1e-9)
    assert holder == 0.0


def test_edge_heat_outputs_and_ledger():
    tool, spec = make_tool()
    edges = F.edge_model()
    nt, ne = edges.elem_z_m.shape
    heat = ContactHeat(0, 0, 0)
    heat.tool_rake_elem_J = np.full((nt, ne), 0.004)
    heat.tool_flank_elem_J = np.full((nt, ne), 0.001)
    heat.h_mean_m = np.full((nt, ne), 4e-5)
    heat.compression = np.full((nt, ne), 2.5)
    in_cut = np.full((nt, ne), 3e-3)
    entries = np.full((nt, ne), 2.0)
    env = ToolEnvironment(in_spindle=True, omega_rad_s=800.0, T_air_K=293.15, T_end_K=293.15)
    settings = ThermalSettings(edge_every_n_macro=1)
    for _ in range(50):
        tool.deposit(edges.elem_z_m, heat.tool_rake_elem_J + heat.tool_flank_elem_J)
        tool.step(0.01, env, 1e-9)
        out = tool.edge_heat(edges, heat, in_cut, entries, None, 0.01, settings)
    assert out.edge_temp_q_K.shape == (nt, ne, 3)
    assert out.q_weights.sum() == pytest.approx(1.0)
    assert np.all(out.edge_peak_q_K >= out.rake_temp_mean_K[:, :, None] - 1e-9)
    assert np.all(out.edge_temp_q_K > out.body_temp_at_elem_K[:, :, None])
    assert np.all(out.edge_cycle_amp_K > 0.0)
    assert out.length_growth_m > 0.0
    assert tool.ledger.worst_rel < 1e-12
    # temperature rises through the heating phase
    assert np.all(np.diff(out.edge_peak_q_K[0, 0]) > 0.0)
