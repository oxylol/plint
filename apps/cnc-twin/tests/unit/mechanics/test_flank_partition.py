"""The flank contact's heat partition (docs/models/mechanics.md 6): the friction on the edge arc and the wear land and
the plastic work of the material pressed under the edge, split between tool and work by Blok's equal mean
temperatures over the contact thermal deposits the tool's part on."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, special

from cnctwin.mechanics import abi, coefficients as CF, contact as C, heat as HT, kernels as K, params as P

H_GRID = np.geomspace(2e-6, 1e-3, 24)


@pytest.fixture(scope="module")
def steel_map(steel, carbide):
    return CF.build_process_map(steel, carbide, 0.0, speed_grid_m_s=np.array([1.5, 3.0]),
                                t1_grid_m=np.geomspace(5e-7, 1e-3, 10), tool_temp_grid_K=np.array([293.15, 800.0]))


def test_stagnation_arc_is_the_edge_contact_of_the_force_model():
    for r, depth, rake in ((3e-5, 8e-6, 0.0), (2e-5, 6e-6, math.radians(12)), (2e-5, 3e-5, math.radians(-20)),
                           (1e-5, 1.5e-5, math.radians(-50)), (0.0, 2e-6, 0.1)):
        out = np.zeros(abi.N_PL)
        K.edge_contact_forces(depth, r, depth, rake, 0.0, 1.0, 0.5, out)
        assert C.stagnation_arc_length(r, depth, rake) == pytest.approx(out[abi.PL_CONTACT], rel=1e-12, abs=1e-18)
    arr = C.stagnation_arc_length(np.array([2e-5, 2e-5]), np.array([0.0, 6e-6]), np.zeros(2))
    assert arr[0] == 0.0 and arr[1] == pytest.approx(2e-5 * math.acos(1.0 - 0.3), rel=1e-12)


def test_edge_power_parts_are_the_pressure_and_friction_integrals():
    """F_t = cos i [p h_min + tau r sin w_s cos i] + tau r w_s sin^2 i (mechanics.md 4): the pressure part p h_min cos i is
    the work of the pressure on the pressed material, the rest the friction power over V."""
    k, mu, r, h_min = 5e8, 0.5, 2.5e-5, 7.5e-6
    p, tau = C.contact_stresses(k, mu)
    w_s = math.acos(1.0 - h_min / r)
    for incl in (0.0, math.radians(30)):
        f_p, f_tau, length = C.contact_power_parts(2e-4, r, h_min, 0.1, incl, k, mu)
        ci, si = math.cos(incl), math.sin(incl)
        assert f_p == pytest.approx(p * h_min * ci, rel=1e-12)
        assert f_tau == pytest.approx(tau * r * (math.sin(w_s) * ci * ci + w_s * si * si), rel=1e-12)
        assert length == pytest.approx(r * w_s, rel=1e-12)
        assert f_p + f_tau == pytest.approx(C.edge_forces(r, h_min, 0.1, incl, k, mu)[abi.PL_FT], rel=1e-12)
    # below h_min: the ploughing contact at depth h; indentation: its forces with the wedge-pressure rake face
    f_p, f_tau, _ = C.contact_power_parts(3e-6, r, h_min, 0.1, 0.0, k, mu)
    assert f_p + f_tau == pytest.approx(C.ploughing_forces(3e-6, r, h_min, 0.1, 0.0, k, mu)[abi.PL_FT], rel=1e-12)
    chisel = math.radians(-55)
    f_p, f_tau, _ = C.contact_power_parts(4e-5, r, h_min, chisel, 0.0, k, mu, indentation=True)
    assert f_p + f_tau == pytest.approx(C.indentation_forces(4e-5, r, h_min, chisel, 0.0, k, mu)[abi.PL_FT],
                                        rel=1e-12)


def _layer_1d(beta: float) -> float:
    """Surface rise of a uniform layer source of depth delta in a semi-infinite body with an adiabatic surface after
    time t, over that of the same surface flux (Carslaw & Jaeger 1959: plane sources and their images), beta =
    delta / sqrt(kappa t)."""
    return math.sqrt(math.pi) / (2.0 * beta) * integrate.quad(
        lambda u: special.erf(beta / (2.0 * math.sqrt(u))) if u > 0.0 else 1.0, 0.0, 1.0)[0]


def test_layer_source_factor_limits():
    # a thin layer acts as the surface source; at a low Peclet number the heat reaches the surface in the passage
    assert HT.layer_source_factor(10.0, 1e-4) == pytest.approx(1.0, abs=2e-3)
    assert HT.layer_source_factor(1e-3, 0.2) > 0.95
    # at a high Peclet number every surface point sees the 1-D buried layer during its transit (0 < s < 2l)
    for L, ratio in ((30.0, 0.05), (100.0, 0.2)):
        n = 200
        num = den = 0.0
        for i in range(n):
            s = (i + 0.5) / n * 2.0
            beta = ratio * 2.0 / math.sqrt(s / (2.0 * L))
            num += _layer_1d(beta) * math.sqrt(s)
            den += math.sqrt(s)
        assert HT.layer_source_factor(L, ratio) == pytest.approx(num / den, rel=0.01)
    values = [HT.layer_source_factor(3.0, r) for r in (0.05, 0.2, 0.7, 1.5)]
    assert all(a > b for a, b in zip(values, values[1:]))


def _closed_form_share(steel, carbide, group, slow, h_min, k_ref, mu, t_ws):
    """Blok 1937: T_ws + M_w (q_f - H) + g M_w q_d = T_t + M_t H, written out from the contact's integrals."""
    p, tau = C.contact_stresses(k_ref, mu)
    r = group.edge_radius_m
    w_s = math.acos(1.0 - h_min / r)
    arc = r * w_s
    sigma0, tau0 = C.flank_reference_stresses(k_ref, mu)
    _, tau_land = C.flank_stresses(slow.VB_m, sigma0, tau0)
    v = slow.speed_m_s
    length = slow.VB_m + arc
    q_f = v * (tau * r * math.sin(w_s) + tau_land * slow.VB_m) / length
    q_d = v * p * h_min / length
    width = slow.contact_width_m
    m_w = HT.work_side_resistance(length, width, v, steel, t_ws)
    m_t = HT.tool_side_resistance(length, slow.tool_flank_temp_K, carbide, width, slow.contact_time_s)
    peclet = v * 0.5 * length / (2.0 * steel.diffusivity(t_ws))
    g = HT.layer_source_factor(peclet, P.number(P.CONTACT, "ploughing_zone_depth_ratio") * arc / length)
    heat = (t_ws - slow.tool_flank_temp_K + m_w * (q_f + g * q_d)) / (m_w + m_t)
    return min(max(heat / (q_f + q_d), 0.0), 1.0), length


@pytest.mark.parametrize("vb, tool_K", [(0.0, 400.0), (0.0, 700.0), (1.5e-4, 500.0)])
def test_the_flank_share_matches_bloks_closed_form(steel, carbide, steel_map, vb, tool_K):
    group = CF.EdgeGroup(rake_rad=0.0, edge_radius_m=2.5e-5)
    slow = CF.SlowState(3.0, tool_temp_K=tool_K, tool_flank_temp_K=tool_K, VB_m=vb, contact_width_m=2e-3,
                        contact_time_s=None)
    gc = CF.group_coefficients(steel, carbide, group, slow, H_GRID, steel_map)
    h_min = gc.h_min_m
    k_ref = CF._reference_shear_flow_stress(steel, steel_map, h_min, slow)
    above = np.nonzero(H_GRID > 2.0 * h_min)[0]
    assert above.size > 3
    for k in above[::4]:
        share, length = _closed_form_share(steel, carbide, group, slow, h_min, k_ref, gc.mu_eff, gc.T_shear_K[k])
        assert gc.frac_flank_to_tool[k] == pytest.approx(share, rel=1e-9, abs=1e-12)
        assert gc.flank_contact_m[k] == pytest.approx(length, rel=1e-12)
        assert 0.0 < gc.frac_flank_to_tool[k] < 1.0


def test_cooling_the_tool_never_raises_its_contact_temperature(steel, carbide):
    """At the same load (fluxes, contact, work surface temperature) the tool side of the contact, T_t + M_t H_t, never
    rises when the tool is colder: with the share computed for the heat it is applied to, Blok's contact temperature
    moves with the tool's by M_w / (M_w + M_t) in [0, 1], and with the share clamped at 0 or 1 by 1 or 0."""
    for length, v, q_f, q_d in ((2e-5, 3.0, 4e8, 9e8), (2.4e-6, 0.27, 1.6e8, 6e8), (2.2e-4, 1.0, 2e8, 0.0)):
        t_ws = 900.0
        prev = -math.inf
        for t_t in np.linspace(293.15, 1400.0, 60):
            share, info = HT.flank_heat_to_tool(steel, carbide, length, 2e-3, v, q_f, q_d, 0.7 * min(length, 2e-5),
                                                t_ws, float(t_t), None)
            contact = t_t + info["M_t"] * share * (q_f + q_d)
            assert contact >= prev - 1e-9
            prev = contact


def test_harness_contact_follows_the_tool_temperature(steel, carbide):
    """contact.py mirrors the run's partition: its flank contact temperature (the tool side over the contact the
    share was computed for) never falls when the tool body warms at the same cut (it rose when the tool was cooled
    while the share of the friction flux was applied to the whole edge power over a different length)."""
    from cnctwin.validation.harnesses.contact import ContactModel
    model = ContactModel(steel, carbide, rake_rad=math.radians(5), clearance_rad=math.radians(7),
                         edge_radius_m=2e-5, work_temp_K=293.15, speeds_m_s=[2.0], h_values_m=[8e-5])
    prev = -math.inf
    for body in np.linspace(293.15, 1100.0, 25):
        st = model.state(8e-5, 2.0, float(body), 2e-3, 2e-3)
        assert st.flank_temp_K >= prev - 1e-6
        prev = st.flank_temp_K
        assert st.flank_contact_m == pytest.approx(C.stagnation_arc_length(2e-5, st.h_min_m, math.radians(5)),
                                                   rel=1e-12)


def test_thermal_deposits_the_flank_heat_over_the_contact_the_share_was_computed_for(steel, carbide, steel_map):
    """The run (thermal.partition, from the bus record) and the share (mechanics) use one flank contact: the wear land
    plus the edge arc below the stagnation point at min(h, h_min)."""
    from cnctwin.thermal.partition import FractionTable
    group = CF.EdgeGroup(rake_rad=math.radians(6), edge_radius_m=2.5e-5)
    vb = 8e-5
    gc = CF.group_coefficients(steel, carbide, group, CF.SlowState(3.0, tool_temp_K=500.0, VB_m=vb), H_GRID,
                               steel_map)
    pc = CF.process_coefficients([gc], tool_slot=0, version=1,
                                 flank_coeff_N_per_m2=np.zeros((1, H_GRID.size)))
    table = FractionTable.from_coefficients(pc)
    h = H_GRID[[2, 8, 16, 20]]
    shape = (1, h.size)
    lengths = table.flank_length(np.zeros(shape, np.int32), h.reshape(shape), np.full(shape, 2.5e-5),
                                 np.full(shape, math.radians(6)), np.full(shape, vb))
    assert np.allclose(lengths[0], gc.flank_contact_m[[2, 8, 16, 20]], rtol=1e-12)
