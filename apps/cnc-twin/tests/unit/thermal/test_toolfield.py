"""3-D cutting-region field (toolfield.py, kernels.field_step): against the dense reference, exact
solutions of wedges and half spaces under rectangular fluxes, energy, coupling to the ladder, the
periodic deviation of the edge kernel, the domain path and the point API."""

import math

import numpy as np
import pytest
from scipy import integrate

from cnctwin.core.bus import Bus
from cnctwin.db.models import params
from cnctwin.thermal import kernels as K
from cnctwin.thermal import props
from cnctwin.thermal import reference as R
from cnctwin.thermal.abi import (EDGE_INTERRUPTED, EDGE_OUT_DEVIATION, FD_AIR, FD_COOL, FD_COUPLE, FD_DU, FD_SP,
                                 FD_SRC, FD_U0, FD_U1, N_FD, new_edge_input, new_edge_output)
from cnctwin.thermal.domain import ThermalDomain
from cnctwin.thermal.response import edge_response, gauss_legendre, heating_phase_nodes
from cnctwin.thermal.schema import ThermalSettings
from cnctwin.thermal.toolfield import FACE_SEAT, ring_edges, unroll_elements
from cnctwin.thermal.toolmodel import ToolEnvironment, ToolSpec, ToolThermal

import fixtures as F

RHO, K_W, CP = 15000.0, 80.0, 200.0
ALPHA = K_W / (RHO * CP)
GLX, GLW = gauss_legendre(24)
AIR = ToolEnvironment(T_air_K=293.15, T_end_K=293.15)


def const_material():
    return props.mat_table({"value": RHO}, {"value": K_W}, {"value": CP}, {"value": 5e-6}, "constant")


def wedge_tool(beta, settings, kind="tooth", contact_edges=()):
    """A large tool whose cutting region is an unclipped wedge (tooth) or an insert section, constant properties."""
    mat = const_material()
    if kind == "insert":
        spec = ToolSpec(kind="turning", tool_material_id="x", insert_ic_m=0.04, insert_thickness_m=0.03,
                        bar_h_m=0.03, bar_b_m=0.03, bar_overhang_m=0.01)
    else:
        spec = ToolSpec(kind="rotating", tool_material_id="x", diameter_m=0.06, flutes=1, flute_length_m=0.05,
                        overall_length_m=0.1, shank_diameter_m=0.06, stickout_m=0.06, core_diameter_m=0.02,
                        fluted_area_m2=math.pi * 0.03 ** 2, fluted_perimeter_m=math.pi * 0.06,
                        holder_gauge_length_m=0.05, holder_nose_diameter_m=0.08, holder_clamp_length_m=0.03,
                        holder_mass_kg=2.0, taper="SK40")
    tool = ToolThermal(0, spec, mat, mat, 2e-3, 293.15, 0.0)
    assert tool.enable_field(settings, wedge_angle_rad=beta, edge_radius_m=1e-4, contact_edges_m=contact_edges)
    return tool


def image_rise(q, l, y0, y1, beta, n, t, points):
    """Exact rise in a wedge beta = pi / n (adiabatic faces) and its end face y = 0 under a flux q on
    [0, l] x [y0, y1] of the rake face: n full-space rectangle sources of strength 2q on the rays
    2 k beta (method of images, Carslaw & Jaeger 1959 ch. X) and their mirror images across y = 0, each
    evaluated by the moving-source kernel at v = 0 (validated in test_jaeger.py)."""
    out = np.zeros(len(points))
    for i, (x, y, z) in enumerate(points):
        total = 0.0
        for kk in range(n):
            th = 2.0 * kk * beta
            u = x * math.cos(th) + z * math.sin(th)
            w = abs(-x * math.sin(th) + z * math.cos(th))
            for yc in (0.5 * (y0 + y1), -0.5 * (y0 + y1)):
                total += K.moving_rect_rise(q, 0.5 * l, 0.5 * (y1 - y0), 0.0, K_W, ALPHA, t, u - 0.5 * l, y - yc, w,
                                            GLX, GLW)
        out[i] = total
    return out


def strip_mean_exact(q, l, y0, y1, beta, n, t):
    gx, gw = gauss_legendre(8)
    pts = [(0.5 * l * (a + 1.0), y0 + 0.5 * (y1 - y0) * (c + 1.0), 0.0) for a in gx for c in gx]
    vals = image_rise(q, l, y0, y1, beta, n, t, pts).reshape(gx.size, gx.size)
    return float(0.25 * np.einsum("i,ij,j->", gw, vals, gw))


def heat_rake_strip(tool, q, l, b, y0, dt, t_end):
    t = 0.0
    while t < t_end - 1e-12:
        tool.deposit_contact(q * l * b * dt, 0.0, l, 1e-6, b, y0)
        tool.step(dt, AIR, 1e-9)
        t += dt


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("dt,t_end", [(0.01, 0.05), (0.25, 2.0)])
def test_wedge_matches_image_solution(n, dt, t_end):
    """Wedges of 180, 90, 60 and 45 deg under a rake-face rectangle flux adjacent to the edge: the field
    (default mesh) against the exact image solution, at the edge, on the rake face in and beyond the
    contact, inside the wedge 1-2 mm away and beside the contact; and the contact mean."""
    beta = math.pi / n
    tool = wedge_tool(beta, ThermalSettings(tool_field_length_m=0.04))
    q, l, b, y0 = 5e7, 2e-4, 2e-3, 5e-3
    heat_rake_strip(tool, q, l, b, y0, dt, t_end)
    yc = y0 + 0.5 * b
    mid = -0.5 * beta
    pts = [(0.0, yc, 0.0), (0.5 * l, yc, 0.0), (l, yc, 0.0), (2 * l, yc, 0.0), (1e-3, yc, 0.0), (2e-3, yc, 0.0),
           (1e-3 * math.cos(mid), yc, 1e-3 * math.sin(mid)), (2e-3 * math.cos(mid), yc, 2e-3 * math.sin(mid)),
           (5e-4, y0, 0.0), (1e-3, y0 - 1e-3, 0.0)]
    sim = tool.temperature_at(np.array(pts)) - 293.15
    exact = image_rise(q, l, y0, y0 + b, beta, n, t_end, pts)
    assert np.all(np.abs(sim / exact - 1.0) < 0.045), np.round(sim / exact - 1.0, 4)
    rake, _, _ = tool.field.strip_means([l], [1e-6], [y0], [y0 + b], [0])
    mean = strip_mean_exact(q, l, y0, y0 + b, beta, n, t_end)
    assert rake[0] - 293.15 == pytest.approx(mean, rel=0.08)        # lateral resolution of the contact ends


def test_mesh_refinement_converges_to_the_image_solution():
    """Halving the ring growth, sector and slab length brings the field within 2 % everywhere (90 deg, t = 2 s)."""
    beta = 0.5 * math.pi
    st = ThermalSettings(tool_field_length_m=0.04, tool_field_growth=1.25, tool_field_sector_deg=10.0,
                         tool_field_dy_m=0.25e-3)
    tool = wedge_tool(beta, st)
    q, l, b, y0 = 5e7, 2e-4, 2e-3, 5e-3
    heat_rake_strip(tool, q, l, b, y0, 0.25, 2.0)
    yc = y0 + 0.5 * b
    pts = [(0.0, yc, 0.0), (l, yc, 0.0), (1e-3, yc, 0.0), (2e-3, yc, 0.0), (1e-3 * math.cos(-0.7), yc,
                                                                            1e-3 * math.sin(-0.7))]
    sim = tool.temperature_at(np.array(pts)) - 293.15
    exact = image_rise(q, l, y0, y0 + b, beta, 2, 2.0, pts)
    assert np.all(np.abs(sim / exact - 1.0) < 0.02), np.round(sim / exact - 1.0, 4)
    rake, _, _ = tool.field.strip_means([l], [1e-6], [y0], [y0 + b], [0])
    assert rake[0] - 293.15 == pytest.approx(strip_mean_exact(q, l, y0, y0 + b, beta, 2, 2.0), rel=0.025)


def test_half_space_rectangle_matches_carslaw_jaeger():
    """180 deg wedge with equal fluxes on both faces = a rectangle 2a x 2b centred on the edge line of a
    half space: centre and area-mean rise against (q a / k) Phi(Fo, m) (Carslaw & Jaeger 1959 10.4,
    the response tables of the edge model)."""
    q, a, half_b, yc = 4e7, 1.5e-4, 6e-4, 15e-3
    tool = wedge_tool(math.pi, ThermalSettings(tool_field_length_m=0.04),
                      contact_edges=(yc - half_b, yc + half_b))
    dt, t_end = 0.005, 0.1
    t = 0.0
    while t < t_end - 1e-12:
        e = q * a * 2 * half_b * dt
        tool.deposit_contact(e, e, a, a, 2 * half_b, yc - half_b)
        tool.step(dt, AIR, 1e-9)
        t += dt
    resp = edge_response()
    im, wm = K.m_index(resp, half_b / a)
    phi_c, phi_m = K.phi_pair(resp, ALPHA * t_end / (a * a), int(im), wm)
    centre = tool.temperature_at(np.array([[0.0, yc, 0.0]]))[0] - 293.15
    assert centre == pytest.approx(q * a / K_W * phi_c, rel=0.03)
    rake, flank, _ = tool.field.strip_means([a], [a], [yc - half_b], [yc + half_b], [0])
    assert 0.5 * (rake[0] + flank[0]) - 293.15 == pytest.approx(q * a / K_W * phi_m, rel=0.08)


def test_insert_section_matches_the_open_wedge_before_the_heat_reaches_its_faces():
    """The clipped insert section (seat, back) gives the unclipped wedge's field while the diffusion
    length is small against the section (cut cells only far from the edge)."""
    st = ThermalSettings(tool_field_length_m=0.04)
    values = []
    for kind in ("insert", "tooth"):
        tool = wedge_tool(0.5 * math.pi, st, kind)
        heat_rake_strip(tool, 5e7, 2e-4, 2e-3, 5e-3, 0.01, 0.05)
        values.append(tool.temperature_at(np.array([[0.0, 6e-3, 0.0], [1e-3, 6e-3, -1e-3], [2e-3, 6e-3, 0.0]])))
    assert np.allclose(values[0], values[1], rtol=0, atol=0.01)


def small_insert(every=1, coolant=False):
    spec = ToolSpec(kind="turning", tool_material_id="carbide-k20-6co", insert_ic_m=0.006, insert_thickness_m=0.003,
                    bar_h_m=0.02, bar_b_m=0.02, bar_overhang_m=0.01, holder_material_id="aisi4140-qt30")
    mt = props.tool_material_table_by_id(spec.tool_material_id)
    mh = props.material_table_by_id(spec.holder_material_id)
    tool = ToolThermal(0, spec, mt, mh, 2e-3, 293.15, 0.0)
    st = ThermalSettings(tool_field_growth=3.0, tool_field_sector_deg=45.0, tool_field_dy_m=3e-3)
    tool.enable_field(st, wedge_angle_rad=math.radians(80), edge_radius_m=20e-6, every_n_steps=every)
    env = ToolEnvironment(T_air_K=293.15, T_end_K=300.0, coolant_on=coolant, T_fluid_K=290.0,
                          h_coolant_W_m2K=5000.0 if coolant else 0.0, reach=0.6 if coolant else 0.0)
    return tool, env


def test_field_kernel_matches_dense_reference():
    tool, env = small_insert(coolant=True)
    f = tool.field
    for _ in range(3):
        tool.deposit_contact(5.0, 2.0, 1.5e-4, 3e-5, 2e-3, 1e-3)
        tool.set_environment(env)
        f.set_environment(env, tool.emissivity_tool)
        d_cells, d_ladder, terms = R.field_step_ref(f.mesh, f.sys, tool.ladder, tool.mat_tool, tool.mat_holder, 0.05)
        out = np.zeros(N_FD)
        K.field_step(f.mesh, f.sys, tool.ladder, tool.mat_tool, tool.mat_holder, 0.05, out)
        n = f.sys.T.size
        act = f.sys.active != 0
        assert np.allclose(f.sys.x[:n], d_cells, rtol=1e-9, atol=1e-9 * np.max(np.abs(d_cells)))
        assert np.allclose(f.sys.x[n:][act], d_ladder[act], rtol=1e-9, atol=1e-12)
        for idx, key in ((FD_AIR, "air"), (FD_COOL, "cool"), (FD_SP, "sp"), (FD_COUPLE, "couple"), (FD_DU, "du"),
                         (FD_SRC, "src")):
            assert out[idx] == pytest.approx(terms[key], rel=1e-10, abs=1e-13)
        assert out[FD_U1] - out[FD_U0] == pytest.approx(out[FD_SRC] - out[FD_AIR] - out[FD_COOL] - out[FD_SP],
                                                        rel=1e-12, abs=1e-12)
        f.sys.src[:] = 0.0
        f.face_cover[:] = 0.0


def test_energy_ledger_and_views_over_a_cut_and_cooling():
    """Field + ladder conserve energy to round-off through heating, a cadence of 3 and cooling with coolant;
    the insert's ladder nodes are views holding exactly the field's mass and energy."""
    tool, env = small_insert(every=3, coolant=True)
    deposited = 0.0
    for i in range(60):
        if i < 40:
            tool.deposit_contact(1.0, 0.4, 1.5e-4, 3e-5, 2e-3, 1e-3)
            deposited += 1.4
        tool.step(0.02, env, 1e-12)
    tool.flush_field(env, 1e-12)
    assert tool.ledger.worst_rel < 1e-12
    assert tool.ledger.sources_J == pytest.approx(deposited, rel=1e-12)
    views = tool.field.view_nodes
    lad = tool.ladder
    assert math.fsum(lad.m[views]) == pytest.approx(math.fsum(tool.field.mesh.m), rel=1e-12)
    assert math.fsum(lad.m[views] * lad.h[views]) == pytest.approx(tool.field.energy_J(), rel=1e-11)
    assert tool.energy_J() == pytest.approx(float(K.ladder_energy(lad)), rel=1e-11)
    # the seat carried heat into the bar
    assert lad.T[tool.layout.holder_nodes[0]] > 293.2


def test_seat_conductance_is_the_contact_in_series_with_the_bar():
    tool, _ = small_insert()
    f = tool.field
    seat = f.face_kind == FACE_SEAT
    area = f.mesh.face_area[seat].sum()
    p = dict(params("thermal-tool"))
    contact = p["insert_seat_area_ratio"] * 0.006 ** 2
    assert np.allclose(f.mesh.face_rc[seat], area / (p["shank_contact_conductance_W_m2K"] * contact))
    assert np.allclose(f.mesh.face_lk[seat], 0.02 / 3.0)
    assert np.all(f.mesh.face_node[seat] == tool.layout.holder_nodes[0])


def test_ring_edges_end_exactly_at_the_outer_radius():
    r = ring_edges(2e-6, 1.4, 1.3e-3)
    assert r[0] == 0.0 and r[1] == 2e-6 and r[-1] == 1.3e-3 and np.all(np.diff(r) > 0)
    assert np.all(r[2:-1] / r[1:-2] == pytest.approx(1.4))
    assert 0.3 <= (r[-1] - r[-2]) / (r[-2] - r[-3]) <= 1.4 * 1.3


def test_deviation_mode_matches_brute_force_and_has_zero_cycle_mean():
    resp = edge_response()
    xi, _ = heating_phase_nodes(3)
    inp = new_edge_input((1, 1), 3, EDGE_OUT_DEVIATION)
    inp.xi[:] = xi
    inp.scal[0], inp.scal[1] = 4, 3
    qr, ar, qf, af, b, th, period, ncyc = 3e7, 1.5e-4, 1e7, 4e-5, 2.5e-4, 1e-3, 6e-3, 40
    k, alpha, wedge = 40.0, 1.2e-5, 1.2
    for name, value in (("mode", EDGE_INTERRUPTED), ("q_r", qr), ("a_r", ar), ("q_f", qf), ("a_f", af), ("b", b),
                        ("t_h", th), ("period", period), ("n_cyc", ncyc), ("k", k), ("alpha", alpha),
                        ("wedge", wedge), ("r_far", 1e-3), ("T_body", 400.0)):
        getattr(inp, name)[:] = value
    out = new_edge_output((1, 1), 3)
    glx, glw = gauss_legendre(3)
    K.edge_temperatures(inp, resp, glx, glw, out)
    rake, peak, flank = R.edge_deviation_ref(qr, ar, qf, af, b, th, period, ncyc, k, alpha, wedge, xi)
    assert np.allclose(out.rake_q[0, 0], rake, rtol=2e-3, atol=0.02)
    assert np.allclose(out.peak_q[0, 0], peak, rtol=2e-3, atol=0.02)
    assert np.allclose(out.flank_q[0, 0], flank, rtol=2e-3, atol=0.02)
    # the deviation of the edge line averages to zero over the whole cycle (heating and cooling)
    def dev(tau):
        return wedge * (qr * R.periodic_ref(tau, th, period, ncyc, ar, b, k, alpha, 0)
                        + qf * R.periodic_ref(tau, th, period, ncyc, af, b, k, alpha, 0))
    mean = integrate.quad(dev, 0.0, period, points=[th], limit=200)[0] / period
    assert mean > 0.0
    tau = xi * th
    assert np.allclose(out.peak_q[0, 0], np.array([dev(t) for t in tau]) - mean, rtol=2e-3, atol=0.05)
    # continuous elements deviate by nothing
    inp.mode[:] = 2
    inp.t_eng[:] = 0.5
    K.edge_temperatures(inp, resp, glx, glw, out)
    assert np.all(out.rake_q == 0.0) and np.all(out.peak_q == 0.0) and out.amp[0, 0] == 0.0


def cnmg_edges():
    from cnctwin.db.database import Database
    from cnctwin.tooling.build import build_tool
    tid = "tungaloy-cnmg120408-tm-t9215-pclnr2525m12e"
    return build_tool(Database.load().tool(tid), None, None, tool_id=tid).edge


def test_insert_elements_unroll_from_the_nose_along_the_major_edge():
    edges = cnmg_edges()
    el = unroll_elements(edges, True, 0.0127)
    kinds = np.asarray(edges.elem_kind[0])[el.elem]
    kappa = np.degrees(np.asarray(edges.elem_kappa_rad[0])[el.elem])
    along = el.side == 0
    assert np.all(kinds[along][:14] == 9) and el.lo[0] == 0.0          # the nose first, from its minor-edge end
    assert np.all(np.diff(el.lo[along]) > 0) and kappa[along][0] > 90   # ... toward the major edge (kappa -5 deg)
    assert np.all(kappa[~along] == pytest.approx(95.0))                 # the minor edge on the prism's end face
    assert el.in_field[0, el.elem].all()


def test_tool_frame_points_of_an_insert_map_into_the_field():
    edges = cnmg_edges()
    spec = ToolSpec(kind="turning", tool_material_id="carbide-p30", insert_ic_m=0.0127, insert_thickness_m=0.00476,
                    bar_h_m=0.025, bar_b_m=0.025, bar_overhang_m=0.032, holder_material_id="aisi4140-qt30")
    mt = props.tool_material_table_by_id("carbide-p30")
    mh = props.material_table_by_id("aisi4140-qt30")
    tool = ToolThermal(0, spec, mt, mh, 1e-3, 293.15, 0.0)
    assert tool.enable_field(ThermalSettings(), edges)
    f = tool.field
    from cnctwin.thermal.toolmodel import points_to_edge_frame
    pos = f.frames["pos"][0]
    pe = points_to_edge_frame(f.frames, pos)
    side0 = f.elements.side == 0
    assert np.allclose(pe[side0, 0], 0.0, atol=1e-12) and np.allclose(pe[side0, 2], 0.0, atol=1e-12)
    assert np.allclose(pe[side0, 1], f.frames["y_mid"][side0], atol=1e-12)
    # heat the nose, read it through the tool frame; a point outside the insert reads the bar's first node
    for _ in range(10):
        tool.deposit_contact(2.0, 0.5, 2e-4, 35e-6, 1.5e-3, 0.0)
        tool.step(0.1, AIR, 1e-9)
    inside = pos[2] + 0.5e-3 * f.frames["x"][0, 2]                      # 0.5 mm along the rake face from the nose
    far = np.array([0.05, 0.05, 0.05])
    temps = tool.temperature_at_T(np.array([inside, far]))
    assert temps[0] > 330.0
    assert temps[1] == pytest.approx(float(tool.ladder.T[tool.layout.tool_nodes[0]]))


def test_domain_uses_the_field_and_resumes_bit_identically():
    import copy
    proj = F.project()
    ctx = F.context(proj)
    state = F.sim_state()
    dom = ThermalDomain(geometry=F.stock(), edges=F.Edges(F.edge_model()))
    dom.init(ctx, state)
    tool = state.thermal.tools[0]
    assert tool.field is not None and tool.field.kind == "tooth" and tool.field.mult == 4
    outs = []
    for n in range(12):
        prev = F.bus_step(n)
        nxt = Bus()
        state.clock.tick = int(prev.motion.tick)
        dom.macro_step(ctx, state, prev, nxt)
        outs.append(nxt)
        if n == 6:
            arrays, meta = state.thermal.to_arrays()
            snapshot = (copy.deepcopy(arrays), copy.deepcopy(meta))
    h = outs[-1].thermal.tools[0]
    assert np.all(np.isfinite(h.edge_temp_q_K)) and np.all(h.edge_peak_q_K > h.body_temp_at_elem_K[:, :, None])
    assert tool.ledger.worst_rel < 1e-12
    # temperature at the edge of tooth 0 (T frame, spindle angle 0) is the field's
    point = tool.field.frames["pos"][0, 4]
    assert state.thermal.tool_temperature_at(0, point[None, :])[0] > float(h.body_temp_at_elem_K[0, 4])
    from cnctwin.thermal.state import ThermalState
    resumed = ThermalState.from_arrays(*snapshot)
    state.thermal = resumed
    for n in range(7, 12):
        prev = F.bus_step(n)
        nxt = Bus()
        state.clock.tick = int(prev.motion.tick)
        dom.macro_step(ctx, state, prev, nxt)
    assert nxt.thermal.tools[0].edge_temp_q_K.tobytes() == h.edge_temp_q_K.tobytes()
    assert resumed.tools[0].field.sys.T.tobytes() == tool.field.sys.T.tobytes()


def test_insert_elements_heat_the_field_and_get_its_edge_temperatures():
    """Per-element heat of a turning contact (nose and the first millimetre of the major edge) goes into the
    field exactly; the elements in cut take the field's contact temperatures, the others its edge there."""
    from cnctwin.thermal.partition import ContactHeat
    edges = cnmg_edges()
    spec = ToolSpec(kind="turning", tool_material_id="carbide-p30", insert_ic_m=0.0127, insert_thickness_m=0.00476,
                    bar_h_m=0.025, bar_b_m=0.025, bar_overhang_m=0.032, holder_material_id="aisi4140-qt30")
    mt = props.tool_material_table_by_id("carbide-p30")
    mh = props.material_table_by_id("aisi4140-qt30")
    tool = ToolThermal(0, spec, mt, mh, 1e-3, 293.15, 0.0)
    assert tool.enable_field(ThermalSettings(), edges, every_n_steps=2)
    el = tool.field.elements
    shape = np.shape(edges.elem_z_m)
    cutting = np.zeros(shape, dtype=bool)
    cutting[0, el.elem[(el.side == 0) & (el.hi <= 2.5e-3)]] = True
    heat = ContactHeat(0, 0, 0)
    heat.tool_rake_elem_J = np.where(cutting, 0.02, 0.0)
    heat.tool_flank_elem_J = np.where(cutting, 0.005, 0.0)
    heat.h_mean_m = np.where(cutting, 1e-4, 0.0)
    heat.compression = np.full(shape, 2.5)
    in_cut = np.where(cutting, 0.01, 0.0)
    settings = ThermalSettings(edge_every_n_macro=1)
    total = 0.0
    for _ in range(20):
        total += tool.deposit_elements(edges, heat, None)
        tool.step(0.01, AIR, 1e-12)
        out = tool.edge_heat(edges, heat, in_cut, np.zeros(shape), None, 0.01, settings)
    assert total == pytest.approx(20 * 0.025 * cutting.sum(), rel=1e-12)
    assert tool.ledger.sources_J == pytest.approx(total, rel=1e-12) and tool.ledger.worst_rel < 1e-12
    hot = out.edge_temp_q_K[0, cutting[0], 0]
    idle = out.edge_temp_q_K[0, el.elem[el.side == 0][-1], 0]
    assert np.all(np.isfinite(out.edge_temp_q_K[edges.elem_valid != 0]))
    assert hot.min() > idle + 50.0 and np.all(out.edge_cycle_amp_K == 0.0)        # continuous cutting
    assert np.all(out.edge_peak_q_K[0, cutting[0]] >= out.rake_temp_mean_K[0, cutting[0], None] - 1e-9)


def test_insert_field_checkpoint_round_trip_is_bit_identical():
    """An insert field mid-cadence (pending heat, warm start, views) resumes bit-identically."""
    tool, env = small_insert(every=3, coolant=True)
    for _ in range(4):
        tool.deposit_contact(1.0, 0.4, 1.5e-4, 3e-5, 2e-3, 1e-3)
        tool.step(0.02, env, 1e-12)
    arrays, meta = tool.to_arrays("tool.0.")
    twin = ToolThermal.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta, "tool.0.")
    for t in (tool, twin):
        for _ in range(5):
            t.deposit_contact(1.0, 0.4, 1.5e-4, 3e-5, 2e-3, 1e-3)
            t.step(0.02, env, 1e-12)
    assert twin.field.sys.T.tobytes() == tool.field.sys.T.tobytes()
    assert twin.ladder.T.tobytes() == tool.ladder.T.tobytes()
    assert twin.field.face_T.tobytes() == tool.field.face_T.tobytes()
    assert twin.ledger.u_J == tool.ledger.u_J
