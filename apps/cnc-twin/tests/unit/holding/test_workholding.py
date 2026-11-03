"""Workholding: vise slip threshold against the hand calculation, clamping force from torque, the chuck's
centrifugal loss against the Kitagawa worked example and VDI 3106, vacuum and magnetic holding, tab release,
the supports handed to structure and the lathe support conditions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.holding import holdfix as fx
from cnctwin.core.findings import Category, Severity
from cnctwin.db.simap import SIMap
from cnctwin.holding import devices as D
from cnctwin.holding import setup as SU
from cnctwin.holding.domain import HoldingDomain
from cnctwin.holding.protocols import NominalTabs, StockGeometry
from cnctwin.holding.schema import settings

G = 9.80665
BLOCK = {"kind": "block", "size_mm": [100, 60, 30]}
CENTROID = np.array([0.05, 0.03, 0.015])


def at(point, force):
    """Wrench on the part about the W origin for a force through a point (W = H here)."""
    f = np.asarray(force, float)
    return np.concatenate([f, np.cross(np.asarray(point, float), f)])


def domain(entries, stock=None, material=None, machine=None, tabs=None, lathe=False):
    proj = fx.project(entries, stock=stock or BLOCK, material=material, machine=machine)
    ctx = fx.context(proj)
    state = fx.sim_state()
    td = fx.tool_data(holder_type="lathe_holder", rotating=False) if lathe else fx.tool_data()
    dom = HoldingDomain(tools={0: td}, tabs=tabs)
    dom.init(ctx, state)
    return dom, ctx, state


def vise_entry(**kw):
    return fx.entry(fx.VISE, translate_mm=(50, 0, 0), input_torque_Nm=80, parallels_mm=10, **kw)


def test_vise_slip_threshold_is_the_hand_calculation():
    """Sliding along the jaws through the centroid: F_slip = 2 mu_jaw F_clamp + mu_base m g."""
    dom, ctx, state = domain([vise_entry()])
    mass = 0.1 * 0.06 * 0.03 * 7850.0
    hand = 2 * 0.2 * 20000.0 + 0.19 * mass * G
    for F in (6000.0, 7600.0):                       # margins in the finely resolved band (< 2)
        w = at(CENTROID, [F, 0, 0])
        rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=0.6 * w, work_peak=w))
        assert rec.parts[0].margin * F == pytest.approx(hand, rel=1.5e-3)
        assert rec.parts[0].mode == "" and not rec.parts[0].free
    # above it the part ratchets along the jaws (only the peaks exceed the capacity) and stops when they drop
    w = at(CENTROID, [8800.0, 0, 0])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=0.6 * w, work_peak=w))
    p = rec.parts[0]
    assert p.mode == "slip" and p.slip_m > 0.0 and p.pose_delta[0, 3] > 0.0
    assert abs(p.pose_delta[1, 3]) < 1e-3 * p.pose_delta[0, 3]
    slipped = p.pose_delta[0, 3]
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=0.3 * w, work_peak=0.5 * w))
    assert rec.parts[0].slip_m == 0.0 and rec.parts[0].pose_delta[0, 3] == pytest.approx(slipped, rel=1e-12)
    cats = [(f.category, f.severity) for f in state.findings]
    assert (Category.PART_SLIP, Severity.ERROR) in cats and (Category.PART_SLIP, Severity.WARNING) in cats


def test_vise_lift_off_by_jaw_friction_and_weight():
    dom, ctx, state = domain([vise_entry()])
    mass = 0.1 * 0.06 * 0.03 * 7850.0
    hand = 2 * 0.2 * 20000.0 + mass * G               # pulled straight up out of the jaws
    w = at(CENTROID, [0, 0, 6000.0])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=w, work_peak=w))
    assert rec.parts[0].margin * 6000.0 == pytest.approx(hand, rel=1.5e-3)


def test_clamping_force_from_torque():
    notes = []
    vise = SIMap(fx.VISE)
    assert D.vise_force(SIMap({"input_torque_Nm": 40}), vise, notes) == pytest.approx(10000.0)
    generic = SIMap({"id": "v", "type": "vise", "jaw_width_mm": 150, "max_opening_mm": 200})
    force = D.vise_force(SIMap({"input_torque_Nm": 100}), generic, notes)
    d_m = 0.022 - 0.0025
    lever = 0.5 * d_m * math.tan(math.atan(0.005 / (math.pi * d_m)) + math.atan(0.15 / math.cos(math.radians(14.5)))) \
        + 0.15 * 0.015
    assert force == pytest.approx(100.0 / lever, rel=1e-12)
    clamp = SIMap(fx.CLAMP)
    assert D.clamp_force(SIMap({"input_torque_Nm": 30}), clamp, notes) == pytest.approx(9790.0 * 30 / 47.5)
    stud_only = SIMap({"id": "s", "type": "clamp", "stud": "M12", "recommended_torque_Nm": 47.5, "lever_ratio": 2.0})
    # Shigley T = K F d with K = 0.2, the strap halves the stud tension (2:1 lever)
    assert D.clamp_force(SIMap({}), stud_only, notes) == pytest.approx(47.5 / (0.2 * 0.012) / 2.0)
    preload = SIMap({"id": "s", "type": "clamp", "recommended_torque_Nm": 47.5, "stud_preload_kN": 19.57,
                     "lever_ratio": 2.0})
    assert D.clamp_force(SIMap({}), preload, notes) == pytest.approx(19570.0 / 2.0)


def test_two_strap_clamps_on_the_table():
    entries = [fx.entry(fx.CLAMP, translate_mm=(20, 30, 30)), fx.entry(fx.CLAMP, translate_mm=(80, 30, 30))]
    dom, ctx, state = domain(entries)
    mass = 0.1 * 0.06 * 0.03 * 7850.0
    F = 9790.0
    hand = 0.19 * 2 * F + 0.19 * (2 * F + mass * G)   # pads and the table under the part
    w = at([0.05, 0.03, 0.0], [5000.0, 0, 0])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=w, work_peak=w))
    assert rec.parts[0].margin * 5000.0 == pytest.approx(hand, rel=1.5e-3)
    assert {s["role"] for s in rec.parts[0].supports} == {"clamp", "base"}


def kitagawa_chuck(**kw):
    item = dict(fx.CHUCK, max_rpm=6000)
    item.update(kw)
    return item


def test_chuck_centrifugal_loss_against_the_kitagawa_example():
    """Kitagawa 'Gripping force setting of power chucks' (BB208): static 60 kN, loss 38 kN at 3500 rpm,
    dynamic 22 kN, gripping torque F_gD mu r_g = 22 kN x 0.1 x 80 mm = 176 Nm."""
    omega = 3500 * math.pi / 30
    notes = []
    assert D.chuck_loss(SIMap(kitagawa_chuck()), omega, 60000.0, 3, notes) == pytest.approx(38000.0)
    assert D.chuck_loss(SIMap(kitagawa_chuck()), omega / 2, 60000.0, 3, notes) == pytest.approx(9500.0)
    stock = {"kind": "cylinder", "size_mm": [160, 20]}
    dom, ctx, state = domain([fx.entry(kitagawa_chuck(), grip_depth_mm=15, jaws="od no_stop")], stock=stock,
                             lathe=True)
    torque = 120.0
    w = np.array([0, 0, 0, 0, 0, torque])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=w, work_peak=w, omega=omega))
    p = rec.parts[0]
    assert p.clamp_force_N == pytest.approx(22000.0)
    assert p.lathe["chuck"]["gripping_force_N"] == pytest.approx(22000.0)
    assert p.margin * torque == pytest.approx(176.0, rel=2e-3)
    assert p.lathe["chuck"]["grip_length_m"] == pytest.approx(0.015)


def test_vdi_3106_jaw_masses_and_the_example_mass_moment():
    """F_c = sum(m r) omega^2: Kitagawa's top jaws (1.2 kg at 72 mm, 3 jaws, mass moment 259.2 kg mm) give
    34.8 kN at 3500 rpm; the published 38 kN includes the master jaws (coefficient 1.09)."""
    omega = 3500 * math.pi / 30
    notes = []
    jaws = SIMap({"id": "c", "type": "chuck_3jaw", "jaws": 3, "jaw_mass_kg": 1.2, "jaw_cg_radius_mm": 72})
    assert D.chuck_loss(jaws, omega, 60000.0, 3, notes) == pytest.approx(3 * 1.2 * 0.072 * omega ** 2, rel=1e-12)
    assert 3 * 1.2 * 0.072 * omega ** 2 == pytest.approx(34820.0, rel=1e-3)
    moment = SIMap({"id": "c", "type": "chuck_3jaw", "jaws": 3, "jaw_mass_moment_kg_mm": 259.2,
                    "centrifugal_loss_coeff": 1.09})
    assert D.chuck_loss(moment, omega, 60000.0, 3, notes) == pytest.approx(38000.0, rel=2e-3)
    # no loss data: the speed rule (dynamic 1/3 of static at the maximum speed)
    rule = SIMap({"id": "c", "type": "chuck_3jaw", "jaws": 3, "max_rpm": 4000})
    assert D.chuck_loss(rule, 4000 * math.pi / 30, 60000.0, 3, notes) == pytest.approx(40000.0, rel=1e-3)


def test_inside_gripping_gains_and_overspeed_loses_the_part():
    omega = 3500 * math.pi / 30
    tube = {"kind": "tube", "size_mm": [160, 100, 20]}
    dom, ctx, state = domain([fx.entry(kitagawa_chuck(), grip_depth_mm=15, jaws="id")], stock=tube, lathe=True)
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), omega=omega))
    assert rec.parts[0].clamp_force_N == pytest.approx(60000.0 + 38000.0)
    dom, ctx, state = domain([fx.entry(kitagawa_chuck(), grip_depth_mm=15)], stock={"kind": "cylinder",
                                                                                     "size_mm": [160, 20]}, lathe=True)
    fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), omega=omega))
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), omega=6500 * math.pi / 30))
    cats = {(f.category, f.severity) for f in state.findings}
    assert (Category.OVERSPEED, Severity.ERROR) in cats
    assert rec.parts[0].free and "speed" in rec.parts[0].free_cause


VACUUM = {"id": "test-vac", "type": "vacuum", "plate_mm": [400, 350, 30], "min_abs_pressure_kPa": 15,
          "area_utilisation": 0.8, "jaw_friction": 0.3}


def test_vacuum_holding_force_and_the_seal_breaking_on_lift():
    dom, ctx, state = domain([fx.entry(VACUUM, translate_mm=(50, 30, 0))])
    mass = 0.1 * 0.06 * 0.03 * 7850.0
    hold = (101325.0 - 15000.0) * 0.1 * 0.06 * 0.8
    w = at([0.05, 0.03, 0.0], [100.0, 0, 0])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=w, work_peak=w))
    assert rec.parts[0].margin * 100.0 == pytest.approx(0.3 * (hold + mass * G), rel=1.5e-3)
    assert rec.parts[0].clamp_force_N == pytest.approx(hold)
    # a pull above the hold-down lifts an edge: the seal opens, the vacuum is lost and the part is free
    w = at([0.09, 0.03, 0.03], [0, 0, 0.8 * hold])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=w, work_peak=w))
    assert rec.parts[0].free
    assert any(f.category == Category.PART_FREE for f in state.findings)


def test_magnetic_chuck_does_not_hold_aluminium():
    mag = {"id": "test-mag", "type": "magnetic", "jaw_friction": 0.2,
           "holding_pressure_N_cm2": {"table": [[3, 15], [10, 80], [25, 157], [50, 157]], "x": "mm", "src": "est"}}
    dom, ctx, state = domain([fx.entry(mag, translate_mm=(50, 30, 0))], material=fx.ALUMINIUM)
    fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6)))
    assert any(f.category == Category.PART_FREE and "ferromagnetic" in f.message for f in state.findings)
    dom, ctx, state = domain([fx.entry(mag, translate_mm=(50, 30, 0))])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6)))
    p30 = 157.0 * 1e4          # 30 mm thick steel: the saturated end of the table
    assert rec.parts[0].clamp_force_N == pytest.approx(p30 * 0.1 * 0.06, rel=1e-9)


class ShrinkingTab:
    """TabSections whose remaining section shrinks by 10 % of the nominal per step (removal cutting the tab)."""

    wired = True

    def __init__(self):
        self.step = 0

    def section_S(self, workpiece, centre_S, normal_S, axis1_S, axis2_S, half1_m, half2_m):
        frac = max(1.0 - 0.1 * self.step, 0.0)
        return 4.0 * half1_m * half2_m * frac, np.asarray(centre_S, float)


def test_tab_release_when_the_remaining_section_cannot_carry_the_load():
    """A 4 x 2 mm tab in shear: capacity (sigma_u / sqrt 3) A f = 2771 N x f; 1000 N through its centre
    breaks it once f < 0.361, i.e. at the step where 30 % of the section is left."""
    tabs = ShrinkingTab()
    entries = [vise_entry(), fx.entry(fx.TABS, translate_mm=(100, 30, 15), rotate_deg=(0, 0, 180), label="tab A")]
    dom, ctx, state = domain(entries, tabs=tabs)
    w = at([0.1, 0.03, 0.015], [0, 1000.0, 0])
    capacity = 600e6 / math.sqrt(3.0) * 8e-6
    free_at = None
    for step in range(10):
        tabs.step = step
        rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=w, work_peak=w, tick0=step * 10000))
        if rec.parts[0].free:
            free_at = step
            break
        assert rec.parts[0].margin == pytest.approx(min(capacity * (1 - 0.1 * step) / 1000.0, 1000.0), rel=2e-3) \
            or rec.parts[0].margin >= 2.0
    assert free_at == 7
    assert "tab" in rec.parts[0].free_cause
    assert any(f.category == Category.PART_FREE and f.severity == Severity.ERROR for f in state.findings)


def test_supports_balance_the_load_and_the_support_stiffness_is_symmetric():
    dom, ctx, state = domain([vise_entry()])
    w = at(CENTROID, [1500.0, -300.0, 200.0])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=w, work_peak=1.5 * w))
    p = rec.parts[0]
    total = sum(s["force_S_N"].sum(axis=0) for s in p.supports)
    mass = 0.1 * 0.06 * 0.03 * 7850.0
    np.testing.assert_allclose(total, -(w[:3] + mass * np.array([0, 0, -G])), atol=1e-6 * 2e4)
    roles = {s["role"] for s in p.supports}
    assert roles == {"fixed_jaw", "moving_jaw", "parallel"}
    for s in p.supports:
        n = s["pos_S_m"].shape[0]
        assert s["normal_S"].shape == (n, 3) and s["stiffness_N_per_m"].shape == (n,) and s["area_m2"].shape == (n,)
    pre = sum(s["preload_S_N"].sum(axis=0) for s in p.supports)
    np.testing.assert_allclose(pre, -mass * np.array([0, 0, -G]), atol=1e-6)
    K = p.support_stiffness_S
    np.testing.assert_allclose(K, K.T, rtol=1e-9, atol=1e-6 * np.abs(K).max())
    assert np.all(np.linalg.eigvalsh(0.5 * (K + K.T)) > -1e-6 * np.abs(K).max())
    np.testing.assert_allclose(p.centroid_S_m, CENTROID)


def test_lathe_support_conditions_for_structure():
    lathe = {"type": "lathe", "kinematics": {"gravity_direction": [-1, 0, 0]}}
    centre = {"id": "c", "type": "tailstock_centre", "angle_deg": 60, "live": True, "max_rpm": 6000}
    steady = {"id": "s", "type": "steady_rest", "rollers": 3, "grip_range_mm": [8, 101],
              "clamp_force_per_roller_kN": 2.0, "roller_stiffness_N_um": 50}
    entries = [fx.entry(kitagawa_chuck(), grip_depth_mm=20), fx.entry(centre, translate_mm=(0, 0, 300),
                                                                       rotate_deg=(180, 0, 0)),
               fx.entry(steady, translate_mm=(0, 0, 150))]
    dom, ctx, state = domain(entries, stock={"kind": "cylinder", "size_mm": [40, 300]}, machine=lathe, lathe=True)
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.array([300.0, 100.0, 50.0, 0, 0, 10.0]), omega=200.0))
    lt = rec.parts[0].lathe
    assert lt["chuck"]["grip_length_m"] == pytest.approx(0.020) and lt["chuck"]["jaws"] == 3
    assert lt["tailstock"]["z_S_m"] == pytest.approx(0.300) and lt["tailstock"]["thrust_N"] == pytest.approx(5000.0)
    assert lt["steady"][0]["z_S_m"] == pytest.approx(0.150) and lt["steady"][0]["force_per_roller_N"] == 2000.0
    assert lt["chuck"]["gripping_force_N"] == pytest.approx(60000.0 - 38000.0 * (200.0 / (3500 * math.pi / 30)) ** 2)
    roles = {s["role"] for s in rec.parts[0].supports}
    assert {"jaw", "stop", "centre", "roller"} <= roles
    # the centre pushes the bar toward the chuck with the thrust
    centre_sup = [s for s in rec.parts[0].supports if s["role"] == "centre"][0]
    assert centre_sup["preload_S_N"].sum(axis=0)[2] == pytest.approx(-5000.0, rel=1e-6)


def test_setup_without_a_grip_and_contacts_on_the_moved_part():
    proj = fx.project([vise_entry()])
    geo = StockGeometry.from_stock(proj.stock)
    st = settings(None)
    s0 = SU.build_setup(proj, 0, geo, NominalTabs(), st)
    T = SU.twist_matrix(np.array([0.2, 0, 0, 0, 0, 0]), CENTROID)        # slid 200 mm along the jaws
    s1 = SU.build_setup(proj, 0, geo, NominalTabs(), st, slip_pose=T)
    assert s0.group("main").held and s1.no_grip and s1.group("main").n == 0


def test_chuck_pressure_from_the_cylinder_text_and_soft_jaw_grip_depth():
    notes = []
    item = SIMap(kitagawa_chuck(gripping_force_kN=86, matching_cylinder="S1552, max pressure 2.65 MPa"))
    assert D.chuck_static_force(SIMap({"pressure_bar": 15}), item, notes) == pytest.approx(86000.0 * 1.5 / 2.65)
    assert any("matching cylinder" in n for n in notes)
    soft = {"id": "sj", "type": "soft_jaws", "jaw_length_mm": 95, "jaw_width_mm": 35, "jaw_friction": 0.1}
    stock = {"kind": "cylinder", "size_mm": [60, 200]}
    dom, ctx, state = domain([fx.entry(kitagawa_chuck()), fx.entry(soft, grip_depth_mm=30)], stock=stock, lathe=True)
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6), omega=100.0))
    assert rec.parts[0].lathe["chuck"]["grip_length_m"] == pytest.approx(0.030)


def test_plate_conventions_and_a_plate_that_carries_the_vise():
    plate = dict(fx.PLATE, plate_mm=[300, 300, 27])
    # the plate frame on its bottom face: the part sits at the plate thickness, it is seated
    dom, ctx, state = domain([fx.entry(plate, translate_mm=(50, 30, -27))])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6)))
    assert rec.parts[0].seated and {s["role"] for s in rec.parts[0].supports} == {"plate"}
    # a vise on a grid plate 60 mm below the part: the plate carries the vise, not the part
    dom, ctx, state = domain([fx.entry(plate, translate_mm=(50, 30, -60)), vise_entry()])
    rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=np.zeros(6)))
    assert rec.parts[0].seated and "plate" not in {s["role"] for s in rec.parts[0].supports}
    assert any("carries other devices" in n for n in state.holding.notes)


def _vise_info(entries, stock=None):
    from cnctwin.holding.protocols import stock_shape
    proj = fx.project(entries, stock=stock or BLOCK)
    shape = stock_shape(SIMap(stock or BLOCK))
    bc = D.BuildCtx(op_name="OP10", workpiece=0, shape=shape, T_H_S=np.eye(4), n_side=5, centre_nodes=8,
                    material_family="steel", ultimate_Pa=6e8, youngs_Pa=2.1e11, poisson_work=0.3,
                    machine=None)
    entry = proj.operations[0]["setup"]["workholding"][0]
    out = D.build_vise(bc, 0, entry, entry["device"], np.asarray(entry["pose"]), "")
    return out


def test_vise_frame_on_the_seat():
    """The vise frame sits on the seat (the parallels' top): the jaws grip from the seat to grip_depth (else jaw
    height - parallels); a part above the seat is not seated, a contact shape below it rests on it."""
    out = _vise_info([vise_entry()])
    assert out.info["grip_depth_m"] == pytest.approx(0.030) and out.info["seated_gap_m"] == 0.0     # 40 - 10 > 30
    out = _vise_info([vise_entry(grip_depth_mm=8)])
    assert out.info["grip_depth_m"] == pytest.approx(0.008) and any("disagree" in n for n in out.notes)
    jaw = next(pa for pa in out.patches if pa.role == "fixed_jaw")
    assert jaw.pos_H[:, 2].min() >= -1e-12 and jaw.pos_H[:, 2].max() <= 0.008 + 1e-12
    # the part placed 0.2 mm above the parallels: clearance, not seated
    out = _vise_info([fx.entry(fx.VISE, translate_mm=(50, 0, -0.2), input_torque_Nm=80, parallels_mm=10)])
    assert out.info["seated_gap_m"] == pytest.approx(2e-4)
    # the contact shape 2.5 mm below the seat (a face machined in an earlier operation): it rests on the seat
    out = _vise_info([fx.entry(fx.VISE, translate_mm=(50, 0, 2.5), input_torque_Nm=80, parallels_mm=10)])
    bed = [pa for pa in out.patches if pa.role == "parallel"]
    assert out.info["seated_gap_m"] == 0.0 and all(abs(pa.pos_H[:, 2] - 0.0025).max() < 1e-12 for pa in bed)
    assert out.info["grip_depth_m"] == pytest.approx(0.0275) and any("below the seat" in n for n in out.notes)


def test_a_descriptive_vise_keeps_its_grip_when_the_part_slips():
    """A vise frame set at its mounting face (the part 61 mm above the seat of the vise frame: the descriptive
    placement over grip_depth) grips like the same vise placed on the seat, also after slip has rebuilt the
    setup at the moved pose (the jaws stay where the programmed setup put them)."""
    w = at(CENTROID, [9400.0, 0, 0])
    runs = []
    for pose_z, extra in ((0.0, {}), (-61.0, {"grip_depth_mm": 30})):
        entry = fx.entry(fx.VISE, translate_mm=(50, 0, pose_z), input_torque_Nm=80, parallels_mm=10, **extra)
        dom, ctx, state = domain([entry])
        out = []
        for k in range(4):
            rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=0.6 * w, work_peak=w, tick0=k * 10000))
            p = rec.parts[0]
            out.append((p.pose_delta.copy(), p.supports_version, p.free, p.clamp_force_N))
        rec = fx.run(dom, ctx, state, fx.bus_with(work_mean=0.1 * w, work_peak=0.2 * w, tick0=50000))
        out.append((rec.parts[0].pose_delta.copy(), rec.parts[0].supports_version, rec.parts[0].free,
                    rec.parts[0].clamp_force_N))
        runs.append(out)
    seated, descriptive = runs
    assert seated[-1][1] > 1 and seated[-1][0][0, 3] > 5e-5            # slipped past the rebuild distance
    for a, b in zip(seated, descriptive):
        np.testing.assert_allclose(a[0], b[0], rtol=0.0, atol=1e-9)
        assert a[1] == b[1] and not a[2] and not b[2] and a[3] == pytest.approx(b[3], rel=1e-9)


def test_strap_clamps_stay_where_the_setup_put_them_when_the_part_slips():
    """A part sliding under two strap clamps past the rebuild distance stays clamped by the pads at their
    programmed points; a pad set beside the part (moved onto its edge in the programmed setup) loses the part
    as soon as the part moves away from that edge."""
    import cnctwin.holding.setup as SU_
    entries = [fx.entry(fx.CLAMP, translate_mm=(20, 30, 30)), fx.entry(fx.CLAMP, translate_mm=(80, 30, 30))]
    proj = fx.project(entries, stock=BLOCK)
    geom = StockGeometry.from_stock(SIMap(BLOCK))
    st = settings({})
    moved = fx.pose((2.0, 0, 0))                   # 2 mm along x: both pads still on the part
    s = SU_.build_setup(proj, 0, geom, NominalTabs(), st, slip_pose=moved)
    pads = [p for p in s.group("main").patches if p.role == "clamp"]
    assert len(pads) == 2 and not s.no_grip
    np.testing.assert_allclose(sorted(p.centre_H[0] for p in pads), [0.020, 0.080], atol=1e-12)
    # a pad beside the part (x 105 > 100): moved onto the edge x 100 as programmed, lost after a 1 mm slide in -x
    entries = [fx.entry(fx.CLAMP, translate_mm=(105, 30, 30)), fx.entry(fx.CLAMP, translate_mm=(20, 30, 30))]
    proj = fx.project(entries, stock=BLOCK)
    s0 = SU_.build_setup(proj, 0, geom, NominalTabs(), st)
    assert any(p.role == "clamp" and abs(p.centre_H[0] - 0.100) < 1e-12 for p in s0.group("main").patches)
    s1 = SU_.build_setup(proj, 0, geom, NominalTabs(), st, slip_pose=fx.pose((-1.0, 0, 0)))
    assert len([p for p in s1.group("main").patches if p.role == "clamp"]) == 1
    assert any("no longer under the pad" in n for n in s1.notes)
