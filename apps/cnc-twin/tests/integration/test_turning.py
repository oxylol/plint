"""(c) A straight OD turning pass on the polar field through the whole chain (Haas ST-20Y, 4140 Q&T bar d45,
CNMG120408 in PCLNR, G97 1400 rpm M04, f 0.25 mm/rev, ap 1.5 mm, rigid): the per-tick forces in T0 against
cutting's stand-alone turning driver for the same tool, table and kinematics (cnctwin.cutting.drive.Driver
run_turning with the exact surface of the earlier revolutions ahead of the tool); every element's chip against
that surface; the removed volume booked exactly on the polar field; the steady removal
rate against the ring pi (R0^2 - r^2) v_f less the nose-radius scallops; chip mass = rho V.

The tool frame is tooling's turning convention (x_T into the work, z_T into the tool body), the chain's turret
frame turned by the hot loop's tool mount (docs/models/hotloop.md "Turning tool frame"). M04: the direction in
which tooling's right-hand tool cuts along its rake face under motion's ISO spindle sign (hotfix.turn_program)."""

import math

import numpy as np
import pytest

from cnctwin.core.drive import OperationDriver
from tests.integration import hotfix as HF

D_MM, L_MM, AP_MM, Z1_MM = 45.0, 80.0, 1.5, -8.0
R0 = 0.5 * D_MM * 1e-3
STEADY_Z = (L_MM - 6.0, L_MM - 1.5)         # tip z in S (mm) of the steady window: 1.5 mm in, 2 mm before the end
TOL_DYN = 0.005                             # steady force with the modes vs rigid (relative to |F|; 1e-4 measured)


@pytest.fixture(scope="module")
def turn_run(tmp_path_factory):
    return run_turn(tmp_path_factory.mktemp("turn"))


def run_turn(tmp):
    drv = OperationDriver(HF.turn_project(tmp / "run", D_MM, L_MM, AP_MM, Z1_MM), use_dynamics=False)

    def observe(d, rec):
        fr = d.state.bus_next.motion.paths[0]
        tips = np.asarray(fr.tip_S_m)
        rec.z_range = (float(tips[:, 2].min()), float(tips[:, 2].max()))
        rec.r_tip = float(np.mean(np.hypot(tips[:, 0], tips[:, 1])))
        rec.vz = float((tips[-1, 2] - tips[0, 2]) / ((len(tips) - 1) * d.ctx.ipo_ticks * d.ctx.tick_s)) \
            if len(tips) > 1 else 0.0
        from cnctwin.core import hot_abi as H
        ct = d.hot.last["contacts"][0] if d.hot.last is not None else None
        if ct is not None and int(ct.ipar[H.CI_ACTIVE]):
            from cnctwin.cutting import abi as CA
            rec.pose = (np.array(ct.cut.pose.R_S_T0), np.array(ct.cut.pose.tip_S_m),
                        float(ct.cut.step[CA.ST_OMEGA_RAD_S]), float(ct.cut.step[CA.ST_DT_S]))
            rec.h = np.array(ct.cut.st.h_last_m)
            if STEADY_Z[0] * 1e-3 < rec.z_range[0] and rec.z_range[1] < STEADY_Z[1] * 1e-3:
                rec.cut_tool = type(ct.cut.tool)(*(np.array(a) for a in ct.cut.tool))
                rec.cut_table = type(ct.cut.table)(*(np.array(a) for a in ct.cut.table))
    drv.observers.append(observe)
    res = HF.run_until_done(drv)
    steady = [st for st in drv.steps if st.in_cut_s >= 0.0099 and hasattr(st, "pose")
              and STEADY_Z[0] * 1e-3 < st.z_range[0] and st.z_range[1] < STEADY_Z[1] * 1e-3]
    return drv, res, steady


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _insert_x(ins, z_T):
    """x_T extent [lo, hi] of removal's insert region at z_T (NaN outside its z range)."""
    lo = np.interp(z_T, ins.zl, ins.xl, left=np.nan, right=np.nan)
    hi = np.interp(z_T, ins.zu, ins.xu, left=np.nan, right=np.nan)
    return lo, hi


def previous_surface(ins, r_tip, z_prev, z_rows):
    """The material a steady external pass meets, per row: the bar [0, R0] less the insert regions at the earlier
    tool positions z_prev (tip z of each earlier revolution), as the hull of its radial segments and the gap
    between the two outermost (cutting's PolarView with its air gap). Returns (r_in, r_out, gap_lo, gap_hi)."""
    r_in = np.zeros(len(z_rows))
    r_out = np.full(len(z_rows), R0)
    g_lo = np.zeros(len(z_rows))
    g_hi = np.zeros(len(z_rows))
    for i, z in enumerate(z_rows):
        segs = [(0.0, R0)]
        for zk in z_prev:
            lo, hi = _insert_x(ins, z - zk)
            if not np.isfinite(lo):
                continue
            a, b = r_tip - hi, r_tip - lo                  # x_T points toward the axis: r = r_tip - x_T
            nxt = []
            for s0, s1 in segs:
                if b <= s0 or a >= s1:
                    nxt.append((s0, s1))
                    continue
                if a > s0:
                    nxt.append((s0, a))
                if b < s1:
                    nxt.append((b, s1))
            segs = nxt
        if segs:
            r_in[i], r_out[i] = segs[0][0], segs[-1][1]
            if len(segs) >= 2:
                g_lo[i], g_hi[i] = segs[-2][1], segs[-1][0]
        else:
            r_in[i] = r_out[i] = 0.0
    return r_in, r_out, g_lo, g_hi


def turning_reference(drv, pose_rec, vz, frac=0.25):
    """cutting's stand-alone turning driver with the run's tool and table: the tool at the run's radius and
    orientation (one steady pose of the loop, turned to angle 0), feeding at the run's axial speed, the work at
    the run's speed, over ``frac`` of a revolution; the polar view ahead of it holds the exact surface of the
    earlier revolutions (``previous_surface`` per bin: the tool positions one, two, ... revolutions before it
    passes the bin), no update; the loop's own tool and table of that step (cutting's coefficients follow the
    tool's state between macro steps). Returns the mean T0 wrench and the driver."""
    from cnctwin.cutting import abi as CA
    from cnctwin.cutting.drive import Driver
    R_S_T0, tip, omega_s, dt = pose_rec.pose
    th = math.atan2(tip[1], tip[0])
    R0_T = _rz(-th) @ R_S_T0
    p0 = _rz(-th) @ tip
    r_tip = float(math.hypot(p0[0], p0[1]))
    state = drv.state
    tc = state.cutting.tools[0]
    ins = drv.hot.last["rm"].inserts[0]
    nt, ne = np.asarray(state.tooling.effective[0].elem_r_m).shape
    nth, dz = 720, 5e-6
    z_lo, z_hi = p0[2] - 0.5e-3, p0[2] + 2.5e-3
    nz = int(round((z_hi - z_lo) / dz))
    z_rows = z_lo + (np.arange(nz) + 0.5) * dz
    polar = CA.empty_polar(nth, nz)
    polar.geom[:] = (2 * math.pi / nth, z_lo, dz, 0.0)
    T = 2 * math.pi / abs(omega_s)
    sense = 1.0 if -omega_s > 0.0 else -1.0             # the tool's angle in S runs at -omega_s
    n_bins = int(frac * nth) + 4
    for k in range(-2, n_bins):
        b = int(sense * k) % nth                         # the bins ahead of the tool, two behind
        theta_b = (b + 0.5) * 2 * math.pi / nth
        t_b = ((sense * theta_b) % (2 * math.pi)) / abs(omega_s)      # first passage of the bin centre
        z_prev = [p0[2] + vz * (t_b - j * T) for j in range(1, 7)]
        polar.r_in_m[b], polar.r_out_m[b], polar.gap_lo_m[b], polar.gap_hi_m[b] = \
            previous_surface(ins, r_tip, z_prev, z_rows)
    tool, table = getattr(pose_rec, "cut_tool", tc.tool), getattr(pose_rec, "cut_table", tc.table)
    ref = Driver(tool, table, nt, ne, rotating=False, polar=polar, rho_kg_m3=drv.rho)
    w = np.array([0.0, 0.0, -omega_s])

    def pose_fn(t):
        Rz = _rz(-omega_s * t)
        q = p0 + np.array([0.0, 0.0, vz * t])
        x = Rz @ q
        v = np.cross(w, x) + Rz @ np.array([0.0, 0.0, vz])
        return Rz @ R0_T, x, v, w

    n = int(round(frac * T / dt))
    ref.run_turning(n, omega_s, dt, pose_fn, update_field=False)
    W = np.array(ref.record.wrench)
    return W[:, :3].mean(axis=0), ref


def test_pass_ends_and_volume_is_exact(turn_run):
    drv, res, steady = turn_run
    assert res.finished
    assert not [f for f in res.findings if f.severity.value in ("fatal", "error")]
    assert res.ledger["time"]["ok"] and res.ledger["time"]["residual_ticks"] == 0
    en = res.ledger["energy_run"]
    assert abs(en["in_J"] - en["out_J"]) <= 1e-9 * en["abs_J"]
    pf = drv.state.removal.workpieces[0].polar
    assert pf is not None
    # the booked removal is the polar field's own change (exact radial quanta)
    assert res.removed_volume_m3 == pytest.approx(res.field_volume_change_m3, rel=1e-12)
    assert res.removed_volume_m3 > 0.0
    assert res.chip_mass_kg == pytest.approx(res.density_kg_m3 * res.removed_volume_m3, rel=1e-12)


def test_steady_removal_rate_against_the_ring(turn_run):
    drv, res, steady = turn_run
    assert len(steady) >= 30
    dt_step = (steady[0].tick1 - steady[0].tick0) * drv.ctx.tick_s
    rate = float(np.mean([st.removed_m3 for st in steady])) / dt_step
    r = float(np.mean([st.r_tip for st in steady]))
    vz = abs(float(np.mean([st.vz for st in steady])))
    f = vz * 2 * math.pi / abs(steady[0].pose[2])
    assert f == pytest.approx(HF.TURN_FEED_MM_REV * 1e-3, rel=2e-3)
    r_nose = 0.8e-3
    ring = math.pi * (R0 ** 2 - r ** 2) * vz
    scallop = 2 * math.pi * r * vz * f ** 2 / (24.0 * r_nose)       # mean height of the nose-arc feed marks
    print("removal rate", rate, "ring", ring, "less scallops", ring - scallop, "ap", R0 - r)
    cpu = sum(st.cpu.get("macro", 0.0) for st in steady) / (len(steady) * dt_step)
    l0 = sum(st.cpu.get("l0_steps", 0.0) for st in steady)
    print("run_macro CPU s per simulated s in cut", cpu, "us per L0 step",
          1e6 * sum(st.cpu.get("macro", 0.0) for st in steady) / max(l0, 1.0))
    assert R0 - r == pytest.approx(AP_MM * 1e-3, abs=10e-6)
    assert rate == pytest.approx(ring - scallop, rel=4e-3)


def test_chip_of_every_element_against_the_earlier_revolutions(turn_run):
    """The loop's chip thickness per element (the polar field cut every L0 step, the view one bin behind) against
    the depth along -n into the exact steady surface of the earlier revolutions."""
    drv, res, steady = turn_run
    last = steady[-1]
    h = last.h
    tool = drv.state.cutting.tools[0].tool
    ins = drv.hot.last["rm"].inserts[0]
    R_S_T0, tip, omega_s, dt = last.pose
    r_tip = float(math.hypot(tip[0], tip[1]))
    f = abs(float(np.mean([st.vz for st in steady]))) * 2 * math.pi / abs(omega_s)

    def material(x_T, z_T):
        if r_tip - x_T > R0:
            return False
        for k in range(1, 7):
            lo, hi = _insert_x(ins, z_T - k * f)
            if np.isfinite(lo) and lo <= x_T <= hi:
                return False
        return True

    worst = 0.0
    n_eng = 0
    for e in range(tool.p_T_m.shape[0]):
        px, pz = float(tool.p_T_m[e, 0]), float(tool.p_T_m[e, 2])
        nx, nz = float(tool.n_T[e, 0]), float(tool.n_T[e, 2])
        nn = math.hypot(nx, nz)
        ana = 0.0
        if material(px, pz):
            while ana < 2e-3 and material(px - ana * nx / nn, pz - ana * nz / nn):
                ana += 0.25e-6
        if h[e] > 0.0 or ana > 0.0:
            n_eng += 1
            worst = max(worst, abs(h[e] - ana) / max(ana, 20e-6))
    print("engaged elements", n_eng, "worst chip error", worst)
    assert n_eng >= 15
    assert worst < 0.05


def test_forces_against_the_stand_alone_turning_driver(turn_run):
    drv, res, steady = turn_run
    W = np.concatenate([st.wrench_T0[0] for st in steady])
    mean = W[:, :3].mean(axis=0)
    vz = float(np.mean([st.vz for st in steady]))
    mid = steady[len(steady) // 2]
    ref_mean, ref = turning_reference(drv, mid, vz)
    print("turning mean F T0 (N)", mean, "reference", ref_mean)
    F = float(np.linalg.norm(mean))
    assert F > 500.0
    assert np.allclose(mean, ref_mean, atol=0.01 * float(np.linalg.norm(ref_mean)))
    # tooling's frame: the cutting force opposes the tool's velocity relative to the work, the feed force pushes
    # the tool back against its feed (it feeds along -z_T), the passive force out of the work (x_T points in)
    R, tip, omega_s, dt = mid.pose
    v_T = R.T @ np.cross([0.0, 0.0, -omega_s], tip)
    assert float(mean @ v_T) < 0.0
    assert mean[2] > 0.0 and mean[0] < 0.0
    # the spindle does positive work (the cut runs along the tool's rake face)
    assert sum(st.energy.get("in_J", 0.0) for st in steady[-1:]) > 0.0


@pytest.fixture(scope="module")
def turn_dyn_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("turn_dyn")
    drv = OperationDriver(HF.turn_project(tmp / "run", D_MM, L_MM, AP_MM, Z1_MM), use_dynamics=True)

    def observe(d, rec):
        tips = np.asarray(d.state.bus_next.motion.paths[0].tip_S_m)
        rec.z_range = (float(tips[:, 2].min()), float(tips[:, 2].max()))
    drv.observers.append(observe)
    res = HF.run_until_done(drv)
    steady = [st for st in drv.steps if st.in_cut_s >= 0.0099
              and STEADY_Z[0] * 1e-3 < st.z_range[0] and st.z_range[1] < STEADY_Z[1] * 1e-3]
    return drv, res, steady


def test_with_the_modes_the_steady_chip_is_the_rigid_one(turn_run, turn_dyn_run):
    """Tool and turret modes on (the mounted frame's deflections exchanged with dynamics): the previous
    revolution was cut with the same static deflection, so the steady forces stay the rigid ones up to the
    depth the radial deflection takes off; energy identity and exact polar volume hold."""
    drv, res, steady = turn_dyn_run
    _, res_r, steady_r = turn_run
    assert res.finished and len(steady) >= 30
    en = res.ledger["energy_run"]
    assert abs(en["in_J"] - en["out_J"]) <= 1e-9 * en["abs_J"]
    assert res.removed_volume_m3 == pytest.approx(res.field_volume_change_m3, rel=1e-12)
    F = np.concatenate([st.wrench_T0[0] for st in steady])[:, :3].mean(axis=0)
    F_r = np.concatenate([st.wrench_T0[0] for st in steady_r])[:, :3].mean(axis=0)
    dv = res.removed_volume_m3 / res_r.removed_volume_m3 - 1.0
    print("steady F with modes", F, "rigid", F_r, "removed volume change", dv)
    assert dv < 0.0 and abs(dv) < 0.02                  # the tool deflects out of the work
    assert np.allclose(F, F_r, atol=TOL_DYN * float(np.linalg.norm(F_r)))
