"""Every calibration target recovers known parameters from noisy synthetic measurements (analytic surrogates)."""

import math

import numpy as np
import pytest

from cnctwin.mdata import surrogates as S
from cnctwin.mdata.align import Alignment
from cnctwin.mdata.calibrate import run_target
from cnctwin.mdata.log import MachineLog
from cnctwin.mdata.targets import BUILDERS, Context, Skip

import mdatakit

OMEGA_MAX = mdatakit.OMEGA_MAX


def _fit(ctx, target, tmp_path):
    problem = BUILDERS[target](ctx)
    outcome = run_target(problem, ctx, "surrogate", out_dir=tmp_path, project_path=None, report=lambda **k: None)
    return {p["name"]: p for p in outcome["fit"].params}, outcome


def _trapezoid(t, moves):
    """Velocity of an axis doing moves [(t_start, duration, speed, accel)] with trapezoidal ramps."""
    v = np.zeros_like(t)
    for t0, duration, speed, accel in moves:
        ramp = abs(speed) / accel
        tau = t - t0
        inside = (tau >= 0) & (tau <= duration)
        profile = np.minimum.reduce([np.full_like(tau, abs(speed)), accel * tau, accel * (duration - tau)])
        v = np.where(inside, np.sign(speed) * np.maximum(profile, 0.0), v)
    return v


def test_servo_gain_and_feedforward(resolved_project, tmp_path):
    t = np.arange(0.0, 3.0, 1e-3)
    v = _trapezoid(t, [(0.1, 0.8, 0.2, 2.0), (1.2, 0.6, -0.15, 3.0), (2.0, 0.8, 0.3, 4.0)])
    x_cmd = -0.381 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    # the machine's servo responds to the commanded path the log records (its velocity by differences)
    e = S.following_error(t, np.gradient(x_cmd, t), kv_1_s=33.0, feedforward=0.7)
    rng = np.random.default_rng(11)
    log = MachineLog()
    log.add("axis.X.pos_cmd_m", t, x_cmd)
    log.add("axis.X.following_error_m", t, e + rng.normal(0, 0.5e-6, len(t)))
    ctx = Context(project=resolved_project, log=log)
    params, outcome = _fit(ctx, "servo", tmp_path)
    assert params["axis_X_kv_1_s"]["prior"] == pytest.approx(40.0)
    assert params["axis_X_kv_1_s"]["value"] == pytest.approx(33.0, rel=0.01)
    assert params["axis_X_feedforward"]["value"] == pytest.approx(0.7, abs=0.005)
    lo, hi = params["axis_X_kv_1_s"]["ci"]
    assert lo < 33.0 < hi
    assert outcome["fit"].metrics["axis.X.following_error_m"]["after"]["rmse"] < 0.6e-6


def test_axis_friction(resolved_project, tmp_path):
    t = np.arange(0.0, 4.0, 2e-3)
    log = MachineLog()
    truth = {"seal": 55.0, "mu": 0.008}
    rng = np.random.default_rng(12)
    machine = resolved_project.machine
    for name, moves in (("X", [(0.2, 1.0, 0.3, 3.0), (1.6, 1.2, -0.2, 2.0)]), ("Z", [(0.5, 1.5, -0.1, 1.0), (2.5, 1.2, 0.25, 2.0)])):
        v = _trapezoid(t, moves)
        x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
        a = np.gradient(np.gradient(x, t), t)
        mass = float(machine.axes[name].moving_mass_kg)
        lift = 1.0 if name == "Z" else 0.0
        normal = 0.0 if name == "Z" else mass * 9.80665
        force = S.axis_force(np.gradient(x, t), a, mass_kg=mass, lift_frac=lift, friction_coeff=truth["mu"],
                             normal_load_N=normal, seal_drag_N=truth["seal"])
        log.add(f"axis.{name}.pos_machine_m", t, x)
        log.add(f"axis.{name}.load_frac", t, force / float(machine.axes[name].motor_force_N)
                + rng.normal(0, 1e-3, len(t)))
    ctx = Context(project=resolved_project, log=log)
    params, _ = _fit(ctx, "axis_friction", tmp_path)
    assert params["axis_seal_drag_N"]["value"] == pytest.approx(55.0, rel=0.05)
    assert params["axis_friction_coeff"]["value"] == pytest.approx(0.008, rel=0.1)


def test_thermal_growth(resolved_project, tmp_path):
    hours = np.arange(0.0, 4 * 3600.0, 60.0)
    speed = np.select([hours < 3600, hours < 7200, hours < 9000], [OMEGA_MAX, 0.5 * OMEGA_MAX, 0.0], 0.8 * OMEGA_MAX)
    growth = S.spindle_growth(hours, speed, growth_max_m=32e-6, tau_s=1200.0, omega_max_rad_s=OMEGA_MAX,
                              exponent=5 / 3, initial_m=3e-6)
    probes = slice(0, None, 5)
    log = MachineLog()
    log.add("spindle.speed_rad_s", hours, speed)
    log.add("spindle.growth_m", hours[probes], growth[probes] + np.random.default_rng(13).normal(0, 0.4e-6,
                                                                                                   len(hours[probes])))
    ctx = Context(project=resolved_project, log=log)
    params, _ = _fit(ctx, "thermal_growth", tmp_path)
    assert params["spindle_growth_max_m"]["value"] == pytest.approx(32e-6, rel=0.03)
    assert params["spindle_thermal_tau_s"]["value"] == pytest.approx(1200.0, rel=0.05)
    assert params["spindle_growth_initial_m"]["value"] == pytest.approx(3e-6, abs=1e-6)
    assert params["spindle_growth_max_m"]["targets"][0]["path"] == "machine.spindles[main].thermal.growth_at_max_speed_um"


def test_tool_life_taylor(resolved_project, tmp_path):
    rng = np.random.default_rng(14)
    vc = np.repeat([250.0, 350.0, 500.0, 700.0], 4)                  # m/min
    life = 1.0 * (1500.0 / vc) ** (1 / 0.32) * np.exp(rng.normal(0, 0.12, len(vc)))
    rows = [{"vc_m_min": float(v), "life_min": float(T)} for v, T in zip(vc, life)]
    ctx = Context(project=resolved_project, options={"tool_life": rows})
    params, outcome = _fit(ctx, "tool_life", tmp_path)
    # C is the speed for T = 1 min, an extrapolation from lives of 10 to 300 min: its interval is wide
    lo, hi = params["taylor_C_m_s"]["ci"]
    assert lo * 60 < 1500.0 < hi * 60 and hi / lo < 1.5
    lo, hi = params["taylor_n"]["ci"]
    assert lo < 0.32 < hi and params["taylor_n"]["value"] == pytest.approx(0.32, abs=0.03)
    predicted = (params["taylor_C_m_s"]["value"] * 60 / 400.0) ** (1 / params["taylor_n"]["value"])
    assert predicted == pytest.approx((1500.0 / 400.0) ** (1 / 0.32), rel=0.1)      # inside the data: 10 %
    assert any(c["a"] == "taylor_C_m_s" for c in outcome["fit"].confounded)          # C and n correlate
    assert params["taylor_a"]["free"] is False                         # feed not varied
    assert outcome["problem"].inputs["entry"] == "tool_class=carbide,coating=uncoated"


def test_usui_wear(resolved_project, tmp_path):
    A, B = 4e-15, 5200.0
    clearance, rake = math.radians(10), math.radians(8)
    rows = []
    rng = np.random.default_rng(15)
    for test, temp_K in (("cold", 820.0), ("hot", 1010.0)):
        t = np.arange(1, 11) * 120.0
        vb = S.usui_flank_wear(np.concatenate([[0.0], t]), sigma_n_Pa=np.full(11, 1.1e9), sliding_m_s=np.full(11, 3.0),
                               temp_K=np.full(11, temp_K), A_per_Pa=A, B_K=B, clearance_rad=clearance, rake_rad=rake)[1:]
        for ti, v in zip(t, vb):
            rows.append({"test": test, "t_min": ti / 60, "VB_mm": float(v * 1e3 * (1 + rng.normal(0, 0.02))),
                         "sigma_n_MPa": 1100.0, "sliding_m_min": 180.0, "temp_degC": temp_K - 273.15})
    ctx = Context(project=resolved_project, options={"wear": rows,
                                                     "target_options": {"usui": {"clearance_deg": 10, "rake_deg": 8}}})
    params, outcome = _fit(ctx, "usui", tmp_path)
    assert params["usui_B_K"]["value"] == pytest.approx(B, rel=0.05)
    assert params["usui_A_per_Pa"]["value"] == pytest.approx(A, rel=0.3)
    assert outcome["problem"].constants  # the new Usui entry's rate_unit


def test_deflection_runout_and_stiffness(resolved_project, tmp_path):
    rng = np.random.default_rng(16)
    sens = np.array([2.0, 2.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0])
    defl = np.array([0.0, -15e-6, -30e-6, -8e-6, 12e-6, -25e-6, -5e-6, 0.0])
    simulated = np.array([10.0, 10.0, 20.0, 15.0, 8.0, 12.0, 30.0, 5.0]) * 1e-3
    truth = S.part_dimension(simulated, runout_sensitivity=sens, deflection_m=defl, runout_m=8e-6, runout_sim_m=5e-6,
                             stiffness_factor=0.8)
    rows = [{"feature": f"f{i}", "measured_mm": float(m * 1e3 + rng.normal(0, 0.3e-3)), "simulated_mm": float(s * 1e3),
             "runout_sensitivity": float(c), "deflection_mm": float(d * 1e3)}
            for i, (m, s, c, d) in enumerate(zip(truth, simulated, sens, defl))]
    ctx = Context(project=resolved_project, options={"part_measurements": rows})
    params, _ = _fit(ctx, "deflection", tmp_path)
    assert params["tool_runout_m"]["prior"] == pytest.approx(5e-6)
    assert params["tool_runout_m"]["value"] == pytest.approx(8e-6, abs=0.5e-6)
    assert params["tool_stiffness_factor"]["value"] == pytest.approx(0.8, abs=0.03)
    assert params["tool_stiffness_factor"]["targets"][0]["path"] == "factors.tool_stiffness[1]"


def _cut_log(cuts, power_of, eta=0.85, idle=700.0, noise=20.0, seed=17):
    log = MachineLog()
    rng = np.random.default_rng(seed)
    for cut in cuts:
        t = np.arange(cut["t0_s"], cut["t1_s"], 0.01)
        omega = cut["rpm"] * 2 * np.pi / 60
        shaft = power_of(cut)
        power = shaft / eta + idle * (omega / OMEGA_MAX) ** 1.6667 + rng.normal(0, noise, len(t))
        log.add("spindle.power_W", t, power)
    return log


def _cuts():
    cuts, t = [], 0.0
    for fz in (0.03, 0.06, 0.1, 0.15):
        for ae in (2.0, 6.0):
            rpm, teeth, D = 9000.0, 3, 10.0
            cuts.append({"t0_s": t, "t1_s": t + 2.0, "ae_mm": ae, "ap_mm": 3.0, "rpm": rpm,
                         "feed_mm_min": fz * teeth * rpm, "diameter_mm": D, "teeth": teeth})
            t += 2.5
    return cuts


def test_kienzle_from_cut_power(resolved_project, tmp_path):
    cuts = _cuts()

    def shaft(cut):
        n = cut["rpm"] / 60
        return S.milling_power_kienzle(spindle_rev_s=n, teeth=cut["teeth"], radius_m=cut["diameter_mm"] / 2e3,
                                       ap_m=cut["ap_mm"] * 1e-3, fz_m=cut["feed_mm_min"] / (cut["teeth"] * cut["rpm"]) * 1e-3,
                                       ae_m=cut["ae_mm"] * 1e-3, kc11_Pa=900e6, mc=0.22, h_ref_m=1e-3)

    ctx = Context(project=resolved_project, log=_cut_log(cuts, shaft), options={"cuts": cuts})
    params, _ = _fit(ctx, "kienzle", tmp_path)
    assert params["kc11_Pa"]["prior"] == pytest.approx(800e6)
    assert params["kc11_Pa"]["value"] == pytest.approx(900e6, rel=0.03)
    assert params["kc_mc"]["value"] == pytest.approx(0.22, abs=0.02)


def test_mechanistic_from_cut_power(resolved_project, tmp_path):
    from cnctwin.project import resolve

    project = mdatakit.write_project(tmp_path / "p7075")
    project.write_text(project.read_text().replace("id: al6061-t6", "id: al7075-t6"))
    resolved = resolve(project)
    cuts = _cuts()

    def shaft(cut):
        return S.milling_power_mechanistic(spindle_rev_s=cut["rpm"] / 60, teeth=cut["teeth"],
                                           radius_m=cut["diameter_mm"] / 2e3, ap_m=cut["ap_mm"] * 1e-3,
                                           fz_m=cut["feed_mm_min"] / (cut["teeth"] * cut["rpm"]) * 1e-3,
                                           ae_m=cut["ae_mm"] * 1e-3, Ktc_Pa=1100e6, Kte_N_per_m=30e3)

    ctx = Context(project=resolved, log=_cut_log(cuts, shaft, noise=10.0), options={"cuts": cuts})
    params, _ = _fit(ctx, "mechanistic", tmp_path)
    assert params["Ktc_Pa"]["value"] == pytest.approx(1100e6, rel=0.03)
    assert params["Kte_N_per_m"]["value"] == pytest.approx(30e3, rel=0.15)
    with pytest.raises(Skip, match="no mechanistic entry"):
        BUILDERS["mechanistic"](Context(project=resolved_project, log=ctx.log, options={"cuts": cuts}))  # 6061


class FakeRun:
    """A timeline of many short and a few long blocks (block durations known)."""

    def __init__(self, motion_s, block_s, lookahead, tick_s=1e-6):
        self.tick_s = tick_s
        start, end = S.block_queue(motion_s, block_s=block_s, lookahead=lookahead)
        self.blocks = []
        for i, (a, b, m) in enumerate(zip(start, end, motion_s)):
            tick0, tick1 = int(round(a / tick_s)), int(round(b / tick_s))
            self.blocks.append({"index": i, "line": 10 + i, "tick0": tick0, "tick1": tick1, "kind": "linear",
                                "ticks": {"cutting": int(round(m / tick_s)),
                                          "data_starvation": tick1 - tick0 - int(round(m / tick_s))}})
        self.dir = "."

    def timeline(self):
        return self.blocks


def test_block_processing_time(resolved_project, tmp_path):
    rng = np.random.default_rng(18)
    motion = np.concatenate([np.full(5, 0.05), rng.uniform(0.2e-3, 0.8e-3, 400), np.full(5, 0.08),
                             rng.uniform(0.5e-3, 3e-3, 300)])
    lookahead = int(resolved_project.machine.controller.lookahead_blocks)
    run = FakeRun(motion, 1e-3, lookahead)                           # simulated with the machine file's 1 ms
    true_start, _ = S.block_queue(motion, block_s=2.4e-3, lookahead=lookahead)
    observed = np.arange(0, len(motion), 7)                          # the machine log saw every 7th block start
    poll = 1e-3
    t_meas = np.ceil((true_start[observed] + 0.3) / poll) * poll      # machine clock 0.3 s ahead, polled
    sim_starts = np.array([b["tick0"] * run.tick_s for b in run.blocks])
    alignment = Alignment(-0.3, 1.0, "markers", {"pairs": [[float(a), float(sim_starts[j])]
                                                           for a, j in zip(t_meas, observed)],
                                                 "poll_interval_s": poll})
    ctx = Context(project=resolved_project, log=MachineLog(), run=run, alignment=alignment)
    params, outcome = _fit(ctx, "block_time", tmp_path)
    assert params["block_processing_s"]["prior"] == pytest.approx(1e-3)
    assert params["block_processing_s"]["value"] == pytest.approx(2.4e-3, rel=0.01)
    assert params["block_processing_s"]["targets"][0]["path"] == "machine.controller.block_processing_ms"


def test_targets_skip_without_their_data(resolved_project):
    ctx = Context(project=resolved_project, log=MachineLog())
    for name in BUILDERS:
        with pytest.raises(Skip):
            BUILDERS[name](ctx)
