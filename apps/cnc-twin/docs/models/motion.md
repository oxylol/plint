# Motion domain (`cnctwin/motion/`)

Look-ahead planner, interpolation, feed drives, spindles, kinematics and timing of every controller
path. Motion turns the controller's IR commands (`cnctwin/controller/ir.py`, each with its
`ready_tick`) into commanded axis values per L1 tick, runs the feed drives and spindles in the
compiled L1 loop, computes the tool and workpiece poses through the machine's kinematic chains (with
the ISO 230-1 error map and the thermal and holding error transforms), and books every executed L1
tick into the timeline and the cycle-time ledger. It is an L1 domain: `kernel_inputs`/`absorb` hand
the hot loop the setpoints and the drive states, and `motion_tick` runs inside it.

| file | what |
|---|---|
| `abi.py` | kernel layouts (NamedTuples of arrays): `Chain`, `Servo`/`ServoState`, `Spindle`/`SpindleState`, `Setpoint`, `LoopState`, `MotionOut`, `MotionInputs`; sync, alarm and mode codes |
| `kernels.py` | Numba kernels (one explicit signature each): `motion_tick`, `run_motion`, `replan_before`, FK/IK, drive and spindle steps, geometry, look-ahead generator; `reference.py` their plain-Python twins |
| `machine.py` | `MachineModel`: axes and drives, paths, chains, spindles, error map from a machine file (the axis entries of `axes`; the source keys beside them, `src` `note` `notes` `page` ..., are skipped) |
| `ik.py` | `PathKinematics`: FK, closed-form + numeric IK, singularities, tilted-plane solutions |
| `geometry.py` | path pieces (joint lines/arcs, TCP tip pieces, corner blends) and their limits |
| `scurve.py` | analytic 7-segment double-S profiles (Biagiotti & Melchiorri 2008) |
| `planner.py` | per-path planner: queue, pieces, phase machine, commit by snapshot |
| `servo.py` | drive and spindle arrays, friction calibration, the circle closed form |
| `spindle.py` | spindle commands, speed prediction, CSS, the per-step spindle timeline |
| `timing.py` | tool change, turret, M functions, coolant, doors, pallets, chucks, dwell, operator stops |
| `ledger.py` | timeline records and the cycle-time categories |
| `domain.py` | `MotionDomain` (core.domain.Domain), bus record, findings, checkpoint |
| `driver.py` | `MotionRunner`: controller + motion without cutting (reference loop protocol) |
| `state.py`, `schema.py`, `channels.py` | SimState field, `physics.motion` settings, channels |

Model coefficients: `db/models/motion.yaml` (read through `cnctwin.db.models.params("motion")`),
sources in `db/bibliography/models-motion.yaml` plus the shared keys cited below. Machine-file
values always win; every default used for a machine is listed in `MachineModel.estimated` and in
`MotionDomain.finalize()["estimated"]`. RNG streams: **none**; the model is deterministic (no
stochastic phenomena, no sampling).

Units: SI everywhere (m, rad, s, N, N m, kg); rotary axes in rad. Where a control mixes linear and
rotary quantities (combined feeds, rotary in-position bands, blend tolerances), 1 deg counts as
1 mm: `rotary_length_per_rad_m` = 57.2958 mm per rad (FANUC/Haas rule).

## 1. Clock, loop and commit

* Base tick 1 us (`tick_s`); L1 tick `ipo_ticks` base ticks (the control's interpolation period,
  1 ms by default); macro step `macro_ipo` L1 ticks (10). The setpoint of tick k is the value at the
  **end** of tick k.
* Per macro step every path is planned for ticks `k_from .. n-1` (`kernel_inputs`); the compiled
  loop runs `motion_tick` per L1 tick and returns at the first sync event; `absorb` commits ticks
  `k_from .. k_stop`; the next call re-plans from `k_stop + 1` with the actual states.
* **Sync codes** (`abi.py`): `QUEUE_EMPTY` (a re-plan point: the plan ran out, or a phase needs the
  actual state), `IN_POSITION`, `PROBE_TRIGGER` (skip or tool setter), `ALARM` (overtravel,
  excess following error), `PATH_WAIT` (multi-path wait reached), `SEGMENT_END` (end of a
  spindle-synchronous stretch).
* **Pre-tick re-plan** (`replan_before`): a path whose setpoint is `SP_END` at tick k > k_from
  makes the loop return *before* tick k (code `QUEUE_EMPTY` at k - 1); a re-plan point costs no
  simulated time.
* **Commit by snapshot**: every planned tick stores a copy of the plan state; a sync at k_stop
  restores the snapshot of k_stop, so only the ticks actually executed change persistent state.
  Events (block start/done, ideal times, spindle and coolant changes, feedback) carry their boundary
  and are committed with the snapshot of the tick that emitted them (`TickPlan.ev_mark`): an item that
  ends at the end of tick k and one that starts (and, without motion, ends) at the start of tick k + 1
  share boundary k + 1, and only the first belongs to the snapshot of tick k. A live read that starts
  inside a step therefore answers the controller once (a duplicate answered the next buffer stop).

## 2. Planner

### 2.1 Pieces and limits (`geometry.py`)

Motion commands become pieces when they are accepted; a piece has one geometry and one set of
limits. A spindle's C axis outside C-axis mode turns with the spindle (its command is the spindle
angle), so it is not part of the geometry of items that do not program it: an item is rebuilt from the
commanded position only when its other axes moved (a blended item rebuilt up to its blend keeps the
spindle angle on C).

* **Joint pieces**: lines `q = q0 + dq u` and arcs (circular, helical, spiral within the control's
  radius tolerance) `q = c + r(u)(cos(phi u) e1 + sin(phi u) e2) + w u`, evaluated by the compiled
  `geo_eval` (also inside the loop for synchronous stretches).
* **TCP pieces**: the tool tip moves on a line or arc in the workpiece frame W. "rotary" mode (the
  default with programmed rotary positions: FANUC G43.4, Haas G234, TRAORI with B/C, Heidenhain
  M128): the rotary axes are linear in u and the linear axes solve the tip onto the path; "vector"
  mode (`modal["tcp_type"] == "vector"`, G43.5): the tool axis turns on the great circle (slerp,
  Shoemake 1985) and every tick is a full IK solve; a start at a singular orientation is preceded
  by turning the free rotary axis there with the tip still. TCP pieces are built with the tool that
  will be in the spindle when they run (a queued tool change changes the tip).
* **Path metric** (what F applies to): FANUC/Haas the vector of linear (m) and rotary (1 deg = 1 mm)
  axes; Siemens/Heidenhain the linear axes when any moves (rotary axes follow); TCP the tip path in
  W.
* **Limits** of a piece, with `dq_i/ds` the axis derivatives along the path:

      v <= F x override (<= machine max feed),   v <= v_i / max|dq_i/ds|
      a <= A_i / max|dq_i/ds|,                   j <= J_i / max|dq_i/ds|
      v <= sqrt(A_i / max|d2q_i/ds2|),           v <= (J_i / max|d3q_i/ds3|)^(1/3)

  On a circle of radius R in the plane of axes i the last two are `v <= sqrt(A R)` and
  `v <= (J R^2)^(1/3)` (centripetal). Feed and override are applied at plan time (feed_v and the
  axis/curvature limit limit_v are stored separately), so an override change acts on the next
  plan call. Per-revolution pieces keep their feed per revolution and move at `f x |omega_c| / 2 pi`
  with omega_c the cutting spindle's commanded speed at plan time (CSS evaluated at the current
  radius).
* **Tolerance-driven limits** of curved feed pieces (programmed arcs, corner blends, TCP arcs; radius
  R of the path, a helix `r (1 + (c/r)^2)` with c its lead per radian), with e the contour tolerance in
  force (2.2):

      chord at the IPO period T:   v <= 2 sqrt(2 R e - e^2) / T  (~ sqrt(8 R e) / T, e capped at R)
      axis setpoint filter T_f:    v <= sqrt(24 R e) / T_f

  The control hands the drives one point per interpolation period and the path between two points is
  their chord, which lies `R - sqrt(R^2 - (v T / 2)^2)` inside the curve (Suh et al. 2008; Yeh & Hsu 2002);
  the twin's commanded points stay on the curve, so this bound is what keeps the chords in the band. A
  control that filters its axis setpoints after interpolation (a moving average of length T_f: FANUC
  linear acc/dec after interpolation, parameter 1769 in AI contour mode; the TNC nominal-position / HSC
  filter; SINUMERIK MD32410 axial jerk filter) pulls a circle in by `v^2 T_f^2 / (24 R)` (FANUC 30i-B
  operator's manual, radius error with acceleration/deceleration after interpolation; Biagiotti &
  Melchiorri 2012) and slows until that stays within e. T_f is the machine file's
  `controller.setpoint_filter_ms` (default 0: jerk limited before interpolation only, as FANUC AI contour
  control, Haas HSM and SINUMERIK SOFT document; no control publishes a filter time, see 14).

### 2.2 Corners

A tangent discontinuity of angle theta between two lines in continuous-path mode with tolerance
eps is blended by a circular arc in the scaled joint space (1 deg = 1 mm):

    R = eps cos(theta/2) / (1 - cos(theta/2)),   l = R tan(theta/2)

with the tangent length l at most `blend_length_frac` (half) of each adjacent piece (the radius
shrinks accordingly so the deviation from the corner stays <= eps). The blend's centripetal limits
give the corner speed `v <= min_i( sqrt(A_i R / rho_i), (J_i R^2 / rho_i)^(1/3) )`, rho_i the
axis share of the curvature, and the blend arcs carry the tolerance-driven limits of 2.1. Corners of
arcs and TCP pieces keep their corner; the node speed is bounded by the same arc bound and by the
per-axis velocity jump `v <= A_i dt / |delta(dq_i/ds)|`.

**Curvature steps**: where tangent pieces meet with different curvature (a line and an arc, two arcs,
an S-turn) each axis' acceleration steps by `v^2 |delta(d2q_i/ds2)|`. The node speed keeps the step within
the axis acceleration, `v <= min_i sqrt(A_i / |delta(d2q_i/ds2)|)` (the same allowance as the velocity
step of one interpolation period at a corner); with a setpoint filter the step is spread over T_f at a jerk
within J_i only up to `J_i T_f`, so `min(A_i, J_i T_f)` then. A line meeting an arc is no slower than the
arc itself; an S-turn of radius R is bounded by `sqrt(A R / 2)`.

**Tolerance in force** (e above and the blend tolerance eps), each control's own high-speed mode, from
the program and the machine file: Heidenhain cycle 32 T (ISO G62), FANUC G05.1 Q1 / G08 / smooth
tolerance (the machine's smoothing table entry of the code), SINUMERIK CYCLE832 S_TOL (G642 + COMPCAD;
`_TOL` 0: the machine's entry of the technology), Haas G187 E, else Setting 85 Maximum Corner Rounding
(the machine file's G187 entry; also at power-on, after G187 alone, RESET and M30). Without one,
`default_contour_tolerance_m` (0.01 mm). Exact stop (G09/G61, `smoothing.corner == "exact_stop"`)
stops at the end of the block and waits for in-position.

**What the tolerance bounds** (`Smoothing.tolerance_kind`, set by the dialect; 2026-11-09):

| control, setting | kind | blend at a corner of turn theta | source of the definition |
|---|---|---|---|
| Heidenhain cycle 32 T (G62) | deviation | `R = T cos(theta/2) / (1 - cos(theta/2))`: the arc passes the corner at distance T | db/controllers/heidenhain-tnc640.yaml (T: the permitted contour deviation) |
| SINUMERIK CYCLE832 S_TOL, G642 | deviation | as above with S_TOL (the contour tolerance) | db/controllers/siemens-840d-sl.yaml (S_TOL contour tolerance, siemens-840dsl-jobplan-ru-2018 pp. 825-827) |
| SINUMERIK G641 ADIS | distance | tangent length `l = ADIS` (the rounding starts ADIS before the transition and ends ADIS after it), `R = ADIS / tan(theta/2)` | db/controllers/siemens-840d-sl.yaml (G641: smoothing with a programmable rounding distance ADIS) |
| FANUC G05.1 Q1 (AI contour control) | deviation | as above with the machine file's entry (est: the manual gives AICC no geometric tolerance, db/controllers/fanuc-31i-b.yaml) | machine file, est |
| Haas Setting 85, G187 E | radius | `R = min(E, R_needed(v))`: the corner is rounded only as far as the axes need at the feed, `R_needed = max_i(rho_i v^2 / A_i, sqrt(rho_i v^3 / J_i))` (the inverse of the arc's corner speed), up to the setting; where the feed needs more, the radius stays at the setting and the corner slows to that arc's speed | haas-ngc-mill-om-2020 p. 456: "the control keeps the radii of corners no bigger than" Setting 85 and "adjusts feeds around corners in all 3 axes to meet the tolerance value ... it might round the corner to a radius up to the tolerance value" (db/controllers/haas-ngc.yaml `corner_rounding_default_in`); factory default 0.0250 in = 0.635 mm (db/machines/haas-vf2ss.yaml `smoothing`) |

Heidenhain's M120 LA is the look-ahead of the radius-compensated contour (up to 99 blocks against undercuts,
db/controllers/heidenhain-tnc640.yaml `radius_comp_lookahead_blocks`), not a corner tolerance; the dialect does not
run it (TW0012, docs/dialects.md). The planner's own look-ahead window is `controller.lookahead_blocks` (2.3).

Until 2026-11-09 every tolerance was taken as a deviation: Haas's 0.635 mm became a 1.53 mm blend radius at a
square corner (2.4 x the documented radius limit) at any feed, and SINUMERIK's ADIS a rounding that started 2.4 x
ADIS before the transition. Measured on the convergence study's pocket corner (docs/convergence.md 2.2: D10, a 90 deg
internal corner at F1600 on the VF-2SS, Setting 85 = 0.635 mm; controller, planner and drives without cutting): before, the
commanded path passed the corner at 0.635 mm (blend radius 1.533 mm) and the actual tool path at 0.646 mm (the drives'
contour error and the error map add 11 um), beyond the tolerance itself; now the axes need R = 0.436 mm at that
feed (the jerk limit, sqrt(v^3 / J) with J = 100 m/s3), within the setting: commanded deviation 0.181 mm, actual
0.193 mm, no slow-down. At F500: R 0.076 mm, deviation 0.032 mm. At F6000 the radius stops at 0.635 mm (deviation
0.263 mm, actual 0.295 mm) and the corner slows from 100 to 33.8 mm/s. The tolerance bounds the control's commanded
path; the drives' contour error (machine file's velocity feed-forward 0.95, est) and the ISO 230 error map come on
top of it, as on the machine (12-31 um at these corners).

The commanded profile never passes a block's target: the generator clamps the path position to the
piece (and the window end), dog-leg rapids follow their analytic double-S to the target, and blends cut
corners inward, so a target on a stroke limit is reached exactly
(`test_commanded_profile_never_passes_the_block_target`).

### 2.3 Look-ahead generator (`plan_ticks`)

Per tick the jerk is the largest in [-J, J] that keeps the cruise approach to v_max and the braking
distance to every node ahead feasible:

    brake_distance(v, a -> u, 0) <= distance to node   for every node in the window

`brake_distance` is the time-optimal jerk-limited braking distance from (v, a) to (u, 0) (jerk -J to
the peak deceleration a_p = -sqrt((a^2 - 2 J dv) / 2), clipped at -A with a constant phase, then +J
back to 0; Biagiotti & Melchiorri 2008 s. 3.4), `track_jerk` the largest jerk whose "ramp the
acceleration to zero" end speed does not overshoot the cruise speed. The largest feasible jerk is
found by bisection: bang-bang jerk with exact switching, i.e. the 7-segment double-S profile,
re-planned every tick as blocks arrive. The window holds the pieces from the current one up to the
control's look-ahead (`controller.lookahead_blocks`) that are **ready** (`ready_tick` <= now); its end
is a stop node. Blocks prepared too slowly (short segments, slow block processing) therefore cap
the feed at the block rate: **data starvation**, reported per tick (`out_lim = 2`) and booked as
`data_starvation`.

* **Feed hold** plans v_max = 0 (the path brakes on the profile and stands, `feed_hold`); **feed
  override** scales F (not threads or taps); **rapid override** scales the rapid rates; **dry run**
  moves cutting feeds at the machine's maximum feed; **machine lock** holds the axes while the
  program is still timed.
* **Rapids**: linear interpolation at the slowest axis' rate (Siemens, Heidenhain), or dog-leg
  (FANUC parameter LRP = 0, Haas): each axis moves on its own double-S profile at its own rate and
  the block ends with the last axis (`default_rapid_mode_<dialect>`).
* **Continuous rotary axes** move the shortest way (half a turn: positive).

### 2.4 Spindle-synchronous motion

* **Feed per revolution** (G95/G99): the run becomes a stretch the loop evaluates from the *actual*
  spindle angle: the planner gives a table (theta, s) at the nominal speed, the loop latches theta0
  at the stretch start and sets `s = s_table(theta - theta0)` each tick (linear in the table, the
  last slope beyond). The table and the pieces' geometry (up to `setp_geo_capacity` = 64 pieces
  ahead) are rebuilt every plan call; a stretch ends when the actual s reaches the end of the run
  (`SEGMENT_END`).
* **Threading** (G32/G76/G92, CYCLE97, cycle 18): a one-revolution trigger (the start angle of the
  thread), then `s = lead x (theta - theta_trig) / 2 pi`, optionally through the control's
  exponential acceleration filter `thread_filter_tc_s` (FANUC parameter 1626; 0 by default: the
  start lag of the thread comes from the servo).
* **Rigid tapping** (M29/G84 rigid, CYCLE84, cycle 209): the spindle is position-controlled and its
  angle command follows the same profile as the tapping axis (pitch per revolution, reversed coming
  out); the tapping axis runs with the spindle's position-loop gain so the two lags match; the
  path speed and acceleration are bounded by the spindle's speed and position-mode acceleration
  (`pitch / 2 pi` per rad).

### 2.5 Phases (`planner.py`)

`MOTION` (the generator over a run of pieces up to the next stop), `TIMED` (non-motion commands,
dog-leg rapids, spindle arrival and C-axis engage waits), `WAIT_INPOS` (`SP_HOLD_WAIT`: the loop
reports in-position), `WAIT_CTRL` (path-wait release, more commands after a buffer stop),
`RESYNC` (a re-plan from the actual state), `ENDED`, `ALARM`. **Rebuilds**: a queued item whose start
no longer matches the commanded position (a tool change sent the tool axis home, the tool in the
spindle changed under a TCP block, a C-axis engage relabelled C) is rebuilt from there, in place, so
the pieces stay in path order; an item that ends in a corner blend is rebuilt up to the blend, which
stays, and the new corner is bounded by the axes' velocity step. The **ideal time** of every item
(length / programmed speed with the override in force, per-revolution feeds at the spindle speed)
is recorded when the planner first reaches it, also for blended blocks that never start on their
own.

## 3. Feed drives (`servo.py`, `servo_axis_step`)

Per axis (Altintas 2012 ch. 6; Erkorkmaz & Altintas 2001 part II):

    v_ref = Kv (x_c + c_b - x_meas) + k_ff dx_c/dt                  position loop + velocity feed-forward
    F_m   = sat(Kp_v (v_ref - v) + Ki_v int(v_ref - v) dt)           PI velocity loop, conditional anti-windup
    m dv/dt = F_m + F_ext + F_g - F_c tanh(v / v_s)                  moving mass, cutting load, gravity, friction

* `Kp_v = 2 zeta omega_v m`, `Ki_v = omega_v^2 m` (closed velocity loop of bandwidth
  omega_v = 2 pi `velocity_bw_Hz`, damping `velocity_loop_zeta` 0.7, Ellis 2012).
* **Backlash**: the load follows the motor through a dead band of width b (Nordin & Gutman 2002);
  the control adds `c_b = +-c/2` when the commanded direction reverses, `c = b (1 - r)` with the
  compensation residual r = `backlash_comp_residual_frac` (0.1), so the lost motion seen by the
  part is `b r` (ISO 230-2 reversal value after compensation).
* **Friction**: Coulomb friction regularised as `F_c tanh(v / v_s)` (Armstrong-Helouvry et al. 1994,
  `stribeck_velocity_m_s` 0.1 mm/s), calibrated so that its sign change at a velocity reversal (a
  force step 2 F_c) produces the machine's reversal spike (circular test, ISO 230-4):
  `F_c = spike / (2 h_peak)`, h_peak the peak position response of the closed loop to a unit force
  step (computed with the drive kernel itself), capped at `friction_max_frac` of the force limit.
* x_meas is the motor position (rotary encoder) or the load position (`scale_feedback`).
* Gravity on vertical axes (minus `gravity_counterbalanced_frac`), held by the integrator at rest.
* Integration: RK4 on n_sub sub-steps per L1 tick with the command fine-interpolated; n_sub from
  `omega x h <= servo_omega_h` and `h F_c / (m v_s) <= friction_h` (RK4 stability).
* **Spindle C axes** follow the spindle: `q_C = kappa theta + c_off` (section 4).
* **Steady contouring** (linear model): `x/x_c = (Kv + k_ff s) H / (s + Kv H)`,
  `H = (2 zeta w s + w^2)/(s^2 + 2 zeta w s + w^2)`; on a circle of angular speed omega_c the radius
  ratio is `|x/x_c|(j omega_c)` (`circle_radius_ratio`); without feed-forward the radius reduction
  tends to `v^2 / (2 R Kv^2)`. Steady following error at speed v: `v (1 - k_ff) / Kv`.
* **Findings**: the commanded position is checked against the stored stroke limits (widened by half
  the least input increment, the control's resolution) -> `overtravel` finding + the control's
  OVERTRAVEL alarm (controller database text) to the controller; `|e| > ferr_limit_factor x rapid /
  Kv` -> excess following error alarm; demanded force beyond the drive's force limit ->
  `axis_overload` finding (one per episode).

## 4. Spindles (`spindle_step`, `spindle.py`)

    J domega/dt = T_m - T_load - T_drag,      T_m = sat(Kp e + Ki int e dt) within T_max(gear, |omega|)
    T_drag = (P_idle / omega_max) (|omega| / omega_max)^(2/3)          (Palmgren 1959 viscous bearing moment)

with `Kp = 2 zeta omega_b J`, `Ki = omega_b^2 J` (`spindle_zeta`, speed-loop bandwidth
`position_loop.velocity_bw_Hz` else `default_spindle_bw_Hz`), T_max the gear range's torque curve
(digitised power/torque curves of the machine file). Modes:

* **Speed** (`SPM_SPEED`): the command ramps at `max_rpm / accel_time_s` (`decel_time_s`) through
  zero on a reversal; with no ramp time given, the torque limit sets the acceleration. **CSS**
  (G96): `omega = v_c / max(|x - x_centre|, css_min_radius)` clamped (G50/G92/LIMS) and to the
  maximum speed, evaluated in the loop from the radius axis command.
* **Position** (`SPM_POSITION`): `omega_ref = Kv (theta_c - theta) + k_ff omega_c`. **Orientation**
  (M19, SPOS): the spindle stops under speed control (the predicted deceleration), then the
  position command moves from the angle at the switch to the nearest angle equal to the programmed
  orientation angle (mod 2 pi) on a quintic `10 tau^3 - 15 tau^4 + 6 tau^5` (zero end speed and
  acceleration) over `orient_time_s`, and the spindle holds it until the next speed command.
  Rigid tapping uses the same mode.
* **C-axis** (`SPM_CAXIS`): the spindle is a position-controlled axis with the C axis' gains;
  its command is the C axis setpoint, `theta_c = (q_C - c_off) / kappa`. A motion block that moves
  a spindle's C axis while the spindle is not in C-axis mode (the program's C target differs from
  the last programmed C) **engages** it: the motion before stops in position, the spindle stops
  (predicted deceleration) and is referenced (the spindle's `c_axis_engage_time_s` when a machine
  file gives it, else its orientation time); then, at the first tick of the next plan call, the C
  coordinate continues from the spindle angle, relabelled by whole turns
  (`c_off = 2 pi round((q_C,queued - kappa theta) / 2 pi)`) so the queued blocks stay valid. Any
  spindle speed command or orientation returns the spindle to speed or position control.
  Kinematics are unaffected by the relabelling (whole turns); outside C-axis mode the C axis'
  command and actual value are the spindle's `kappa theta + c_off` (following error 0).
* **Speed arrival**: a cutting feed after a speed command waits until `|omega - omega_c| <=
  spindle_arrival_frac x |omega_c|` (FANUC SAR), booked as `spindle_ramp`.
* **Gear change**: stop, `gear_change_s`, then run-up; dwell in revolutions counts the actual speed.
* **Sign convention** (ISO 841): angles and speeds right-handed about the spindle reference axis +z
  (from the work toward the tool spindle; from the work toward the headstock for work spindles), so
  M03 gives omega < 0; a work spindle turns its chain's last rotary link with `q_C = kappa theta`,
  `kappa = (-z_F) . d_C`.

## 5. Kinematics (`machine.py`, `ik.py`, `fk_poses`)

Each path has a workpiece chain and a tool chain from the machine frame M. Link i:

    T_i = T_{i-1} J_i(q) E_i(q) X_i

J_i the joint motion (translation along d, or rotation about the line through point p with
direction d), X_i the fixed pose, E_i the 6-component error transform `[dx, dy, dz, rx, ry, rz]`
(translation, rotation vector), `E = e_const + p0 + xi (p1 + xi p2)` with `xi` the position in the
travel (0..1), plus the thermal screw strain `d x strain x (q - q_ref)` along linear links:

* **ISO 230-1 error map**: squareness tilts the second axis of each pair toward the first;
  straightness is a bow `4 D xi (1 - xi)` normal to the axis (D = value x travel); angular errors
  (pitch, yaw, roll) vary linearly from -e to +e over the travel; pivot errors move and tilt a
  rotary axis line; `y_axis_centre_height_um` offsets the Y link. Errors live in the chain, never in
  the axis values: the control's own model is the **nominal** chain (IK, TCP), the loop's FK uses
  the physical one.
* **Thermal drift and holding**: `thermal.axis_drift` (6 components per axis), screw strain
  `axis_drift_strain` with its reference `axis_drift_ref_m`, `spindle_growth_M_m` (at the tool
  spindle's nose link or the work spindle's rotor link), `structure_drift_M_m` (first tool link);
  `holding.parts[w].pose_delta` moves W in the fixture frame. Tool pull-out and tool/holder growth
  are not in the chain (tooling composes them into the edge); they lengthen the tool the setter
  measures.
* **FK outputs** (`FK_*`): T0 in M, W, W0 (the non-rotating workpiece frame of lathes), T0 in S and
  W0, the tool-chain end N and the workpiece-chain end F. The T frame is T0 spun by the tool spindle
  angle.
* **IK**: orientation by the product of exponentials `z_W(theta) = exp(w1^ phi1) exp(w2^ phi2)
  z_W(0)` (Murray, Li & Sastry 1994 ch. 3): any two orientation joints are Paden-Kahan subproblem 2
  (two, one or no solutions), one is subproblem 1; position by Newton steps on the 3x3 linear axes
  (compiled `ik_position`); other machines, or positions the closed form cannot place, use damped
  Gauss-Newton on the full residual from the reference configuration. Covered: table A/C, table
  B/C, head B/C, head A/B, head-table B/C, lathes, Swiss (two paths), mill-turn (B head with C in
  C-axis mode).
* **Singularities**: a joint whose axis is parallel to the vector it must turn is undetermined
  (numerically below 1e-10); it keeps its reference value and the solve reports `singular` when the
  tool axis is within `singularity_cone_rad` (0.01 rad) of a rotary axis. A TCP block through a
  singular region raises a `singularity` finding with the machine's declared behaviour
  (`kinematics.singularities[*].behaviour`: a warning for slow_down, an error and a controller
  alarm for alarm).
* **Tilted planes** (`PathKinematics.tilted_plane`): the rotary solutions of a plane orientation,
  chosen by rule: `nearest`, `positive`/`negative` (sign of the master rotary axis: Heidenhain PLANE
  SEQ+/-, CYCLE800 direction, FANUC G53.1), `travel_shortest` (Heidenhain without SEQ). Defaults:
  Heidenhain travel_shortest, Siemens positive, others nearest.

## 6. Auxiliary functions and the operator (`timing.py`)

* **Tool change (ATC)**: spindle stop (predicted deceleration) + orientation while the tool-axis
  slide rapids to its tool-change position, then `tool_to_tool_s`; chip-to-chip adds the approach
  and the run-up of the next tool (compared with the published `chip_to_chip_s`).
* **Turret**: `adjacent + (max - adjacent)(steps - 1)/(half - 1)` for `steps` stations away (the
  farthest a bidirectional turret indexes is half the stations).
* **M functions** `m_function_s`; through-spindle coolant adds `tsc_pressure_delay_s`; chucks,
  collets, tailstock, steady `chuck_s`; doors `doors.open_time_s`; pallets
  `pallet_changer.change_time_s`.
* **Dwell**: seconds, or revolutions at the actual speed of the path's spindle.
* **Operator stops** (`operator_stops` setting): M00, M01 with the optional-stop switch, wait for
  `stop_s + 2 door_s` of the production operator model (project `production.operator` over
  `db/models/production-operator.yaml`); single-block `single_block_s`; message stops
  `message_stop_s`; then the cycle restarts.
* **Probing** (G31, TCH PROBE, measuring cycles): the loop flags `PROBE_TRIGGER` when removal's
  sweep reports the stylus touching (`out.probe_hit`); the planner latches the actual axis values
  (plus the stylus pre-travel along the move: the machine's `probes.spindle_probe.pretravel_m`, else
  `probe_pretravel_m`), cancels the rest of the move and brakes along it within `brake_distance`.
  **Tool setter**: the loop triggers when the physical tool tip (with pull-out and thermal growth)
  crosses the setter plane, or enters the cylinder over a setter disc of the given radius.

## 7. Multi-path waits

A `SYNC` command with reason `path_wait` (FANUC M100-M999 waits, Siemens WAITM) makes the path stop
in position (`SP_HOLD_WAIT` with `WAIT_PATH`); the loop reports `PATH_WAIT`; the controller gets
feedback and releases the paths once all have arrived, and the wait is booked `path_wait` in the
next block of each path. Other sync reasons (live reads of machine variables) return feedback with
the actual position at the tick.

## 8. Timeline and ledger (`ledger.py`)

One record per executed NC block (the commands sharing `(path, block_seq)`): `{index, op, program,
line, block, path, kind, tick0, tick1, part, tool, ticks, stop, pumps_on, motion_ticks,
ideal_ticks}` in base ticks over `[tick0, tick1)`. Records of a path are contiguous: waits before a
block (data starvation, spindle arrival, path waits, operator) belong to the block that follows, so
records sum exactly to the span and `ticks` sums exactly to `tick1 - tick0` (every L1 tick booked
once, integer base ticks). Categories: `cutting`/`air_cutting` (feed in/out of contact: the hot
loop's per-tick contact flag `out.in_cut`, else removal's ToolContact of the step), `rapid`,
`acceleration` (motion time beyond the ideal time: acceleration, corner slow-down, centripetal
limits), `data_starvation` (from the generator's starvation flag, before acceleration), `feed_hold`,
`in_position`, `dwell`, `tool_change`, `spindle_ramp`, `coolant_delay`, `probing`, `program_stop`,
`auxiliary`, `path_wait`. The timeline (`ctx.timeline` = `MotionDomain.timeline`, JSON by
`timeline_json`) is the format `cnctwin.production` reads.

## 9. Validity and error bounds

| quantity | bound | how established (test) |
|---|---|---|
| generator vs analytic double-S, rest to rest | position <= 2 J dt^3, duration within 1 tick, v <= V, a <= A, jerk steps <= J dt | `test_generator_reproduces_the_analytic_double_s` |
| braking distance | relative 1e-12 vs closed form, 2e-4 vs brute-force integration | `test_brake_distance_matches_brute_force` |
| arc feed clamp | v <= min(sqrt(A R), (J R^2)^(1/3)) (1e-6), radius 1e-9 m | `test_arc_feed_is_clamped_by_centripetal_acceleration_and_jerk` |
| data starvation, 0.05 mm blocks | feed = 0.05 mm per block time within 0.2 % | `test_data_starvation_on_short_segments_limits_feed_to_block_rate` |
| corner deviation | commanded <= eps (1e-9 relative); blend apex at eps (1e-6) | `test_corner_blend_deviation_within_tolerance` |
| Haas corner radius | blend radius `R_needed(v)` (1e-9) inside Setting 85 with no slow-down; at the setting (1e-12) and the corner speed of that arc (0.5 %) when the feed needs more | `test_haas_corner_rounding_is_a_radius_rounded_as_far_as_the_feed_needs` |
| SINUMERIK ADIS | tangent length ADIS (1e-9), deviation ADIS (sqrt 2 - 1) at a square corner (1e-6) | `test_siemens_adis_is_the_rounding_distance_along_the_blocks` |
| feed velocity at a tool change | no tick above the axes' rapid vector (the tip's jump to the new tool is no motion), macro steps of 1, 2 and 10 ticks | `test_a_tool_change_is_no_feed_motion` |
| chord error at the IPO period | arc speed at 2 sqrt(2 R e - e^2) / T (generator ramp J dt^2/8, 0.5 %); every chord within e (1e-3) | `test_chord_error_at_the_ipo_period_limits_a_small_arc_with_a_tight_tolerance` |
| setpoint filter band | arc speed at sqrt(24 R e) / T_f (0.5 %) | `test_setpoint_filter_band_slows_arcs_to_keep_the_radius_error_in_the_tolerance` |
| curvature step | S-turn junction at sqrt(A R / 2) (+ J dt^2/8) | `test_curvature_reversal_between_tangent_arcs_bounds_the_junction_speed` |
| block targets | no commanded value passes its block's target or a travel end on it (0 m) | `test_commanded_profile_never_passes_the_block_target` |
| circle radius reduction | within 3 % of the closed form; 15 % of v^2/(2 R Kv^2) without feed-forward | `test_circle_radius_reduction_matches_closed_form` |
| steady following error | v (1 - k_ff)/Kv within 1e-6 | `test_steady_following_error_is_v_one_minus_ff_over_kv` |
| backlash at reversal | lost motion b r within 2-4 nm | `test_backlash_lost_motion_is_the_compensation_residual` |
| reversal spike | within 25 % of the published spike | `test_reversal_spike_from_the_calibrated_friction` |
| FK/IK round trip | tip and axis < 1e-10, five 5-axis types; lathe < 1e-9 | `test_fk_ik_round_trip_for_each_five_axis_type`, `test_lathe_fk_ik_and_rotating_workpiece_frames` |
| TCP | tip on the programmed line < 1e-9 m | `test_tcp_motion_keeps_the_tip_on_the_programmed_line` |
| tool change, turret, dwell, M functions | exact to the L1 tick | `test_timing.py` |
| rigid tapping synchronisation | spindle-Z error within 5 % of v[(1-k_ff,sp)/Kv_sp - (1-k_ff,z)/Kv_tap] | `test_rigid_tapping_spindle_z_synchronisation_error` |
| per-revolution feed / thread lead / CSS | 1e-6 / 1e-9 relative / 1 % | `test_fanuc_lathe_program_css_per_rev_and_threading`, `test_long_per_revolution_contour_runs_as_one_stretch` |
| orientation | angle held within 1e-4 rad after the orientation time | `test_orientation_turns_the_spindle_to_the_programmed_angle` |
| C axis | positions within the C-axis loop's settling (2e-3 rad at the next block's start, 1e-4 rad after it) | `test_c_axis_moves_stop_the_spindle_and_position_the_work` |
| ledger | records and categories sum exactly (integer base ticks) | `test_records_sum_exactly_to_the_cycle_time` |
| checkpoint | continuation bit-identical (axes, spindles, timeline) | `test_checkpoint_round_trip_*` |
| kernels vs Python references | 1e-15 .. 1e-9 (RK4 states) | `test_kernels.py` |

Validity: rigid-body axes with one dominant mode (the velocity loop bandwidth); structural modes,
motor current loops and encoder quantisation below the input increment are not modelled; friction
is Coulomb plus a tanh regularisation (no Stribeck dip, no pre-sliding hysteresis), so quadrant
glitches are reproduced in height (25 %) but not in shape; the look-ahead is a time-optimal
jerk-limited profile, which matches controls with jerk limitation (FANUC AI contour II, Haas
G187, SINUMERIK BRISK/SOFT + G64, Heidenhain cycle 32) in bounds but not in their proprietary
smoothing filters, so cycle times on short-segment programs are within the block-rate and limit
bounds rather than to the millisecond; spindle drives are a speed loop on a torque-limited rotor.

## 10. Bus

Read (`bus_prev`):
* `thermal.axis_drift` (6 components per axis), `axis_drift_strain`, `axis_drift_ref_m`,
  `spindle_growth_M_m`, `structure_drift_M_m`, `thermal.tools[slot].length_growth_m` and
  `holder_growth_m` (tool setter only);
* `holding.parts[w].pose_delta`, `holding.tools[slot].pullout_m` (tool setter only);
* `cutting.contacts[*].axis_force_N` (last row, columns in the order of `PathFrame.axis_names` of the
  contact's path) and `spindle_torque_Nm` (last value): the first tick's disturbance of the step
  (one-tick lag; the hot loop passes the per-tick loads afterwards);
* `removal.contacts[*]` (`contact` -> `path`; `geometry.in_cut`, `geometry.air_cut_time_s` for the
  cutting/air split when the hot loop gives no per-tick contact flags);
* `motion.paths[*].axis_names` (its own previous record, for the force columns).

Written (`bus_next.motion`, a `MotionFrame` per macro step, one `PathFrame` per path): tick,
block_index (into the timeline), tip_S_m, tip_M_m, axis_T_M, axis_names, axis_actual, axis_cmd,
following_error, spindle_angle_rad (wrapped), spindle_rev, spindle_omega_rad_s, feed_vel_M_m_s,
rapid, tool_slot, spindle_id, workpiece, pose_M_T, pose_S_T, pose_W0_T, coolant_pumps,
nozzle_settings, events, axis_inertial_force_N. `feed_vel_M_m_s` is the tip's velocity relative to the work
(differenced tip positions in W0); at the tick where a tool change moves the tip to the new tool's point (motion
swaps the tool offset in its chain at the first tick of the next plan call: `step.tip_swap`) that tick takes the
velocity of the tick before (2026-11-09: the jump, the tool length difference in one tick, went to the bus as up to
102 m/s, and chips sized its occupancy grid by it: docs/models/chips.md). **Added** fields (instance attributes, additive):
`axis_force_N` (drive force per axis per tick), `spindle_torque_Nm`, `work_angle_rad` (work spindle
angle per tick), `tip_W0_m` (tip in the non-rotating workpiece frame), `phase` (planner phase at the
end of the step).

Channels: per tick `motion.axis_actual`, `motion.axis_cmd`, `motion.following_error`,
`motion.axis_force_N`, `motion.path_speed_m_s`, `motion.tip_M_m`, `motion.spindle_omega_rad_s`,
`motion.spindle_torque_Nm`; per macro step `motion.queued_commands`, `motion.phase`,
`motion.cycle_time_s`.

Findings: `overtravel`, `alarm` (excess following error), `axis_overload`, `singularity` (TCP). Controller feedback: retire, feedback (positions, probe/setter results, alarms),
cycle_start (`MotionDomain.controller_actions`).

## 11. Kernel ABI and loop protocol

Layouts (all arrays; P paths, NA axes, S spindles, L links, T ticks per step, G geometry rows):

* `Chain` [P, 2 chains, L]: kind, axis, direction, point, fixed 4x4, err_poly [6, 3], err_lo,
  err_span, err_const [6], strain, strain_ref; per path work_spindle_link, pose_N_T0, pose_F_W,
  pose_W_S, tool_spindle, tool_phase.
* `Servo` [NA]: path, rotary, kv, kff, kp_v, ki_v, mass, force_max, friction, v_stribeck, gravity,
  backlash, comp, scale_fb, in_pos, lo, hi, ferr_max, spindle, kappa, tap_kv; n_sub, dt.
  `ServoState` [NA]: x, v, integ, load, comp, dirn, cmd, force, acc, vel_prev, **c_off** (added: the
  whole-turn offset of spindle C axes).
* `Spindle` [S]: inertia, kp, ki, drag_max, omega_max, accel/decel rate, pos_kv, pos_kff, c_kv,
  c_kff, c_axis, torque curves [S, gears, points]; `SpindleState`: angle, omega, integ, torque,
  cmd_omega, gear, cmd_angle.
* `Setpoint` [T, NA] cmd; [T, P] mode (`SP_TIME`, `SP_PER_REV`, `SP_THREAD`, `SP_HOLD_WAIT`,
  `SP_END`, `SP_IDLE`), block, rapid, tap, probe, setter, sync_start; [T, S] sp_mode, sp_omega,
  sp_angle, sp_omega_ff, css_vc, css_clamp; CSS axis and centre; the synchronous table
  (sync_theta, sync_s [P, T+1]) and geometry (geo_* [P, G]); threading lead, trigger, filter; tool
  setter point, normal, radius, extra length.
* `LoopState` [P]: stage, theta_trig, s_filt, s_act, theta_prev, theta0. `MotionOut`: per tick
  actual, cmd, ferr, force, inertial, overload [T, NA]; tip_M, tip_S, tip_W0, axis_M [T, P, 3];
  sp_angle, sp_omega, sp_torque [T, S]; sync, alarm, alarm_axis, in_cut, probe_hit, sync_s [T, P];
  poses [P, 7, 4, 4]; k_done.

`motion_tick(chain, servo, servo_state, spindle, spindle_state, setpoint, loop, out, k,
dist_force[NA], dist_torque[S]) -> sync code` advances one L1 tick: stretch start latch, spindles
(CSS, ramp, step), synchronous setpoints, axis drives (overload flag, stroke check on the command,
excess error), FK per path, probe/setter/in-position/end checks. `dist_force[a]` is the cutting
force on axis a along +axis (N); `dist_torque[s]` the cutting torque opposing positive omega (N m).
The hot loop sets `out.in_cut[k, p]` when the path's tool is in contact during tick k and
`out.probe_hit[k, p]` when removal's sweep finds the stylus touching, calls `replan_before(setp, k)`
before every tick after the first of a call (and returns without executing tick k when it gives a
path), and returns at the first non-zero sync code with `inputs.result = (code, k, path)`.
`run_motion` is the same loop without cutting (constant disturbances).

## 12. Settings (`physics.motion`, `schema.py`)

`every_n_macro` 1 (fixed), `servo_omega_h` 0.25, `friction_h` 1.0, `operator_stops` 1,
`contact_from_loop` 1.

## 13. Cost (measured, one core, bin/tsafe, motion without cutting)

| program | cost |
|---|---|
| 3-axis zigzag, 400 blended 40 mm blocks at 3 m/min | 0.33 s per simulated second |
| lathe, 300 per-revolution blocks (one synchronous stretch) | 0.21 s/s |
| 5-axis TCP (table B/C), 60 blocks | 0.19 s/s |
| kernel compilation (cold cache) | 20-30 s once per cache key |

## 14. Limitations

* A spindle's C axis engages only when a block changes the programmed C value (the IR has no
  C-axis on/off command): the first target of C in a program is taken as its modal value, and a
  speed command or orientation always leaves C-axis mode. Polar and cylindrical interpolation are
  the controller's transforms; motion executes their joint moves.
* Corner blends are geometric between joint lines only; arc and TCP corners keep their corner at the
  bounded node speed (no blend).
* One rigid-body mode per axis; no structural dynamics, cross-coupling between axes, motor current
  loop, encoder quantisation or thermal effects on the drive parameters.
* Proprietary smoothing of controls (FANUC nano smoothing, SINUMERIK COMPCAD, Heidenhain ADP) is
  represented by the tolerance-band blend, the tolerance-driven limits and the jerk-limited profile only.
* The setpoint-filter band needs the filter time from the machine file (`controller.setpoint_filter_ms`);
  no machine file in db/ gives one, because no control publishes it. The TNC 640 documents a feed
  reduction on arcs that keeps its HSC filter within cycle 32 T (heidenhain-tnc640-cycles-2023, cycle 32),
  which this band reproduces once the filter time is known. Taking it as the jerk-equivalent filter
  A/J (Biagiotti & Melchiorri 2012) fits one published TNC 640 parameter set and contradicts the other
  (validation/cases/cycle_time: the A 3.1 / J 157 set's 10 um tests go from -46 % and -27 % to -6 % and
  -8 %, the A 3 / J 50 set's 10 and 50 um tests from -7 % and -17 % to +63 % and +33 %), so it is not a
  default.
* Curvature steps are bounded by the axis acceleration only: the controls' jerk control at such
  junctions (FANUC AI contour control II jerk control, SINUMERIK MD32432 PATH_TRANS_JERK_LIM) has no
  published default and is not applied.
* The operator model times M00/M01; if production also books operator time for the same stops, the
  integration must count it once.

## 15. References

Biagiotti & Melchiorri 2008 (double-S, braking distance); Biagiotti & Melchiorri 2012 (FIR filters,
jerk limitation as moving averages); Suh, Kang, Chung & Stroud 2008 and Yeh & Hsu 2002 (chord error of
sampled-data interpolation); Altintas 2012 (feed drives, contouring);
Erkorkmaz & Altintas 2001 (feed drive identification); Ellis 2012 (loop tuning); Nordin & Gutman
2002 (backlash); Armstrong-Helouvry, Dupont & Canudas de Wit 1994 (friction); Palmgren 1959 (bearing
drag); Murray, Li & Sastry 1994 (product of exponentials, Paden-Kahan subproblems); Shoemake 1985
(slerp); ISO 230-1:2012, ISO 230-2:2014, ISO 230-4:2005 (geometric errors, positioning, circular
tests); ISO 841:2001 (axis nomenclature); FANUC 30i-B operator's manual; Haas NGC mill operator's
manual 2020; Heidenhain TNC 640 Klartext 2023 and PLANE; Siemens SINUMERIK 840D cycles. Full
entries: `db/bibliography/models-motion.yaml` and the shared bibliography files.
