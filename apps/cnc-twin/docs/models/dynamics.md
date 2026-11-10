# Dynamics: machine dynamics and vibration

Package `cnctwin/dynamics/` (domain `dynamics`). Integrates the relative tool-workpiece vibration of every
active tool, in contact (L0, coupled to the cutting forces) and out of contact (L1, unbalance and axis
accelerations), books its energy exactly, detects chatter, and computes stability lobe diagrams per tool and
engagement with a semi-discretisation check.

Files: `abi.py` (kernel layouts), `kernels.py` (compiled kernels), `reference.py` (plain-Python references),
`discretize.py` (exact discretisation, energy integrals, the free interval), `modal.py` (mode assembly, basis
projection), `operating.py` (operating point and its linearised cut), `sld.py` (lobes), `semidisc.py`
(semi-discretisation), `chatter.py` (detection), `drive.py` (verification harness, cost loops), `state.py`,
`channels.py`, `schema.py`, `params.py`, `domain.py`.

## 1. Systems and modes

One **system** per controller path with an active tool (`PathFrame.tool_slot >= 0`): that tool against the
path's workpiece. Modes are mass-normalised (`u = phi q`, energy `E = sum (qdot^2 + omega^2 q^2) / 2`), ordered
`[tool] + [machine] + [workpiece]`; row r of `ModalModel.tool_shape` belongs to the r-th non-workpiece mode
(cutting's `deflection_from_modal` uses the same convention).

| source | modes | shape |
|---|---|---|
| structure `ToolModes` of the slot | tool + holder + spindle (structure's RCSA / FE) | 6 components per station (T0), mapped from structure's stations to the EdgeModel stations by linear interpolation in z; outside structure's range moved rigidly with the end station's rotation (`dx += dz r_y`, `dy -= dz r_x`); residual flexibility mapped with the same weights (`P R P^T`) |
| fallback until structure publishes the slot | machine `spindles[].frf_modes` (rotating tools, direction in the spindle-nose frame) or `structure.turret_modes` (static tools, direction in M) | rigid tool on the flange: `d / sqrt(m)` at every station, `m = k / omega^2` |
| machine `structure.modes`, `table_modes` | relative tool-workpiece translation along `direction` (M) | `R_M_T0^T d / sqrt(m)` at every station |
| machine `structure.rotary_modes` | torsion of a rotary axis (`k_Nm_rad`) | relative translation `s (e x (p_st - p_axis)) / sqrt(I)` and rotation `s e / sqrt(I)` (T0), `I = k / omega^2`, s = -1 workpiece chain, +1 tool chain; the axis line is placed by the product of exponentials of the upstream axes at the step's axis values (Murray, Li & Sastry 1994; joints are screws in the base frame at home) |
| structure `WorkModes` of the workpiece | workpiece modes on structure's grid (S) | trilinear interpolation between cell centres (`origin + (i + 1/2) cell`, clamped) at the contact point (`refresh_work_shapes`, per L1 tick for mills, per L0 step for rotating workpieces) |

Residual (static) flexibility: the tool's from structure (`static_residual_m_per_N`, translations of the
stations), the workpiece's per cell (`static_compliance`) at the contact; when the workpiece has no modes the
fixture compliance (`StructureFeedback.fixture_compliance_m_per_N`) is added at every point instead (with modes
it is assumed to be inside them - structure's FE includes the supports). Both enter quasi-statically: the station
deflection adds `R F_st` and the workpiece `-C F_S` with the wrenches of the last evaluation (one L0 step lag;
the lagged static term is stable while `c_res K_c < 1`, which a statically stable cut satisfies).

Reference station (vibration output, chatter samples, the lobe FRF): the station nearest the height of the
engaged region's centroid (removal's contact geometry), else the lowest station.

## 2. Time integration

### 2.1 Exact first-order hold

Per mode, `q'' + 2 zeta omega q' + omega^2 q = f(t)`. Over a step h with f linear between f0 and f1:
`x1 = Phi x0 + G0 f0 + G1 f1` from the augmented exponential of `[[A h, B h, 0], [0, 0, 1], [0, 0, 0]]`
(Franklin, Powell & Workman 1998, 6.3.2): exact for a force linear in time (tested to 1e-10 against the
analytic ramp response for `omega h` from 1e-4 to 30); a sampled sinusoid converges as `(W h)^2 / 12`.

### 2.2 Exact energy integrals of a step

With `z = [q0, v0, f0, s]`, `s = (f1 - f0) / h`, the structural damping dissipation and the work of a force part
`f_p(t) = f0_p + s_p t` on the exact trajectory are quadratic / bilinear forms with Gram matrices from Van
Loan's block exponential (Van Loan 1978) for the exosystem-augmented state (`A_z`, `[q, v, f, s]`):

    D = int 2 zeta omega v^2 dt = z^T M_D z,       W_p = int f_p v dt = [f0_p, s_p] N z,
    M_D = int e^(A_z^T t) (2 zeta omega e2 e2^T) e^(A_z t) dt,   N = int e^(A_z^T t) e3 e2^T e^(A_z t) dt.

`E1 - E0 = W - D` holds to round-off (tested to 1e-11 of the flows over a long mixed run). Cached per
`(omega, zeta, dt)` bits.

### 2.3 Predictor-corrector in contact

Nodes are the ends of L0 steps. The cutting kernel's force of a step (mean over the step, evaluated at the
step's mid angle with the deflection predicted for the step end) is the first-order-hold sample at the node.
Per step:

1. `x* = Phi x_n + G0 f_n + G1 (f*_cut + f_unb + f_base)_(n+1)` with the cutting force extrapolated
   `f*_cut = 2 f_cut,n - f_cut,n-1` (`modal_predict`);
2. the cutting kernel evaluates the wrenches at the predicted deflection (`modal_deflection`);
3. `f_x` = their modal projection; relaxed correction `df = (I + K_m diag(G1_q))^-1 (f_x - f*_cut)` with the
   modal cutting stiffness `K_m = -Psi^T K_c Psi` (`K_c = a_p K0`, the mean force sensitivity of the operating
   point, section 5.1; identity without one); `x_(n+1) = x* + G1 df`, `f_cut,(n+1) = f*_cut + df`
   (`modal_correct`).

With an exact `K_c` the scheme reproduces the implicit first-order hold; the stiffness is not delayed. The
extrapolation assumes a force that is smooth from node to node: a force component that switches on and off with
the predicted state (cutting's process damping as it is evaluated today, a one-sided function of the predicted
normal velocity; docs/models/loop.md 15) is overshot by `2 f_n - f_n-1` and the pair can lock into a
step-to-step relay; the relaxation, a linearisation, does not cover it.

**Process damping's flank contact in the estimate** (2026-11-10). Cutting's process damping is now the wave-memory
indentation K_sp dS A evaluated at the predicted state (docs/models/cutting.md "Process damping"): with the
remembered surface held, A grows by the zero-clearance land L per unit of the current displacement along the
element's normal, so each engaged element adds a contact stiffness K_sp dS L n n^T to the force sensitivity. The
relaxation's estimate is `K_c = a_p (mean of the cut's sensitivity + mean of the flank contact)` (`CutModel.
contact_M`, `contact_mean`; the lobes' delay model does not use it). On examples/drill-tap-plate-1045's D10 spot
drill in 1045 (K_sp 3.4e14 N/m3, L 26-35 um) at 6.79 s: |cut| 3.1e8 N/m, |flank contact| 5.4e7 N/m. The modal
scheme's limit (tests/unit/dynamics/test_contact_relaxation.py, one mode, zeta 0.01): the plain predictor-corrector
holds to a contact-stiffened 2 pi f_c dt of about 1.2, the relaxed one to about 2; the L0 stride rule keeps
2 pi f_c dt <= `stride_contact_omega_dt` (1.0), f_c the largest eigenfrequency of diag(omega^2) + Psi^T sym(-K_c)
Psi over the tool and machine modes at the reference station (`modal.f_contact_Hz`, `SystemRecord.f_contact_Hz`;
19.3 kHz on the spot drill, 2 pi f_c dt = 0.24 at its 2 us step: not binding there).

**The residual flexibilities against the contact** (2026-11-10). The quasi-static residual flexibilities (the tool's
static residual over its stations, the workpiece's at the contact) take the last evaluated forces, one L0 step
behind. Against a contact that is the explicit iteration F_n = F(u_modal) + K_c C F_(n-1), C the residual
compliance in series with the contact at the reference station (tool block plus R^T C_w R): it alternates in sign
from step to step and grows once |K_c C| > 1, whatever the step. On the spot drill C = 1.1-1.5e-8 m/N laterally (the
tool's residual; the workpiece's 3e-10). That was the drill's divergence: the lateral force alternated from one 2 us
step to the next and grew by 8 % per step from 6.798 s (+-270 N -> +-1.4 kN in 12 steps; a real loop gain of
-1.08, a restoring sensitivity of about -8e7 N/m laterally), then 15-27 kN on the Z axis and the chip-thickness
divergence finding at 6.80-6.95 s depending on the variant; it went away with process damping off. The corrector now
stores, for the residuals, the fixed point of F = F(u_modal) + K C (F - F_(n-1)):
`F_(n-1) + (I - K_c C)^-1 (F_n - F_(n-1))` at the load stations and for the workpiece (`modal.residual_relaxation`,
DynModel.scal[MS_RES_RELAX], kernels.modal_correct and reference.modal_correct_ref; the identity without an
estimate). For a linear contact it is exact in one step; with an estimate off by a factor it converges for any true
gain below 1 + 2 x the estimated one (tests/unit/dynamics/test_contact_relaxation.py: gains 0.5, 2.4 and 5, and
estimates of 0.5 and 3 x the true stiffness). The vibration sample reads the residuals' forces as they hold them.
K here is the flank contact's stiffness alone (`K_res_T0`; -2.0e7 N/m per lateral axis on the drill, gain 0.26-0.3,
so the measured -1.08 is within 1 + 2 x 0.3). The cut's linearised sensitivity (the delay model's `CutModel.M`) does
not enter: on the drill its symmetric part is not restoring (+1.3e8 N/m laterally, +1.5e8 axially, trace +4e8, where
each element's trace is -db (K_r - K_a a.n) / a_p; and the run's own alternation shows a restoring loop), with it I - K C had
eigenvalues 0.48 and -0.52 +- 1.92i and the first relaxed step blew up (1e50 mm chip thickness at 6.791 s). Why
the delay model's mean has that sign on a drill (the incremental K_r at the mean chip thickness, the axial-force
direction of the lip elements, the engaged arc) is open; the modal relaxation still takes it.
The estimate exists only while the step has an operating point: removal reports a contact in the steps that
subtracted, and a slow tool's tooth subtracts every few steps (the spot drill's every 36 ms against 10 ms steps), so
dynamics keeps the last subtracting step's geometry while cutting reports the tool in cut
(`DynamicsDomain._engaged_contact`, persisted in `state.dynamics.meta["held_contact"]`).
**Numerical damping** of the closed loop `F = -K_c u` against the exact damping ratio
`zeta / sqrt(1 + K_c / k)` (tests/unit/dynamics/test_predictor_corrector.py, the real kernels):

| 2 pi f dt | K_c / k | with the stiffness estimate (zeta 0.002-0.05) | without (zeta 0.01 / 0.05) |
|---|---|---|---|
| 0.05 | 0.1-1 | < 0.05 % | < 0.05 % |
| 0.15 | 0.1-1 | < 0.2 % | - |
| 0.2 | 1.0 | -0.16 % | +0.34 % / -0.07 % |
| 0.3 | 0.5 | -0.18 % | +0.52 % / -0.05 % |
| 0.3 | 1.0 | **-0.37 %** (frequency -0.37 %) | +3.3 % / +0.33 % |

A delayed stiffness (the previous node's force held) gives -730 % at the corner (unstable where the exact loop
is damped). Consequence of the node convention: the cutting force's angle dependence is attributed half a step
late; the periodic system is only shifted in time by dt/2, which leaves its stability unchanged and shifts the
forced response by dt/2 (4 us at 10 000 rpm with 0.5 deg steps).

### 2.4 Out of contact: exact free interval

Per L1 tick (or any interval h from the current node): `f(t) = Re(P e^(i W t)) + g0 + g1 t` - harmonic
unbalance at the tick's mean speed `W = (theta1 - theta0) / h` (unwrapped) with `P = P_k (W^2 - i alpha)
e^(i theta0)`, `alpha = (omega1 - omega0) / h`, and the base excitation by first-order hold. Exact solution:
harmonic particular solution `Q = P / (omega^2 - W^2 + 2 i zeta omega W)`, polynomial particular solution,
homogeneous part `Re(C_q e^(lambda t))`; work and dissipation in closed form from `J0(k) = int e^(k t)` and
`J1(k) = int t e^(k t)` (`phi1`, `psi1` with series below |z| = 0.2). Below `omega h = 0.1` the particular
solutions cancel and a Taylor series of the solution is used instead (sub-intervals of `|W| h <= 4`, 44 terms).
Tested against an independent exosystem-exponential + Gauss-Legendre reference (40 random cases, both branches)
and the identity `dE = W - D` (<= 1e-9 of the flows). The unbalance amplitude is held per tick at the mean
speed: during a ramp the phase is exact at the tick ends and the amplitude errs by about `alpha h / W`
(< 0.5 % of the response in the 9000 rad/s^2 ramp test).

Transitions: the first L0 step after a free tick starts from the tick-boundary node with `f_cut = 0`; at contact
end the free tick drops the cutting force to zero at the node (the last step's half impulse
`F_last dt / 2` is not applied; small because the exit step of a tooth carries little force).

## 3. Forces on the modes

* **Cutting**: `f_k = sum_s phi_k(s) . W_s` over the station wrenches (force and moment about the station's axis
  point, T0, cutting's `StepOut.station_T0`; the rotational components carry the moments) for tool and machine
  modes; `f_k = -phi_w . F_S` for workpiece modes (cutting's `StepOut.force_S_N`, the force on the tool in S).
* **Unbalance** (holding's `ToolHolding`: static vector U at height z_U and couple C, both in T): the
  d'Alembert load on the rotor `F = (W^2 - i alpha) U_c e^(i theta)`, moment `i (W^2 - i alpha) C_c e^(i theta)`
  (complex x-y components; `e_z x` is multiplication by i), so `f_k = Re[P_k (W^2 - i alpha) e^(i theta)]`,
  `P_k = conj(phi_c) U_c + i conj(psi_c) C_c` with the shape's translation `phi_c = phi_x + i phi_y` and rotation
  `psi_c` at z_U (tested against the direct projection of the rotating loads to 1e-12; steady-state amplitude and
  phase against the receptance to 1e-6, the couple through the rotation shape, the ramp against an ODE reference).
  Workpiece (lathe chuck) unbalance is not on the bus and not modelled.
* **Base excitation** (machine and fallback modes, which have a modal mass): `f_k = -sqrt(m_k) d_k . a_rel`,
  `a_rel = sum_axes side_a acc_a dir_a` the rigid-body acceleration of the tool side relative to the workpiece side
  (M; side from the kinematic chain), `-sqrt(I_k) alpha_axis` for rotary modes (Craig & Kurdila 2006: a mass on a
  spring carried by an accelerating base). Quasi-static check: `u = -m a / k` to 1e-6. The axis accelerations
  come from the hot loop (motion's servo state, `relative_axis_acceleration`) or, in the L2 catch-up, from second
  differences of `PathFrame.axis_actual`.
* **Process damping**: the flank indentation force `F = K_sp db A(v_n / V)` along -n (Wu 1989; Altintas, Eynian &
  Onozuka 2008), `A = int_0^x* (s x - max(0, x - L) tan gamma) dx` with the land `L = VB + 1.0 r_e` and the
  clearance face (Ahmadi & Ismail 2011), one-sided (only motion into the material), its dissipation `F v_n dt`
  into `Accum[ACC_PROCESS_DAMPING_J]` (`process_damping_element`). For small slopes `A -> s L^2 / 2`, whose
  energy-equivalent viscous coefficient over a vibration cycle is `K_sp db L^2 / (4 V)` (the lobes' model).
  **Booking**: cutting's force kernel applies the same model per element (`CutTool.pd`) and books the same
  accumulator, and cutting's record includes it in `ploughing_J`; `DynamicsFrame.process_damping_J` therefore
  stays 0 (the ledger would count it twice) and `process_damping_total_J` reports the accumulator.

## 4. Chatter detection (L2)

Samples: the relative deflection at the reference station at every L0 node in contact (`DynState.samp_*`, with
the spindle speed), kept per path over the last `chatter_window_max_s`, from the steps in which the path's tool
cut (cutting's record of the step: an element in cut); a step without cutting clears the window (2026-11-09:
the broad phase runs L0 steps near the stock before the first chip, and the machine's frame and table modes
(55-130 Hz on the VF-2SS), excited by the base excitation of the rapids and the plunge, rang there with a few
um: the tiny project, examples/thin-wall-al7075 and two-setup-flip reported that free vibration as
"regenerative chatter" at 65-91 Hz at their entry). Each macro step the last contiguous run covering
`chatter_window_revs` revolutions (at least `chatter_min_revs`) is analysed (`chatter.analyse`):

* detrend (mean and slope per component), Hann window, periodogram summed over components;
* **forced lines**: every spindle order k f_s (tooth-passing harmonics included: a stable cut with runout, pitch
  or helix is periodic with the revolution; Delio, Tlusty & Smith 1992; Schmitz, Medicus & Dutterer 2002), each
  masked +-(2 / T + k (f_max - f_min) / 2);
* **chatter index** `CI = E_non / (E_non + E_forced)` above f_s / 2; chatter frequency the largest non-forced
  peak (parabolic interpolation on log P), amplitude `A = 2 sqrt(sum_band |X|^2 / (n_fft sum w^2))`;
* **period doubling** of an even-tooth cutter lies on spindle orders ((2k + 1) N/2): those lines move to the
  non-forced side when their energy grew by `flip_growth_ratio` (1.5) between the window halves and carries at
  least `flip_min_share` (5 %) of the second half's energy (odd N: the half lines are not spindle orders);
* chatter when `CI >= 0.5` and `A >= max(0.5 um, 0.05 f_z)`, unless the band's envelope decays at least at half
  the structural rate `zeta omega` of the mode nearest the chatter frequency (its fitted growth rate): a free
  vibration dying out (an entry shock, a mode excited before the cut) is a transient, chatter grows or holds a
  limit cycle because the cut feeds it at least that much energy; **onset** = first sample where the Hilbert
  envelope of the band around the chatter frequency reaches that floor; the episode ends after
  `chatter_clear_steps` steps below `CI = 0.35`.

Finding (category `chatter`, at the onset tick, once per episode, at most `max_findings` per tool and operation):
severity **error** when `A >= 1.0 f_z` (the edge leaves the cut, Tlusty & Ismail 1981), else warning; quantities
`chatter_freq_Hz`, `chatter_amp_m`, `chatter_index` (+ the lobe limit and depth when known); cause chain
`dynamics:chatter_onset` <- `dynamics:stability` (operating point) <- `dynamics:modes`. A second finding kind:
"lobe diagram predicts chatter" (warning) when an operating point's margin is below 1 (section 5.4).
The Hann weighting favours the window centre: a chatter growing within the window is flagged once it dominates
the window's energy (up to one window, ~20 revolutions, after its start); its onset is dated back by the envelope.

## 5. Stability lobes

### 5.1 Linearised cut

`F(t) = a sum_j K_j(t) (u(t) - u(t - tau_j))`, u relative (T0), a the depth; per element
`M_e = -(db_e / a_p)(K_t c + K_r n - K_a a) n^T` (incremental coefficients `d(K h)/dh` of the element's group at
the mean chip thickness of the arc, cutting's ProcessCoefficients; element frames from tooling's EdgeFrameT),
rotating as `R(theta) M_e R(theta)^T` while the element is in the engaged arc. Means and Fourier coefficients
over the rotation are analytic (`R M R^T = sum_m C_m e^(i m theta)`, m = -2..2). `classic_milling` builds the
straight-tooth benchmark model (Altintas' frame) in the same form. FRF: `G = sum psi psi^T / (w_k^2 - w^2 +
2 i z_k w_k w) + G_static` at the reference station (tool residual block + workpiece residual).

### 5.2 Methods

* **ZOA** (Altintas & Budak 1995): eigenvalues lambda of `K0 G(i w)`; for `Re lambda > 0`:
  `a_lim = 1 / (2 Re lambda)`, `w tau = pi + 2 arg(lambda) + 2 k pi`, speed `2 pi / (N tau)`; lobes from a
  chatter-frequency scan (branches matched by nearest neighbour), lower envelope on a speed grid.
  Turning / boring: N = 1, constant K, tau one work revolution - Tlusty's `a_lim = -1 / (2 K_f Re G_or)`
  (minimum reproduced to 1e-7 of `2 zeta (1 + zeta) k / (K_f mu)`).
* **General solver at a speed**: real positive eigenvalue crossings of `B(w)` along the frequency scan
  (bisection on the branch): variable pitch `B = sum_j (1 - e^(-i w tau_j)) Kbar_j G` (Altintas, Engin & Budak
  1999); multi-frequency `B = (1 - e^(-i w tau)) [G(w + p w_T) K_(p-q)]` (Budak & Altintas 1998, h harmonics).
* **Process damping** (Altintas, Eynian & Onozuka 2008): `G -> (I + i w a C_pd G)^-1 G`,
  `C_pd = mean K_sp db L^2 / (4 V a_p) n n^T`; the depth is the fixed point of `a = limit(G_pd(a))` (secant
  iteration, frequency scan narrowed to +-15 % after the first step).
* **Mode coupling** (Tobias 1965; Tlusty & Polacek 1963): smallest a with an unstable eigenvalue of the modes with
  the current-displacement stiffness `a Psi^T K0 Psi` only (no regeneration); reproduces the closed-form flutter
  onset to 0.08 %.
* **Semi-discretisation** (Insperger & Stepan 2002, 2004, first order): interval means of K (and of the process
  damping), the delayed displacement linear between its samples, exact first-order hold per interval; monodromy
  over the tooth period (uniform pitch) or the revolution (variable pitch, delays rounded to intervals); at least
  `sd_m` intervals per period and `sd_points_per_period` per period of the highest mode over the delay. The limit
  search scans geometrically, then scans `[a_u / 2, a_u]` with ratio 1.02 for **period-doubling lenses** (two
  real multipliers collide and the cut is briefly stable again: at 10 000 rpm in the 1-DOF a/D = 0.5 benchmark
  the first loss of stability is a flip at 2.11 mm below a Hopf boundary at 2.35 mm, confirmed by the time domain).

### 5.3 Accuracy (how established)

| comparison | result |
|---|---|
| SD (m = 40) vs Insperger & Stepan 2004 charts (validation cases), lobe troughs, 1-DOF a/D = 0.5 and 2-DOF a/D = 1 | within the digitisation uncertainty + 3 % |
| time domain (96 steps per tooth, real kernels) vs converged SD, 1-DOF a/D = 0.5 (4 speeds incl. a flank and a flip point) | < 2 % |
| time domain vs SD, 2-DOF slotting | < 3 % |
| time domain (192 steps per tooth) vs SD, a/D = 0.05 | < 4 % |
| ZOA vs SD, slotting lobe minima | < 3 %; a/D = 0.5 within 15 % where the boundary is Hopf; ZOA has no flip boundary (43 % high at 10 000 rpm, 3.8 x at the 18 000 rpm flip lobe of a/D = 0.05) |
| multi-frequency (3 harmonics) vs SD at the a/D = 0.05 flip lobe | < 6 % |
| turning: ZOA minimum vs Tlusty closed form; SD vs ZOA | 1e-7; < 1 % |
| variable pitch (90 +- d deg, 4 flutes): ZOA vs SD | < 6 %; limit at the design speed > 1.3 x uniform |
| process damping, 1500-3000 rpm: time domain (one-sided nonlinear indentation, f_z = 0) vs SD with the linear equivalent | < 4 %; ZOA + process damping overestimates by 30 % at 1500 rpm (it averages the damping over the revolution) |

The digitised flanks of the Insperger charts are offset from the converged SD, the time domain and the ZOA (which
agree with each other) by 50-90 rpm, and the a/D = 0.05 chart reads 7-15 % above them in its troughs; the case
notes attribute such deviations to the source's 400 x 200 grid and rendering. The forced vibration changes the
process damping: with the feed (f_z = 0.1 mm) the flank is loaded during each pass and the time-domain limit at
1500-3000 rpm is about twice the linear one (an amplitude effect no linear method has).

### 5.4 Operating points and sld/<tool>.json

At every macro step with a contact and the slot's ProcessCoefficients (from `bus_prev`): engaged elements
(cutting's per-element in-cut time, else the stations within a_p around the centroid height; a turning contact,
whose removal record carries no depth and no centroid (docs/models/loop.md 15), takes cutting's engaged elements,
the sum of their chip widths as the depth, and the tool tip as its contact point and reference), the engaged arc
(removal's engaged angle around the centroid direction, T0), feed per tooth from motion's feed velocity, the mean
chip thickness over the arc; the limit at the operating speed (ZOA / variable pitch / multi-frequency, process
damping when K_sp > 0); refreshed when the engagement signature (arc, thickness, coefficient and structure
versions, engaged count) or speed / depth change by `engagement_tol`, at least every `sld_refresh_macro` steps.
The limit is cached by engagement class (the signature), FRF and spindle speed (relative step 1e-3) in the
checkpointed state (`meta.limits`), so a repeated engagement costs nothing; the chatter-frequency scan evaluates
the characteristic matrices of all frequencies at once (one stacked eigenvalue call), the sign changes of all
frequency intervals are bisected together (one stacked call per bisection level) and the cut model's per-tooth
means are cached. During the run that limit (the bus `depth_limit_m`, the "lobe diagram predicts chatter"
finding when the margin is below 1) is all the stability analysis does: lobe diagrams and the SD check are report
artefacts and an input to nothing in the loop (chatter detection is the time-domain index of every step). Each
engagement class (its linearised cut and FRF) is kept in the checkpointed `pending` state and, at the end of the
operation (`end_operation`, or `finalize` for what is left), gets its lobe diagram over `[sld_speed_lo_frac, 1] x`
the spindle's maximum speed and its mode-coupling limit (at most `sld_max_diagrams` per tool, the classes of the
operating points with the smallest margins first; a class keeps its diagram for later operations); then up to
`sd_max_points` operating points, smallest margins first, are checked by SD at their depth (with process damping):
`stable = margin > 1 and sd_stable`, and a point SD finds unstable that the zero-order limit passed raises the
finding then, at the point's tick. The check resolves the highest checked mode over the delay (`sd_points_per_period`
intervals per its period), so its state 2 n + r n grows with the delay: a point whose state would exceed
`sd_max_states` (2000) keeps the frequency-domain verdict and is recorded as not checked (`sd_stable` null,
`sd_states_needed`, the run's `sd_not_checked` count). Before 2026-11-09 the state was not bounded: a D6.8 drill at
830 rpm or a 4-flute cutter at 573 rpm (tooth periods of 36 and 26 ms against modes up to 10-20 kHz) asked for
r of tens of thousands, a monodromy matrix of 1e10 entries, and examples/ti-slot-throughtool was killed by the
1.8 GB cap at the end of its run (tests/unit/dynamics/test_domain.py, test_stability.py).
`sld/T<n>.json`: `format cnctwin-sld`, `tool`, `slot`, `number`,
`speed_rad_s`, `depth_limit_m`, `chatter_freq_Hz` (the lower envelope of the diagram of the worst operating
point), `radial_immersion_m`, `method`, `diagrams [{method, ae_m, arc, n_teeth, h_mean_m, speed_rad_s,
depth_limit_m, chatter_freq_Hz, lobes [{k, branch, speed_rad_s, depth_limit_m, chatter_freq_Hz}],
mode_coupling_limit_m, modes}]`, `operating_points [{op, program, line, time_s, tick, speed_rad_s, ap_m, ae_m,
stable, depth_limit_m, margin, chatter_freq_Hz, method, diagram, sd_stable, sd_states_needed (when not checked),
path}]`.

## 6. Basis change

When the system's mode set changes (structure version, fallback -> structure, tool change), the state is mapped
at the active contact points (every station's 6 components, rotations weighted by max(tool radius, 10 mm), and the
workpiece contact point): modes with the same identity whose frequency (1e-6) and shape (MAC >= 0.999,
Allemang & Brown 1982) match keep their coordinate scaled by the least-squares coefficient (exact for a
republished mode); the physical displacement and velocity of the others are projected onto the unmatched new
modes by least squares, minimum strain energy (displacement) / kinetic energy (velocity) where underdetermined.
`basis_change_J = E_new - E_old` (a dropped system books `-E`); `modal_energy_change_J` excludes it.

## 7. Domain, protocol, bus

`kernel_inputs(ctx, state, bus_prev)` -> `DynamicsInputs`: per path `systems[p]` (DynSystem), `grids[p]`
(WorkGrid), `axes[p]` (AxisBase), `contact_offset_S[p]` (contact point minus tip, S), `ring_columns[p]`.
**Hot-loop protocol** per path and L1 tick k:

* tick in contact: per L0 step n: `modal_begin(sys, theta_end, omega_end, alpha, base_in)` (= external_force +
  modal_predict + modal_deflection; theta, omega at the step end, base_in from `relative_axis_acceleration`),
  the cutting kernel with `Deflection(*sys.defl)` (+ structure's d(X) added to `u_work_S_m` if cutting expects
  it there), `modal_correct(sys, out.station_T0, out.force_S_N, tick_end, acc)`, optionally
  `ring_write(sys, ring, slot, ring_columns[p])`; per tick `refresh_work_shapes(sys, grid, tip_S + offset)` (per L0
  step for rotating workpieces);
* tick out of contact: `modal_free(sys, theta0, theta1, omega0, omega1, base0, base1, tick_end)`;
* every tick end: `record_tick(sys, k)`.

The L0 stride must match `ModalModel.dt_s` (`set_l0_ticks(path, l0)` before `kernel_inputs`, else
`physics.resolved` / the stride rule `l0_ticks_rule`); a mismatch is flagged by the kernels and raised at absorb.
Ticks the loop did not advance are caught up at the next `kernel_inputs` from the motion frame, out of contact
(counted in `meta.caught_up_ticks`, trace `dynamics:free_catch_up`).

`absorb`: energy identity per system (`dE = W_cut + W_unb + W_base - D`, tolerance 1e-9 of the terms; violation
raises), `bus_next.dynamics`, `SimState.ledger.unbalance_work_J -= (W_unb + W_base)` (the ledger lists that term
on its out side), samples into the chatter window. `macro_step`: stability, chatter. `end_operation`: windows and
episodes reset. `finalize`: sld files and a summary. State: `DynamicsState.to_arrays/from_arrays` (round trip
continues bit-identically, tested).

**Bus read** (`bus_prev`): `motion.paths[]` (tool_slot, workpiece, spindle_id, spindle_angle_rad, spindle_rev,
spindle_omega_rad_s, pose_M_T, pose_S_T, axis_names, axis_actual, feed_vel_M_m_s, tick, tip_W0_m, block_index),
`structure` (tools[slot], work_modes[workpiece], fixture_compliance_m_per_N, version), `holding.tools[slot]`
(unbalance_static_kg_m, unbalance_z_T_m, unbalance_couple_kg_m2), `removal.contacts[]` (tool_slot, path,
geometry: in_cut, point_S_m, ae_m, ap_m, engaged_angle_rad), `cutting.coefficients[slot]`, `cutting.contacts[]`
(in_cut_time_s), `wear.tools[slot].VB_m`; `state.tooling` (effective EdgeModel, EdgeFrameT); machine file.

**Bus written** `bus_next.dynamics` (DynamicsFrame): vibration_pp_m (max over paths), chatter_index,
chatter_freq_Hz (worst path), modal_energy_change_J, damping_dissipation_J, process_damping_J (0, section 3),
basis_change_J; **added**: `contacts` (DynamicsContact per path: n_modes, vibration_pp_m, chatter_index,
chatter_freq_Hz, chatter_amp_m, chatter, chatter_onset_tick, modal_energy_J, modal_energy_change_J,
damping_dissipation_J, cutting_work_J, cutting_work_quadrature_J, unbalance_work_J, base_work_J, basis_change_J,
deflection_peak_T0_m, depth_limit_m, depth_of_cut_m, structure_version), chatter_amp_m, chatter_onset_tick,
cutting_work_J, cutting_work_quadrature_J, unbalance_work_J, base_work_J, modal_energy_J,
process_damping_total_J. Core Accum written in contact: ACC_DAMPING_J, ACC_UNBALANCE_WORK_J (work done on the
modes). RNG streams: none (dynamics has no stochastic element).

**Ledger note**: cutting books the deflection work with its own quadrature (impulse x predicted rate,
`cutting_work_quadrature_J`); the exact modal work is `cutting_work_J`. They differ by 0.5-3 % of the vibration
work: relative to the shaft work about 4e-7 in a stable cut and 1.4e-5 in saturated chatter (benchmark, R 10 mm),
the latter above the ledger's 1e-6 tolerance.

Channels: fast `dynamics.vibration_rel_m` [4 paths x 3]; tick `dynamics.tool_deflection_{x,y,z}_m`,
`work_deflection_{x,y,z}_m`, `deflection_rel_{x,y,z}_m` [4]; macro scalars `dynamics.vibration_pp_m`,
`chatter_index`, `chatter_freq_Hz`, `chatter_amp_m` and per path `vibration_pp_path_m`, `chatter_index_path`,
`chatter_freq_path_Hz`, `modal_energy_J`, `damping_power_W`, `cutting_power_W`, `unbalance_power_W`,
`base_power_W`, `basis_change_J`, `depth_limit_m`, `stability_margin`, `n_modes` (`DynamicsDomain.channel_values`).

## 8. Kernel ABI, signatures, cost

Layouts (`dynamics/abi.py`): DynModel, DynState, DynAccum, ModalDeflection (= cutting's Deflection fields),
WorkGrid, AxisBase, DynSystem (ModalModel, ModalState, DynModel, DynState, DynAccum, ModalDeflection, xp, fe, fx,
scratch); constants `MI_*`, `MS_*`, `FL_*`, `DA_*`, `TU_*`, `BASE_*`.

| kernel | signature |
|---|---|
| `modal_begin` | `void(SYSTEM_T, f8 theta, f8 omega, f8 alpha, f8[::1] base_in[7])` |
| `external_force`, `modal_predict`, `modal_deflection` | `void(SYSTEM_T, f8, f8, f8, f8[::1])`, `void(SYSTEM_T)`, `void(SYSTEM_T)` (inlined into modal_begin) |
| `modal_correct` | `void(SYSTEM_T, f8[:, ::1] station_T0[nst, 6], f8[::1] force_S_N[3], i8 tick, ACCUM_T)` |
| `modal_free` | `void(SYSTEM_T, f8 theta0, f8 theta1, f8 omega0, f8 omega1, f8[::1] base0, f8[::1] base1, i8 tick1)` |
| `record_tick` | `void(SYSTEM_T, i8 row)` |
| `refresh_work_shapes` | `void(SYSTEM_T, GRID_T, f8[::1] point_S)` |
| `relative_axis_acceleration` | `void(AXES_T, f8[::1] acc_axis, f8[::1] out[7])` |
| `process_damping_element` | `f8(f8 K_sp, f8 db, f8 land, f8 tan_clear, f8 max_len, f8 v_n, f8 V, f8 dt, ACCUM_T)` |
| `interference_area` | `f8(f8 slope, f8 land, f8 tan_clear, f8 max_len)` |
| `deflection_at_point` | `void(DEFL_T, i8 station, f8[::1] r_T0, f8[::1] out[6])` |
| `relative_deflection`, `ring_write` | `void(SYSTEM_T, f8[::1] out[9])`, `void(SYSTEM_T, RING_T, i8 slot, i8 col0)` |
| `free_interval` | `void(f8 omega, f8 zeta, f8 h, f8 q0, f8 v0, f8 Pr, f8 Pi, f8 W, f8 g0, f8 g1, f8[::1] out[5])` |
| `phi1`, `psi1`, `j0`, `j1`, `_deflection`, `_relative` | helpers (one signature each) |
| `bench_run`, `cost_contact`, `cost_free` (drive.py) | verification harness |

**Measured cost** (Plint, one core, bin/tsafe, from a compiled loop; `test_cost_and_settings.py`, and the hot
loop's instrumented slot, docs/performance.md; 2026-11-09, a load of 4-6):

| system | L0 step (modal_begin + modal_correct) | free tick |
|---|---|---|
| 5 modes, 1 station | 0.76 us (1.3-1.7 us with reference counting) | 4.7 us |
| 13 modes, 20 stations | 4.5 us (5.1-7.9 us) | 11 us |
| the slot's shape: 4 tool / machine + 4 workpiece modes, 119 stations, deflections at 61 (`act`), loads at 15 | 1.8 us (5.2 us) | |
| the same inside the hot loop (caches shared with cut_step and removal) | 3.5 us (7.6 us) | |

`modal_begin` and `modal_correct` allocate nothing and are compiled without reference counting (`RAW`; with it,
every member array of the system tuple taken in a kernel cost an atomic incref / decref, about half of the former
cost); kernels bind single arrays, never sub-tuples, and the helpers are inlined at the IR level. Their station
loops run over the stations' 6 components as one contiguous run (`u_st`, `v_st`, a mode's `tool_shape` row and the
station loads are C-contiguous [station, 6]): the deflection update is vectorised, every element still summing the
modes in their order, and the load projection sums four modes side by side, each in its order: the results are
bit-identical to the station loops. A tool's static residual flexibility adds a dense 3 x 3 block per (deflected,
loaded) station pair (+2.6 us for 27 x 15 stations). Per simulated second at 10 000 rpm (8 us steps): 0.1 s (5
modes, 1 station) to 0.56 s (13 modes, 20 stations) in contact, 0.44 s for the slot inside the hot loop; 5-11 ms
out of contact.

## 9. Parameters and settings

`db/models/dynamics-chatter.yaml`: chatter_index_threshold 0.5, clear_index 0.35, min_amplitude_m 0.5 um,
min_amplitude_per_feed 0.05, severe_amplitude_per_feed 1.0 (tlusty-ismail-1981-nonlinearity), flip_growth_ratio
1.5, flip_min_share 0.05 (est). `dynamics-process-damping.yaml`: edge_land_per_radius 1.0, max_interference_m
0.5 mm (est, = cutting-contact). `dynamics-modes.yaml`: stride_compliance_share 0.01, stride_omega_dt 0.3, stride_contact_omega_dt 1.0 (2.3)
(architecture 5.1), basis_mac_threshold 0.999, basis_freq_tol 1e-6. Bibliography: `models-dynamics.yaml`.
Settings `physics.dynamics` (`schema.py`): sample_margin, chatter_window_revs (20), chatter_window_max_s,
chatter_min_revs, chatter_clear_steps, max_findings, sld_n_freq, sld_n_lobes, sld_n_speeds, sld_speed_lo_frac,
sld_freq_lo_factor, sld_freq_hi_factor, sld_refresh_macro, sld_max_classes (64 engagement classes kept per tool and
operation), sld_max_diagrams (8 end-of-operation diagrams per tool), engagement_tol, mf_harmonics (0: ZOA), pd_max_iter,
pd_tol, mode_coupling_max_m, sd_check, sd_m, sd_points_per_period, sd_max_points, sd_max_states, sd_a_max_m,
sd_scan, sd_tol,
max_operating_points, relax_estimate, rotation_length_m. Numerical constants of the closed forms (series radius
0.2, Taylor switch omega h = 0.1, 44 terms) are floating-point accuracy choices, documented in kernels.py.

## 10. Assumptions, validity and limitations (consequences)

* Linear modal structure; modes with 0 < zeta < 1 (InputError otherwise). Stations of structure's tool modes
  outside the edge stations are reached by rigid extrapolation (error ~ curvature x dz^2 / 2 of the shape).
* Simultaneous cuts on one workpiece (two turrets) each carry their own copy of its modes: the coupling of the two
  forces through the workpiece is not modelled (each sees the workpiece's response to its own force; pinch turning
  under-predicts the workpiece vibration by up to the other cut's share).
* Machine frame modes: base excitation by the relative rigid-body acceleration with the modal mass k / omega^2 (the
  file does not say which bodies a mode involves: an over-estimate when only part of its mass moves with an axis);
  linear axes keep their home direction (exact when no rotary axis precedes them); the rotary axis' modes use the
  POE axis line at the end of the previous step. Tool-mode base excitation (unknown participation) is not applied
  (tool d'Alembert loads m_tool a <= 5 N vs cutting loads >= 50 N).
* Out of contact the unbalance amplitude is held per tick (ramp error ~ alpha h / W); the tool's own residual
  flexibility under the unbalance is not added.
* Lobes: straight-cut engagement (removal's arc), one mean chip thickness, per-element coefficients of the
  element's group; uniform-pitch diagrams by ZOA (no flip boundary: the SD check and the multi-frequency option
  cover it), variable-pitch diagrams on a coarse speed grid; process-damped ZOA overestimates at low speed in
  interrupted cutting (up to 30 %; the SD check at the operating point uses the time-varying damping).
* Chatter: detection lags onset by up to one window while the chatter grows; saturated (non-growing) period
  doubling of an even-tooth cutter is reported at its onset but not counted afterwards; a stable cut dominated by
  runout of an even-tooth cutter cannot be told from saturated flip chatter spectrally.
* The workpiece grid is taken as cell-centred (the codebase's thermal convention); structure must publish its
  WorkModes the same way.
* Cost per L0 step (section 8) exceeds the architecture's 3.7 us average budget share for systems above ~10 modes.

## 11. References

Allemang & Brown 1982 (allemang-brown-1982-mac); Altintas & Budak 1995 (altintas-budak-1995-zoa); Altintas, Engin
& Budak 1999 (altintas-1999-variable-pitch); Altintas, Eynian & Onozuka 2008 (altintas-2008-process-damping);
Altintas 2012 (altintas-2012-mfg-automation); Ahmadi & Ismail 2011 (ahmadi-ismail-2011-process-damping); Bayly
et al. 2003 (bayly-2003-stability-interrupted); Budak & Altintas 1998 (budak-altintas-1998-multifrequency); Budak
& Tunc 2010 (budak-tunc-2010-process-damping); Craig & Kurdila 2006 (craig-kurdila-2006-structural-dynamics);
Delio, Tlusty & Smith 1992 (delio-1992); Franklin, Powell & Workman 1998 (franklin-1998-digital-control);
Insperger & Stepan 2002, 2004 (insperger-stepan-2002-sd, insperger-2004-updated-sd); Murray, Li & Sastry 1994
(murray-li-sastry-1994); Schmitz, Medicus & Dutterer 2002 (schmitz-2002-once-per-rev); Schmitz & Smith 2009
(schmitz-smith-2009-machining-dynamics); Smith & Tlusty 1991 (smith-tlusty-1991-milling-simulation); Tlusty &
Ismail 1981 (tlusty-ismail-1981-nonlinearity); Tlusty & Polacek 1963 (tlusty-polacek-1963-stability); Tobias 1965
(tobias-1965-machine-tool-vibration); Van Loan 1978 (van-loan-1978-integrals); Wu 1989 (wu-1989-process-damping).
