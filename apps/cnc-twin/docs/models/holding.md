# Holding: tool holding and balance, workholding

Package `cnctwin/holding/` (domain `holding`, an L2 slow domain), tests `tests/unit/holding/`, coefficients
`db/models/holding-tool.yaml`, `holding-work.yaml`, `holding-balance.yaml`, sources
`db/bibliography/models-holding.yaml` (plus catalogue keys already in `holding.yaml`, `materials-2.yaml`,
`models-tooling.yaml`, `models-cutting.yaml`).

| module | content |
|---|---|
| `toolhold.py` | holder clamping per type (ER chain, interference and hydraulic fits, side-lock, tapping, positive), `ToolData` from tooling's `ToolGeometry` |
| `balance.py` | ISO 21940-11 unbalance of tool + holder (vector, height, couple), speed limits |
| `devices.py` | workholding devices as contact patches (Simpson nodes), clamping forces from torque, chuck centrifugal loss |
| `setup.py` | contact groups, locate / clamp / lock solve, location error, seating, support stiffness, lathe support summary |
| `workhold.py` | per-step loads, capacity, mechanism, slip and lift ratchet, bond failure, seal loss, supports |
| `kernels.py`, `reference.py`, `abi.py` | compiled kernels, plain-Python references, layouts |
| `protocols.py` | geometry queries (`PartGeometry`, `TabSections`, `BurrSource`) and stock-based defaults |
| `locators.py` | chips' `LocatorFaces`: the next setup's locating and clamping faces in S |
| `state.py`, `domain.py`, `channels.py`, `schema.py`, `params.py` | domain plumbing |

## 1. Bus record

Reads `bus_prev` only:

* `motion.paths[*]`: `tool_slot`, `workpiece`, `tick`, `spindle_omega_rad_s` and `spindle_angle_rad` per tick
  (speed, tooth phase, angular acceleration), `pose_M_T` / `pose_S_T` (orientation of S in M for gravity),
  `tip_M_m` / `tip_S_m` per tick (the part's translation in M: table acceleration).
* `cutting.contacts[*]`: `tool_slot`, `wrench_tool_T0_mean/_peak` (torque `[:, 5]`, axial `[:, 2]` on the
  tool, T0), `wrench_work_W_mean/_peak` (on the part about the W origin, W), `float_u_max_m` (floating tapping
  holders).
* `removal.contacts[*]` (`contact -> workpiece`, `removed_volume_m3`), `removal.loose_bodies[*]`
  (`workpiece`, `volume_m3`, `mass_kg`, `tick`).
* `chips.locator_debris_m` (patch id -> chip / burr height under it, measured by chips at the end of the
  previous operation on the faces holding gives it, section 6).
* `thermal.tools[slot].holder_temp_K`, `.body_temp_K` (interference fits lose pressure when the holder runs
  warmer than the shank).

Writes only `nxt.holding` (`HoldingFeedback`):

* `tools[slot]` (`ToolHolding`): `pullout_m` (> 0: the tool longer, toward the work; tooling adds it to the
  tool length), `slip_rad` (> 0: the tool lags the spindle, tooling subtracts it from the tooth phases),
  `unbalance_static_kg_m` [3] in the rotating frame T at height `unbalance_z_T_m`, `unbalance_couple_kg_m2` [3]
  (the rotor exerts `omega^2 U` at z_U and the moment `omega^2 e_z x C` about z_U, rotating with T).
  **Additive fields**: `torque_capacity_Nm`, `axial_retention_N` (inf: positive), `holding_margin` (static
  friction-circle capacity / peak demand of the step), `walk_per_rev_m`, `slip_friction_J`,
  `unbalance_force_N` (`|U| omega_max^2`), `balance_quality_m_s` (worst-case G), `max_rad_s`, `float_bottomed`.
* `parts[workpiece]` (`PartHolding`): `pose_delta` (4x4, `p_H = pose_delta p_W`; motion composes it into
  `pose_F_W`), `free`, `seated`, `location_error_S_m`, `supports` (below). **Additive fields**:
  `slip_pose_delta`, `location_pose_delta` (`pose_delta = slip_pose_delta @ location_pose_delta`), `margin`
  (capacity factor of the step's load; inf: no load), `mode` (`slip` | `lift` | ""), `slip_m` (largest
  displacement of the part's bounding-box corners by this step's slip), `lift_m` (the same at the largest
  reversible lift), `clamp_force_N`, `seat_gap_m`, `support_stiffness_S` (6x6 about `centroid_S_m`, S axes, the
  locked contact set's tangent), `centroid_S_m`, `supports_version` (increments when contacts are placed
  again), `lathe` (section 7), `slip_friction_J`, `free_cause`.

`supports`: one dict per contact patch with per-node arrays in S (material configuration): `id`, `role`
(fixed_jaw, moving_jaw, parallel, base, plate, locator, clamp, jaw, stop, centre, roller, bond, tab), `law`
(locator | locked | bond), `device`, `group`, `pos_S_m` [m, 3], `normal_S` [m, 3] (direction of the support's
compressive force on the part), `t1_S` [m, 3], `force_S_N` [m, 3] (on the part, at the last tick's mean load),
`preload_S_N` [m, 3] (clamped state), `stiffness_N_per_m` [m], `tangential_stiffness_N_per_m` [m, 2] (along
t1 and n x t1), `friction`, `area_m2` [m], `state` [m] int8 (`abi.ST_*`: 0 stick, 1 slip, 2 open, 3 bonded,
4 off), `rotating`. Structure applies the forces (clamping distortion, spring-back after unclamping) and the
springs (fixture compliance) from them.

Slip and walk friction work is **reported, not booked** in `ledger.friction_J`: the slip enters the hot loop as
a phase / pose change between macro steps, so the spindle and feed work that drives it is not in cutting's
shaft and feed work either; booking only the dissipation would put an unbalanced term into the energy check.

## 2. Tool holders (`toolhold.py`)

**ER collets** (ISO 15488, 8 deg setting angle). Nut torque T_n = the REGO-FIX recommended torque for the
collet size and shank diameter (14.03, Hi-Q ER column; the project gives no nut torque). Power screw with
collar friction (Budynas & Nisbett 2015 eq. 8-5/8-6) on the ISO thread (d2 = d - 0.649519 P, ISO 68-1):

    T_n = F_a [ (d2 / 2) tan(lambda + rho') + mu_c r_c ],  tan lambda = P / (pi d2),  tan rho' = mu_t / cos 30 deg
    N = F_a / tan(alpha + atan mu_k)            (wedge of the 8 deg cone: total radial force on the shank)
    T_cap = mu_s N d / 2,   F_ret = mu_s N

ER 32 on 20 mm: T_n = 136 Nm, F_a = 28.7 kN, N = 117 kN, T_cap = 176 Nm (test: hand calculation to 1e-12).
Collet closing force is not deducted (shanks at nominal size h6; the closing share grows toward the bottom of
the 1 mm clamping range and lowers the capacity there).

**Friction fits** (hydraulic, shrink-fit, milling chucks), contact pressure p over the clamped length L:
from the catalogue torque `p = 2 T_cat / (mu_s pi d^2 L)` (then `F_ret = 2 T_cat / d`), else shrink-fit and
milling chucks from the interference (Lame, Budynas & Nisbett eq. 3-56, solid shank in a hub of outer
diameter D, `delta_d = 1.3e-3 d` calibrated on the SCHUNK CELSIO 12 mm minimum torque):

    p = delta_d / ( d [ (D^2 + d^2) / (D^2 - d^2) + nu_h ] / E_h + d (1 - nu_s) / E_s )

and hydraulic chucks without a torque at 70 MPa (back-calculated from TENDO E compact 12 mm). Interference fits
lose pressure with the holder warmer than the shank: `p = p0 max(0, 1 - d (alpha_h dT_h - alpha_s dT_s) /
delta_0)` (thermal record; release at about 290 K of difference for the default 12 mm fit).

**Side-lock**: positive torque drive; axial retention `2 mu_s sigma_pre A_s` of the set screw (diameter 0.8 d,
stress area 0.76 pi d^2/4, preload stress 227 MPa: Carr Lane's 33 % of a 100 ksi yield); the shank walks up to
`side_lock_walk_limit_m` (0.2 mm) before the screw bears on the end of the flat. **Tapping holders**: square
drive and ball lock (positive); the axial float is cutting's (`float_u_max_m`), holding raises a warning when
it reaches the holder's travel (`axial_float_mm`, smaller side of a [-c, +t] range). **Positive interfaces**
(modular, arbor, boring head) and **non-rotating holders** have no frictional slip mode.

**Combined loading and slip.** The friction at the bore is one Coulomb circle in (torque / r, axial pull):
`|D| <= C`, `C = min(T_cap / r, F_ret)` (equal for every model above; when a catalogue gives both and they
differ, the smaller is used, conservative). Demand per L1 tick: mean and first tooth harmonic
`D(t) = D_mean + (D_peak - D_mean) cos(N theta_cut(t))` (Altintas 2012 ch. 2: milling forces are periodic at
the tooth frequency, the first harmonic carries most of the fluctuation; the amplitude makes the maximum equal
the tick's peak sample), with `D1 = -sign(omega) M_z / r + sign(omega) J alpha / r` (lag positive, inertia of
the tool in spindle acceleration) and `D2 = -F_z` (pull-out positive). When the demand can leave the circle
(static check with the tooth harmonic's extremes) the 2-DOF ratchet integrates the tool as a rigid body on
the interface, `m1 = J / r^2`, `m2 = m_tool`, implicit Coulomb friction (`v = (M + lambda I)^-1 p`,
`lambda |v| = h c_k`, exact stick test `|p| <= h c`), back stop at the set length, walk limit (side-lock),
the tool leaving the holder at its clamped length (fatal). A torque above the capacity slips the tool and lets a
small axial force walk it out (the friction circle: the physical mechanism of pull-out under torsional
overload).

**The demand's tooth-period shape and the stall (2026-11-09).** The cutting torque is the teeth's pulse train
and it only resists the tool's turning against the work. Within the tick its shape is the pulse of
`abi.Wave`: `D(t) = D_mean + (D_peak - D_mean) g(N theta; alpha)` with `g = (s - <s>)/(1 - <s>)`,
`s(u) = max(0, cos u - cos alpha)/(1 - cos alpha)`; alpha follows the tick's mean / peak of the torque
(`abi.pulse_alpha`: `<s> = mean / peak`): the first tooth harmonic (alpha = pi, `g = cos`) for mean / peak of
1/2 or more, a narrower pulse below (a short engagement, a shock), mean and peak as recorded either way; the
axial pull has the torque's shape. Every sub-step applies the demand's exact mean over it (`shape_mean`: its
impulse), so a shock of a few microseconds is not stretched over half the tooth period: before, a 169 N m peak
sample of a 1 N m mean (examples/thin-wall-al7075, a chip recut, section 13) was integrated as a cosine above
the 88 N m capacity for a third of every tooth period of the whole step, and the ER collet turned 792 deg. The
slip speed is bounded by the stall: the cutting torque cannot turn the tool backwards against the work, so the
circumferential slip speed stays at most `|omega| r` (`wave.v_max`; the demand above it is not transmitted, a
reaction of the stalled cut). An overload therefore stalls the tool at most at the spindle speed (a slip of at
most `|omega| dt` per step) instead of accelerating its small inertia without bound.

**Energy.** The ratchet returns the work the demand did on the slip (`RT_IN_J`), the dissipation (`RT_WORK_J`:
the friction impulse along the new velocity, what a sticking contact takes of the remaining speed, the stops'
losses) and the kinetic energy follows from the state; `in = dissipation + change of (m1 v1^2 + m2 v2^2)/2`
holds to round-off for the discrete scheme (test). The dissipation is booked into `SimState.ledger.friction_J`
(architecture 12: slip friction) and the bus record carries `slip_in_J`, `slip_loss_J`, `slip_mech_change_J`:
the run's ledger books the loads' work on the slip as an in term and the kinetic change as an out term
(docs/models/loop.md 6). The hot loop's own shaft work and heat by zone still assume the tool turning with the
spindle (it applies `slip_rad` as a phase only): the slip's friction heat is booked beside the cut's, not taken
out of it.

Findings (`tool_pullout`): walk below the depth limit (min of `pullout_error_m` 0.02 mm and the project's
tightest tolerance) warning, above it error, the depth error (`depth_error_m` = `pullout_m`) in the quantities;
any slip a warning, above `slip_error_rad` an error; float at its travel a warning; out of the holder fatal.

## 3. Balance (`balance.py`, ISO 21940-11)

`G = e_per Omega`, `U_per = m e_per = G m / Omega` (`U [g mm] = 9549 G [mm/s] m [kg] / n [rpm]`).
Tool mounting: the EdgeModel's runout offset o and tilt (a_x, a_y) displace a point p of the axisymmetric tool
to `p + o + a x p`, giving `U_t = m (o_x + a_y z_c, o_y - a_x z_c)` at z_c and the couple about z_c
`C_t = (a_y, -a_x)(I_t - I_a)` (test: point-mass integration of a displaced cylinder to 1e-4). Holder residual
`|U_h| = G m_h / Omega_G` (its published grade at its speed; G 6.3 at its maximum speed when unrated), tool
residual from its own grade when published; directions drawn once per slot from the counter-based stream
**`holding.unbalance_phase`** (`hash_uniform(seed, key, slot, 0 | 1, 0)`). Assembly: `U = sum U_i` at the
assembly centre of mass, `C = sum U_i (z_i - z_U) + C_t`. Check at the step's largest |omega|: worst case
(magnitudes added) `G_eff = U_worst omega / m` against the required grade (published grade of holder or tool,
else G 2.5 of machine-tool drives, ISO 21940-11 Table 1): `unbalance` warning. Speed above the lower of tool
and holder limits: `overspeed` error.

## 4. Workholding devices (`devices.py`)

Every device becomes patches on the part's bounding box in the device frame D (conventions below); each patch
carries n x n Simpson nodes (`physics.holding.patch_nodes`, default 5) whose weights integrate a uniform
pressure's force, first and second moments exactly (stiffness and tipping edges exact); the torsional friction
capacity of a rectangle (the mean radius) is within 1 % at 5 x 5 nodes for aspect ratios to 10:1 (21-30 % low
at 3 x 3; measured against a 2001 x 2001 grid).

Contact laws: `locator` (stiff unilateral support that reacts), `locked` (clamped force-controlled to its
clamping force, then a preloaded spring), `bond` (bilateral with tensile and shear strength). Normal stiffness
per patch: device stiffness in series with the joint stiffness (1000 N/um per cm2, est.); tangential stiffness
`2 (1 - nu) / (2 - nu)` of the normal (Mindlin; Johnson 1985 7.2). Friction from the item (`jaw_friction`), else
0.19 (Carr Lane) for jaws, clamps and bases, 0.1 (Kitagawa) for chuck jaws.

Frames D: vise / soft jaws: origin on the seat (the top of the parallels, else the vise bed: where the part's
bottom rests), centred on the jaw width, x along the jaw width, y the clamping direction, z up; the jaws rise from
the bed (z = -parallels) to `grip_depth_mm` above the seat (else jaw height - parallels, else the part top) - the
convention of the project template, the examples ("bottom on the jaw steps" at z 0) and sanity's jaw boxes (jaw
tops `grip_depth_mm` above the stock bottom); clamp: origin at the pad, pressing along -z; fixture plate, vacuum,
magnetic, adhesive: origin at the centre of the plate top, z up (or at the plate bottom: when the item gives its
thickness `plate_mm[2]`, the plate face nearer the part bottom, z = 0 or z = thickness, is the one it rests on);
chucks and collets: chuck face centre, z along the spindle axis out of the chuck; tailstock centre: tip, z along
the part axis toward the chuck; steady rest: on the axis in the rollers' plane, rollers at 90 deg + 360 deg j / n;
tabs: section centre, x into the held part, section (y, z) = item `dims_mm` width, height (length = tab length,
default the width); point locators: entries whose `label` starts with `locator`, `stop` or `pin` touch the part at
the pose origin pushing along +x, `support` / `rest` along +z.

* **Vise**: jaw faces where the part's y-faces are; grip zone = the jaws (jaw width x seat to jaw tops) on the
  part; `grip_depth_mm` wins over jaw height - parallels when both are given and disagree (noted); a setup that
  does not place the part between the jaws of the vise frame (a vise frame set at its mounting face, as on the
  grid plates of batch-50, blade-5axis-ti, ti-slot-throughtool) is gripped over the part's bottom `grip_depth_mm`
  (or jaw height - parallels) on the central jaw width and seated (descriptive, noted); a contact shape below the
  seat (the stock shape of a face machined in an earlier operation, e.g. two-setup-flip OP20 with the default
  geometry) rests on the seat (noted); clamping force from the input torque by the item's force per torque, or the
  published force scaled linearly with torque, else the generic power-screw (22 mm Acme, lead 5 mm, est.); base:
  two parallels (6.35 mm) under the jaws, else the vise bed; a part above them is not seated (clearance gap: it
  can slide down until it touches). Kurt-style anti-lift pull-down is not modelled (the lift capacity is
  conservative).
* **Clamps**: force from the catalogue relation, else the stud preload scaled by torque over the lever ratio,
  else `T / (K d)` (K = 0.2) over the lever ratio; a pad of 10 mm on the part's top at the pad origin (a pad
  beside the part is moved onto its nearest edge, noted); setups of clamps only rest on the part face the
  clamps press toward (table). Friction acts at the pads and at the base: the lateral capacity of n strap
  clamps is `2 mu n F + mu m g` (drill-tap-plate: 14.9 kN; its EXPECTED.md counts one interface, 7.4 kN).
* **Fixture plate** + point locators: the 3-2-1 scheme from labels. A plate whose top is more than
  `mounting_gap_m` (1 mm, est.) below the part carries other devices (a vise or risers on a grid plate, a
  zero-point plate): it is skipped with a note and the part rests on the devices on it.
* **Vacuum**: hold-down `(p_atm - p_abs) A_seal` at the footprint centroid (min absolute pressure, else the
  vacuum level, else the holding pressure; `pressure_bar` of the entry read as the absolute pressure reached),
  `A_seal` = footprint x area utilisation; the plate as locator with seal (perimeter) nodes; a seal node that
  opens or a lift of the part breaks the vacuum (persistent) - with nothing else clamping, the part is free.
* **Magnetic**: holding pressure from the item's table over part thickness, times the material factor
  `(B_s / 2.1 T)^2` (Maxwell stress, Furlani 2001; saturation inductions Bozorth 1951: steel 1, tool steel 0.73,
  cast iron 0.38, cobalt 0.73, nickel 0.08, stainless 0 (austenitic, conservative), others 0), times
  `1 / (1 + g / g_c)^2` for chips on the plate (g_c = 0.3 mm, est.). A non-ferromagnetic part: error at setup.
* **Adhesive**: bonded nodes, shear strength (aluminium value for aluminium parts) x bond area utilisation,
  tensile = shear (est.); hot strength not modelled (not digitised in the database).
* **Chucks** (3-, 4-jaw): jaws as cylinder strips over the grip length (`grip_depth_mm` of the chuck entry,
  else of the soft-jaw entry, else the item's grip length, else the soft jaw length, else 15 mm), width from the
  soft jaws (else 26 mm); static force from the item scaled by `pressure_bar / max_actuation_pressure_bar`
  (capped at 1; an item without that field: the "max pressure X MPa / bar" of its `matching_cylinder` text,
  noted; neither: the maximum force, noted), or by force per torque; dynamic force `F_s -+ loss(omega)` (OD /
  ID gripping: `jaws` text `id`), loss by the item's model (`quadratic_in_speed`, `measured` curve, or VDI 3106
  `c sum m r omega^2` with jaw masses or mass moment), else the Kitagawa speed rule (2/3 of static at the
  maximum speed); axial stop at the jaw steps unless `jaws` says `no_stop`/`bar`; the setup is clamped again
  when the centrifugal loss has moved the gripping force by more than `speed_rebuild_rel` (2 %) of a chuck's
  static force since the build (the loss at the device's maximum speed, `loss_ref_N`, scaled by omega^2): before
  2026-11-09 the rule was 2 % of omega^2 itself, so a spindle running up from rest rebuilt the setup at every
  step of the run-up (examples/lathe-shaft-4140: 55 setups in 2.4 s, each a new work model in structure and a new
  modal basis in dynamics) for grip changes of a fraction of a percent; zero gripping force: the part is free;
  device speed limits: overspeed.
* **Collets** (lathe): three segments over most of the circumference, gripping force from the item (or max
  input force x pull-back ratio), range check; centrifugal loss of the segments not modelled (light, closed by
  the collet closer).
* **Tailstock centre**: n nodes on the 60 deg cone at `max(0.08 R, 1.5 mm)`, normal `sin(alpha_h) a +
  cos(alpha_h) e_r`, thrust (the machine's lower thrust, else 5 kN) as force-controlled preload, frictionless
  about the axis (a live centre turns with the part), friction 0.1 along the cone; it slides along its cone
  while the thrust is applied.
* **Steady rest**: roller nodes at the rollers' angles, force from the item's force-pressure relation at the
  entry pressure (else the maximum), roller stiffness from the item, frictionless in the rolling direction.
* **Tabs**: bonded nodes over the section, `sigma_t = sigma_u`, `tau = sigma_u / sqrt 3` (distortion energy),
  axial stiffness `E A / L`, shear `G A / L`; the section area from `TabSections` (removal) scales nodes and
  strengths; the tab-held part rests on the setup's primary datum (its weight goes there, the tabs carry the
  cutting load: for a part hanging in air only on its tabs this misses the tab bending from its weight).
* **Rebuilds after slip** (the part moved more than `rebuild_slip_m`): the devices stay where the setup as
  programmed put them (the jaw zone and seat of a vise, descriptive or not, the pad point of a clamp, a plate)
  and grip the moved part where it still meets them; a part slid out of the jaws, from under a pad or off its
  plate is not gripped there (`no_grip`), a lifted part is not seated (gap).

## 5. Statics, slip, lift and freeing (`setup.py`, `workhold.py`, kernels)

Rigid part on elastic contacts with Coulomb friction (Li & Melkote 1999): twist delta = (u, phi) about the
part's centre of mass O, `d_i = u + phi x r_i`, `f_n = pre_n + k_n (gap - n.d)` (unilateral, closed at f_n =
0), `f_t = pre_t - k_t (t.d)` with the return mapping `|f_t| <= mu f_n` (holonomic friction from the reference
configuration), bonded nodes bilateral; residual `R = w + sum G f - K_reg delta` and its consistent tangent;
Newton with backtracking line search (Armijo), 1 load increment with a retry at 8 when Newton does not converge,
a step beyond 100 x the displacement limits is a mechanism; "held" = converged with |u| <= 1 mm, |phi| <= 0.01
rad. Deterministic: zero start, fixed iteration order and tolerances.

Clamped state (per setup and contact group; groups: `main` = all devices but tabs, `tabs` = tabs + the primary
datum): (1) *locate*: clamps force-controlled, no friction, gravity and hold-down forces, debris as
interference at the worst node of each locating patch (the node whose lift moves O most, `|u| + l_c |phi|`);
if a load component has no frictionless locator, seat under gravity with friction instead; (2) *clamp*:
re-anchored, friction on; (3) *lock*: forces become preloads, clamps stiff springs.

Per macro step: loads per tick in H about O (cutting mean / peak + inertia: `-m a` from the part's origin in M
by second differences of `tip_M - R_M_S tip_S`, or `-I alpha` about the chuck axis; gravity change since the
build: orientation and mass removed), bonded nodes fail progressively at the worst tick (elliptic tension /
shear interaction, worst node first, persistent), the capacity factor lambda of the worst tick's peak load by
doubling and bisection (`capacity_tol` 1e-3 below a margin of 2, 10 % above), the other ticks solved unless
within 1.8 deg of the worst load's direction and not larger with 5 % reserve. Below 1: the mechanism m (tangent
with the critical contacts released: utilisation >= 1 - eps slip, normal force <= eps of the largest open;
regularised at 1e-8 of the stiffest contact) normalised to `m^T M m = mass`, `Q_cap = m^T (w0 + lambda w_peak)`,
demand `Q(t) = m^T (w0 + w_mean) + m^T (w_peak - w_mean) g(N theta; alpha)` (the tooth-period pulse of section 2,
alpha from the tick's `|F_mean| / |F_peak|` of the cut; the exact mean per sub-step); the 1-DOF ratchet
integrates the motion: *friction mode* (a slipping contact: slip, rotation in the jaws or chuck, pull-out from a
chuck) leaves a persistent displacement - the pose delta, contacts placed again on the moved part after 50 um;
*separation mode* (lift-off, tipping against gravity and clamps) comes back to the seat (inelastic landing at
its exact instant) unless the clamps lose the part. In friction mode the slip is no faster than the cut can push
the part (`StepLoads.v_limit_m_s` at the cut's contact point, `wave.v_max = v_limit / |m_v + m_w x r_c|`): twice
the tool's feed speed relative to a table-mounted part (a part retreating faster than the tool advances plus the
chip thinning of one feed per tooth per tooth period loses the cut and the load with it), the cutting speed at
the contact of a part turning in a chuck (the cutting torque cannot turn it backwards); the demand above that is
not transmitted. A sustained overload then pushes the part along with the tool instead of accelerating it
freely (examples/two-setup-flip moved its blank 38 mm in one step before). The energies (`RP_IN_J`,
`RP_LOSS_J`, the kinetic and lift potential energy) are booked as for the tool (`PartHolding.slip_in_J`,
`slip_friction_J`, `slip_mech_change_J`); a part held again after moving dissipates what is left of its speed
(and lands from a lift).

Free (`part_free` error, cause chain): every contact failed (tabs released), the moved part out of its devices,
a lift beyond the clamps' engagement, the vacuum lost with nothing else clamping, the gripping force zero at
speed, the setup unable to hold the part, a removal-released body holding >= 5 % of the workpiece volume
(warning on a rotating workpiece: parted off - check the part catcher). `part_slip`: persistent slip or lift
above 1 um (error), margin below the safety factor while cutting (warning; 2.0 Carr Lane, 2.5 Kitagawa for
chucks).

## 6. Location error, seating, chips and burrs

`location_error_S_m` = displacement of O by the debris (located twist with debris minus without; small-twist
rigid body, Asada & By 1985 in its elastic form); `location_pose_delta` its transform; `seated` = every primary
datum node in contact or within 5 um. Findings `mislocation`: `|u| + l_c |phi|` above 0.01 mm warning, above the
tighter of 0.05 mm and the project's tightest tolerance error; not seated warning. Chips: `HoldingLocators`
(chips' `LocatorFaces`) gives `faces_S(op)` = the locating and clamping patches of operation op + 1 in S (ids
`<op>/<entry>:<device>/<role>`); chips measures debris there at its end of operation; holding reads
`locator_debris_m` when it builds that setup. Burrs from the surface domain come through `BurrSource` (not wired
by default: burrs then come only through chips' debris, which includes removal's burr events).

## 7. Lathe support conditions

`PartHolding.lathe` (cylinder stocks): `chuck` {z_face_S_m, grip_centre_S_m, grip_length_m, jaws,
gripping_force_N (dynamic), radial_stiffness_N_per_m (half the jaw sum: the radial stiffness of three jaws at
120 deg), tilt_stiffness_Nm_per_rad (jaw stiffness distributed along the grip length), inside}, `tailstock`
{z_S_m, thrust_N, radial_stiffness_N_per_m, live}, `steady` [{z_S_m, rollers, force_per_roller_N,
stiffness_N_per_m}] - positions along S z (the stock axis), for structure's beam model and dynamics.

## 8. Settings (`physics.holding`)

`patch_nodes` 5, `centre_nodes` 8, `newton_tol` 1e-9, `newton_max_iter` 60, `load_increments` 1,
`max_disp_m` 1e-3, `max_rot_rad` 1e-2, `regularisation` 1e-12, `capacity_tol` 1e-3, `margin_cap` 1000,
`critical_eps` 1e-2, `sub_steps_per_tooth` 256, `max_sub_steps` 16384, `speed_rebuild_rel` 0.02,
`rebuild_slip_m` 5e-5, `max_findings` 3 (per kind, tool or part and operation), `every_n_macro` fixed 1.

## 9. Parameters and sources

Every coefficient is in `db/models/holding-*.yaml` with its source or `est` note. Sourced: REGO-FIX ER tables
(sizes, threads, collet front diameter and length, recommended nut torques), ISO 15488 (8 deg), ISO 68-1
(thread profile), Budynas & Nisbett 2015 (steel E, nu, alpha; screw and bolt torque; distortion energy), Carr
Lane (friction 0.19, safety factor 2, stud preload basis), Kitagawa (chuck friction 0.1, safety factor 2.5,
grip length 15 mm, speed rule), ISO 21940-11 (G 2.5). Estimates (with their basis in the notes): thread, collar,
cone and shank friction, collar radius, interference ratios, hydraulic pressure, clamp length and nose ratios,
side-lock screw data and walk limit, joint and device stiffnesses, parallels, pad, centre and roller data,
tailstock thrust, magnetic factors and gap, adhesive tension, mislocation and slip thresholds, default holder
grade.

## 10. Validity, error bounds and how they were established

| quantity | validity | error bound | basis |
|---|---|---|---|
| ER collet torque | ER 8-50, shanks in the collet range, recommended nut torque | +-25 % | chain vs REGO-FIX 89 Nm measurement on 12 mm (model 105 Nm, +18 %); friction coefficients 0.1-0.16 |
| fit torque from catalogue | as published (minimum torque, h6 dry shank) | catalogue value | oil on the shank lowers mu by up to half: not seen |
| fit torque from interference | 6-32 mm carbide or HSS shanks | +-35 % | interference tolerance class and mu 0.1-0.2 |
| vise / clamp slip threshold | rigid part, flat jaws, force through the contacts | 0.15 % numerical (capacity tolerance); friction coefficient +-30 % physical | test vs 2 mu F + mu_b m g; Carr Lane's value is one example |
| chuck gripping torque | OD/ID, 3-4 jaws | numerical 0.2 %; loss model as published | test vs Kitagawa example 176 Nm |
| VDI 3106 loss | jaw masses known | exact formula | 34.8 kN vs 38 kN published (master jaws, factor 1.09) |
| torsional friction of a patch | rectangles to 10:1 | 1 % (5 x 5 nodes) | Simpson vs 2001^2 grid |
| location error | debris << part size, stiff locators | 1 % of the rigid-body value | test 3-2-1 vs h/2, h/30 mm |
| slip ratchet | gross slip | numerical: part < 0.01 %, tool 0.1 % (unequal masses) at the default 256 sub-steps per tooth | convergence studies against 8192-16384 sub-steps (tests); physically an upper bound (section 11) |
| unbalance | rigid rotor, small runout and tilt | 1e-4 (couple), exact (static) | point-mass integration |

## 11. Assumptions and limitations (consequences)

* Gross slip is integrated with the step's cutting loads (their tick means and peaks, the tooth-period pulse of
  section 2) held fixed within the step, bounded by the slip speeds a cut can drive (the stall of a tool, a part
  pushed along by the tool): the cut's change by the slip within the step is not resolved, so slip and walk
  magnitudes above the capacity are upper bounds; the onset (margin 1) is the reliable prediction. Micro-slip ratcheting below gross capacity (fretting walk of um per revolution at 70-100 % load)
  is not modelled: such walks are not predicted.
* Holonomic friction from the clamped state for every load sample (history only through gross slip and failed
  bonds): partial-slip hysteresis between ticks is not carried.
* Kinetic = static friction (one coefficient per pair in the databases).
* Contacts on the bounding box of the part in the device frame: exact for prismatic and cylindrical parts set
  square to their devices; a part rotated in its vise is gripped at its extreme edges. Contacts of a moved part
  are placed again only after 50 um of slip; until then the original contacts move with it.
* The workpiece is one rigid body for holding; the tab-held part carries the full cutting wrench and rests on
  the primary datum; part flexibility (clamping distortion) is structure's.
* Part acceleration from the tool tip and the S orientation at the end of the step: rotary-axis accelerations
  of a tilting table within a step are not included; rotating workpieces: angular acceleration about the axis
  only (no unbalanced-part centrifugal load).
* Mass falls with the removed volume, the centroid and shape of the default geometry stay at the stock's
  (gravity moment error <= m g x centroid shift); wire `PartGeometry` for the real shape. Defaults:
  `StockGeometry` (block, cylinder, tube) and `MeshStockGeometry` for `kind: mesh` stocks (volume, centroid
  and inertia of the closed mesh by signed tetrahedra, exact for a closed outward mesh; contacts on its
  bounding box, a cylinder about S z when the box is centred on the axis with equal x and y extents within 1 %;
  an off-axis or non-square turned blank is a box).
* A push into a stiff support "fails" at the elastic displacement limit (1 mm at the support stiffness): margins
  against pushing into locators are numerically capped, never a real failure mode.

## 12. Cost (measured on this box, single core, after JIT)

Per macro step and workpiece: vise (100 contact nodes) 0.8 ms without load, 1.4 ms at margin 16, 2.1 ms at
margin 1.1; 3-jaw chuck (75 nodes) 0.7 ms / 1.9 ms; tool holding < 0.1 ms per active tool when no slip is
possible (static check), the ratchet when it runs 0.1-0.7 ms. Setup build 12 ms (vise), 89 ms with a debris
search (once per operation). Kernels: contact evaluation 10 us per 100 nodes, an equilibrium solve 40 us.

## 13. Integration

`HoldingDomain(geometry=..., tabs=..., burrs=..., tools=...)`: wire `geometry` (PartGeometry: `shape_S`,
`mass_properties_S`, `face_fraction`) and `tabs` (TabSections: `section_S`) to removal's queries, `burrs`
(BurrSource) to the surface domain; `tools` defaults to `state.tooling.geometry(slot)` (tooling's init runs
first). Chips: `ChipsDomain(..., locators=holding_domain.locator_faces(ctx))`. Motion already composes
`pose_delta`, tooling already applies `pullout_m` / `slip_rad`; the hot loop must apply the unbalance force and
couple (`omega^2 R_z(theta) U`, `omega^2 e_z x R_z(theta) C` about `unbalance_z_T_m`). Structure reads
`supports`, `support_stiffness_S`, `lathe`.

## References

* Altintas, Y. (2012) Manufacturing Automation, 2nd ed., Cambridge UP (`altintas-2012-mfg-automation`).
* Asada, H.; By, A.B. (1985) Kinematic analysis of workpart fixturing ..., IEEE J. Robotics and Automation
  RA-1(2) 86-94 (`asada-by-1985-fixturing`).
* Bozorth, R.M. (1951) Ferromagnetism, Van Nostrand (`bozorth-1951-ferromagnetism`).
* Budynas, R.G.; Nisbett, J.K. (2015) Shigley's Mechanical Engineering Design, 10th ed.
  (`budynas-nisbett-2015-shigley`).
* Carr Lane: clamping force calculations; strap clamp forces (`carrlane-clamping-force-calc`,
  `carrlane-strap-clamp-force`).
* Furlani, E.P. (2001) Permanent Magnet and Electromechanical Devices, Academic Press (`furlani-2001-magnets`).
* ISO 15488:2003, ISO 68-1:1998, ISO 21940-11:2016 (`iso-15488-2003`, `iso-68-1-1998`, `iso-21940-11-2016`).
* Johnson, K.L. (1985) Contact Mechanics, Cambridge UP (`johnson-1985-contact-mechanics`).
* Kitagawa: Gripping force setting of power chucks; B-200 manual (`kitagawa-gripforce-setting`,
  `kitagawa-2021-b200-manual`).
* Li, B.; Melkote, S.N. (1999) An elastic contact model for the prediction of workpiece-fixture contact forces
  in clamping, J. Manuf. Sci. Eng. 121(3) 485-493 (`li-melkote-1999-elastic-contact`).
* REGO-FIX (2009) Technical Information System ER (`regofix-2009-er-technical-information`); ER flyer 2025.
* VDI 3106:2004 Determination of permissible speed of lathe chucks (`vdi-3106-2004`).
