# Thermal domain (`cnctwin/thermal/`)

Heat in tools, holders, spindles, machine axes, the structure, fixtures and workpieces; thermal
damage of the machined surface; one energy ledger per thermal solver. The domain is an L2 (macro
step) domain: it takes no part in the hot loop (`kernel_inputs`/`absorb` return `None`); its
coupling to the forces runs through the tool and preheat temperatures that cutting's
coefficient pipeline reads (architecture 5.3).

| file | what |
|---|---|
| `partition.py` | single-owner split of the zone heat into chip, tool elements, workpiece |
| `toolmodel.py` | tool + holder ladder (1-D finite volumes), per-element edge field, growths |
| `workpiece.py` | cut-cell finite volumes on the stock lattice, fixture body, two-scale contact temperature |
| `machine.py` | spindle, ballscrew axes, structure (first-order bodies, exact integration) |
| `damage.py` | burn / over-temper / white layer / melt thresholds -> `thermal_damage` findings |
| `kernels.py` | Numba kernels (one signature each); `reference.py` their plain-Python references |
| `response.py` | dimensionless step-response tables of a heated rectangle (edge model) |
| `convection.py` | free convection, rotating cylinder, pipe flow, radiation |
| `props.py`, `inputs.py` | properties from the databases; robust reading of project/machine files |
| `protocols.py` | `WorkGeometry`, `ToolEdges`, `FixtureContact` (wired by the integration) |
| `state.py`, `schema.py`, `channels.py`, `domain.py`, `ledger.py` | domain plumbing |

Model coefficients: `db/models/thermal-{convection,tool,work,machine,partition}.yaml`, sources in
`db/bibliography/models-thermal.yaml`. RNG streams: **none** (no stochastic thermal phenomena).

## 1. Heat distribution (single owner, architecture 12)

Per contact the hot loop reports the generation by zone: shear `Qs`, rake (secondary) `Qr`, flank
(tertiary) `Qf` and ploughing `Qp` (J in the step). ProcessCoefficients give per element group
and chip thickness the fractions `f_sc` (shear -> chip), `f_rt` (rake -> tool), `f_ft`
(flank -> tool). Thermal splits once:

    chip = sum_e Qs_e f_sc + Qr_e (1 - f_rt)
    tool = sum_e Qr_e f_rt + (Qf_e + Qp_e) f_ft          (per element, into the tool ladder)
    work = sum_e Qs_e (1 - f_sc) + (Qf_e + Qp_e)(1 - f_ft)

* **Ploughing** (rubbing below h_min, edge forces) happens at the edge-flank contact and takes the
  flank fraction.
* **Element shares** of a zone (`Q*_e`) follow the per-element integrals of the step: shear ~
  `int_h_dt * db * r` (removed volume x speed), rake ~ `int_sigma_v_rake * h zeta^n * db` (normal
  stress x sliding over the contact area `l_c db`, uniform friction coefficient along the edge),
  flank and ploughing ~ `int_sigma_v_flank * l_f * db`. If an integral is zero everywhere the next
  proxy is used (sliding distances, then in-cut time x db x r, then equal shares of valid
  elements). Fractions are interpolated linearly in log h on the table of the element's group at
  its mean chip thickness `int_h_dt / in_cut_time`.
* **Tables in force**: the heat in `bus_prev` was generated with the tables cutting published one
  step earlier; thermal keeps them (`ThermalState.fractions`) and uses those, falling back to the
  current tables for a new tool. With no table at all (an invariant violation by cutting) the
  literature fallback of `thermal-partition.yaml` is used and a trace event
  `thermal:partition_fallback` is written.
* The destinations are computed independently; `chip + tool + work = generation` to round-off
  (test `test_partition_destinations_sum_to_the_generation`). The chip share is published per
  contact (`ThermalFeedback.heat_to_chip_J`, one step later in `bus_prev`): the chips domain's
  chip enthalpy should be this energy (single owner), plus `unplaced_heat_J`.
* **Work heat location**: cutting's `work_heat_S` rows give *where*; their energies are scaled so
  that their sum is thermal's work share (the scale is the channel `thermal.work_deposit_scale`,
  1 when cutting used the same fractions). Without rows the heat is put at removal's contact
  point (trace event `thermal:work_heat_at_contact_point`).

## 2. Tool, holder and spindle nose

### 2.1 Body: axial ladder

Tool strand from the tip: fluted section (solid area `pi D^2/4 - n_flutes * gullet_area`, or
`fluted_solid_fraction * pi D^2/4`; wetted perimeter `fluted_perimeter_factor * pi D`), neck,
shank; exposed over the stickout, inside the bore above the holder nose. Holder strand from the
nose to the gauge line (outer diameter from `profile_mm`, bore removed over the clamp/bore depth),
cross-sections scaled so that the strand plus a lumped **taper node** carry the catalogue
`mass_kg` (taper node mass = catalogue mass - strand mass, at least the taper frustum). Over the
clamp length tool and holder nodes at the same height are coupled by a contact conductance
`h_shank * pi d_shank dz`; the taper couples to the spindle nose by `h_taper * pi d_mean L_taper`
(7/24 steep tapers 30/40/50 and HSK 40-100 from `thermal-tool.yaml`, an unknown taper as size
40). Turning tools: insert (length ic, section ic x thickness) - seat contact
(`insert_seat_area_ratio * ic^2`) - holder bar (overhang, then the clamped length coupled to the
turret at the structure temperature).

Per node `C dT/dt = conduction + Q_elem - g_air (T - T_air) - g_cool (T - T_fluid) - g_sp (T - T_spindle)`:

* axial conductance between nodes: series half cells `1 / (dz_a / 2 k_a A_a + dz_b / 2 k_b A_b)`
  with k(T) of the tool material / holder steel (db tables);
* lateral air: rotating cylinder `Nu_D = 0.133 Re^(2/3) Pr^(1/3)`, `Re = omega D^2 / nu` (Kays &
  Bjorklund 1958, valid Re < 4.3e5; above it the value at the limit), never below free convection
  of a horizontal cylinder (Churchill & Chu 1975b); plus linearised radiation (emissivity of tool
  steel for HSS and holders, the default otherwise) - in the magazine free convection only;
* coolant on the fluted (insert) surface: `reach * h_coolant` (coolant's `h_flank` of the contact,
  else `h_work_far`), the rest of that surface `(1 - reach) * h_air`; through-tool channels:
  Dittus-Boelter / laminar Nu = 4.36 with the coolant's properties at the fluid temperature;
* heat of each element enters the node at its height `elem_z_m` (linear weights).

Backward Euler in delta form, banded Cholesky (bandwidth 2 by node ordering: tool nodes, then
interleaved tool/holder pairs over the overlap, then holder nodes, then the taper). Energy
identity per step: `sum C dT = sum Q - dt sum_c g_c (T^(n+1) - T_c)` exactly (ledger `tool.<slot>`).
The active tool of each path is stepped every macro step; idle tools (magazine/turret) are brought
up to date lazily in `idle_tool_steps` implicit steps when used again or at operation end.

**Growth** (relative to the temperature at the start of the run, when lengths were set):
tool length = `sum over stickout nodes of eps(T) dz`, holder = `sum over the holder strand`,
radial per element = `r_e eps(T_body(z_e))`, with the free strain from the mean-CTE tables
`eps = alpha_m(T)(T - 20 C) - alpha_m(T0)(T0 - 20 C)`. For turning tools `length_growth_m` is along
the bar (insert + overhang) and `radial_growth_m` is zero.

### 2.2 Edge: periodic local field (Stephenson & Ali 1992)

The edge is a quarter space (rake and flank faces adiabatic away from the contacts). The rake
contact `l_c x db` (from the edge along the rake) mirrored across the flank is a rectangle
`2 l_c x db` on a half space; the flank contact `l_f x db` likewise. A unit flux switched on at
`t = 0` raises the temperature by `(a/k) Phi(Fo, m)`, `Fo = alpha t / a^2`, `m = b/a` (b = db/2):

    Phi_c(Fo, m) = int_0^Fo (pi s)^-1/2 erf(1 / 2 sqrt s) erf(m / 2 sqrt s) ds        (centre = edge)
    Phi_m(Fo, m) = int_0^Fo (pi s)^-1/2 M(1, s) M(m, s) ds,  M(c, s) = erf(X) - (1 - e^-X^2)/(X sqrt pi), X = c/sqrt s

(product of the instantaneous plane source and two strip factors, Carslaw & Jaeger 1959 10.4).
Tables of `log10 Phi` on 24 points/decade in Fo (1e-8..1e8) and 32/decade in m (10^-2.5..10^2.5),
16-point Gauss-Legendre per interval; below the table the 1-D limit `2 sqrt(Fo/pi)`, above it
`Phi_inf - 2m/(pi^1.5 sqrt Fo)`. `Phi_c,inf = (2/pi)(asinh m + m asinh 1/m)`; the square's mean
`Phi_m,inf(1) = 0.9464`.

Contact sizes: `l_c = h * zeta^n` (Poletika, n = 1.5; zeta = 1 / chip_ratio or the compression
ratio, clipped to `max_contact_ratio`, at least the edge radius), `l_f = VB + edge_radius`
(VB from wear). Fluxes while in cut: `q_r = Q_rake,tool / (l_c db t_in)`, `q_f = Q_flank,tool / (l_f db t_in)`.

Interrupted elements (entries >= 1, in-cut time < step): heating `t_h = t_in / n_entries` every
period `T = dt / n_entries`; with `N` cycles since the element entered continuous engagement the
rise at phase `tau` of the current cycle is the superposition

    theta(tau) = q sum_{n=0}^{N-1} [F(tau + nT) - F(tau + nT - t_h)]

the first `sa_direct_cycles` (4) term by term, the rest by the midpoint integral
`(W(t_hi) - W(t_lo)) / T` with `W(t) = int_{t-t_h}^t F` (Gauss-Legendre, `window_points`).
Continuous elements (turning, slotting): `theta = q F(t_eng)`. From the semi-infinite field the
part that the 1-D body already holds is removed: the far field of the point source at the edge of
a quarter space `P / (pi k r)` at the body radius `r_far = max(far_field_radius_ratio * r_e,
2 max(l_c, l_f, db))`, times the duty and `erfc(r_far / 2 sqrt(alpha t))`. The real wedge angle
`beta = pi/2 - rake - clearance` scales the rise by `(pi/2) / beta` (exact for the far field of a
point source on the edge of a dihedral wedge). Cross terms: the flank source seen over the rake
contact (and vice versa) is the other source's edge value times the mean axial attenuation of a
uniformly heated disc of the same area, `(sqrt(R^2 + z^2) - z) / R` averaged over the own contact.

Outputs per element at the nq Gauss points of the heating interval (`q_weights` sum to 1, so wear
integrates nonlinear rates as `sum_q w_q rate(T_q)`): `edge_temp_q_K` (mean over the rake
contact), `flank_temp_q_K` (mean over the flank contact), `edge_peak_q_K` (the cutting-edge line,
both sources), `rake_temp_mean_K` (weighted mean, for the coefficient pipeline), and
`edge_cycle_amp_K` = edge-line temperature at the end of heating minus at the end of cooling.
`edge_every_n_macro` (default 2): between evaluations the last local rise is kept on top of the
current body temperature for elements still in cut (the field is quasi-steady on the scale of a
macro step; the counters N and t_eng advance every step).

### 2.3 Validity and error

* `l_c, l_f, db << r_e` (quarter-space approximation of the edge region; error ~ l / r).
* Local field established within `l^2 / alpha` (1-4 ms for carbide at l = 0.1-0.3 mm): the
  quasi-periodic field is accurate after the first macro step of an engagement; for HSS and large
  contacts (l^2/alpha up to ~40 ms) the first steps of an engagement are overestimated. An element
  out of cut for one macro step restarts its cycle count (cooling over l^2/alpha assumed complete).
* Convection at the edge itself is neglected (`h sqrt(alpha T)/k < 0.05` for h < 1e4 W/m2K);
  coolant acts on the body.
* Table interpolation error < 2e-4 relative (test against adaptive quadrature); superposition tail
  < 3e-3 relative against brute force over 300 cycles (`test_stephenson_ali.py`).
* Cross terms between rake and flank by the disc attenuation: within ~10 % of the exact rectangle
  field for l_f / l_c in [0.05, 1] (steady-state estimate).
* Wedge correction: exact for the far field; overestimates the near field of a source adjacent to
  the edge by up to `((pi/2)/beta - 1)/2` (~12 % at beta = 72 deg).
* Body: 1-D cross-section mean; radial gradients in the shank and holder are not resolved (Biot
  number h r / k < 0.01 for the convective boundary; the contact zone gradient is the local field).
* Overall tool temperature uncertainty is dominated by the partition fractions and contact length
  (each +-30 %): expect +-20-30 % on the local rise.

## 3. Workpiece

### 3.1 Grid and cut cells

A uniform grid of cell `physics.thermal.cell_mm` over the stock's bounding box in S (origin at the
box's low corner, i.e. co-registered with the stock lattice; material configuration). Material
fractions come from removal through `WorkGeometry.fill_fraction` (the full grid at the start, then
only the box that `changed_box_S` reports, plus one cell). Cells with `f <= min_fraction` are
empty. Mass `rho(20 C) dx^3 f` (architecture 4.1); state = specific enthalpy h (J/kg rel. 20 C) per
cell, T = T(h) from the exact piecewise-quadratic enthalpy of the piecewise-linear cp table.

* **Exposure**: a face is exposed when its neighbour is empty or outside the grid.
* **Conduction** between cells i, j along axis d: `G = 1 / (R_i,d + R_j,d)`, half-cell resistance
  `R = [w_d f + (1 - w_d) / f] / (2 k(T) dx)` with the interface orientation weight `w_d` from the
  exposure pattern (`e_d^2 / sum e^2`), else from the fraction gradient, else 1/3. This is exact
  for a slab normal to d (series path `f dx / 2`) and for a slab along d (section `f dx^2`) and
  reduces to `k dx` for full cells. Stored as float32 (memory); every sum in float64.
* **Exposed areas** (`face_fraction`): full cells `dx^2` per exposed face; a cut cell with exactly
  one opposite pair exposed (a wall or floor thinner than the cell) `dx^2` for both faces of the
  pair and `f dx^2` for its edges - this is the **shell model** of thin walls: through-thickness
  uniform, in-plane conduction over the wall's own section, both faces convecting, capacity of the
  wall (valid for Bi = h t / k < 0.1 and `t^2/alpha << dt`; test
  `test_thin_wall_inside_one_cell_cools_through_both_faces`); other cut cells one oblique facet,
  `sqrt(sum e_d^2)/sum e_d dx^2` per exposed face (exact for 45 deg staircases; a voxel surface
  error up to sqrt 3 otherwise).

### 3.2 Time stepping: locally one-dimensional backward Euler

Every `every_n_macro` macro steps (energy accumulated in between exactly as heat rows):

1. fraction changes: removed material leaves with its cell's h (`removed_enthalpy_J`), re-added
   material (side flow, burrs, forming) enters with the cell's h, the mean of its material
   neighbours or the ambient enthalpy;
2. the accumulated rows are deposited into the remaining material: half a cell inside the
   generated surface along its normal, trilinear weights times the remaining fraction, marching
   inward up to `deposit_search_cells`, then the nearest material cell; turning rows are spread over
   the arc the part carried past the tool (`-omega dt / n_rows` about z_S, sub-points one cell
   apart). Heat that finds no material within the search is **unplaced** (it left with removed or
   freed material): it is booked as removed enthalpy and published (`unplaced_heat_J`);
3. one step `(C + dt (A_x + H_x)) T_x = C T^n + src + dt H_x T_env`, then the same along y from
   `T_x` and along z from `T_y` (Yanenko's fractional steps), each a tridiagonal (Thomas) solve along
   the grid lines, empty cells cutting the lines; exposed faces are implicit in the sub-step of
   their axis. Each factor is an M-matrix solve with unit row sums, hence a stochastic matrix: the
   scheme satisfies the discrete maximum principle and is unconditionally stable for any cut-cell
   fraction and step (test `test_stability.py`: 20 % and 1 % cut cells, steps 10-10^4 times the
   explicit limit). Douglas-Gunn ADI was implemented first and **rejected**: with cut cells its
   split operators do not commute and the 20 % test diverged (to -1e6 K) while conserving energy.
   The splitting is first order in dt (as backward Euler); its steady state with sources differs
   by O(dt).
4. Conservation: `sum C (T_z - T^n) = sum src - sum_faces dt H (T_after_its_sub-step - T_env)`
   exactly; h += C dT / m. The ledger `work.<w>` compares the stored internal energies before and
   after with deposits - (air + coolant + radiation + fixture contact) - (removed - added).

Boundary classes on exposed faces: **air** free convection per face orientation from gravity in S
(pose of the path machining the part): vertical faces Churchill-Chu (1975a) with the part height,
horizontal faces Incropera 9.6.2 (`0.54 Ra^1/4` / `0.15 Ra^1/3` up, `0.27 Ra^1/4` down, L = A/P),
evaluated at the mean exposed temperature of the previous step and the material bounding box;
**radiation** linearised at the cell temperature to the enclosure at ambient, emissivity per
material family (`thermal-convection.yaml`); **coolant** on dry faces replaced by: near-cut zones
(radius `near_cut_radius_ratio x tool diameter` around each contact, coolant `h_work`), wetted
columns of coolant's 2-D wetted grid over the S xy-plane (`h_work_far`), or every exposed face when
the coolant is on and no wetted grid is given; **fixture contact** on contact faces
(`h_contact = 3000 W/m2K`) to the lumped fixture. Contact faces come from `FixtureContact` when
wired, else the faces most aligned with gravity on the lowest material layer (part resting on its
base; vise jaws are not inferred).

**Fixture**: one lumped body (catalogue workholding mass or `fixture_default_mass_kg`, steel),
area `fixture_area_factor V^(2/3)`, surface h `fixture_surface_htc`, contact to the table
(structure) over `V^(2/3)`; it takes the part's contact heat and chips' `heat_to_fixture_J`.
Backward Euler, ledger `fixture.<w>`; its table loss is given to the structure body.

### 3.3 Two-scale local temperature at the contact

    T_local = T_FV(cell) + [theta_feed - <theta_feed>_cell] + theta_flash

* `theta_feed`: the work-bound heat rate of the contact (`work_J / dt`) on a rectangle moving at
  the feed speed over a half space, half length along the feed = half chord of the engagement
  `R sin(psi/2)` (milling; psi = removal's engaged angle) or half the chip width (turning), width =
  ap (milling) or the circumference (turning), active for the time since the last finite-volume
  deposit (capped by the contact's continuous engagement):

      theta = (q alpha / k) int_0^t (pi alpha tau)^-1/2 exp(-z^2/4 alpha tau)
              1/2 [erf((x + V tau + l)/2 sqrt(alpha tau)) - erf((x + V tau - l)/2 sqrt(alpha tau))]
              1/2 [erf((w - y)/2 sqrt(alpha tau)) + erf((w + y)/2 sqrt(alpha tau))] dtau

  (Jaeger 1942; integrated in pieces split at the footprint passage, the diffusion cut-off and a
  geometric ladder of l^2/alpha, Gauss-Legendre with tau = s^2 on the first piece). The surface
  maximum is sought over six positions of the band (trailing edge to centre).
* `<theta_feed>_cell`: at finite-volume steps (when that heat has just been put into the cell) the
  energy the source deposited in the contact's cell during the window divided by the cell's heat
  capacity (path and footprint shares of the cell; capped at theta_feed); between steps the
  pending heat is not yet in T_FV and nothing is subtracted.
* `theta_flash`: one edge passage, the per-tooth flux `work_J / (L_s b t_cut)` on a band of length
  `L_s = VB + r_edge + h / tan(phi)` (flank contact plus shear-plane footprint; phi from the
  coefficient tables or 30 deg), width b (engaged edge length), moving at the cutting speed for
  the passage duration; its energy is part of theta_feed, so its cell mean is not subtracted.
* Preheat (material entering the cut) = `T_FV` one feed half length plus one cell ahead + the
  feed source's rise at its leading edge.

Validity: half-space approximation of the part near the contact (error grows when sqrt(alpha t)
of the window approaches the distance to other free surfaces: < 1 mm for steel at 50 ms, 1.9 mm
for aluminium); flat generated surface; the work-bound heat treated as one band source (the
shear-plane heat source of Komanduri & Hou 2000 is not resolved: its effect on the surface is
within the flash band). The two-scale peak is an estimate within about +-30 % (partition and
contact size dominate), adequate for threshold findings with that margin.

Published values (`test_jaeger.py`): high-Peclet band mean `0.754 q l/(k sqrt L)` and maximum
`1.128 q l/(k sqrt L)` (Jaeger 1942) within 3 % at L = 20 and 100; stationary square centre
`1.1222 q a/k` (Carslaw & Jaeger) within 0.5 %; 1-D short-time limit to 1e-6; kernel against
adaptive quadrature within 2e-4.

### 3.4 Thermal damage

The two-scale peak at each contact is compared with the material database `thermal_damage`:
`burn_degC` (temper/oxide colours, warning), `overtemper_degC` where given (soft layer of hardened
steels, warning), `white_layer_degC` (Ac1 rehardening, error), `melt_smear_degC` (error). One
`thermal_damage` finding per (operation, workpiece, kind, program block) at the tick of the step,
position from the tick and the contact point in S in the quantities; a trace event
`thermal:surface_temperature` is its parent; repeats are counted (`finalize`). Time at
temperature (tempering kinetics) is not modelled: a threshold crossing is reported.

## 4. Machine

First-order bodies integrated exactly over each step (excess temperature over 20 C as state, so
micro-kelvin changes keep their precision; `expm1` for the decay):

    theta(t+dt) = theta_eq + (theta - theta_eq) exp(-dt/tau),  loss = G[(theta_eq - theta_env) dt + (theta - theta_eq) tau (1 - e^-dt/tau)]

* **Spindle**: bearing heat `P = P_max (|omega|/omega_max)^(5/3)` (Palmgren viscous moment, time
  mean over the ticks); time constant and growth at max speed from the machine file
  (`growth_um_at_max_rpm` or `growth_at_max_speed_um`, `time_constant_min`,
  `bearing_heat_W_at_max_rpm` / `bearing_heat_at_max_speed_W`), else the estimates of
  `thermal-machine.yaml` (listed in `finalize()["estimated"]`). With `L_eff` (spindle growth length,
  est. 0.25 m) and the spindle steel CTE: `dT_max = growth_max / (alpha L_eff)`, `G = P_max / dT_max`,
  `C = tau G`. Growth `delta = alpha L_eff (T - T_ambient)`; published growth since the start of the
  run (when tool lengths were set) times `comp_residual_fraction` when the control compensates.
  Direction in M: milling/live spindles `-axis_T_M` (nose toward the work), turning `+z_M`, sub
  spindles `-z_M`. The holder's taper conduction enters the spindle body.
* **Axes** (linear, ballscrew): heat `heat_per_m_travel_J x travel` (from the actual axis values
  per tick), C from the screw steel volume, G = C / tau. Error at position s:
  `alpha (T - T0)(s - s_fixed)` along the axis' kinematic direction (zero with `scale_feedback`,
  times the residual with `electronic_comp`; `fixed_end: both` pre-tensioned: no growth). Published
  as `axis_drift[axis]` (6-vector at the end-of-step position) and as strain + fixed-end position
  for per-tick use. T0 resets at each operation start (new work offsets).
* **Structure**: relaxes to ambient with `structure_time_constant` (4 h est.); drift =
  `structure_sensitivity_um_per_K x (T - T0)` along `+axis_T_M` (warmer column lifts the head);
  T0 = start of the operation.
* **Start state** (`environment.machine_state`): `cold` = ambient; `warm` = after the warm-up time
  `warmup.cold_to_warm_min` at `warmup_speed_fraction` of max speed (axes at
  `warmup_axis_speed_m_s`); a mapping gives spindle/axes/structure temperatures directly.
* **Between operations** (`end_operation`): pending heat deposited, unload temperatures recorded
  (`unload`, `work_unload_mean_K`), the part unclamped and cooled in air for
  `time_between_ops_min` (`cooldown_steps` implicit steps), idle tools, stopped spindles, axes and
  structure advanced the same time; then clamped again for the next setup. `finalize` reports the
  lumped time for the part to come within `unload_settle_band_dK` of ambient.

Error: spindle growth of a first-order model vs a real spindle: +-30 % of the growth (two time
constants in reality: shaft and housing); axis drift: first-order lumped screw, the temperature
profile along the screw (hotter in the travelled range) is not resolved (+-50 %); structure:
sensitivity is a machine-file estimate.

## 5. Ledgers (architecture 12)

Solvers: `tool.<slot>`, `work.<w>`, `fixture.<w>`, `spindle.<id>`, `axis.<name>`, `structure`.
Each step: `dU = sources - losses - removed`, U from the stored enthalpies / excess temperatures,
the losses from the face/link conductances the scheme used; a relative residual above
`physics.thermal.energy_tol` (1e-9, relative to |U0| + |U1| + |terms|) raises ConservationError.
Measured residuals: 1e-16 .. 2e-12. Core ledger terms booked by thermal: `heat_to_chip_J`,
`heat_to_tool_J`, `heat_to_work_J` (partition), `heat_to_coolant_J` (tools' coolant and channels,
wetted workpiece faces), `removed_enthalpy_J` (net of re-added material, plus unplaced heat).
Work heat between finite-volume steps is published as `pending_heat_J`.

## 6. Bus

Read (`bus_prev`): `cutting.contacts[*]` (zone heat, per-element integrals `in_cut_time_s`,
`entries`, `int_h_dt`, `int_sigma_v_*`, `*_sliding_m`, `work_heat_S`, `in_cut_s`),
`cutting.coefficients[slot]` (fractions, chip ratio, shear angle, h grid, version),
`removal.contacts[*]` (workpiece, path, geometry: point_S_m, normal_M, ap_m, engaged_angle_rad),
`coolant` (on, fluid_temp_K, h_flank/h_work per contact, h_work_far, reach_fraction, film_boiling
(into the finding's quantities), wetted grid, through_tool_flow_m3_s), `chips` (heat_to_part,
heat_to_fixture_J, and `heat_to_machine_J` when chips publishes it: into the structure body),
`motion.paths[*]` (tool_slot, spindle_id, workpiece, spindle_omega_rad_s,
axis_actual/axis_names, feed_vel_M_m_s, axis_T_M, pose_M_T, pose_S_T, tick, block_index),
`wear.tools[slot].VB_m`.

Written: `nxt.thermal` (a fresh ThermalFeedback each step): `tools[slot]` (ToolHeat: edge/flank
temperatures at q points, q_weights, cycle amplitude, rake mean, body and holder temperature,
radial/length/holder growth; **added** `edge_peak_q_K`, `body_temp_at_elem_K`), `spindle_temp_K`,
`spindle_growth_M_m`, `axis_drift`, `work_temp_origin_S_m`, `work_temp_cell_m`, `work_temp_K` (a
new array at every finite-volume step, NaN outside material, never modified afterwards),
`work_surface_temp_K`, `work_preheat_K` (max over the workpiece's contacts), `work_bulk_mean_K`,
`work_bulk_max_K`, `heat_to_coolant_J`, `absorbed_J` (per solver dU of the step). **Added** fields:
`heat_to_chip_J`, `heat_to_tool_J`, `heat_to_work_J` (per contact), `unplaced_heat_J`,
`pending_heat_J`, `contact_surface_temp_K`, `contact_preheat_K`, `spindle_growth_abs_M_m`,
`axis_drift_strain`, `axis_drift_ref_m`, `structure_drift_M_m`, `fixture_temp_K`,
`work_unload_mean_K`, `ledger_residual_rel`.

## 7. Settings (`physics.thermal`, `schema.py`)

`every_n_macro` 5, `cell_mm` 2, `tool_cell_mm` 1, `edge_quadrature` 3, `edge_every_n_macro` 2,
`sa_direct_cycles` 4, `window_points` 3, `jaeger_points` 24, `min_fraction` 1e-9,
`deposit_search_cells` 3, `cooldown_steps` 24, `idle_tool_steps` 8, `energy_tol` 1e-9.

## 8. Cost (measured, one core of the i5-4690, bin/tsafe)

| part | cost |
|---|---|
| workpiece finite-volume step | 150-225 ns per material cell (5.5 ms for 26k cells; 14 ms for 63k; 74 ms for 490k) |
| edge kernel | 9 us per in-cut element and evaluation (every `edge_every_n_macro` steps) |
| domain macro step, one cutting tool (40 in-cut elements), small part | 1.9 ms (Python plumbing dominates) |
| response tables | 0.6 s once per process |
| kernel compilation (cold cache) | ~25 s once per cache key |

Per simulated second at 10 ms macro steps: tool, machine and contact work ~0.19 s, plus the
finite volumes `cells x 190 ns / (every_n_macro x 10 ms)`. For the reference part (300 x 200 x
100 mm, ~490k material cells at 2 mm) choose `cell_mm: 4` (63k cells: 0.28 s/s at every 5) or
`every_n_macro: 20` (0.37 s/s at 2 mm): the two-scale contact temperature covers the sub-cell
and sub-step scales, so a coarser field changes the bulk only.

## 9. References

Jaeger 1942 (moving sources); Carslaw & Jaeger 1959 (conduction, disc and rectangle sources);
Stephenson & Ali 1992 (interrupted cutting); Blok 1937 (flash temperature, partition);
Boothroyd 1963; Trent & Wright 2000; Astakhov 2006 (Poletika contact length); Komanduri & Hou
2000; Douglas & Gunn 1964 (ADI, rejected); Churchill & Chu 1975a,b; Incropera et al. 2007;
Kays & Bjorklund 1958; Dittus & Boelter 1930; White 2006 (Sutherland); Palmgren 1959,
Harris & Kotzalas 2007 (bearing friction); Madhusudana 2014 (contact conductance); Bryan 1990,
Mayr et al. 2012 (machine thermal errors); ISO 7388-1, ISO 12164-1 (tapers); Abramowitz & Stegun
1964 (Gauss-Legendre). Full entries: `db/bibliography/models-thermal.yaml`.
