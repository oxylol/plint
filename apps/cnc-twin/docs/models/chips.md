# Chips after the cut (`cnctwin.chips`)

What happens to the chips once they leave the edge: flight through air, coolant jets and the wall
film, contacts with the rotating tool, the in-process workpiece, fixture, table, enclosure and the
chip pile, cooling, settling into a height field, avalanches, flushing, conveying, chips in the
flutes and the packing loads they cause, the frozen occupancy grid the hot loop recuts against, and
the findings that follow (recutting, packing, welding, bird-nesting, wrapping, chips nested in a
pocket, debris on locating faces, blocked probe or nozzle, dust, fire risk, pan overflow).

Package: `abi.py` (layouts), `kernels.py` (Numba, explicit signatures), `reference.py` (plain-Python
twins), `state.py`, `schema.py` (`physics.chips`), `channels.py`, `params.py`, `protocols.py`,
`formation.py`, `scene.py`, `emission.py`, `flutes.py`, `hazards.py`, `domain.py` (`ChipsDomain`).
Coefficients: `db/models/chips-{aero,heat,contact,bulk,flute,flush,hazards}.yaml`; sources:
`db/bibliography/models-chips.yaml`.

## 1. Timing, data flow and frames

* **Hot loop -> chips.** `ChipsDomain.kernel_inputs()` hands the compiled loop
  `abi.ChipsKernelInputs(emission, occupancy, packing)`:
  * `emission`: an `EventBuffer` of rows `abi.EMISSION_COLUMNS` (tick, contact, tool slot, tooth,
    rho x analytic chip volume of the tooth pass, separation point and chip velocity in M, chip
    temperature, mean h, chip width b, uncut length l, cutting speed, rake, inclination, dominant
    edge kind and coefficient group). The hot loop writes one row per tooth pass with
    `kernels.write_emission` (explicit signature, sets `n[1]` on overflow; `absorb` then doubles
    the capacity for the next step and keeps the rows that fit - the mass rule below still books
    the exact field mass).
  * `occupancy`: `abi.Occupancy` - the frozen chip-occupancy grid in S (origin, cell, mass, size,
    thickness per voxel), aligned with the pile cells, covering what the tools can sweep before
    the next grid is published (tool envelope + feed x lag). The hot loop applies recut impulses
    where the swept envelope meets it and reports `CuttingContact.recut` rows.
  * `packing`: `abi.PackingLoads` - per tool slot the additive packing torque (opposing the
    spindle rotation: apply `-sign(omega) torque` about +z_T) and force in T0 (+z_T).
* **One macro step behind.** `absorb()` (step n) stores the rows. `macro_step(prev, nxt)` of step n
  reads only `bus_prev` (records of step n-1) and integrates the chips over the window of that
  motion record: it releases the rows with `tick < window end`, scaled by the same window's
  `RemovalContact.removed_volume_m3`, processes the same window's recut rows, and writes
  `bus_next.chips`. Hence the chip ledger terms written at step n cover the removed volume up to
  the window of `bus_prev` (`ChipFeedback.booked_until_tick`), and the occupancy grid the hot loop
  of step n+1 uses describes the chips at the end of window n-1 (20 ms old for settled chips; fast
  particles are left out of it, see 6).
* **Frames.** Particles, tools, nozzles and jets are in M. The pile grid G is fixed to S with axes
  `axes_S` (columns e_u, e_v, e_w), a signed permutation of the S axes with e_w the S axis closest
  to "up" (-g mapped into S with the window's rotation). The S origin in M is taken per L1 tick from
  `tip_M - R_M_S tip_S` (the table moves under the chips); `R_M_S` is the end-of-step rotation
  (exact for 3-axis and positional 5-axis; continuous rotary motion errs by the rotation in one
  step times the lever arm). Gravity: `machine.kinematics.gravity_direction`, else -z for mills and
  -x for lathes (turret above the spindle - slant beds should state their direction).

### Bus fields read (all from `bus_prev`)

| record | fields |
|---|---|
| motion.paths[] | tick0, ipo_ticks, tick, tip_M_m, tip_S_m, axis_T_M, spindle_omega_rad_s, feed_vel_M_m_s, tool_slot, spindle_id, workpiece, pose_M_T, pose_S_T, events (`kind` probe / probe_trigger / probe_hit / tool_setter with `tick`, optional `pos_S_m`) |
| removal | contacts[].contact, tool_slot, removed_volume_m3, geometry.in_cut, geometry.point_M_m, geometry.ae_m; events[] with `kind: burr`, `point_S_m`, `height_m` (end of operation) |
| cutting | contacts[].contact, tool_slot, recut (`[tick, tooth, elem, mass_kg, x_M, y_M]`); coefficients[slot].h_grid_m, shear_angle_rad, chip_ratio, T_shear_K |
| coolant | on, kind, jets[] (nozzle_id, kind, origin_M_m, direction_M, speed_m_s, flow_m3_s, half_angle_rad), fluid_temp_K, wetted (+ wetted_origin_S_m, wetted_cell_m), flush_pressure_Pa, through_tool_flow_m3_s |
| thermal | work_temp_K[wp], work_temp_origin_S_m[wp], work_temp_cell_m[wp] |

`wetted` and `flush_pressure_Pa` are used as they are when they have the pile grid's shape
(coolant should publish them on `pile_origin_S_m`, `pile_cell_m`, `pile_axes_S` of the previous
chips record); otherwise they are sampled at the cell centres from their own origin and cell
(along the two in-plane S axes). `flush_pressure_Pa` is read as the dynamic pressure q = rho u^2/2
of the coolant flow over the cell.

### Bus record written: `bus_next.chips` (`ChipFeedback`)

Contract fields: `formation` (per released contact: shear angle, chip ratio, strain, strain rate,
flow angle, chip temperature, form code, BUE, `EmissionBatch` of the parcels), `flute_fill`,
`packing_torque_Nm`, `packing_force_T0_N`, occupancy grid (`occupancy_origin_S_m`,
`occupancy_cell_m`, `occupancy_mass_kg`, `occupancy_size_m`), pile (`pile_origin_S_m` = corner of
cell (0, 0) at w = 0 in S, `pile_cell_m`, `pile_height_m` along e_w), `blocked_nozzles`,
`probe_blocked`, `wrap_mass_kg`, `dust_mass_rate_kg_s`, `heat_to_part` (`[x_S, y_S, z_S, J]`, base
point of each part cell), `heat_to_fixture_J`, `heat_to_coolant_J`, `locator_debris_m`.

Additive fields (added by chips, commented in core/bus.py): `booked_until_tick`,
`occupancy_thickness_m`, `pile_axes_S`, `pile_base_m`, `pile_mass_kg`, `packing_power_W`,
`flute_mass_kg`, `nest_fraction`, `part_chip_mass_kg`, `recut_mass_kg`, `heat_to_machine_J`,
`heat_to_air_J`, `kinetic_energy_J`, `in_flight_kg`, `settled_kg`, `conveyed_kg`, `adhered_kg`,
`in_flutes_kg`, `dust_kg`, `emitted_kg`, `conveyor_backlog_kg`.

Ledger (architecture 12), written every step: `chips_in_flight_kg` (flying particles),
`chips_settled_kg` (pile + chip pan), `chips_conveyed_kg`, `chips_adhered_kg` (welded, wrapped,
on nozzles), `chips_in_flutes_kg`, `dust_kg`. Their sum equals the booked mass `emitted_kg`
exactly (tested to 1e-9 over 1000 random steps; observed ~1e-16).

## 2. Emission

**Mass rule (architecture 5.4).** For each contact of the window the rows carry
`m_i = rho V_analytic,i`. The parcels are scaled so their total is `M = rho V_field`
(`removed_volume_m3` of the same window): `m_group = M sum(m_i in group) / sum(m_i)`. A contact
with field volume but no row (e.g. a rapid into material) gets one row at its contact point with
the tool's surface velocity there and the slot's last chip (elemental when none); rows without a
removal record are booked with their analytic mass and reported once (finding *sanity*), because
the balance against the removed volume then cannot close.

**Representative particles.** Rows of a contact (sorted by tick, tooth) are grouped by
`K = round(1 / particles_per_tooth)`; a group becomes one parcel carrying its mass with the
position, velocity, temperature and shape of one row drawn as
`j = floor(u (hi - lo))`, `u = hash_uniform(seed, "chips.select", group, contact, 0)` (unbiased
in expectation). The departure direction is turned by two normal angles
`sigma_a n_1, sigma_a n_2` about two axes normal to v and the speed scaled by `1 + sigma_s n_3`
(`hash_normal(seed, "chips.scatter", group, contact, k)`).

**Chip shape.** The formation model (Protocol `FormationModel`; default `MechanicsFormation`:
Oxley's orthogonal solution with the Johnson-Cook flow stress and `mechanics.chip_form` for curl,
breaking, ISO 3685 class, segmentation, BUE and colour, solved once per 2 % bin of h, v, b, l and
0.01 rad bin of rake and inclination) gives per contact the deformed chip. Per row: thickness
`t = t_F h_row / h_mean`, width `b_F`, length of one separated chip `L_F` (if the chip is not broken
into pieces: `l_row r_c`), curl radius, form code, colour. No chip below the minimum chip
thickness: the field's removed volume leaves as elemental fragments of the uncut thickness.

**Shares.** Dust `m f_d` (composite `dust_mass_fraction`, `powder_dust_fraction` for powder-forming
materials) leaves to extraction; the confined share `(m - m_d) c_f` of the slot (7) enters its
flutes; the rest flies.

Per chip: `V = t w L`, `S = 2(tw + tL + wL)`, `d_v = (6V/pi)^(1/3)`, sphericity
`psi = pi^(1/3) (6V)^(2/3) / S`, drag reference area `pi d_v^2 / 4`.

## 3. Flight (kernel K1, `advance_particles`)

Per particle, sub-steps until the window end:

* **Drag** (Haider & Levenspiel 1989), Re on `d_v`:
  `C_D = 24/Re (1 + A Re^B) + C/(1 + D/Re)`, `A = exp(2.3288 - 6.4581 psi + 2.4486 psi^2)`,
  `B = 0.0964 + 0.5565 psi`, `C = exp(4.905 - 13.8944 psi + 18.4222 psi^2 - 10.2599 psi^3)`,
  `D = exp(1.4681 + 12.2584 psi - 20.7322 psi^2 + 15.8855 psi^3)`; `F = 1/2 rho_f C_D (pi d_v^2/4) |u - v|(u - v)`,
  written as `m k (u - v)` with `k = rho_f A_ref (C_D |v_r|) / (2 m)` (finite as Re -> 0).
* **Fluid at the point**: the jet with the largest dynamic pressure there, else the wall film over
  a wet cell (within `max(d_v, h_film)` of the pile/part top: film velocity, liquid properties,
  force scaled by the immersed fraction `min(1, h_film / d_v)`), else still air. Liquid jets keep the
  exit velocity inside the spray cone `r <= r0 + s tan(theta)`, their momentum flux scaled by the
  liquid fraction `(r0 / r_j)^2`; gas jets (air, mist, MQL) are self-similar round jets
  `u_c = u_0 min(1, B d_0 / s)`, `u = u_c exp(-ln2 (r / (S s))^2)` (Pope 2000, B = 5.8, S = 0.094).
  Exit radius `r0 = sqrt(Q / (pi u_0))`. Buoyancy of the liquid share: `g (1 - alpha rho_l / rho_p)`.
* **Integrator**: exact solution of `dv/dt = g - k (v - u)` with k, u frozen over dt:
  `v1 = v + (u - v) z phi1 + g dt phi1`, `x1 = x + u dt + (v - u) dt phi1 + g dt^2 phi2`,
  `z = k dt`, `phi1 = (1 - e^-z)/z`, `phi2 = (z - 1 + e^-z)/z^2` (series below z = 1e-3). Exact
  for k = 0 (ballistic) and at the terminal state; `dt <= min(dt_max, drag_dt_factor / k)`.
* **Contacts** along the chord of the sub-step, earliest first: enclosure box (floor -> chip pan;
  walls -> pan when flood coolant runs, else reflection), the height field in G (column tops and
  side walls, exact DDA through the cells, moving with the table), the tool and holder cylinders
  and the turning workpiece (finite cylinders moving per L1 tick), nozzle spheres. The particle
  moves to the contact with the exact propagator, the response is applied, and it continues.
* **Impact** (Brach 1991): relative to the surface velocity `s`, `v_n' = -e v_n`, tangential
  Coulomb impulse `min(mu (1 + e)|v_n|, |v_t|)`. Restitution and friction by material pair:
  `e = e_family f_e(surface)`, `mu = mu_family f_mu(surface)` (chips-contact).
* **Flute face** (paddle) of a rotating tool, surface velocity `omega a x r`: radial reflection with
  e, tangential restitution `paddle_e` along the flute motion when the flute catches the chip,
  Coulomb friction on the axial component. A chip at rest struck leaves at `(1 + e_p) omega R`.
* **Settling** on a face with `g_n > 0` when the rebound apex would be below `settle_bounce_sizes`
  chip sizes: `e v_in < sqrt(2 g_n s d_v)`; on wetted faces also when `e v_in < v_esc`,
  `v_esc = sqrt(2 F_cap h_film / m)`, `F_cap = gamma cos(theta) 2 (w + L)` (Wilhelmy 1863), or when
  the viscous Stokes number `St = 4 rho_p v_in d_v / (9 mu_l)` is below
  `(1 + 1/e) ln(h_film / h_a)` (Ennis et al. 1991). A settled particle deposits into its cell.
* **Welding**: a chip hotter than `T_weld = 0.6 T_solidus` (Kazakov 1985: diffusion bonding at
  0.5-0.8 T_m) striking the cutting part of a tool sticks to it (`ON_TOOL`), except at its first
  contact with its own tool (the separation itself). It is recut (released as a fragment, counted
  as recut mass) at the slot's next engagement.
* **Wrapping**: a long form (`wrap_forms`) at least `wrap_turns` circumferences long striking a
  rotating tool or turning part wraps (`WRAPPED`) until that body stops or the tool leaves the
  spindle. **Nozzles**: chips stick on an idle nozzle (capillary escape as above) and hook on a
  running one when longer than `hook_radii` nozzle radii; short ones are blown off when it runs.
* **Cooling** (lumped, Bi = h t / k < 0.1 for chips): `h_c = k_f / d_v (2 + 0.6 Re^0.5 Pr^(1/3))`
  (Ranz & Marshall 1952; liquid share alpha, air the rest), radiation
  `h_r = eps sigma (T^2 + T_a^2)(T + T_a)`; `T1 = T_inf + (T - T_inf) exp(-h_tot S dt / (m_chip cp))`.
  The heat is the exact enthalpy difference of the piecewise-linear cp table, split by the fluxes
  into coolant (liquid convection) and air (gas convection, radiation).
* Guards: after `max_bounces` contacts or `max_zero_hits` contacts without time advance a particle
  is laid into the cell below (event `chips:laid_down`).

## 4. Pile (height field over the table/part region)

* Grid G over the stock footprint plus `pile_margin_mm`, cell `pile_cell_mm`. Base `w(u, v)` =
  highest surface along e_w from the `ChipGeometry` provider (in-process workpiece, fixture, table;
  `-inf` where nothing is below), refreshed in the changed box when the geometry version changes.
  Cells that lose their surface (breakthrough) drop their chips into the pan. A new up axis
  (rotary axes) or another workpiece: the chips move again at rest and a new grid is built. A
  turning workpiece (|omega| > `rotating_part_rad_s`) holds no pile: chips on part cells leave with
  its surface speed.
* **Deposit**: bulk volume `m R(form) / rho` (Spanraumzahl R: ribbon/snarled 100, helical 60, short
  helical 30, spiral 10, broken chips 3 - chemie-de-span; cross-check with screw-conveyor bulk
  densities: dry aluminium chips R 11-24, cast-iron chips 2.2-3.5, crushed steel turnings 3.3-4.9),
  height = bulk / cell^2. Extensive fields (bulk, mass, enthalpy, mass-weighted size, thickness,
  width, length, tan(repose), long-form and fines mass) add, so mixing and moving are exact.
* **Avalanches** (K2): Gauss-Seidel in fixed order, `dV = (dh - tan(phi_r) cell) cell^2 / 2` to a
  lower neighbour; over an edge with nothing below the pile keeps at most `tan(phi_r) cell / 2`
  and spills the rest into the pan. Pockets and holes fill this way.
* **Tools moving through the pile** (K7): at every L1 sample the overlap of each pile column with
  the tool cylinders is thrown out with the velocity a chip at rest gets from that surface.
* **Recut rows** (K8) of the window: the requested mass is taken from the cell of (x_M, y_M) at the
  tool tip's height, then its 8 neighbours; it comes back as fragments (length <= width, arc or
  elemental) with `(1 + e_p) omega a x r + v_feed`; mass already gone is counted as
  `recut_missing_kg`.
* **Flushing** (K4): Shields number `theta = C_f q / ((rho_s - rho_f) g d)`, d the cell's mass-weighted
  chip size; above `theta_c` the cell loses `rho_s q_s cell dt`,
  `q_s = 3.97 (theta - theta_c)^1.5 sqrt((rho_s/rho_f - 1) g d^3)` (Wong & Parker 2006,
  theta_c = 0.0495). Beds wetted but not submerged (MQL, mist) add the capillary adhesion
  `mu_bed 8 gamma cos(theta) / d` to the critical stress (Israelachvili 2011). Flushed chips leave
  with the film velocity `sqrt(2 q / rho_l)` down the flush-pressure gradient.
* **Heat of resting chips** (K5): bed-to-surface conductance `h_bed` to the surface below (part
  temperature from the thermal grid, else ambient), top to the coolant film (`h_top_wet`) or air
  (`h_top_air` + radiation); exact exponential per step, exact enthalpy. Part cells give
  `heat_to_part` rows, fixture cells `heat_to_fixture_J`, table cells `heat_to_machine_J`.
* **Chip pan and conveyor**: lumped mass/enthalpy/bulk; cooled through `bulk / pan_layer_m` toward
  the coolant (flood running) or air; removed at `machine.chip_conveyor.rate_kg_h` (none: 0).
* **End of operation**: the last window is booked, attached chips and flying chips are laid down,
  flutes empty into the pan, the debris on every patch the next setup clamps or locates on is
  measured (`LocatorFaces`; default: the six stock faces, only up-facing faces hold settled chips;
  burr events of removal on the patch count too), then part and fixture are cleaned into the pan.
  The operator cleans fixture and locating faces before loading (`production.operator.clean_s` > 0, the
  template's 15 s: blow chips off fixture and locators, cnctwin.production.operator): the settled chips on
  those faces then leave no debris and only burrs (on the part, not blown off) count; with `clean_s: 0` the chips
  stay and mislocate the next setup (2026-11-09: before, the chips were always measured as left there, and with
  the burrs of a batch's previous part every next blank was mislocated and every batch unsafe).

## 5. Flute fill and packing (`flutes.py`)

* **Confinement**: fraction `e` of a ring at `R + ring_cells` cells around the tip whose base lies
  above the tip by more than a cell; confined share `c_f = clamp((e - e_0)/(1 - e_0), 0, 1)`,
  `e_0 = 0.75` (1 for a drilled hole; a full slot is open behind the tool, its chips fly and pile in
  the slot as particles). Evacuation length `L_e` = mean height of the confining ring above the tip.
  Spiral-point taps push chips ahead (not confined). Evaluated at each window end, applied to the
  next window's chips.
* **Conveying**: high-speed limit of a screw turning in a stationary casing (Roberts 1999): the chip's
  absolute velocity is normal to the flute face turned by the friction angle, so
  `v_s = |omega| R cos(delta)(sin(delta) - mu_f cos(delta))` (0 when `tan(delta) <= mu_f`, e.g.
  straight flutes), `v_eff = v_s f_form exp(-k L_e / D) + c_slip Q_tt / A_g`. The exponential
  depth attenuation reproduces the exponential growth of evacuation forces with depth observed in
  drilling (Mellinger et al. 2003); `f_form` by chip form.
* **Transport**: `dV/dt = Q_in - V/tau`, `tau = L_e / v_eff`, exact per step; fill `f = V / (A_g L_e)`,
  steady `f = Q_in / (A_g v_eff)` (tested against this hand calculation to 1e-3 through the
  domain and to 1e-13 in the unit). `L_e = 0` (retract, peck, tool change) empties the flutes;
  the leaving chips are released at the hole mouth (`tip + L_e a`, on the tool radius at a hashed
  azimuth `chips.flute`, velocity `omega a x r + v_eff a + v_feed`).
* **Packing** above the critical fill (Kawakita & Luedde 1971): `C = 1 - f_c / f`, `a = 1 - 1/R`,
  `sigma = P_half C / (a - C)`, `P_half = c sigma_y`, `sigma <= 3 sigma_y`; torque
  `T = mu_w sigma (pi D L_e) R`, axial force on the flute bottom `F = sigma A_g` (+z_T), friction
  power `T |omega| + F |v_ax|` (published; the hot loop integrates the work of the loads it
  applies). Gullet area: tool file `gullet_area_mm2`, else `gullet_area_fraction pi D^2 / 4`.
  Chips stay at their enthalpy in the flutes (adiabatic).

## 6. Occupancy grid (`fill_occupancy`)

Voxels of S aligned with the pile cells inside the union of the active tools' reach (tool and
holder envelope + `R + |v_feed| lag + cell`, lag = `slow_lag_steps` macro steps): every pile
column spread uniformly over its height, plus flying particles slower than `cell / lag` (fast ones
will be elsewhere when the hot loop reads the grid). Size = mass-weighted volume-equivalent chip
diameter, thickness = mass-weighted chip thickness. Empty when no chips are within reach.

## 7. Findings (thresholds: chips-hazards)

| condition | category, severity | cause chain |
|---|---|---|
| flute fill >= 0.7 / >= 1 | chip_packing warning / error (hole-making tools; warning otherwise) | `chips:flute_fill` <- `chips:formation` |
| recut mass / chip mass of the slot over 0.5 s >= 2 % | chip_recutting warning | `chips:recut` <- formation, `chips:recut_rows` |
| chips in the footprint of a tool entering the cut >= 0.2 g | chip_recutting warning | `chips:nested_at_entry` |
| welded mass on an edge >= 0.01 g | chip_hazard warning | `chips:weld` <- formation |
| long chips in the ring R..3R above the tip / ring volume >= 0.3 | chip_hazard warning (bird's nest) | `chips:nest` <- formation |
| wrapped mass >= 0.5 g / 20 g (tool or turning part) | chip_hazard warning / error | `chips:wrap` |
| chips >= 0.02 mm at a probing / tool-setter point | chip_hazard warning, `probe_blocked` | `chips:probe_debris` |
| stuck bulk on a nozzle >= pi r^3 | chip_hazard warning, `blocked_nozzles` | `chips:nozzle_blocked` |
| dust generated | chip_hazard warning | - |
| chip temperature >= chip ignition / hazard temperature (material `ignition`) | chip_hazard error | `chips:ignition_risk` |
| fines (thinner than the critical thickness) >= 1 g, dry, ignitable material | chip_hazard warning | - |
| pan bulk > floor area x 0.15 m | chip_hazard warning | - |
| chips/burrs > 0.01 mm on a face the next setup clamps or locates on | mislocation warning | `chips:locator_debris` |

A rule fires once when its condition starts and re-arms when it clears: packing below
`flute_fill_rearm`, recutting and nesting below half their threshold, the others below their
threshold; dust is reported once per run.

## 8. Parameters

| file | parameters (SI) | source |
|---|---|---|
| chips-aero | Haider-Levenspiel constants (24, A, B, C, D), min sphericity 0.026 | haider-levenspiel-1989 |
| | round jet B = 5.8, S = 0.094 | pope-2000 |
| | spray half angle 5 deg, scatter 10 deg / 15 % | est |
| chips-heat | Ranz-Marshall 2, 0.6, 1/2, 1/3 | ranz-marshall-1952 |
| | emissivity by family, h_bed 200, h_top_air 10, h_top_wet 2000 W/m2K, pan layer 30 mm | est (h_bed from the Schlunder model) |
| chips-contact | restitution / friction by family and surface factor, paddle e 0.3, settle 1 size, film 50 um, asperity 2 um, cos theta 0.9, wrap 0.5 turns, hook 4 radii, nozzle radius 1.5 mm | est |
| | Wilhelmy factor 1 | wilhelmy-1863 |
| | weld at 0.6 T_solidus (range 0.5-0.8) | kazakov-1985 |
| chips-bulk | R by form | chemie-de-span (+ est for 61, 80, 90, 95) |
| | repose angles by form, long/wrap forms, fines 0.05 mm, powder dust 5 % | est |
| chips-flute | flute mu 0.3, wall mu 0.5, depth attenuation 0.1 / D, coolant slip 0.5, confinement 0.75, critical fill 1, Kawakita 1/b = 0.003 sigma_y, cap 3 sigma_y, gullet fraction 0.25, conveying by form | est (forms of roberts-1999-screw, kawakita-ludde-1971, mellinger-2003) |
| chips-flush | C_f 0.006 | est (schlichting-gersten-2017) |
| | theta_c 0.0495, 3.97, 1.5 | wong-parker-2006 |
| | capillary bridge 8 | israelachvili-2011 |
| chips-hazards | thresholds of section 7 | est |

Numerical settings (`physics.chips`, `schema.py`): `pile_cell_mm` 2, `pile_margin_mm` 50,
`particles_per_tooth` (default `physics.particles_per_tooth`), `max_particles` 50 000, `dt_max_ms`
1, `drag_dt_factor` 0.2, `contact_offset_um` 1, `max_bounces` 400, `max_zero_hits` 8,
`relax_max_iter` 50, `relax_tol_fraction` 1e-3, buffer capacities, `rotating_part_rad_s` 1,
`slow_lag_steps` 2, `ring_cells` 2, `enclosure_margin_mm` 1000, `formation_bin_rel` 0.02,
`formation_bin_rad` 0.01. `every_n_macro` must be 1 (the motion record of skipped steps is not kept).

## 9. Assumptions and their consequences

* Representative parcels: chips of a group share one trajectory; with `particles_per_tooth = 0.1`
  the spatial distribution of a pile is resolved to about ten tooth passes. Mass is exact.
* Chips are rigid strips (curled forms keep the strip's volume and surface; no tumbling dynamics,
  orientation-averaged drag). Flat chips (psi < 0.3) fall with an uncertain drag (see 10).
* 2.5-D height field: no overhangs (undercuts, vise jaws overhanging the part are not
  represented); chips under an overhang land on the surface below it. Grid tilt up to 54.7 deg.
* The tool is a swept cylinder (cutting part + holder); turning tools, turrets and the chuck are
  not obstacles (chips pass through them).
* Coolant: liquid jets coherent within their cone, no breakup, no boiling regimes (single-phase
  Ranz-Marshall even above the boiling point; chips in coolant cool within ~10 ms anyway); the jet
  does not lose momentum to the chips.
* Air at rest outside jets (no spindle-induced air flow).
* Chips in the flutes are adiabatic and invisible to the occupancy grid.
* Recut location: (x_M, y_M) of the row at the tool tip's height; mass from that column and its
  neighbours.
* Pan chips: one lumped store; walls are washed down when flood coolant runs.
* Cleaning between operations: all resting chips go to the pan after the locator debris is
  measured; the operator's cleaning before loading removes the chips on the locating faces (above).

## 10. Validity range and error bounds

| part | validity | error and how it was established |
|---|---|---|
| ballistic flight | any | exact (test: 1e-12 against x0 + v0 t + g t^2/2 over 0.2 s) |
| drag | Re < 2.6e5, psi >= 0.026 | correlation: ~5 % for spheres, ~20 % compact chips, up to a factor 2 for flakes and ribbons (scatter of non-spherical data reported for the correlation); terminal velocity reproduced to 1e-6 (test) |
| frozen-drag integrator | dt k <= drag_dt_factor | transient velocity error ~ (k dt)^2 / 2 = 2 % per sub-step at 0.2, exact at the terminal state and for k = 0 |
| jets | coherent liquid jets, round gas jets beyond ~6 d0 | entrainment to the jet speed tested; jet spreading and breakup not modelled: chip landing points in a jet +-30 % of the travel |
| cooling | Bi < 0.1 (t < ~1 mm), Re < 1e4 | Ranz-Marshall with d_v: factor ~2 for flakes against flat-plate correlations; radiation +-50 % (emissivity); bookkeeping exact (exponential solution tested to 1e-10; enthalpy closure 1e-9 over 1000 steps) |
| contacts | 1-30 m/s impacts | e, mu estimates +-50 %: chip landing positions +- a few cm, pile spread +-30 % |
| pile | chip sizes ~ cell size | heights: R within its tabulated range, factor ~2 (aluminium 11-24); slopes to +-10 deg; mass exact |
| flutes | drilling, tapping, reaming, closed plunges | fill: kinematic formula, mu_f +-50 % gives v_s -50/+100 % for delta = 30 deg, so fill within a factor ~2; packing torque order of magnitude (Kawakita 1/b est); peck reset exact |
| flushing | non-cohesive beds, theta < ~1 | threshold factor 2-3 (C_f, theta_c for flat chips), rate factor 2-5 (typical of bed-load formulas); mass exact (tested against the hand calculation to 1e-12) |
| recut | grid lag 1-2 macro steps | mass exact (round trip tested); location within one cell |
| findings | thresholds of practice | judgement values (est) |

## 11. RNG streams

`chips.select` (representative row of a group; keys group, contact, 0), `chips.scatter` (departure
direction and speed; keys group, contact, draw), `chips.flute` (azimuth of chips leaving the
flutes; keys particle id, slot, 0). All through `core.rng.hash_uniform` / `hash_normal` with
counters in `ChipsState` (`group_counter`, `next_pid`); no stateful generator.

## 12. Cost per macro step (measured)

On this box (single core, `bin/tsafe`, kernels cached), reference pocket (stock 300 x 200 x 100 mm,
2 mm cells -> 200 x 150 grid, D10 4-flute at 10 000 rpm, 7 emission rows per step):

| case | macro_step | of it kernels |
|---|---|---|
| no chips (idle) | 1.3 ms | 0.2 ms |
| particles_per_tooth 0.1, dry | 2.6-2.9 ms | 0.7 ms (K1 0.5 ms) |
| particles_per_tooth 0.1, flood coolant with flush field | 2.7-3.1 ms | 0.6 ms (K1 0.2, film 0.13 ms) |
| particles_per_tooth 1.0, flood | 4.0-4.3 ms | 1.5 ms (K1 1.1 ms) |

That is 0.26-0.43 s per simulated second, dominated by the Python orchestration of grid-sized
arrays (record snapshots, heat rows, occupancy, frames). `absorb` 0.01 ms, `kernel_inputs`
0.05 ms. A 4 mm cell saves only ~10 %. Cold start compiles the kernels in ~27 s (cached on disk).

## 13. What the integration has to wire

* `ChipsDomain(formation=..., geometry=..., tools=..., locators=...)`: `geometry` implements
  `protocols.ChipGeometry` over removal's dexel field (surface height along a signed S axis at
  given points, with part/fixture/table kinds; `version`, `changed_box_S`, `stock_bounds_S`,
  `rotation_radius_S`); without it the analytic stock box (`BoxScene`) is used and recorded as a
  fallback. `MeshSurfaces(part_provider, [(mesh_in_S, abi.SURF_FIXTURE | SURF_TABLE), ...])` adds
  fixture, vise, pallet and table meshes through `cnctwin.geom.bvh` (one downward ray per cell).
  `tools` defaults to the project tool list (`ProjectTools`), `locators` to the stock faces,
  `formation` to `MechanicsFormation` (needs a Johnson-Cook flow stress).
* The hot loop writes rows with `kernels.write_emission` into `ChipsKernelInputs.emission`, applies
  `packing` and recuts against `occupancy`, and accumulates the work of the packing loads.
* The core ledger must compare the chip terms with the removed mass of the windows up to
  `booked_until_tick` (chips book a window one macro step after removal does).
* `ChipsDomain.macro_values(state)` gives the macro-tier channel values; `keyframe_arrays(state)`
  the particles and pile summary for `chips/kf_<k>.npz`.

## References

* Haider, A.; Levenspiel, O. (1989) Drag coefficient and terminal velocity of spherical and
  nonspherical particles. Powder Technology 58, 63-70.
* Ranz, W.E.; Marshall, W.R. (1952) Evaporation from drops. Chem. Eng. Prog. 48, 141-146, 173-180.
* Ennis, B.J.; Tardos, G.; Pfeffer, R. (1991) A microlevel-based characterization of granulation
  phenomena. Powder Technology 65, 257-272.
* Wilhelmy, L. (1863) Annalen der Physik und Chemie 119, 177-217.
* Israelachvili, J.N. (2011) Intermolecular and Surface Forces, 3rd ed., Academic Press.
* Shields, A. (1936) Mitteilungen der Preussischen Versuchsanstalt fuer Wasserbau und Schiffbau 26.
* Wong, M.; Parker, G. (2006) J. Hydraul. Eng. 132, 1159-1168.
* Kawakita, K.; Luedde, K.-H. (1971) Some considerations on powder compression equations. Powder
  Technology 4, 61-68.
* Roberts, A.W. (1999) The influence of granular vortex motion on the volumetric performance of
  enclosed screw conveyors. Powder Technology 104, 56-67.
* Mellinger, J.C.; Ozdoganlar, O.B.; DeVor, R.E.; Kapoor, S.G. (2003) Modeling chip-evacuation
  forces in drilling for various flute geometries. J. Manuf. Sci. Eng. 125, 405-415.
* Brach, R.M. (1991) Mechanical Impact Dynamics. Wiley.
* Pope, S.B. (2000) Turbulent Flows. Cambridge University Press.
* Schlichting, H.; Gersten, K. (2017) Boundary-Layer Theory, 9th ed., Springer.
* Schluender, E.U. (1984) Chem. Eng. Process. 18, 31-53.
* Kazakov, N.F. (1985) Diffusion Bonding of Materials. Pergamon/Mir.
* Janssen, H.A. (1895) Z. VDI 39, 1045-1049; Jaky, J. (1944) J. Soc. Hung. Archit. Eng. 78, 355-358
  (background of the depth attenuation; no coefficient used).
* chemie.de, Span (Spanraumzahl table), read 2026-11-03; KWS Manufacturing, Screw Conveyor Bulk
  Material Table, read 2026-11-03; ISO 3685:1993 Annex G (chip forms).
