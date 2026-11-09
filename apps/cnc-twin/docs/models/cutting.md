# Cutting mechanics (`cnctwin.cutting`)

Forces, torques, heat and chip of every engaged edge element of every tool, per L0 step, from the
surface the previous tooth (or revolution, or pass) left; the coefficient tables they use, once per
macro step per tool; the spindle's ability to deliver the torque; and what the cut hands to the other
domains (wrenches, per-element integrals, heat by zone, chip emissions, surface events, work-heat
deposits, section loads, recut impulses, packing loads, axis loads, findings).

Package: `abi.py` (layouts and column constants), `kernels.py` (Numba, explicit signatures),
`reference.py` (plain-Python twin of the milling kernel), `coefficients.py` (L2 coefficient pipeline),
`geometry.py` (CutTool from tooling's EdgeModel, table rows, memory rows of taps), `memory.py` (stale
bins, field planes, analytic stocks), `spindle.py`, `analytic.py` (closed forms and ideal tools for
verification), `drive.py` (a stand-alone driver of the kernels over prescribed motion), `state.py`,
`schema.py` (`physics.cutting`), `channels.py`, `domain.py` (`CuttingDomain`). Coefficients:
`db/models/cutting-contact.yaml`, `db/models/cutting-spindle.yaml`; sources:
`db/bibliography/models-cutting.yaml` (plus the mechanics and tooling bibliographies for shared keys).

## 1. Timing, frames and conventions

* **L2 (once per macro step per tool slot).** `CuttingDomain.macro_step` reads `bus_prev`
  (thermal, wear, coolant, motion), snaps each table row's slow state to the rebuild grid
  (`coefficients.quantised_state`: speed and friction factor on log grids of `tol_speed_rel` /
  `tol_friction_rel`, temperatures, VB, BUE height and edge radius on uniform grids of their tolerances in
  `physics.cutting`, engaged edge length and engagement time on 5 % log grids) with hysteresis (a value keeps its
  row's last cell while it stays within 0.75 cell of that cell's centre: plain rounding switched a value lying at a
  cell boundary between two cells from one macro step to the next, a limit cycle with the slow feedback's one-step
  stagger), recomputes a row only when its
  grid cell changed - at the snapped state, so equal cells give equal rows (`CuttingDomain._rows` caches them by
  cell and the model's map context) - calls the coefficient model (`MechanicsModel` on `cnctwin.mechanics`, its
  process maps built lazily, or any `CoefficientModel`), linearises the rows into the kernel's `CutTable` and
  publishes `ProcessCoefficients` in `bus_next.cutting.coefficients`. While a tool is not cutting (previous and
  current step) and its spindle is stopped or changing speed (more than `SETTLE_SHARE` = 0.1 of `tol_speed_rel`
  within the last step, or more than `tol_speed_rel` since the last call), its rows wait for the speed to settle: a
  run-up is not rebuilt at every speed it passes through. Rows a tool needs before then (the first build after a
  tool change) are provisional (`MechanicsModel.rows(..., provisional=True)`: the process map's fallback nodes keep
  the kernel's own fallback, no continuation) and are recomputed in full once the speed settles or the tool cuts.
  End mill run-up: 7.0 s -> 0.8 s, drill: 211 s -> 7 s (docs/performance.md). It also integrates the spindle load
  (section 7).
* **L0 (every step, in the compiled hot loop).** `cut_step` (rotating tools) or `turn_step` (lathe
  tools on the rotating work) per contact. The hot loop never calls Python: stale memory is refilled
  from field planes computed at L2 (section 3.4), and missing planes become refill requests answered
  by a deterministic replay of the macro step.
* **Frames.** T rotates with the spindle (+z_T toward the spindle, origin at the tool tip); T0 is T at
  spindle angle 0 (non-rotating); S is the stock (material configuration, moves with the part); M is
  the machine; W the part frame of the wrench on the part. M03 is omega < 0 about +z_T. Element
  angles are right-handed about +z_T: alpha = theta + h (phase + lag) with the hand sign h = -1 for
  right-hand tools. Element frames from tooling: tangent t, chip-thickness normal n (into the
  material), cutting direction c.
* **Force convention** (per element, components t, r, a):
  F_tool = f_t (-c) + f_r (-n) + f_a a, a = -(t + sin(i) c) / cos(i) (i: inclination), with
  `f_j = K_j(h) h db + Ke_j dS`. Edge coefficients are **per unit edge length**; Altintas' Ke per unit
  axial depth of a helical edge equals ours / cos(i) (tests use Ke cos(beta) for helical cutters).
* SI everywhere; names carry units (`_m`, `_s`, `_rad`, `_N`, `_J`, ...).

## 2. Coefficient pipeline (`coefficients.py`, L2)

* **Rows.** Elements are grouped into table rows by tooling's coefficient group, edge kind, a
  low-speed variant (elements below v_min: indentation rows, `row_plan`) and forming lobes. A row is
  `Rows`: over a log-spaced h grid (`log_grid`, `physics.cutting.log_nodes` nodes from 5e-8 m to
  h_hi) the shearing coefficients K_t, K_r, K_a (Pa), the edge / ploughing forces (N/m), h_min, the
  power shares (shear plane, rake), the rake normal stress, chip ratio, chip temperature, shear angle,
  heat-to-chip/tool fractions, the process-damping coefficient K_sp (N/m^3), the flank contact
  stresses for the wear land and v_min. Sources: `MechanicsModel` (Oxley / Johnson-Cook via
  `cnctwin.mechanics`, with speed, rake, rake temperature (`bus_prev.thermal` tools'
  `rake_temp_mean_K`, `flank_temp_q_K`), workpiece preheat (`contact_preheat_K`, `work_preheat_K`),
  wear (edge radius, VB, BUE height from `bus_prev.wear`), coolant friction factor), or
  `LinearModel` (Altintas' linear edge-force model or Kienzle, for verification). The flank partition takes
  thermal's `flank_entry_K` (the flank at the entry into the cut, the background of mechanics' stationary-source
  partition; the in-cut `flank_temp_q_K`, which wear needs, already holds the step's own rise: reading it counted
  it twice), the rake partition `rake_temp_mean_K` (the rake contact's entry temperature).
* **Kienzle form** (Kienzle 1952): F_c = kc11 h_ref (h / h_ref)^(1 - mc) b, feed / passive forces by
  ratios; `LinearModel` with `kc11` builds it. Linear form (Altintas 2012 ch. 2): F = Kc h b + Ke dS.
* **Below h_min** (Waldorf 1998; Jun 2006): no shear-plane force, the edge force ramps linearly from
  0 to Ke at h_min (ploughing branch); continuous at h_min for mechanics rows.
* **Linearisation** (`linearise_row`, `build_table`): per row two uniform grids, [0, h_min] in n_lo
  segments and [h_min, h_hi] in n_hi segments; each segment stores intercept and slope of every
  column: FT, FR, FA (per chip width), ET, ER, EA (per edge length), PS, PR (power shares), SIG_RAKE,
  RC, TCHIP, WS, WF. Exact at the nodes, continuous across segments; piecewise-linear error of a
  smooth curve: second order (Kienzle h^0.74 with n_hi = 384: < 0.2 % above 20 um, tested). Above
  h_hi the last segment extrapolates and `table_over` counts it. `h_write` per row: h_min for
  cutting rows, +inf for rubbing rows (margins), 0 for forming lobes.
* **Flank wear land** (Waldorf 1999; Smithey, Kapoor & DeVor 2000, 2001): wear-land forces sigma(VB) VB dS
  (normal) and tau(VB) VB dS (tangential) while in contact (`flank_stress`), stresses from the row: the edge's
  ploughing stresses sigma_0, tau_0 at the land's edge end, elastic contact (quadratic decay to the land end,
  mean one third) below the critical land VB*, a plastic region of width VB - VB* ahead of an elastic one of width
  VB* beyond it (mechanics.md 4; the regions were swapped before, which made short lands fully plastic).
* **Process damping** (Wu 1989; Altintas, Eynian & Onozuka 2008; Budak & Tunc 2010; Ahmadi & Ismail 2011): the
  flank indents the surface the same edge has just generated behind it. The flank is a zero-clearance land
  L = VB + c_e r_e (`pd_edge_land_per_radius` c_e) followed by the clearance face receding max(0, x - L) tan(gamma)
  at the distance x behind the edge; the surface there was left where the edge was when it passed, x / V ago, so it
  stands delta(x) = (u(t) - u(t - x / V)) . n - max(0, x - L) tan(gamma) above the flank, u the element's relative
  vibration displacement (tool deflection at the element minus the work's) and n its normal. Indented area
  A = integral of max(0, delta) over [0, x_max] (Budak & Tunc's indentation volume per edge length), force K_sp dS A
  (normal) plus its friction. Each element keeps the surface it generated as a **wave memory** (`CutState.pd_ring`,
  `pd_aux`): u (T0, three components, so a rotating tool's static deflection gives no false wave) sampled every
  `pd_max_flank_m` / 15 of its travel along the material since it entered the cut, 16 samples spanning the flank;
  the profile is linear between the samples, the last evaluation and the current point, and its positive part is
  integrated exactly (`kernels.wave_area`, `wave_push`; `reference.wave_area_ref`). The memory restarts at every
  entry (the surface behind an entering edge was not generated by it) and ends where the surface was generated by
  the previous tooth: x_max = min(`pd_max_flank_m`, 2 pi r / teeth) for rotating tools, min(`pd_max_flank_m`,
  2 pi r) on the turned work. A wave much longer than the flank is the straight line of slope v_n / V, and the area
  is the linearised model's s x*^2 / 2 - tan(gamma) (x* - L)^2 / 2, x* = min(L tan(gamma) / (tan(gamma) - s),
  `pd_max_flank_m`) (`flank_interference_area`, tested to 1e-9); a vibration much shorter than the flank indents it
  by at most its own amplitude over the land and the few microns behind it before the clearance rises above it.
  A sharp edge without a land (VB = 0, r_e = 0) indents only for a slope beyond the clearance; the dissipation of a
  long wave scales ~ L^2. The force depends on the current displacement against the remembered surface (a contact
  stiffness K_sp dS times the contact length), not on the predicted velocity: it is evaluated at the predicted
  state the chip thickness uses, consistent with the surface memory, which also holds the predicted positions.
  Before 2026-11-09 the force was the linearised model at the instantaneous predicted normal velocity: an L0 step's
  velocity of nanometres of vibration of the work's 32-48 kHz modes (wavelength 60-100 um at 3 m/s) or of the
  predictor's step-to-step alternation read as a slope over the whole 0.5 mm flank, near a drill's centre (V -> 0)
  as a slope above the clearance for any vibration; the one-sided force switched on and off from step to step and
  the predictor's extrapolation made it a period-2 relay that diverged (warmup's turning pass at 2.707 s,
  drill-tap's D6.8 entry; docs/models/loop.md 15).
  **Only the vibration makes the wave**: u is the tool's deflection (station translation and rotation) minus the
  work's, not the whole relative motion. The steady path (feed) is the surface the edge itself generates, which
  its land follows (a worn land wears to it); its steady contact is the edge and wear-land force above. With the
  whole normal velocity the feed slope of every steady cut indented the land: twice the same contact, and near a
  drill's axis, where the feed is comparable to the cutting speed (s up to 0.8 at r = 0.05 mm, f = 0.26 mm/rev), a
  chisel element of 0.1 mm carried 1.7 kN (Sklenicka's thinned 8 mm drill at f = 0.26 mm/rev: thrust 4.8 kN against
  1.15 kN measured; Wang's worn end mill: +134 N of normal force).
* **Forming lobes** (Chowdhary et al. 2002, 2003; Hill 1950; Johnson 1985): contact pressure
  p = (2 + pi) k, k = sigma / sqrt(3) (sigma: Johnson-Cook strain term at `forming_strain`), contact
  length l(h) = (1 + c_er) sqrt(2 rho_eff h), rho_eff = 2 r^2 / (e N^2) (lobe relief
  e = `forming_lobe_relief_per_D` D, N lobes), F_r = p l, F_t = p h + min(mu p, k) l (mu < 0:
  sticking, tau = k). The frontal term makes the forming work of a revolution p times the displaced
  volume (tested to 2 %).
* **Rebuild policy.** A row is recomputed when its state (speed, temperatures, friction, VB, BUE, edge
  radius) moved beyond the tolerances since its last computation; the domain counts computations.

## 3. Surface memory (`abi.SurfaceMemory`, `memory.py`, kernels)

### 3.1 Representation and chip thickness

Row = memory row (a station, or a tap's handover row, 3.5), column = angle bin (720 by default) in
T0. A bin holds a point M in S, its tick and a valid flag. Chip thickness of element e is the vector
form `h_e = (p_e - M) . n_e` (architecture 5.4) with M interpolated between the two bins around the
element's angle (inside-bin only next to an interval boundary, where the outer bin is not renewed).

* **Chord correction.** The bins hold points of the swept circle; the linear stencil reads a chord.
  The kernel subtracts `r dphi^2 [(1 - w) u^2 + w (1 - u)^2] / 2 (n . p_xy / r)` (u: the element's
  position between the bin centres, w: the stencil weight). Writes add the rotation's second-order
  term so the written point lies on the circle. Residual: fourth order (~r dphi^4 ~ 3e-11 m at
  r = 5 mm, 720 bins).
* **Delayed write.** An element crossing a bin centre writes that bin one bin later (pending), so the
  element's own stencil never reads what it just wrote.
* **Elastic recovery** (Jun et al. 2006; Vogler et al. 2004): h < h_min leaves the bin unchanged (the
  ploughed layer recovers); h in [h_min, (1 + rho) h_min] moves the point by (h - h_min) / (rho h_min)
  toward the edge; above, the point is set (rho = `recovery_transition_ratio`). A bin without a point
  is set whenever h >= h_min.

### 3.2 Staleness

A bin is stale when older than one revolution plus one tooth period (ticks), written before
`stale_before` (rapid, tool change, new operation) or never written.

### 3.3 Exactness checks

1-DOF regenerative chip `h = (c + u(t) - u(t - tau)) sin(phi)` through the memory: < 1e-13 m.
Runout tooth skipping (Kline & DeVor 1983) and variable-pitch delays (Altintas et al. 1999) follow
from "the most recent passer wrote the bin".

### 3.4 Refill from the field

For the bins that are or will be stale inside (or `refill_margin_rad` around) the last engaged
intervals, `memory.candidates` lists (row, bin) pairs (at most `max_refill_queries`, spread evenly)
and `memory.compute_planes` asks removal's `SurfaceQuery.surface_planes` for a point q and outward
normal m of the material boundary along -n from the station circle. The kernel refills a stale bin
with the point on that plane along -n from the element (`refill_bin`). A stale bin without a plane
writes a refill request row (`abi.REFILL_COLUMNS`) and the element sees h = 0 at that bin; the
integration answers the requests (`planes_for_requests`) and replays the macro step
deterministically (as for event-buffer overflow). A plane is tangent to the boundary at its bin:
second-order error <= r dphi^2 / 4 (tested). Analytic stocks for tests and validation:
`HalfSpaceStock`, `BoxStock`, `BoreStock`.

### 3.5 Taps: the thread helix

A tap advances one lead per revolution. The *predecessor* of an element is the element that passed
the same angle just before it at the same S height: on the tooth that passed g / omega earlier
(smallest positive cutting-sense gap g of the tooth phases), the element whose lifted profile
(z + lead g / 2 pi) the chip line of the element crosses (`geometry.helix_rows`, actual positions,
chamfer clipping included). For the lands after the first it is the element of the same station:
read and write the station row. The first land's predecessor is the last land's element one thread
row lower, which passes the angle at the same moment as the first land's element of its own station
(which writes the station row): it writes a **handover row** nst + its station instead, which only
its first-land successors read (no read/write conflict at any angle). An element whose chip line
misses the predecessor's profile (below the first teeth, or where the chamfer clipped the
predecessor's land) **reads the field** (read row -1: the refill plane of its station, uncached).
When the read row differs from the write row the element passes the surface it read on, renewed by
the recovery rule (copy rule), so the surface follows the helix even where an element does not cut.

## 4. Force kernel (`cut_step`, per L0 step, per contact)

Per station: engaged intervals (T0 angles from removal), deflection of the station (translation and
rotation, from `deflection_from_modal`), floating-holder axial float. Per element:

1. **Engaged part of the step's arc** (`gate`): fraction, centroid, start and end; the force is
   evaluated at the arc centroid (closed-form integral of a force linear in the angle over the clipped
   arc); positions and velocities of clipped elements are rotated with the second-order series.
2. **Chip thickness** from the memory (section 3); slope over the step; velocity of the element
   relative to the material (tip, frame rotation, deflection rates, work vibration).
3. **Table** row segment, forces f_t, f_r, f_a, edge forces, wear-land and process-damping forces;
   rotated to T0 and S; moments about the station and the tip.
4. **Energy** (exact identity, tested to 1e-12): shaft + feed work = heat in the shear plane, at the
   rake, at the flank, ploughing (edge, wear land, forming, process damping) + work on the deflection
   rates (+ recut + packing). Heat by zone with the row's partitions; work-heat deposits in S. The work on the
   deflection rates (`CA_DEFLECTION_J`) holds every load the modes receive: the element forces, the recut loads and
   the packing load at their stations, and the total force on the work's vibration velocity; it equals dynamics'
   quadrature of the modal work exactly (test `test_chip_loads_close_the_energy_identity_and_agree_with_the_modal_work`).
5. **Accumulators**: per element integrals (`CutState.elem_acc`: in-cut time, h dt, force, energy),
   per tooth pass rows (chips' `EMISSION_COLUMNS`: rho x analytic chip volume, separation point and
   velocity, chip temperature, mean h, width, length, speed, rake, inclination, edge kind, group),
   entry / exit / breakthrough / breakout shocks, surface events (interior where v.n changes sign,
   boundary tangency |v.n| <= eps_t |v_rel| at entries and clipped exits, floor events at the trailing
   angle for |n_z| > 0.7), section moments and stresses at the critical sections, recut impulses
   against the frozen chip occupancy, packing loads, axis forces (axis map from the machine) and the
   spindle torque per tick.

### Recutting and chip packing (loads of the chips on the tool)

* **Recut strikes** (`recut_scan`, once per tooth period: in a tooth period the teeth together sweep every angle of
  the envelope once). Chips' frozen occupancy grid (pile cells, 2 mm) holds each voxel's chip mass spread over its
  volume, with the chips' mass-weighted thickness and volume-equivalent size. The teeth meet the share of a voxel's
  chips inside the cutting envelope (sub-cell centres, 3 x 3 x 3, inside the envelope cylinder), less what was
  struck from the voxel in the macro step (`taken_kg`). They cut them like material: uncut thickness h_r = the
  chips' thickness, width b_r = their width sqrt(V_chip / h_r) (a platelet of the volume-equivalent size) up to the
  edge length in the voxel, times the chips side by side along the edge (packing: solid fraction x edge length /
  chip width, at least one); the force F = f(h_r) b_r n from the table row `TC_RECUT_ROW` (tangential and thrust
  per unit width at h_r, edge forces included) acts at the struck part's centroid (at least 0.1 of the cutting
  radius from the axis), opposite to the cutting direction there plus the thrust away from the chips, for the
  crossing time V_struck / (h_r b_r n V) (the struck volume cut at that section at the edge speed V): impulse
  F V_struck / (A V), energy f_t(h_r) V_struck / h_r (the table's specific energy times the volume). The loads
  are kept per contact with their absolute start and end times (`CutState.recut_act`, 16 rows; a new load joins
  the one that ends last with its remaining impulse kept when every row still acts) and each L0 step applies the
  part of their time inside it: a load whose time passes while the tool is out of contact (cut_step is not called)
  has ended, it is not carried to the next engagement. Before 2026-11-09 a voxel's whole mass took
  J = m V + u m / (rho V) within one L0 step: 17 kN, then MN after the tool's response, at the tiny slot's, the
  thin wall's and the flip's first cuts (docs/models/loop.md 15). Test `test_a_recut_strike_is_cut_like_material_over_its_crossing_time`
  (a 2 mg voxel of 40 um chips: two thirds struck, 117 N for 1.7 ms instead of kilonewtons in 5 us).
* **Packing** (chips' flute packing load: a torque opposing the rotation and a force in T0) at the middle of the
  engaged edge.
* Both act at a station dynamics takes loads at (the engaged range of the step, as the hot loop's `DynSystem.act`;
  with no engaged station the work modes still take them through the total force): the station loads, the
  critical-section impulses and the total wrench carry them; their work on the deflection rates of their station and
  on the work's vibration velocity is part of `CA_DEFLECTION_J`, and the rest of their rigid-body work (spindle and
  feed) is dissipated in the chips (`CA_RECUT_J`, `CA_PACKING_J`: the run ledger's friction). Before, they reached
  the work modes through the total force but neither the tool modes nor `CA_DEFLECTION_J`: the run's energy balance
  missed their work on the vibration (the four examples that stopped at their first cut on residuals of 1e-6 to
  2e-5).

### Processes (all through the same element model)

* **Milling** (all envelopes: square, ball, bull-nose, face, chamfer, engraving, slitting, thread
  milling, boring heads): average forces against Altintas' closed forms (slot, up, down, helix:
  < 0.5 %), instantaneous single-tooth force, runout skipping, variable pitch, plunge and helical
  ramp end-edge load (2e-3).
* **Drilling**: lips + chisel (thrust and torque vs the integrals, incl. the web-offset torque term,
  2e-3); breakthrough (centre station lost) and cross-hole breakout (partial exit, once per pass);
  lateral walk force of a one-sided entry; inclined entry via the intervals.
* **Taps**: cutting taps through the helix memory (3.5): each chamfer land cuts the cone's radial rise
  over one land pitch, h = rise sin(kappa) (1 %); the first engaged teeth cut from the bore wall; the
  full-thread lands follow the thread (< 1e-4 rise); synchronisation error through the holder's axial
  float: k u = F_z (implicit static equilibrium per step with the cut's axial stiffness; tested 10 %);
  rigid holder: flanks recut, axial load and torque rise. Forming taps: lobe rows (section 2).
* **Reaming, boring**: rubbing margins (h_write = +inf), the cutting chamfer through the memory.
* **Turning family** (`turn_step`): the chip of every element is its depth inside removal's polar
  field (`PolarView`: r_out, r_in over (theta, z) of the rotating S) along -n, so threading passes,
  grooves and interrupted cuts need no special case. Straight edge vs Kienzle: the element strips
  match the strip quadrature to 0.5 %; Kienzle's uniform h over b differs by the corner triangle at
  the tip (relative area f sin(kr) cos(kr) / (2 ap)) and the truncation at the uncut surface.
  Option `turn_colwell`: forces lumped on Woxén's equivalent chip thickness A / L with the thrust
  normal to Colwell's chord (Colwell 1954; Young et al. 1987). Threading: per pass by the infeed (the
  polar field carries the previous pass). **Polar view requirement**: the bin under the tool must
  still hold the previous revolution; removal updates a bin after the tool has left it.
  **Chip section of a curved edge** (2026-11-09): h is measured along -n, so an element's chip is the strip
  between its neighbours' normals, `db (1 - h curv / 2)` wide at mid-depth (`abi.edge_curvature`, packed as
  `PK_CURV`: 1 / r_eps on a nose arc, 0 on straight edges; an annular sector, exact for h <= r_eps); the shearing
  terms, Colwell's area A and the chip volume use that section, the edge terms the edge length, the chip width of
  the chip model (`TS_STEP_B`) the engaged edge. The strip turns about the spindle axis at its centroid, h / 2
  along -n (Pappus). `h db` alone overstated the chip by h / 2 r_eps on the nose (19 % at f 0.3 mm/rev on
  r_eps 0.8 mm) and by 5-6 % on the lathe-shaft example's G71 pass (cutting's chip volume 1.06 x the polar field's
  removal); a nose insert's chip of a revolution now matches the exact region between the profile and its shift by
  f to 0.04 % (tests/unit/cutting/test_turning.py). `cut_step` keeps `h db` and the edge radius: its h is the
  projection on n of the surface's shift between passes at the element's own station, so `h db` tiles the chip
  exactly (strips along the shift, Cavalieri) and `r_xy` is the exact Pappus radius (a slot's chip volume is
  2 R ap v_f t; the narrowing and the centroid radius took 0.4 % off it). **Depth ray**: `polar_depth` marches in
  half rows and bisects the bracketing half row on the sign (12 halvings); `polar_inside` is a radial distance that
  jumps where the ray enters an air gap or the next row's surface, and the linear root of the two samples sat near
  the outside one (+1-2 % on the G71 chip).

## 5. Kernels, signatures and measured cost

All `@njit(<explicit signature>, cache=True, nogil=True, error_model="numpy")`, one signature each
(tested), no fastmath / parallel / jitclass, C-contiguous arrays and NamedTuples of them only.

| kernel | signature (abi types) |
|---|---|
| `cut_step` | `void(TOOL_T, TABLE_T, MEMORY_T, INTERVALS_T, REFILL_T, f8[::1] step, i8[::1] istep, POSE_T, DEFLECTION_T, OCCUPANCY_T, PACKING_T, AXES_T, STATE_T, ACCUM_T, CACCUM_T, OUT_T, EVENTS_T)` |
| `turn_step` | `void(TOOL_T, TABLE_T, POLAR_T, f8[::1], i8[::1], POSE_T, DEFLECTION_T, AXES_T, STATE_T, ACCUM_T, CACCUM_T, OUT_T, EVENTS_T)` |
| `deflection_from_modal` | `void(ModalModel, ModalState, f8[:, ::1] residual force T0, Deflection)` |
| `copy_element_accumulators` | `void(STATE_T, ACCUM_T, i8 n_elem)` |

Python reference: `reference.cut_step_ref` (rotation matrices, vectors in S, direct sums) agrees with
`cut_step` to round-off over 60 steps with runout, clipping, deflection, rotary motion, work
vibration, wear land and process damping (`test_kernels_reference.py`).

**Measured cost** (docs/performance.md "cut_step"; `test_tools_state_perf.py::test_milling_kernel_cost`, called
from a compiled loop so the 17 NamedTuple arguments are not boxed): slotting with a 4-flute 30 deg helix cutter,
160 engaged elements per step: ~34 us per L0 step (~210 ns per engaged element; 2026-11-03: ~50 us, ~290 ns). In
the integrated hot loop: 2.8-3.0 us per step for the pocket-islands helical entry, 8.2-8.3 us for the D10 4FL slot
(27 engaged elements; 16.9 us and 8.5 us before). How: `cut_step` visits only stations with engaged intervals (or loads written in the last two
steps), classifies an element's swept arc against a station's single interval with two comparisons (the general
`gate` within 1e-12 rad of a boundary, so the results are those of `gate`), skips the boundary tests of interior
elements, reads every element constant from one packed row (`CutTool.pk` / `pki`, built at L2 by
`abi.pack_elements`, rows in station-list order), inlines its small helpers at the IR level and is compiled
without reference counting (`RAW`: it allocates nothing). Per element and step it evaluates only table lines and
products (no pow / exp; sin and cos of the spindle angle by angle addition in the caller). Called from Python each
call adds ~40 us of argument unboxing.

## 6. Spindle (`spindle.py`, L2)

Per spindle and gear: torque and power curves from the machine (`spindles`), available torque
T_av(omega) = min(T_curve, P_curve / omega). Continuous rating: S1 curves as given; S6 x % curves
times sqrt(x) (equivalent thermal load); short-time "t min" curves times sqrt(1 - exp(-t / tau))
(IEC 60034-1). Winding load theta (1 = S1 limit) by the exact first-order update with time constant
`motor_thermal_time_constant_s`. When the cutting torque exceeds T_av the speed drops to the first
balance below the commanded speed with T_cut ~ omega^-(1 - m) (Kienzle exponent m,
`default_force_exponent`); none above `stall_speed_ratio` x commanded: stall. Findings:
`spindle_overload` (torque above the curve, theta > 1, stall) with the gear and the margin.

## 7. Domain (`CuttingDomain`)

`init` (needs `state.tooling`): settings, spindles, axis map, density, one coefficient model per tool
slot. `kernel_inputs`: per motion path a `ContactInputs` (tool, table, memory with handover rows,
intervals, refill planes for the stale bins of the coming step, frozen chip occupancy and packing
loads, step arrays). `absorb`: event-buffer overflow doubles the capacity and asks for a replay; so do
refill requests (after computing their planes); outputs go to `bus_next.cutting.contacts`.
`macro_step`: coefficient rebuilds and `ProcessCoefficients`, spindle load and findings.
`end_operation`: every memory bin becomes stale. `finalize`: per-tool summary. State checkpoint:
`CuttingState.to_arrays` / `from_arrays` (bit-identical continuation, tested).
A row's slow state takes thermal's tool temperatures and wear's VB / BUE as means over the row's member elements;
a row without members in the step (a low-speed variant no element runs at) or a non-finite mean leaves the value
unset (None: the model's default), never a NaN (it stopped the drill-tap example's run in the rebuild grid).

### Bus fields read (from `bus_prev`)

| record | fields |
|---|---|
| motion.paths[] | tool_slot, workpiece, spindle_omega_rad_s, spindle_angle_rad, pose_S_T, tip (S, M), events |
| thermal | tools[slot].rake_temp_mean_K, flank_entry_K (flank_temp_q_K, q_weights for a record without it); contact_preheat_K, work_preheat_K |
| wear | tools[slot].VB_m, bue_height_m, edge radius (via tooling's effective edge) |
| coolant | friction_factor |
| chips | occupancy_origin_S_m, occupancy_cell_m, occupancy_mass_kg, occupancy_size_m, occupancy_thickness_m; packing loads |
| removal | contacts (engaged intervals, workpiece) and the SurfaceQuery for refill planes |

### Bus record written: `bus_next.cutting` (`CuttingFrame`)

Contract: `contacts[]` (`CuttingContact`: wrenches, per-element accumulators, heat by zone, chip volume,
surface events, deposits, section peaks, recut rows, axis force and spindle torque per tick) and
`coefficients[slot]` (`ProcessCoefficients`). Additive fields (commented in core/bus.py):
`emissions`, `displaced_volume_m3`, `forming_J`, `process_damping_J`, `recut_J`, `packing_J`,
`deflection_work_J`, `h_max_step_m`, `engaged_edge_m`, `pass_time_s`, `mean_speed_m_s`,
`omega_mean_rad_s`, `spindle_available_Nm`, `refill_missing`, `table_over`, `float_u_max_m`,
`axis_names`.

## 8. Parameters

| file | parameter | value | source |
|---|---|---|---|
| cutting-contact | recovery_transition_ratio | 1.0 | est. (Jun 2006, Vogler 2004 for the regime) |
| cutting-contact | pd_edge_land_per_radius | 1.0 | est. (Altintas 2008, Budak & Tunc 2010) |
| cutting-contact | pd_max_flank_m | 0.5 mm | est. (clearance land widths) |
| cutting-contact | forming_pressure_factor | 2 + pi | Hill 1950 |
| cutting-contact | forming_strain | 0.8 | est. |
| cutting-contact | forming_lobe_relief_per_D | 0.004 | est. (Chowdhary 2003 geometry) |
| cutting-contact | forming_recovery_contact_factor | 0.3 | est. (Chowdhary 2003: recovery raises loads) |
| cutting-spindle | motor_thermal_time_constant_s, default_peak_rating_s, stall_speed_ratio, default_force_exponent | see file | IEC 60034-1, est. |

## 9. Determinism and RNG streams

No random numbers (no RNG streams). Every loop runs in a fixed order; replays after refill requests or
buffer overflow are bit-identical.

## 10. Assumptions, validity and error bounds

* Mechanistic forces per element, quasi-static (no chip inertia); validity of the coefficient model's
  range (mechanics: Oxley regime; LinearModel: as calibrated).
* Memory discretisation: 720 bins; chord corrected to fourth order; refill planes second order
  (<= r dphi^2 / 4); stencil inside-bin rule at interval edges (<= one bin of angle).
* Element strips normal to the edge: at convex corners of a profile (tap tooth crest-flank) the chip
  columns of neighbouring elements overlap; the chip volume of a cutting tap is 1.0-1.2 x the thread
  groove (tested); in turning the corner triangle at the tip is missed (f sin cos / (2 ap) of the
  area, tested).
* Tap helix rows need at least two lands; first-land elements whose discretisation differs from the
  last land's by dz see |n_z| dz of the bore wall (0.1 x rise, tested).
* Floating holder: quasi-static (no inertia), linear spring, travel limit.
* Process damping: flank indentation of the wave the vibration leaves on the surface the same edge generated (the
  vibration's displacement only), from the element's wave memory at a spacing of pd_max_flank / 15 (waves shorter
  than two spacings, about 70 um, are under-resolved: their indentation is bounded by their amplitude either way);
  plastic indentation coefficient K_sp from the row (linear in the indented area: no hardness bound of the
  indentation pressure); the flank's clearance face is straight (the circle's curvature behind a rotating edge,
  x^2 / 2 r, is not subtracted).
* Recutting: every struck chip is cut (trapped against the work), none knocked away freely; the voxel's chips are
  spread uniformly over it (2 mm cells).
* Indentation rows (chisel edges, normal rakes below -45 deg, elements below v_min): the rake face carries the
  wedge-indentation pressure of its inclination (mechanics.md 4), the rounded edge the flat-punch pressure.

## 11. Limitations

* Kernel cost ~200-300 ns per engaged element and step (docs/performance.md): 8.3 us per step for the D10 4FL
  slot's 27 engaged elements (the 3 us target is met for the pocket's helical entry, 2.8 us, not for the slot).
* Tap retraction (spindle reversal) reverses predecessor relations; the handover rows then hold the
  wrong turn until refilled (approximate rubbing on retraction).
* The turning chip of a radial edge (kappa_r = 90 deg) resolves its previous wall to the polar field's
  z bin (h error <= dz).
* Single-land taps (N = 1) keep a read/write conflict on the handover row.

## 12. References

Altintas 2012 (`altintas-2012-mfg-automation`); Kienzle 1952 (`kienzle-1952`); Smithey et al. 2000, 2001
(`smithey-2000-worn-tool-3d`, `smithey-2001-worn-tool-forces`); Waldorf 1999 (`waldorf-1999-worn-tool-ploughing`);
Waldorf 1998 (`waldorf-1998-ploughing-slipline`); Hill, Lee & Tupper 1947 (`hill-lee-tupper-1947-wedge`); Jun et al. 2006
(`jun-2006-microendmilling-dynamics`); Vogler et al. 2004 (`vogler-2004-micro-endmilling-surface`);
Liu et al. 2007 (`liu-2007-microendmilling-surface`); Wu 1989 (`wu-1989-process-damping`); Altintas,
Eynian & Onozuka 2008 (`altintas-2008-process-damping`); Ahmadi & Ismail 2011
(`ahmadi-ismail-2011-process-damping`); Budak & Tunc 2010 (`budak-tunc-2010-process-damping`);
Colwell 1954 (`colwell-1954-chip-flow`); Young et al. 1987 (`young-1987-nose-radius`); Woxén 1932
(`woxen-1932-equivalent-chip`); Kaymakci et al. 2012 (`kaymakci-2012-unified-force-model`);
Chowdhary et al. 2002, 2003 (`chowdhary-2002-thread-forming`, `chowdhary-2003-thread-forming-forces`);
Agapiou 1994 (`agapiou-1994-tapping`); Mezentsev et al. 2002 (`mezentsev-2002-tapping-faults`); Dogra,
Kapoor & DeVor 2002 (`dogra-kapoor-devor-2002`); Chandrasekharan et al. 1998
(`chandrasekharan-1998-drill-forces`); Kline & DeVor 1983 (`kline-devor-1983-runout`); Altintas et al.
1999 (`altintas-1999-variable-pitch`); Tlusty & Ismail 1981 (`tlusty-ismail-1981-nonlinearity`); Hill
1950 (`hill-1950-plasticity`); Johnson 1985 (`johnson-1985-contact-mechanics`); IEC 60034-1:2017
(`iec-60034-1-2017`).
