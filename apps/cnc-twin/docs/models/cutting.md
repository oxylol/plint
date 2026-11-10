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
  element's own stencil never reads what it just wrote. A crossing at a step's end belongs to both steps: the range
  is widened by 1e-7 bins and a bin already pending is not crossed again (2026-11-10: with a stride commensurate with
  the bins, 0.25 or 0.05 deg on 0.5 deg bins, round-off left some bin centres outside both steps' ranges, the passing
  tooth did not write them and the next read a surface three passes old: h = 3 f_z, the force doubled, the mean
  +2 %; tests/unit/cutting/test_boundary_and_peak.py `test_a_commensurate_stride_writes_every_bin`).
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
   rotated to T0 and S; moments about the station and the tip. **Engaged length** (2026-11-10): an element whose
   station is engaged at its angle while a neighbouring station along the edge is not lies at the material's boundary
   (the depth-of-cut line, a floor's or a step's edge); its half toward that neighbour is replaced by the engaged
   length up to the boundary, so its chip width, edge length, wear-land and process-damping forces (and with them
   heat, chip volume and torque) carry the weight 1 + d (`CutState.st_edge`, `SE_D_LO` / `SE_D_HI`, each where that
   neighbour is out of the material at the element's angle). The hot loop measures d after every refresh of the
   engaged intervals (`core.hotloop.edge_reach`: at the middle of the angles where the neighbour is out, the gating's
   own point test (removal's `classify`, the test `engaged_intervals` makes on the station circles) is sampled along
   the edge from s toward n, first along s's element (its meridian tangent) to the element's end, then on to n's
   point, 16 samples refined by 12 bisections; the boundary lies at length d_b from s, d = d_b / db_s - 1/2. Past a
   corner of the edge (n's element turned by more than 45 deg, a flat end mill's corner) the search ends at s's
   element end: the material found beyond it is s's own uncut chip across its thickness, f_z sin(phi) along the
   floor, which counted as width gave +1.6 % at 1 mm elements); rows
   carry the gating's tick and are not used with other intervals; `drive.edge_plane` does the same for analytic
   planes. A first version (2026-11-10, same day) took removal's `ray_cast` exit along the chord between the two
   station circles: the station's point lies on the surface being cut, where the ray's start test often found air,
   and at the corner of a flat end mill the chord runs through the cleared inside of the cut rather than along the
   edge, so the lowest flank element lost its lower half (measured `st_edge` -0.48 at the D10's z 0.3 mm station;
   element_flat el1 -9.6 %, el0.5 -6.2 %). Before, the element was engaged
   whole or not at all: the forces erred by (engaged length - ap) / ap, up to half an element's height over ap (-4.8 %
   at ap 3 mm, -32 % at 0.6 mm with 0.5 mm elements; docs/convergence.md 3). The boundary is taken at one angle per
   station side (exact for a top or floor normal to the tool axis; an inclined boundary is averaged over the angles
   where the neighbour is out), and a layer that ends between the tool's end and its first station (thinner than half
   the first element) is not seen by the gating at all. Tests: tests/unit/cutting/test_boundary_and_peak.py (slot ap
   2.86 mm with 1, 0.5, 0.25 and 0.125 mm elements: force per depth and chip volume within 0.2 %; whole elements
   spread by several per cent; 0.7 mm with 1 mm elements: weight 0.7).
4. **Energy** (exact identity, tested to 1e-12): shaft + feed work = heat in the shear plane, at the
   rake, at the flank, ploughing (edge, wear land, forming, process damping) + work on the deflection
   rates (+ recut + packing). Heat by zone with the row's partitions; work-heat deposits in S. The work on the
   deflection rates (`CA_DEFLECTION_J`) holds every load the modes receive: the element forces, the recut loads and
   the packing load at their stations, and the total force on the work's vibration velocity; it equals dynamics'
   quadrature of the modal work exactly (test `test_chip_loads_close_the_energy_identity_and_agree_with_the_modal_work`).
5. **Peak** (2026-11-10): besides the step mean, the instantaneous wrench at the step's start and end: each element
   engaged at that instant at its chip of that instant (h linear over the step, the table read at that chip in its own
   segment; constant flank and process-damping terms), plus the chip loads' mean, rotated to T0 at the spindle angle of that instant.
   The step's peak wrench is whichever of start, end and mean has the largest |F|; it feeds every peak (`ACC_FORCE_PEAK_N`,
   `CA_FPEAK_*` and the new `CA_FPEAK_INST_N` = `CuttingContact.force_peak_inst_N`, the per-tick `wrench_tool_T0_peak`
   and `wrench_work_W_peak`); the means are unchanged. Before, a peak was the largest step mean: an element entering
   the cut (its edge force, in down milling its whole chip) is averaged over the step, 5-11 % low at 0.5 deg per step
   (docs/convergence.md 1, 4.1). Test: half-immersion down milling at 2 deg per step gives the instantaneous peak of a
   0.05 deg stride within 1 % (its largest step mean was more than 3 % lower), the same mean force. A first version
   (same day) extrapolated the centroid chip's segment slope to the ends: a chip that fell from 49 um to 0.09 um within
   a step sat on the steep segment below h_min and gave 1.7 kN at the step's start (stride_slot F_peak 850 N at 1 deg
   per step against 258 N converged).
6. **Accumulators**: per element integrals (`CutState.elem_acc`: in-cut time, h dt, force, energy),
   per tooth pass rows (chips' `EMISSION_COLUMNS`: rho x analytic chip volume, separation point and
   velocity, chip temperature, mean h, width, length, speed, rake, inclination, edge kind, group),
   entry / exit / breakthrough / breakout shocks, surface events (interior where v.n changes sign,
   boundary tangency |v.n| <= eps_t |v_rel| at entries and clipped exits, floor events at the trailing
   angle for |n_z| > 0.7), section moments and stresses at the critical sections, recut impulses
   against the frozen chip occupancy, packing loads, axis forces (axis map from the machine) and the
   spindle torque per tick.

### Recutting and chip packing (loads of the chips on the tool)

* **Recut strikes** (`recut_scan`, once per tooth period: in a tooth period the teeth together sweep every angle of
  the tool's surface of revolution once). Chips' frozen occupancy grid (pile cells, 2 mm) holds each voxel's chip mass
  spread over its volume, with the chips' mass-weighted thickness and volume-equivalent size. **Which chips the teeth
  meet** (2026-11-10): only those in the layer the edges advance into through air. In the tooth period the surface of
  revolution moves by d = v_tip T_tooth relative to the stock; its leading side sweeps max(0, n . d) per unit area
  (n the outward meridian normal of each station's band, of area r db per radian; sampled every 1/16 of a pile cell
  along each station's circle, `RECUT_ARC_PER_CELL`), and where the station is engaged at that angle the layer is
  material (the chip being formed), not chips. The struck mass of a voxel is that layer's air volume in it times its
  chip density, less what was struck from the voxel in the macro step (`taken_kg`). Chips inside the envelope - in the
  flutes, carried round and out by the rake faces - are not cut (their impulse m V is ~1e-7 N s). Before, every chip
  inside the envelope cylinder was cut once per tooth period: a drill took the chips of its own flutes (the D6.8 of
  examples/drill-tap-plate-1045 at 5.136 s: 0.39 mg struck near the axis at 0.17 m/s, 431 N sideways for 2.2 ms, the
  drill thrown into a 0.15-0.4 mm vibration; docs/models/loop.md 15), and a standing tool took chips around it. A peck
  drill re-entering a hole with chips below its point, or a cutter advancing on chips in a pocket, still meets them.
  The teeth cut the struck chips like material, at the table's specific energy f_t(h_r) / h_r (row `TC_RECUT_ROW`,
  h_r = the chips' thickness, edge forces included): energy f_t(h_r) V_struck / h_r. **Force** (2026-11-10): the
  voxel's chips are spread uniformly over the layer, so the edges meet them all along the band's arc in the voxel,
  over the time they take to sweep it, t_pass = min(arc / omega, T_tooth) (each angle once per tooth period): the
  mean force F = f_t V_struck / (h_r t_pass V), V the edge speed at the struck layer's centroid (at least 0.1 of the
  cutting radius from the axis), bounded by the section force of whole chips side by side, f(h_r) b_r n (width b_r =
  the chips' width sqrt(V_chip / h_r), a platelet of the volume-equivalent size, up to the edge length in the voxel;
  n = solid fraction x edge length / chip width, at least one), which dense chips reach. It acts opposite to the
  cutting direction plus the thrust (f_r / f_t of it) against the struck surface's normal, for V_struck f_t / (h_r F V)
  (t_pass, or the crossing time at the section force). Before, every strike took the section force for the crossing
  time of its volume: 13 ug of chips near the D6.8's axis gave 431 N for 50 us, a kick a light drill mode turned into
  a 0.16 mm vibration (the mean force of the same strike is about 6 N). The loads are kept per contact with their absolute start and end times
  (`CutState.recut_act`, 16 rows; a new load joins the one that ends last with its remaining impulse kept when every
  row still acts) and each L0 step applies the part of their time inside it: a load whose time passes while the tool
  is out of contact (cut_step is not called) has ended, it is not carried to the next engagement. Before 2026-11-09 a
  voxel's whole mass took J = m V + u m / (rho V) within one L0 step: 17 kN, then MN after the tool's response, at the
  tiny slot's, the thin wall's and the flip's first cuts (docs/models/loop.md 15). Tests (tests/unit/cutting/
  test_chip_loads.py): chips spread round a D10 feeding through air - the struck mass is f_z x 2R x ap at their density
  (0.2 %); a plunging cutter (chips in its flutes) and a slot (the leading side in material) strike nothing; a strike's
  energy is the specific energy times its volume, its force the mean over the arc's sweep (14 N for 2 mg in a voxel
  ahead) or, for a nearly solid voxel, the section force of whole chips.
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

**`_row_state` (the coupling round's change of 2026-11-09, reviewed 2026-11-10).** The means are weighted by the
previous macro step's in-cut time of each member, summed over the teeth at its position along the edge (a helical
edge's row spans the whole flute while a pass engages a few of its elements: the plain mean was dominated by the idle
part, 2 of 44 elements per tooth cut on the convergence side step; summed over the teeth so that the weights do not
follow how many passes of each tooth one step held). With no member in cut (a tool entering, or between cuts) the plain
mean over the members stands in. Non-finite members get no weight. The rake partition reads `rake_temp_mean_K` and the
flank partition `flank_entry_K` (the entry temperatures: thermal publishes the background of mechanics' stationary
source; the in-cut values would count the step's own rise twice), `flank_temp_q_K` only for a record without it. The
contact time of a rotating tool's element is its engaged arc over the spindle speed, min(arc, 2 pi) / omega, with
removal's `engaged_angle_rad` (the largest engaged angle of any station in the step); before, the step's in-cut time
over its entries, which followed how many passes crossed the step's ends. Review: correct and consistent with
thermal's record; three limits, documented rather than changed: (1) the arc is the largest of the stations', an upper
bound for tools whose engagement varies along the edge (a ball nose's small radii, a helical entry), so their contact
time and the transient partition it feeds are those of the most engaged station; (2) a continuously engaged element
(a drill's lips, a full slot's 180 deg is not one) gets one revolution, not a steady contact; (3) with several contacts
of one slot in a step (two workpieces) the last contact's width, time and arc are used. The coupling monitor uses the
same weights (section "Coupling-lag monitor" of `_lag_monitor`: the tangential coefficient at the reference chip
averaged over the rows with their elements' in-cut time, under the loop's alternation rule, db/models/loop.yaml).

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
  x^2 / 2 r, is not subtracted). The surface x behind a rotating edge lies x / r rad back on the element's circle,
  where the generated surface's normal is turned by that angle, but the wave memory is projected on the element's
  current normal and the force acts along it: exact for x << r (end mills: x / r <= 0.1), not near a drill's axis
  or a ball nose's tip, where the memory spans up to 2 pi r / teeth (pi rad for a two-lip drill). There the
  indentation of a large lateral vibration is a delayed stiffness of arbitrary phase; on the drill example the
  response to a 0.4 mm kick became a sustained vibration with process damping on and decayed with it off
  (docs/models/loop.md 15; open: project each node on its own normal and apply the force along it).
* Recutting: every struck chip is cut (trapped against the work), none knocked away freely; the voxel's chips are
  spread uniformly over it (2 mm cells), also over its material and its part inside the tool, so the air layer the
  edges sweep takes the voxel's mean density.
* A chip beyond the tool's reach (thicker than its diameter, or not finite) is a diverged state, not a cut: it is
  counted (`CA_H_BEYOND`, `CuttingContact.h_beyond`), the table is read at its last segment (the index is compared
  before its integer conversion: a 649 km chip wrapped it and the kernel read outside its table, a general protection
  fault of the drill example), and the domain stops the run with a fatal finding naming the chip.
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
