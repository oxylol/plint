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
| `toolfield.py` | 3-D field of the cutting region (insert, or the tooth tip of a solid tool), solved with the ladder |
| `workpiece.py` | cut-cell finite volumes on the stock lattice, fixture body, two-scale contact temperature |
| `machine.py` | spindles (network or lumped body), ballscrew axes, structure (first-order bodies, exact integration) |
| `spindle.py` | physical spindle network: bearing friction, motor losses, shaft / bearing sets / housing / jacket, growth |
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
* Where the 3-D cutting-region field is active (2.4) it replaces the far-field composition: the
  elements it carries take its temperatures, and the 1-D body no longer stands in for the first
  millimetres of the tool.

### 2.4 Cutting region: 3-D field (`toolfield.py`, `kernels.field_step`)

Beyond about 2 max(l_c, l_f, db) the composition of 2.2 leaves only the 1-D body, a mean over the whole
section, so the first millimetres of an insert (where IR windows, spots and embedded thermocouples
measure) read close to the body temperature. The field resolves them: transient 3-D conduction in the
tool's cutting region, driven by the rake and flank fluxes over their contact areas, convecting on its
exposed faces, solved as one implicit system with the ladder. `physics.thermal.tool_field` (1) builds it
for every tool whose EdgeModel the domain receives; `ToolThermal.enable_field` builds it directly.

**Frame and mesh.** Edge frame E: x along the rake face normal to the edge into the tool, y along the
edge (unrolled: cumulative chip width of the elements), z the rake face's outward normal (tool at
z < 0). Cross-section: a polar grid about the edge line, rings r_k = r_1 g^(k-1) (`tool_field_ring_mm`
2 um, at most a quarter of the smallest edge radius; `tool_field_growth` 1.4), equal sectors over the
wedge angle beta = pi/2 - rake - clearance (median over the field's elements; sectors at most
`tool_field_sector_deg` 15 deg), so the rake face is theta = 0 and the flank face theta = -beta
exactly, the contact scale (um) and the body scale (mm) are both resolved and cells near the edge are
isotropic. Slabs along y: the element boundaries (closer than `tool_field_dy_mm` / 2 merged, intervals
longer than `tool_field_dy_mm` 0.5 mm split; contact ends given to `enable_field` are kept), so a
contact ends on a cell boundary.

* **Insert**: the section edge - rake back (x = ic) - bottom back - flank bottom (depth = insert
  thickness, flank inclined at beta), over y in [0, ic] from the corner (the prism of the ladder's insert
  volume ic^2 s); cut cells only where the polygon clips the far rings (exact areas of the clipped
  sectors, 32-point arcs). The corner is the prism's end face y = 0 (a 90 deg corner; the nose and
  major edge are unrolled along y from the nose end at the minor edge, the minor edge lies along x on the
  end face). The bottom is the seat: per unit bottom area 1 / [A_bottom / (h_c A_seat) + H_bar / 3 k_bar]
  to the bar's first ladder node, h_c = `shank_contact_conductance_W_m2K`, A_seat =
  `insert_seat_area_ratio` ic^2 (thermal-tool.yaml) and H_bar / 3k the bar's internal resistance from its
  top face to its section mean (quasi-steady profile of a slab heated through one face, Carslaw & Jaeger
  1959 ch. III: T_s - T_mean = q H / 3k). Back and end faces convect. The ladder's insert nodes become
  **views**: mass and mass-weighted enthalpy of the cells by radial overlap with each node's distance
  range from the edge (they hold exactly the field's mass and energy, so `body_temp_at`, `growths`,
  `mean_temp` and the checkpoint keep their meaning).
* **Tooth** (solid rotating tools): the open wedge of one tooth to the flute depth R_w = (D - D_core)/2
  (`core_diameter_mm` of the tool item, else D/2 (1 - sqrt(A_fluted / (pi D^2/4)))), along the first
  `tool_field_length_mm` (12 mm) of the edge from the first element. Its outer arc couples to the ladder
  node at the slab's height through the quasi-steady parabolic profile of the fluted section (cylinder
  with a uniform surface flux, Carslaw & Jaeger 1959 ch. VII: T_s - T_mean = q R / 4k), 8 pi k per unit
  length shared by the teeth (per unit root area of the representative tooth n_t R_w beta / (8 pi k)).
  One representative tooth carries the sum of the teeth's heat with every capacity and conductance
  times n_t (the system stays symmetric); tooth j's rise above the core at element e is the
  representative's times Q_je / mean_j Q_je (linear superposition). The wedges' section, mass and wetted
  perimeter are taken out of the ladder nodes they lie in (each keeps at least a fifth); the prism's two
  ends are cuts through the tooth (adiabatic); elements beyond the field use the ladder and 2.2.

**Equations.** Per cell, backward Euler in delta form with the properties at the start of the step:

    (C_i + dt sum_j G_ij + dt sum_c G_ic) dT_i - dt sum_j G_ij dT_j = Q_i + dt sum_j G_ij (T_j - T_i) + dt sum_c G_ic (T_c - T_i)

internal faces G = A / (d_a / k_a + d_b / k_b) (two-point flux, d the centroid's distance from the face;
consistent on the polar grid, whose neighbour lines are normal to the faces), exposed faces
G = A_w / (d / k + 1 / h) per medium (air: the ladder's correlation for that tool at the face
temperature plus linearised radiation; coolant: h of the bus over the reach fraction; faces under a
contact do not convect), coupled faces G = A / (d / k + r_c + l_k / k_node). The ladder's nodes, links
and end/air/coolant conductances are the same unknowns and terms as in `ladder_step`. Sources: each
element's rake heat as a uniform flux over [0, l_c] x its y-range on the rake face and its flank heat over
[0, l_f] on the flank face (l_c, l_f of 2.2), mapped onto the faces by overlap and renormalised (every
joule placed). Identity: sum C dT = sum Q - dt sum_c G_c (T + dT - T_c) - sum(b - A dT); the last term is
the solver's residual (|b - A dT| <= `tool_field_tol` |b|, 1e-12), booked by the tool's ledger with the
rest (measured residuals 1e-16 .. 4e-13).

**Solver.** The symmetric M-matrix is solved by conjugate gradients (Hestenes & Stiefel 1952) with a
block symmetric Gauss-Seidel preconditioner (Saad 2003, ch. 10): the diagonal blocks are the slabs (banded
within the sector count by the ring-major numbering) and the ladder (bandwidth 2), each factored exactly
by banded Cholesky every step; the start is the previous increment when it beats zero. IC(0) was tried
first and needed ~3.5x the iterations on the graded mesh. Fixed operation order: deterministic; the
previous increment is checkpointed with the state.

**Cadence.** In a run the field and the ladder are solved every `tool_field_every_n_macro` (10) macro
steps over the accumulated time; the heat of the steps in between is accumulated per element exactly
(energy and heat-weighted contact sizes) and mapped once per solve; idle steps, operation ends and
`finalize` solve what is pending. The near-edge field is quasi-steady on this scale (l^2 / alpha = 1-10 ms)
and the insert body's time constant is ~1 s; a new engagement reaches the edge temperatures within one
interval.

**Edge temperatures** of the elements the field carries: the field's cycle-mean face temperatures over
the element's rake strip (`edge_temp_q_K` base), flank strip (`flank_temp_q_K`) and the innermost ring
(`edge_peak_q_K`; T_face = T_cell + (q_in - q_out) d / k), plus the **periodic deviation** of the edge kernel
(`EdgeInput.scal[2] = EDGE_OUT_DEVIATION`): wedge (theta(tau) - <theta>) with the exact cycle mean
<theta> = q W(N T) / T, W(t) the integral of the unit response over [t - t_h, t]; zero for continuous
cutting. The cycle amplitude is the kernel's, as before; `body_temp_at_elem_K` stays the ladder's (the
core, or the insert's first view).

**API.** `ThermalState.tool_temperature_at(slot, points_T)` (and `ThermalDomain.tool_temperature_at(state,
slot, points_T)`): temperatures at points of the tool frame T (rotating tools at spindle angle 0; the
point is taken into the edge frame of the nearest field element: x = (p - p_e) . (cos g (-n) - sin g m),
z = (p - p_e) . (cos g m - sin g n), y = y_e + (p - p_e) . t with n, m, t the chip-thickness normal,
cutting direction and edge tangent made orthonormal, g the normal rake), the ladder's body temperature
outside the field. `ToolThermal.temperature_at(points_E, outside="nan" | "nearest")` in E,
`contact_temperatures(l_c, l_f, width, start)` the field's contact means, `deposit_contact(rake_J, flank_J,
l_c, l_f, width, start, wedge_angle_rad=, edge_radius_m=)` one contact patch (for a library user
without an EdgeModel; builds a requested field), `deposit_elements(edges, heat, vb)` the domain's path,
`flush_field`.

**Validity and error** (tests/unit/thermal/test_toolfield.py, default mesh):

* Exact solutions (method of images for wedges beta = pi/n, Carslaw & Jaeger 1959 ch. X, evaluated with
  `moving_rect_rise` at v = 0; rectangle flux adjacent to the edge, constant properties, 0.05 s and
  2 s): beta = 180, 90, 60, 45 deg within 4.5 % of the rise at the edge, on the rake face in and beyond
  the contact, 1-2 mm inside the wedge and beside the contact; the contact mean within 8 % (+4 %
  typical): a lateral-resolution bias at the contact's two ends (0.5 mm slabs against l_c = 0.2 mm),
  2.5 % at growth 1.25 / 10 deg / 0.25 mm, where every point is within 2 %. Half-space rectangle
  centred on the edge (both faces heated) against (q a / k) Phi(Fo, m): centre 3 %, mean 8 %.
* Kernel against a dense direct solve of the same couplings: increments 1e-9, every energy term 1e-10.
* Geometry: the corner is 90 deg (the insert's included angle is not represented), the edge is
  unrolled straight (nose curvature not represented), one wedge angle per field, the chip-breaker
  groove beyond the wedge and the insert's contact with its pocket walls are not represented; the
  tooth's flute depth is a geometric estimate where the tool item gives no core diameter.
* Interrupted cutting: the field holds the cycle mean (solve interval >> tooth period); the periodic part
  is the semi-infinite kernel's (penetration sqrt(alpha T) << tool size), its tail approximation
  (error < 3e-3 relative) shifts the cycle mean slightly.
* First order in time (backward Euler) and in the lagged properties; 4 % of the rise at the diffusion
  front after 5 steps (0.05 s at 10 ms).

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

* **Spindle**: the physical network of 4.1 (the default), or the lumped first-order body when the
  machine file gives only the lumped data or says `thermal.model: lumped`: bearing heat
  `P = P_max (|omega|/omega_max)^(5/3)`, time constant and growth at max speed from the file
  (`growth_at_max_speed_um`, `time_constant_min`, `bearing_heat_at_max_speed_W`), `L_eff` (est. 0.25 m),
  `dT_max = growth_max / (alpha L_eff)`, `G = P_max / dT_max`, `C = tau G`, growth `alpha L_eff (T - T_ref)`.
  Either way the published growth is the growth since the start of the run (when tool lengths were set)
  times `comp_residual_fraction` when the control compensates. Direction in M: milling/live spindles
  `-axis_T_M` (nose toward the work), turning `+z_M`, sub spindles `-z_M`. The holder's taper conduction
  enters the spindle (the shaft node of the network). Defaults used are listed in `finalize()["estimated"]`.
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

Error: the network's growth is uncertain by about +-50 % when the bearings, lubricant, housing and
cooling are estimates (Palmgren's f0 alone spans a factor 2 between makers' and textbook values; the
housing's exposed area and growth length each move the plateau by tens of per cent), +-30 % with a
described spindle; the lumped body +-30 % against a measured warm-up curve; axis drift: first-order lumped
screw, the temperature profile along the screw (hotter in the travelled range) is not resolved (+-50 %);
structure: sensitivity is a machine-file estimate.

### 4.1 Spindle network (`spindle.py`)

Nodes: the **shaft** (with the inner rings and a built-in motor's rotor), one node per **bearing set**
(rolling elements, cage, lubricant: `bearing_rolling_mass_fraction` of the set's mass) and the **housing**
(cartridge and the part of the head casting that warms with it). Boundaries: the ambient air, the jacket
coolant, the machine structure (the structure body of this section, which receives that heat).

Heat at the speed n of the step (the time mean over its ticks):

* **bearings**, per bearing (Palmgren 1959; Harris & Kotzalas 2007 ch. 6): `M = M_v + M_l`,
  `M_v = 1e-7 f0 (nu n)^(2/3) dm^3` N mm (nu mm2/s, n rpm, dm mm; `nu n < 2000`: `160e-7 f0 dm^3`),
  `M_l = f1 F_beta dm`, `f1 = z (P0/C0)^y` (ball bearings; rollers `f1 = z`), P0 the static equivalent load
  (ISO 76), F_beta Palmgren's friction load (angular-contact ball `max(0.9 Fa cot(alpha) - 0.1 Fr, Fr)`, Fa the
  preload); f0, z, y per bearing type and lubrication from `thermal-machine.yaml`. nu is the lubricant's
  kinematic viscosity at the **bearing node's** temperature (Walther's relation, ASTM D341, through its 40
  and 100 degC values), so a cold machine (thick oil) makes more heat and grows more, and the friction falls
  as the bearings warm. Heat `M omega`: half enters the rolling elements (Blok's equal partition at a rolling
  contact of like bodies), the other half the rings, `bearing_heat_inner_ring_fraction` of it the inner ring
  (shaft), the rest the outer ring (housing).
* **motor**: no-load loss `r P_rated min(1, n / n_base)` (P_rated: `drive_power_kW`, else the S1 rating or
  the main power curve; n_base where that curve reaches 95 % of its maximum); built-in motors put
  `motor_rotor_loss_fraction` into the shaft and the rest into the housing (Bossmanns & Tu 2001), separate
  motors `motor_heat_to_housing_fraction` (coupled, gear 0.3, belt 0.15) into the housing. Load losses
  (copper, proportional to the torque squared) are not included (the domain passes speeds only).
* **holder**: the heat the tool holder conducts through the taper (tool ladder) enters the shaft.

Links: bearing node to shaft and to housing through the rolling contacts, per bearing and side
`G_ref (dm / dm_ref) sqrt(omega / omega_ref)` (Hertzian contacts with the moving-source enhancement of
Jaeger 1942); shaft to housing through the air annulus between the bearing sets (rotating cylinder, Kays &
Bjorklund); shaft to air at the exposed nose (rotating cylinder plus radiation); housing to air (vertical
plate, Churchill & Chu, plus radiation at the paint's emissivity 0.9) over its exterior; housing to the
jacket coolant `G = m c_p (1 - exp(-h A / (m c_p)))` (h of pipe flow in the channels, the inlet at the
ambient plus an offset unless stated); housing to the structure `default_housing_to_structure_W_K` (guide
carriages and nut). Shaft and housing capacities from their dimensions and masses (steel, cast iron).

Integration: sub-steps of at most `spindle_network_max_step_s` (20 s); in each, the convection coefficients
at the start temperatures and the bearing heat linearised in its node temperature
(`P = P0 + dP/dT (T - T0)`, dP/dT < 0 through the viscosity: it stiffens the system), then the linear system
`C dT/dt = P - K T + G_b T_b` integrated exactly for constant sources and boundary temperatures through the
symmetric eigen-decomposition of `C^-1/2 K C^-1/2`; every loss is the exact integral of its link, so the
ledger `spindle.<id>` closes to round-off (1e-15 measured). First order in the sub-step: a 1 h warm-up at
6000 rpm is 0.14 % from the converged growth at 20 s (test). The domain steps a network every
`spindle_every_n_macro` macro steps (time, speeds and holder heat accumulated exactly in between; flushed at
operation ends and `finalize`; the pending part is in the checkpoint).

Growth toward the work: `eps_s(T_shaft) L_n + eps_h(T_housing) L_h`, the free strains of the mean-CTE tables
from the reference (ambient at the start), L_n = nose to the axially fixed bearing set
(`nose_to_fixed_bearing_mm`), L_h = housing length from the head's axial datum (the Z nut of a vertical head,
the bed mounting of a headstock) to that set (`housing_growth_length_mm`).

Parameters: the machine file (`drive_type`, `bearings`, `thermal.*`, docs/formats.md "Kind notes"); absent
keys take the 40-taper defaults of `thermal-machine.yaml` scaled with the front bearing bore (from
`bearings`, `bearing_id_mm` or the nose's taper size): mass and areas with (d / 70 mm)^3 and ^2, lengths with
d. The shipped machines carry estimated bearing arrangements (noted in their files). Cost: 0.33 ms per
network step (one core).

Not modelled: the radial thermal expansion of the rings (thermally induced preload change), centrifugal and
gyroscopic ball loads at high speed, the axial shift of angular-contact contacts, the head's temperature
gradient (one housing node), forced convection by the motor's fan, the test mandrel or tool of a warm-up test
(the tool ladder holds the run's tool).

## 5. Ledgers (architecture 12)

Solvers: `tool.<slot>` (the ladder, or the cutting-region field and the ladder nodes it does not
replace), `work.<w>`, `fixture.<w>`, `spindle.<id>`, `axis.<name>`, `structure`.
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
radial/length/holder growth; **added** `edge_peak_q_K`, `body_temp_at_elem_K`; the edge temperatures of
the elements a cutting-region field carries come from it, 2.4), `spindle_temp_K`,
`spindle_growth_M_m`, `axis_drift`, `work_temp_origin_S_m`, `work_temp_cell_m`, `work_temp_K` (a
new array at every finite-volume step, NaN outside material, never modified afterwards),
`work_surface_temp_K`, `work_preheat_K` (max over the workpiece's contacts), `work_bulk_mean_K`,
`work_bulk_max_K`, `heat_to_coolant_J`, `absorbed_J` (per solver dU of the step). **Added** fields:
`heat_to_chip_J`, `heat_to_tool_J`, `heat_to_work_J` (per contact), `unplaced_heat_J`,
`pending_heat_J`, `contact_surface_temp_K`, `contact_preheat_K`, `spindle_growth_abs_M_m`,
`axis_drift_strain`, `axis_drift_ref_m`, `structure_drift_M_m`, `fixture_temp_K`,
`work_unload_mean_K`, `ledger_residual_rel`. `finalize()["spindles"][id]` adds `model` (network | lumped)
and, for a network, `node_temps_K`, `growth_parts_m` (shaft, housing) and `energy_J` (bearing, motor and
holder heat, losses to air, structure, jacket).

## 7. Settings (`physics.thermal`, `schema.py`)

`every_n_macro` 5, `cell_mm` 2, `tool_cell_mm` 1, `edge_quadrature` 3, `edge_every_n_macro` 2,
`sa_direct_cycles` 4, `window_points` 3, `jaeger_points` 24, `min_fraction` 1e-9,
`deposit_search_cells` 3, `cooldown_steps` 24, `idle_tool_steps` 8, `energy_tol` 1e-9; cutting-region
field (2.4): `tool_field` 1, `tool_field_ring_mm` 0.002, `tool_field_growth` 1.4, `tool_field_sector_deg`
15, `tool_field_dy_mm` 0.5, `tool_field_length_mm` 12, `tool_field_every_n_macro` 10, `tool_field_tol`
1e-12, `tool_field_max_iter` 5000; `spindle_every_n_macro` 10 (spindle networks, 4.1).

## 8. Cost (measured, one core of the i5-4690, bin/tsafe)

| part | cost |
|---|---|
| workpiece finite-volume step | 150-225 ns per material cell (5.5 ms for 26k cells; 14 ms for 63k; 74 ms for 490k) |
| edge kernel | 9 us per in-cut element and evaluation (every `edge_every_n_macro` steps) |
| domain macro step, one cutting tool (40 in-cut elements), small part | 1.9 ms (Python plumbing dominates) |
| cutting-region field, 12.7 x 4.76 mm insert (4.1k cells) | 5.5 / 8.6 / 10 / 12.6 ms per solve at 10 / 50 / 100 / 250 ms steps (10-32 iterations) |
| cutting-region field, D10 4-flute end mill (1.1k cells) | domain macro step 1.70 -> 2.05 ms (mean, solve every 10 steps) |
| response tables | 0.6 s once per process |
| kernel compilation (cold cache) | ~25 s once per cache key |

Per simulated second at 10 ms macro steps: tool, machine and contact work ~0.19 s, plus the
finite volumes `cells x 190 ns / (every_n_macro x 10 ms)`, plus the cutting-region field: ~0.1 s per
turning tool, ~0.035 s per solid tool in cut (at `tool_field_every_n_macro` 10; set 0 in
`tool_field` to drop it). For the reference part (300 x 200 x
100 mm, ~490k material cells at 2 mm) choose `cell_mm: 4` (63k cells: 0.28 s/s at every 5) or
`every_n_macro: 20` (0.37 s/s at 2 mm): the two-scale contact temperature covers the sub-cell
and sub-step scales, so a coarser field changes the bulk only.

## 9. References

Jaeger 1942 (moving sources); Carslaw & Jaeger 1959 (conduction, disc and rectangle sources; quasi-steady
slab and cylinder under a constant flux, ch. III and VII; sources and images, ch. X); Hestenes & Stiefel
1952 (conjugate gradients, J. Res. NBS 49(6) 409-436); Saad 2003 (Iterative Methods for Sparse Linear
Systems, 2nd ed., SIAM: block preconditioners);
Stephenson & Ali 1992 (interrupted cutting); Blok 1937 (flash temperature, partition);
Boothroyd 1963; Trent & Wright 2000; Astakhov 2006 (Poletika contact length); Komanduri & Hou
2000; Douglas & Gunn 1964 (ADI, rejected); Churchill & Chu 1975a,b; Incropera et al. 2007;
Kays & Bjorklund 1958; Dittus & Boelter 1930; White 2006 (Sutherland); Palmgren 1959,
Harris & Kotzalas 2007 (bearing friction); ISO 76 and ISO 281 (bearing load factors); ASTM D341 (Walther's
viscosity relation); Bossmanns & Tu 1999, 2001, Holkup et al. 2010 (spindle thermal networks, motor loss
split); IEC 60034-2-1 (motor loss segregation); Madhusudana 2014 (contact conductance); Bryan 1990,
Mayr et al. 2012 (machine thermal errors); ISO 7388-1, ISO 12164-1 (tapers); Abramowitz & Stegun
1964 (Gauss-Legendre). Full entries: `db/bibliography/models-thermal.yaml` (Hestenes & Stiefel 1952 and
Saad 2003 are not in it yet).
