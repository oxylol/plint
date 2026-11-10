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
| `surfacetemp.py` | temperature history of the machined surface under the edge (shear-zone, edge-arc, ploughed-layer and wear-land sources of the work share) |
| `damage.py` | white layer (austenite start at the heating rate, minimum depth), over-tempering and burn colours (Hollomon-Jaffe tempering parameter), melt / smear (solidus) on that history -> `thermal_damage` findings |
| `kernels.py` | Numba kernels (one signature each); `reference.py` their plain-Python references |
| `response.py` | dimensionless step-response tables of a heated rectangle (edge model) |
| `convection.py` | free convection, rotating cylinder, pipe flow, radiation |
| `props.py`, `inputs.py` | properties from the databases; robust reading of project/machine files |
| `protocols.py` | `WorkGeometry`, `ToolEdges`, `FixtureContact` (wired by the integration) |
| `state.py`, `schema.py`, `channels.py`, `domain.py`, `ledger.py` | domain plumbing |

Model coefficients and their sources: section 7. RNG streams: **none** (no stochastic thermal phenomena).

## 1. Heat distribution (single owner, architecture 12)

Per contact the hot loop reports the generation by zone: shear `Qs`, rake (secondary) `Qr`, flank
(tertiary) `Qf` and ploughing `Qp` (J in the step). ProcessCoefficients give per element group
and chip thickness the fractions `f_sc` (shear -> chip), `f_rt` (rake -> tool), `f_ft`
(flank -> tool). Thermal splits once:

    chip = sum_e Qs_e f_sc + Qr_e (1 - f_rt)
    tool = sum_e Qr_e f_rt + (Qf_e + Qp_e) f_ft          (per element, into the tool ladder)
    work = sum_e Qs_e (1 - f_sc) + (Qf_e + Qp_e)(1 - f_ft)

* **Ploughing** (rubbing below h_min, edge forces) happens at the edge-flank contact and takes the
  flank fraction. `f_ft` is the share of the flank contact's whole heat that enters the tool: the friction on the
  wear land and on the edge arc below the stagnation point and the plastic work of the material pressed under the
  edge, from Blok's equal mean temperatures with the plastic work as a heat source inside the work
  (docs/models/mechanics.md 6). Thermal deposits the tool's part over the contact the share was computed for, per
  element `l_f = VB + r arccos(1 - d / r)` with `d = min(h, h_min of the element's group)` (the arc below the
  stagnation point, `mechanics.contact.stagnation_arc_length`, continued along the rake face beyond its tangency;
  `ContactHeat.flank_length_m`). Until 2026-11-09 mechanics computed the share for the land friction alone and
  thermal applied it to the whole edge power over `VB + 1.0 r_e`: the flank contact came out about twice Blok's
  value and rose when the tool was cooled (the friction share grows as the tool cools). Without a table the
  literature fallback keeps `VB + flank_contact_edge_factor r_e`.
* **Element shares** of a zone (`Q*_e`) follow the per-element integrals of the step: shear ~
  `int_h_dt * db * r` (removed volume x speed; r the element's radius from the tool axis for a rotating tool,
  1 for a turning tool, whose cutting speed is the work's surface speed along the whole edge to within ap / r and
  whose `elem_r_m` is its x_T coordinate, <= 0: until 2026-11-09 those negative weights sent the shear heat to an
  even split over every valid element and the idle elements' share, at the table's thinnest chip, into the work:
  75 % of warmup's turning pass heat went into the bar, now 19 %, with 75 % in the chip and 5 % in the insert),
  rake ~ `int_sigma_v_rake * h zeta^n * db` (normal
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
ratio, clipped to `max_contact_ratio`, at least the edge radius), `l_f` the flank contact of the partition
(section 1: the wear land plus the edge arc below the stagnation point; `VB + flank_contact_edge_factor r_e`
without a table).

**Heat history in run time (2026-11-09).** An element's periodic heating is reconstructed from its own history, not
from the counts of one macro step. It is engaged from its first pass until it misses one (no in-cut time for longer
than one period after the step that held its last pass); while engaged it keeps running sums of its steps - tool
heat (rake, flank), in-cut time, passes, heat-weighted contact sizes - that forget with the time constant
`edge_window_periods` (4) periods in run time. From them: the fluxes while in cut `q_r = sum Q_rake,tool / (l_c db
sum t_in)` and `q_f` likewise, the heating time per pass `t_h = sum t_in / sum passes` (for a rotating tool over the
teeth at the element's position along the edge, which pass the same arc: a pass across a step boundary has its entry
in one step and part of its in-cut time in the next, and the teeth's boundary passes fall in different steps), the
period `T = 2 pi / |omega|` for a rotating tool (each element passes the cut once per revolution) and the mean time
between passes otherwise, and `N` = passes since the engagement began. Interrupted: at least two passes; continuous: one entry, in cut since.
The rise at phase `tau` of the current cycle is the superposition

    theta(tau) = q sum_{n=0}^{N-1} [F(tau + nT) - F(tau + nT - t_h)]

the first `sa_direct_cycles` (4) term by term, the rest by the midpoint integral
`(W(t_hi) - W(t_lo)) / T` with `W(t) = int_{t-t_h}^t F` (Gauss-Legendre, `window_points`).
Continuous elements (turning, slotting): `theta = q F(t_eng)`, t_eng the contact time. Until 2026-11-09 the inputs
were the step's: `T = dt / n_entries`, `t_h = t_in / n_entries`, and an element whose step held no pass restarted
its cycle count and published its body temperature (or, with the field, the cycle mean) as its entry temperature:
with a 10 000 rpm end mill (a 6 ms revolution) the period came out 5 ms at 5 ms steps, 5 or 10 ms at 10 ms and 5 or
6.7 ms at 20 ms, about one 5 ms step in six held no pass of an element, and the rake temperature at the entry
alternated from step to step (+-50 K at 5 ms; docs/convergence.md 5 and 9 item 8). A step without a pass now leaves
the element's engagement and its inputs unchanged (test `test_edge_history.py`: the entry and in-cut rises at 5, 10
and 20 ms agree within 2 %, with and without the cutting-region field). On that test's interrupted cut (a 6 ms
revolution, 0.43 ms passes) the entry rise at 5, 10 and 20 ms steps is 1.09, 1.09, 1.09 K with every tooth alike
(the former reconstruction: 1.04, 1.45, 1.45 K, the teeth 0 to 1.45 K, alternating by 1.1 K from step to step at
5 ms), with the cutting-region field 5.24, 5.25, 5.24 K (formerly, with its cadence of 10 macro steps, 7.3, 8.2,
7.6 K, the teeth -2.4 to 8.2 K, alternating by 7 K at 5 ms). From the semi-infinite field the
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
both sources), `edge_cycle_amp_K` = edge-line temperature at the end of heating minus at the end of cooling,
and the **entry temperatures** `rake_temp_mean_K` / `flank_entry_K`: the rake and flank contact means at the end
of the cooling interval, when the edge enters the cut again (what is left of the earlier engagements on the body
temperature; a continuous element's own steady rise excluded: its entry value is the body temperature),
`rake_contact_mean_K` the in-cut mean (the q-weighted `edge_temp_q_K`).

**The coefficient pipeline's tool temperature (2026-11-09).** cnctwin.mechanics evaluates the rake heat
partition inside ProcessCoefficients with the tool rake temperature T_tool as the background of its
stationary-source tool side: `T_w + dT_SZ + psi R dT_M = T_tool + (1 - R) q M_t` (docs/models/mechanics.md):
the rise the engagement's own heat makes in the tool, `(1 - R) q M_t` with the transient cap of one contact
time, is mechanics', solved with the forces as one consistent state (architecture 5.3: the fast
thermo-mechanical coupling inside ProcessCoefficients). `rake_temp_mean_K` was the in-cut mean of the
quadrature temperatures, which already holds that rise from the previous step's heat: the partition counted it
twice, and through the one-step stagger the loop closed explicitly with a gain near -1 (a hotter tool -> a larger
chip share R -> less heat into the tool -> a colder tool next step): the tiny project's rake temperature
alternated by 20-37 K from one macro step to the next (loop.coupling_lag, cutting's own oscillation check). The
entry temperature is the background the partition expects; it changes with the accumulated heat (the residual of
earlier cycles, the body), a fraction of the in-cut rise, so the staggered loop gain falls well below the
alternation limit. `flank_entry_K` is the flank's equivalent for mechanics' flank partition (cutting's row
state reads it, and `flank_temp_q_K` only from a record without it: docs/models/loop.md 15); wear and coolant read
the in-cut temperatures (`flank_temp_q_K`, `rake_contact_mean_K`). A coefficient row stands for a whole edge group
(the peripheral edge of a helical end mill: the whole flute) while a pass engages a few of its elements: since
2026-11-09 cutting's row state takes the members' temperatures (and VB and BUE height) weighted by the in-cut time of
the step at their position along the edge, summed over the teeth (cnctwin/cutting/domain.py `_row_state`; the plain
mean over the members when none cut; summed over the teeth so that the weights do not follow how many passes of each
tooth a step held). The plain mean was dominated by the idle members: in a 1 mm deep side step about 2 of 44
elements per tooth cut, so the rows saw the body temperature and a twentieth of the contact's rise. The contact time
of one engagement that mechanics' transient cap reads is, for a rotating tool, removal's engaged arc over the
spindle speed (the step's in-cut time over its entries moved by up to a third at 5 ms steps with the passes that
cross the step's ends). With the cutting-region field the entry values of its elements are
the field's cycle-mean strip temperatures plus the edge kernel's deviation at the entry (`EDGE_OUT_DEVIATION`:
`rake_entry` / `flank_entry` of the kernel). A continuous element (turning, slotting) has no cycle: its strip mean
is its in-cut temperature and holds the steady rise of its own contact heat, which mechanics' stationary-source
partitions add again. Since 2026-11-10 its entry values are the strip means less that rise
(`ToolThermal._continuous_own_rise`): the edge kernel's continuous branch at the element's fluxes, contact lengths
and contact time over the width the contact has (the engaged edge of the element's tooth, the width the
coefficient rows' stationary-source tool side takes from cutting's `engaged_edge_m`) and with nothing removed for a
far field (the field holds all of it), and not below the body temperature at the element (the rest of the field, the
body and every other source, all heating, is not colder): the field is solved at intervals (100 ms) and has not yet
taken up a new engagement's heat when the analytic rise already has it, so without the bound the entry of the
aluminium side step's first contact fell to 109 K, mechanics put far too much heat into the tool and the flank
spiked to 810-950 K. Until then the entry of a continuous field element was its in-cut mean, the
double count the interrupted elements had before the entry temperature was introduced (test
`test_edge_history.py::test_a_continuous_field_element_enters_at_the_field_less_its_own_rise`).
`edge_interval_ms` (default 20): the edge field is evaluated at the first macro step ending at or after every
multiple of it in run time (the same instants whatever the macro step); between evaluations the last local rise is
kept on top of the current body temperature for the engaged elements (the field is quasi-steady on the scale of a
macro step; the running sums and N advance every step). `edge_interval_ms: 0` restores the former cadence of every
`edge_every_n_macro` (2) macro steps.

### 2.3 Validity and error

* `l_c, l_f, db << r_e` (quarter-space approximation of the edge region; error ~ l / r).
* Local field established within `l^2 / alpha` (1-4 ms for carbide at l = 0.1-0.3 mm): the
  quasi-periodic field is accurate after the first macro step of an engagement; for HSS and large
  contacts (l^2/alpha up to ~40 ms) the first steps of an engagement are overestimated. An element
  that misses a pass (no in-cut time for more than one period) restarts its cycle count and running sums
  (cooling over l^2/alpha assumed complete); the running sums lag a change of the cut by about
  `edge_window_periods` periods (24 ms at 10 000 rpm).
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

**Cadence.** In a run the field and the ladder are solved at the first macro step ending at or after every multiple
of `tool_field_interval_ms` (100 ms) in run time, over the accumulated time (`tool_field_interval_ms: 0`: every
`tool_field_every_n_macro` (10) macro steps, the former cadence, which put the solves 50, 100 or 200 ms apart at 5,
10 or 20 ms steps); the heat of the steps in between is accumulated per element exactly
(energy and heat-weighted contact sizes) and mapped once per solve; idle steps, operation ends and
`finalize` solve what is pending. The teeth's split of the representative tooth's rise uses their running sums'
heat per unit in-cut time (the teeth at one position along the edge pass the same arc; a step shorter than a
revolution holds the passes of some teeth only, and a pass across a step boundary has its entry in one step and part
of its heat in the next, so neither the step's heat nor the heat per entry is a steady split). The near-edge field is quasi-steady on this scale (l^2 / alpha = 1-10 ms)
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
* `theta_flash`: one edge passage over the new surface, the peak of the machined-surface history of 3.4 (the work's
  shares of the shear-zone, edge and wear-land heat as moving sources over their own contacts at the cutting speed);
  its energy is part of theta_feed, so its cell mean is not subtracted. Contacts without the per-zone split (callers
  that build a `ContactSource` by hand) keep the former band: the per-tooth flux `work_J / (L_s b t_cut)` on
  `L_s = VB + r_edge + h / tan(phi)`.
* **Preheat** (material entering the cut, what cutting's coefficients read as the work temperature; 2026-11-09):
  the temperature at the leading edge of the feed source from two scales that each converge,

      T_pre = T_FV(x_D) + theta_pend(x_D) - theta_eng(x_D) + theta_eng(lead)

  `theta_eng`: the feed source on since the contact's engagement began, carrying the contact's work-bound heat rate
  averaged over `edge_window_periods` revolutions (a step holds a whole number of passes only on average: 3 or 4 of a
  4-flute tool at 10 000 rpm in a 5 ms step; its half length and width are heat-weighted running means of removal's
  contact geometry, which moves with the passes a step held, and where removal gives a step no contact point (a zero
  point) the tool tip plus the last offset of the point from it stands in), `theta_pend`: the same over the heat not yet
  in the finite volumes (since their last step); `x_D` lies one tool diameter (insert inscribed circle for turning)
  beyond the leading edge along the feed, in the material the tool reaches next, where the grid resolves the field:
  `T_FV(x_D)` is interpolated between cell centres with trilinear weights times the material fractions
  (`WorkThermal.temperature_at`; the point moves back toward the leading edge cell by cell while it is outside the
  material). The source's own near field, which no cell can hold, is taken off there and put back analytically at
  the leading edge. The former definition, `T_FV` of the cell one feed half length plus one cell ahead plus the
  source's rise over the window since the last finite-volume step, sat inside that near field: its cell value
  doubled with each halving of the cell (docs/convergence.md 6) and its window moved with the cadence. Test
  `test_preheat.py`: a 15 W band at 20 mm/s on aluminium, 14.08-14.10 K at cells of 4, 2, 1 and 0.5 mm (the
  former cell ahead 1.5, 3.6, 4.6, 6.2 K). Error: the band is the half-space solution (at a convex edge of the part,
  a side step at the top of a wall, the near field is up to twice that); grid-independent once the cell is below
  about a third of the tool diameter.

Validity: half-space approximation of the part near the contact (error grows when sqrt(alpha t)
of the window approaches the distance to other free surfaces: < 1 mm for steel at 50 ms, 1.9 mm
for aluminium); flat generated surface; the edge passage resolved as the shear-zone, edge-arc, ploughed-layer and wear-land
sources of 3.4 (the shear plane as its footprint on the cut plane, not the inclined source of Komanduri & Hou 2000). The two-scale peak is an estimate within about +-30 % (partition and
contact size dominate), adequate for threshold findings with that margin.

Published values (`test_jaeger.py`): high-Peclet band mean `0.754 q l/(k sqrt L)` and maximum
`1.128 q l/(k sqrt L)` (Jaeger 1942) within 3 % at L = 20 and 100; stationary square centre
`1.1222 q a/k` (Carslaw & Jaeger) within 0.5 %; 1-D short-time limit to 1e-6; kernel against
adaptive quadrature within 2e-4.

### 3.4 Machined-surface temperature and thermal damage (2026-11-10)

**What was wrong before.** The damage check compared the two-scale peak `T_FV + theta_feed + theta_flash` with fixed
thresholds, where `theta_flash` put the contact's whole work-bound heat (`work_J`: the shear zone's work share plus
the flank and edge heat's work share) on one band `VB + r_edge + h / tan(phi)` long, with the flux computed for the
engaged edge length but the rectangle given the circumference as its width in turning, and judged white layer,
over-tempering and burn by a threshold crossing at the surface for any duration. The warmup's 4140 pass (132 m/min,
f 0.25 mm) was reported at 912 / 794 degC with white layer, over-tempering and burn.

**Sources** (`surfacetemp.py`). In the frame of the edge (xi along the cutting direction, xi = 0 at the edge, xi < 0
the new surface), the work's share of each zone heat (thermal.partition: `ContactHeat.work_shear_J`, `work_flank_J`,
`work_plough_J`, additive fields) is put where it is generated:

* shear zone, `Q_s (1 - f_sc)` (Boothroyd 1963): uniform band `0 < xi < L_s = h / tan(phi)` ahead of the edge, the
  footprint of the shear plane on the cut plane through which it enters the material below the cut line;
* edge arc: the friction part of the hot loop's edge (ploughing) heat, `(1 - f_ft)` of it, on `-l_a < xi < 0`, l_a the
  arc below the stagnation point (`l_f - VB` of the coefficient tables, heat-weighted);
* ploughed layer: the pressure part (the plastic work of the material pressed under the edge), its work share,
  uniform to the depth `delta = ploughing_zone_depth_ratio x l_a` (mechanics-contact; Hill 1950) under the arc, as
  buried plane sources with their images (adiabatic surface; Carslaw & Jaeger 1959 ch. X), three Gauss points.
  The friction / pressure split is `contact.contact_power_parts` at the elements' hone radius, rake, inclination and
  their group's h_min (heat-weighted); with Prandtl's pressure `(2 + pi) k` and friction `min(mu p, k)` it does not
  depend on k, nor on mu above `1 / (2 + pi)`, where it is evaluated (about a third friction for a honed edge);
* wear land: the hot loop's flank (land) heat, its work share, on `-(l_a + VB) < xi < -l_a` (in-cut mean VB).

Each is Jaeger's rectangle of width b (the engaged edge length per tooth) moving at V over a half space
(`kernels.moving_rect_rise`, any Peclet number), on for one passage (turning: the engagement). Rates are the step's
work shares over the time in cut. A material point at depth z meets them in turn and then cools into the bulk:
`T(z, t) = T_b + theta(xi0 - V t, z)` with `T_b = T_FV + theta_feed - <theta_feed>_cell` (3.3), sampled ahead of the
shear band, across the arc and the land (the maximum of a fast band lies at 0.95-0.98 of its length) and on a
geometric ladder behind, at depths 0, 1 um and `sqrt(alpha t_c) x (0.1 ... 4)`. `theta_flash` (3.3) is the surface
maximum. Properties are taken at `T_b`.

**Criteria** (`damage.py`, parameters `db/models/thermal-damage.yaml`, material `thermal_damage`):

| kind | criterion | data |
|---|---|---|
| white layer (error) | peak at the minimum depth `min_depth_m` (1 um) above the austenite start at the heating rate there (mean rate between 10 % and 90 % of the rise); the start is the data sheet's Ac1 (`white_layer_degC`, slow heating) moving in log(rate) to `white_layer_fast_degC` at `white_layer_fast_rate_kelvin_per_second` where the material gives it, held beyond (no extrapolation) | continuous-heating TTA diagrams (Orlich & Pietrzeniuk 1976; AISI 4140 to 1e4 K/s: Miokovic et al. 2004, 2005); no material carries a verified fast point yet, so the data-sheet Ac1 applies (the lower, conservative value) |
| over-tempering (warning) | Hollomon-Jaffe parameter `P = T (C + log10 t[h])` of the history at the minimum depth, accumulated by additivity (each interval adds its time to the equivalent time of the P so far at its own temperature), the bulk and feed-scale temperature appended for the feed source's dwell `2 l / v_feed` (at most the engagement), above the parameter of the tempering the steel received (`overtemper_degC` for `tempering_hold_min`, default 2 h) | Hollomon & Jaffe 1945, C = 20 (material may override) |
| burn colours (warning) | the same parameter at the surface above that of `burn_degC` held `burn_hold_s` (1 h): the colour table is a tempering-practice table and tempering temperature and time are interchangeable | wikipedia-de-anlassen; hold assumed |
| melt / smear (error) | peak at the minimum depth above the solidus `melt_smear_degC` | material (7075: 477 C, NASA) |

A history is evaluated when the surface peak exceeds the lowest reference temperature of the material (a tempering
parameter cannot exceed the reference while the temperature stays below the reference temperature for less than the
reference time). One `thermal_damage` finding per (operation, workpiece, kind, program block) as before; its
quantities add the criterion value and limit, the affected depth (deepest sampled depth where the criterion holds),
the heating rate and the sources (`shear_work_W`, `arc_work_W`, `plough_work_W`, `land_work_W`, lengths, width, hold);
the trace event `thermal:surface_temperature` is its parent. `ThermalFeedback.contact_damage` (added) carries
`{kind: affected depth}` per judged contact; surface integrity (surface/integrity.py) takes its white-layer /
over-temper / burn / melt flags and the white layer's depth from it instead of its own threshold crossings.

**Results** (tests `tests/unit/thermal/test_surface_damage.py`; the warmup run of 2026-11-10):

* fast band: trailing-edge maximum within 3 % of the 1-D flash `2 q sqrt(t / pi) / e` at L = 22 and 89; the layer
  source tends to the surface source as delta -> 0 and is 10 %+ cooler at the surface for delta = 20 um;
* AISI 52100 62 HRC, CBN, 200 m/min, f 0.1, ap 0.2 mm, VB 0.3 mm (the worn-tool hard-turning case of Chou & Evans
  1999, white layers growing with flank wear): white layer to 37 um, over-tempered (dark) layer, depth more than twice
  the sharp tool's 6 um;
* Chen et al. 2017 (52100 58-62 HRC, PCBN, 100 m/min, ap 0.1 mm on the 0.8 mm nose): measured peak machined surface
  temperatures at the intersection of edge and new surface 500-800 C rising with feed; the chain (coefficient
  pipeline -> partition -> sources) gives 690 / 782 / 796 C at the edge for f 0.05 / 0.1 / 0.15 mm (fresh edge), in
  range and rising with feed. The mean over the edge contact behind the edge is 1050-1160 C, 250-400 K above the
  measured range: the edge heat entering the work is about 35 W on a 16 um x 0.45 mm arc (5 GW/m2), where Chen's
  inverse analysis found flank fluxes of 0.13-0.73 GW/m2;
* the warmup's 4140 pass (132 m/min, f 0.25, ap 1.5, 40 um hone, initial VB 50 um; `twin warmup --only lathe`):
  over-tempering is no longer reported (P 15 400 / 16 600 K at 1 um against 17 700 K for 600 C / 2 h); a white
  layer and burn colours still are: surface peaks 1138 / 1253 C, 1110 / 1219 C at 1 um, above Ac1 to 9 / 13 um, at
  heating rates of 1.3e7 / 7e6 K/s. The sources at the second finding: shear 146 W over 0.37 mm, edge arc 95 W
  (friction) + 182 W (plastic work to 20 um) over 28 um, land 85 W over 67 um, 2.1 mm wide, 1.98 m/s: the arc carries
  4.7 GW/m2, from edge forces of about 70 N/mm. The same chain through the harness (`test_warmup_od_pass_on_4140_is_not_
  damaged`, xfail) gives 940 C at 1 um. That is an input of the cutting / mechanics domains (the edge force's flow
  stress and its share in the work), see the open problem below.

**Validity and error.** Half space, flat new surface, constant properties at `T_b` (k of steel falls 30-40 % to
800 C: the peak is underestimated by up to that share of the rise); sources uniform over their contacts (the
trailing-edge maximum of a non-uniform land is within the same order); the tempering parameter is extrapolated from
hour-scale tempering to 1e-5-1e-1 s (C = 20 +- 2 moves the equivalent temperature by about 10 %); the austenite start
is the slow-heating Ac1 where no fast point is given (white layers flagged at temperatures where rapid heating would
still need 40-150 K more: the criterion errs toward reporting); the minimum depth is an assumed detectability limit.
The history is as good as its sources: with the coefficient pipeline's edge heat it overestimates the edge-contact
mean against Chen et al. by 250-400 K (above); the temperature at the edge itself agrees within 30-100 K.

**Open problem.** The edge (ploughing) heat that enters the work dominates the new surface's peak in every case
checked (warmup 4140, Chen's fresh PCBN edge): about 70 N/mm (4140, 40 um hone) and 50 N/mm (52100, 25 um) times V,
86-95 % of it into the work, on the arc below the stagnation point. Its flow stress is the primary zone's (k_AB at the
strain cap) while the arc and the land run several hundred kelvin hotter; a land or arc stress softened at the
contact temperature (Johnson-Cook's thermal term) would lower the heat where it is hottest (the worn 52100 case's
3000 C surface peak, above the melting point, shows the missing feedback). Until that is resolved the warmup's
4140 pass is reported with a white layer (`test_warmup_od_pass_on_4140_is_not_damaged`, xfail strict).

## 4. Machine

Axes: every mapping of the machine file's `axes` block is an axis (the block's own `src` / `note` entries are not:
haas-ec400's `axes.src` crashed the set-up of examples/ti-slot-throughtool before 2026-11-09).

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
  `(structure_sensitivity_um_per_K + s_sp) x (T - T0)` along `+axis_T_M` (warmer column lifts the head);
  T0 = start of the operation. The file's sensitivity is the tool-to-work drift of the whole loop per K of a
  uniform temperature change, the spindle head included (the shipped files' notes: "the loop from the pallet
  through the base and column to the spindle nose"); the spindle model expands with its own temperatures, so
  the structure body carries the sensitivity plus the uniform growth per K of the spindle in its loop
  (`s_sp = d growth / dT` of the network, eps_s L_n + eps_h L_h at 1 K, or alpha L_eff of the lumped body;
  `machine.link_structure`): a uniform warming of head and structure then drifts by the file's sensitivity.
  Until 2026-11-09 the head's share was counted twice (once in the spindle growth, once in the sensitivity):
  for a 40-taper network (s_sp 4.7 um/K) a structure warmed 1 K by the head and the ambient put 4.7 um too much
  toward the work. The loop spindle is the machine's first milling spindle; lathes without one share nothing
  (a turning spindle grows along the work axis, across the radial loop of the sensitivity).
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
radial/length/holder growth; **added** `edge_peak_q_K`, `body_temp_at_elem_K`, `flank_entry_K`,
`rake_contact_mean_K` (2.2: `rake_temp_mean_K` is the rake contact's entry temperature); the edge temperatures of
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

## 7. Parameters and settings

Model coefficients: `db/models/thermal-{convection,tool,work,machine,partition}.yaml`, sources in
`db/bibliography/models-thermal.yaml`; material properties (k(T), cp(T), mean CTE, emissivity family,
thermal-damage thresholds) from the material and tool databases. Machine-file values replace the
`thermal-machine` defaults; every default used is listed in `finalize()["estimated"]`.

| file | parameters (SI) | source |
|---|---|---|
| thermal-convection | vertical plates 0.825, 0.387, Pr 0.492; horizontal cylinders 0.6, 0.387, Pr 0.559 | churchill-chu-1975-vertical-plate, churchill-chu-1975-horizontal-cylinder |
| | horizontal plates 0.54 Ra^1/4 / 0.15 Ra^1/3 (switch 1e7) up, 0.27 Ra^1/4 down; laminar pipe Nu 4.36, transition Re 2300 | incropera-2007-heat-mass-transfer |
| | rotating cylinder 0.133 Re^(2/3) Pr^(1/3), Re <= 4.3e5 | kays-bjorklund-1958-rotating-cylinder |
| | Dittus-Boelter 0.023 Re^0.8 Pr^0.4 | dittus-boelter-1930-tubes |
| | air: Sutherland constants at 273.15 K | white-2006-viscous-fluid-flow |
| | emissivity by material family (aluminium 0.08, steels 0.35, stainless, titanium, nickel 0.3, cast iron 0.45, polymers 0.9, default 0.3) | est |
| thermal-tool | Poletika exponent 1.5, default compression ratio 2.5, flank contact edge factor 1.0, fluted solid fraction 0.65 and perimeter factor 1.6, shank and taper contact conductances 5000 and 3000 W/m2K, holder proportions, insert and bar defaults (12.7 x 4.76 mm, bar 25 mm, 30 mm overhang), wedge angle limits 30-180 deg | est |
| | steep tapers 30/40/50, HSK 40-100 (gauge diameters and lengths) | iso-7388-1-2007-steep-taper, iso-12164-1-2001-hsk |
| thermal-work | fixture contact 3000 W/m2K, default fixture 20 kg of 1045 steel, area factor 8, surface h 8 W/m2K, default shear angle 30 deg (flash band), near-cut coolant radius 1.5 tool diameters, unload band 1 K | est |
| thermal-damage | minimum affected depth 1 um, Hollomon-Jaffe C 20, default tempering hold 2 h, burn hold 1 h, slow heating rate of the data-sheet Ac1 0.05 K/s, heating-rate band 10-90 % | est; hollomon-jaffe-1945-tempering |
| thermal-partition | literature fallback when cutting publishes no table: shear to chip 0.85, rake to tool 0.15, flank to tool 0.4 | est |
| thermal-machine | Palmgren M_v = 1e-7 f0 (nu n)^(2/3) dm^3 (160e-7 f0 dm^3 below nu n 2000), lumped bearing heat ~ n^(5/3) | palmgren-1959-bearings |
| | f0, f1 = z (P0/C0)^y per bearing type and lubrication, F_beta factors 0.9 / 0.1 / 3 | harris-kotzalas-2007-bearings |
| | static equivalent load factors X0, Y0 | iso-76-2006-static-load-ratings, iso-281-2007-dynamic-load-ratings |
| | Walther's relation, -20 to 150 degC | astm-d341-2020-viscosity-temperature |
| | half the bearing heat into the rolling elements | blok-1937-flash-temperature |
| | housing paint emissivity 0.9 | incropera-2007-heat-mass-transfer |
| | motor no-load loss 0.03 of rated power; rotor share 0.33 | iec-60034-2-1-2014-motor-losses, bossmanns-tu-2001-power-flow |
| | 40-taper spindle defaults (bearing arrangement, bores, masses, areas, lubricant viscosities, jacket), lumped spindle (growth 20 um, 25 min, 400 W, L_eff 0.25 m), compensation residual 0.3, housing to structure 8 W/K, warm-up at 0.5 of max speed, structure time constant 4 h, ballscrew 50 J per m of travel and 40 min (69 entries) | est (ranges in the file) |

Numerical settings (`physics.thermal`, `schema.py`):
`every_n_macro` 5, `cell_mm` 2, `tool_cell_mm` 1, `edge_quadrature` 3, `edge_every_n_macro` 2,
`edge_interval_ms` 20 (0: every `edge_every_n_macro` steps), `edge_window_periods` 4,
`sa_direct_cycles` 4, `window_points` 3, `jaeger_points` 24, `min_fraction` 1e-9,
`deposit_search_cells` 3, `cooldown_steps` 24, `idle_tool_steps` 8, `energy_tol` 1e-9; cutting-region
field (2.4): `tool_field` 1, `tool_field_ring_mm` 0.002, `tool_field_growth` 1.4, `tool_field_sector_deg`
15, `tool_field_dy_mm` 0.5, `tool_field_length_mm` 12, `tool_field_every_n_macro` 10, `tool_field_interval_ms` 100
(0: every `tool_field_every_n_macro` steps), `tool_field_tol` 1e-12, `tool_field_max_iter` 5000;
`spindle_every_n_macro` 10 (spindle networks, 4.1).

## 7a. Assumptions and their consequences

* **One split of the zone heat** (section 1) by the partition fractions cutting published one step earlier
  (literature fallback 0.85 / 0.15 / 0.4 without a table), shares per element from the step's integrals with
  a uniform friction coefficient along the edge. Consequence: chip + tool + work = generation to round-off;
  the fractions and the contact length (each about +-30 %) dominate the tool temperature: +-20-30 % on the
  local rise (2.3).
* **Tool body as a 1-D axial ladder** (cross-section means) with the edge region as a quarter space whose
  faces are adiabatic away from the contacts and without convection at the edge. Consequence: radial
  gradients in shank and holder are not resolved (Biot number below 0.01); the edge field is off by ~ l / r
  when the contacts are not small against the edge radius, and convection there would change it by less
  than 5 % for h < 1e4 W/m2K (2.3).
* **Quasi-periodic edge field** established within `l^2 / alpha`. Consequence: accurate after the first macro
  step of an engagement for carbide (1-4 ms); for HSS and large contacts (up to ~40 ms) the first steps of an
  engagement are overestimated; running sums lag a change of the cut by `edge_window_periods` periods.
* **Workpiece on a uniform cut-cell grid** (`cell_mm` 2) stepped by first-order fractional steps every
  `every_n_macro` (5) macro steps. Consequence: the steady state with sources differs by O(dt); exposed areas
  of oblique cut cells carry a voxel error up to sqrt 3 except at 45 deg; walls thinner than a cell are shells
  (valid for Bi < 0.1); the bulk, hottest-cell and contact-surface temperatures move with the cell and the
  cadence (docs/convergence.md 6) while the contact's local temperature comes from the two-scale model.
* **Two-scale contact temperature** on a half space with a flat generated surface; the edge passage as the sources
  of 3.4.
  Consequence: an estimate within about +-30 %; at a convex edge of the part the near field is up to twice the
  half-space value (3.3).
* **Fixture as one lumped body**; without a wired `FixtureContact` the part rests on the faces most aligned
  with gravity on its lowest material layer. Consequence: vise jaws are not inferred, so heat conducted into
  jaws through the part's side faces is missed.
* **Free convection per face orientation** at the previous step's mean exposed temperature, radiation
  linearised at the cell temperature, coolant coefficients over the reached share only. Consequence: the
  boundary coefficients lag the surface temperature by one finite-volume step.
* **Removed material leaves with its cell's mean enthalpy**. Consequence: the chip's own heat is the
  partition's chip share (single owner); the cell mean only decides what leaves the part's ledger.
* **Machine bodies are first-order** (axes, structure) and the spindle a network with one housing node,
  no load (copper) losses, no thermally induced preload change. Consequence: spindle growth +-50 % with
  estimated bearings and housing, +-30 % with a described spindle, the lumped body +-30 %, axis drift +-50 %
  (the screw's temperature profile is not resolved), structure drift as good as the machine file's
  sensitivity (section 4).
* **Machined-surface history from moving sources** (3.4): the work's zone shares on their own contacts over a half
  space at constant properties, a material point's history sampled at fixed positions and depths. Consequence: the
  history is as good as the partition and the zone heats (the edge heat dominates, 3.4 open problem); the peak is
  low by up to the conductivity's fall with temperature (30-40 % of the rise for steel at 800 C); sampling of the
  maximum within 2 %.
* **Damage criteria** (3.4): white layer by the austenite start at the heating rate (the slow Ac1 where the material
  gives no fast point) at a minimum depth of 1 um; over-tempering and burn colours by the Hollomon-Jaffe parameter
  with additivity (C = 20, burn hold 1 h and tempering hold 2 h assumed where not given); melt at the solidus.
  Consequence: a white layer may be reported where rapid heating would still need 40-150 K more (errs toward
  reporting); the tempering parameter is extrapolated from hours to microseconds; damage thinner than 1 um is not
  reported; the reported depth is the deepest sampled depth (a ladder of factor 2).
* **Grids stay in their build frame** after a removal transfer (docs/models/loop.md 15). Consequence: after
  a flip the first operation's workpiece temperatures are misplaced by the turn.

## 8. Cost (measured, one core of the i5-4690, bin/tsafe)

| part | cost |
|---|---|
| workpiece finite-volume step | 150-225 ns per material cell (5.5 ms for 26k cells; 14 ms for 63k; 74 ms for 490k) |
| edge kernel | 9 us per in-cut element and evaluation (every `edge_interval_ms` of run time); the elements' running sums cost a few array operations per step |
| domain macro step, one cutting tool (40 in-cut elements), small part | 1.9 ms (Python plumbing dominates) |
| cutting-region field, 12.7 x 4.76 mm insert (4.1k cells) | 5.5 / 8.6 / 10 / 12.6 ms per solve at 10 / 50 / 100 / 250 ms steps (10-32 iterations) |
| cutting-region field, D10 4-flute end mill (1.1k cells) | domain macro step 1.70 -> 2.05 ms (mean, solve every 10 steps) |
| response tables | 0.6 s once per process |
| kernel compilation (cold cache) | ~25 s once per cache key |

Per simulated second at 10 ms macro steps: tool, machine and contact work ~0.19 s, plus the
finite volumes `cells x 190 ns / (every_n_macro x 10 ms)`, plus the cutting-region field: ~0.1 s per
turning tool, ~0.035 s per solid tool in cut (at `tool_field_interval_ms` 100; set 0 in
`tool_field` to drop it). For the reference part (300 x 200 x
100 mm, ~490k material cells at 2 mm) choose `cell_mm: 4` (63k cells: 0.28 s/s at every 5) or
`every_n_macro: 20` (0.37 s/s at 2 mm): the two-scale contact temperature covers the sub-cell
and sub-step scales, so a coarser field changes the bulk only.

## 9. References

Jaeger 1942 (moving sources); Carslaw & Jaeger 1959 (conduction, disc and rectangle sources; quasi-steady
slab and cylinder under a constant flux, ch. III and VII; sources and images, ch. X); Hestenes & Stiefel
1952 (conjugate gradients, J. Res. NBS 49(6) 409-436); Saad 2003 (Iterative Methods for Sparse Linear
Systems, 2nd ed., SIAM: block preconditioners);
Stephenson & Ali 1992 (interrupted cutting); Blok 1937 (flash temperature, partition); Hollomon & Jaffe 1945
(tempering parameter); Orlich & Pietrzeniuk 1976 (TTA diagrams), Miokovic et al. 2005 (AISI 4140 to 1e4 K/s);
Griffiths 1987, Chou & Evans 1999 (white layers); Chen et al. 2017 (machined surface temperature in hard turning);
Boothroyd 1963; Trent & Wright 2000; Astakhov 2006 (Poletika contact length); Komanduri & Hou
2000; Douglas & Gunn 1964 (ADI, rejected); Churchill & Chu 1975a,b; Incropera et al. 2007;
Kays & Bjorklund 1958; Dittus & Boelter 1930; White 2006 (Sutherland); Palmgren 1959,
Harris & Kotzalas 2007 (bearing friction); ISO 76 and ISO 281 (bearing load factors); ASTM D341 (Walther's
viscosity relation); Bossmanns & Tu 1999, 2001, Holkup et al. 2010 (spindle thermal networks, motor loss
split); IEC 60034-2-1 (motor loss segregation); Madhusudana 2014 (contact conductance); Bryan 1990,
Mayr et al. 2012 (machine thermal errors); ISO 7388-1, ISO 12164-1 (tapers); Abramowitz & Stegun
1964 (Gauss-Legendre). Full entries: `db/bibliography/models-thermal.yaml` (Hestenes & Stiefel 1952 and
Saad 2003 are not in it yet).
