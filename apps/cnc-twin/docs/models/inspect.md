# Result geometry and inspection

Package `cnctwin/inspect/`, tests `tests/unit/inspect/`, numerical settings `db/models/inspect.yaml`, ISO 286 tables
`db/models/inspect-iso286.yaml`, thread formulas and tables `db/models/inspect-threads.yaml`, sources
`db/bibliography/models-inspect.yaml` (plus iso-1101-2017, iso-286-1-2010, iso-1-2016 in models-report, iso-965-1-2013
and iso-68-1-1998 in models-tooling, oberg-2016 and kiefer-1953-golden already in the bibliography).

Inspection turns each operation's result into the geometry a metrology lab would see and judges it against the
CAD: the deviation map, remaining material and gouges, the chord errors of the linearised toolpath, the features of
the B-rep and their measured size, location and orientation, the datum reference frames, every GD&T and size
characteristic of the project (hand-written and PMI), the thread classes, a virtual CMM run with a CMM-style report,
and the stock the next operation starts from.

| module | content |
|---|---|
| `result.py` | `ResultPart`, `DisplacementGrid`: material configuration X and the displaced configurations; part files |
| `cadref.py` | the CAD in S: tree, analytic faces, boundaries, adjacency, exact nominal distances; `cad_from_mesh` |
| `specs.py` | datum and tolerance specifications (project form or SI view), hand entries merged with PMI |
| `sampling.py` | result points assigned to CAD faces (`Assignment`, `PointSet`) |
| `deviation.py` | deviation map, per-face statistics, per-brick `deviation_m` |
| `columns.py` | dexel columns, field minus target, connected regions |
| `chordal.py` | toolpath segments from controller commands, chord errors |
| `fitting.py` | least-squares, minimum-zone and envelope associations; zones linear in free displacements |
| `features.py` | feature recognition (planes ... tori, slots, tabs, pockets) and measurement |
| `datums.py` | datum feature simulators, datum reference frames, registration, free degrees of freedom |
| `gdt.py` | every characteristic kind of the project schema |
| `fits286.py`, `threads.py` | ISO 286 limits; ISO 965-1 / ASME B1.1 thread limits and gauging |
| `cmm.py` | virtual CMM: probe plans, probing, compensation, on-demand probes |
| `stock.py` | the next operation's stock (transfer conventions, mislocation) |
| `inspector.py` | one operation's inspection; inspection.json |
| `abi.py`, `kernels.py`, `reference.py` | kernel layouts, Numba kernels, their plain-Python twins |
| `protocols.py` | what inspection needs from other parts, with default implementations |
| `schema.py`, `state.py`, `channels.py`, `domain.py` | settings, checkpointable state, (no) channels, the Domain |

## 1. Inputs and outputs

**Bus read: none directly; bus written: none.** Inspection runs at operation ends and at the end of the run and is
a sink (nothing it computes feeds back into the physics within an operation). It reads other domains' results
through the protocols of `protocols.py` (section 18): removal's mesh of the field in the material configuration with
its `surface_event` vertex attribute and structure's `Displacement` grids (`thermal_m`, `clamp_m`, `residual_m` at
the end of cutting, the residual after unclamping and the unload thermal state) as a `ResultPart`; removal's level-0
z rays of the field and of the CAD target; surface.json `features` and `threads`; cutting's surface-event rows
(`tool_slot`, `u_rel_x/y/z_m`, bus.SURFACE_EVENT_COLUMNS) for the error budget; the controller's commands for the
chord errors. Project: `cad` (path, units, `placement` = `cad_pose_S_C`, `datums`, `tolerances`), `operations[]`
(`name`, `stock_pose_W_S`, `from_previous`), `stock.kind`, `physics.lattice_m`; the tool table (slot -> T number).

**Outputs**: `inspection.json` (section 16), `part_<op>.glb|stl|3mf` (the inspected geometry) and
`part_<op>_unload.stl` (architecture 16; `part_<op>_w<w>.*` for workpieces other than 0), `keyframes/attr/brick_<key>
.npz` arrays `deviation_m` (merged with the other writers' arrays), findings (section 17), the trace event
`inspect:init`, and per transfer a `NextStock` for the integration to hand to removal.

## 2. Result geometry (`result.py`)

removal stores material in the material configuration X (20 degC, unclamped, before stress redistribution;
architecture 4.1); structure publishes `d(X)` on node grids in S. A vertex X of removal's mesh moves to `x = X + d(X)`
with `d` trilinear in the grid nodes `origin + (i, j, k) cell`, clamped at the grid border (constant extrapolation) -
the rule of removal's own meshing (`k_mesh.displace`), so both give the same vertices:

| configuration | `d(X)` | what it is |
|---|---|---|
| `material` | 0 | what the tools left (X, in S) |
| `as_cut` | thermal + clamp + residual | the part in the machine at the end of cutting |
| `cold` | clamp + residual | clamped, at 20 degC |
| `unload` | unload_thermal + final | unclamped, at the unload temperature (`part_<op>_unload.stl`) |
| `final` | final (residual after unclamping, 20 degC) | spring-back and distortion: **what is inspected** (ISO 1: 20 degC) |

A grid that is not given counts as zero and is listed in the record's `assumptions`. Precomputed meshes per
configuration may be handed in (`ResultPart.meshes`). The configuration mesh keeps X per vertex (`X_m`), so the
displacement components can be evaluated at the material points (error budget, section 11).

## 3. The CAD in S (`cadref.py`)

The CAD model (cnctwin.geom `CadModel`: welded mesh with B-rep face ids, analytic face parameters, features, PMI) is
placed by `cad_pose_S_C`; after each transfer the domain moves it by the nominal transfer transform (the drawing
follows the part's nominal frame; a mislocation displaces the material, not the drawing). Selectors stay in the CAD
frame C and are resolved by `cnctwin.geom.selectors.select_faces` (whole features).

Signed distance to a face's nominal surface, positive outside the material (`concave` faces - holes, inner fillets
- have the material outside the surface and the sign reversed):

    plane      d = (p - p0) . n
    cylinder   d = rho - R                     rho = |(p - a) - ((p - a) . u) u|
    sphere     d = |p - c| - R
    cone       d = rho cos(psi) - z sin(psi)   z = (p - apex) . u, u from the apex into the cone
    torus      d = |p - q| - r                 q = c + R e_rho (nearest point of the tube's centre circle)

The gradient is the outward normal. Freeform faces use the distance to their own triangles (pseudo-normal sign,
valid near the surface). Face boundaries: a mesh edge whose two triangles belong to different faces (or an open
edge); adjacency and per-edge concavity (`edge_concavity`: the triangle of b beyond the shared edge rises on the
outward side of a's triangle) come from the same edges. A CAD given as STL / 3MF / OBJ becomes one face per mesh group,
a plane where the group is planar within 1 nm, freeform otherwise (`cad_from_mesh`).

## 4. Point assignment (`sampling.py`)

Every vertex of the inspected mesh gets the CAD's closest point by BVH query (`kernels.deviation_nb`: geom's
closest-point kernel and pseudo-normal sign), its B-rep face, the signed deviation, and whether the closest
feature lies on a face boundary. A vertex is **usable** for its face when |d| <= `assign_band_m` (0.5 mm), its
angle-weighted vertex normal is within `assign_angle_rad` (30 deg) of the CAD normal there, and its closest feature
is not on a boundary. Weights are a third of the adjacent triangle areas (they sum to the surface area). Dense
evaluations keep at most `max_points_per_face` (4000) usable points per face by a deterministic stride in vertex
order. The share of a face's usable area whose vertices a cut generated (`surface_event >= 1`) is the face's
`machined_fraction`.

## 5. Deviation map (`deviation.py`)

Per face (usable points and all points) and for the whole surface: area-weighted mean, RMS, standard deviation,
minimum, maximum, the 1 % / 99 % quantiles (weighted), the largest |d| and where, and the areas in excess / missing
beyond `remaining_tolerance_m`. Sign: + material outside the CAD (excess), - CAD material missing. The same signed
distance is written per keyframe-brick vertex (`deviation_m`, float32) after moving the brick vertices to the
inspected configuration.

## 6. Remaining material and gouges (`columns.py`)

Both solids as z-family dexel columns on one lattice (`abi.Columns`): column `c = i ny + j` through the node
`(x_i, y_j)` holds sorted, disjoint material intervals and owns the cell `cx x cy` around its node, so a volume is
`cx cy sum(hi - lo)` - removal's own midpoint rule for its z family (architecture 5.4). Meshes are dexelised by +z
rays (`rays_all_hits`) and a winding count over the crossings (`columns_from_hits_nb`: a crossing with outward
normal `n_z < 0` enters, `n_z > 0` leaves, `n_z = 0` grazes; crossings of the same sense within `column_merge_m` (1
nm) count once - a ray through a shared edge hits both triangles; zero-length intervals are dropped). The fallback
lattice puts its nodes at half-cell offsets (`lattice_for(offset=0.5)`), so rays never run inside a face at a round
coordinate and walls at round coordinates lie on cell boundaries.

* remaining material = field minus target per column (pieces thinner than `remaining_tolerance_m`, 5 um, dropped);
  gouges = target minus field (thinner than `gouge_tolerance_m`, 5 um, dropped); the raw volumes without the
  thresholds are reported too;
* regions: intervals of face-adjacent columns overlapping in z by a positive length are connected; union-find with
  the smallest index as root (`label_components_nb`), regions numbered by their first interval, ordered by volume
  (`component_stats_nb`: volume, centroid, box, largest thickness and its point, columns, intervals); regions below
  `region_min_volume_m3` (0.001 mm^3) are counted, not listed; each listed region names its nearest CAD face.

The remaining-material comparison is made in the material configuration (what the tools left); spring-back is in the
deviation map.

## 7. Chordal deviation of the toolpath (`chordal.py`)

With `d(p)` the signed distance of a tool reference point to the CAD and a segment `P(s)`, `s in [0, 1]`, from a
point at `d0` to one at `d1`, the chord error is

    e(s) = d(P(s)) - ((1 - s) d0 + s d1)

the most negative value (toward the part: the chord cuts into a convex surface's offset) and the most positive
(away: material left on a concave one) are found from `chordal_samples` (8) interior samples and `chordal_iterations`
(40) golden-section steps on the bracket of the best sample (Kiefer 1953). For a chord of length L on a curve of
radius R it is the sagitta `R (1 - cos(L / 2R))`. Lines are G01 chords; arcs (G02/G03) are evaluated along the exact
helical / spiral arc (`segment_point_nb`), so their error measures the programmed arc against the CAD. The
reference point is the programmed tool tip, optionally moved along the tool axis (`reference_offset_m`, e.g. +R for a
ball centre); the end-point offsets d0, d1 are reported, so no tool model is needed. Segments with both ends farther
than `chordal_max_distance_m` (5 mm) are approach moves and skipped; segments above `chordal_tolerance_m` (10 um) are
counted. Tool tips come from `TipResolver`: `CartesianTips` follows sanity's 3-axis convention (TIP-space commands
carry the tip; machine space `x_W = X - offset_x`, `z_W = Z - offset_z - h`), mapped to S by the operation's stock pose.

## 8. Associated features (`fitting.py`)

**Least squares** (ISO 10360-6 Gaussian association; Forbes 1989; Shakarji 1998): plane and line by SVD of the
centred points; circle (in a given plane; algebraic start, Kasa 1976), sphere, cylinder `|(p - a) x u| - R`, cone
`(rho - r0 - z tan psi) cos psi` (axis point, direction, radius at the point, half angle: no apex singularity) and
torus `sqrt((rho - R)^2 + z^2) - r` by Levenberg-Marquardt with central-difference Jacobians in a local
parametrisation re-centred after every outer step, started from the CAD nominal (deterministic), stopped when the
step is below `ls_step_tolerance` (1e-13) of the feature size or after `ls_iterations` (60). Standard
uncertainties from `s^2 (J^T J)^-1`, `s^2 = sum r^2 / (n - p)`.

**Chebyshev associations by sequential linear programming** (Chetwynd 1985; Carr & Ferreira 1995a, b): the
distances `rho_i(x)` of the points to the feature are linearised in a small displacement x of the feature (the
small-displacement torsor, Bourdet et al. 1996), the LP (HiGHS, Huangfu & Hall 2018) minimises the zone `max - min`
(MZ), maximises `min` (MIC) or minimises `max` (MCC) within a trust region (`lp_trust_m`, 1 mm, and the same divided
by the feature size for rotations); the step is then searched exactly along the LP direction (golden section on the
exact zone: the LP cannot see directions that matter only to second order, such as the free side of two diametral
points), accepted if the exact zone does not grow, else the trust region is halved; it stops when the zone changes by
less than `lp_tolerance_m` (1 pm) or after `lp_iterations` (12). MZ plane (ISO 12781 MZPL), MZ line (ISO 12780),
MZ / MIC / MCC circle (ISO 12181), cylinder (ISO 12180; axis direction optionally fixed) and sphere. The datum
simulator of a plane - constrained minimax outside the material (ISO 5459), the ASME tangent plane - is the outer
plane of the minimum zone: for a fixed orientation the tangent plane touches the highest point, and minimising the
largest gap over the orientation is minimising the zone. Zones that are linear in free displacements (profile,
orientation and location with unconstrained degrees of freedom) are single LPs: `symmetric_zone` (min max |d + J xi|),
`band_zone` (min (max - min)), `radial_zone` (cylindrical zones: a 64-gon LP start, then the exact minimax by the
sequential circumscribed LP on the exact norms). `min_enclosing_circle` is Welzl's construction in the input order.

## 9. Features (`features.py`)

Single-surface features are the CAD's features (adjacent faces of one analytic surface): plane, cylinder (hole when
concave, boss otherwise), cone, sphere, torus, freeform. **Width features** (ISO 14405 size between opposite planes):
plane features with antiparallel outward normals (within `pair_angle_rad`, 1 deg) whose outlines overlap in
projection, mutually nearest per kind: a **slot** when the normals face each other (air between), a **tab** when they
are back to back (material between: wall thicknesses); a face may be in one slot and one tab. **Pockets**: a plane
whose neighbours meet it at concave edges and rise perpendicular to it (walls), with at least one face around the
opening parallel to the floor with the same outward sense (the top, which gives the depth); a convex neighbour makes
the pocket open on that side (at least two walls needed then). Measurement: least squares on the usable points and
the deviations from the CAD - plane: offset, tilt, MZ and LS flatness; cylinder: diameter, axis offset at the
nominal axis point, axis tilt, form; sphere; cone: half angle, apex, axis; torus: both radii, centre; slot / tab:
least-squares width (the two planes' distance at their mid point) and the two-point widths (every point of each side
to the other side's plane); pocket: floor flatness and depth (top to floor along the floor's normal).

## 10. Datums and datum reference frames (`datums.py`)

Simulators on the result points, in precedence order (ASME Y14.5-2018 7, Y14.5.1-2019; ISO 5459): plane - tangent
plane (primary), the same with the normal kept perpendicular to the primary direction (one rotation left), a plane
whose orientation the higher datums fix only translates to the outermost point; cylinder - maximum inscribed (hole)
or minimum circumscribed (boss) cylinder, free when primary, parallel to the primary direction otherwise (least
squares with a note when the points do not surround the axis); sphere - MIC / MCC centre; cone - least-squares axis
and apex. The constraints a datum gets are taken from the frame of the measured higher datums (`constraints_for`).

Frame rules (the same code builds the nominal frame from the CAD's analytic faces and the measured frame from the
simulators): a primary plane sets `e3 = -n` (into the material) and the origin on it; a primary axis sets e3 along it
and the origin on it (a cone also fixes the origin at its apex); a primary point sets the origin. A secondary plane
sets `e1 = -n` made perpendicular to e3 and moves the origin onto it; a secondary axis parallel to e3 puts the origin
on its trace; a plane perpendicular to a primary axis fixes the origin along it. A tertiary plane sets e2 (or clocks e1
when the secondary was an axis) and moves the origin onto it; a tertiary axis clocks e1 toward its trace.
`e2 = e3 x e1` unless a tertiary plane's inward normal points the other way (a left-handed frame, so basic dimensions
are measured into the material from every datum). Degrees of freedom no datum locks stay **free** and every
characteristic minimises over them (ASME Y14.5-2018 7.11). `registration = O_n + A_n A_m^T (p - O_m)` maps result
points into the nominal frame; characteristics with datums are evaluated there against the CAD. The small-displacement
columns of the free degrees of freedom are `n . e_j` (translation along e_j) and `((p - O) x n) . e_j` (rotation about
e_j through O); the optimised motion is applied exactly (Rodrigues) and the zone re-evaluated on the exact geometry.

## 11. Characteristics (`gdt.py`)

Every characteristic runs twice: on the dense result surface (the twin's truth: `measured_m`, `pass`) and on the
virtual CMM's probe points (`cmm`: what the CMM program would report; `agrees` says whether both judge alike).

| kind | measured value |
|---|---|
| flatness | MZ width of the plane points (LS width reported) |
| straightness | plane: max MZ width of `sections` line profiles along the face's long direction; cylinder / cone: max over generator lines, or the derived median line's minimum circumscribed cylinder diameter when the modifier names the axis (`axis`, `DML`) |
| circularity | max over `sections` cross sections (covering at least half a turn) of the MZ circle zone |
| cylindricity | MZ cylinder zone |
| profile_surface | `2 max |d|` from the nominal surface in the datum frame; without datums the zone floats in all six degrees of freedom (form and size remain; ISO 1660) |
| profile_line | the same per cross section; without datums each section floats in its plane |
| perpendicularity, parallelism, angularity | plane: width along the nominal normal in the datum frame (free rotations optimised); axis (derived median line of section circle centres): a cylindrical zone (minimum enclosing circle of the centres projected) when the nominal axis lies along the primary direction, else a planar zone normal to the axis in the plane of the axis and the datum direction |
| position | hole / boss: `2 max` distance of the axis of the unrelated actual mating envelope (MIC / MCC) from the true position axis over the feature length (ASME Y14.5 10.3); MMC bonus = envelope - MMC size (hole) / MMC - envelope (boss); LMC bonus = LMC - minimum material envelope (hole: MCC) / MIC - LMC (boss); sphere: 2 x centre distance; slot / tab: 2 x the largest centre-plane offset at the outline corners; plane: `2 max |d|` from the true plane (ISO 1101). True position: the CAD nominal, or `basic_mm` as coordinates in the datum frame |
| concentricity (coaxiality) | 2 x the largest distance of the section centres from the datum axis |
| symmetry | 2 x the largest deviation of the median plane's offset from the datum plane (the CAD's offset subtracted) |
| runout, total_runout | range of the normal deviation from the nominal surface coaxial with the datum axis: per measuring circle (axial bins; radial bins on faces perpendicular to the axis) or over the surface. The datum axis is one axis datum's envelope axis, the common axis of several (through their envelope axes' mid points), or the frame's e3 |
| distance | between least-squares features: plane-plane at the second's centroid along the first's normal, axis-axis, plane-axis, centres; `nominal_cad_m` and a note when the specification differs from the CAD by more than the tolerance |
| angle | between least-squares directions (the supplement when the specification counts it) |
| diameter, radius | least-squares size (ISO 14405 GG) as `measured_m`; two-point local sizes (LP: opposite 10-deg bins per section about the LS axis), the envelope (GX hole / GN boss) and the LS size all within the limits (ASME Rule #1) for `pass` |
| roughness | surface.json's Ra / Rz / Rt of the feature id (kinematic profile simulation, surface domain) |
| thread | a size whose fit is a thread class (`6H`, `5g6g`, `2A`, or a full designation): section 13 |

**Error budget** (`contributions_m`): for each displacement component carried by the points - the spring-back /
distortion `d_final(X)` and, when the surface events are wired, each tool's deflection at generation `u_rel` - the
measured value minus the value with that component removed from every point (datum features included):
`[{source: residual_distortion | tool_deflection, tool, value_m}]`. For a nonlinear characteristic (a zone) the
contributions need not add up to the value.

## 12. Size limits (`fits286.py`, ISO 286)

IT1-IT18 from ISO 286-1 Table 1 by main size step (up to 500 mm; IT14-IT18 not up to 1 mm). Shafts: a-g upper
deviation es from the ISO 286-2 tables (intermediate steps), h `es = 0`, js `+-IT/2`, k (the IT4-IT7 value, else 0), m,
n, p, r, s lower deviation ei. Holes by ISO 286-1 5.3.2: A-H `EI = -es`; JS `+-IT/2`; K, M, N up to IT8 and P-S up to
IT7 `ES = -ei + Delta`, `Delta = IT(n) - IT(n - 1)` (0 up to 3 mm); K above IT8 `ES = 0`; M above IT8 `ES = -ei`; N above IT8
`ES = 0` (above 3 mm); P-S above IT7 `ES = -ei`. Positions j / J, t-zc, cd / ef / fg and IT01 / IT0 raise with the reason
(the characteristic is then "not evaluated" with that reason).

## 13. Threads (`threads.py`)

**ISO 965-1** (P and d in mm, tolerances in um, d the geometric mean of the basic major diameter range, ranges
0.99-1.4-2.8-5.6-11.2-22.4-45-90-180-355-600 mm):

    T_d2(6) = 90 P^0.4 d^0.1          grades 3 4 5 6 7 8 9: x 0.5 0.63 0.8 1 1.25 1.6 2.0
    T_D2(n) = k(n) T_d2(6)            grades 4 5 6 7 8: k = 0.85 1.06 1.32 1.7 2.12
    T_d(6)  = 180 P^(2/3) - 3.15 P^(-1/2)            grades 4 6 8: x 0.63 1 1.6
    T_D1(6) = 433 P - 190 P^1.22 (P <= 0.8),  230 P^0.7 (P >= 1)   grades 4-8: x 0.63 0.8 1 1.25 1.6
    es = -(50 + 11 P) e, -(30 + 11 P) f, -(15 + 11 P) g, 0 h;   EI = +(15 + 11 P) G, 0 H

Each value is rounded to the nearest R40 number (ISO 3; ties to the larger), then to a whole um (half to even): this
reproduces es(g) = -20 / -22 / -24 / -26 / -28 / -32 / -34 / -38 um for M3 ... M16 and every tolerance of the M4-M16
coarse 6g / 6H limits. Four table cells differ from the rounded formula and are taken from the tables
(`iso_table_values`): T_d2(6) = 75 and T_D2(6) = 100 um for P 0.5 in 2.8-5.6 mm (formula 80, 106), T_D2(6) = 180 um for
P 1.5 in 5.6-11.2 mm (formula 170), T_D1(6) = 140 um for P 0.5 (formula 132). Limits: external `d_max = d + es`,
`d_min = d_max - T_d`, `d2_max = d2 + es`, `d2_min = d2_max - T_d2`; internal `D_min = D + EI`, `D2_min = D2 + EI`,
`D2_max = D2_min + T_D2`, `D1_min = D1 + EI`, `D1_max = D1_min + T_D1`; ISO 68-1 basic `d2 = d - 0.75 H`, `d1 = d - 1.25 H`,
`H = (sqrt 3 / 2) P`. The coarse pitch comes from ISO 261 when a designation gives none (`M10-6g`).

**ASME B1.1** (inch): class 2A pitch-diameter tolerance `0.0015 D^(1/3) + 0.0015 LE^(1/2) + 0.015 P^(2/3)`, LE = D
for UNC, UNF, 4/6/8-UN and 9 P otherwise; class factors 1A 1.5, 2A 1, 3A 0.75, 1B 1.95, 2B 1.3, 3B 0.975; allowance 0.3 x
the 2A tolerance for 1A / 2A; external major tolerance 0.060 P^(2/3) (2A, 3A) or 0.090 P^(2/3) (1A); internal minor
tolerance 0.25 P - 0.4 P^2 (1B, 2B). As the standard's tables: external limits from the rounded allowance and
tolerances (maximum = basic - allowance, minimum = maximum - tolerance, 0.0001 in), internal limits basic + tolerance
rounded (minor of 1B / 2B to 0.001 in).

**Gauging** (ISO 1502; Galyer & Shotbolt 1990): the virtual pitch diameter is the simple one plus the diameter
equivalents of the cumulative pitch deviation over the length of engagement (`|dP| LE / P`, LE the ISO 965-1 normal
upper limit `6.7 P d^0.2` when not given) and of the flank half-angle deviations:

    f_P = |dP_sum| cot(alpha / 2)          f_alpha = sum_i 2 h_i |d alpha_i| / sin(alpha)

with `h_i` the engaged flank height on the interfering side of the pitch line: H/4 below it (to the internal crest),
3H/8 above it (to the external crest); an external flank with a larger half angle interferes below, a smaller one
above (reversed for internal threads). For 60 deg this is `0.0727 P (K1 |da1| + K2 |da2|)` um per arc minute with K = 2
and 3 - ISO 1502's `0.073 P (K1 |da1| + K2 |da2|)`. External: GO accepts `d2_virtual <= d2_max`, NOT GO accepts
`d2_simple >= d2_min`; internal: GO `D2_virtual >= D2_min`, NOT GO `D2_simple <= D2_max`; major and minor diameters
against their limits directly. `pass` = both gauges accept and the diameters lie within their limits. The measured
thread (surface.json `threads`, simple pitch diameter, else the three-wire one) is matched by kind and by major diameter
within `thread_match_rel` (2 %) of the nominal.

## 14. Virtual CMM (`cmm.py`)

**Probing**: a ball of radius r (`probe_radius_m`, 1 mm: a 2 mm ruby, Renishaw 2020) starts at `nominal + (r + prehit) n`
(n the nominal outward normal, `probe_prehit_m` 2 mm) and moves along `-n` by sphere tracing on the exact unsigned
distance to the result mesh (Hart 1996; `probe_nb`): each step advances by the free distance, so the ball never passes
through the surface; contact when the free distance is below `probe_trace_tolerance_m` (1 nm); no contact within
`prehit + search` (`probe_search_m` 2 mm) means material missing; a ball already in contact at the start means excess
beyond the clearance. **Compensation** in vector mode: the reported point is the ball centre minus r along the nominal
vector (the exact contact point is kept); the cosine error of vector mode on a surface tilted by theta from the vector
is `r (1 / cos(theta) - 1) ~ r theta^2 / 2` (0.5 nm at r = 1 mm, 1 mrad). On-demand probes: `VirtualCMM.probe(point,
direction, radius_m, vector, travel_m)`.

**Plans** (ISO 10360-6 sampling counts as defaults): a plane gets a grid of about `probe_points_plane` (9) points
over its outline, inside the face and at least `probe_radius + probe_inset` (1.5 mm) from its edges; a cylinder or cone
`probe_levels` (3) circles of `probe_points_circle` (8) points over its axial extent and angular span (inset from the
ends); spheres, tori and freeform faces points spread over the face by farthest-point selection from the largest
triangle. A probe plan point is the projection of the candidate onto the nominal surface. **Report**: `cmm_report` rows
(characteristic, kind, feature, nominal, limits, tolerance, actual, deviation, pass, points) and per characteristic
`points [{x_S_m, y_S_m, z_S_m, deviation_m}]`.

## 15. The next operation's stock (`stock.py`)

    p_new = M (R p_old + t)

applied to the inspected geometry `X + d_final` (the part as it is seated next: unclamped, sprung back, distorted,
20 degC). `R, t` mirror removal's transfer conventions exactly (flip about an S axis, sub-spindle = half turn about S y,
the turned box at the S origin or a turned axis kept on S z with z from 0; a test compares both); pass removal's level-0
box so both agree. **M is the mislocation input** (a small rigid pose in the new S: a re-chucking error drawn by an
uncertainty study, a measured seating error, a sub-spindle grip error). The field comes from the `StockFieldBuilder`:
removal's exact remap when R is a signed permutation, M is the identity and the part has no distortion above 1 nm (the
material configuration then is the part); otherwise the closed transformed mesh is rasterised (removal's mesh_field
keeps the tri-dexel endpoints exact to 10 nm along its rays, so a distortion well below one cell is carried).

## 16. inspection.json (format `cnctwin-inspection`, version 1)

`temperature_K` (293.15), `configuration`; `tolerances [{id, kind, feature, faces, datums, source (hand | pmi), op,
status (evaluated | not evaluated), reason, pass, method, nominal_m, plus_m, minus_m, lower_m, upper_m (sizes,
distances) | tolerance_m, allowed_m, bonus_m, mmc_m, lmc_m (zones) | nominal_rad ... (angles), measured_m, deviation_m,
fit, size {least_squares_m, envelope_m, two_point_min_m, two_point_max_m, limits, checks, nominal_cad_m, rule},
machined_fraction, tools, points [{x_S_m, y_S_m, z_S_m, deviation_m}], contributions_m [{source, tool, value_m}],
cmm {measured, pass, method, points, agrees, row}}]` (kind `thread`: `designation, class, pitch_diameter_m {measured,
virtual, min, max}, major_diameter_m, minor_diameter_m {measured, min, max}, gauges {GO, NOT GO}, virtual {...},
fundamental_deviation_m, tolerances_m, pass`) - the last operation of each workpiece (the finished part); `cmm_report
[...]`; `features [...]` (recognised features with their measurement); `datums {label set: {nominal, measured,
registration}}`; `deviation {surface, usable, faces [...]}`; `remaining [...]`, `gouges [...]` (regions: volume_m3,
centroid_S_m, box_lo_S_m, box_hi_S_m, max_thickness_m, max_point_S_m, columns, intervals, nearest_face); `chordal
[{segments, evaluated, limit_m, exceeding, max_m, mean_m, worst [...], op}]`; `unload [{op, workpiece, time_s,
mean_temp_K, max_temp_K}]`; `operations [...]` (every operation record); `next_stock [...]`; `files`;
`summary {checked, failed, not_evaluated, cmm_disagrees}`; `settings`; `models`; `attributes` (brick files written).
These are the keys docs/report.md "Inputs" reads (tested with the report's own `limits` / `_judged`).

## 17. Findings

| condition | category, severity |
|---|---|
| a gouge region (target material missing deeper than `gouge_tolerance_m`), at each operation end | `gouge`, error |
| an evaluated characteristic fails | `out_of_tolerance`, error |
| a characteristic cannot be evaluated (no matching face, too few points, a fit or class not encoded ...) | `out_of_tolerance`, info, with the reason |
| remaining material on the finished part (per listed region) | `remaining_material`, warning |

Findings carry their quantities (SI) and are raised once (keys kept in the state, so a resumed run does not repeat
them). removal also raises per-step gouge events and a remaining-material warning per operation; inspection's are per
connected region of the result.

## 18. Domain, protocols and integration

`InspectDomain(results, columns, surface, events, stock_builder, tips, commands, cad, settings, mislocation, bricks)`
implements core.domain.Domain; `channels()` is empty (no value per step to record), `kernel_inputs` / `absorb` return
None, `macro_step` does nothing. Its state (`InspectState`: operation records, next-stock summaries, finding keys,
files) lives on the domain (SimState has no inspect field) and is checkpointed through the participant protocol
(`checkpoint_cut`, `checkpoint_state`, `restore_state`, `truncate_outputs`; the CAD poses are rebuilt from the
recorded transfers).

**Wiring**: `results` - a ResultSource returning removal's `to_geom_mesh(mesh_level(field))` (material configuration,
`surface_event` kept) with structure's grids (`DisplacementGrid.from_bus(displacement, "thermal_m" | "clamp_m" |
"residual_m")`, the unclamped residual as `final`, the unload thermal grid as `unload_thermal`) and thermal's unload
temperatures, taken **before** removal transfers the field; order the operation ends structure (unclamp, spring-back)
-> surface (threads) -> inspect -> removal (transfer). `columns` - a FieldColumns over removal's level-0 z rays of the
field and its TargetField (origin `lv_origin_m[0][:2]`, cell `lv_cell_m[0]`, dims `lv_dims[0][:2]`, intervals = z origin +
quanta x 10 nm); without it, both are dexelised from meshes (`MeshColumns`). `surface` -
`SurfaceJson(lambda: surface_domain.document(ctx, state))`. `events` - `TableEvents(rows, first_id)` over cutting's
surface-event rows (removal's `surface_ids`). `stock_builder` - remap: removal.transfer.apply for signed permutations;
from_mesh: removal.stock.mesh_field; the NextStock is `domain.next_stock(op_index, workpiece)`. `tips` -
`CartesianTips(setup work offsets, tool length offsets)` for 3-axis mills (motion's kinematics for others); `commands`
- the controller's executed commands per operation. Register InspectDomain as a checkpoint participant and the part
files (`state.files`) with RunDirectory.register_output. `physics.inspect` is declared (`schema.physics_schema()`) but
the project schema routes physics sections only for SimState domains; until it adds `inspect`, the defaults apply.

## 19. Settings (`db/models/inspect.yaml`, `schema.py`)

| setting | default | role |
|---|---|---|
| assign_band, assign_angle | 0.5 mm, 30 deg | point assignment (section 4) |
| max_points_per_face, sections, section_min_points | 4000, 8, 8 | dense sampling, cross sections |
| ls_iterations, ls_step_tolerance | 60, 1e-13 | least squares |
| lp_iterations, lp_tolerance, lp_trust, zone_polygon_sides | 12, 1 pm, 1 mm, 64 | Chebyshev solvers |
| pair_angle | 1 deg | slots, tabs, pocket walls |
| probe_radius, probe_prehit, probe_search, probe_inset | 1, 2, 2, 0.5 mm | CMM |
| probe_points_plane, probe_points_circle, probe_levels | 9, 8, 3 | CMM plans |
| probe_trace_iterations, probe_trace_tolerance | 400, 1 nm | sphere tracing |
| remaining_tolerance, gouge_tolerance, region_min_volume, column_merge | 5 um, 5 um, 0.001 mm^3, 1 nm | remaining material, gouges |
| chordal_samples, chordal_iterations, chordal_max_distance, chordal_tolerance | 8, 40, 5 mm, 10 um | chord errors |
| thread_match_rel | 2 % | matching measured threads |

Each value's source and reason is in the file; out-of-range overrides are rejected.

## 20. Kernels and their references

Every kernel: `@njit(<explicit signature>, cache=True, nogil=True, error_model="numpy")`, no fastmath / parallel,
writable C-contiguous arrays, scalars and the NamedTuples of geom.abi / inspect.abi; `KERNELS` is checked for one
signature each, and the suite passes with `NUMBA_BOUNDSCHECK=1`.

| kernel | reference | check |
|---|---|---|
| `trilinear_nb` | `trilinear_ref` | random grid and points, 1e-14; node values exact |
| `deviation_nb` (+ `_signed_nb`) | `deviation_ref` (brute force, winding-number sign) | sphere mesh, 30 points, 1e-13 |
| `columns_from_hits_nb` | `columns_from_hits_ref` | pocket block, rays through its edges and corners |
| `columns_difference_nb` | `columns_difference_ref` | 60 random columns |
| `label_components_nb` (+ `_find_nb`), `component_stats_nb` | `components_ref` (breadth first), `component_stats_ref` | random field, same partition, volumes 1e-14 |
| `segment_point_nb` | `segment_point_ref` | helical-spiral arc |
| `chord_deviation_nb` (+ `_chord_error_nb`) | `chord_error_at`, `chord_error_ref` (dense brute force) | value at the kernel's s exact, at least as extreme as 121 samples |
| `probe_nb` | `probe_ref` (marching + bisection, brute force) | box, oblique and frontal approaches, 1e-9 m |

## 21. Validity, error bounds and how they were established

| quantity | bound | established by |
|---|---|---|
| LS fits | parameters within 4 standard uncertainties of the truth for 1 um noise (plane, cylinder, sphere, circle, cone, torus) | tests on synthetic points |
| minimum-zone flatness | exact (1e-9 relative) on the hand example `z3 x2 / sqrt(x3^2 + z3^2)`; LS 5 % larger there | test |
| MZ straightness, roundness (3-lobe: 2 a), cylindricity (bow), MIC / MCC | exact to 1e-6 relative | tests |
| radial (cylindrical) zones with free displacements | exact minimax (the polygon start is within 0.12 %, refined to 1e-9) | test |
| datum registration | exact (1e-12) for a rigidly moved part (rotation about a tilted axis, translation) | test |
| position | 50 um vs 2 x 25 um offset to 1e-9 m through a misplaced part; MMC / LMC bonus to 1e-9 m | tests |
| coaxiality, runout, total runout, profile with datums | 2 e for an eccentricity e to 0.2 % | tests (stepped shaft) |
| deviation map | the offset of a grown box exactly per face; sqrt(3) delta at the corners | test |
| remaining material / gouge volume | exact for the lattice (1e-12); vs the true geometry up to `perimeter x cell / 2 x height` per region (walls off the cell boundaries) | tests (pocket 0.5 mm shallow: 96 mm^3; 0.2 mm deep: 38.4 mm^3) |
| chord error | the sagitta within the CAD tessellation (0.2 um chords); kernel vs brute force exact at its parameter | tests |
| ISO 965 limits | equal to the ISO 965-2 table to 1 um for M3-M16 coarse 6g / 6H (M8 x 1.25, M10 x 1.5 included) | tests against the chart |
| ASME B1.1 limits | equal to Machinery's Handbook for 1/4-20 and 1/2-13 UNC 2A / 2B to 0.0001 in | test |
| ISO 286 limits | 22 tolerance classes (holes and shafts, 2-40 mm) equal to ISO 286-2 values | test |
| flank-angle diameter equivalent | the ISO 1502 coefficient to 0.4 % (0.0727 exact vs 0.073 rounded) | test |
| probing | compensated points within 1 nm on surfaces normal to the vector; a hole within its facets' sagitta | tests |
| determinism | inspection.json, part files and findings bit-identical across runs | test |

Validity: rigid-body small displacements in the Chebyshev solvers (each step re-linearises; form errors well below
the feature size); features with points around enough of the surface (an envelope of points that do not surround the
axis is unbounded and falls back to least squares with a note; a cross section needs half a turn); ISO 286 up to 500
mm; ISO 965-1 for basic diameters 0.99-600 mm and its grades; ASME B1.1 for sizes from 1/4 in (internal minor diameter
formula); result meshes from removal's marching cubes (vertex spacing = lattice).

## 22. Assumptions and limitations (with consequences)

* The characteristics are evaluated on the result mesh's vertices: form below the lattice spacing is not resolved by
  the dexel field itself (the surface domain carries the sub-lattice finish); a marching-cubes mesh chamfers sharp
  edges within about one cell, which the assignment's boundary and normal filters exclude from fits.
* The dense evaluation is the twin's truth; the CMM evaluation uses a few points and can miss a local defect: both are
  reported and `cmm.agrees` flags a disagreement.
* Stylus shaft and probe head collisions and reach are not modelled (plans approach along the nominal normal).
* Common datums (A-B) are recognised for runout and coaxiality only (all-axis datum lists); elsewhere the datum list
  is primary / secondary / tertiary. Datum targets, movable datum targets and datum feature modifiers (MMB) are not
  encoded: a feature of size as datum is simulated RFS (its envelope), so datum shift is not granted (conservative).
* Thread positions E / F (internal) and a-d (external), ASME class 3B's minor diameter maximum and ISO 286 positions
  j / J, t-zc are not encoded: such a characteristic is "not evaluated" with the reason (3B: limits without the minor
  maximum, with a note).
* Symmetry is measured against the datum plane its selector names (a centre-plane datum of two faces cannot be written
  with one selector).
* The error budget separates the spring-back / distortion and each tool's deflection at generation; thermal growth,
  wear and runout stay in the remainder (they are part of the event positions removal subtracted at).
* Orientation zones of axes are cylindrical when the axis lies along the primary datum direction, planar otherwise (the
  project form has no diameter symbol for the zone).
* The fallback remaining-material comparison dexelises meshes (0.25 mm columns unless wired to removal's rays); with
  removal's own rays it is exact to removal's quantum.

## 23. Cost (measured, single core of this box)

Inspection is not in the macro loop: **0 per macro step**. At an operation end (100 x 80 x 30 mm block with a pocket,
result mesh of 139 266 vertices, 278 528 triangles, 19 recognised features, 4 characteristics): point assignment 0.68 s
(2.0e5 points/s), probing 19 us per probe, chord errors 8.4 us per segment, the mesh-dexelised remaining-material
comparison at 0.25 mm 3.4 s (131 000 columns; reading removal's rays instead is a copy), the whole `inspect()` 7.9 s.
Memory (computed from the arrays): the assignment holds about 140 bytes per result vertex plus 24 per error-budget component (about 23 MB for the example with one component).

## 24. RNG streams

None: inspection draws no random numbers (plans, thinning and solvers are deterministic).

## References

asme-y14-5-2018, asme-y14-5-1-2019, iso-5459-2011, iso-1101-2017, iso-14405-1-2016, iso-286-1-2010, iso-286-2-2010,
iso-12781-1-2011, iso-12780-1-2011, iso-12181-1-2011, iso-12180-1-2011, iso-1660-2017, iso-10360-6-2001, iso-1-2016,
iso-965-1-2013, iso-68-1-1998, iso-261-1998, iso-3-1973, iso-1502-1996, asme-b1-1-2019, oberg-2016,
engineersedge-unified-thread-limits, amesweb-metric-thread-chart, carr-ferreira-1995a, carr-ferreira-1995b,
chetwynd-1985, bourdet-1996-sdt, forbes-1989-npl, shakarji-1998, kasa-1976, welzl-1991, hart-1996-sphere-tracing,
huangfu-hall-2018, galyer-shotbolt-1990, renishaw-2020-styli, kiefer-1953-golden (db/bibliography).
