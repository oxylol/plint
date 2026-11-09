# Surface and edges: finish, marks, burrs, surface integrity, thread accuracy

Package `cnctwin/surface/` (the `surface` SimState field, an L2 slow domain), tests `tests/unit/surface/`,
coefficients `db/models/surface-roughness.yaml`, `surface-burr.yaml`, `surface-integrity.yaml`, sources
`db/bibliography/models-surface.yaml` (plus keys already in other bibliography files).

The domain turns cutting's **surface events** (one row per edge element where a pass leaves the final
surface: X in the material configuration, outward normal, tick, tooth, element, h, edge radius, VB,
u_rel at generation; `core.bus.SURFACE_EVENT_COLUMNS`) into the finish of every machined face by
**kinematic profile simulation**, records the marks a stop, a dwell, a reversal or a step-over leaves,
predicts the burr at every exit edge, the residual stress and hardness of the machined layer, and
measures generated threads. It writes `surface.json`, per-brick attribute arrays and findings.

| module | content |
|---|---|
| `abi.py`, `kernels.py`, `reference.py` | kernel layouts, Numba kernels, their plain-Python twins |
| `lines.py` | profile lines: frames, pass rows, templates, evaluation of one line |
| `roughness.py` | ISO 4287 / 4288 / 3274 / 16610-21 evaluation of a profile |
| `gather.py` | one contact's events of a step -> patches, passes, lines (classification, frames, rows) |
| `store.py` | the per-face patch store (bounded memory) |
| `faces.py` | roughness specifications, feature matching, step-over profiles, face grouping, brick maps |
| `marks.py`, `burrs.py`, `integrity.py`, `threads.py` | the models of sections 4-6 and 8 |
| `kinematics.py` | tool motion at the event ticks from motion's PathFrame |
| `protocols.py` | ToolInfo, WorkpieceQuery, BrickGrid, BrickMeshes and their default implementations |
| `state.py`, `schema.py`, `channels.py`, `domain.py`, `params.py` | domain plumbing |

## 1. Inputs and outputs

**Bus read (prev)**: `cutting.contacts[*]` (`surface_events`, `shocks` exit rows, `recut`, `in_cut_s`,
`ploughing_J`, `mean_speed_m_s`, `engaged_edge_m`, `h_max_step_m`, `wrench_tool_T0_mean` thrust, `tool_slot`,
`contact`), `cutting.coefficients[slot]` (`chip_ratio`, `Kte_N_per_m`, `Kre_N_per_m`), `motion.paths[*]`
(`tick`, `tip_S_m`, `axis_T_M`, `pose_S_T`, `pose_M_T`, `pose_W0_T`, `spindle_omega_rad_s`,
`spindle_angle_rad`, `feed_vel_M_m_s`, `axis_names`, `axis_cmd`, `following_error`, `block_index`,
`tool_slot`, `workpiece`, `path`), `removal.contacts[*]` (`contact`, `path`, `workpiece`, `geometry.ap_m`),
`wear.tools[slot]` (`radial_wear_m`, `axial_wear_m`, `bue_height_m` per element), `thermal.contact_surface_temp_K`,
`dynamics.chatter_index`, `chatter_freq_Hz`. **Bus written: none** (the domain is a sink; other parts query
`SurfaceDomain.view(state)`, section 9). Project: `material` (physical, cutting.min_chip_thickness_ratio,
cutting.work_hardening, thermal_damage, friction, composite), `stock`, `cad.tolerances` (kind roughness) and
`cad.placement`, `environment.ambient_degC`, `physics.lattice_mm`, `physics.surface` (section 10).

**Outputs**: `surface.json` (section 11), `keyframes/attr/brick_<brick>.npz` (merged with what other writers put
there), findings `finish_out_of_spec` (warning), `burr` (warning; delamination: error), the trace event
`surface:init`, the macro channels of `channels.py`.

## 2. Kinematic profile simulation

**Principle.** Each surface event is a point the edge element actually passed through while generating the
final surface (X includes runout, tilt, deflection and vibration at generation, wear recession and thermal
growth, because cutting evaluates the effective, deflected element: architecture 4.1). The surface a pass leaves
near that point is the **swept edge** placed there: the generating outline of the tool moved rigidly so that
the element sits at X. The profile along a line is the **lower envelope** of those swept edges in time order
(heights w grow away from the material):

    z(u) = min_k [ w_k + g_k(u - s_k) ]                                                  (2.1)

The local swept edge of an element moving with the unit cutting velocity v near the generating point is, to
second order in the distance t along v, the outline P (the edge's meridian, or a turning insert's outline in
the work's meridian plane) translated along v and lifted by the path curvature:

    Q(P, t) = P + t v + (kappa_v t^2 / 2) n_out                                          (2.2)

and its section by the profile plane (line normal m) is obtained by solving (Q - X).m = 0 for t (pass_polyline):
this reproduces Euler's normal curvature kappa(theta) = kappa_v cos^2(theta) + kappa_e sin^2(theta) for a
profile at angle theta to v, the silhouette g_e of the edge for theta = 90 deg (turning, floors, ball noses
fed along the inclination) and the path itself for theta = 0 (side walls: within 5 deg the parabola
w = t v_n + kappa_v t^2 / 2 is used directly). The path curvature of a rotating tool's element at radius r is the
trochoid's (Martellotti 1941):

    kappa_v = sin(kappa_e) / r / (1 + 2 Vf.t / (r |w|) + (Vf / (r w))^2)                   (2.3)

(sin kappa_e = -n_out.r_hat; Vf the feed velocity relative to the work, t the edge's direction of motion).

**Generation classes** (`gather.py`): `turn` (non-rotating tool, the work turns: cylindrical line about S z),
`floor` (rotating tool, normal within 45 deg of the tool axis: the centre line of the pass, the meridian of the
end edge/corner as template), `bore` (rotating tool fed along its axis: hole walls, cylindrical about the tool
axis), `wall` (the rest: one line per station - every tooth sweeps the same circle there - with the trochoid
parabola). Cylindrical lines live in the meridian half-plane of their axis, so a revolved surface has no
sagitta in line coordinates; cartesian lines have origin, normal and feed direction from their first event.

**Floors and the leading side.** End elements write their events at the trailing side of the feed. When the
tool axis leans so that the leading side lies lower (shift -2 r r_hat, height change < 0), the leading
passage generates the floor: a mirrored twin row half a revolution earlier is added; a floor line is then
evaluated only where both sides' valleys reach (crossed marks when both cut). The elements of one tooth pass
are one rigid edge in one pose: per tooth pass only the element nearest the tool's lowest point is kept (its
template window carries the rest).

**Minimum chip thickness and side flow** (Brammertz 1961, Grzesik 1996; Lambert 1961 and Brown & Schoop 2020
for the plastic flow of the uncut layer). A pass removes material only where the chip thickness normal to its
edge, (z_old - z_new) / sqrt(1 + g'^2), reaches h_min = (min_chip_thickness_ratio) x (edge radius at generation,
from the event). Where the pass lies below the surface by less than h_min the layer is **ploughed**, not cut:
it is pressed to the pass and, being plastically incompressible, its volume (the profile area above the pass)
flows into the free space under the pass next to the ploughed run (side flow: the samples where the pass lies
above the surface, nearest first, both sides alternating, filled up to the pass; never into the chip, i.e. not
across samples the pass cuts); what finds no free space stays where it was ploughed, spread over the run (a
uniform layer under a wide edge cannot escape sideways: steady plane ploughing returns the surface behind the
edge, Challen & Oxley 1979; a later pass then meets it as a thicker layer, as cutting's surface memory does).
Consequences, all tested: Brammertz's uncut chip tail moves to the mirror position across the intersection of
two passes, which keeps his R_t = f^2 / 8 r + (h_min / 2)(1 + r h_min / f^2) (within 1 %, f 0.125-0.353 mm,
r 0.8 mm); a flat trailing edge (a wiper or a nearly flat minor or end edge) passing over the feed-mark ridges by
less than h_min levels them into the valleys (Pt 1.7 um against 1.3 um kinematic for a 0.5 deg minor edge at
f 0.2 mm, h_min 9 um, where keeping the layer in place left 9.6 um); the profile area is conserved by every
ploughing pass. Below f = sqrt(2 r h_min) (the minimum of Brammertz's formula, where the bottom of the nose no
longer cuts in a pass) the surface is generated where the per-pass chip thickness reaches h_min on the leading
side of the nose: R_t = h_min, a regular sawtooth of period f (Brammertz's formula, read beyond its validity,
would grow as r h_min^2 / 2 f^2). `recovery_ratio` (0 by default) blends linearly between h_min and
(1 + ratio) h_min like cutting's surface memory (with 1 the peaks rise by up to h_min).

**Wear** rounds the outline: per element radial wear (r - VB_r, the edge recedes from the material) and axial
wear (z + VB_a) from the bus are interpolated along the outline's arc length between the generating tooth's
element midpoints; the element itself is placed with its worn position. Uniform wear leaves the profile shape
unchanged; wear concentrated on one side of the generating point reshapes the feed marks (test against an
independent envelope of the worn outline).

**Built-up edge.** With wear's `bue_height_m` h_B > 0 at the generating element, the edge protrudes by
delta = c_over h_B U (U uniform, stream `surface.bue`, counters (tick, tooth, element)) over a half-width
c_w h_B along the outline (BUE as an unstable extension of the tool, Trent & Wright 2000), and with
probability p_frag (stream `surface.bue_fragment`) a fragment of height c_frag h_B U' (stream
`surface.bue_fragment_height`) is welded onto the surface.

**Recut damage.** A recut row of the contact is pressed into the surface the same tooth generates nearest in
time: a dent of depth t_chip (1 - H_work / H_chip) = t_chip (1 - 1 / c_H), t_chip = h_max / (chip ratio), and
half-width c_d x depth (a chip fragment harder than the surface, pushed by the flank).

**Chatter marks.** The vibration at generation is in every event position (u_rel): chatter is in the profile
itself. Patches generated while dynamics' chatter index exceeds `chatter_index` are flagged with the chatter
frequency.

**Step-over (cross) profile** (`faces.cross_profile`). Passes of one patch with parallel feed directions
(within 20 deg) are placed across the feed at the tool tip's lateral position U and at the height of the
tool's lowest point H (each event corrected by its element's height above that point along the normal); the
template is the tool's silhouette seen along the feed, the lower envelope of the revolved outline projected
onto the cross plane (`kernels.silhouette`; Vickers & Quan 1989). This gives ball-nose scallops
h = R - sqrt(R^2 - s^2 / 4) (test, 3 %), and for inclined cutters the effective profile: a ball stays a circle
of radius R, a torus (corner r on radius R0) has the effective radius r + R0 / sin(lead) at its lowest point
(the corner-centre circle projects to an ellipse of curvature radius R0 / sin(lead), offset by r; test 1 %).

**Lay.** Feed lines measure across the tooth marks (lay = n x d); cross profiles across the scallops (lay = d).
The reported finish is the roughest direction (ISO 4288: perpendicular to the lay); `multidirectional` when
the two directions are within 30 % in Ra, `circular` / `circumferential` for faced / turned faces.

## 3. ISO evaluation (roughness.py)

Form removal by a least-squares quadratic; ISO 4288 cut-off: periodic profiles by RSm (RSm <= 0.04, 0.13,
0.4, 1.3, 4 mm -> lambda_c 0.08, 0.25, 0.8, 2.5, 8 mm; RSm evaluated with the width discrimination of the
cut-off tried, the feed spacing's cut-off tried first), otherwise by Ra (Table 1); ISO 3274 lambda_s and the
sampling spacing for the cut-off (0.5 um up to lambda_c 0.8 mm, and at most RSm / 40); the Gaussian filter of
ISO 16610-21, s(x) = exp(-pi (x / (alpha lambda_c))^2) / (alpha lambda_c), alpha = sqrt(ln 2 / pi), weights
renormalised at the ends, the lambda_c mean line evaluated every lambda_c / 50 and interpolated; half a cut-off
discarded at each end, up to 5 sampling lengths; ISO 4287 Ra, Rq, Rsk, Rku, Rp, Rv, Rz (mean of the sampling
lengths), Rt (evaluation length), RSm (height discrimination 10 % of Rz, width 1 % of the sampling length),
Rdq. A profile shorter than two cut-offs is reported with its primary-profile parameters (`method` "P",
`short_evaluation`). Each line also reports the unfiltered kinematic Pt (`kinematic_Pt_m`), the value the
analytic formulas describe. Acceptance against a specification: ISO 4288's 16 % rule over the patches of the
feature (every value for a `max` modifier, and for Rt).

## 4. Marks (marks.py)

* **Dwell and rub** (G04 in the cut, feed hold, program stop, data starvation): feed speed below
  `dwell_speed` with the spindle running and the tool cutting for at least one revolution: depth =
  max(0, u_rel.n) of the last generation (the unloaded tool springs back into the surface by its standing
  deflection); kind `dwell` / `program_stop` / `feed_stop` from the timeline block kind; the feed marks under
  the edge are removed too (a smooth mark).
* **Spindle stopping in the cut**: speed below `spindle_stop_ratio` of the running speed while cutting: a
  stop mark (deflection relief) and a finding.
* **Servo marks at reversals** (quadrant glitches): at a sign change of a linear axis's commanded velocity
  during generation, the tool-position error -sum_axis e_axis u_axis (u_axis the ISO 841 direction of the
  axis in S, R_S_M u_M) projected on the generating normal: height = peak - median within
  `reversal_window` (> 0 material left, < 0 dig). Rotary-axis reversals are not converted (section 12).
* **Witness marks** at step-overs and re-entries: overlapping passes (track distance below the tool
  diameter) whose lowest-point heights differ: height |H_b - H_a|, deflection share the difference of their
  mean u_rel.n, the remainder (thermal growth, wear, positioning) reported separately.
* A mark deeper than `mark_limit` (or than the face's Rz/Rt specification, the larger) raises a
  `finish_out_of_spec` warning at the mark's program line.

## 5. Burrs and composite exit damage (burrs.py)

Exit rows (cutting shocks of kind exit, breakthrough, breakout) give the tooth, the last element, the uncut
chip thickness h_x and the T0 angle at the exit. The exit point is that element at that angle (turning: the
tool tip in the work's meridian at the work angle of the tick); the exit face is found by casting from just
inside the generated surface along the pushing direction (cutting velocity for rotating tools, the feed for
turning) to the first material exit (`WorkpieceQuery.ray_cast`): no exit within 4 (h + r_beta) or 1 mm means
the edge left the cut inside material (counted as `exits_interior`, no burr).

    H = C_H * b_x * (sin kappa / sin 45)^e_k * (cos psi)^e_psi * F_eos * min(1, eps_f / eps_ref)
    T = C_T * (h_x + c_r r_beta)

b_x: turning, the depth of cut (it lies in the exit face the feed pushes through); rotating tools,
ap |n_x x a| + h |n_x . a| (axial depth on a side face, chip thickness on a top face). psi: angle between
the pushing direction and the exit face's outward normal. F_eos (exit order, Hashimura et al. 1999): 1 when the
tool tip leaves last (end, corner, lip, chisel element), C_eos otherwise. eps_f: true fracture strain (the file's
fracture_strain, else ln(1 + elongation)); below eps_brittle the edge **breaks out** (negative burr of depth
C_bo H; Ko & Dornfeld 1996). Calibration: C_H = 1 from Machado et al. 2003 (AISI 1045, SNMG 120408, kappa_r 45
deg: the leaned exit burr is as high as the depth of cut over ap 0.55-2.0 mm; Fig. 8: lower at 20 deg, higher
beyond 45 deg, reproduced by e_k = 1: 0.48 ap at 20 deg, 1.41 ap at 85 deg).

**Drilling** (breakthrough rows; Kim & Dornfeld 2002): the remaining disc yields when the thrust reaches its
limit load F = k pi sigma_f t^2 (Hopkins & Prager 1953, simply supported, uniform load: k = 1.5; sigma_f the
mean of yield and ultimate strength): t_c = sqrt(F / (k pi sigma_f)) with the largest thrust of the last two
steps; H = C_D t_c F_duct, T = C_TD t_c; a **crown** burr when eps_f < eps_crown (Min, Kim & Dornfeld 2001).

**Composites** (material.composite): every macro step a hole-making tool cuts, the uncut thickness h under the
tip (ray along the feed to the exit face) is compared with the Hocheng & Dharan (1990) critical thrust

    F* = pi sqrt(8 G_IC E h^3 / (3 (1 - nu^2)))

(G_IC, E, nu from composite.delamination, or its critical_thrust_N); a thrust above it is push-out
delamination: a `damage` record and an error finding (once per hole). Exits through composite edges leave
uncut / pulled-out fibres of depth C_po r_beta instead of a metal burr.

**Edges, survival and limits.** Burrs are aggregated per edge record (workpiece, brick, exit-face class): count,
largest and mean height, root thickness, exit order, exit-angle cosine, last tick and block. At each operation
end (and at finalize) a burr whose exit face is no longer material (point just behind the face outside the
field) was machined away (`removed`); a surviving burr higher than `burr_limit` (the drawing's ISO 13715 edge
allowance) raises a `burr` warning at its program line. `SurfaceView.burr_height_at` answers the burr under a
later locator or clamp (holding); `since_tick` restricts it to the burrs formed since then (the loop passes the
start of the part in the machine: the burrs of a batch's earlier parts left with them).

## 6. Surface integrity (integrity.py)

Mechanical load: the flank/edge contact ploughing the new surface is a sliding line contact; the normal force
per edge length P_n = (ploughing power / (v L_engaged)) x (Kre / Kte of the coefficients, else 1 / mu) acts over
l_c = VB + r_beta with peak pressure p0 = (4 / pi) P_n / l_c. With k = sigma_y / sqrt(3), first yield at
p_y = 3.1 k (Johnson 1985), shakedown at p_s = min(4 k, k / mu) (Merwin & Johnson 1963):

    sigma_mech = -sigma_y clip((p0 - p_y) / (p_s - p_y), 0, 1),   d_m = C_m l_c / 2

Thermal load (thermal's two-scale peak temperature of the generated surface, dT above ambient): a biaxially
constrained hot layer carries -E alpha dT / (1 - nu) (Timoshenko & Goodier 1970), relieved beyond yield, so after
cooling sigma_th = min(sigma_y, max(0, E alpha dT / (1 - nu) - sigma_y)) in tension, over the heat-penetration
depth d_th = sqrt(kappa l_c / v) of the contact time (Carslaw & Jaeger 1959). Profile
sigma(z) = sigma_mech e^(-z/d_m) + sigma_th e^(-z/d_th), the surface value clipped to +-sigma_y, the affected
depth where it falls to 10 %. Hardness: the material file's work_hardening (layer depth and increase at the
reference edge radius) scaled by min(1, p0 / p_s) and by l_c over the reference contact; white layer above
white_layer_degC (+ white_layer_hardness_ratio, depth >= d_th), over-tempering above overtemper_degC
(- overtemper_softening_ratio), burn and melt/smear flags. Each patch carries the integrity of its last
generation.

## 7. Store and bounded memory (store.py, gather.py)

A **patch** is one facet of one brick (32^3 lattice cells, removal's level-0 numbering) of one workpiece in one
operation: normal within `face_angle` and plane within `plane_tol` of the facet (at most `max_facets` per brick
and operation). Events join only the patches their operation opened (the domain's running operation count, one
per operation of every part): the next part of a batch is a new blank in the same place and the next setup has
another stock frame, so neither may join the patches before it. (Until 2026-11-09 they did: a batch's second part
joined the first part's patches, its step-over profiles compared the two parts' tracks as overlapping passes and
reported witness marks of 2-143 um on every second part; tests/unit/surface/test_domain.py.) A **pass**
is one uninterrupted generation of a patch by one tool (a gap longer than `pass_gap` and four revolutions opens
a new one). At most `lines_per_pass` **lines** per patch and pass collect pass rows; a line is evaluated and
restarted at `line_rows` (`floor_line_rows`) rows, and evaluated when no event arrived for max(pass_gap, 4
revolutions, 4 of its own intervals). Memory: open lines x rows x 31 floats (1024 x 31 x 8 B = 0.25 MB per line;
a few open at a time) plus 71 numbers per patch and 23 per pass: the reference part (300 x 200 x 100 mm, about
5000 machined patches and a few passes each) stays below 10 MB; burr edges (`max_burr_edges`), marks (`max_marks`) and thread points
(`thread_points`, counter-based thinning, stream `surface.thread_thinning`) are capped and the excess counted.

## 8. Threads (threads.py)

Events of a threading tool (tooling ThreadProfile present: inserts, taps, thread mills) are gathered per (tool,
operation). About the thread axis (S z for turning; the tool axis through the mean tool position for tapping
and thread milling) each point has (rho, z, phi); the helix is unwrapped, zeta = z - s L phi / 2 pi folded into
one pitch, s chosen by the smaller scatter, the lead L refined by golden-section search (+-1 % of nominal) on the
scatter. Flanks (normals more than 10 deg from radial, split by the sign of the normal's axial component) are
fitted with lines: half-angles atan(1 / |slope|), the included angle and its error. Major and minor diameters
from the crest and root points. Three-wire method on the simulated points: best wire d_w = P / (2 cos(alpha / 2)),
the wire resting on both flanks of the groove (centre radius rho_c from `kernels.wire_centre`, the axial
position where it sits deepest by grid + golden-section search), M = 2 (rho_c +- d_w / 2) and

    E = M - d_w (1 + 1 / sin(alpha / 2)) + (P / 2) cot(alpha / 2)     (external; internal with the signs reversed)

(Oberg et al. 2016, Machinery's Handbook; lead-angle correction neglected, < 0.1 um for standard metric threads),
plus the simple pitch diameter (ridge = groove) and the ISO 724 basic d2 = d - 0.649519 P of the measured major
diameter. On ideal ISO 68-1 profiles (M6x1, M10x1.5, M20x2.5, external and internal) E equals ISO 724's d2
within 0.05 um (test); pitch errors of 3 um and flank-angle errors of 2 deg are recovered within 0.2 um and
0.05 deg. The thread class (6g, 6H) is inspect's decision.

## 9. Domain, protocols, integration

`SurfaceDomain(tools=None, query=None, bricks=None, meshes=None)` implements core.domain.Domain:
`channels`, `init`, `kernel_inputs` (None: not in the hot loop), `absorb` (None), `macro_step(prev, nxt)`,
`end_operation` (closes passes and lines, step-over profiles and witness marks, burr survival and findings,
specification findings, thread measurement of the operation), `finalize` (the same for every operation, the
document, surface.json and the attribute arrays when `ctx.run_dir` is set), `channel_values`, `document`,
`view(state) -> SurfaceView` (`burr_height_at(workpiece, point_S_m, normal=None, radius_m=1e-3)`,
`finish_at(workpiece, point_S_m, radius_m=5e-3)`).

Protocols (protocols.py): `ToolInfo` (`edge_model(slot)`, `outline(slot) -> ToolOutline`; `GeometryToolInfo`
over tooling's ToolGeometry list - envelope chain 0 cutting pieces for rotating tools, the insert's cutting chain
with an exact nose arc for turning tools; `StaticToolInfo` + `endmill_outline`, `ball_outline`,
`turning_insert_outline` for inline data), `WorkpieceQuery` (`point_inside`, `ray_cast`: the signatures of
removal's FieldGeometry; default `StockQuery` on the analytic stock), `BrickGrid` (`brick_of`, `brick_key`,
`brick_size_m`, `coords`; `LatticeBricks` reproduces removal's level-0 numbering, `LatticeBricks.for_stock` its
stock lattice), `BrickMeshes` (`final_bricks()`; default `RunDirBricks`, the last keyframe records).
**Integration**: wire `tools=GeometryToolInfo(state.tooling.tools)`, `query=RemovalDomain.geometry(state)`
(the final field), `bricks=LatticeBricks(field.lv_origin_m[0], field.lv_cell_m[0], field.lv_dims[0])` (one
grid per workpiece via `per_workpiece`), and hand `SurfaceDomain.view(state)` to holding for burrs under
locators; run surface after cutting, motion, wear and thermal in the slow-domain order (it reads only prev).
Defaults without wiring are documented in the trace and `estimates` of surface.json.

## 10. Settings (`physics.surface`, schema.py)

`face_angle_deg` 10, `plane_tol_mm` 0.3, `pass_gap_s` 0.05, `lines_per_pass` 1, `line_rows` 1024,
`floor_line_rows` 4096, `half_spacings` 4, `turn_bins` 36, `burr_limit_mm` 0.1 (0: no burr finding),
`mark_limit_um` 2, `dwell_speed_mm_min` 1, `spindle_stop_ratio` 0.05, `reversal_window_s` 0.02,
`servo_min_um` 0.1, `witness_min_um` 0.1, `chatter_index` 0.5, `silhouette_angles` 720, `thread_points` 8192,
`max_marks` 5000, `max_burr_edges` 20000, `max_facets` 16, `every_n_macro` 1.

## 11. Results (surface.json, format cnctwin-surface)

`features [{id, op, tool, lines, Ra_m, Rz_m, Rt_m, spec_Ra_m, spec_Rz_m, spec_Rt_m, pass, patches, lay, lay_S,
direction, cutoff_m, short_evaluation, kinematic_Pt_m, integrity {residual_stress_Pa, affected_depth_m,
hardness_change, hardened_depth_m, surface_temp_max_K, white_layer, burn, overtemper, melt_smear}, chatter_marks,
bue, recut_marks, recut_dent_max_m, marks, mark_max_m}]` (one per roughness tolerance), `faces [...]` (the other
machined faces, same keys plus normal and centroid), `burrs [{edge, kind, height_m, breakout_depth_m,
mean_height_m, root_thickness_m, location_S_m, exit_normal_S, exit_order, exit_angle_cos, exits, removed, tool,
op, program, line}]`, `marks [{kind, height_m, tick, location_S_m, normal_S, patch, duration_s, revolutions,
axis?, op, program, line}]`, `damage [...]` (delamination), `threads [{id, kind, pitch_m, lead_m, pitch_error_m,
flank_half_angles_rad, included_angle_rad, flank_angle_error_rad, major_diameter_m, minor_diameter_m,
wire_diameter_m, M_over_wires_m, pitch_diameter_3wire_m, simple_pitch_diameter_m, iso724_pitch_diameter_m, ...}]`,
`patches [...]`, `summary` (counters), `estimates` (defaults in use), `models`. Attribute arrays per brick vertex:
`Ra_m Rz_m Rt_m lay_x lay_y lay_z burr_height_m residual_stress_Pa hardness_change white_layer burn
mark_height_m` (NaN where no machined patch matches the vertex normal and plane).

## 12. Validity, error bounds and how they were established

| quantity | bound | established by |
|---|---|---|
| turned feed marks Pt | exact circle envelope (0.5 % below f^2 / 8 r at f / r = 0.4) | tests, f = 0.1-0.3 mm, r = 0.8 mm |
| Ra of the parabolic profile | 3 % vs f^2 / (18 sqrt 3 r) after the ISO filter | test |
| minimum-chip term | 1 % vs Brammertz's formula | test, f 0.125-0.353 mm |
| ploughed layer (side flow) | profile area conserved to 1e-12; a 0.5 deg trailing edge keeps Pt within 0.25 h_min of the kinematic Pt | tests |
| turned Ra vs measured | -13 % ... +21 % (mean 11 %) | Cirstoiu 2024, 42MoCr11, TNMG 220408, f 0.125-0.353 mm, r_beta 25 um assumed |
| wall feed marks, runout marks | 2 % vs fz^2 / 8 R_eff, (N fz)^2 / 8 R_eff; RSm = N fz within 5 % | tests |
| floor marks, leading-side floors | 2-3 % vs an independent envelope of the outline | tests |
| ball-nose scallop | 3 % vs R - sqrt(R^2 - s^2/4) | test |
| inclined ball / torus profile | 0.05 um vs the sphere; 1 % vs r + R0 / sin(lead) | tests |
| ISO filter | 50 % transmission at lambda_c within 0.3 % | test |
| 3-wire pitch diameter | 0.05 um vs ISO 724 on ideal profiles; 2 um with 4000 thinned points | tests |
| exit burr height | factor 2 (-50 % / +100 %) for ductile metals; the published case reproduced within 20 % | Machado 2003 |
| drilling burr | factor 2-3 (coefficients estimated) | plastic-collapse scaling |
| delamination onset | Hocheng-Dharan closed form (about +-20 % vs their drilling tests) | published model |
| residual stress | +-150 MPa or +-50 %, depth within a factor of 2 | model class (line-contact shakedown, thermal relief) vs typical XRD scatter |

Validity: feed marks with spacing well below the generating radius (second-order path expansion; the parabola
error is (t / r)^2 / 4 relative, < 0.1 % for t / r < 0.06); rigid generating outline per tooth pass (per-station
deflection differences enter through the event positions, not the template shape); ISO profile rules for
lambda_c 0.08-8 mm; burr model for ductile metals and the exit kinematics above; integrity for metals with a
stated yield stress (composites: delamination and pull-out only).

## 13. Assumptions and limitations (with consequences)

* Event sampling is cutting's: walls have one envelope point per element and tooth pass, floors the trailing
  passage (leading twins added from the tool axis tilt; the leading-side deflection is taken equal to the
  trailing one: error up to the deflection difference, typically < 1 um), turning one event per
  `surface_event_deg` of work rotation.
* Side flow is volume-conserving lateral flow of the ploughed layer into the free space under the generating
  pass, nearest first; its split between the two sides, the filling up to the pass and the absence of flow into
  the chip are the model's choices (none is measured). Material pushed into the chip would lower the profile;
  burrs rising above the pass (outside the pass's envelope) are not represented. At feeds below
  sqrt(2 r h_min) the profile carries the h_min sawtooth (R_t = h_min): the measured low-feed roughness then
  follows h_min, i.e. the edge radius and min_chip_thickness_ratio, which most cases do not state (Magalhaes 2026,
  r 0.2 mm, f 0.025-0.05 mm with the 35 um default insert hone: Ra 2.3 um simulated, 0.44-0.92 um measured).
* BUE overcut, fragment and recut-dent coefficients are estimates (db/models/surface-roughness.yaml): the BUE
  and recut contributions to Rt are indications within a factor of 2.
* Servo marks are converted for linear axes named X, Y, Z (ISO 841); reversals of rotary axes and of axes with
  other names leave no servo mark (their contour error still reaches the surface through the event positions).
* Witness marks use the pass heights of overlapping tracks; turning re-entries are not compared (the later pass
  removes the earlier one where they overlap).
* Burr heights use the exit-face orientation from the workpiece query: with the analytic stock (no removal
  field wired) only exits through the stock outline are found.
* The thread lead-angle correction of the three-wire method is neglected (< 0.1 um for standard pitches).
* The surface integrity model is a load-class correlation, not an elastic-plastic FE solution: compressive
  mechanical and tensile thermal parts superposed.
* Cost (measured on this box under load average 7-10, single core): about 5 ms per macro step for a D10
  four-flute end mill finishing a wall and a floor at 10 000 rpm (~200 events per step, 0.5 s per simulated
  second including the line evaluations), about 2.5 ms per step for turning at 1000 rpm (6 events per step).
  This exceeds the surface's share of the L2 budget of architecture 20 for dense finishing; `lines_per_pass`
  (cost is proportional) and `line_rows` set the trade-off.

## 14. Kernels (architecture 5.5)

All `@njit(explicit signature, cache=True, nogil=True, error_model="numpy")`, one signature each (test), each
with a plain-Python twin in reference.py compared in tests/unit/surface/test_kernels.py (1e-18 m absolute):
`pass_polyline(PASSES_T, TEMPLATES_T, i8, f8[::1], f8[::1], i8) -> i8`,
`envelope_profile(PASSES_T, TEMPLATES_T, GRID_T, i8, f8)` (with `_place_ploughed(GRID_T, i8, i8, i8, i8, f8)`), `pass_valleys(PASSES_T, TEMPLATES_T, GRID_T, i8, f8,
f8[::1]) -> i8`, `gaussian_mean(f8[::1], f8, f8, f8, i8, f8[::1])`, `profile_params(f8[::1], f8, i8, i8, i8, f8,
f8, f8[::1])` (with `_rsm_core`), `silhouette(f8[::1] x 7, i8, f8, f8, f8[::1])` (with `_raster_min`),
`wire_centre(f8[::1], f8[::1], f8, f8, i8) -> f8`. Layouts in abi.py (`Passes`, `Templates`, `Grid`).

## 15. RNG streams

`surface.bue` (BUE overcut per pass), `surface.bue_fragment` (fragment occurrence), `surface.bue_fragment_height`
(fragment height), `surface.thread_thinning` (thread point budget); all `hash_uniform(seed, stream, tick, tooth,
element)` or the arrival index: no state, resume-exact.

## 16. Parameters

db/models/surface-roughness.yaml: min_chip_ratio 0.3 (est; the material's ratio first), recovery_ratio 0
(brammertz-1961-roughness; the ploughed layer's side flow has no parameter), gauss_alpha 0.4697 (iso-16610-21-2011), sampling_lengths 5, cutoff_by_rsm_m and
cutoff_by_ra_m (iso-4288-1996), stylus_table_m (iso-3274-1996), rsm discriminations 0.1 / 0.01 (iso-4287-1997),
bue_overcut_ratio 0.5, bue_width_ratio 2, bue_fragment_probability 0.2, bue_fragment_height_ratio 0.3,
recut_chip_hardness_ratio 1.6, recut_dent_width_ratio 4 (est). db/models/surface-burr.yaml: rollover_height_ratio
1.0 (machado-2003-burr-turning), rollover_kappa_exponent 1, rollover_thickness_ratio 1, edge_radius_ratio 1,
exit_angle_exponent 1, eos_top_last_factor 0.5, brittle_fracture_strain 0.05, ductile_reference_strain 0.1,
breakout_depth_ratio 0.5, drill_plate_load_factor 1.5 (hopkins-prager-1953-plates), drill_burr_height_ratio 1,
drill_burr_thickness_ratio 0.5, crown_fracture_strain 0.15, pullout_depth_ratio 2 (est).
db/models/surface-integrity.yaml: yield_onset_ratio 3.1 (johnson-1985-contact-mechanics), shakedown_ratio 4
(merwin-johnson-1963-rolling-contact), friction_default 0.4, mech_depth_ratio 2, wh_reference_edge_radius 20 um,
white_layer_hardness_ratio 0.3, overtemper_softening_ratio 0.15 (est).

## References

* Boothroyd, G.; Knight, W.A. (2006) Fundamentals of Machining and Machine Tools, 3rd ed., CRC.
* Brammertz, P.H. (1961) Industrie-Anzeiger 83(2), 25-32. Grzesik, W. (1996) Wear 194, 143-148.
* Brown, I.; Schoop, J. (2020) An iterative size effect model of surface generation in finish machining, J. Manuf.
  Mater. Process. 4(3), 63 (material below h_min deforms plastically and side-flows; Lambert 1961).
* Challen, J.M.; Oxley, P.L.B. (1979) Wear 53, 229-243 (wave model of plane ploughing).
* Carslaw, H.S.; Jaeger, J.C. (1959) Conduction of Heat in Solids, 2nd ed., Oxford.
* Cirstoiu, A. (2024) Scientific Bulletin of Valahia University - Materials and Mechanics 20(22).
* Gillespie, L.K.; Blotter, P.T. (1976) ASME J. Eng. Ind. 98(1), 66-74.
* Hashimura, M.; Hassamontr, J.; Dornfeld, D.A. (1999) ASME J. Manuf. Sci. Eng. 121(1), 13-19.
* Hashimura, M.; Chang, Y.P.; Dornfeld, D.A. (1999) ASME J. Manuf. Sci. Eng. 121(1), 1-7.
* Ho-Cheng, H.; Dharan, C.K.H. (1990) ASME J. Eng. Ind. 112(3), 236-239.
* Hopkins, H.G.; Prager, W. (1953) J. Mech. Phys. Solids 2, 1-13.
* ISO 1302:2002; ISO 3274:1996; ISO 4287:1997; ISO 4288:1996; ISO 16610-21:2011; ISO 13715:2017; ISO 68-1:1998;
  ISO 724:1993; ISO 965-1:2013.
* Johnson, K.L. (1985) Contact Mechanics, Cambridge. Merwin, J.E.; Johnson, K.L. (1963) Proc. IMechE 177, 676-690.
* Kim, J.; Dornfeld, D.A. (2002) ASME J. Eng. Mater. Technol. 124(2), 192-198.
* Ko, S.L.; Dornfeld, D.A. (1991) ASME J. Eng. Mater. Technol. 113(1), 75-87; (1996) J. Mater. Process. Technol.
  62, 24-36.
* Machado, A.R.; Kaminise, A.K.; da Silva, M.B.; Ariza, R.G. (2003) Study on burr formation in turning, COBEM 2003.
* Martellotti, M.E. (1941) Trans. ASME 63, 677-700.
* Min, S.; Kim, J.; Dornfeld, D.A. (2001) J. Mater. Process. Technol. 113, 4-9.
* Oberg, E. et al. (2016) Machinery's Handbook, 30th ed., Industrial Press.
* Pekelharing, A.J. (1978) CIRP Annals 27(1), 5-10.
* Timoshenko, S.P.; Goodier, J.N. (1970) Theory of Elasticity, 3rd ed., McGraw-Hill.
* Trent, E.M.; Wright, P.K. (2000) Metal Cutting, 4th ed., Butterworth-Heinemann.
* Vickers, G.W.; Quan, K.W. (1989) ASME J. Eng. Ind. 111(1), 22-26.
