# Material removal domain (`cnctwin/removal/`)

The in-process workpiece of every operation: a tri-dexel field (Van Hook 1986; Benouamer & Michelucci
1997; Zhang & Leu 2009) of the stock in the stock frame S, a polar field for turning, the CAD target on the
same rays, and the kernels the compiled loop (`core.hotloop.run_macro`) calls to subtract tool envelopes,
gate the cutting elements, detect collisions and answer geometric queries. At L2 the domain publishes
the removed volume (the core Ledger books it), finds loose bodies, syncs the polar field into the
tri-dexel, raises findings and carries the workpiece from one operation to the next. Architecture: 5.3
(lifecycle), 5.4 (hot loop, gating, turning, mass), 5.5 (kernel rules), 10 (volumes), 19 (workpiece).

| file | what |
|---|---|
| `abi.py` | every layout (NamedTuples of C-contiguous writable arrays) and index constant |
| `nbt.py` | Numba types of the layouts; compile options `OPTS`, `INLINE`, `RAW` |
| `field.py` | lattice, levels (refine boxes), pool growth, exact volumes, memory, checkpoint arrays |
| `stock.py`, `k_stock.py` | block / cylinder / tube / mesh stock, CAD target (watertight rasterisation) |
| `profile.py` | exact max-radius profiles from tooling envelopes, tooth slabs, tap sweeps |
| `k_core.py`, `k_ray.py` | array-level ray primitives (lookups, pool load/store, bricks, D_f, classify) |
| `k_envelope.py`, `k_subtract.py` | ray / revolved-envelope intervals (line, sphere, torus quartic); subtraction |
| `k_helix.py` | tap tooth profiles swept along the relative helical path |
| `k_sd.py`, `k_gate.py` | interpolated boundary; engaged intervals, contact geometry, engaged-sector box |
| `k_query.py`, `views.py` | ray cast, surface point, surface height, fill fraction, wall thickness, broad phase, probe; protocol views |
| `k_collide.py`, `collision.py` | non-cutting envelope parts, fixture / machine meshes, swept checks |
| `k_connect.py`, `connect.py` | connectivity of the column graph, loose bodies |
| `mctable.py`, `k_mesh.py`, `mesh.py` | marching cubes per brick with exact crossings, welding, keyframes |
| `polar.py`, `k_polar.py` | polar field: resolution, cut per L0 step, radius, exact tri-dexel rebuild |
| `transfer.py`, `k_transfer.py` | flips and transfers between operations (exact lattice remap, else resampling) |
| `k_drive.py` | `tooth_pass`, `l1_contact`, `excess_material` (compositions for the hot loop) |
| `kernels.py`, `reference.py` | kernel index; plain-Python references of every kernel |
| `state.py`, `schema.py`, `channels.py`, `domain.py` | domain plumbing |

Model parameters: `db/models/removal.yaml` (loose-body hazard sizes), sources in
`db/bibliography/models-removal.yaml`. Numerical settings: `physics.removal` (`schema.py`, section 13).
RNG streams: **none** (removal is deterministic geometry; no stochastic phenomenon is modelled here).

## 1. Layouts (`abi.py`)

### 1.1 Tri-dexel field `DexelField`

* **Levels** share the arrays: level 0 is the base lattice over the stock (`physics.lattice_mm`, spacing
  `a`), levels >= 1 are refine boxes (`physics.refine`, spacing `a / factor`, whole quanta). Level `l` has
  nodes `origin[l] + (i, j, k) a_l`; `origin = lo + a/2 - margin a`: the stock faces lie half way between
  nodes and `margin_nodes` (default 1) layers of empty nodes surround the stock (meshes close; gating
  treats points beyond the outermost nodes as air).
* **Families**: family 0 along x through (y_j, z_k), 1 along y through (x_i, z_k), 2 along z through
  (x_i, y_j); the families share the nodes. Ray index `ray0[l, f] + iu dims[l, v] + iv`.
* **Segments**: sorted, disjoint, non-touching half-open `[lo, hi)`, endpoints int32 fixed point in
  `QUANTUM_M = 10 nm` from the level origin (21 m range). `K_INLINE = 4` segments inline, the rest in pool
  blocks of 4 linked by int32 `link`/`pnext` (free list, deterministic index order). Every endpoint carries
  the int32 surface event id of the cut that made it (`SEV_STOCK = 0` for the stock).
* **Bricks** of 32 cells: `bmat[brick, f]` material length of the family-f rays the brick owns, `bdirty`
  (keyframes), `bconn` (connectivity), `bver` (version per brick).
* `info` (`FI_*`): version, levels, rays, overflow flag, free list, pool size, largest segment count,
  peak pool use, workpiece, topology counter (`FI_TOPO`: a level-0 z ray changed its segment count).

Write semantics: a write that needs more pool blocks than are free is refused (`FI_OVERFLOW`, ray
unchanged); every kernel that writes is idempotent, so the loop grows the pool (`field.grow_pool`,
doubling) and repeats the step (`RemovalDomain.needs_replay` / `prepare_replay`).

Volume: `V = a^2 QUANTUM_M sum(z-family lengths)` of level 0, an exact integer sum (architecture 10). The
x and y families give two further estimates of the same material; their spread is the representation
uncertainty of the field (published per contact as `removed_volume_xyz_m3`).

### 1.2 Other layouts

| layout | content |
|---|---|
| `TargetField` | CAD part on the same rays (CSR: `start [nray + 1]`, `seg [n, 2]` int32 quanta) |
| `Profile` | max-radius profile R(z) about +z from the tip: sorted sub-pieces, each a line or a z-monotone arc (`PK_LINE`, `PK_ARC`), part tag (tooling `PART_*`), `info` (z range, R_max, dead radius) |
| `SubtractSpec` | one subtraction: axis, per-slab axis points (S) and radius offsets, slab heights, box, ints (event id, tick, contact, rapid, plunge, target check, level mask), floats (gouge tolerance, dead zone, slack) |
| `ToothSlabs` | per tooth of a path's tool the slab heights and radius offsets (`profile.tooth_slabs`) |
| `HelixSweep` | tap lands: convex polygons in (z_T, r) per land, land phases, hand and box |
| `PolarField` | turning: `geom` (bins, rows, axial columns), radial rays `rcnt/rseg/rsev [nth nz, KPR, 2]`, axial rays `acnt/aseg/asev [nth nr, KPA, 2]`, `zmax_q` (largest radius per bin and block of `POLAR_BLOCK = 16` rows), `info` (`PI_*`: version, overflow bits, dirty ranges, KPR, KPA) |
| `InsertRegion` | non-rotating tool region in (z_T, x_T): lower / upper chains in z, left / right chains in x, exact nose arc, box `[6]` (z, x ranges, upper-chain peak z, unimodal flag) |
| `RemovalAccum` | per contact: removed quanta per level and family, `ivals` (`IA_*`), `fvals` (`FA_*`) |
| `Scratch` | per caller work lists (segments, pieces, intervals) and small fixed work (`fw`, `iv2`, `tg`, `iw`, `rows`) so kernels allocate nothing per ray |
| `RemovalInputs` | what `kernel_inputs` hands to the loop (section 12) |

## 2. Stock and target (`stock.py`, `k_stock.py`)

Block `[0, sx] x [0, sy] x [0, sz]` in S; cylinder / tube with the axis on S z, `z in [0, L]`; mesh stock in
S. Analytic fills (`fill_box`, `fill_tube`) write the exact chords (circle chords to the quantum). Meshes
(stock or CAD target, via `cnctwin.geom.io`) are rasterised per ray family with a watertight scheme:
canonical edge functions with the top-left rule (Pineda 1988; Woop, Benthin & Wald 2013) so every ray
crossing an edge or vertex is counted exactly once, crossings sorted and the winding number decides the
inside (mode 0), parity (1) or the union of both shells (2). Error: crossing positions to the quantum
(5 nm rounding); a mesh that is not closed is reported (`open` rays) and the winding rule keeps the
consistent side. Verified: a meshed block equals the analytic block on every ray (test_02).

## 3. Subtraction (`k_envelope.py`, `k_subtract.py`)

One call subtracts the swept envelope of one tooth pass (architecture 5.4 "Removal"): the tooth's
envelope at its generation instant, per axial slab `k` with that slab's axis point `c_k = X_e + u_rel,k`
(bent tool, bent wall, mapped to the material frame by the loop) and radius offset `dr_k` (the tooth's
radius relative to the profile: runout, wear, tooth-to-tooth differences), within the engaged-sector box.

**Ray against one slab.** For the ray `x(tau) = p0 + tau d` (d a lattice axis, tau from the point nearest
the slab axis point so all coefficients are of tool size): `z(tau) = z0 + s tau` along the axis (s = d.w),
`r^2(tau) = A0 + A1 tau + A2 tau^2`. Per profile sub-piece clipped to the slab:

* line (cylinder, cone): `r^2 <= (R0 + R1 tau)^2`, `R0 + R1 tau >= 0` - one quadratic in the
  cancellation-free form (Press et al. 2007, 5.6); exact;
* arc centred on the axis (ball ends, sphere): one quadratic; exact;
* arc off the axis (corner radius, ball with a radius offset: torus): boundary roots of the implicit
  quartic `(r^2 + (z - zc)^2 + rc^2 - rho^2)^2 - 4 rc^2 r^2`, isolated through the derivative cascade (P''
  quadratic, P' and P monotone between the roots of the next derivative), refined by safeguarded Newton
  to 1e-11 m, inside test of each sub-interval at its midpoint on the explicit form; error < 1e-9 m;
* parallel (d = +-w) and perpendicular (d.w = 0) rays: closed forms.

Intervals of all slabs and pieces are merged (touching within 1e-12 m), quantised to the nearest quantum
and subtracted from the ray; every removed piece is recorded. The removed volume is the exact integer sum
of removed z-family quanta (the X and Y family sums are kept too). Tool axes in any direction (5-axis)
use the same algebra.

**Passes.** The hot pass walks the rays of the box (rejects: no segments; distance to the axis beyond
R_max + max dr + slab spread, decided without the square root except within 1e-12 of the bound; no material in
the box range) and computes, quantises and subtracts; `post_pieces` books the pieces into the bricks (`bmat`,
`bdirty`, `bconn`, `bver`), checks the target and the dead zone and accumulates the contact sums; events are
written once per call. What each ray costs is limited to what can change it:

* **slabs**: only the slabs that can reach the ray's material are evaluated. The call leaves the spread of the
  slab axis points (|c_k - c_0|, so a slab's frame differs from slab 0's by at most that along the axis) and
  the ray's material extent in the box range in the scratch (`FW_ENV`, read by `_env_ray` as `work[16:]`);
  the ray's height along the axis over that extent (widened by a quantum and the spread) gives the slab
  window by a search over the ascending slab heights (a guess from the mean height, then a walk), and a slab
  is skipped when the ray's height over the box range misses it by more than 1e-10 m. Intervals that cannot
  reach the ray's material change nothing, so the subtraction is that of all slabs (a full circle of 33-41
  slabs per tooth collapses to one or two per ray);
* **rays already clear**: a ray whose quantised intervals meet no material (the columns under the tool that
  the previous teeth cut down to the floor) is left without loading or writing it;
* **trims**: a ray of one inline segment that loses one end is written in place (the new end with the
  pass's surface event id, as load / subtract / store would write it);
* **booking**: pieces are booked in batches of 512 (they stay in the cache; the order of every sum is the
  same); a piece within one brick updates it directly (the next piece strictly inside the same brick reuses
  its index), and the brick indices come from a multiplication by the reciprocal with exact integer
  corrections instead of integer divisions; the contact sums run in registers;
* **box**: the envelope box takes, per axis, the extent of the slab axis points along it (their 3-D spread
  only for the quick-reject radius), so the columns under the tool whose material ends at the envelope's
  bottom fall outside it (the box contains the envelope; margin 1e-12 m);
* **axis rays**: on a ray along an axis-aligned tool axis a cylinder piece is the test A0 <= R0^2 on the
  slab's range (the branch of the quadratic that these coefficients take).

Every field array, accumulator and event is bit-identical to the evaluation of every slab on every ray of the
box (200 tooth passes of the slot and 120 of the pocket's helical entry, `tests/unit/gpu` parity of the GPU
path, which compiles this kernel with its own interval lookup and needs `_env_ray` called with the box range
for every candidate ray: the window is inside `_env_ray`, not in its arguments).

**Events** (aggregated, one row per kind and call, `EV_*`, columns `EC_*`):
* rapid into material (`SI_RAPID`: any removal during a rapid; volume and centroid);
* gouge: removed target material deeper than `gouge_tolerance_m` along a ray (depth = the smallest over
  the families of the per-family largest overlap: the family most normal to the gouged face);
* breakthrough: a target segment left without any material on its ray;
* non-centre-cutting plunge: material removed inside the dead cylinder (radius `PI_RDEAD`, or half the
  core diameter; height `R tan(end_dish_rad)` above the lowest envelope point, R the tool radius) while the
  tool feeds axially faster than the ramp limit `tan(end_dish_rad)` (tooling model parameter);
* pool overflow (the caller retries).
Collisions of non-cutting parts are a separate kernel (section 8).

**Contact sums** (`FA_*`): centroid and outward normal of the removed z-family volume (normal from the
removed point toward the tool axis at its height), extent of removed material along the axis from the tip
(ap) and largest distance from the axis.

Error: endpoints to the quantum (5 nm); envelope geometry exact (quadratics) or < 1e-9 m (torus). The
dexel representation: a ray carries material only along its line, so the volume of a pass is the integer
sum over the rays it crosses; for a wall parallel to a family the z-family volume of a single pass is
quantised in columns of a x a (a pass of width f_z < a removes whole columns every a / f_z passes), exact
on average over passes; the per-step consistency ratio accounts for this (section 13).

## 4. Taps and thread mills (`k_helix.py`, `profile.helix_from_thread`)

Taps and threading tools are never revolved bodies (architecture 5.4): the tooth profile of every land
(convex polygons in the meridian half-plane) is swept along the relative helical path. Within one call
the tip moves linearly with the spindle angle (feed / speed mismatch of a floating or rigid cycle comes
from the motion); a point at T0 azimuth alpha is passed by land l at `theta* = alpha - h phase_l + 2 pi k`
(every passage inside the call's angle range); there the tool is at `p(theta*)` and the point is cut when
`((x - p).w, |(x - p) x w|)` lies in one of the land's polygons. Along a ray the inside function (largest
outward edge distance) is sampled at most every `min(a/2, pitch/16)` and each sign change bisected to
1e-10 m. Thread mills (annular rows, rotating) use the revolved envelope of `profile_from_thread`.
Verified: a V tooth swept two turns in a 6 mm bore removes the Pappus volume within 5 % (lattice 0.25 mm,
groove 0.6 mm deep: the error is the lattice's, test_09).

## 5. Gating (`k_sd.py`, `k_gate.py`)

**Interpolated boundary D_f.** For a point and a family f, the four rays of f around it give the signed
distance along the ray to their nearest endpoint (+ inside); `D_f` is their bilinear interpolation. A plane
boundary crossing all four rays has endpoints linear in the transverse coordinates, so `D_f = distance /
|n . e_f|` exactly; a curved boundary has the zero-set error `a^2 kappa / (8 cos^2 theta)` (bilinear
interpolation, de Boor 1978 ch. 2), at most `a^2 kappa / 4` for the most normal family (theta <= 45 deg).
The most normal family has the smallest endpoint spread; it is valid when all four rays have an endpoint
within `SPREAD_CELLS` cells. `classify`: the family most aligned with the tool axis decides alone when its
four rays agree and are farther than `SPREAD_CELLS` from an endpoint, else the valid family with the
smallest spread.

**Engaged intervals** (architecture 5.4, `abi.EngagedIntervals`): per station circle `c + r (cos phi x0 +
sin phi y0)` (T0), once per tooth period:
1. reject when no brick with material meets the circle's box;
2. sample every `gate_sample_cells` lattice cells along the circle (at least 16 samples); a sample whose
   stencil lies in empty bricks or beyond the outermost nodes is air without classification (what
   classify would answer);
3. locate each state change between neighbouring samples by Illinois regula falsi (Dowell & Jarratt 1971)
   on D_f of the family with the smallest spread at the bracket, warm-started at the previous interval
   boundary that lies in the bracket, to `gate_tolerance_m` along the arc; bisection of the
   classification when no family brackets;
4. intervals `[lo, hi]`, `lo in [0, 2 pi)`, `hi <= lo + 2 pi`; more than the capacity merge the smallest gap.

Cost: a sample's lattice coordinates are computed once for the empty-brick test and the three families'
interpolations (a horizontal circle, x0 and y0 without a z component, has one height for all its samples;
such a circle beyond the outermost nodes is air without sampling). A sample of a horizontal circle of a tool
along z whose four z columns each hold one segment ending at or below it, all farther than SPREAD_CELLS cells
or all at the same height within that, is air without further work (the z family decides alone, or is valid
with zero spread and D <= 0: what classify answers; the end-edge stations above a floor, the air behind the
cut). Otherwise, after the family most aligned with the
axis, the others are followed only while they can still change the choice: a valid family with zero endpoint
spread decides at once (the choice is strict), a family stops at its first ray that shows it is invalid or
cannot beat the best spread, and the smallest-spread choice among invalid families is evaluated in full only
when none is valid. The refinement evaluates the families at the bracket's middle and ends from coordinates
computed once. The intervals are those of the full classification, bit for bit.

Error: planar walls and floors exact to the tolerance and the quantum (verified for walls at 0/15/30/45
deg to the lattice, slots, plunges, ramps, a ball nose on an inclined plane: test_04); curved walls `a^2
kappa / 4` normal to the wall (verified on circular walls, test_04), divided by the sine of the crossing
angle along the circle; a new engagement narrower than the sample spacing s along the circle is found
once it reaches it (undetected depth below `s^2 / (8 r)`: 1.6 um for s = a = 0.25 mm, r = 5 mm). The
inner (freshly cut) boundary is never gated here (cutting's surface memory, architecture 5.4); for feed
per tooth below the lattice the gating sees the outer boundaries only (verified: f_z < a, test_04).

**Contact geometry** (`contact_geometry`): largest engaged angle and arc, radial engagement ae across the
feed direction at the largest engaged radius, engaged edge length (sum of the engaged stations' element
lengths). **Engaged-sector box** (`sector_box`): the union of the engaged angle intervals as a wedge of
radius `R_max + max dr` from the axis, swept over the envelope's axial range, grown by `sector_margin_cells`
(material outside it inside the envelope is in the trailing half, removed by the previous tooth).

## 6. Queries (`k_query.py`, `views.py`)

| query | model | error |
|---|---|---|
| `point_inside`, `classify` | finest level containing the point; D_f near boundaries | plane exact; curved a^2 kappa/4 |
| `ray_cast` | brick 3D-DDA (Amanatides & Woo 1987) over occupied bricks, half-cell samples, crossing refined on D_f by regula falsi to `tol`; outward normal from D_f's gradient | oblique plane 3 nm (test_05) |
| `surface_point`, `refill_points` | along n: ahead from air, behind from material; h = (p - M).n for stale surface-memory bins | as ray_cast |
| `surface_height` | four rays of the axis family: bilinear in their extreme endpoints when they agree within a cell (plane), else nearest ray | plane exact; edges one cell |
| `fill_fraction` | level-0 z family as a x a columns distributed over a coarse grid by exact overlap | exact for the column model |
| `wall_thickness` | entry and exit along -n | as ray_cast |
| `broad_phase` | occupied level-0 bricks inside the envelope's box at the pose | conservative |
| `sweep_contact` | stylus (ball + shank) moving linearly: leading-half samples cast along the move, earliest refined 3x by 4x sub-sampling | plane 0.07 um, bore 0.2 um (test_05) |
| connectivity | section 9 | |

`views.FieldGeometry` serves thermal (`WorkGeometry`: stock bounds, fill fraction, changed boxes, version,
rotation radius) and chips (`ChipGeometry`: surface height of the part).

## 7. Air cutting

`l1_contact` runs the broad phase of every path at every L1 tick; feed motion (spindle on) without an
occupied brick inside the cutting envelope's box is air cutting: ticks and first / last tip are
accumulated, the domain publishes `air_cut_time_s` per contact and reports runs longer than
`air_cut_report_s` (info finding with the start and end positions).

## 8. Collision (`k_collide.py`, `collision.py`)

* `collide_envelope`: material inside the whole envelope but outside the cutting envelope, level 0. Only
  the axial range and radius of the non-cutting pieces (part != `PART_EDGE`: body, neck, shank, holder,
  spindle, insert body) are searched; their box is rejected by brick occupancy while the tool cuts
  normally (0.3 us). Overlap beyond `collision_tolerance_m` along a ray is an `EV_COLLISION` with the part.
* `mesh_field_overlap`: a closed mesh (fixture, machine part) at a pose against the material: its interior
  intervals on the field's rays (watertight rasterisation) intersected with the material.
* `swept_mesh_field`: motion between two poses sub-sampled so no point moves more than
  `sweep_step_cells` cells (a rapid cannot tunnel through a feature thicker than that); poses interpolated
  as rotation + translation (never a matrix lerp).
* `collision.envelope_vs_mesh`: the whole tool envelope against a static mesh with cnctwin.geom's exact
  revolved-solid / triangle clearance on the mesh's static tree, at sub-sampled poses.
Machine-part collisions need the link poses from the motion / kinematics domain (integration).

## 9. Loose bodies (`k_connect.py`, `connect.py`)

The material graph is the level-0 z family: segments are nodes, segments of 4-neighbouring z rays whose z
ranges overlap are joined (face-connected columns of the column model the ledger books). Union-find with
path halving, deterministic union (smaller root wins). Components not meeting a held region (the
workholding contact boxes of the setup; without any, the largest component is held) are loose: volume,
centroid and box measured on the same columns, then removed from every ray of every level. A body below
`slug_max_cube_edge_m^3` (12.6 mm: 2000 mm^3) is a `chip_hazard`, a larger one `part_free`; below
`swarf_cube_edge_m^3` (1 mm^3) it is booked but not reported (db/models/removal.yaml, estimated). All
are published in `loose_bodies` (the core Ledger books them as `freed_body_mass_kg`). The check runs only
when a level-0 z ray changed its segment count (`FI_TOPO`): shortening columns cannot separate the graph.
Error: webs thinner than a cell may join or separate one cell early or late.

**Incremental check** (`connect.release_loose_incremental`, `k_connect.local_connected`, every macro step).
While the material is known to be one component (`Workpiece.one_body`: the last check found one body, or one
held body was left after a release) and only subtractions changed it since (each marks the bricks it changed
in `bconn`), only the changed region is labelled: the rectangle R of node columns of the brick columns with
a mark (a brick's nodes and the next boundary node, which a change there marks too) and a ring of one column
around it, unchanged. When that labelling is one component and it reaches the ring, the material is still one
body: a path from any column outside into R crosses the ring, and every column outside joins the ring's
component as it did before. Otherwise (the cut's two sides join only outside the region, the region holds no
ring material, or more than one component) the whole field is labelled as above, so a body separates at
exactly the macro step it would with the full labelling: verified for a trepanned core (the step that closes
the ring through the stock), a part on two tabs (the step that cuts through the second) and a blind groove
(no release, every step decided locally; `test_06`). After every check the marks and `FI_TOPO` start from
zero; a polar sync or a transfer (which may add material) and the operation's end label the whole field.
`one_body` is kept in the checkpoint (meta `one_body`, default false: the first check labels everything).

## 10. Meshing (`mctable.py`, `k_mesh.py`, `mesh.py`)

Marching cubes (Lorensen & Cline 1987) per 32^3 brick with the exact endpoints as edge crossings: a cell
edge along axis f lies on a ray of family f, so its vertex is that ray's endpoint inside the edge. Node
state: majority of the three rays through the node. The case table resolves ambiguous faces with the face
rule (Nielson & Hamann 1991; Montani, Scateni & Scopigno 1994) consistently on both sides, so the mesh is
closed (verified: no open edge, test_07); lattice-boundary nodes count as air. Every vertex carries the
surface event id of its endpoint and a global edge id (welding bricks exactly). Keyframes mesh only the
bricks marked dirty; an optional displacement grid (trilinear) moves vertices to X + d(X) (as-cut, cold,
unload geometry). Polar fields are meshed through their exact tri-dexel rebuild. Error: vertices exact on
edges; faces are chords (a^2 kappa / 8 on curved surfaces); sharp edges chamfered within one cell.

## 11. Polar field: turning (`polar.py`, `k_polar.py`)

The workpiece turns about S z; S rotates with it. Bin b covers `theta in [b dth, (b + 1) dth)`
(`n_theta >= physics.polar_theta_bins`, default 720), row i covers `[z0 + i dz, z0 + (i + 1) dz)` with
`dz <= min(f_min / 8, r_nose / 16)` (`f_min = polar_min_feed_m_per_rev`), coarsened to fit `polar_max_bytes`
with a warning finding. The radial ray of (b, i) at the bin and row centres holds r segments; the axial ray
of (b, j) at radius `(j + 1/2) dr` (`dr = max(dz, a/2)` unless `polar_axial_spacing_m`) holds z segments
(faces, shoulders, grooves).

**Cut per L0 step** (`polar_cut`): every bin whose centre angle the tool's reference plane passes in the
step (rotating-frame angle th0 -> th1) is cut with the insert region at the tool position interpolated to
that angle (helical feed marks and threads come out exactly), for external (`sx = -1`) and internal tools
and a centre-height offset `yoff` (radius `sqrt(rho^2 + yoff^2)`). Radial rows are taken in blocks of 16:
a block is skipped when its largest radius (`zmax_q`, kept exact) is not above the insert's deepest point
there (first overall, then over the block's rows: O(1) for a unimodal upper chain); in the remaining rows
the deep side is evaluated first. The removed volume is the exact integer `sum delta(q_out^2 - q_in^2)` of
the radial family times `dth dz Q^2 / 2`. A ray needing more inline segments than KPR / KPA sets
`PI_OVERFLOW` (unchanged ray); the domain doubles the family (`grow_kp`) and the step is replayed.
Verified against the reference row by row over successive steps (external, internal, offset), volume
consistent to 1e-12, radius behind the tool within the scallop `f^2 / (8 r_nose)`, threads leave the helix
(test_08).

**Chip area per element**: `polar_radius(theta, z)` (nearest bin and row: error the radius change over half
a bin and half a row).

**Sync**: the tri-dexel of the same workpiece is rebuilt from the polar field for the dirty rows once per
`polar_sync_revolutions` (collisions, coolant, meshing) and exactly over the whole field at operation end
(`rebuild_tridexel`): z rays read the radial rays row by row with state changes placed at the axial rays'
exact endpoints; x / y rays cross the bins of their row (only those within the field's radius) with the
exact chord of every radial segment. Verified pointwise (> 10^4 samples, test_08) and by volume (1 %).
The polar field is authoritative for the removed volume while turning (`RemovalContact.polar = True`).

## 12. Integration with the hot loop

`kernel_inputs` returns `RemovalInputs(fields, targets, polars, cut, whole, inserts, helix, slabs, accum,
events, scratch, iparams, fparams)`: per workpiece the field, target, polar field (a placeholder when not
turning); per path the active tool's cutting profile, whole profile, insert region, helix sweep, tooth
slabs (placeholders of the same types); per contact (contact index = path index) an accumulator; one event
buffer and one scratch. `iparams`: `RI_NPATHS`, `RI_NWORK`, per path (`RI_PATH0 + RI_PSTRIDE p + RP_*`)
workpiece, slot, kind (`TK_ENVELOPE`, `TK_TURNING`, `TK_HELIX`), centre cutting, rotating; after the paths
the next surface event id of every workpiece (the loop increments it per subtraction and writes it into
`SI_EVENT_ID`). `fparams`: gouge / collision tolerances, gate sample spacing, tolerance and margin, sector
margin, sweep step, per path dead radius / height, ramp limit, slab height.

Per path, in the compiled loop:
* every L1 tick: `l1_contact` (broad phase, air ticks); `collide_envelope` (whole vs cutting profile);
* rotating tool (`TK_ENVELOPE`), at each tooth's generation instant: `tooth_pass(f, prof, slabs, tooth,
  slab_c, tip, x0, y0, w, st_z, st_r, active, eng, st_len, feed_x, feed_y, gate_params, spec, tgt, acc, ev,
  sc, stats)` = engaged intervals (warm start from `eng`, which cutting owns) + contact geometry + sector
  box + `subtract_envelope`; `slab_c` holds the per-slab axis points `X_e + u_rel` in S;
* taps (`TK_HELIX`): `helix_subtract` per L0 step with the poses at both ends and the spindle angles;
* turning (`TK_TURNING`): `polar_cut` per L0 step with `polar.polar_step` values;
* the loop may write `FA_M_FROM_S` (M <- S at the last removal, 12 values); left at identity, `absorb`
  uses motion's `pose_M_T` and `pose_S_T` of the path in `bus_prev` for the contact point and normal in M;
* a return status 1 (pool or polar overflow): finish the step, then `needs_replay` -> `prepare_replay` ->
  `kernel_inputs` -> replay (all writes are idempotent).

`absorb` turns accumulators and events into `bus_next.removal`; `macro_step` (L2) closes the volume
consistency, connectivity, polar sync and air cutting; `end_operation` rebuilds from the polar field,
releases loose bodies, checks the remaining material (last operation) or transfers the part.

## 13. Domain, bus, ledger, settings

**Bus read** (`bus_prev`): `motion.paths[*]` (`tool_slot`, `workpiece`, `spindle_omega_rad_s`, `ipo_ticks`,
`pose_M_T`, `pose_S_T`), `cutting.contacts[*]` (`contact`, `analytic_chip_volume_m3`). The hot loop itself
reads the rapid flag, spindle angle and poses from motion's kernel inputs. State read: `state.tooling` (`geometry(slot)`,
`edge_model(slot)`, `version`) for the tool shapes, `state.clock`. **Bus written**: `removal` only
(`RemovalFrame`: `contacts`, `events`, `loose_bodies`, `field_version`).

`RemovalContact` per contact: `removed_volume_m3` (exact field volume of the step: z family, or the polar
radial family), `displaced_volume_m3` (0: no forming / burr model, section 15), `geometry` (ToolContact:
in_cut, point and normal in M and S, ae, ap, engaged length and angle, in-cut fraction, air-cut time),
`volume_consistency`, and the additive fields `removed_volume_xyz_m3` (x, y, z family volumes),
`polar`, `surface_ids` (first and last surface event id of the step).

**Volume consistency** (architecture 5.4): field volume / analytic chip volume per contact. Cutting's
analytic volume of step n arrives in `prev` at step n + 1, so the ratio published in step n + 1 is that of
step n (lag one step). A pass whose ratio leaves `1 +- (volume_consistency_tol + family spread)` raises a
`mass_balance` warning.

**Ledger**: removal writes only its bus record; the core Ledger (cnctwin.core.ledger) books from it the
removed volume and mass (`removed_volume_m3`, rho at 20 degC x the field volume), the displaced volume (0,
section 15) and the freed bodies (`loose_bodies[*].mass_kg` -> `freed_body_mass_kg`). Chips scales its
emitted mass to rho x `removed_volume_m3` of each contact.

**Findings** (once per episode): rapid into material, collision, gouge, breakthrough, non-centre-cutting
plunge, fixture contact (errors); part_free, chip_hazard; remaining material after the last operation
(warning, from `excess_material`); air cutting (info); polar coarsening and scratch overflow (sanity).

**Channels** (macro): `removal.removed_volume_m3`, `removal_rate_m3_s`, `field_volume_m3`,
`freed_mass_kg`, `pool_used_blocks`, `events`, `field_version`, `polar_volume_m3`; per path
`volume_consistency`, `ae_m`, `ap_m`, `engaged_angle_rad`, `in_cut_fraction`, `air_cut_time_s`.

**Settings** `physics.removal` (`schema.py`, range-checked): `margin_nodes` 1, `gouge_tolerance_um` 5,
`collision_tolerance_um` 10, `gate_sample_cells` 1, `gate_tolerance_nm` 10, `gate_margin_um` 0,
`sector_margin_cells` 1, `pool_headroom` 2, `pool_min_free` 1024, `event_capacity` 1024,
`sweep_step_cells` 0.5, `connect_every_n_macro` 1, `polar_min_feed_mm_rev` 0.05, `polar_max_mb` 150,
`polar_kp` 2, `polar_axial_spacing_mm` 0, `polar_sync_revolutions` 1, `volume_consistency_tol` 0.02,
`air_cut_report_s` 2, `probe_rings` 8, `probe_around` 32, `mesh_capacity` 20000. Top level:
`physics.lattice_mm`, `physics.refine`, `physics.polar_theta_bins`.

**Checkpoint**: `RemovalState.to_arrays` / `from_arrays` (format 1) store every field, target, polar
field, pool, brick flags, accumulators, pass statistics, episodes and tool profiles; the run continues
bit for bit (test_10).

## 14. Transfers between operations (`transfer.py`, `k_transfer.py`)

`from_previous`: none (identity), `{flip: {axis, angle_deg}}` about the stock box centre, `{transfer:
sub_spindle | main_spindle | pallet | manual}` (sub-spindle: half turn about S y, theta -> pi - theta and
z -> L - z for the polar field). Rotations by multiples of 90 degrees are lattice symmetries: level 0 is
remapped exactly (signed axis permutation, mirrored segments on reversed axes; verified bit-exact
volumes, test_11). Other rotations resample through the mesh (error a^2 kappa / 8 on curved faces, up to
one cell at sharp edges). Refine levels are dropped when the material moves (note in the run).

## 15. Assumptions, validity and limitations

* Lattice: `twin check` requires `lattice_mm <= min(ae) / 4` on every pass, or a refine box over the
  finishing passes; features narrower than a cell in a family's transverse directions are seen by fewer
  families (the majority rule and the most normal family decide).
* Tool bodies are rigid per axial slab (deflection enters as the slab axis points); the profile is the
  maximum radius per height (undercut tools like T-slot cutters are filled to the axis as a revolved
  max-radius solid).
* Forming taps: the tooth profile is subtracted like a cutting tap's and booked as removed (chips scale
  to it, so the mass balance closes); the displaced material is not re-added (`displaced_volume_m3 = 0`,
  cutting's analytic `CuttingContact.displaced_volume_m3` is not used): the thread crest is the cut crest.
  A crest-forming model would re-add material, report `displaced_volume_m3` and needs chips to scale to the
  net volume. Burrs and side flow are not modelled.
* Removal below one quantum per endpoint (10 nm) is lost; the volume lost this way is below
  `5 nm x a^2` per ray per pass.
* Gating error bounds as section 5; polar chip area as section 11.
* Machine-part collisions need the link poses (meshes and poses come from the machine / kinematics).
* Inner (ID) turning: the block skip uses the largest radius only, so a bore costs more per step than OD
  turning (correct either way).

## 16. Cost (measured, one core of the i5-4690 on the shared box, `bin/tsafe`, process CPU time, best of repeats)

Targets of the task (architecture 20), test `tests/unit/removal/test_12_perf.py` (`pytest -s` prints; ranges
over the runs on this shared box, load average 4 to 8):

| what | measured | target |
|---|---|---|
| engaged intervals, 40 stations (20 peripheral over ap 10 mm, 20 end-edge), D10 side milling, 3629 samples | 184 - 197 us per call | 120 us |
| one tooth-pass subtraction, D10 end mill, ae 1 mm, ap 10 mm, f_z 50 um, 0.25 mm lattice (641 rays changed) | 100 - 110 us | 200 us |
| polar L0 step (one 0.5 deg bin), CNMG-like insert, ap 2 mm, f 0.2 mm/rev, dz 25 um (about 20 radial + 16 axial rays changed, 36 block bounds) | 3.1 - 3.9 us | 2 us |
| memory, 300 x 200 x 100 mm block at 0.25 mm (1 769 612 rays) | 129.7 MB (+ 28 MB for a one-segment target), built in 2.6 - 3.2 s | 300 MB |

Other kernels (compiled callers unless noted):

| kernel | case | cost |
|---|---|---|
| `broad_phase` | D10 envelope box | 0.66 us |
| `l1_contact` | with the broad phase | 1.1 us |
| `point_inside` | | 0.63 us |
| `ray_cast` | 40 mm through a block | 7.7 us |
| `surface_point` | | 9.0 us |
| `polar_radius` | | 0.01 us |
| `collide_envelope` | tool cutting, shank clear | 0.29 us |
| `collide_envelope` | shank 3 mm into material | 314 us |
| `sweep_contact` | 8 x 32 stylus samples (from Python) | 3.4 ms |
| `label_z` | 300 x 200 x 100 mm at 0.25 mm | 31 ms |
| `excess_material` | 300 x 200 x 100 mm at 0.25 mm | 9.2 ms |
| `mesh.mesh_bricks` | 18 surface bricks (about 0.3 ms per brick) | 5.7 ms |
| `rebuild_tridexel` | D40 x 60 mm bar, 720 bins x 2400 rows, full / 0.5 mm dirty rows | 3.2 s / 26 ms |
| `transfer.apply` | flip about x, 60 x 30 x 20 mm, exact remap | 31 ms |

How the hot kernels reach these costs (and why the gating and polar targets are not met):

* `RAW` kernels (`subtract_envelope`, `post_pieces`, `_engaged`, `_refine`, `contact_geometry`,
  `sector_box`, `tooth_pass`, `polar_cut`, `_polar_rows`, `_polar_axial`, `polar_radius(i)`,
  `polar_to_ray`, `collide_envelope`, and `k_sd.{level_at, dexel_sd, gradient}`, which the hot loop's refill
  planes call for every stale memory bin: `gradient` without its work arrays, the six D values from
  `k_core.sd_d`) are compiled with Numba's reference counting off (`_nrt=False`):
  they allocate nothing (work arrays come from `Scratch` or a thin wrapper), and every inlined helper would
  otherwise pay an atomic incref / decref per array argument and call (measured: envelope intervals of
  one ray 274 ns with, 30 ns without; subtraction 510 -> 108 us; gating 2957 -> 192 us). Numba refuses an
  allocation in such a function, so the property is checked at compile time.
* Gating: about 3600 samples per call; samples in empty bricks or beyond the lattice cost about 10 ns,
  classified samples 60 to 190 ns (four rays per family, all three families near boundaries). Reaching
  120 us needs fewer classifications per sample (a per-station cache of the axis family's rays) or a
  coarser default sample spacing (`gate_sample_cells`, with the undetected depth of section 5).
* Polar: about 26 000 instructions per step for the rays the insert actually changes (radial rows at the
  nose and leading edge, axial rays across ap), the block bounds behind the nose and the axial rays'
  chain evaluation; the per-step overhead is 0.06 us and air steps 0.15 us.

## 17. Kernels

Rules (architecture 5.5): `@njit(<explicit signature>, cache=True, nogil=True, error_model="numpy")`,
one specialisation each (`len(kernel.signatures) == 1` is tested for every kernel of every module), no
fastmath / parallel / jitclass / typed containers / closures; C-contiguous writable arrays and NamedTuples
only; each kernel has a plain-Python reference in `reference.py` or an analytic check, and a comparison
test. `inline` helpers are inlined at the Numba IR level (one call site each where compile time matters);
`call` kernels are separate functions; `raw` kernels run without reference counting (section 16).

Inlined helpers: `k_core.{rindex, bindex, nearest, inside, overlap, extent, blocks_for, load, _alloc, _free,
store, floordiv, mark, account, book_piece, level_at, sd_pre, sd, sd_d, sd_beats, classify_pre, classify}`,
`k_gate.{_empty_near_pre, _above_columns}`, `k_subtract._point`, `k_ray.{iv_subtract, iv_union, target_overlap}`,
`k_envelope.{quad_leq, _p4, _dp4, _torus_g, _push, _piece, merge_intervals, _env_ray, _event, _aabb,
node_range}`, `k_gate.{_box_material, _empty_near, _sin_range_max}`, `k_query.{_occ, _extreme}`,
`k_mesh._crossing`, `k_polar.{_chain, _chain_w, _x_range, _z_range, _z_range_w, _cut_inline, _xhi_max,
_xhi_max_w, _rescan, _q_of, _rad, _add_piece}`.

Compiled kernels (types: `nbt.py`; F1/F2 float64 C arrays, I1 int64, I4_* int32, B1 int8; inputs and
outputs in the docstrings):

| kernel | kind | signature | what |
|---|---|---|---|
| `k_ray.blocks_for` | call | `i8(i8)` | Pool blocks a ray with n segments uses. |
| `k_ray.pool_alloc` | call | `i8(FIELD_T)` | One pool block from the free list (FIELD_T wrapper of k_core). |
| `k_ray.pool_free` | call | `void(FIELD_T, i8)` | Return a pool block to the free list. |
| `k_ray.ray_load` | call | `i8(FIELD_T, i8, I4_1, I4_1, I4_1, I4_1)` | Copy the segments of ``ray`` into the scratch lists; |
| `k_ray.ray_store` | call | `i8(FIELD_T, i8, I4_1, I4_1, I4_1, I4_1, i8)` | Write n segments to ``ray`` (inline first, then its pool chain). |
| `k_ray.ray_index` | call | `i8(FIELD_T, i8, i8, i8, i8)` | Ray index of (level, family, iu, iv). |
| `k_ray.brick_index` | call | `i8(FIELD_T, i8, i8, i8, i8)` | Brick index of (level, bx, by, bz). |
| `k_ray.mark_range` | call | `void(FIELD_T, i8, i8, i8, i8, i8, i8, i8)` | Mark changed the bricks whose closed node ranges contain the ray part [q0, q1] (both ends inclusive, both bricks of a shared boundary node). |
| `k_ray.brick_account` | call | `void(FIELD_T, i8, i8, i8, i8, I1, I1, i8, i8)` | Book removed (sign -1) or added (+1) pieces of one ray into the owning bricks' material and mark the touched bricks changed. |
| `k_ray.ray_nearest` | call | `types.UniTuple(i8, 3)(FIELD_T, i8, f8)` | (inside, nearest endpoint in quanta, has_endpoint) of ``ray`` at the coordinate q (quanta). |
| `k_ray.ray_inside` | call | `b1(FIELD_T, i8, f8)` | Is q (quanta) inside material on the ray. |
| `k_ray.ray_overlap` | call | `types.UniTuple(i8, 2)(FIELD_T, i8, i8, i8)` | (material length inside [q0, q1), segments touching it) of ``ray``. |
| `k_ray.family_length` | call | `i8(FIELD_T, i8, i8)` | Total material length (quanta) of one family of one level: the exact integer the volume is. |
| `k_ray.target_overlap_t` | call | `types.UniTuple(i8, 2)(TARGET_T, i8, i8, i8)` | (target length inside [q0, q1), index of the first target segment touched or -1). |
| `k_ray.quantize` | call | `i8(f8)` | Fixed-point coordinate (quanta) of a float coordinate in quanta: round half up, saturated. |
| `k_stock.add_interval` | call | `i8(FIELD_T, i8, i8, i8, i8, i8, i8, i8, SCRATCH_T)` | Unite [q0, q1) (quanta, clipped to the ray) into ray (lvl, fam, iu, iv); |
| `k_stock.replace_ray` | call | `i8(FIELD_T, i8, i8, i8, i8, I1, I1, i8, i8, SCRATCH_T)` | Replace the segments of one ray by the sorted, disjoint intervals [qlo, qhi) (quanta, clipped to the ray); |
| `k_stock.fill_box` | call | `i8(FIELD_T, i8, F1, F1, i8, SCRATCH_T)` | Unite the axis-aligned box [lo, hi) (S) into every ray of a level. |
| `k_stock.fill_tube` | call | `i8(FIELD_T, i8, f8, f8, f8, f8, f8, f8, i8, SCRATCH_T)` | Unite the tube r_in <= r < r_out, z0 <= z < z1 about the axis through (cx, cy) along S z (r_in = 0: a cylinder) into every ray of a level. |
| `k_stock._canon_edge` | call | `f8(f8, f8, f8, f8, f8, f8)` | Edge function of the directed edge p -> q at x, evaluated in canonical vertex order. |
| `k_stock._covers` | call | `types.boolean(f8, f8, f8)` | Top-left rule of the watertight rasteriser (edge owns the sample). |
| `k_stock.rasterize` | call | `i8(FIELD_T, i8, i8, F2, I4_2, i8, i8, i8, i8, I1, I1, F1, B1, i8)` | Hits of the triangles (V, Fc) on the rays (iu0..iu1, iv0..iv1) of family ``fam``. |
| `k_stock._ray_intervals` | call | `i8(F1, B1, i8, i8, I1, I1)` | Sort hits a..b-1 (entering before leaving at equal t) and sweep the winding number; |
| `k_stock.resolve_hits` | call | `i8(FIELD_T, i8, i8, i8, i8, i8, i8, I1, F1, B1, i8, i8, I4_2, I1, I1, F1, SCRATCH_T)` | Turn the hits of a window into material intervals per ray. |
| `k_envelope._monotone_root` | call | `f8(f8, f8, f8, f8, f8, f8, f8, f8, i8)` | Root of P (deriv 0) or P' (deriv 1) on [lo, hi] where it is monotone and changes sign. |
| `k_envelope.poly4_roots` | call | `i8(f8, f8, f8, f8, f8, f8, f8, F1)` | Real roots of p0 + p1 t + ... |
| `k_envelope.torus_leq` | call | `i8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, F2, I4_1, i8, i8, F1)` | Append the parts of [ta, tb] inside r <= rc' + sigma sqrt(rho^2 - zeta^2) (/zeta/ <= rho there). |
| `k_envelope.push_event` | call | `void(EVENTS_T, i8, i8, i8, f8, f8, f8, f8, f8, i8, i8, i8, i8)` | Append one event row (EVENT_WIDTH columns) to an EventBuffer. |
| `k_envelope.envelope_aabb` | call | `f8(PROFILE_T, SPEC_T, F1, F1)` | Box of the slabbed envelope in S (written to lo, hi); |
| `k_envelope._dead_cyl` | call | `i8(f8, f8, f8, i8, F1, f8, f8, f8, f8, f8, f8, f8, f8, F2, I4_1)` | Intervals (t, metres from the ray origin) of the ray inside the cylinder r <= r_dead, z in [z0, z1]. |
| `k_subtract.post_pieces` | raw | `void(FIELD_T, TARGET_T, SPEC_T, F1, ACCUM_T, I2, i8, F1, SCRATCH_T)` | Second pass of a subtraction over the recorded pieces (ray, level, family, iu, iv, q0, q1): brick material and change flags, target checks (gouge: ... |
| `k_subtract.subtract_envelope` | raw | `i8(FIELD_T, PROFILE_T, SPEC_T, TARGET_T, ACCUM_T, EVENTS_T, SCRATCH_T)` | Subtract the slabbed revolved envelope of a SubtractSpec from every level of the field (bit l of the level mask), inside the spec's box. |
| `k_helix.helix_inside` | call | `f8(HELIX_T, f8, f8, f8, f8, f8, f8, F1, F1, F1, F1, F1, f8, f8, f8)` | Largest of the per-land inside values at a point (<= 0: inside the swept region of some land). |
| `k_helix.helix_subtract` | call | `i8(FIELD_T, HELIX_T, F1, F1, F1, F1, F1, f8, f8, F1, F1, i8, i8, i8, TARGET_T, ACCUM_T, EVENTS_T, SCRATCH_T)` | Subtract the regions the tap's lands sweep while the spindle turns from th_a to th_b and the tip moves from pa to pb (S). |
| `k_collide.env_ray` | call | `i8(F1, F1, types.int32[::1], F1, F1, F1, F1, types.int32[::1], f8, f8, f8, i8, F1, F2, F1, F1, f8, f8, F2, I4_1, i8, F1)` | Compiled (not inlined) _env_ray, for the callers off the subtraction's hot path. |
| `k_collide.envelope_ray` | call | `i8(PROFILE_T, f8, f8, f8, i8, F1, F2, F1, F1, f8, f8, F2, I4_1, i8)` | Intervals of the ray o + t e_fam (t in [t_lo, t_hi], metres) inside the slabbed envelope, sorted and merged when ``merge``; |
| `k_collide.collide_envelope` | raw | `i8(FIELD_T, PROFILE_T, PROFILE_T, F1, F2, F1, i8, i8, f8, ACCUM_T, EVENTS_T, SCRATCH_T)` | Material inside the whole envelope (holder, shank, neck, spindle nose, insert body) but outside the cutting envelope, at the pose (tip_c [1, 3], ... |
| `k_collide.mesh_field_overlap` | call | `i8(FIELD_T, F2, types.int32[:, ::1], F1, F1, f8, F1, SCRATCH_T)` | A closed triangle mesh (vertices in S) against the material of level 0, over the rays inside the box (the mesh box clipped to the field). |
| `k_collide.pose_lerp` | call | `void(F2, F1, F1, F2, F1, f8, F2)` | out (3 x 4) = the rigid pose a fraction lam from (R0, t0) to (R1, t1): translation linear, rotation along the geodesic R0 exp(lam log(R0^T R1)) ... |
| `k_collide.swept_mesh_field` | call | `f8(FIELD_T, F2, types.int32[:, ::1], F2, F1, F2, F1, f8, f8, F1, SCRATCH_T)` | A mesh (vertices in its own frame) moving from pose (R0, t0) to (R1, t1) in S: sub-sampled so no vertex moves more than ``step`` between samples ... |
| `k_sd.level_at` | raw | `i8(FIELD_T, f8, f8, f8)` | Finest level whose authoritative box contains the point (0 otherwise). |
| `k_sd.dexel_sd` | raw | `f8(FIELD_T, i8, i8, f8, f8, f8, F1)` | D_f at (x, y, z) in metres (positive inside). |
| `k_sd.classify` | call | `types.Tuple((b1, f8, i8))(FIELD_T, i8, f8, f8, f8, i8, F1)` | (inside, D of the chosen family (m), chosen family) at a point (k_core.classify). |
| `k_sd.point_inside` | call | `b1(FIELD_T, f8, f8, f8)` | Is the point inside material (finest level containing it; |
| `k_sd.gradient` | raw | `types.void(FIELD_T, f8, f8, f8, i8, f8, F1)` | Central-difference gradient of D_fam (unit vector, pointing into the material). |
| `k_gate.box_has_material` | call | `b1(FIELD_T, i8, f8, f8, f8, f8, f8, f8)` | Any brick of the level meeting the box (grown by one cell) holds material. |
| `k_gate._refine` | raw | `f8(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, F1, F1, F1, f8, f8, f8, b1, f8, f8, F1, F1, F1)` | Angle in [pa, pb] where the material state changes (``a_inside``: the state at pa). |
| `k_gate._engaged` | raw | `void(FIELD_T, F1, F1, F1, F1, F1, F1, B1, ENGAGED_T, i8, F1, F1, F1, F1, F1, B1, F1, F1, F1, B1)` | engaged_intervals with its work arrays (states, bd_phi, bd_in: MAX_SAMPLES + 1; |
| `k_gate.engaged_intervals` | call | `void(FIELD_T, F1, F1, F1, F1, F1, F1, B1, ENGAGED_T, i8, F1, F1)` | Engaged intervals of the active stations (``out`` updated in place; |
| `k_gate.contact_geometry` | raw | `void(ENGAGED_T, F1, F1, F1, f8, f8, ACCUM_T)` | Fold one gating result into the contact accumulator: largest engaged angle and arc, radial engagement ae across the feed direction (feed_x, ... |
| `k_gate.sector_box` | raw | `i8(ENGAGED_T, F1, F1, F1, F1, F1, f8, f8, f8, f8, F1, F1)` | Axis-aligned box (S) of the engaged sector of the whole envelope: the union of the engaged angle intervals of all stations, as a wedge of radius ... |
| `k_query._brick_occupied` | call | `b1(FIELD_T, i8, i8, i8, i8)` | Does brick (level, bx, by, bz) hold material in any family. |
| `k_query._refine_t` | call | `f8(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, F2, F2, i8, f8, f8, f8, f8, f8, f8, f8, f8, b1, f8, F1)` | Crossing on [ta, tb] of a ray whose material state is ``a_inside`` at ta and the opposite at tb. |
| `k_query._state` | call | `b1(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, F2, F2, i8, f8, f8, f8, i8, F1)` | Material state at a point (finest level); |
| `k_query.ray_cast` | call | `i8(FIELD_T, f8, f8, f8, f8, f8, f8, f8, i8, f8, F1)` | First crossing along o + t d (/d/ = 1), 0 <= t <= t_max: mode 0 entry into material, mode 1 exit. |
| `k_query.surface_point` | call | `i8(FIELD_T, f8, f8, f8, f8, f8, f8, f8, F1)` | Surface point along n from p (n into the material): ahead (+n) when p is in air, behind (-n) when p is in material. |
| `k_query.refill_points` | call | `void(FIELD_T, F2, F2, f8, F2, B1)` | surface_point for many points (out [n, 4] = x, y, z, h; |
| `k_query.points_inside` | call | `void(FIELD_T, F2, B1)` | point_inside for many points (out [n] 1/0). |
| `k_query._extreme_endpoint` | call | `types.UniTuple(i8, 2)(FIELD_T, i8, i8, i8, i8)` | (top, bottom) endpoints (quanta) of a ray; |
| `k_query.surface_height` | call | `void(FIELD_T, i8, i8, F1, F1, F1, I1)` | Highest (sign +1) / lowest (sign -1) material surface along S axis ``axis`` above the points whose other two coordinates (increasing axis order) ... |
| `k_query.fill_fraction` | call | `void(FIELD_T, F1, f8, I1, F3)` | Material fraction of the cells of a grid (origin, cubic cell, shape [nx, ny, nz]) from the level-0 z family: each z ray is the column [x +- a/2] x ... |
| `k_query.wall_thickness` | call | `f8(FIELD_T, f8, f8, f8, f8, f8, f8, f8)` | Material thickness behind the surface point p along -n (n the outward normal): distance from the entry into material to the next exit; |
| `k_query.broad_phase` | call | `i8(FIELD_T, PROFILE_T, F1, F1, f8)` | Occupied level-0 bricks inside the box of the profile at the pose (tip, unit axis w), grown by ``margin``: 0 means the envelope cannot touch material. |
| `k_query.sweep_contact` | call | `f8(FIELD_T, PROFILE_T, F1, F1, F1, F1, i8, i8, f8, F1)` | First contact of a stylus (solid of revolution ``prof`` about w, tip at c0 -> c1, translation only) with material: the fraction lambda in [0, 1] ... |
| `k_connect._find` | call | `i8(I4_1, i8)` | Union-find root with path halving. |
| `k_connect._segs` | call | `void(I4_1, types.int32[:, :, ::1], I4_1, types.int32[:, :, ::1], I4_1, i8, I1, I1)` | Segments of one ray (inline and pool) into work lists. |
| `k_connect.label_z` | call | `i8(FIELD_T, I1, I4_1)` | Union-find over the level-0 z segments. |
| `k_connect.component_stats` | call | `i8(FIELD_T, I1, I4_1, i8, F2)` | Per root segment id r: out[r] = [volume (m^3), sum x V, sum y V, sum z V, x_lo, y_lo, z_lo, x_hi, y_hi, z_hi] (columns of a^2 x length, level 0). |
| `k_connect.components_in_box` | call | `void(FIELD_T, I1, I4_1, i8, F1, F1, types.int8[::1])` | flags[root] = 1 for every component with a level-0 z column (node inside the box's x-y range) whose segment overlaps the box's z range (the held ... |
| `k_connect.local_connected` | call | `i8(FIELD_T, I1)` | Incremental connectivity: 1 when the labelling of the changed bricks' column rectangle and a ring of one column is one component reaching the ring (the material is still one body), 0 not proven (label the whole field), -1 no change marked; box = [i0, i1, j0, j1, columns, segments]. |
| `k_connect.remove_loose` | call | `i8(FIELD_T, I1, I4_1, i8, types.int8[::1], f8, f8, f8, i8, SCRATCH_T)` | Remove the segments of loose components (``loose[root] = 1``) from every ray of every level: z rays of level 0 by their labels; |
| `k_mesh.ray_crossing` | call | `types.UniTuple(i8, 3)(FIELD_T, i8, i8, i8, i8)` | (endpoint in [q0, q1], its surface event id, found) of the kind an edge needs (see _crossing). |
| `k_mesh.displace` | call | `void(F1, f8, F4, f8, f8, f8, F1)` | out = (x, y, z) + d(x, y, z), trilinear in the grid (clamped at its border); |
| `k_mesh.mesh_brick` | call | `i8(FIELD_T, i8, i8, i8, i8, I1, I3_64, F2, I4_2, I4_1, I1, I1, F1, f8, F4, B3, I4_4)` | Mesh one brick of one level into out_* (appending at counts[0] vertices, counts[1] faces). |
| `k_polar.insert_x_range` | call | `types.UniTuple(f8, 2)(INSERT_T, f8)` | (x_lo, x_hi) of the insert at z_T (x_lo > x_hi: no insert there); |
| `k_polar.insert_z_range` | call | `types.UniTuple(f8, 2)(INSERT_T, f8)` | (z_left, z_right) of the insert at x_T (left > right: none). |
| `k_polar.insert_xhi_max` | call | `f8(INSERT_T, f8, f8)` | Largest x_T of the insert over z_T in [za, zb] (-inf when it has none there). |
| `k_polar._rescan_block` | call | `void(POLAR_T, i8, i8)` | Recompute zmax_q of one (bin, row block). |
| `k_polar._polar_rows` | raw | `types.UniTuple(i8, 4)(types.int32[:, :, ::1], types.int32[:, :, ::1], types.int8[::1], I1, F1, F1, F1, F1, F1, i8, i8, i8, i8, i8, f8, f8, f8, f8, f8, f8, f8, i8, i8, i8)` | The radial rays of rows ra0..ra1 of bin b under the insert (one block of polar_cut; |
| `k_polar._polar_axial` | raw | `types.UniTuple(i8, 3)(types.int32[:, :, ::1], types.int32[:, :, ::1], types.int8[::1], F1, F1, F1, F1, F1, i8, i8, i8, i8, f8, f8, f8, f8, f8, f8, f8, i8, i8, i8, i8)` | The axial rays j0..j1 of bin b under the insert (polar_cut). |
| `k_polar.polar_cut` | raw | `i8(POLAR_T, INSERT_T, F1, i8, ACCUM_T)` | Cut the bins the tool's reference plane passes in one L0 step (PolarStep values ``ps``). |
| `k_polar.polar_radius` | raw | `f8(POLAR_T, f8, f8)` | Outer material radius at (theta, z) in the rotating frame (0: no material). |
| `k_polar.polar_radii` | raw | `void(POLAR_T, F1, F1, F1)` | polar_radius for many (theta, z). |
| `k_polar.polar_q2_sum` | call | `i8(POLAR_T, i8)` | Sum over radial rays of q_hi^2 - q_lo^2 (the exact integer volume / (dtheta / 2 dz quantum^2)); |
| `k_polar.polar_to_ray` | raw | `i8(POLAR_T, f8, f8, f8, i8, f8, f8, I1, I1, I1, I1, i8, F1, F1, I1, I1, F2)` | Material of the polar field along the ray o + t e_fam, t in [t_lo, t_hi] (metres from o): sorted, merged intervals written as quanta of t; |
| `k_polar.bin_trig` | call | `F2(i8, f8)` | [nth + 1, 4]: sin, cos of the bin edges b dth (rows 0..nth) and of the bin middles 0.5 (b dth + (b + 1) dth) (columns 2, 3 of rows 0..nth-1), ... |
| `k_polar.rebuild_tridexel` | call | `i8(FIELD_T, POLAR_T, i8, f8, f8, TARGET_T, f8, EVENTS_T, i8, i8, SCRATCH_T)` | Replace the material of every ray of level ``lvl`` inside the polar region (r <= the polar field's largest radius, z in [zlo, zhi]) by the polar ... |
| `k_polar.refresh_zmax` | call | `void(POLAR_T, i8)` | zmax_q[b * nblk + blk] = largest outer radius (quanta) of the rows of each block in each bin. |
| `k_transfer.remap_level0` | call | `i8(FIELD_T, FIELD_T, I1, I1, SCRATCH_T)` | Fill level 0 of ``new`` (empty, dims[i] = old dims[perm[i]]) from level 0 of ``old``. |
| `k_drive.tooth_pass` | raw | `i8(FIELD_T, PROFILE_T, SLABS_T, i8, F2, F1, F1, F1, F1, F1, F1, B1, ENGAGED_T, F1, f8, f8, F1, SPEC_T, TARGET_T, ACCUM_T, EVENTS_T, SCRATCH_T, F1)` | Gate, bound and subtract one tooth pass. |
| `k_drive.l1_contact` | call | `b1(FIELD_T, PROFILE_T, F1, F1, f8, b1, i8, f8, ACCUM_T)` | Broad phase of one path at one L1 tick; |
| `k_drive.excess_material` | call | `i8(FIELD_T, TARGET_T, f8, F1)` | Material of level 0 outside the target, from the z family: out = [volume (m^3), largest excess along a z ray (m), x, y, z of its middle, rays with ... |

## 18. Additive changes to shared records and layouts

* `core/bus.py` `RemovalContact` (written by removal only): `removed_volume_xyz_m3: np.ndarray` (x, y, z
  family volumes of the step), `polar: bool` (the volume is the polar radial family's), `surface_ids:
  tuple` (first and last surface event id of the step); commented in place. No existing field changed.
* `Scratch.fw` sub-ranges: `FW_ENV` (`N_ENV` = 11: the slab bounds and the ray's material extent that
  `subtract_envelope` leaves for `_env_ray`, right after the root-isolation workspace `FW_WORK`); `FW_AGG` moved
  from 34 to 45 (`N_FW` unchanged). `Workpiece.one_body` (the incremental connectivity check; checkpoint meta
  `one_body`, default false for older checkpoints).
* Removal's own layouts (`removal/abi.py`, not shared before this domain): `Scratch` gained `fw`, `iv2`,
  `tg`, `iw`, `rows` (fixed work of the RAW kernels; `FW_*` sub-ranges), `InsertRegion.box` has 6 entries
  (upper-chain peak and unimodal flag), `POLAR_BLOCK = 16` sets the `zmax_q` blocks. cnctwin.gpu's dexel
  twin recompiles `subtract_envelope` from its source with `_env_ray` rebound: the function still calls
  `_env_ray` by name and takes its work arrays from the scratch, so that path is unchanged (its parity
  tests pass).

## 19. References

* Amanatides, J., Woo, A. (1987). A fast voxel traversal algorithm for ray tracing. Eurographics 87, 3-10.
* Altintas, Y. (2012). Manufacturing Automation, 2nd ed. Cambridge University Press (chip geometry, engagement).
* Benouamer, M. O., Michelucci, D. (1997). Bridging the gap between CSG and Brep via a triple ray
  representation. ACM Symposium on Solid Modeling and Applications, 68-79.
* de Boor, C. (1978). A Practical Guide to Splines. Springer, ch. 2 (interpolation error).
* Dowell, M., Jarratt, P. (1971). A modified regula falsi method for computing the root of an equation.
  BIT 11, 168-174.
* Lorensen, W. E., Cline, H. E. (1987). Marching cubes. SIGGRAPH 87, 163-169.
* Montani, C., Scateni, R., Scopigno, R. (1994). A modified look-up table for implicit disambiguation of
  marching cubes. The Visual Computer 10, 353-355.
* Nielson, G. M., Hamann, B. (1991). The asymptotic decider. IEEE Visualization 91, 83-91.
* Pineda, J. (1988). A parallel algorithm for polygon rasterization. SIGGRAPH 88, 17-20.
* Press, W. H., Teukolsky, S. A., Vetterling, W. T., Flannery, B. P. (2007). Numerical Recipes, 3rd ed.,
  5.6 (quadratic and cubic equations), 9.4 (safeguarded Newton).
* Van Hook, T. (1986). Real-time shaded NC milling display. SIGGRAPH 86, 15-20.
* Woop, S., Benthin, C., Wald, I. (2013). Watertight ray/triangle intersection. JCGT 2(1), 65-82.
* Zhang, W., Leu, M. C. (2009). Surface reconstruction using dexel data from three sweep directions.
  Computer-Aided Design 41, 413-423.

Keys in `db/bibliography/models-removal.yaml`.
