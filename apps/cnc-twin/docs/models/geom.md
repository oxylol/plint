# Geometry library (`cnctwin.geom`)

The geometry every domain builds on: poses and the tilted-plane conventions of the controls, triangle
meshes, primitive solids and tool envelopes, mesh and CAD files (STL, OBJ, 3MF, PLY, GLB, STEP with PMI),
bounding-volume trees with ray, distance and inside/outside queries, interference tests, and the 3D-DDA
walk of uniform grids. It is a library (architecture 2): it owns no SimState field, reads and writes no
bus record and draws no random numbers. Everything is SI (metres, radians) inside; file units are
converted at the file boundary.

| module | contents |
|---|---|
| `transform` | 4x4 poses, rotations, Euler sequences, project poses, control frames (Fanuc G68.2, Siemens CYCLE800, Heidenhain PLANE), quaternions, slerp; Numba kernels |
| `mesh` | `Mesh` (vertices, faces, face groups, attributes), measures, topology, weld, merge |
| `primitives` | box, cylinder, cone, sphere, torus, revolved profiles (polylines with arcs) as meshes; analytic ray kernels; polynomial roots |
| `bvh` | bounding-box tree (`MeshTree`), rays, segments, closest point, signed distance, point in mesh, winding number |
| `overlap` | triangle-triangle, triangle-box, box-box, mesh-mesh, mesh-box, mesh-vs-revolved-envelope |
| `dda` | 3D-DDA traversal of a uniform grid (`Grid`) |
| `io` | STL, OBJ, PLY, 3MF, GLB; `io.step` STEP import (OpenCASCADE) with PMI and an AP242 PMI writer |
| `cad`, `selectors` | `CadModel` / `FaceInfo` (an imported part), face selectors |
| `abi` | kernel layouts `MeshTree`, `Grid`; feature and classification codes |
| `reference` | plain-Python references of the kernels (tests only) |
| `settings` | defaults from `db/models/geom.yaml` |

`from cnctwin.geom import X` resolves lazily (each submodule compiles or loads its kernels on first use).

## 1. Conventions

* A pose is a 4x4 matrix on column vectors, `pose_A_B` maps coordinates in frame B to frame A:
  `p_A = R p_B + t` (the columns of R are B's axes in A). `pose_A_C = pose_A_B @ pose_B_C`.
* Rotations are right-handed and active: `R_z(a) = [[c, -s, 0], [s, c, 0], [0, 0, 1]]` etc.
* Intrinsic sequence `axes = ijk`, angles `(a1, a2, a3)` in sequence order: `R = R_i(a1) R_j(a2) R_k(a3)`;
  extrinsic (fixed axes): `R = R_k(a3) R_j(a2) R_i(a1)`.
* Project and database poses `{translate_mm, rotate_deg: [a, b, c], order: xyz}` index angles by axis and
  apply them intrinsically in `order` (`R = R(order[0]) R(order[1]) R(order[2])`), identical to
  `cnctwin.db.simap.pose_matrix` (tested for all six orders) and to three.js `Euler(a, b, c, 'XYZ')`.
* Meshes: faces counter-clockwise seen from outside (normals out of the material), `face_group` = B-rep
  face id / object / primitive patch / -1.

## 2. Transforms (`transform.py`)

* Rodrigues: `R = I cos a + sin a [k]x + (1 - cos a) k k^T`.
* `euler_angles` inverts every one of the 12 Euler and Tait-Bryan sequences, intrinsic or extrinsic
  (Shoemake 1994 reduction written out per family). At a gimbal lock (middle angle at a pole) the third
  angle is 0 and the first carries the rotation. Round trip error below 4e-16 (tested, gimbal cases included).
* Quaternions `(w, x, y, z)` by Shepperd (1978): the largest of `{tr, R00, R11, R22}` selects the formula,
  so the division is never by a small number; sign normalised to `w >= 0`.
* Slerp (Shoemake 1985) on the shorter arc, `q(s) = sin((1 - s) th)/sin th q0 + sin(s th)/sin th q1`, with the
  angle by Kahan's `th = 2 atan2(|q1 - q0|, |q1 + q0|)` (accurate at 0 and pi). Below `th* = eps^(1/4)`
  (1.2e-4 rad) a normalised lerp is used: its angular error grows as `th^3/12`, the sine formula's rounding
  as `eps/th`; they balance at `th*` (error <= 1.5e-13 rad).
* `vec_slerp(a, b, s)`: the great-circle interpolation of the tool axis that 5-axis controls use between two
  orientations (Fanuc tool-axis interpolation, Siemens ORIVECT, Heidenhain TCPM). Opposite vectors: the
  circle passes through a given `pole`, else through the deterministic `perpendicular(a)` (a crossed with
  the coordinate axis it is least aligned with).
* `pose_interp(T0, T1, s)`: quaternion slerp + linear translation (architecture 5.4: never a matrix lerp).
* Kernels: `pose_apply_points_nb`, `pose_apply_dirs_nb`, `pose_mul_nb`, `pose_inv_nb`, `axis_angle_nb`,
  `quat_from_matrix_nb`, `matrix_from_quat_nb`, `quat_slerp_nb`, `vec_slerp_nb`, `pose_interp_nb`, each
  checked against the NumPy function (agreement <= 1e-14).

### 2.1 Tilted working planes of the controls

All return the pose of the tilted (feature) frame in the active workpiece coordinate system (SI).

| control | function | definition |
|---|---|---|
| Fanuc `G68.2 X Y Z I J K` (P0) | `fanuc_g68_2` | origin (X, Y, Z); Euler Z-X'-Z'': `R = R_z(I) R_x(J) R_z(K)` (Fanuc 30i-B manual) |
| Fanuc `G68.2 P1 Q.. I J K` | `fanuc_g68_2_rpy` | roll-pitch-yaw: I about X, J about Y, K about Z of the workpiece system, in the order Q (123 = X, Y, Z), fixed axes: `R = R_q3 R_q2 R_q1` |
| Fanuc `G68.2 P2` | `fanuc_g68_2_points` | origin point 1, +X toward point 2, point 3 on the +Y side |
| Fanuc `G68.2 P3` | `fanuc_g68_2_vectors` | Z along the second vector, X along the first made perpendicular |
| Siemens `CYCLE800` | `siemens_cycle800(mode, A, B, C, X0.., X1..)` | `TRANS(X0) R TRANS(X1)`; `_MODE` bits 7-6: 00 axis by axis `R = R_ax1(A) R_ax2(B) R_ax3(C)` (each about the already rotated axes; bits 1-0, 3-2, 5-4 name the axes, 01 X 10 Y 11 Z; 57 = X, Y, Z), 01 solid angle `R_z(A) R_y(B) R_z(C)`, 10 projection angle (below), 11 rotary axes direct: refused (machine kinematics, not a frame) |
| Heidenhain `PLANE SPATIAL SPA SPB SPC` | `heidenhain_plane_spatial` | about the non-tilted X, Y, Z in the order A, B, C (fixed axes): `R = R_z(SPC) R_y(SPB) R_x(SPA)`; the datum does not move |
| Heidenhain `PLANE EULER` | `heidenhain_plane_euler` | precession about Z, nutation about X', rotation about Z'' |
| Heidenhain `PLANE VECTOR` | `heidenhain_plane_vector` | Z along the normal vector, X along the base vector corrected to be perpendicular |
| Heidenhain `PLANE POINTS` | `heidenhain_plane_points` | X along P1 -> P2, P3 on the +Y side |
| Heidenhain `PLANE RELATIVE` | `heidenhain_plane_relative` | an incremental turn about an axis of the active tilted frame |

**Projection angles** (CYCLE800 mode 10, `projection_angles`): angle 1 is the angle, about axis 1, of the
line where the tilted plane cuts the coordinate plane normal to axis 1; angle 2 likewise about axis 2; the
tool axis is the remaining axis z. With `R = R_1(a) R_2(b)` the rotated axis 2 stays in the plane normal to
axis 1, so `a = angle1` exactly; the plane normal is `n = R_1(a) R_2(b) e_z` and its cut with the plane
normal to axis 2 has `tan(angle2) = tan(b) / cos(a)`, so `b = atan(tan(angle2) cos(a))` (derived for XY and
YX, checked numerically by recomputing both projections). The third angle turns about the new Z.

Choices made (and their consequence): Fanuc P1 is taken about fixed axes (the robotics meaning of
roll-pitch-yaw); if a control applied P1 about moving axes, frames with two or more non-zero angles would
differ. CYCLE800 projection mode is defined for tool axis Z (orders XY, YX; others are refused with an
error naming the supported orders). Not provided: Fanuc P4, Heidenhain PLANE PROJECTED and PLANE AXIAL
(PLANE AXIAL is machine-axis positioning, i.e. kinematics); the controller reports such blocks.

## 3. Meshes (`mesh.py`)

* Volume by the divergence theorem about the bounding-box centre c: `V = 1/6 sum (v0 - c) . ((v1 - c) x
  (v2 - c))` (exact for a closed mesh, positive when outward); volume centroid `c + sum vol_i (v0+v1+v2-3c)/4 / V`.
* Vertex normals angle-weighted (Thuermer & Wuethrich 1998), corner angles by `atan2(|a x b|, a . b)`.
* Watertight: every undirected edge in exactly two faces traversing it in opposite directions, no
  degenerate face (`edge_report` counts boundary, non-manifold, inconsistent edges and degenerate faces).
* `weld(tol)`: tol 0 merges bit-identical coordinates (first occurrence represents); tol > 0 is a greedy
  clustering in vertex order (a vertex joins the lowest-index representative within `tol`, searched in the
  27 neighbouring cells of size `tol`, kernel `weld_representatives_nb`, reference
  `weld_representatives_ref`). Deterministic; collapsing faces dropped.
* `transformed(T)` maps a `normal` attribute by the inverse transpose and flips the winding of a mirroring T.
* `merge` fills attributes missing in some parts with NaN (float) / -1 (int); `offset_groups` renumbers groups.

## 4. Primitives and revolved profiles (`primitives.py`)

Tessellation: a circle of radius r gets `n = max(n_min, ceil(pi / acos(1 - tol / r)))` segments so the
sagitta `r (1 - cos(pi/n)) <= tol` (`geom.tessellation_chordal_tolerance_m` = 10 um, `n_min = 12`). Vertices
lie on the true surface, so the meshes are inscribed: `0 < V_true - V_mesh <= A_true * 2 tol` (two
tessellation directions), checked for cylinder (equal to the exact prism `n/2 r^2 sin(2 pi/n) h` to 1e-13),
cone, sphere, torus and a ball-end envelope.

Revolved profiles are polylines in the meridian plane `(z, r)` with a bulge per segment `b = tan(theta/4)`
(theta the signed included angle, positive counter-clockwise in (z, r)); arc centre `M + n_left (L/2)(1 -
b^2)/(2b)`, radius `L / (2 |sin(theta/2)|)`. The orientation sign of a profile is the exact Green integral
`1/2 oint (z dr - r dz)` (arcs: `rho^2 theta + rho [c_z (sin phi1 - sin phi0) - c_r (cos phi1 - cos phi0)]`),
which sets the triangle winding (outward normals) and the ray normals. Face group k = segment k (caps are
segments to the axis). `profile_polyline(conservative=True)` replaces arcs that bulge away from the axis by
their circumscribed polygons (tangent intersections at radius `rho / cos(dphi/2)`), so a collision
envelope never lies inside the true profile; arcs bulging toward the axis are enclosed by their chords.

Machine geometry primitives (`db/_templates/machine.yaml` `geometry[].primitives`: box, cylinder, cone,
sphere, mesh file with scale and pose; plus torus) become meshes through `mesh_from_primitive` (raw mm or SI views).

**Analytic rays** (first hit in `[t0, t1]`, outward normal): box by slabs (Kay & Kajiya 1986; a zero
direction component is tested explicitly, so no `0 * inf`); sphere by the stable quadratic
(`q = -(b + sign(b) sqrt(disc))/2`, Press et al. 2007 5.6); cone bands `x^2 + y^2 = (m z + c)^2` and annuli;
arcs as torus patches `(|p|^2 + rc^2 - a^2)^2 = 4 rc^2 (px^2 + py^2)` (spheres when the centre is on the
axis), solved after moving the ray origin to its closest approach to the arc centre (conditioning), the
mirrored branch `(rho + rc)^2 + z^2 = a^2` and points outside the arc's span rejected.
Real roots of degree <= 4 (`poly_roots_nb`): recursive bracketing between the roots of the derivative
(each piece is monotone), the interval clipped to Cauchy's bound `1 + max |c_i/c_0|`, each bracketed root
by Newton safeguarded by bisection (rtsafe, Press et al. 9.4) to 8 ulps; a critical point within
`64 eps sum |c_i||t|^i` of zero is a double root (tangency). Checked against companion-matrix roots
(`numpy.roots`) on random quartics (<= 1e-9) and on the torus quartic for 200 random rays.

## 5. Bounding-volume trees (`bvh.py`)

**Build**: binned surface-area heuristic (MacDonald & Booth 1990; Wald 2007): split cost
`C = C_trav + (A_L N_L + A_R N_R) / A_P` over 12 bins of the centroid extent along the widest axis,
`C_trav = 1/8`, nodes with more than 4 triangles always split (pbrt, Pharr et al. 2016 4.3). Deterministic:
sequential, two-pointer partitions in index order, median of the index order when all centroids coincide,
a leaf forced at depth `MAX_DEPTH = 60` (so traversal stacks of 2 (60 + 2) entries always suffice). The
Numba build equals the plain-Python build (`reference.build_tree_ref`) bit for bit. Layout `abi.MeshTree`:
depth-first, first child at `i + 1`, second at `node_right[i]`; triangle corners copied in leaf order
(`tri_v`, 72 bytes/triangle) plus face normals, edge pseudo-normals and vertex pseudo-normals by face id
(about 180 bytes per triangle in total). `static_tree(mesh)` memoises by content hash (machine, fixture);
`save_tree`/`load_tree` persist.

**Rays**: watertight ray/triangle test (Woop, Benthin & Wald 2013): shear into ray space
(`kz = argmax |d|`, `S = (d_kx/d_kz, d_ky/d_kz, 1/d_kz)`), edge functions `U = Cx By - Cy Bx`,
`V = Ax Cy - Ay Cx`, `W = Bx Ay - By Ax`; a hit when all have one sign, `t = (U Az + V Bz + W Cz) / (U + V + W)`.
The edge functions of a shared edge are computed from the same numbers with exactly opposite sign, so a
ray is never lost between two triangles; an exactly zero edge function marks a degenerate (edge/vertex)
hit. Boxes are tested conservatively (far distance times `1 + 2 gamma(3)`, Ize 2013). First hit: nearer
child first, ties by lowest face id. All hits: sorted by (t, face). Batches: `rays_first_hit`,
`segments_hit`, and `rays_all_hits` in CSR form (count pass, fill pass) for dexelising a stock mesh.
Agreement with Moller-Trumbore brute force: t to 1e-12 relative.

**Closest point / signed distance**: closest point on a triangle by Voronoi regions (Ericson 2005
5.1.5), depth-first with the nearer child first and boxes beyond the best distance pruned. Sign
(Baerentzen & Aanaes 2005): `sign((p - c) . n)` with n the pseudo-normal of the closest feature: the face
normal (interior), the normalised sum of the two face normals (edge), the angle-weighted vertex normal
(vertex). Valid for closed, consistently oriented meshes (`info[INFO_CLOSED]`; `signed_distance` refuses
others unless `allow_open`). Exact for the tessellation up to rounding (agreement with brute force to
1e-12); against the true surface within the tessellation's chord error (sphere test: <= 2 tol).

**Point in mesh**: parity of crossings along fixed Fibonacci-spiral directions (no coordinate axis or
diagonal); a ray with a degenerate hit is discarded, a hit at t = 0 classifies the point ON_SURFACE; the
majority of `geom.point_in_mesh_rays` = 3 decided rays wins (outvotes one leak through a hole); points no
ray decides fall back to the generalised winding number `w = 1/(4 pi) sum Omega_f`, `tan(Omega/2) =
a.(b x c) / (|a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|)` (Van Oosterom & Strackee 1983; Jacobson et al.
2013), O(faces) per point, also available directly (`winding_numbers`). ON_SURFACE needs the point exactly
on a triangle; points within rounding distance may be classified either way.

## 6. Overlap (`overlap.py`)

* Triangle-triangle (Moller 1997): signed plane distances; a distance `<= 32 eps |n| |p - v0|` (the
  rounding of its own evaluation) counts as zero; intervals on the line of intersection (projected on its
  largest component) must overlap; coplanar pairs by 2-D edge-edge tests and containment; zero-area
  triangles by segment tests. Touching counts. Reference: edge-segment versus triangle tests (3000 random
  pairs agree).
* Triangle-box: 13 separating axes (Akenine-Moller 2001); reference: Sutherland-Hodgman clipping.
* Box-box: 15 separating axes (Gottschalk et al. 1996) with `|R_ij| + 64 eps` (Ericson 2005 4.4.1);
  reference: box B's surface triangles against A plus centre containment.
* Mesh-mesh: two trees descended together under `pose_A_B` (B's node boxes as oriented boxes in A), leaf
  triangles by Moller; all pairs or the first. A mesh entirely inside the other touches nothing: combine
  with `points_in_mesh` when containment matters.
* Mesh-box: the tree against an oriented box (half extents about its frame's origin) grown by a margin.
* **Mesh-vs-revolved envelope** (tool, holder; profile `(z_k, r_k)` z-monotone in frame S, axis z): broad
  phase by the envelope's oriented bounding box; narrow phase per triangle and profile segment. With
  `g(z)` the distance from the axis to the triangle's chord at height z, the clearance on a segment is
  `h(z) = g(z) - R(z)`. `g` is convex: it is the partial minimisation over the triangle of the convex
  distance-to-axis subject to the linear constraint `z(p) = z` (Boyd & Vandenberghe 2004, 3.2.5); R is
  linear on the segment, so h is convex and golden-section search (Kiefer 1953) converges to its minimum.
  h is Lipschitz with `L = s + |R'|`, s the largest horizontal run per unit height of the triangle's
  non-horizontal edges (the speed of the chord's end points), so a final bracket of `tol / L` bounds the
  error of the minimum by `tol` (`geom.clearance_tolerance_m` = 1 nm; at most 200 steps, after which the
  bracket is below one ulp). Beyond the ends (within `margin`) the clearance is `max(axial gap, g(z) -
  R_end)` (also convex, L + 1). Negative clearance = radial penetration depth. Exactness and meaning:
  overlap (clearance <= 0) is exact; on cylindrical portions and end faces the clearance equals the
  Euclidean gap; on a conical portion of slope angle alpha it exceeds it by `1/cos(alpha)`; beyond the ends
  it is between `1/sqrt(2)` and 1 times the Euclidean distance to the end disc. Checked against analytic
  gaps (1 mm axial, 2 mm radial, a face tilted 1 mrad: `(gap - R tan a) / (1 + tan a)`, where the axial and
  radial gaps are equal) and against dense sampling of random triangles.

## 7. Grid traversal (`dda.py`)

Amanatides & Woo (1987): the ray is clipped to the grid box and `[t0, t1]`, the entry cell is
`floor((p - lo)/cell)` clamped to the grid, and the walk advances the axis with the nearest boundary.
Every boundary is computed from the cell index, `t = (lo + (i + [step > 0]) cell - o) / d`, never by
accumulating increments, so long walks do not drift. Simultaneous crossings (edges, corners) advance one
axis at a time in x, y, z order: the cells touched only at that edge or corner appear with zero-length
intervals (conservative for occlusion tests). `ray_cells`, `segment_cells`, `first_occupied` (first cell
with non-zero `uint8` occupancy), `first_occupied_batch`. Reference: per-cell slab tests over the whole
grid (300 random rays: same cells in the same order, intervals to 1e-12).

## 8. Files (`io/`)

| format | read | write | unit | notes |
|---|---|---|---|---|
| STL (3D Systems 1989) | ASCII, binary | binary (default), ASCII | `unit` (mm default) | bit-identical corners welded; several ASCII solids = face groups; non-zero binary attribute words kept as face attribute `stl_attribute`; header `cnctwin binary STL` (never `solid`); float32 binary |
| OBJ (Wavefront) | v, f (all index forms, negative indices, polygons as fans), g/o | v, vn, f, g where the group changes | `unit` | vt, vn, usemtl, s, l, p are not surfaces: counted in `report['skipped']` |
| PLY (Turk 1994) | ascii, binary LE/BE | binary LE (default) or ascii | `unit` | normals `nx ny nz`; attributes as float properties (`name_k` columns regrouped); `face_group` int |
| 3MF (core 1.3) | model part from `_rels/.rels`, unit, objects, components (recursive), build items | objects + build items | the file's | 3MF transforms are 3x4 row-vector matrices (p' = p M): R = M^T, t = (m30, m31, m32); materials not read; production-extension external parts refused; zip entries with fixed timestamps |
| GLB (glTF 2.0) | nodes (matrix or TRS), triangle primitives (strips, fans), float and normalised integer accessors with strides, KHR_mesh_quantization | nodes, meshes, POSITION/NORMAL/indices, `_name` float attributes, materials from `color_rgba`, `split_groups` primitives with `extras.face_group`, quantised positions | metres | sparse accessors, external buffers, Draco/meshopt refused |

GLB quantisation: positions `q = round((p - lo)/step)` as uint16 (byte stride 8), one step for all axes
(`max extent / 65535`), dequantised by a child node `<name>#q` with matrix `diag(step) + lo`; error
<= `step / 2` per axis (0.06 um for an 8 mm keyframe brick, 4.6 um over 300 mm); normals as normalised
int8 (direction error <= 0.5/127 per component). A uniform step keeps the dequantisation a similarity
transform, so normals need no correction. Bytes are deterministic: JSON with sorted keys and fixed
separators, shortest round-trip floats, zero padding to 4 bytes.

## 9. STEP import and PMI (`io/step.py`, `cad.py`, `selectors.py`)

`read_step(path, chordal_tolerance_m, angular_tolerance_rad)` (OCP 8 / OCCT 8):
1. `STEPCAFControl_Reader` with names, colours and GD&T into an XCAF document. Shapes arrive in the session
   unit (`UnitsMethods::GetCasCadeLengthUnit`, mm) and are scaled to metres; the file's unit is reported
   from `STEPControl_Reader::FileUnits` (`millimetre`, `INCH`, ...).
2. `BRepTools::Clean`, then `BRepMesh_IncrementalMesh` sequentially (deterministic) with linear deflection
   = chord tolerance (`geom.step_chordal_tolerance_m` = 5 um, a quarter of an IT7 tolerance at 10-18 mm)
   and angular deflection (`geom.step_angular_tolerance_rad` = 10 deg).
3. Face id = 0-based index in `TopExp::MapShapes(FACE)`; triangles of REVERSED faces flipped; shared edge
   nodes are bit-identical and welded, so a closed solid gives a watertight mesh (`face_group` = face id).
4. Per face (`FaceInfo`): surface type and parameters from `BRepAdaptor_Surface` (plane point + outward
   normal; cylinder axis point, axis, radius; cone apex, axis from the apex, semi-angle, reference point
   and radius; sphere centre, radius; torus centre, axis, major and minor radius; B-spline/Bezier degrees,
   pole counts, rationality, periodicity; revolution/extrusion/offset/other by name). The outward sense of
   planes and the concavity of curved faces (`concave`: material outside the surface, i.e. a hole) come from
   the oriented tessellation (area-weighted sign of normal . radial), independent of OCC's surface
   parametrisation. Area, bounds, a point on the face (the largest triangle's centroid projected onto the
   analytic surface), the XCAF name.
5. Features: faces sharing an edge (`MapShapesAndAncestors(EDGE, FACE)`) whose analytic surfaces agree
   within `geom.feature_merge_distance_m` (1 um) and `geom.feature_merge_angle_rad` (0.01 deg): a cylinder
   split at its seam, a plane split by a fuse.
6. PMI (`XCAFDoc_DimTolTool`), the AP242 semantic subset of the CAx-IF recommended practices:
   `dimensional_size` / `dimensional_location` with `plus_minus_tolerance` or `limits_and_fits`,
   `geometric_tolerance` subtypes with `..._with_datum_reference` (`datum_system`,
   `datum_reference_compartment`) and `..._with_modifiers`, `datum` / `datum_feature`. Each item's shapes are
   mapped to face ids (`GetRefShapeLabel`; a face of an instanced part by `IsPartner`).

Mapping to the project's `cad` form (mm; validated in the tests with `cnctwin.project.schema`):

| PMI | project entry |
|---|---|
| datum | `{label, feature}` |
| Flatness, Position, Perpendicularity, Parallelism, Angularity, Cylindricity, CircularityOrRoundness, Straightness, ProfileOfSurface, ProfileOfLine, Concentricity, Coaxiality, Symmetry, CircularRunout, TotalRunout | `{id: pmi-gt-<n>, kind, feature, tolerance_mm, datums: [labels by precedence], modifier: M or L}` (`circularity`, `profile_surface`, `profile_line`, `concentricity` for coaxiality, `runout`, `total_runout`) |
| Size_Diameter / Size_SphericalDiameter, Size_Radius / Size_SphericalRadius | `{id: pmi-dim-<n>, kind: diameter or radius, feature, nominal_mm, plus_mm, minus_mm, diameter_mm: {nominal, fit}}` |
| Location_LinearDistance (all variants), Location_CurvedDistance, Size_Thickness with two shape sets | `{kind: distance, between: [feature, feature], nominal_mm, plus_mm, minus_mm}` |
| Location_Angular, Size_Angular | `{kind: angle, between, nominal_deg, plus_deg, minus_deg}` |

A `feature` is a selector `{face, normal or axis, near_mm, radius_mm, id: "face:<n>"}` (`selector_for_face`).
Limit dimensions become nominal = mid-range, plus = minus = half range. The fit (`H7`, `g6`) is the ISO 286
designation from the class of tolerance (hole: upper case; read through the object's `DumpJson`, the only
route to those out-parameters from Python). Items without a project form (datum targets, dimensions on
edges or vertices, oriented/path dimensions, tolerances on no face) are kept in `pmi_unmapped` with the
reason; `pmi_raw` holds everything read in SI (zone type, material and zone modifiers, modifier lists,
datum modifiers, XCAF entries).

OCCT 8 behaviours found and handled: (a) the reader returns the lower deviation as a magnitude, so a range
entirely above the nominal (+a/+b) reads as +b/-a; (b) the AP242 writer stores a fit grade as (enum value
+ 1), two above the IT number its reader expects, and (c) writes PMI length measures unconverted with the
file unit attached. `write_step_pmi` compensates both (b, c), and the tests check the written text
(`LIMITS_AND_FITS('H','hole','7',..)`, inch measures) and the round trip.

`write_step_pmi` output carries OCCT's write time in the STEP header (it makes test parts and is not part
of any result set). `read_step_isolated` runs the import in a child process (`python -m cnctwin.geom.io.step in out`) and
returns the `CadModel` through `save`/`load` (`.npz` mesh + `.json`), so OpenCASCADE's memory never lives in
the simulation process (architecture 10). `box_with_hole` and `write_step_pmi` build AP242 test parts.

**Selectors** (`select_faces`): `id` (`face:<n>` or a STEP face name), then surface type (`freeform` for
non-analytic, `any`), `normal` (planes: outward normal; curved faces: the normal at the point nearest
`near_mm`), `axis` (cylinder, cone, torus, either sign; a plane's normal up to sign) within
`geom.selector_angle_tolerance_rad` (1 deg), `radius_mm` (cylinder/sphere radius, torus minor radius, cone
radius at the nearest point) within `geom.selector_radius_tolerance_m` (0.01 mm). A direction the face type
has no notion of does not match. Best match: the smallest exact point-triangle distance from `near_mm`,
then angle error, then face id; without `near_mm` the smallest angle error, then the largest area. The
result is every face of the best face's feature. `pose_C_X` maps selectors written in another frame (the
stock frame) into the CAD frame.

## 10. Parameters (`db/models/geom.yaml`)

| name | value | source |
|---|---|---|
| tessellation_chordal_tolerance_m | 10 um | est (below control smoothing bands and collision margins) |
| tessellation_min_segments | 12 | est |
| step_chordal_tolerance_m | 5 um | est (quarter of IT7 at 10-18 mm) |
| step_angular_tolerance_rad | 10 deg | est |
| bvh_leaf_size, bvh_bins, bvh_traversal_cost | 4, 12, 1/8 | pharr-2016-pbrt |
| selector_angle_tolerance_rad, selector_radius_tolerance_m | 1 deg, 0.01 mm | est |
| feature_merge_distance_m, feature_merge_angle_rad | 1 um, 0.01 deg | est (STEP modelling tolerances 1e-4..1e-3 mm) |
| clearance_tolerance_m | 1 nm | kiefer-1953-golden (bracket rule) |
| point_in_mesh_rays | 3 | est |

Derived numerical constants (not parameters): slerp switch `eps^(1/4)`, Ize factor `1 + 2 gamma(3)`,
plane-distance zero `32 eps |n||p - v0|`, SAT guard `64 eps`, root tolerance 8 ulps, `MAX_DEPTH` 60, rigid
tolerance of computed poses `1000 eps` and of poses handed in by callers `1e-6` (`RIGID_INPUT_TOLERANCE`:
single-precision rotations pass, any real scale or shear does not).

## 11. Kernels and their references

Every kernel has an explicit signature (one specialisation, `cache=True, nogil=True,
error_model="numpy"`, no fastmath/parallel), writable C-contiguous arrays, scalars and the NamedTuples of
`geom/abi.py`. Each module's `KERNELS` tuple is checked for `len(signatures) == 1`.

| kernels | reference / check |
|---|---|
| transform `*_nb` | the NumPy functions of `transform` |
| `weld_representatives_nb`, `_lex_lower` | `weld_representatives_ref`, `bisect` |
| `build_nodes_nb` | `reference.build_tree_ref` (bit-identical) |
| `ray_triangle_nb`, `ray_first_hit_nb`, `ray_all_hits_nb`, batches | Moller-Trumbore brute force (`ray_hits_ref`) |
| `closest_point_triangle_nb`, `closest_point_nb`, `signed_distance_nb` | projection + edge distances, brute force, winding sign |
| `point_in_mesh_nb`, `winding_numbers_nb` | analytic sphere/box, `winding_number_ref` |
| `tri_tri_overlap_nb` (+ 2-D helpers) | edge-segment tests (`tri_tri_overlap_ref`) |
| `tri_box_overlap_nb`, `obb_overlap_nb` | clipping, triangle soup + containment |
| `tree_tree_overlap_nb`, `tree_box_overlap_nb` | all pairs |
| `chord_axis_distance_nb`, `triangle_clearance_nb`, `tree_revolved_overlap_nb` | dense sampling, analytic gaps |
| `poly_roots_nb` and helpers | `numpy.roots` + Newton (`poly_roots_ref`), `numpy.polyval` |
| ray kernels of primitives | closed forms, the torus quartic, tessellated meshes |
| `dda_cells_nb`, `dda_first_occupied_nb`, `grid_clip_nb` | per-cell slab tests (`dda_cells_ref`) |

## 12. Validity and error bounds (summary)

| quantity | bound | how established |
|---|---|---|
| transforms, Euler round trips | <= 4e-16 | tests over all sequences, gimbal cases |
| tessellated volume | `0 < V - V_h <= 2 A tol` | exact prism formula; analytic solids |
| ray t (mesh) | 1e-12 relative to brute force; watertight | tests; Woop et al. 2013 |
| analytic ray t | 8 ulps of the root (polished) | quartic reference |
| distance, sign | rounding (1e-12 rel.) on the mesh; chord tol to the true surface | brute force; sphere |
| revolved clearance | <= 1 nm | Lipschitz bracket; analytic and sampling tests |
| GLB quantised position | <= step/2 per axis | test |
| STEP mesh to B-rep | <= chord tolerance (BRepMesh linear deflection) | OCCT contract; volume test |
| selector distances | exact on the tessellation | vectorised Ericson test |

Validity: meshes up to the int32 index range; trees to any size (memory ~180 B/triangle); revolved
envelopes must be z-monotone (non-decreasing z, steps allowed; checked); point-in-mesh parity assumes a
closed mesh (holes are outvoted up to one leaking ray of three; the winding number handles larger holes).

### Assumptions and their consequences

* **Inscribed tessellation.** Curved surfaces become meshes whose vertices lie on the true surface, at a
  chordal tolerance of 10 um for primitives and envelopes (`tessellation_chordal_tolerance_m`, at least 12
  segments) and 5 um for STEP faces (`step_chordal_tolerance_m`, with 10 deg angular deflection).
  Consequence: a mesh lies inside a convex true surface: volumes are low by `0 < V - V_h <= 2 A tol`,
  distances to the true surface are off by up to the chord tolerance (sphere test <= 2 tol), selectors
  and STEP features are exact on the tessellation only. Collision envelopes are made conservative instead
  (`profile_polyline(conservative=True)`: circumscribed polygons), so they never lie inside the true profile.
* **Closed, consistently oriented meshes** for signed distance and point-in-mesh. Consequence: an open mesh
  is refused by `signed_distance` (unless `allow_open`); point-in-mesh parity survives one leaking ray of
  three, larger holes fall back to the winding number at O(faces) per point; a point within rounding
  distance of the surface may be classified either way.
* **Rigid poses.** Poses are rotations plus translations: computed ones are rigid to `1000 eps`, poses
  handed in to `1e-6` (`RIGID_INPUT_TOLERANCE`). Consequence: a scaled or sheared transform is rejected,
  never silently used (single-precision rotations pass).
* **Revolved envelopes are z-monotone about z** (non-decreasing z, steps allowed). Consequence: a profile
  that turns back in z is refused (checked); the mesh-vs-envelope clearance is exact for overlap, equals
  the Euclidean gap on cylindrical portions and end faces, exceeds it by `1/cos(alpha)` on a cone of slope
  alpha (a near-miss margin reads larger than the true gap) and lies between `1/sqrt(2)` and 1 times it
  beyond the ends.
* **Mesh-mesh overlap tests surfaces.** Consequence: a mesh entirely inside another touches nothing;
  callers combine it with `points_in_mesh` where containment matters.
* **File units.** STL, OBJ and PLY carry no unit: the `unit` argument (default mm, the CAD convention)
  scales them; 3MF uses its own unit, GLB metres, STEP the OCCT session unit (mm) scaled to metres.
  Consequence: a file written in another unit than declared comes in scaled by that ratio (25.4 for
  inches); STEP reports its file unit so a caller can check.
* **PMI mapping.** Limit dimensions become nominal = mid-range with symmetric plus/minus; PMI items without
  a project form (datum targets, dimensions on edges or vertices, oriented or path dimensions) are kept in
  `pmi_unmapped` with the reason. Consequence: those items are not inspected; the asymmetric limits of a
  dimension are kept as a symmetric band of the same width.
* **IEEE double arithmetic** everywhere; the tolerances of the predicates (`32 eps |n||p - v0|` plane
  distance, `64 eps` SAT guard, 8-ulp roots) are rounding bounds of their own evaluation. Consequence:
  touching counts as overlap; results are reproducible bit for bit on one build (determinism, section 13).

## 13. Determinism, RNG, bus

No random numbers (no RNG streams). Builds, traversals, ties (lowest face id), sorting of all-hits (by t,
then face), weld clustering, JSON and zip output are deterministic; nothing iterates over unordered sets.
Bus fields read: none. Bus fields written: none. Bus/ABI additions: none (the library's own layouts are in
`cnctwin/geom/abi.py`).

## 14. Integration notes

* removal: dexelise a stock or STEP mesh with `bvh.rays_all_hits` (CSR, sorted crossings per ray); classify
  points with `points_in_mesh`; walk bricks with `dda`.
* motion / collision: build machine and fixture trees once (`static_tree`), query per tick with relative
  poses (`mesh_mesh_overlap`, `mesh_box_overlap`, `mesh_revolved_overlap` for tool and holder envelopes
  from `profile_polyline(conservative=True)`); the trees are never rebuilt for motion.
* coolant / chips / probe: `segments_hit` and `rays_first_hit` against meshes, `dda.first_occupied` against
  brick occupancy; analytic primitives for nozzles and simple fixtures.
* render / keyframes / scene: `glb_bytes` with `GlbNode`s, `quantize=True` for bricks, `_name` attributes
  for per-vertex values, `split_groups` for B-rep faces; `read_glb` for tools and tests.
* project / inspect: `read_step_isolated` for `cad.path`, `CadModel.pmi` to merge with hand-given
  `cad.datums`/`cad.tolerances`, `select_faces` for their `feature` selectors (with the `cad.placement`
  pose as `pose_C_X`).
* controller: `fanuc_g68_2*`, `siemens_cycle800`, `heidenhain_plane_*` for tilted planes; `vec_slerp_nb` /
  `pose_interp_nb` for orientation interpolation in compiled code.

## References

pharr-2016-pbrt, wald-2007-bvh, macdonald-booth-1990, woop-2013-watertight, ize-2013-robust-bvh,
moller-trumbore-1997, moller-1997-tritri, akenine-moller-2001-tribox, gottschalk-1996-obbtree,
ericson-2005-rtcd, baerentzen-aanaes-2005, thurmer-wuthrich-1998, jacobson-2013-winding,
vanoosterom-strackee-1983, amanatides-woo-1987, kay-kajiya-1986, williams-2005-raybox, shoemake-1985-slerp,
shepperd-1978-quaternion, shoemake-1994-euler, kahan-2006-angles, press-2007-nr, kiefer-1953-golden,
boyd-vandenberghe-2004, fanuc-30i-b-operator, siemens-840d-cycles, heidenhain-tnc640-plane,
iso-10303-242-2020, cax-if-pmi-2014, occt-8-reference, khronos-gltf-2, khronos-khr-mesh-quantization,
threemf-core-1-3, stl-3dsystems-1989, turk-1994-ply, wavefront-obj (db/bibliography/models-geom.yaml).
