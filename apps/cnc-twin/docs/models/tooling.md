# Tooling: tool geometry as the physics sees it

Package `cnctwin/tooling/` (library + the `tooling` SimState field), tests `tests/unit/tooling/`,
coefficients `db/models/tooling.yaml`, sources `db/bibliography/models-tooling.yaml`.

One owner for every tool's geometry: the cutting edges of each tool instance as an
`EdgeModel` (core/abi.py), the revolved envelope, the thread tooth profiles, the 2-D insert outline
of non-rotating tools, the properties other domains read (chip space, sections, masses, balance,
speed limits, coolant holes), the project's `ToolTable`, and the per-macro-step composition of
the *effective* edges from wear, thermal growth, pull-out and slip (architecture 4.1).

| module | content |
|---|---|
| `spec.py` | tool item / holder / project instance -> SI values (`ToolSpec`, `HolderSpec`, `InstanceSpec`) |
| `profile.py` | meridian profile pieces (line, arc, parametric curve) |
| `edges.py` | discretisation, element geometry, face models, groups, EdgeModel packing |
| `mills.py`, `holemaking.py`, `indexable.py`, `turning.py`, `inserts.py` | type builders |
| `envelope.py` | revolved envelope, thread profiles, spindle nose from the machine |
| `props.py` | properties |
| `build.py` | `build_tool()` -> `ToolGeometry` (runout, tilt, everything above) |
| `table.py` | `build_tool_table()` -> `core.state.ToolTable` |
| `update.py`, `kernels.py` | composition and compiled kernels |
| `state.py`, `domain.py`, `channels.py`, `schema.py`, `catalog.py`, `abi.py` | domain plumbing |

## 1. Inputs

* **Tool item** (db/_templates/tool.yaml; a database id resolved by `cnctwin.project`, or an inline
  item): type, hand, tool material, coating, geometry (every field of `cnctwin.db.schemas`
  `GEOMETRY_FIELDS`), max_rpm, balance_grade, mass_g. Catalogue keys beyond the template that the
  database files carry are read when present: `wrap_angle_deg` (lollipops), `included_angle_deg`
  (dovetails), `serration {pitch_mm, depth_mm}`, `inserts.wiper_flat_mm`, `hub_diameter_mm`,
  `side_clearance_deg` (saws), `bore_mm`.
* **Holder** (db/_templates/holder.yaml): gauge length, nose diameter, `profile_mm` (z from the
  nose toward the spindle, diameter), bore, clamp length, runout (TIR), balance grade, max rpm,
  mass.
* **Project instance** (`tools.list[i]`): number, path (1-based in the file, 0-based inside),
  spindle, stickout, length/radius offsets, `runout_um` (TIR measured at the cutting edge),
  `runout_phase_deg`, `tilt_urad`, life (group, sister_of), `initial_wear.edge_radius_um`.
* **Settings** `physics.edge_element_mm` (default 0.5) and `physics.tooling` (`schema.py`
  `PHYSICS_TOOLING`, found by the project schema as `PHYSICS_TOOLING`): `rake_class_deg` (2),
  `incl_class_deg` (5), `max_kappa_step_deg` (7.5), `max_lag_step_deg` (5),
  `margin_element_factor` (4), `wave_elements` (8), `max_elements_per_tooth` (20000).
* **Defaults** for values a catalogue does not state come from `db/models/tooling.yaml` (table in
  section 13); every use is written to `ToolGeometry.notes` (and `finalize()` lists them).

## 2. Conventions

### 2.1 Rotating tools

T: origin at the tool tip (control point), +z_T toward the spindle (architecture 4). The edge
geometry is built in the tool's **cutting-sense frame C**: x = r cos psi, y = r sin psi, z, where
psi grows in the direction the edges move when the tool runs in its cutting direction. With the
hand sign **h** (`EdgeModel.hand_sign`, -1 right-hand: cuts with omega < 0 = M03; +1 left-hand) the
right-handed angle about +z_T is alpha = h psi, so C is T mirrored in y for a right-hand tool.

* `tooth_phase_rad[j]` is the cutting-sense position of tooth j, wrapped to [0, 2 pi); tooth j+1
  trails tooth j by `pitch_deg[j]` (the spacing in the order the teeth pass a fixed point), so
  phases are 0, -p0, -p0-p1, ... ; `elem_lag_rad[j, k]` is the element's cutting-sense position
  relative to its tooth (helix: negative = trailing).
* Element midpoint in T at spindle angle theta (right-handed): alpha = theta + h (phase + lag),
  p = (r cos alpha, r sin alpha, z). The kernel form uses `EdgeTrig` (cos/sin of h (phase + lag))
  and angle addition. The EdgeModel docstring's psi = sign theta + phase + lag is this with
  sign = h (= sign(omega) when the spindle turns the tool's way): psi = h alpha. **Use h, not
  sign(omega), for positions**; a spindle turning against the hand is detected by comparing
  sign(omega) with h.
* Normal into the material: n = sin(kappa) e_r - cos(kappa) e_z (kappa = 0 end edge, pi/2
  peripheral edge, > pi/2 back-cutting edges of keyseats, dovetails, lollipops). The chip side of
  the rake face is -n.
* Unit cutting direction of an element: +e_psi in C (`EdgeFrameT.cut_dir` in T).
* Runout: offset e = TIR/2 at the tip toward the azimuth `runout_phase_deg` (cutting sense from
  tooth 0, right-handed alpha_off = h phase); tilt (scalar) tilts the axis toward the same azimuth:
  `tilt_rad` = (-tau sin alpha_off, tau cos alpha_off) about (x_T, y_T), so a point at height z is
  displaced by tau z. Positions: p + omega x p + runout, omega = (tilt_x, tilt_y, 0) (small
  angles; second-order error tau^2 z, < 1e-12 m for tau < 1e-3 rad). Without a measured TIR the
  holder's catalogue TIR and the spindle's taper runout are combined RSS at phase 0
  (`runout_source` "catalogue").

### 2.2 Non-rotating tools (turning family, boring bars, form tools)

T: origin at the **imaginary tool tip** P (the tangents of the nose circle along x_T and z_T);
x_T points **into the work material** (toward the work axis for external tools, away from it for
internal tools); z_T points from the tip into the tool body along the work axis, so the major
edge faces -z_T and leads when the tool feeds along -z_T (grooving tools feed along +x_T);
y_T = z_T x x_T. The intended cutting velocity of the tool relative to the work is h y_T (the rake
face faces it). Elements lie in the reference plane: psi = 0, `elem_r_m` is the x_T coordinate
(<= 0), `elem_z_m` the z_T coordinate, the same normal formula holds with e_r = x_T.
kappa on the major edge is 90 deg - kappa_r (ISO entering angle from the feed direction), the
nose spans kappa_major .. 90 deg + kappa_r' (kappa_r' = 180 deg - kappa_r - epsilon_r), the minor
edge has 90 deg + kappa_r'. Threading inserts use the tip apex as x_T = 0 and the profile's
centre line as z_T = 0. `ToolProps.shank_dir_T` gives the shank direction in T (-x_T for
external holders, +z_T for boring bars); runout and tilt are not applied (no tool spindle).

## 3. Discretisation (edges.py)

Each tool is a **layout** (profile pieces in the meridian plane, shared by all teeth) and **teeth**
(phase, lag parameters, mask, optional per-tooth profile change or 3-D curve). Every piece gets
n elements, n = ceil(integral of the cost density), boundaries equidistant in the cost

    c(u) = max( ds/du / (h_e s),  max_j dS_j/du / (h_e s),  |d kappa/du| / k_max (<= 1 per sample at kinks),
                max_j |d psi_j/du| / psi_max )

(h_e = `edge_element_mm`, s = `element_scale` (margins: `margin_element_factor`; waves:
lambda / (`wave_elements` h_e)), ds meridian arc length, dS true 3-D edge length). Consequences:
no element is longer than h_e along the true edge, normals turn by at most `max_kappa_step` across
an element on arcs, angular positions change by at most `max_lag_step` (drill lips near the
chisel), and corners (kinks) add at most one boundary. **Stations** are the layout elements:
station k of every tooth is the same place on the layout (`elem_station[j, k] = k`,
`station_z_m/station_r_m` the layout midpoint), so teeth sweep the same circle there; teeth that
lack part of the profile (short end teeth, pads, insert halves of indexable drills, staggered
T-slot teeth, clipped tap teeth) have `elem_valid = 0` there (mask evaluated at the element
midpoint, so a partial element is counted with its midpoint: error < one element length).

Per element (16 sub-chords, exact for straight pieces; arcs give exact chip widths rho dtheta;
helical and parametric pieces have a chord error < (dtheta/16)^2/24 < 1e-6 relative):

* `elem_db_m` = meridian arc length (chip width; Lee & Altintas db = dz / sin kappa on the ball,
  dr on end edges: finite at kappa = 0). Margins and pads: the land width (contact width of
  rubbing).
* `elem_len_m` = true 3-D edge length, sqrt(ds^2 + (r dpsi)^2) integrated.
* `elem_kappa_rad` = atan2(dz, dr) of the meridian tangent at the midpoint.
* `elem_incl_rad` = lambda_s = asin(-t . e_v) (ISO 3002-1: angle between the edge and the reference
  plane normal to the cutting velocity), t oriented along the profile; tan(lambda) = -r dpsi/ds:
  the helix angle on a peripheral edge, r tan(i0) sin(kappa)/R0 on a constant-lead ball (the true
  inclination; Lee & Altintas' "local helix" r tan(i0)/R0 is the angle to the axis and equals it
  only at kappa = 90 deg), asin((w/r) sin p) on a drill lip.
* Rake and clearance from the **face model** (ISO 3002-1 normal plane P_n, m = cutting direction in
  P_n): rake face trace cos(g) (-n) - sin(g) m, flank trace -cos(a) m - sin(a) n.
  * `radial`: rake/clearance measured in the local radial plane (normal to the meridian tangent;
    the transverse plane of an end mill): tan g_n = tan g_r cos(lambda), tan a_n = tan a_r /
    cos(lambda) (Armarego & Brown 1969; Budak, Altintas & Armarego 1996). Solid mills, reamers,
    taps, gun drills, chamfer edges.
  * `helicoid`: the rake face is the helical flute of lead 2 pi R / tan(beta0) through the lip
    (drills): normal = t x h with the flute helix h; tends to tan g_n = tan(beta(r)) / sin p for a
    thin web (Armarego & Cheng 1972), falls to negative rake toward the chisel.
  * `insert`: seat rake plane N_r (from axial/radial rake or holder rake/inclination at the major
    edge) plus the chip-breaker rake normal to every edge; constant insert wedge, so
    a_n = alpha_i - g_seat (alpha_i: ISO 1832 normal clearance); turning tools take the tangent
    lifted into the rake plane for the inclination.
  * `normal`: given directly (chisel, margins, forming lobes).
* `elem_edge_radius_m` = `edge_radius_um`, else by edge preparation (sharp, hone, chamfer,
  waterfall; inserts) from the defaults; T-land width/angle go to the group table.

## 4. Tool families

**End mills** (square, ball, toroidal, roughing, serrated; mills.py). Profile: dished end edge
(kappa = -dish, `end_dish_rad`), corner radius arc (kappa 0..90 deg) or corner chamfer, periphery
to the cutting length. Helix lag of flute j: psi = -z tan(beta_j) / R (constant lead, also on the
corner/ball: Altintas 2012 ch. 2; Lee & Altintas 1996). End teeth are radial (lag 0); long end
teeth reach the axis when centre cutting (two opposite teeth for even flute counts, one for odd),
the others start at the core radius. Ball: arc of radius R about (R, 0); the element nearest the
tip has r ~ R sin(k_max/2) > 0: the vanishing cutting speed at the tip is handled by the element
velocity v_e = omega x r_e + v_feed (architecture 5.4). Roughing: sinusoidal knuckle wave
r_j(z) = R - d (1 - cos 2 pi (z/lambda + j/N)) / 2, serrated: trapezoidal notches (Merdol &
Altintas 2004), both shifted by lambda / N per flute and ramped in over the first wavelength.

**Chamfer mills, engravers**: tip flat (end edge) and cone at half tip angle; the cone is
peripheral (kappa = 90 deg - half angle). **Keyseat / T-slot / slitting saws**: disc head of width w,
periphery plus dished side edges on both faces from R to the neck (keyseat; T-slot: staggered,
even teeth bottom side, odd teeth top side, for even tooth counts >= 4); saws cut on the periphery
only. **Dovetails**: dished bottom, corner radius tangent to bottom and cone, cone at the half
included angle going inward (kappa = 90 deg + half); a neck wider than the cone top truncates the
cone (note). **Lollipops**: sphere arc from kappa = 0 to half the wrap angle (> 90 deg), constant
lead helix. **Thread mills**: rows of flank-crest-flank teeth (ISO 68-1 proportions, crest P/8,
depth 17H/24, H = P / (2 tan(form/2)), applied to any form angle), annular (no lead shift).

**Twist and spot drills** (holemaking.py). Half web w (from `web_thickness_mm`, `chisel_length_mm`
= 2w / sin(psi_ch), or `drill_web_ratio`). Chisel half from the axis to the corner
(x_c, w), x_c = -w / tan(psi_ch), r_c = w / sin(psi_ch): orthogonal edge (inclination 0, kappa 0),
rake g_ch = phi - 90 deg and clearance phi with tan(phi) = sqrt(cot^2 p + tan^2 a_c),
tan(a_c) = 1 / (tan(pi - psi_ch) tan p): the planar flanks of both lips meeting along the chisel
(about -54 deg for p = 59 deg, psi = 125 deg; Wiriyacosol & Armarego 1979 treat the chisel as
orthogonal cutting, Stephenson & Agapiou 2016 quote -55 to -60 deg). Lip: straight 3-D line from
the chisel corner to the outer corner at height (x_R - x_c) cot p, x_R = sqrt(R^2 - w^2); a lip
point at radius r sits at the web angle: lag = asin(w/r) - asin(w/R) (inner points lead),
inclination sin(i) = (w/r) sin p, rake from the flute helicoid (Armarego & Cheng 1972). Margins:
helical lands from the outer corner to the flute length with back taper `drill_back_taper`, chip
width = margin width, rake = clearance = 0. Spot drills: no margins, small web.
**Centre drills** (DIN 333 A): pilot drill, pilot margin, 60 deg countersink (lip-type edge).
**Gun drills**: inner and outer edge in a plane through the axis meeting at the point offset
D/4 (standard angles 30/20 deg, or split from `point_angle_deg`), two pad teeth (margins) at
`gun_pad_angles_rad` (Astakhov 2014). **Indexable drills**: central and peripheral insert on one
straight edge line (kappa = 90 deg - kappa_r), overlapping by `idrill_overlap_ratio`, nose radius
at the periphery. **Reamers**: chamfer lead (`corner_chamfer_mm`, half angle from `tip_angle_deg`
or 45 deg) and cylindrical lands with back taper (margins).

**Taps**: ISO 68-1 tooth rows on every land, each land shifted axially by
lead x (trailing angle) / 2 pi (spiral flutes: the implicit shift z' = (z + L phi_t/2pi) /
(1 - L tan(beta) / (2 pi R))), and every point ground down to the chamfer cone
r_c(z) = r_tip + (R - r_tip) z / L_ch, r_tip = R - 17H/24, L_ch = chamfer threads x P. Flank parts
above the cone become chamfer (crest) elements. Successive teeth along the helical path therefore
rise by h / (n_ch N): **the chamfer teeth share the thread depth** (Dogra, Kapoor & DeVor 2002);
the vector-form chip thickness of the hot loop sees it without any special case, because station
k = (row, profile position) is shared by the lands and each land's tool-frame z carries its lead
shift (`station_z_m` is land 0's). Full-thread lands are concentric (clearance 0), chamfer teeth
carry the relief; spiral-point taps add -`tap_spiral_point_angle_rad` to the chamfer inclination.
Forming taps: lobes as teeth, `forming` groups, rake -90 deg (no rake face: the coefficient
pipeline must use its forming model for these groups).

**Indexable mills** (face and shoulder mills; indexable.py): per insert at its position, the
minor edge or wiper flat (kappa 0), nose radius, major edge (kappa = kappa_r, milling entering
angle from the face plane) around the theoretical corner at D/2 on the face; the seat's rake plane
from tan(lambda_s) = tan(g_p) sin(kappa_r) - tan(g_f) cos(kappa_r), tan(g_n) = (tan(g_f) sin(kappa_r)
+ tan(g_p) cos(kappa_r)) cos(lambda_s) (Armarego & Brown 1969); every point is lifted onto the
plane (exact psi from r (cos psi N_x + sin psi N_y) = Q.N - z N_z), so the inclination and rake of
a straight inclined edge drift along the edge (-0.2 deg/mm for the 45 deg test cutter) as they
physically do. **Boring heads**: the same with one (or two) inserts at the set diameter (item
`diameter_mm`, else 2 x the instance's radius offset), axial feed (kappa = 90 deg - kappa_r).

**Turning family** (turning.py): turning inserts and boring bars (nose arc, major and minor edge
of the insert's edge length, wiper flat parallel to the feed for wiper inserts, round inserts as
the half circle facing the work: ap <= ic/2), grooving and parting (front edge, two corner radii,
sides with `groove_side_clearance_rad`; parting with an approach angle tilts the front edge),
threading inserts (V of the thread form with tip radius), form tools (`profile_mm` rows
[z_T, x_T], shifted so the deepest point is x_T = 0). Holder rake/inclination default to -6/-6 deg
for negative (0 deg clearance) inserts, 0/0 otherwise. Elements carry no y offset: the edge's
out-of-plane position from the inclination (<= edge length x sin(lambda)) is dropped (its angle
is kept); the generated surface comes from the nose, which lies in the plane.

## 5. Coefficient groups (`GroupTable`)

Group = (edge kind, round(rake / rake_class), round(incl / incl_class), T-land width and angle,
forming flag), numbered in sorted key order (deterministic). Rows: kind, mean rake, inclination,
clearance, hone radius, T-land, forming, member count — the columns of
`cnctwin.mechanics.coefficients.EdgeGroup`. Groups are fixed at build time; wear changes the
element hone radius (`elem_edge_radius_m`), not the grouping.

## 6. Envelope, thread profile, insert profile (abi.py)

`Envelope` (layout defined here; removal reads it): pieces in one closed chain per solid —
chain 0 the tool (cutting head with the cutting flag, fluted body to the flute length, neck to
its reach from the tip, shank to the overall length), chain 1 the holder (nose face at the
stickout, `profile_mm`, to the gauge line), chain 2 the spindle nose housing (a cylinder primitive
of the machine geometry coaxial with the spindle-nose link and starting at the nose plane,
`spindle_nose_from_machine`). Each chain runs counter-clockwise in (z, r) from the axis to the
axis; a piece is a frustum/disc (z0, r0)-(z1, r1) or an arc (centre, rho, a0..a1,
point(a) = (zc - rho cos a, rc + rho sin a)); drill lips are 16 frusta. The envelope is the union:
radius(z) = max over covering pieces (`envelope_radius_at`). `ThreadProfile`: per (tooth, row)
closed CCW polygons in (z, r) (taps: clipped by the chamfer cone, the tip and the thread length,
lead shift per land; thread mills: annular rows, lead 0; threading inserts: the insert's V in
(z_T, x_T)), plus pitch, lead, starts, thread hand, form angle, crest radius, depth. `InsertProfile`
(non-rotating tools): the closed outline of the insert in (z_T, x_T) with per-segment cutting
flags (parallelogram for rhombic/rectangular shapes, triangle for T/W, regular polygon for H/O/P),
nose centre and radius.

## 7. Properties (props.py, `ToolProps`)

* **Chip space**: gullet area per flute = `gullet_area_mm2` (the database's per-flute convention),
  else `flute_void_fraction` x pi (R^2 - r_c^2) / N (drills: `drill_flute_area_fraction` x pi R^2 /
  N); gullet volume per unit length = N x per-flute area (m^3/m); chip space = that x flute length.
  Core diameter from `core_diameter_mm` or `core_ratio_by_flutes`.
* **Fluted section**: core disc r_c plus N annular sectors (r_c..R) of angular width 2 pi f_s / N at
  the tooth phases, f_s the solid share of the annulus (Young & Budynas 2012 App. A):
  I_xx = pi r_c^4/4 + sum (R^4 - r_c^4)/4 (a/2 - cos(2 phi_j) sin(a)/2), I_yy with +, I_xy =
  sum (R^4 - r_c^4)/4 sin(2 phi_j) sin(a)/2. **Flute-count dependence**: N >= 3 evenly spaced is
  isotropic, I = pi r_c^4/4 + f_s pi (R^4 - r_c^4)/4; N = 2 is anisotropic,
  I_1,2 = pi r_c^4/4 + (R^4 - r_c^4)(a +- sin a)/4; N = 1 more so; variable pitch gives a small
  anisotropy. With the tool's own core and gullet the sector model gives I_min, I_max (D10 4FL,
  core 6, gullet 9.5 mm^2 per flute: D_eq = 0.765 D, 4 % below Kops & Vo); without them the mean is Kops & Vo's equivalent diameter
  0.8 D (Kops & Vo 1990, as used for the fluted section in Schmitz & Smith 2009) and the sector
  model only splits it into I_max, I_min. D_eq = (64 I_min / pi)^(1/4).
* **Critical sections** (z from the tip; `Section`): flute run-out (end of the fluted part,
  equivalent circular section of I_min), neck end (pi d^3/32, pi d^3/16), shank at the holder nose;
  turning shanks h x b at the clamp (b h^2/6, torsion 8 a^2 b^2 / (3a + 1.8b), Young & Budynas Table
  10.7), boring bars (pi d^3/32 at the overhang). Structure uses them for `ToolModes.sections_*`.
* **Mass, centre of mass, inertia**: tool = density x revolved tool chain x fill (fluted area /
  pi R^2 over the fluted length), density from the item's tool material in the database (or the
  class's representative grade), scaled to `mass_g` when given (COM kept); holder = revolved
  profile minus the bore over the clamp length, steel density, scaled to the catalogue mass;
  assembly by the parallel-axis theorem. 2000 slices: COM error < 1e-6 of the length; the holder's
  taper above the gauge line is not in its profile (its mass is smeared over the profile when the
  catalogue mass scales it: COM error up to ~(taper mass share) x (taper length), about 20-30 mm
  for 40-taper holders).
* **Balance** (ISO 21940-11): e_per = G / Omega at the speed the grade names (else the max speed),
  U_per = m e_per, for tool and holder (holding builds the unbalance vector from them).
* **Max speed**: min of the tool's and holder's published limits (None when neither is published).
* **Coolant holes**: count, diameter, total area, exit kind and exit positions/directions in T
  (tip/face: r = `coolant_exit_radius_ratio` R on the tip surface, trailing each flute by half a
  pitch; flute: mid-flute at the cutting length; periphery: radial), hand and hand sign.

## 8. Composition of the effective edge (update.py, `compose_effective`)

Architecture 4.1 order, the tool-side part (spindle growth and the chain pose are motion's,
deflection the hot loop's): with shank direction s = (s_r, s_z) in (e_r at psi = 0, z_T)
((0, 1) for rotating tools), free length L_f (stickout; overhang for non-rotating tools),
d = r s_r + z s_z:

    shift = holder_growth + pullout + length_growth * clip((L_f - d) / L_f, 0, 1)   (uniform strain)
    r_eff = r - shift s_r + radial_growth - radial_wear
    z_eff = z - shift s_z + axial_wear
    phase_eff = phase - slip            (rotating tools; slip > 0: the tool lags the spindle)
    edge_radius_eff = worn radius where wear gives one (> 0), else nominal (initial_wear at start)
    valid_eff = valid and not broken;   stations shifted by the rigid and thermal part.

Inputs: `prev.wear.tools[slot]` (radial_wear_m, axial_wear_m, edge_radius_m, broken),
`prev.thermal.tools[slot]` (radial_growth_m, length_growth_m, holder_growth_m),
`prev.holding.tools[slot]` (pullout_m, slip_rad). Per-element arrays must have the EdgeModel's
[n_teeth, n_elem] shape (empty arrays mean zero; any other shape is an error). For non-rotating
tools radial growth acts along +x_T (toward the work), length growth along -shank_dir. Chipping
enters through wear's radial/axial wear and worn hone radius (`ToolWear.chipped`, `VB_m`, `KT_m`
are wear's and cutting's quantities and do not move the edge here). Signs other domains must
follow: radial wear > 0 recedes the edge away from the material, axial wear > 0 along +z_T,
growths > 0 toward the material / longer tool, pull-out > 0 = tool longer, slip > 0 = the tool
lags the spindle (rotated against the cutting direction).

## 9. Kernels (kernels.py)

All `@njit(explicit signature, cache=True, nogil=True, error_model="numpy")`, one signature each
(test), each with a `*_ref` Python reference (test, equality to 1e-15 or bitwise). Types:
`EDGE_T = typeof(example_edge())` (EdgeModel with tooling's dtypes: float64, int32 kind/group/
station, int8 valid), `TRIG_T`, `FRAME_T`, `STATE_T`, `ENV_T`.

| kernel | signature | does | measured cost (this box, single core) |
|---|---|---|---|
| `edge_points_T0` | `void(EDGE_T, TRIG_T, f8 cos_t, f8 sin_t, f8[:,:,::1] out)` | element midpoints in T0 at the spindle angle, runout and tilt included | 12 ns/element (+5.8 us call overhead from Python) |
| `edge_frames_T0` | `void(FRAME_T, f8, f8, out_t, out_n, out_v, out_rake)` | rotate tangent, normal, cutting direction, rake normal to the spindle angle | 22 ns/element (+2 us) |
| `compose_effective` | `void(EDGE_T nominal, STATE_T, EDGE_T out)` | section 8 | 39 ns/element (+10 us), once per macro step per tool |
| `envelope_radius_at` | `f8(ENV_T, f8 z, i8 chain)` | envelope radius at z (chain < 0: union) | 13 ns per query for 10 pieces in compiled code (3.6 us from Python) |
| `envelope_radius_profile` | `void(ENV_T, f8[::1] z, i8 chain, f8[::1] out)` | the same for many z | 13 ns per point |

Non-rotating tools are evaluated at theta = 0 (cos 1, sin 0): their elements sit in T and the
workpiece rotates. Hot-loop use: per L0 step only the engaged stations are evaluated; with the trig tables and
cos/sin(theta) by angle addition, `edge_points_T0` costs ~12 ns per element and step (D10 4FL with
240 elements: 2.9 us per step when all are evaluated).

## 10. Domain (`ToolingDomain`, architecture 19)

* `init`: settings, `state.tools` = ToolTable (run tooling's init before the domains that size
  per-tool arrays), `state.tooling` = `ToolingState` (ToolGeometry per slot + effective edge,
  EdgeState, trig, frames, version); one trace event `tooling.tool` per slot (diameter, flutes,
  runout TIR, stickout, max speed, element count).
* `kernel_inputs(bus)`: `ToolingInputs(edges, trig, frames, slot)` per controller path from
  `bus.motion.paths[p].tool_slot` (no tool: zero-size EdgeModel of the same types).
* `absorb`: nothing (tooling has no hot-loop outputs). `end_operation`: nothing (geometry carries
  over; wear/growth are composed from the bus). `finalize`: per tool summary with the estimates used.
* `macro_step(prev, nxt)`: section 8 for every slot whose inputs changed (version += 1).
* **Bus read**: `wear.tools`, `thermal.tools`, `holding.tools` (prev), `motion.paths[*].tool_slot`.
  **Bus written: none** (tooling has no bus record).
* Channels (macro, max_shape (16,) per path): `tooling.slot`, `tooling.radius_eff_max_m`,
  `tooling.radius_change_m`, `tooling.tip_shift_m`, `tooling.edge_version`; values from
  `ToolingDomain.channel_values(ctx, state, bus)`.
* Checkpoint: `ToolingState.to_arrays()/from_arrays()` store everything (arrays + JSON meta);
  round trip is bitwise (test).
* Adapters (`catalog.py`): `ToolingCatalog.edge_model(slot)` (thermal's `ToolEdges`),
  `ToolingCatalog.tool(slot)` (the fields of chips' `ToolInfo`; `gullet_area_m2` there is the total
  of all flutes).
* **RNG streams: none** (tooling is deterministic geometry).

## 11. Validity, error bounds and how they were established

| quantity | bound | established by |
|---|---|---|
| element chip width db | exact on lines and arcs; < 1e-6 relative on helical/parametric pieces | tests vs pi R/2 (ball), L (periphery), r_c (chisel), r_eps (pi - eps) (nose) |
| edge length | exact on straight 3-D edges; < 1e-6 on helices | tests vs L / cos(beta), drill lip (x_R - x_c)/sin p |
| projected widths (sum db sin/cos kappa over a ball) | midpoint rule, < k_max^2/24 = 7e-4 | test vs R |
| inclination, rake, clearance (radial mode) | closed form to 1e-6 (chord tangents) | tests vs tan g_n = tan g_r cos(l), tan a_n = tan a_r / cos(l), ball true inclination |
| drill lip rake (helicoid) | -> tan(beta(r))/sin p for a thin web within 2e-3 rad | test |
| chisel rake | planar-flank model; literature -55..-60 deg vs model -54 deg (p 59, psi 125) | closed form test; comparison |
| insert seat angles | exact at the corner (classic relations); drift along straight edges is geometric | test of the face model at psi = 0 |
| tap depth sharing | crest on the cone to 1e-12 m, rise per tooth h/(n_ch N) to 1e-6 | test |
| runout radii | exact geometry; first order R + e cos within e^2/2R | test |
| fluted I | sector model +-20 % vs real flute shapes (D_eq +-5 %); Kops & Vo mean when data are missing | closed forms; 0.765 D vs 0.8 D (Kops & Vo) for the reference tool |
| masses | integration < 1e-6; holder COM error from the taper (section 7) | closed-form cylinder test, catalogue scaling test |
| composition | exact (linear superposition) | kernel vs reference bitwise; closed-form shifts |

Validity: rigid tool geometry at 20 degC (thermal and wear changes enter through section 8);
small runout/tilt (TIR << R, tilt < 1e-3 rad); straight-lipped conventional drill points
(split/S-points, thinned webs and multi-facet points are represented only through web thickness,
chisel angle and point angle); thread forms of ISO 68-1 proportions (other forms via the form
angle with the same proportions).

## 12. Assumptions and limitations (with consequences)

* End-tooth inner radius of short teeth and non-centre-cutting tools = core radius (maker gash
  geometry unknown; the uncut post when plunging may be 0.1-0.3 D in reality).
* End-tooth rake = the stated radial rake (the gash rake is not in the template); corner and ball
  elements use the constant radial rake converted with the local inclination (Lee & Altintas'
  constant-rake assumption).
* Drill lip relief is taken constant (stated or default) along the lip; conical grinding raises it
  toward the centre (the chisel model accounts for it through the chisel edge angle).
* Margins/pads are coarse elements (`margin_element_factor` x h_e) and rub only (chip width =
  land width); the gun drill's own corner land is not modelled (pads only).
* Turning elements keep y = 0 (section 4); insert seat errors (axial/radial insert runout of
  indexable cutters) are not modelled: the project instance has only the tool TIR.
* Thermal length growth is distributed as a uniform strain over the free length (ThermalFeedback
  gives one number); error up to length_growth x z / L_f at the top of the flutes.
* The envelope does not contain coolant holes, flute voids or holder clamping nuts beyond the
  profile; chain 1 starts at the stickout even if that is shorter than the flute length (note).
* Catalogue roughing/serration profiles are not in the template: defaults (est) apply unless the
  item gives `serration {pitch_mm, depth_mm}`.

## 13. Parameters (db/models/tooling.yaml; all est unless a standard is cited)

end_dish 1 deg; sharp / hone / T-land corner / insert edge radius 3 / 20 / 8 / 35 um
(Denkena & Biermann 2014 ranges); core ratio by flutes 0.5, 0.5, 0.55, 0.6, 0.65, 0.7; flute void
fraction 0.75; Kops & Vo ratio 0.8 (kops-vo-1990); knuckle wavelength 0.15 D, depth 0.03 D;
serration pitch 0.08 D, depth 0.015 D, notch fraction 0.35; drill web 0.15 D (spot 0.06 D),
chisel edge angle 125 deg, helix 30 deg (spot 20 deg), margin 0.07 D, back taper 5e-4, flute area
0.45, lip relief 10 deg; gun drill outer/inner angle 30/20 deg, point offset D/4, pads at 90/180
deg, pad width 0.1 D; indexable drill split 0.5 R, overlap 0.06 R; reamer lead 45 deg x 0.05 D,
land 0.02 D; thread form 60 deg (iso-68-1-1998), crest P/8 (iso-68-1-1998), depth 17H/24
(iso-965-1-2013), chamfer 4 threads, tap rake 8 deg, chamfer relief 6 deg, spiral-flute helix 40
deg, spiral point 12 deg, lands 3 (D <= 6.5 mm) / 4, forming lobes 4; centre drill 60 deg
(din-333-1986), body 2.5 d1, pilot 1.2 d1; neck 0.5 D; lollipop wrap 270 deg; saw hub 0.4 D;
wiper flat 0.15 ic (turning 1.5 r_eps); indexable cutter body 0.8 D (chip pockets); minor-edge
dish 1 deg; holder nose 2.6 x bore when a holder gives neither nose diameter nor profile (outline
a cylinder to the gauge line, conservative for collisions); insert shapes and clearances
(iso-1832-2017); negative holder -6/-6 deg; groove side relief 3 deg; boring bar / head approach
95 / 90 deg; densities steel 7850, heavy metal 17500, damped bar 8000 kg/m^3; representative tool
material per class; coolant exit at 0.5 R.

## 14. Additive interface changes

* `cnctwin.core.abi.EdgeModel.hand_sign` ([1] float64, default [-1.0] for fixtures; tooling always
  sets it): the tool's hand, section 2.1.
* New layouts in `cnctwin.tooling.abi`: `Envelope` (with `chain`), `ThreadProfile`, `InsertProfile`,
  `GroupTable`, `EdgeTrig`, `EdgeFrameT`, `EdgeState` (+ `STATE_*` indices), `ToolingInputs`.

## References

* Altintas, Y. (2012) Manufacturing Automation, 2nd ed., Cambridge University Press (altintas-2012-mfg-automation).
* Armarego, E.J.A.; Brown, R.H. (1969) The Machining of Metals, Prentice-Hall (armarego-brown-1969).
* Armarego, E.J.A.; Cheng, C.Y. (1972) Int. J. Mach. Tool Des. Res. 12(1) 17-35 (armarego-cheng-1972).
* Astakhov, V.P. (2014) Drills: Science and Technology of Advanced Operations, CRC (astakhov-2014).
* Budak, E.; Altintas, Y.; Armarego, E.J.A. (1996) ASME J. Manuf. Sci. Eng. 118(2) 216-224.
* Denkena, B.; Biermann, D. (2014) Cutting edge geometries, CIRP Annals 63(2) 631-653.
* DIN 333:1986 Centre drills 60 deg; ISO 68-1:1998; ISO 965-1:2013; ISO 1832:2017; ISO 3002-1:1982;
  ISO 21940-11:2016.
* Dogra, A.P.S.; Kapoor, S.G.; DeVor, R.E. (2002) ASME J. Manuf. Sci. Eng. 124(1) 18-25.
* Kops, L.; Vo, D.T. (1990) CIRP Annals 39(1) 93-96.
* Lee, P.; Altintas, Y. (1996) Int. J. Mach. Tools Manuf. 36(9) 1059-1072.
* Merdol, S.D.; Altintas, Y. (2004) ASME J. Manuf. Sci. Eng. 126(2) 317-326.
* Schmitz, T.L.; Smith, K.S. (2009) Machining Dynamics, Springer.
* Stephenson, D.A.; Agapiou, J.S. (2016) Metal Cutting Theory and Practice, 3rd ed., CRC.
* Wiriyacosol, S.; Armarego, E.J.A. (1979) CIRP Annals 28(1) 87-91.
* Young, W.C.; Budynas, R.G.; Sadegh, A.M. (2012) Roark's Formulas for Stress and Strain, 8th ed.
