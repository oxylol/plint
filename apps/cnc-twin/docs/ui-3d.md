# The 3D view (`web/js/view3d`)

The browser's 3D view of a run: the machine, the fixture, the in-process workpiece (IPW), the tool and the
deflected tool, chips and coolant, in real units at the time the shell gives, with result layers, capped
sections, measuring, picking and a tool-following camera. It reads nothing but the run's API (docs/api.md,
architecture 17), works offline (three.js r170 vendored in `web/vendor/three`, no CDN, no build step:
vanilla ES modules), on a desktop and on a phone, and is mounted by the UI shell (docs/ui.md "The 3D view
contract").

Contents: [contract](#the-contract) · [inputs](#inputs) · [frames](#frames-and-the-machine-state) ·
[scene](#scene-placement) · [workpiece](#the-in-process-workpiece) · [tool](#tool-and-deflected-tool) ·
[chips](#chips-and-piles) · [coolant](#coolant) · [layers](#layers) · [sections](#sections) ·
[picking](#picking) · [measuring](#measuring) · [camera](#camera) · [performance](#performance-and-phones) ·
[parameters](#parameters) · [decisions](#decisions-and-their-consequences) · [limits](#limits) ·
[tests](#tests) · [integration](#integration) · [files](#files) · [references](#references)

## The contract

```js
import { createView3D } from './view3d/index.js'
const view = await createView3D(containerEl, { api, runId })
view.setTime(t_s)                                  // simulated seconds; every animation frame while playing
view.setLayer(name | null)                         // 'temperature' 'chips' 'coolant' 'finish' 'deviation' 'deflection' 'wear'
view.setSection({ axis: 'x' | 'y' | 'z' | null, offset_m, flip })
view.setMeasure(on); view.followTool(on); view.resetCamera()
view.setVisible({ machine, fixture, chips, coolant, shadows, tool, deflectedTool })
view.on('pick', cb)     // cb({ point_m, normal, layerValue, object, cap, brick, layer })
view.on('measure', cb)  // cb({ distance_m, points, kind, ... })
view.legend(layer)      // {min, max, unit, map, source} in display units, {categories}, or null while computing
view.destroy()
```

| call | what it does |
|---|---|
| `createView3D(container, {api, runId})` | reads the run record, its file list, scene.json, the keyframe index, bricks.idx, channels.json, timeline.json and the frames overview; loads the scene meshes and tool envelopes; appends a WebGL canvas (class `view3d-canvas`, first child) and an HTML overlay to the container (which the shell gives `position: relative` and `touch-action: none`). Rejects with a message when the run has no scene.json yet, the api object has no `get`/`getBinary`, or the browser has no WebGL 2. |
| `setTime(t_s)` | the time to show; the frame is drawn on the next animation frame (on demand, never in an idle loop). Times between L1 ticks are interpolated; before the first frame or after the last the nearest record holds. |
| `setLayer(name)` | colours the part, the tool and the chips by the layer ([layers](#layers)); `null` for material colours; an unknown name throws. |
| `setSection({axis, offset_m, flip})` | a plane normal to x, y or z of the stock frame S at `offset_m` (0 to the stock size); `flip` false removes the material beyond the offset (larger coordinate), true the material before it; `axis: null` turns it off. |
| `setMeasure(on)` | measuring mode: a toolbar in the view chooses distance, to plane or hole diameter; picks add points. |
| `followTool(on)`, `resetCamera()` | the orbit centre follows the tool tip / the setup view again. |
| `setVisible({...})` | the seven keys (missing keys keep their state); `shadows` are the coolant shadows (blocked jets). |
| `on(event, cb)` | `'pick'` on every tap or click on a surface, `'measure'` when a measurement is complete; returns an unsubscribe function. |
| `legend(layer)` | the colour bar of a layer once its range is known (the shell then shows it; it equals the shell's own computation, [ranges](#colour-ranges)). |
| `destroy()` | stops drawing and loading, removes the canvas and the overlay, frees GPU memory. |

Additional methods (not used by the shell): `setExaggeration(factor)` (deflected tool), `pickAt(x, y)`
(container pixels; returns the pick event), `project(point_S)` (container pixel of an S point), `debug()`
(state for tests: brick versions shown, chunks and levels, sample, layer mode, section, measure, tools, jets,
notes, renderer statistics).

`api` is any object with `get(path) -> JSON` and `getBinary(path) -> ArrayBuffer` for the paths of
architecture 17; the shell's client (web/js/api.js) is accepted with its own spellings (`json`,
`arrayBuffer`, `binary` ...). `point_m` is in S, metres; `normal` a unit vector in S; `layerValue` SI (K for
temperature, m for chips, finish, deviation, deflection and wear, 0/1 wetted for coolant) or null.

## Inputs

Only files the run's listing shows are requested (no 404s while a run is being written); a running run's
index files are read again every `refresh_period_s` (3 s) until it ends, then the final attributes.

| request | used for |
|---|---|
| `GET /api/runs/{id}` · `/files` | status (running or not), the material id; which files exist |
| `/scene` | machine components, fixture, enclosure, stock (`size_m`, `mesh`, `pose_W_S`, `pose_M_S`, `pose_F_S`, `bricks`, `spin`), `chain`/`kinematics`, `axes`, `tools`, `nozzles`, `gravity_M`, `workpieces`, `frames` (the frames.bin header) |
| `/files/scene/<name>.glb` | component meshes, tool envelopes and holders (three.js GLTFLoader: node transforms, KHR_mesh_quantization) |
| `/frames?format=bin&max_frames=20000` · `?t0=&t1=` | the whole-run overview and 4096-tick full-resolution chunks around the time shown |
| `/keyframes` · `/files/keyframes/bricks.idx` | keyframe times; every brick's versions |
| `/files/keyframes/kf_<k>.glb` · `/bricks/{b}?t=` | brick records: a whole keyframe file, or single bricks when few are needed from a large file |
| `/keyframes/{k}/fields?format=npz` | fields of keyframe k: `pose_M_S`, temperature grids, `pile_height_m`, `wetted`, displacement grids, `tool_deflection_m`, `jet_*` (the rows `/jets?t=` returns) |
| `/chips?t=<t_k>&format=npz` | chips in flight (`pos_M_m`, `vel_M_m_s`, `mass_kg`, `length_m`, `temp_K`, `colour_rgb`) and the pile (`pile_*`) |
| `/files/keyframes/attr/brick_<b>.npz` | final per-vertex attributes (deviation, finish, wear) of a brick's last version |
| `/channels` · `/timeline` · `/series?tier=...` | the tool displacement channel (tick tier), the active tool per path, edge temperature and VB for the tool's layer colour |
| `/api/db/materials` · `/api/db/materials/{id}` | the material family (appearance colour) and density (chip size) when run.json's project does not carry them |

## Frames and the machine state

`frames.js` is the only module that knows the frames.bin layout (docs/formats.md "frames.bin"): the header in
scene.json names each field's dtype, shape and offset; records are decoded generically (unknown fields are
kept). Axis values are the recorded ones plus `axes[].origin` (machine coordinates; relative to the stock
origin when the header gives no origins), rotary axes unwrapped with `axis_rev`; the spindle angle unwrapped
with `spindle_rev`.

Between L1 ticks i and i+1 (w in [0, 1]): axis values linear in axis space, spindle angle linear in its
unwrapped value, tool tip linear in S, tool axis normalised linear, tool number of tick i. Error bound of the
linear interpolation over one 1 ms tick: `a dt^2 / 8` for an axis accelerating at `a` (0.6 um at 5 m/s^2), the
chord error `v^2 dt^2 / (8 r)` on an arc of radius r at feed v (0.06 um for 3 m/min on a 5 mm arc), and
`alpha dt^2 / 8` on the spindle angle during a ramp (1e-4 rad at 800 rad/s^2): below what the screen shows. While
a full-resolution chunk loads, the overview (every n-th record) is interpolated and the status line says so.
The tool pose is read, never computed: the view does not re-implement the machine's kinematics.

## Scene placement

The view draws in the stock frame S of workpiece 0 (metres, z up for mills). A spinning stock
(`scene.stock.spin = {path, axis_S, point_S_m}`, lathes) is drawn in the non-rotating W0 instead: everything
placed in S is turned by the work spindle angle of that path about `axis_S` (its reference axis, toward the
headstock), so the part turns and the tool stays still, as on the machine.

| component placement | placed with |
|---|---|
| `frame: "S"` + `pose` | the pose in S |
| `frame: "W"` | `inv(stock.pose_W_S) pose` |
| `frame: "M"` | `pose_S_M(t) pose` |
| `frame: "T0"` | the non-rotating tool frame of its `path` (default 0) |
| `frame: "T"` | the tool frame turning with the spindle; components of kind spindle, head, machine or housing are drawn in T0 (housings do not turn) |
| `link: <name>` without a frame | in the tool chain: T0 (meshes written about the tool tip, as cnctwin.render reads them); in the work chain: the chain's pose of that link from the axis values, `pose_S_M(t) pose_M_link(q(t)) pose` |
| neither | M |

Chains: `scene.chain.{work, tool}` (or `kinematics.{workpiece_chain, tool_chain}`) as lists of link names,
axis names, or objects `{link, axis, direction, point_m | axis_point_m, fixed | matrix | pose}`;
`T_i = T_{i-1} J_i(q) X_i` with J a translation along `direction` or a rotation about the line through
`point_m` (docs/models/motion.md 5, nominal: the error transforms are in the tool and part poses only). A chain
is used only when every joint has a direction.

`pose_S_M(t) = inv(pose_M_S(t))`, `pose_M_S` from, in order: the work chain when the scene gives the stock in
the chain-end frame (`stock.pose_F_S`); the keyframes' `pose_M_S` (interpolated between keyframes: translation
linear, rotation by slerp; or moved with the work chain between two keyframes when the chain reproduces the
second keyframe's pose within 0.1 mm and 1 mrad from the exact axis values at both keyframe ticks); the scene's
static `stock.pose_M_S`. Without any, objects in M are not drawn and the status line says so. The tool frame T0
has z along the recorded tool axis and x the machine's x axis carried into S made perpendicular to it (exactly
T0 for three-axis mills and table-tilting five-axis machines; for spindle-head machines it is the nearest frame
to it, which only affects the drawn spindle angle of an axisymmetric tool).

## The in-process workpiece

**Versions.** Each brick shows its newest bricks.idx entry whose keyframe time is at or before t (`time <= t
+ 1e-12`, the rule of `/bricks/{b}?t=`, so a brick read alone is the version the index names; KeyframeReader
rounds t to base ticks instead, which differs only within half a tick of a keyframe). Before the first
keyframe the stock mesh is shown; it stays until every shown brick is on screen. An emptied brick (a record
without vertices) is removed.

**Loading** (`BrickLoader`): records already in the cache are shown at once; missing ones are read per brick
(`/bricks/{b}?t=`) when at most `max_bricks_per_file_read` (6) are needed from a file and they are under a
quarter of it, else the keyframe file is read once and every indexed record in it is cached. At most
`max_parallel_requests` (4) requests are in flight; the cache keeps `brick_cache_bytes` (96 MiB; 32 MiB on
phones) of records, least recently used first out, never a shown one. While playing forward the next keyframe's
records, fields and chips are read ahead (`prefetch_keyframes`). Replacement is incremental: only the chunks
of changed bricks are rebuilt, and a chunk keeps showing its old meshes until the new ones are built (no
flicker). Records are decoded from the BIN chunk directly (they lie one after the other in node order); a file
whose records are not BRK1 is read through the glTF accessors of its `brick_<b>` nodes instead.

**Chunks and level of detail** (`ipw.js`, `lod.js`): bricks are merged into chunks of `chunk_bricks`^3
(4^3 = 64 bricks, one draw call). Each chunk is drawn at the coarsest level whose clustering cube projects to at
most `max_lod_cluster_px` device pixels (1.5; 2.5 on phones): level 0 is the recorded mesh, levels 1 and 2
vertex clusters (Rossignac and Borrel 1993) of 4 and 16 lattice cells: vertices in one cube merge into their
mean, triangles with two corners in a cube vanish; every vertex moves by less than the cube diagonal
(sqrt(3) cells), so the shape changes by at most about two pixels on screen. Levels are built when first
needed, within `rebuild_budget_s` (8 ms) per frame, nearest chunks first. With a section on, level 0 is drawn
everywhere (vertex clustering can open a closed surface, and the caps need closed ones). Shading is flat
(normals from screen-space derivatives): the recorded facets are shown as they are and no normals are stored.
Multi-workpiece runs: bricks `w<w>.<b>` are placed with `scene.workpieces[w].pose_S0_S`; without it they are
not drawn (note).

## Tool and deflected tool

Each path's tool is its scene.tools entry by frames.bin's `tool_number`/`tool_slot` field, else by
timeline.json's tool of the block running on that path, else the first tool; drawn from its envelope (GLB in
T: origin at the tip, +z toward the spindle) and holder envelope, or as a cylinder of its diameter and
stickout when it has no envelope, placed at the recorded tip with z along the recorded tool axis and turned by
the spindle angle (right-handed about +z_T: M03 turns it clockwise seen from the spindle).

The **deflected tool** is the envelope bent as a cantilever clamped at the holder face (z_T = stickout_m, else
length_m, else the envelope's top) under a load at its tip (Euler-Bernoulli beam, Timoshenko and Gere 1972):
lateral displacement `u_xy (3 xi^2 - xi^3) / 2`, axial `u_z xi`, `xi = (z_c - z) / z_c` (0 at the holder face, 1
at the tip), nothing inside the holder, all times the exaggeration factor (default 1000, steps 1-2-5 from 1 to
100 000, control in the view). The shape is exact for a uniform prismatic beam loaded at its tip; for the same
tip deflection, a load spread evenly along the flutes gives a shape within 0.043 u of it (at 0.58 of the
stickout). u is, for path 0, the tick-tier value at t of a tool displacement channel of the dynamics domain
(unit m, `*tool*displacement*` or `*tool*deflection*`, 2 or 3 components in T0; the `.mean` of a fast
channel's envelope), else the keyframe's `tool_deflection_m` (T0 components; `tool_deflection_S_m` in S),
held until the next keyframe. It is drawn translucent in orange over the tool where it really is.

## Chips and piles

Chips in flight are one instanced mesh: a curled ribbon (`chip_aspect` thickness : width : length = 1 : 5 : 12,
curl 150 deg) per chip, its length along the chip's velocity (else along a direction hashed from its index,
stream `view3d.chip_orientation`), of length `length_m`, else `cbrt(m / (rho a))` from its mass and the
material density at that aspect (a = 5/144), else 1.2 mm; never smaller than 1.5 device pixels; in its own
colour (`colour_rgb`, the temper colour of the chips model) or, with the temperature layer, by `temp_K`. They are
given in M and drawn with their own keyframe's `pose_M_S`: the snapshot is exact at the keyframe time and held
until the next (the chips model's trajectories include drag, collisions and coolant forces the view does not
repeat). Piles: the pile grid (chips file `pile_*`, else the fields' `pile_height_m`) as a height field of the
cells higher than 0.1 mm on the grid plane (or on `pile_base_m`), corner heights the mean of the adjacent
occupied cells, coloured by the chips' mean colour or by height with the chips layer.

## Coolant

Nozzles are drawn from `scene.nozzles` (a body behind the orifice, along the aim); head-mounted ones
(`attached_to` a tool-chain link, a head, spindle or tool) at the origins of their jets instead, since the
scene's position of a moving nozzle belongs to no particular time. Jets are the `jet_*` rows of the newest
keyframe: along the coolant
model's own path `jet_path_M_m` when recorded, else along the free trajectory from the nozzle: straight for
gas, mist and MQL, for liquid jets the ballistic curve `p(s) = p0 + v0 t + g t^2 / 2`, `t = s / |v0|`, g =
standard gravity along `gravity_M`. Validity: the coherent jet; the momentum it loses to skin friction in still
air, `2 rho_air C_f L / (rho_w d)`, is 0.4 % per metre for a 3 mm water jet (C_f ~ 0.005, an estimate), so the
curve is its centreline until it breaks up; beyond break-up the drops slow and the curve overestimates the
spray's reach (the model's path is drawn when the run records it). The centreline is cut at the first obstacle
it meets (part bricks, fixture, machine, tool: a ray per segment) or at `jet_max_length_m` (600 mm). The radius
starts at `jet_diameter_m / 2`, else at the continuity radius `sqrt(Q / (pi v))`, and widens with
`jet_half_angle_rad`; opacity fades along it (as cnctwin.render draws jets), and beyond `jet_breakup_m` (when
recorded) the stream is drawn at its end opacity: spray, not a coherent jet. Jets of nozzles `attached_to` a
tool-chain link (or a head, spindle or tool) ride with the tool between keyframes; the others stay where the
keyframe has them.

**Shadows** (`setVisible({shadows: true})`): a jet that the run marks blocked (`jet_blocked_by`) or whose
centreline is stopped before reaching the cutting zone (the coolant model's tool zone, docs/models/coolant.md
4.2: a sphere of `zone_radius_factor` (1.5) tool radii, at least 1 mm, centred `tool_zone_height_ratio` (1)
tool radii above the tip) is continued past the obstacle along its unobstructed trajectory to its closest
approach to the zone, in the "shadowed flow" colour: the flow the obstacle takes away from the cut. A
continuation that would miss the zone by more than four zone radii (and 20 mm) is not drawn: that jet was not
aimed at the cut. **Spray**: drops leave the impact point on short ballistic arcs; drop i of a jet has a phase
from hash(`view3d.spray`, jet, i) and restarts every 0.3 s of simulated time with new directions, so the same
time always shows the same drops and playback animates them. A visual cue, not a drop field.

## Layers

| layer | the part | also | display |
|---|---|---|---|
| temperature | a 3-D `*_temp_K` grid (`work_temp_K`) per fragment, else a 2-D one (`surface_temp_K`) on upward faces | tool: edge temperature (`thermal.tool_edge_max_K`, else `thermal.edge_temp_K`); chips by `temp_K` | °C, the shell's orange ramp |
| chips | pile height above the part (2-D pile grid) where it is above 0.1 mm | piles coloured by height | mm, blue ramp |
| coolant | wetted cells (2-D `wetted` grid, its own or the pile/surface grid) wetted / dry | jets in the jet colour, shadows in the shadowed-flow colour | categories |
| finish | `Ra_m` (or `finish_Ra_m`, `_FINISH`) per vertex | | µm, blue ramp |
| deviation | `deviation_m` (or `_DEV`) per vertex | | µm, diverging, symmetric about 0 |
| deflection | the magnitude of a 3-D displacement grid (`*displacement*`/`*deflection*`, [.., 3]) when recorded | tool: \|deflection\| | µm, blue ramp |
| wear | `generating_VB_m` (or `VB_m`, `_WEAR`) per vertex | tool: largest VB of the teeth (`wear.VB_m`) | µm, blue ramp |

**Fields** are sampled per fragment from RG half-float textures: R the value mapped to [0, 1] over the grid's
own range (so half precision keeps 2^-11 of that range whatever the unit), G the validity; linear filtering of
R and G and their ratio R/G is normalised convolution (Knutsson and Westin 1993): cells without material never
bleed into the part; inside a field the error is that of linear interpolation, `(cell^2 / 8) |f''|` per axis.
Grid conventions (fields.js): origin and cell from `<name>_origin_S_m`/`<name>_cell_m`, else with the name's
last tokens dropped, else the pile grid, then the surface grid; 3-D grids are `values[nx, ny, nz]`
(FieldGrid); a 2-D grid is read `[nx, ny]` unless only `[ny, nx]` covers the stock with less than one cell of
padding (the shell's rule, so both read a grid alike); a 2-D grid's axes are `<prefix>_axes_S` (default x, y of
S) and it is projected along its normal onto faces within 60 deg of that normal (`min_surface_field_cos`): a
top-face map does not paint the walls. Points more than half a cell outside a grid have no value.
**Attributes** are per vertex of a brick's final version (keyframes/attr/brick_<b>.npz, in the vertex order of
the brick's last record); a brick shown at an earlier version keeps the material colour (its vertices are not
those of the attribute file). Values average within level-of-detail clusters. A share
`layer_unlit_fraction` (0.35) of a layer colour is shown unlit, so a face turned away from the light still
reads as its value on the bar.

### Colour ranges

Every layer's range is computed with the shell's rules (web/js/panels/layers.js), so the part and the shell's
colour bar always agree: fields over 12 keyframes sampled evenly (`field_sample_keyframes`), attributes over 24
attribute files (`attribute_sample_bricks`), minimum to maximum of the finite values, deviation symmetric about
zero, a degenerate range widened by 1e-9 display units; float16 and float32 values taken at numpy's shortest
decimal digits, as the API's JSON carries them (`shortestRepr`, ties to the even digit). The browser test
checks the two ranges are equal for every layer. Colour maps are the shell's (`lut(LAYER_MAPS[layer], mode)`
of web/js/plot/colormap.js, OKLab-interpolated, light/dark mode), as 256-entry sRGB lookup textures.

## Sections

One clipping plane normal to x, y or z of S cuts the part, the fixture, the chips and the piles (machine,
tool, enclosure and coolant stay whole, as in CAD section views). The plane keeps `n.p + c >= 0`: for `flip`
false the material beyond the offset is removed. The cut faces are capped flat (unlit) in red for the part and
lilac for the fixture with the stencil parity of closed solids (McReynolds and Blythe 2005, 16.10): the
clipped solids' back faces increment and front faces decrement the stencil, a plane is drawn where it is not
zero; the part and the fixture are capped one after the other. Exact for closed meshes (the union of the brick
meshes is; so are the fixture solids); an open mesh leaks cap colour along its opening; a mesh face lying
exactly in the section plane z-fights with the cap.

## Picking

A tap or click (moving less than 6 CSS pixels within 0.5 s; longer or farther is an orbit) casts a ray: against
the shown bricks' own triangles (level 0, Moeller and Trumbore 1997; per-brick bounding boxes first), against
the fixture, machine and tool meshes (three.js Raycaster), and with a section on, skipping hits on the removed
side; a ray that enters the kept side inside material (its first hit beyond the plane is a back face) hits the
cap. The nearest hit gives `point_m` and the face `normal` (outward: triangles are counter-clockwise from
outside), the `object` (part, fixture, machine, tool), the brick, and the layer value there: the attribute
interpolated barycentrically over the hit triangle, the field sampled as the shader samples it, wetted 0/1 for
coolant; on the tool the tool's layer value.

## Measuring

`setMeasure(true)` shows a toolbar: **Distance** (two picks: `distance_m` between them, and `delta_m`), **To
plane** (the first pick gives the plane through it normal to the surface there, the second the point:
`distance_m` = |(p2 - p1) . n|, `signed_distance_m`, `foot_m`), **Hole Ø** (three picks on a circle of the hole,
its rim or the wall at one height: the circumcircle, centre `p1 + (|b|^2 (n x a) + |a|^2 (b x n)) / (2 |n|^2)`
with a = p2 - p1, b = p3 - p1, n = a x b; `distance_m` = `diameter_m`, `center_m`, `normal`; three points closer
to a line than `min_circle_sine` (the sine of the triangle's smallest angle, 0.02) are refused). Markers keep 8
device pixels; labels sit at the measurement in display units (µm below 1 mm, mm below 1 m). Accuracy: picks lie
on the recorded mesh, which is within half a lattice cell of the machined surface plus the brick quantisation
(step / 2, 0.06 um for 8 mm bricks); a circle from three points off one height of a tilted wall is the circle
through those points, not the hole's.

## Camera

Perspective (30 deg vertical field), +z of S up. OrbitControls: one finger or left mouse turns, two fingers
pinch to zoom and drag to pan, right mouse or shift-drag pans, the wheel zooms toward the pointer; damped. The
setup view frames the stock from azimuth -55 deg, elevation 32 deg (the run pictures' follow direction).
`followTool(true)` frames 70 mm around the tip when the camera is far, then carries the orbit centre with the tip
(exponential smoothing, 0.12 s of wall time; a jump of more than half the view, a seek, is followed at once) while
the user can still turn and zoom. Near and far planes follow the orbit distance (near at 0.2 % of it: a 24-bit
depth buffer resolves 3e-5 of the distance, 3 um at 0.1 m), so a micrometre deflection and a metre-high machine
share one depth buffer.

## Performance and phones

* Drawn on demand: a frame when the time, the camera (and its damping), the data or an animation changes.
* One draw call per 64-brick chunk; levels of detail by projected size; three.js frustum culling per chunk and
  per mesh; chips as one instanced mesh; CPU copies of chunk geometry released after upload (bricks are kept
  as their compact records, 8 bytes per vertex and 12 per triangle).
* Device pixel ratio capped at 2 (desktop) and 1.5 (coarse pointers); multisampling only up to 1.5.
* Phones (`(pointer: coarse)`): coarser levels (2.5 px), a 32 MiB brick cache, 44 px buttons.
* Measured on the synthetic rich run in SwiftShader (software GL, one CPU): 16-21 draw calls, 25 k triangles,
  mount in about 0.2 s after the API answers; a 64-chunk rebuild budget of 8 ms per frame.
* A lost WebGL context (phones in the background) rebuilds the workpiece from its records when restored.

## Parameters

`db/models/view3d.yaml` (sources in `db/bibliography/models-view3d.yaml`), read in the browser from
`web/js/view3d/params.js`, which `tests/unit/view3d/sync_params.py` writes from it (a test fails when stale).
Values shared with other models are kept equal by a test: the cutting zone (coolant-occlusion.yaml), standard
gravity (cnctwin.units), appearance colours and cameras (render.yaml).

| parameter | value | source / note |
|---|---|---|
| `max_pixel_ratio`, `max_pixel_ratio_touch`, `max_antialias_pixel_ratio` | 2, 1.5, 1.5 | est; foley-1990 (supersampling) |
| `max_lod_cluster_px`, `max_lod_cluster_px_touch`, `lod_cell_factors` | 1.5, 2.5, [4, 16] | rossignac-borrel-1993 |
| `chunk_bricks`, `rebuild_budget_s` | 4, 8 ms | est |
| `brick_cache_bytes`, `brick_cache_bytes_touch`, `max_parallel_requests`, `max_bricks_per_file_read` | 96 MiB, 32 MiB, 4, 6 | est |
| `max_overview_frames`, `frames_chunk_records`, `frames_chunks_kept` | 20000, 4096, 8 | the shell's DRO sizes |
| `keyframes_cached`, `prefetch_keyframes`, `refresh_period_s` | 8, 2, 3 s | est |
| `field_sample_keyframes`, `attribute_sample_bricks` | 12, 24 | the shell's colour-bar rule |
| `min_surface_field_cos`, `layer_unlit_fraction` | 0.5, 0.35 | est |
| `fov_rad`, `view_azimuth_rad`, `view_elevation_rad`, `framing_margin`, `follow_view_height_m` | 30 deg, -55 deg, 32 deg, 1.15, 70 mm | render.yaml |
| `orbit_damping`, `follow_smoothing_s`, `near_fraction`, `far_scene_factor` | 0.15, 0.12 s, 0.002, 4 | est |
| `key_light_intensity`, `fill_light_intensity`, `key_light_azimuth_rad`, `key_light_elevation_rad`, `crease_angle_rad` | 2.0, 1.2, 35 deg, 50 deg, 35 deg | est; render.yaml |
| `material_srgb`, `tool_srgb`, `holder_srgb`, `machine_srgb`, `fixture_srgb`, `enclosure_srgb`, `enclosure_opacity`, `door_opacity` | render.yaml's | est; porter-duff-1984 |
| `cap_srgb`, `fixture_cap_srgb` | red, lilac | mcreynolds-blythe-2005 |
| `tool_sides`, `deflected_tool_srgb`, `deflected_tool_opacity` | 32, #eb6834, 0.55 | est |
| `deflection_exaggeration_default`, `min_...`, `max_...`, `deflection_window_s` | 1000, 1, 100 000, 1 s | est |
| `chip_aspect`, `chip_curl_rad`, `chip_default_length_m`, `min_chip_size_px`, `chip_default_srgb`, `min_pile_height_m` | 1:5:12, 150 deg, 1.2 mm, 1.5, grey, 0.1 mm | render.yaml; est |
| `standard_gravity_m_s2` | 9.80665 | cgpm-1901-standard-gravity |
| `jet_max_length_m`, `jet_samples`, `jet_opacity`, `jet_end_opacity_fraction`, `jet_sides` | 600 mm, 48, 0.45, 0.5, 12 | render.yaml; est |
| `nozzle_radius_m`, `nozzle_length_m` | 4 mm, 25 mm | est |
| `zone_radius_factor`, `min_zone_radius_m`, `tool_zone_height_ratio` | 1.5, 1 mm, 1 | coolant-occlusion.yaml |
| `max_shadow_miss_m`, `shadow_opacity` | 20 mm, 0.35 | est; porter-duff-1984 |
| `spray_drops_per_jet`, `spray_drop_size_m`, `spray_life_s`, `spray_speed_fraction` | 120, 0.8 mm, 0.3 s, 0.3 | est (visual cue) |
| `measure_srgb`, `measure_marker_px`, `min_circle_sine` | #6250d6, 8, 0.02 | est |
| `pick_tolerance_px`, `max_tap_s` | 6, 0.5 s | est |
| `pose_chain_tolerance_m`, `pose_chain_tolerance_rad` | 0.1 mm, 1 mrad | est |

RNG: none. Everything drawn "at random" is a counter-based hash (lowbias32, Wellons 2018) of a stream name and
integers: `view3d.chip_orientation` (chips without a velocity), `view3d.spray` (drops). The same time and data
always give the same picture.

## Decisions and their consequences

* **Draw in S** (as cnctwin.render): the part never moves, so removal, layers, sections and measurements are
  exact there; the machine moves around it. Consequence: objects in M need `pose_M_S` from the run; without it
  they are not drawn. Lathes opt into W0 with `stock.spin`.
* **The workpiece changes at keyframes** (every `keyframe_period_s`, 0.5 s by default): carving the swept tool
  between keyframes would be a second removal model. Consequence: at 1x the cut grows in half-second steps
  while the tool moves continuously.
* **Ranges by the shell's rules, colours by the shell's maps**: one colour bar for the part. Consequence: a
  range is the run's sampled minimum to maximum (12 keyframes, 24 bricks), not a percentile range; one outlier
  can flatten a map (the pictures use 1st-99th percentiles instead).
* **2-D maps colour upward faces only**: a top-face temperature or wetted map is not smeared down the walls.
  Consequence: walls show material colour with a 2-D layer.
* **Attributes only on final bricks**: keyframes/attr describes the finished part. Consequence: during playback
  the finish, deviation and wear layers colour a brick only once it is at its last version.
* **Chips and jets as keyframe snapshots**: exact at the keyframe, held between keyframes (head-mounted jets
  move with the tool). Consequence: chips in flight do not fly between keyframes.
* **Level of detail by vertex clustering**, fast and data-independent. Consequence: thin features (a wall
  thinner than a cluster cube) can vanish at a distance where they are under two pixels anyway; sections draw
  level 0 everywhere.
* **Flat shading**: the recorded facets are visible (the dexel staircase of a lattice) rather than smoothed
  over. Consequence: curved surfaces look faceted at the lattice resolution, as they are in the data.
* **Brick time rule of the API** (`time <= t + 1e-12` s) rather than KeyframeReader's tick rounding: a brick
  read alone is the version the index names. Consequence: within half a microsecond of a keyframe the two
  readers can disagree.
* **Settings in a model file**: every number is in db/models/view3d.yaml with its source; the browser reads a
  generated copy. Consequence: a change to the YAML needs `sync_params.py` (a test says so).
* **Deflected tool as a tip-loaded cantilever**: a shape from one recorded vector per path. Consequence: the
  real deflected shape (several modes, spread loads) differs by up to 4 % of the tip value for an evenly
  loaded tool; the tip is exact.

## Limits

* Per-path deflection: the tool displacement channel is used for path 0; other paths use the keyframes'
  `tool_deflection_m` rows.
* A rotating workpiece needs `scene.stock.spin`; multi-workpiece runs need `scene.workpieces[w].pose_S0_S`.
* Component meshes are drawn in kind colours (the GLB's own colours are not used).
* Chips are not advanced between keyframes; spray is a visual cue.
* Fixture and machine components are placed by their scene poses; a part that slips in its fixture (the
  keyframe field `part_pose`, holding) is not shown moving against the fixture: the workpiece is drawn in S, and
  the scene does not say in which frame a slip should move the fixture meshes.
* A 2-D field's plane height is not used (projection along its normal): a pile grid under the part also colours
  the part's top faces above it with the chips layer (only where the pile is above 0.1 mm).
* WebGL 2 is required (three.js r170 has no WebGL 1 path).
* Brick meshes closed each on their own (the synthetic run's boxes and carved columns) meet in coincident walls
  at brick boundaries, which show as thin seams on the faces they cross; the removal meshing writes bricks as
  pieces of one welded closed surface (docs/models/removal.md 10), which have no such walls.

## Tests

`tests/unit/view3d/` (run with `cd /opt/plint/apps/cnc-twin && TSAFE_MEM=1500M bin/tsafe python -m pytest -q tests/unit/view3d`):

* `test_view3d_unit.py` runs `view3d_unit.mjs` in node (182 checks) against a synthetic run read with the
  Python readers (`view3d_expect.py`): frames.bin fields and interpolation (rotary axes through 2 pi), brick
  records and keyframe GLBs against `cnctwin.core.keyframes`, bricks.idx versions with the API's time rule, the
  loader (versions per time, cache on scrubbing back), level of detail (cube bound, no degenerate triangles),
  poses against `cnctwin.geom.transform` (pose dicts in three orders, `perpendicular`), chains (linear, rotary
  about a line, fixed), stock motion (interpolation, chain validation both ways), .npz (float16, bool, unicode,
  int64, Fortran order, big-endian, stored and deflated), field grids (orientation rule, cell-centre sampling,
  exact trilinear on a linear field, no bleeding), float16 rounding and shortest digits as numpy, the
  circumcircle and point-to-plane, ballistic jets, the cantilever bend, hashing, colour lookups against the
  shell's maps, the api adapter, the parameters.
* `test_view3d_browser_desktop.py` (79 checks) and `test_view3d_browser_runs.py` (30 checks) run
  `browser_test.mjs` in `mcr.microsoft.com/playwright:v1.48.2-jammy` (`--memory 1g --cpus 1 --user 1002:1002
  --rm --network host`, docker reached through `sg docker` under bin/tsafe) against `serve_view3d.py`: the real
  API on port 5996 (else the next free port in 5990-5999) with three synthetic runs and the test page
  `harness/view3d-harness.html`. They check every contract call, brick versions at six times, playback, the tip
  against frames.bin, every layer's shader mode, picture change and colour range against the shell's
  `layerLegend`, section caps by colour, picks with normal and layer value, the three measurements (2.000 mm
  pocket depth to 1e-5 m, a 12 mm circle), follow and reset, visibility, exaggeration, coolant shadows, destroy,
  the default run, a running run through a minimal `{get, getBinary}` api, a copy of the rich run with a
  spinning stock (the view frame turns by the work spindle angle) and a component placed by the work chain with
  `stock.pose_F_S` (it moves by minus the X travel in S), a phone (pixel-ratio cap, tap to pick, 44 px buttons),
  and no console errors. Screens: `rich-*.png`, `default-*.png`, `partial-0.png`,
  `phone-0.png` in the test's output folder.
* `test_view3d_params.py`: the YAML is valid, params.js is current, shared values equal their sources.
* `test_view3d_vendor.py`: the vendored files are the release's (sha256), the examples differ only in their
  rewritten import, the view imports three.js only through `three.js`.
* `test_view3d_fixture.py`: the generator's `view3d` additions (below) are deterministic, also across a
  checkpoint and resume, change nothing when off, and are consistent (volumes, closed bricks, grids, jets,
  attributes).
* The shell's own suite (tests/ui/run_ui_tests.sh) passes with the real view in place (desktop 125, phone 42
  checks, every layer's view canvas rendered, no console errors).

By hand (the same steps the fixtures take):

```
cd /opt/plint/apps/cnc-twin
TSAFE_MEM=1200M bin/tsafe python tests/unit/view3d/serve_view3d.py --data /tmp/cnctwin-v3d/data --web /tmp/cnctwin-v3d/web --port 5996 --build
mkdir -p /tmp/cnctwin-v3d-node && (cd /tmp/cnctwin-v3d-node && echo '{"private": true}' > package.json && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm i --no-audit --no-fund playwright@1.48.2)
docker run --rm --network host --user 1002:1002 --memory 1g --cpus 1 -e HOME=/tmp -e SUITE=all \
  -v /tmp/cnctwin-v3d-node:/deps:ro -v $PWD/tests/unit/view3d:/tests:ro -v /tmp/cnctwin-v3d/out:/out \
  mcr.microsoft.com/playwright:v1.48.2-jammy node /tests/browser_test.mjs
node tests/unit/view3d/view3d_unit.mjs web <expect.json from view3d_expect.py>
```

`http://127.0.0.1:5996/view3d-harness.html?run=00002-v3d-rich&t=1.2` (add `&api=minimal` for the contract's
two-method api) shows the view on its own; the shell at `http://127.0.0.1:5996/#/runs/00002-v3d-rich/playback`.

The synthetic run generator (tests/fixtures/make_synthetic_run.py) has an additive `view3d=True` option
(`--view3d`): the pocket carved along the tool path as the IPW (closed brick meshes written with
cnctwin.core.keyframes' `brick_record` and `brick_glb`, only changed records indexed), final attributes on it,
`work_temp_K` and `work_displacement_m` 3-D grids, the chip pile and wetted cells on their own pile grid, and a
second, head-mounted low jet (`n2`, `attached_to: head`) with `jet_path_M_m`, `jet_diameter_m` and
`jet_blocked_by` (the uncut stock blocks it at one keyframe). Without the option the run is byte-identical to
before (tested).

## Integration

* **Shell**: already mounts the view through web/js/panels/viewhost.js; nothing to change. The view appends
  its canvas first, so the shell's `.pb-view canvas` locator finds it. `legend(layer)` returns the same range
  the shell computes.
* **scene.json writer (the loop)**: give every component `frame` (S, M, W, T, T0) and `pose`, or a `link` into
  `chain`; axes with `direction` (and `point_m` for rotary ones); `stock.pose_M_S` (static) or keyframe field
  `pose_M_S` when the stock moves in M, or `stock.pose_F_S` with a complete work chain (then M objects follow
  the axes exactly between keyframes); tools with `number`, `slot`, `envelope`, `holder_envelope`,
  `diameter_m`, `stickout_m` (or `length_m`); nozzles with `attached_to` for head-mounted ones; `gravity_M`; a
  `stock.spin {path, axis_S, point_S_m}` for turning; `workpieces[w].pose_S0_S` for more than one workpiece;
  `bricks` (or `stock.bricks`) `{cells, lattice_m, grid, origin_S_m}` so chunks follow the brick grid.
* **frames.bin**: a `tool_number` (or `tool_slot`) field per path lets the view switch tools without
  timeline.json.
* **Keyframe fields**: grids with `<name>_origin_S_m`/`<name>_cell_m`; 3-D grids `[nx, ny, nz]`; 2-D maps
  `[nx, ny]` with `<prefix>_axes_S` when not in x, y of S; vector grids with a last dimension of 3; jets with
  `jet_path_M_m` [n, P, 3] (NaN padded), `jet_diameter_m`, `jet_blocked_by` when the coolant model has them.
* **Attributes**: `keyframes/attr/brick_<b>.npz` per vertex of the brick's last record (`deviation_m`, `Ra_m`,
  `generating_VB_m`, or the `_DEV`/`_FINISH`/`_WEAR` names).
* **Dynamics**: a tick (or fast) channel `dynamics.tool_displacement_m` (T0, 2 or 3 components, m) drives the
  deflected tool at tick resolution.
* **Caddy / Plint theme**: the overlay uses Plint's `--p-*`-backed tokens (`--ink`, `--bg`, `--edge`,
  `--accent`, `--font`, `--mono`) with fallbacks; no theme file needed.

## Files

```
web/js/view3d/index.js      createView3D: the contract, frame update, layers, picking (orchestration)
web/js/view3d/three.js      the one import path of the vendored three.js and its example modules
web/js/view3d/params.js     settings in SI (generated from db/models/view3d.yaml)
web/js/view3d/apiio.js      api adapter, RunData (run record, files, scene, keyframes, bricks.idx, channels, timeline)
web/js/view3d/frames.js     frames.bin layout, decoding, FrameTrack (overview + chunks, interpolation)
web/js/view3d/bricks.js     brick records, keyframe GLBs, BrickIndex, BrickLoader
web/js/view3d/ipw.js        chunks, level of detail, stencil companions, brick ray casting
web/js/view3d/lod.js        vertex clustering, level choice
web/js/view3d/poses.js      poses, tool frames, chains, stock motion
web/js/view3d/scene3d.js    scene components, GLB baking, materials, placement
web/js/view3d/tool.js       tools, holders, deflected tool
web/js/view3d/chips.js      chips (instanced), piles
web/js/view3d/coolant.js    nozzles, jets, shadows, spray
web/js/view3d/fields.js     field grids, sampling, textures, colour-range helpers, float16 rounding
web/js/view3d/layers.js     layer definitions, colour lookups, the part shader
web/js/view3d/section.js    clipping plane and capped sections
web/js/view3d/measure.js    measuring
web/js/view3d/camera.js     camera, orbit controls, follow
web/js/view3d/overlay.js    colour bar, measure toolbar, labels, exaggeration control, status
web/js/view3d/npz.js        .npz / .npy reader
web/js/view3d/hash.js       counter-based hashing
web/vendor/three/           three.js r170: build/three.module.min.js, examples/jsm/{controls/OrbitControls,
                            loaders/GLTFLoader, utils/BufferGeometryUtils}.js, LICENSE (MIT)
db/models/view3d.yaml  db/bibliography/models-view3d.yaml
tests/unit/view3d/          tests, harness, serve_view3d.py, sync_params.py, view3d_expect.py
```

`web/vendor/three` is the npm registry tarball `three-0.170.0.tgz` (sha512
`FQK+LEpYc0fBD+J8g6oSEyyNzjp+Q7Ks1C568WWaoMRLW+TkNNWmenWeGgJjV105Gd+p/2ql1ZcjYvNiPZBhuQ==`, checked on download):
`build/three.module.min.js` (self-contained in r170) and `LICENSE` as they are; the three example modules with
their one `from 'three'` import rewritten to `from '../../../build/three.module.min.js'` (no import map is
needed, and there is one module instance); nothing else changed (test_view3d_vendor.py checks the hashes).

## References

Rossignac and Borrel 1993 (vertex clustering); McReynolds and Blythe 2005 (stencil capping); Moeller and
Trumbore 1997 (ray-triangle); Knutsson and Westin 1993 (normalised convolution); Timoshenko and Gere 1972
(cantilever deflection); CGPM 1901 (standard gravity); Shoemake 1985 (slerp); Foley et al. 1990
(supersampling); Porter and Duff 1984 (compositing); Wellons 2018 (lowbias32); Khronos glTF 2.0 and
KHR_mesh_quantization; three.js r170. Full entries in db/bibliography/models-view3d.yaml (and models-geom.yaml,
models-render.yaml).
