# Pictures and video (`cnctwin.render`)

A deterministic software rasteriser (Numba z-buffer, one thread, no GPU, no external renderer) and the pictures
and video of a run directory: the in-process workpiece, fixture, machine, the tool moved by `frames.bin`, chips,
coolant streams, chip piles, glow, sparks, smoke and steam, colour layers with colour bars, labels and section
views; PNG through Pillow and H.264 MP4 through FFmpeg, both bit-exact for the same inputs and build.

Package: `kernels.py` (Numba, explicit signatures), `reference.py` (plain-Python twins), `raster.py`
(`render(scene, camera, w, h, ssaa)`), `scene.py` (meshes, materials, sprites, labels, colour bars, section
planes), `camera.py`, `colour.py` (sRGB, colour maps, Planck and CIE), `cues.py` (visual cue model),
`layers.py`, `shapes.py`, `text.py`, `rundir.py` (run-directory readers, shared with `cnctwin.sound`),
`compose.py` (one picture of a run), `pipeline.py` (`render_run`, `render_images`, `render_video`,
`render_image`, `render_frame`), `output.py` (PNG, MP4), `__main__.py`. Parameters and their sources:
section 8.

```
python -m cnctwin.render RUN_DIR                       # images/*.png, video.mp4, render.json
python -m cnctwin.render RUN_DIR --at 1.2 --out a.png --camera follow --layer temperature
```

## 1. Rasteriser

**Pipeline** (`raster.render`), every pass at `ssaa` times the output size, box-filtered down at the end
(supersampling, Foley et al. 1990 14.10):

1. **Vertex shading** (NumPy, per triangle corner, so flat and Gouraud need no special case): base colour
   (material, per-vertex layer colour or per-face colour; sRGB -> linear), normal (flat: the face normal;
   Gouraud: crease-aware angle-weighted corner normals, `kernels.corner_normals`, faces within
   `crease_angle_rad` = 35 deg are smoothed, sharper edges stay sharp), and two linear colours:
   `ambient = base (k_a + k_f max(0, n.l_f)) + emission` and
   `direct = base k_k max(0, n.l_k) + k_s max(0, n.h)^e [n.l_k > 0]` (Lambert, Blinn 1977 half vector). The
   key and fill lights are fixed to the camera (key upper left, fill right), so every view is lit the same
   way. Gouraud shading interpolates these lit colours (Gouraud 1971); open surfaces (`two_sided`) are lit on
   the side facing the eye.
2. **Shadow map** (Williams 1978): depth of all shadow casters from the key light, orthographic,
   `shadow_map_size` = 1024 texels across the bounding sphere of the *focus* (follow camera: the cube of one view
   height around the tool tip on every side; other cameras: the fitted view), with a depth range reaching across
   the whole scene so tall casters outside the focus still shade it. Per fragment: percentage-closer filtering
   over 3 x 3 texels (Reeves et al. 1987) with a depth bias of 0.002 (light NDC) and the lookup point moved
   1.5 texels along the normal (no self-shadowing acne; the shadow edge shifts by up to that much).
3. **Opaque pass** (`kernels.raster_opaque`): clip space as OpenGL (`-w <= x, y, z <= w`); triangles are clipped
   against the near plane `z + w >= 0` in clip space (Sutherland & Hodgman 1974, attributes interpolated
   linearly there, which is exact); a pixel is covered when its centre `(px + 1/2, py + 1/2)` is inside, by edge
   functions (Pineda 1988) in float64 with the **top-left rule** (in the orientation used: an edge is top-left
   when `dy < 0` or `dy = 0, dx > 0`), so triangles sharing an edge never both cover and never both miss a
   pixel. Depth is z/w, interpolated with the screen barycentrics; attributes perspective-correctly (weights
   lambda_i / w_i). Depth test strictly nearer (`z < stored`): of two coincident surfaces the first drawn stays,
   decided by the face order, never by scheduling (Catmull 1974 z-buffer).
   **Section planes** `a x + b y + c z + d >= 0` discard fragments by their interpolated world position.
   Solids flagged `cap` show their back faces through a cut in the flat cap colour (red), tested `1e-7` (z/w)
   farther than they are so a coplanar front face (a vise standing on a table) always wins: caps show only
   through cuts, as in CAD section views. Section planes cut the part, the fixture and the chips; machine,
   tool and enclosure stay whole.
4. **Transparent pass** (`kernels.raster_blend`): enclosure walls, doors and coolant streams, triangles sorted
   back to front by the view depth of their centroid (stable sort, ties by index), depth-tested against the
   opaque surfaces (no depth write), blended `c = a src + (1 - a) dst` (Porter & Duff 1984). Not in shadow
   maps.
5. **Sprites** (`kernels.raster_sprites`): capsules around a screen segment (sparks are streaks, smoke and
   steam round puffs) with weight `(1 - (d/r)^2)^2`, depth-tested; over-blended ones far to near first, then the
   additive ones (glow).
6. Box filter, sRGB encoding (IEC 61966-2-1; rounding half up), then text with Pillow (DejaVu Sans from
   Matplotlib, outlined for contrast): title, time and NC block, tool label at the tool, the newest findings
   (four lines), colour bars with round ticks (1, 2, 2.5, 5 x 10^k).

Every kernel has an explicit signature (`cache=True, nogil=True, error_model="numpy"`), a plain-Python twin in
`reference.py` (an independent NumPy formulation) and a test comparing the two; one compiled specialisation each.

**Cameras** (`camera.py`): look-at view, perspective (`fov_y`) or orthographic (`view_height_m`;
`FrameOptions(projection="orthographic")` for any run picture, the follow camera keeping its view height); `fit` frames a box
from a direction with `framing_margin`; `follow` aims at the tool tip, smoothed over `follow_smoothing_s`
(0.25 s of frames.bin, a centred mean, so rapids do not shake the picture), from `follow_azimuth_rad`,
`follow_elevation_rad` at the distance where the view is `follow_view_height_m` (70 mm) high; near plane at 2 % of
the eye distance, far 20 view heights behind the tip.

### Error bounds of the rasteriser

| item | bound | how |
|---|---|---|
| coverage | exactly the pixel centres inside, ties by the top-left rule; count within perimeter/2 of the area | test against a brute-force point-in-triangle count and the analytic area |
| depth, interpolation | float64; perspective-correct attributes agree with the ray-plane intersection to 1e-5 (float32 colour buffer) | test |
| Gouraud vs per-pixel lighting | highlights inside large faces are lost (lighting is computed at the corners) | by construction |
| shadows | texel = 2 R_focus / 1024 (0.14 mm at a 70 mm view); edges shift <= 1.5 texels (normal offset), blur 3 texels | parameters |
| transparency | per-triangle sort: intersecting transparent triangles can blend in the wrong order inside the intersection | by construction (the enclosure and jets do not intersect each other in practice) |
| anti-aliasing | box filter over ssaa^2 samples (images 3 x 3, video 2 x 2) | parameters |

## 2. Pictures of a run (`compose.py`, `pipeline.py`)

### Inputs (the run directory, architecture 16)

| file | what is read |
|---|---|
| `run.json` | `project` (name, material id or family, coolant fluid, tools list with `comment`, `outputs.video`), `clock`, `seed` |
| `scene.json` | `frames` (frames.bin header), `stock` {`mesh` (S), `pose_W_S`, `pose_M_S`}, `machine.components[]` and `fixture[]` and `enclosure[]` {`id`, `mesh`, `frame` S/M/T, `pose` 4 x 4, `kind` machine/spindle/holder/fixture/enclosure/door/window, `opacity`}, `tools[]` {`number`, `envelope` (GLB in T: origin at the tip, +z_T toward the spindle), `holder_envelope`, `diameter_m`, `length_m`, `teeth`}, `nozzles[]`, `gravity_M` (default -z), `chain` (tool-chain axes: components linked to them default to frame T) |
| `scene/*.glb` | read with `cnctwin.geom.io.read_glb` (node transforms applied) |
| `frames.bin` | per L1 tick (memory-mapped; a record being written is ignored): `tip_S_m`, `axis_T_S`, `spindle_angle_rad` + `spindle_rev` (path 0), `tool_number` if the layout has it |
| `timeline.json` | the active tool number (blocks' `tool`), the block shown under the title, cutting time for the `cut.png` choice |
| `keyframes/index.jsonl`, `bricks.idx`, `kf_<k>.glb` | the in-process workpiece at keyframe k: per brick the newest BRK1 record at or before k (`<4sIIffff` header, uint16 cell positions, uint32 indices); a file whose records are not BRK1 is read as glTF nodes `brick_<b>` |
| `keyframes/attr/brick_<b>.npz` | `deviation_m`, `Ra_m`, `generating_VB_m` per vertex of the final brick mesh (attached when the vertex count matches) |
| `keyframes/kf_<k>.npz` | `pose_M_S` (4 x 4, or [paths, 4, 4]: path 0), `work_temp_K` + `work_temp_origin_S_m` + `work_temp_cell_m` (3-D) or `surface_temp_K` [ny, nx] + `surface_origin_S_m` + `surface_cell_m` (top face), `pile_height_m`, `wetted` (+ `wetted_origin_S_m`, `wetted_cell_m`, else the surface grid), `tool_deflection_m`, `jet_*` |
| `chips/kf_<k>.npz` | `pos_M_m`, `vel_M_m_s` (optional), `temp_K`, `mass_kg`, `length_m` (optional), `colour_rgb`, `pile_origin_S_m` + `pile_axes_S` + `pile_cell_m` + `pile_height_m` |
| `series/macro` | `thermal.tool_edge_max_K` (else `thermal.edge_temp_K`) and `wear.VB_m[i]` for the tool colour |
| `findings.json` | labels of the findings near the frame time |
| `db/materials/<id>.yaml`, `db/coolants/<id>.yaml` | family (appearance colour) and density; boiling and smoke points (read raw, no validation; missing: fall-backs) |

### Placement (everything is drawn in the stock frame S)

* The workpiece is in S (keyframes; before the first keyframe the stock mesh). It changes at keyframes
  (`recording.keyframe_period_s`, 0.5 s by default): between them the picture shows the newest keyframe at or
  before the frame (removal appears in steps of that period; the tool moves continuously).
* The tool: its envelope placed at `tip_S_m` with +z_T along `axis_T_S`, turned by the spindle angle, from the
  frames.bin record at or before the frame tick (zero-order hold, 1 ms). Components with `frame: T` move with it.
* Components with `frame: S` are placed by their `pose` in S.
* Objects in M (chips, coolant jets, components with `frame: M`) are mapped with `pose_S_M = inv(pose_M_S)`,
  from the newest keyframe at or before the frame, else `scene.stock.pose_M_S`. A run that gives neither has
  them left out and says so (`FrameInfo.notes`, the `notes` of render.json): the renderer never guesses
  kinematics. Between keyframes the pose is held (an M object is off by the table travel since the keyframe,
  at most `keyframe_period_s` x feed).

### What is drawn

* **Part, fixture, machine**: appearance colours by material family (`material_srgb`), fixture and machine colours;
  solids capped in section views. **Enclosure, doors**: transparent (`enclosure_opacity` 0.10, `door_opacity`
  0.22), two-sided, not cut, cast no shadows.
* **Chips**: one octahedron per particle, `(thickness, width, length) = length (1, 5, 12) / 12` (`chip_aspect`),
  length from `length_m`, else from the mass and the material density with that aspect, else
  `chip_default_length_m`; never smaller than `chip_min_size_px` on screen; length axis along the velocity when
  known, else a direction drawn by `hash_uniform(seed, "render.chip_orientation", i, axis, 0)`; the chip's own
  colour (`colour_rgb`, the temper colour of the chips model) and its glow.
* **Coolant streams**: one open cone per jet from `jet_origin` along `jet_direction`, radius at the nozzle from
  continuity `r0 = sqrt(Q / (pi v))`, widening with `jet_half_angle_rad`; it ends at the **first obstacle** the
  axis meets (ray cast with `cnctwin.geom.bvh` against the part, the S components and the tool; else
  `jet_max_length_m`), so a tool or a wall that shadows the cut visibly stops the stream (coolant shadowing).
  Opacity `jet_opacity`, fading to `jet_end_opacity_fraction` at the end.
* **Chip piles**: a height field over the occupied pile cells (> `pile_min_height_m`), standing on the surface
  found by casting rays down (-e_w) onto the part and fixture; corner heights are the means of the adjacent
  occupied cells.
* **Labels**: title, time and the block being executed, the tool (`T1 D10 4FL CARBIDE`), the findings raised in
  the last `finding_label_hold_s` of *video* time (2 s; simulated time = 2 s x speed for the video, 2 s for
  pictures; none on the setup and finished-part pictures).

### Colour layers (`layers.py`)

| layer | part (per vertex) | tool | display unit, map |
|---|---|---|---|
| temperature | `work_temp_K` trilinear, else `surface_temp_K` bilinear (NaN cells skipped) | edge temperature (macro series) | degC, plasma |
| chip_pile | material colour; the pile is drawn and coloured by height | - | mm, cividis |
| coolant_reach | wetted cell under the vertex (0 / 100 %) | - | %, Blues |
| surface_finish | `Ra_m` (final brick attributes) | - | um, viridis |
| deviation | `deviation_m` (final brick attributes), symmetric limits | - | um, RdBu_r |
| deflection | - | `tool_deflection_m` magnitude | um, YlOrRd |
| wear | `generating_VB_m` (VB of the edge that made the surface) | VB of the teeth (macro series) | um, YlOrBr |

Colour maps are Matplotlib's published 256-entry tables (viridis family: Smith & van der Walt 2015, cividis:
Nunez et al. 2018, ColorBrewer: Harrower & Brewer 2003), linearly interpolated, limits at the 1st and 99th
percentile (`range_percentiles`) so one outlier does not flatten a map. A video uses limits over the whole run
(`layers.run_range`) so frames compare. Vertices without a value keep the material colour.

### The standard set (`render_images`, `render_run`)

| file | time | camera | layer |
|---|---|---|---|
| `setup.png` | end of the first tool change | setup (part, fixture, tool) | - |
| `cut.png` | middle of the first block that cuts for most of its time | follow | - |
| `part_<op>.png`, `part_<op>_section.png` | end | part (section through the middle, normal +y) | - |
| `temperature.png` | keyframe with the hottest field | follow | temperature |
| `chip_pile.png` | keyframe with the highest pile | part | chip_pile |
| `coolant.png` | keyframe with the most jets | setup | coolant_reach |
| `deflection.png` | keyframe with the largest tool deflection | follow | deflection |
| `deviation.png`, `finish.png`, `wear.png` | end | part | deviation, surface_finish, wear |

Pictures without data (no keyframes with that field, no final attributes) are not made. `render.json`
(`format: cnctwin-render`) lists each picture's file, caption, time, tick, keyframe, camera (all its numbers),
layer, colour limits, counts and notes, and the video's frames, fps, speed, time span, size, layer and notes.
The setup sheet picks the picture whose name contains "setup"; the report embeds them all.

**Entry point for the end of a run**: `write_media(run_dir)` makes what the project's `outputs` ask for
(`images`, `video {fps, speed}`, `audio`): the pictures, the video and render.json here, `audio.wav` +
`audio.json` through `cnctwin.sound.render_audio`. Media run in their own process after the simulation
(architecture 10).

**Video** (`render_video`): frames at simulated times `t0 + i speed / fps` that start before t1 (so `[t0, t1)`
plays for `(t1 - t0) / speed` s), camera following the tool, `video.mp4` H.264 yuv420p `+faststart`; defaults
`video_speed` 10 and `video_frame_rate_Hz` 25, the project's `outputs.video {fps, speed}` take precedence.

## 3. Visual cue model (`cues.py`, `colour.py`; `db/models/render-cues.yaml`)

**Incandescence.** A body glows when it is hotter than the **Draper point**, 798 K (525 C, Draper 1847: all
solids start to glow dull red at the same temperature in the dark). Its colour is the chromaticity of a black
body at T:

* spectral radiance `B(l, T) = 2 h c^2 / l^5 / (exp(h c / (l k T)) - 1)` (Planck 1901; h, c, k exact, CODATA 2018),
* tristimulus `X, Y, Z = sum B(l) x(l), y(l), z(l) dl` over 380-780 nm in 5 nm steps (trapezoid) with the CIE
  1931 observer (CIE 015:2018) as fitted by Wyman, Sloan & Shirley (2013, multi-lobe piecewise Gaussians, table 1;
  mean squared error 3e-5 / 7e-6 / 2e-5, below the observers' own variance),
* linear sRGB by the IEC 61966-2-1 matrix, clipped at 0 and scaled to a maximum component of 1.

Checks: the computed radiance peaks at Wien's wavelength (`l_max T = 2.8978e-3 m K` within 0.1 %); 1000 K is red
(r > g > b), 6500 K nearly neutral. Its **drawn intensity** rises with the logarithm of the visible radiance Y,
from 0 at the Draper point to 1 at `glow_saturation_K` (1400 K) — a display mapping (perceived brightness is about
logarithmic in luminance), not a radiometric one: pictures are not exposure-calibrated. Glowing chips get this
emission on their faces and an additive **spark** streak of their velocity x `spark_exposure_s` (0.02 s, the
integration time of the eye or a camera), half-width `spark_radius_m` (0.35 mm); tool edges glow within one tool
radius of the tip when the edge temperature passes the point; part surfaces where their temperature does.

**Smoke and steam** need the coolant on (jets in the keyframe) and a hot spot above a threshold of the run's
fluid:

| fluid kind | smoke (oil) | steam (water) |
|---|---|---|
| emulsion, semi_synthetic | fluid `smoke_point_degC`, else `oil_smoke_point_K` 165 C (concentrate) | fluid `boiling_degC`, else 99.974 C |
| synthetic, solution | - | as above |
| straight_oil, mql | fluid `smoke_point_degC`, else 165 C | - |
| dry, air | - | - |

The boiling point default is the saturation temperature of water at 101.325 kPa (IAPWS-95, Wagner & Pruss 2002);
the oil smoke point default is the database estimate for a VG 22 mineral cutting oil (db/coolants/straight-oil.yaml,
about 20 K below its 186 C COC flash point). Hot spots are the hottest workpiece vertices (up to `max_hot_spots`)
and the tool tip; a spot is identified by the `hot_spot_cell_m` cell it lies in (2 mm,
`hash_u64(seed, "render.hot_spot", i, j, k)`), so its puffs stay put as the mesh changes between keyframes.
Density `rho = clip((T - T_threshold) / cue_full_scale_dK, 0, 1)` (60 K to full). Per spot
`ceil(rho puffs_per_source)` puffs; puff i has a phase `u = hash_uniform(seed, "render.smoke"|"render.steam", spot,
i, 0)`, age `((t / life + u) mod 1) life`, rises along -g at `smoke_rise_m_s` 0.25 / `steam_rise_m_s` 0.45 m/s
(inside the 0.1-0.5 m/s centre-line velocities of watt-level plumes a few cm above the source, Morton, Taylor &
Turner 1956), drifts sideways by `hash_uniform(..., 1|2)`, grows from 3 to 14 mm (steam 2 to 9 mm) and fades as
`min(1, age / (0.25 life)) (1 - age / life)^2` over 1.2 s (steam 0.5 s). The same time always draws the same
puffs and a puff moves continuously from frame to frame.

What the cues are not: smoke and steam are indicators of a threshold being crossed at a place, drawn with an
assumed plume; their amount is not a predicted aerosol mass. Spark streak lengths are a convention of the
exposure time. Thresholds are as good as the fluid data (`est` values in the coolant files).

RNG streams: `render.chip_orientation`, `render.hot_spot`, `render.smoke`, `render.steam` (all
`cnctwin.core.rng` counter-based hashes; nothing stateful).

## 4. Files and determinism (`output.py`)

* PNG: Pillow, RGB, no text chunks or time stamps, compression level 6: the same pixels give the same bytes
  (tested in the same process and in a fresh one).
* MP4: FFmpeg (imageio-ffmpeg's build, else `/usr/bin/ffmpeg`), libx264 `-preset medium -crf 18`, yuv420p,
  `-threads 1 -x264-params threads=1:lookahead-threads=1`, `-fflags +bitexact -flags +bitexact -map_metadata -1
  -movflags +faststart`, `SOURCE_DATE_EPOCH=315532800`; frames streamed to FFmpeg's stdin (memory does not grow
  with the length); written to a temporary name and renamed. The same frames give the same bytes with the same
  FFmpeg build (tested). Sizes must be even.
* Everything else is float64 NumPy and sequential Numba loops in a fixed order; no wall clock, no threads.

## 5. Performance (one core of this box)

Picture at 1280 x 720 with 3 x 3 supersampling: about 1.5-3 s; video frame at 960 x 540, 2 x 2: about 0.4-0.8 s
(the synthetic run: 50 frames at 192 x 108 in 6 s). Cost grows with pixels x overdraw (the shadow map adds one
1024^2 depth pass per frame) and linearly with triangles. Memory: colour buffer 12 bytes and depth 8 bytes per
supersample (1280 x 720 x 9: 160 MB at the image default; 960 x 540 x 4: 41 MB).

## 6. Decisions and their consequences

* **Draw in S.** The part never moves in its own frame, so removal, layers and sections are exact there; the
  machine moves around it. Consequence: objects in M need `pose_M_S` from the run (keyframe field or scene);
  without it they are not drawn (and the notes say so).
* **Gouraud per corner, lights fixed to the camera.** Flat and smooth shading share one pipeline and every view
  is lit alike. Consequence: no specular highlight inside a large face; lighting does not show where the shop
  light is.
* **IPW per keyframe.** The keyframes are what the run records; carving the swept tool between them would be a
  second removal model. Consequence: at speed-up 1 the cut grows in steps of `keyframe_period_s`.
* **Per-triangle transparency sort.** Cheap and deterministic. Consequence: intersecting transparent triangles
  can blend in the wrong order where they cross.
* **Display mapping of glow, plume conventions for smoke and steam.** The pictures show *that* and *where*
  thresholds are crossed; they are not radiometric or aerosol predictions.
* **Layer attributes only on final bricks.** `keyframes/attr` describes the finished part; a brick whose mesh
  changed later than the frame keeps its material colour on those layers.

## 7. Tests (`tests/unit/render/`)

`test_raster.py`: coverage of random triangles equals the pixel centres inside and the area within the perimeter
bound; shared edges and a fan cover every pixel once; z-buffer order and ties; back-face culling;
perspective-correct interpolation against the ray-plane intersection; near-plane clipping; every kernel against
its reference (opaque with planes, caps and shadows; depth; blend; sprites; corner normals); caps through a
section; shadows on a floor; blending; sprites hidden behind surfaces; one signature per kernel.
`test_colour_cues.py`: sRGB, colour maps (ends, monotonic lightness), ticks and robust limits, Wien's law, blackbody
hues, CIE fit peaks, glow from the Draper point, thresholds per coolant kind, deterministic continuous puffs,
sparks. `test_pipeline.py` (synthetic run): readers, PNG determinism (two renders and a fresh process), a
2-second video of 50 frames with identical bytes twice, video settings, the standard pictures and render.json,
layers. `test_scene_rules.py` (a run directory written by hand): coolant streams stop at the tool or part they hit,
M objects follow pose_M_S, the tool follows frames.bin, glowing chips spark, hot wet surfaces steam, runs without
pose_M_S leave M objects out and say so.

## 8. Parameters

`db/models/render.yaml` (appearance, sizes, cameras: settings, not physics) and `db/models/render-cues.yaml`
(visual cue model); sources: `db/bibliography/models-render.yaml`.

| file | parameters (SI) | source |
|---|---|---|
| render | supersampling 3 x 3 (images), 2 x 2 (video) | foley-1990 |
| | x264 crf 18, preset medium | ffmpeg-x264-2026 |
| | specular intensity 0.22, exponent 40 | blinn-1977 |
| | shadow map 1024 texels, bias 0.002, PCF radius 1 texel | williams-1978, reeves-1987 |
| | enclosure opacity 0.1, door 0.22 | porter-duff-1984 |
| | colour maps per layer | smith-vanderwalt-2015 |
| | sizes (1280 x 720 images, 960 x 540 video at 25 Hz and speed 10), lights (key 0.75 at 35 / 50 deg, fill 0.28, ambient 0.22), crease angle 35 deg, colours, jets (opacity 0.45, 600 mm, 12 sides), chip and pile display sizes, cameras (follow: 70 mm view, -55 / 32 deg, 30 deg field, 0.25 s smoothing; framing margin 1.15), labels and colour bars | est (display choices, ranges in the file) |
| render-cues | Draper point 798 K | draper-1847 |
| | water boiling point 99.974 degC at 101.325 kPa | wagner-pruss-2002 |
| | h, c (exact) | codata-2018 |
| | CIE 1931 observer 380-780 nm in 5 nm steps; multi-lobe fit | cie-015-2018, wyman-2013-cmf |
| | XYZ to linear sRGB matrix | iec-61966-2-1-1999 |
| | plume rise 0.25 m/s (smoke), 0.45 m/s (steam) | morton-taylor-turner-1956 |
| | glow saturation 1400 K, oil smoke point 165 degC, full-scale 60 K, spark exposure 0.02 s and radius 0.35 mm, puff sizes, lifetimes (1.2 / 0.5 s), opacities and colours, 10 puffs per source, hot-spot cell 2 mm, 12 hot spots | est |

## 9. Validity range

* **Sizes**: images 64 x 48 to 7680 x 4320, video 64 x 48 to 3840 x 2160 with even sizes (yuv420p), frame
  rates 1-120 Hz (the ranges of `render.yaml`); memory is 20 bytes per supersample (160 MB at the image
  default), so the largest sizes need `ssaa` 1 inside bin/tsafe's cap.
* **Geometry**: any triangle meshes up to the int32 index range; depth z/w in float64 between the near plane
  (2 % of the eye distance) and the far plane (20 view heights behind the tip for the follow camera); colours
  in float32 (interpolation agrees with the ray-plane intersection to 1e-5). Objects in M are drawn only when
  the run gives `pose_M_S`.
* **Incandescence**: the black-body colour is computed for any temperature; it is drawn above the Draper point
  (798 K) and its intensity saturates at `glow_saturation_K` (1400 K), so hotter bodies look alike. The CIE fit
  covers 380-780 nm (mean squared error 3e-5 / 7e-6 / 2e-5 against the 1931 observer); colours are checked
  against Wien's law within 0.1 % and the hues at 1000 K and 6500 K.
* **Smoke and steam**: with the coolant on and a hot spot above the fluid's threshold (smoke point, else
  165 degC; boiling point, else 99.974 degC), full density at 60 K above it; they mark that and where a
  threshold is crossed, valid as indicators only (no aerosol mass), as good as the fluid's threshold data.
* **Files**: byte-identical output for the same inputs on the same Pillow and FFmpeg builds (section 4); a
  different FFmpeg build gives other video bytes of the same frames.

## References

Catmull 1974; Pineda 1988; Sutherland & Hodgman 1974; Gouraud 1971; Blinn 1977; Williams 1978; Reeves, Salesin &
Cook 1987; Porter & Duff 1984; Foley, van Dam, Feiner & Hughes 1990; IEC 61966-2-1:1999; CIE 015:2018; Wyman,
Sloan & Shirley 2013; Planck 1901; CODATA 2018 (Tiesinga et al. 2021); Draper 1847; Wagner & Pruss 2002; Morton,
Taylor & Turner 1956; Smith & van der Walt 2015; Nunez, Anderton & Renslow 2018; Harrower & Brewer 2003; Matplotlib
3.11; FFmpeg/x264 (full entries in `db/bibliography/models-render.yaml`).
