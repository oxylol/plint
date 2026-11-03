# Coolant domain (`cnctwin/coolant/`)

The coolant system of the machine with the project's fluid: pumps and pipe networks, through-spindle
and through-tool supply, air blast, MQL and mist; the jets in the machine frame M; their occlusion by
the rotating tool and holder, the in-process workpiece, the machine and fixture meshes and the chip
pile; the heat-transfer coefficients and the boiling regime at every cut; the lubrication (friction
factor); and the tank (coolant owns it: thermal mass, heat in and out, chiller, evaporation,
concentration, consumption). The domain is an L2 (macro step) domain: `kernel_inputs` and `absorb`
return `None`; it writes only `nxt.coolant`.

| file | what |
|---|---|
| `fluid.py` | fluid properties at temperature (database tables), vapour, saturation under pressure, friction-factor lookup, air |
| `supply.py` | Darcy-Weisbach network, pump curves (radial, displacement + relief, given curve), nozzles, through-spindle path, compressible air, MQL, mist |
| `jets.py` | jets in M from mount frames, programmable nozzles, through-tool exits, break-up, drop size, ray sampling |
| `kernels.py` | Numba: `trace_rays` (trajectory + all in-kernel occluders + zones), `hit_meshes`, `film_map`, helpers (one signature each) |
| `reference.py` | plain-Python references of every kernel (brute-force geometry) |
| `occlusion.py` | job assembly, aggregation (reach, shadows, arrivals), film map, checkpointable result |
| `heat.py` | Martin 1977 impinging jets, boiling curve, aerosols |
| `lubrication.py` | friction factor with capillary access |
| `tank.py` | tank energy, evaporation, chiller, concentration |
| `kinematics.py` | link poses of the machine chains, nozzle mount frames |
| `protocols.py` | `ToolShapes`, `WorkpieceBricks`, `SceneMeshes`, `MountPoses` and complete default implementations |
| `state.py`, `schema.py`, `channels.py`, `domain.py`, `abi.py` | domain plumbing |

Model coefficients: `db/models/coolant-{fluid,supply,jet,heat,lubrication,tank,occlusion}.yaml`;
sources: `db/bibliography/models-coolant.yaml` (plus existing keys: incropera-2007-heat-mass-transfer,
white-2006-viscous-fluid-flow, pope-2000, churchill-chu-1975-vertical-plate, schlichting-gersten-2017,
nist-webbook-water-2026, wikipedia-leidenfrost-2026, dguv-209-025-mql-2010). **RNG streams: none**
(every sampling pattern is deterministic: sunflower rays, equally spaced spindle angles).

## 1. Macro step

1. **Switches** from `prev.motion.paths[*]`: `coolant_pumps` (pump id -> on; the keys `air` or
   `air_blast` switch the air blast), `nozzle_settings` (programmable nozzle id -> position, 1 = home).
   A programmable nozzle without a setting takes the operation's `nozzle_positions` entry for the tool
   in the spindle, else home.
2. **Supply** (section 2) for the configuration (pumps, air, chip-blocked nozzles, tool in the spindle,
   fluid temperature bucket of `supply_temp_step_dK`); the operating point is a pure function of that
   key and is cached in memory (recomputed identically after a resume).
3. **Jets** in M at the end-of-step pose (section 3).
4. **Occlusion jobs** (section 4) every `every_n_macro` steps and whenever the jet configuration or the
   set of cutting contacts changes; a job started at macro n is applied at n + `latency_macro`
   (architecture 5.1: k = 1), never earlier. Until the first job of a configuration is applied the
   reach is 0 (one macro step after M08 with the defaults). Coolant off drops pending and applied jobs.
5. **Per contact**: reach, h_rake / h_flank / h_work and the boiling regime (section 5), friction
   factor (section 6); h_work_far; wetted and flush maps (section 4.5).
6. **Tank** (section 7), consumption, **findings** (section 8), trace events, `nxt.coolant`.

## 2. Supply

### 2.1 Pipe friction and losses

Darcy-Weisbach per element: `dp = (f L/d + K) rho v^2 / 2`, `v = Q / (n pi d^2 / 4)` for `n` identical
parallel passages (coolant holes). Darcy factor:

* laminar, `Re <= laminar_re_hi = 2000`: `f = 64/Re` (Hagen-Poiseuille);
* turbulent, `Re >= turbulent_re_lo = 4000`: Colebrook-White (1939)
  `1/sqrt(f) = -2 log10(eps/(3.7 d) + 2.51/(Re sqrt(f)))`, Newton on `x = 1/sqrt(f)` from Haaland's
  (1983) explicit value, stopped at `|dx| <= solver_rel_tol x`;
* transition: linear in Re between the two end values (continuous and monotone in Q, so the network
  inverse exists; the intermittent flow there is uncertain by about +-30 % in f).

Minor losses on the smaller passage's velocity (Crane TP-410): sudden contraction
`K = 0.5 (1 - beta^2)`; rotary union, holder, line fittings as lumped K (estimates, `coolant-supply`).
Nozzles (converging orifices, no vena contracta): `dp = (1 - beta^4) rho (Q/(C_d A))^2 / 2` (velocity of
approach, ISO 5167 form) which includes the jet's kinetic energy; jet speed `Q/A`. Coolant holes end in
their own velocity head (`K = 1` at the exit).

### 2.2 Circuits

* **Flood** (pump kind `flood`): pump -> common line (`flood_line_*`, static head `static_head_m`) ->
  manifold -> one branch per open nozzle (segmented hose `flood_branch_*` + orifice of the nozzle's
  `diameter_mm`, `count` orifices in parallel).
* **Through-spindle / high-pressure / through-tool** pumps: pump -> line (`tsc_line_*`) -> rotary union
  -> drawbar bore -> holder passage (length = the holder's gauge length) -> contraction into the holes
  -> `count` holes of the tool (tooling `CoolantHoles`: diameter; length = overall length - exit height,
  divided by cos(helix) for helical tip/flute holes) -> exit. A tool without holes closes the branch
  (deadhead: the relief valve opens, the shaft power still heats the fluid) and a
  `coolant_shadowing` warning is raised. A pump of these kinds with no nozzle entry feeds the active
  tool's holes.
* **Pumps**: radial pumps `p(Q) = p0 [1 - (1 - p_r/p0)(Q/Q_r)^2]` through the database's rated point,
  `p0 = centrifugal_shutoff_ratio x p_r` (1.25, Guelich 2014), shaft power
  `P = P_r (r0 + (1 - r0) Q/Q_r)`, `P_r = p_r Q_r / eta` (`r0` = shut-off power ratio 0.5). Displacement
  pumps deliver `Q_r` while the system needs at most `p_r`; above, the relief valve holds `p_r` and the
  outlets take the flow that pressure drives (Karassik 2008), `P = p Q_pumped / eta`. A
  `pressure_flow_bar_l_min` curve of the machine file replaces either (interpolated; extrapolated with
  the last segment's slope, not below 0). All shaft power heats the fluid (pump losses, friction, jet
  impact, relief throttling); motor losses go to the air.
* **Network solution**: the manifold pressure by Brent's method on
  `p_pump(Q(p_m)) - dp_line(Q(p_m)) - rho g h_static - p_m = 0`, each branch flow `Q_i(p_m)` by Brent's
  method on its monotone `dp_i(Q) = p_m`; fixed tolerances, so the same inputs give the same iterations.

### 2.3 Air, MQL, mist

* **Orifice flow** of air at stagnation `p0`: `m = C_d A p0 / sqrt(R T0) psi(p_a/p0)`, choked
  `psi* = sqrt(gamma) (2/(gamma+1))^((gamma+1)/(2(gamma-1)))` below the critical ratio (Anderson 2003).
  When the orifices could pass more than the supply delivers (database `air_blast.flow_l_min` of free air,
  or the fluid's `typical_flow`), the nozzle pressure falls until they pass the supply flow (Brent).
* **Jet** at its fully expanded notional nozzle (Birch et al. 1984): `u_e = sqrt(2 c_p (T0 - T_e))`,
  `T_e = T0 (p_a/p0)^((gamma-1)/gamma)`, `d_e` from mass conservation. Choice: isentropic expansion to
  ambient (the expanded velocity is an upper bound; entrainment in the shock cells lowers it by
  ~10 %).
* **Air blast** issues from the machine's `air` nozzles, else from its flood nozzles (the "coolant/air
  blast switch" of the machine files). A project fluid of kind `air` turns every running pump's outlets
  into air jets.
* **MQL** (fluid kind `mql_oil`): every running pump's outlets (external nozzles, or the tool's holes
  for through-spindle pumps) emit an aerosol: `mql_air_m3_s` of free air per channel at the fluid's
  `min_air_pressure` (else `mql_air_pressure_Pa`), oil `dosage_default_ml_h` (else `mql_dosage_m3_s`)
  shared by the channels. Mist pumps (kind `mist`) atomise the pump's liquid flow in `mist_air_m3_s`.
  Aerosol droplets deposit with `1 - mist_loss_fraction` of the fluid file.

## 3. Jets in M

* **Mount frames** (`kinematics.py`): a nozzle on a link of a tool chain is written in the spindle-nose
  frame N of that chain (the machine files' notes: "head frame at spindle nose"; their geometry
  primitives use the literal link frames). N's origin is `tip_M + z_gauge axis_T_M` from the motion
  record (jets stay registered to the tool the hot loop moved), its axes those of the chain's last link
  at the step's axis values (forward kinematics of the machine chains: linear links translate along
  `direction`, rotary links rotate about it through `axis_point`). A nozzle on a workpiece-chain link is
  in that link's frame. Consequence: a machine file that gives tool-side nozzle positions in the literal
  head frame needs them rewritten relative to the spindle nose (or a MountPoses wiring that says so).
* **Programmable nozzles**: aim at position k = home aim rotated by `(k - 1) step` toward -z of the
  mount frame, about the axis perpendicular to the aim and z (M34 lowers the jet for longer tools).
* **Through-tool jets** leave `CoolantHoles.exit_pos_T_m` along `exit_dir_T` rotating with the
  spindle; the bus carries them at the end-of-step pose, occlusion samples `through_tool_angles` equally
  spaced spindle angles per revolution, each with `rays_through_tool` rays and 1/n of the flow.
* **Liquid jets**: spread half-angle `liquid_jet_half_angle_rad` (2 deg; nozzle `spread_deg` overrides),
  through-tool 3 deg. Intact length (Grant & Middleman 1966): turbulent nozzles (`Re >= 2300`)
  `L/d = 8.51 We^0.32`, laminar `L/d = 19.5 [We^0.5 (1 + 3 Oh)]^0.85`, `We = rho v^2 d / sigma`. Beyond it
  the jet is a spray of drops `1.89 d` (Rayleigh 1878; turbulent atomisation makes smaller drops, whose
  stronger drag is then underestimated).
* **Trajectory** (`kernels.trace_rays`): `dv/dt = g - k |v| v`; coherent column `k = 2 rho_a C_f /
  (rho_l d)` (air skin friction `C_f = 0.005`: < 0.2 m/s^2 for coolant jets), drops
  `k = 3/4 rho_a C_D / (rho_l d)` with Schiller-Naumann `C_D = 24/Re (1 + 0.15 Re^0.687)` (`C_D = 0.44`
  above Re 1000). Explicit midpoint steps; the step length keeps the chord within `sagitta_m` of the
  arc (`ds = sqrt(8 sagitta / kappa)`, between `step_min_m` and `step_max_m`), and a step ends exactly at
  break-up. The midpoint rule integrates constant gravity exactly (test: points on the parabola to
  1e-12), drag to second order.
* **Gas jets** are straight; centreline speed `u_c = u_0 min(1, B d_e / s)`, `B = 5.8`, spreading
  `r_1/2 = 0.094 s`, visible half-angle `atan(2 x 0.094)` = 10.6 deg (Pope 2000).
* **Dynamic pressure** along the axis (bus `Jet.path_q_Pa`): `1/2 rho v^2`; beyond break-up the liquid
  spreads over the widening spray cone, `q x (d / (d + 2 (s - L) tan(spray half-angle)))^2`.
* **Rays**: per jet one axis ray (weight 0; the bus `Jet.path_M_m`) and `rays_per_jet` rays on a
  sunflower pattern of the exit disc (Vogel 1979), ray k at `r_k = sqrt((k + 1/2)/n)` of the exit radius,
  azimuth `k x 137.5 deg`, tilted by `r_k x half-angle`, each carrying `Q/n` (top-hat jet).

## 4. Occlusion

### 4.1 Occluders (one Numba call per job, then one per static mesh)

| occluder | geometry | test |
|---|---|---|
| rotating tool + holder + spindle nose (per path) | tooling `Envelope` union -> radius table `R(z)` (`profile_dz_m`, part of the widest piece per interval) in T0 | exact quadratic `rho^2(t) = R(z(t))^2` per table interval |
| non-rotating tool body | oriented box: shank `h x b` along `shank_dir_T`, overhang + 2 max(h, b) | slabs |
| in-process workpiece | `WorkpieceBricks`: bricks empty / full / partial with `n_sub^3` sub-cells, in S | Amanatides & Woo (1987) 3-D DDA over bricks, nested DDA in partial bricks |
| chip pile and the surface under it | bus `chips.pile_*` height field in G (porous: rays stop at the chip top) | 2-D DDA over cells |
| machine and fixture meshes | `SceneMeshes`: trees in their link frames (default: the machine file's `geometry` primitives) | `cnctwin.geom.bvh.ray_first_hit_nb` per stored segment |
| enclosure | machine `enclosure.box_mm` | slabs (a ray leaving it is lost) |

A rotating tool is occluded by its swept envelope (a jet hitting the fluted part between teeth is
deflected by the next tooth within milliseconds). Through-tool rays ignore their own tool. The chip
pile's base coincides with the part or table top: a base hit wins only when clearly earlier than a hit
already found. Occluder names: `tool T3`, `shank of T3`, `neck of T3`, `holder of T3`, `spindle nose`,
`workpiece 0`, `chips`, `surface below (part, fixture or table)`, mesh names (`machine:table`).

### 4.2 Zones, aimed and arriving flow

* **Cutting zone** of a contact: a sphere at removal's engaged-region centroid `point_M_m`, radius
  `zone_radius_factor x max(R_tool, ap/2, ae/2, zone_min_radius_m)` (1.5): rays that wet the tool within
  about +-90 deg of the engagement, or the work within 1.5 R of the cut, count as reaching it (the
  rotating tool carries the fluid into the engagement within a quarter revolution). Every path's tool
  also gets a **tool zone** (centred `tool_zone_height_ratio R` above the tip) used until a new contact
  has its own traced zone.
* A ray is **aimed** at a zone when its unobstructed trajectory (followed past the first hit up to
  `s_aim = min(1.5 x max(distance to the zone + 2 r), aim_length_m)`) enters the sphere; it **arrives**
  when it enters before (or at) its first hit; otherwise its mass flow is **blocked** by the occluder it
  hit (`shadows`).
* **reach** = arriving / aimed mass flow (0 when nothing is aimed), times `reach_scale` (uncertainty
  studies), clipped to [0, 1]. With several nozzles, one well-aimed clear nozzle gives reach 1 however
  many others point elsewhere: reach measures shadowing of what is aimed; `aimed_flow_kg_s` and
  `arriving_flow_kg_s` carry the amounts.
* Per jet and zone the arrival statistics: arriving mass and volume flow, liquid carried, mean speed,
  distance and offset from the zone centre at the impact (the hit, when inside the sphere; else the
  closest approach), share beyond break-up.

### 4.3 Sampling error

Reach converges with the ray count: in the test geometries (clear nozzle, holder-shadowed nozzle) the
default 64 rays give `|reach(64) - reach(1024)| <= 0.02`; 32 rays up to 0.1. Coefficients change by
< 1 %. Envelope tables at 0.25 mm and 2 mm stock voxels add < 1 % of a jet diameter of position error.
The zone definition is a modelling choice (factor 1.5 +- 0.5 changes reach by up to +-0.2 for jets
grazing the zone): this, not sampling, dominates the reach uncertainty.

### 4.4 Latency and cadence

`every_n_macro` (default 5; the project template uses 1) sets the cadence; a configuration or contact-set
change starts a job at once. `latency_macro` (default 1, the contract value) delays its application.
Results in flight are part of the checkpoint (`CoolantState.pending`).

### 4.5 Film map (wetted cells, flush pressure)

On the chips pile grid (bus `pile_origin_S_m`, `pile_axes_S`, `pile_cell_m`, `pile_base_m`), or, when
chips publish none, on an own grid of `wetted_cell_m` over the workpiece footprint (axes like the pile
grid: `e_w` the S axis closest to up; surface = top of the material by a vertical brick ray per cell,
cached by the workpiece version). Impingements per jet: the flow-weighted centroid of the rays that hit
the part, fixture, table or chips, with their flow, mean impact speed and the arriving jet diameter
`sqrt(4 Q / (pi v))`; liquid intercepted by the tool or holder drains onto the surface below
(free-fall speed). `kernels.film_map`:

* **liquid**: flood fill from the impinged cell to 4-neighbours whose solid surface (under the chips: the
  film percolates through the porous bed) is at most `film_step_m` (1 mm) above the current cell: the
  film spreads over flat surfaces and drains downhill, never climbs walls. Filled cells are wetted; flush
  pressure `1/2 rho u_s(r)^2` with Watson's (1964) radial film: `u_s = U` inside
  `r0 = 0.1833 d Re^(1/3)`, `u_s = 27 c^2 Q^2 / (8 pi^4 nu (r^3 + l^3))` beyond (`c = 1.402`, `l` from the
  continuity of `u_s` at `r0`);
* **gas**: the wall jet follows the surface (|dw| <= step), `u_m = u_imp min(1, a d / r)` (Glauert 1956,
  `a = 1`) until 1 % of the impact speed; no wetting. **Mist** wets the spray footprint.
* The largest flush pressure of all impingements is kept per cell.

Approximation: the film's hydraulic jump and drainage off part edges are not resolved (the downhill
fill wets every connected flat cell, which is what flooded parts show); flush pressure beyond the jump
is overestimated (the slow layer there carries little dynamic pressure but Watson's decay already
makes it small: < 1 % of the impact value beyond ~10 r0).

## 5. Heat transfer

Per contact and arriving jet, Martin's (1977) single-round-nozzle correlation

    Nu = h D / k = Pr^0.42 G(r/D, H/D) F1(Re)
    G = (D/r)(1 - 1.1 D/r) / (1 + 0.1 (H/D - 6) D/r),   F1 = 2 Re^0.5 (1 + 0.005 Re^0.55)^0.5

with properties at the film temperature `(T_w + T_fluid)/2`; validity 2000 <= Re <= 4e5,
2.5 <= r/D <= 7.5, 2 <= H/D <= 12. The arriving part of a jet acts as a jet of diameter
`d sqrt(Q_arr / Q_jet)`; liquid jets at their impact speed with the diameter of continuity
`d0 sqrt(v0 / v)`.

* **Liquid jets** arrive undecayed: H/D is capped at the potential-core length B = 5.8 (coherent jet,
  and the spray of drops close behind it).
* **Gas jets** beyond H/D = 12 use the momentum-equivalent nozzle `D' = H/12` at H/D' = 12: the
  self-similar jet conserves `u D`, so Re is unchanged.
* r/D below 2.5 uses 2.5 (stagnation zone); above 7.5 G is extrapolated (~ D/r, the wall-jet decay).
  Re outside the range: the formula is extrapolated (`Re^0.5` laminar, `Re^0.775` high Re).
* **Targets**: `h_rake` and `h_flank` on the tool (averaging radius `max(2.5 d, R_tool)`; walls at the
  mean rake temperature `ToolHeat.rake_temp_mean_K` and at the body temperature at the in-cut elements
  `body_temp_at_elem_K`), `h_work` on the work (radius `max(2.5 d, offset + heat_zone_radius_factor x
  max(R_tool, ap/2, ae/2))`, wall `T_pre + 0.5 (T_peak - T_pre)` of thermal's contact preheat and
  generated-surface peak). The strongest jet sets each coefficient (no superposition: colliding wall
  jets do not add). Curvature of the tool is neglected (+-25 % on cylinders with D_target/d_jet > 2).
* Coefficients are "where the coolant arrives": thermal applies them over the reached share
  `reach_fraction` of the wetted surface.
* **Through-tool coolant** at a drill tip is the same correlation at H/D >= 2 (exits 0.5-1.5 mm above the
  hole bottom), radius `max(2.5 d_hole, R_drill)`: 1e4-2e5 W/m2K for 1 mm holes at 20-100 m/s.
* **h_work_far** (wetted surfaces away from the cut): the strongest jet that hits part / fixture / table /
  chips, Martin averaged out to `far_radius_fraction` (0.5) of the stock diagonal, at the work bulk
  temperature.

### 5.1 Boiling (liquids, `T_w > T_sat`)

The saturation temperature rises with the mean impingement pressure over the averaging disc,
`p = p_atm + rho v^2 d^2 / (4 r^2)`, by Clausius-Clapeyron through the file's boiling point and latent
heat (water at 10 bar: 452 K, steam tables 453 K); the Leidenfrost temperature shifts with it
(Bernardin & Mudawar 1999).

* **Nucleate** (Rohsenow 1952): `q_nb = mu_l h_fg sqrt(g (rho_l - rho_v)/sigma) (c_pl dT/(C_sf h_fg
  Pr_l^n))^3`, `C_sf = 0.0130`, n = 1 water-based, 1.7 others; superposed on the jet's forced
  convection `q = (q_fc^2 + q_nb^2)^(1/2)` (Kutateladze 1961).
* **Peak flux**: `q_max = 0.149 h_fg rho_v [sigma g (rho_l - rho_v)/rho_v^2]^(1/4)` (Zuber 1959,
  Lienhard & Dhir 1973) x `[1 + 0.1 (rho_l/rho_v)^0.75 c_pl dT_sub / h_fg]` (Ivey & Morris 1962) plus the
  forced-convection share `h_fc dT_sub`; the superheat where the nucleate curve reaches it by a
  safeguarded Newton iteration.
* **Film** above the database Leidenfrost temperature: tools (cylinders) Bromley (1950)
  `Nu_D = 0.62 [g (rho_l - rho_v) h'_fg D^3 / (nu_v k_v dT)]^(1/4)`, work (plates) Berenson (1961)
  `h = 0.425 [k_v^3 rho_v g (rho_l - rho_v) h'_fg / (mu_v dT L_c)]^(1/4)`, `h'_fg = h_fg + 0.8 c_pv dT`,
  forced film boiling `2.7 [V k_v rho_v h'_fg/(D dT)]^(1/2)` (Bromley, LeRoy & Robbers 1953) when larger;
  radiation across the film `h^(4/3) = h_c^(4/3) + h_r h^(1/3)` (emissivity 0.3), solved by fixed-point
  iteration. Vapour: water vapour at 1 atm (Incropera Table A.4) for water-based fluids, an ideal
  heavy-organic vapour (estimates) for oils, at `(T_w + T_sat)/2`.
* **Transition**: `log q` linear in `log dT` between the peak and the Leidenfrost point.
* The coefficient returned is `q / (T_w - T_fluid)` (thermal applies `h (T - T_fluid)`); the regime per
  target is on the bus (`boiling_regime`), `film_boiling` when the work or rake target film-boils.

Checks: Rohsenow and Zuber reproduce Incropera Example 10.1 (836 kW/m2 at 18 K for C_sf 0.0128, scaled
by the model's C_sf; 1.26 MW/m2 within 1.5 %); q is continuous at the Leidenfrost point; regimes come in
order.

### 5.2 Aerosols (MQL, mist)

Carrier-air Martin coefficient plus the deposited liquid flux
`m'' = rho_l Q_l (1 - mist_loss_fraction) / (pi r^2)`:
`h = h_air + m'' [c_pl (min(T_w, T_b) - T_f) + h_fg (T_b <= T_w < T_L)] / (T_w - T_f)`
(drops bounce above the Leidenfrost point).

### 5.3 Validity and error

| part | range | error |
|---|---|---|
| Martin, gas jets | 2000 <= Re <= 4e5, 2.5 <= r/D <= 7.5, 2 <= H/D <= 12 | about +-15-20 % (fit to Martin's data) |
| Martin, free-surface liquid jets (our use) | same | +-35 % (free-surface jets differ from submerged ones in the wall-jet region; Lienhard's reviews) |
| outside the range (extrapolated) | | +-50 % |
| Rohsenow | | +-100 % in q at given dT (+-25 % in dT at given q) (Incropera 10.4) |
| peak flux, subcooling, jets | | +-30 % (pool correlation plus forced share under a jet: conservative, jets raise it) |
| film boiling | | +-25 % (Bromley, Berenson); subcooled film boiling up to 2x higher (not modelled: conservative) |
| wall temperatures | thermal's means | the near-cut work wall fraction 0.5 +- 0.3 moves h_work by up to a factor 2 near the boiling transitions |

## 6. Lubrication

    f = clamp(1 - reach_lub (1 - f_DB) Phi, f_floor, 1)

* `f_DB`: the fluid's `friction_factor` for the tool class (tooling) and work family (material); when
  the pair is missing the mean of the same work family, else of the same tool class, else of all entries
  (the source is reported); air: 1.
* `Phi = phi / phi_ref`, the capillary access relative to the database conditions (Williams & Tabor
  1977; Godlevski et al. 1997): a chip element stays on the rake `t = l_c / v_chip`; Washburn (1921)
  filling of interface capillaries of radius `r_c = 1 um` gives `l_pen = sqrt(sigma r_c cos(theta) t /
  (2 mu))`, so `phi = min(1, sqrt(sigma r_c cos / (2 mu v_chip l_c)))`, `l_c = 2 h / r` (contact length 2
  chip thicknesses, Childs et al. 2000), `v_chip = v_c r` (chip ratio r from cutting's tables at the
  contact's mean h, `int_h_dt / in_cut_time_s`). Reference `v_chip = 1 m/s`, `l_c = 0.3 mm`.
* `reach_lub = reach` for liquids; aerosols also `x min(1, Q_deposited / (delta b v_chip))` with a
  boundary film `delta = 50 nm` and chip width b = engaged length.
* `f_floor = 0.2` (boundary lubrication).

Consequences: lubrication falls as `v_chip^(-1/2)` above the reference speed and saturates at full
penetration below; the access ratio cancels the fluid's own viscosity at the reference (its database
factor already contains its boundary chemistry). Error: the database factors are themselves estimates
(+-0.1); the speed law is a scaling argument (Williams & Tabor observe the trend, not this constant):
+-50 % on `1 - f` away from the reference speed.

## 7. Tank

    C dT/dt = P_thermal + P_chips + P_pump - Q_amb(T) - Q_evap(T) - Q_chill(T),    C = rho V c_p

* `P_thermal` = `prev.thermal.heat_to_coolant_J / dt`; `P_chips` = `prev.chips.heat_to_coolant_J` booked
  once per chips window (`booked_until_tick`); `P_pump` = shaft power.
* `Q_amb`: walls (Churchill & Chu 1975 vertical plate of the liquid depth) and free surface (Lloyd &
  Moran 1974 hot plate up, `0.27 Ra^(1/4)` cold surface up) with air at the film temperature, plus
  radiation `epsilon sigma (T^4 - T_a^4)` (0.9) of both. Geometry: free surface = database tank volume /
  `tank_depth_m` (0.35 m), square plan, liquid depth from the current volume.
* `Q_evap = m_evap h_fg(T)` for water-mixed fluids: the file's `evaporation_l_per_h_per_m2` scaled by
  Dalton's law `(p_s(T) - RH p_s(T_a)) / (p_s(T_ref) - RH p_s(T_a,ref))` (Magnus, Alduchov & Eskridge
  1996; reference 30/20 degC, RH 0.5); h_fg linear between the NIST values at 25 and 100 degC.
* `Q_chill = capacity x clamp((T - T_set)/band, 0, 1)` when `machine.coolant.chiller` gives
  `capacity_kW` and `setpoint_degC` (band 2 K); a chiller entry without them (spindle coolers) is
  reported and ignored for the tank.
* **Integration**: the losses are linearised at the step start and the linear equation integrated
  exactly; every loss term's integral follows from the same solution, so `C (T1 - T0) = (in - losses) dt`
  to round-off (residual on the bus). Against an RK45 integration of the nonlinear model over an 8 h
  shift (1.5 kW, 187 l): within 0.02 K (test).
* **Mass**: drag-out `drag_out_ml_per_kg_chips x` chip mass booked (chips `emitted_kg` increments) and mist
  `mist_loss_fraction x` liquid delivered leave as mix (concentration unchanged); evaporation leaves as
  water (concentration rises). The enthalpy leaving with the mass is booked (`mass_out_J`), so the
  ledger stays closed. No make-up during a run; `finalize` reports the water and mix to restore.
* MQL, air and dry: no tank; the fluid temperature is the shop air.
* Supply temperature at the outlets: `T_tank + P_shaft / (rho c_p Q_pumped)`.
* Between operations the tank idles for `environment.time_between_ops` (steps of `idle_step_s`).

Validity: free convection 1e4 <= Ra <= 1e11 (clamped below: the tank is then within 1-2 K of the shop);
error +-25 % on the losses (convection +-15 %, the tank geometry estimate +-20 %), evaporation +-50 %.
c_p is held at the start temperature (< 0.5 % change over 20-60 degC for water-mixed fluids).

**Consumption** per operation and per part (`finalize`): pump energy, pumped volume, drag-out, mist,
evaporated water, concentrate lost, MQL oil, free air, chip mass, coolant-on time.

## 8. Findings and trace

* `coolant_shadowing` (warning) when a cutting contact's reach falls below `shadow_warning_reach` (0.5):
  "only 12 % of the coolant aimed at the cut of T3 reaches it (blocked by holder of T3 81 %, ...)", with
  `blocked_frac[<occluder>]`; or when no jet is aimed at it: "no coolant jet is aimed at the cut of T3: jet
  n1 passes 34.0 mm from the cutting zone" (`nearest_miss_m`). Once per episode per tool slot, re-armed
  when the reach recovers above `shadow_recover_reach` (0.7) or at the next operation.
* `coolant_shadowing` (warning) when through-spindle coolant runs with a tool without coolant holes.
* Cause chain: finding -> `coolant:occlusion` (per zone: reach, aimed and arriving flow, blocked flow by
  occluder) -> `coolant:jets` (per job: reach of every jet) -> `coolant:supply` (pressures and flows,
  on every configuration change) and `coolant:nozzle_position` (programmable nozzle moves). Relief-valve
  and chip-blocked nozzle events hang off the supply event. A job records `coolant:jets` and a
  `coolant:occlusion` per zone (cutting contacts and tool zones) only when its result changed
  materially (a reach or blocked share by more than 0.01, the set of jets or occluders); otherwise the
  previous event stands for it, so steady machining adds no events (test).

## 9. Bus

**Reads** (`prev`): `motion.paths[*]` (`coolant_pumps`, `nozzle_settings`, `tool_slot`, `tip_M_m`,
`tip_S_m`, `axis_T_M`, `pose_M_T`, `pose_S_T`, `axis_names`, `axis_actual`, `spindle_omega_rad_s`,
`workpiece`, `tick`), `motion.tick`; `removal.contacts[*]` (`contact`, `tool_slot`, `workpiece`, `path`,
`geometry.in_cut/point_M_m/point_S_m/ap_m/ae_m/engaged_length_m`); `cutting.contacts[*]` (`in_cut_s`,
`in_cut_time_s`, `int_h_dt`), `cutting.coefficients[slot]` (`h_grid_m`, `chip_ratio`);
`thermal.heat_to_coolant_J`, `thermal.tools[slot]` (`rake_temp_mean_K`, `body_temp_at_elem_K`,
`body_temp_K`), `thermal.contact_surface_temp_K`, `contact_preheat_K`, `work_surface_temp_K`,
`work_preheat_K`, `work_bulk_mean_K`; `chips.blocked_nozzles`, `heat_to_coolant_J`, `booked_until_tick`,
`emitted_kg`, `pile_origin_S_m`, `pile_axes_S`, `pile_cell_m`, `pile_height_m`, `pile_base_m`.

**Writes** `nxt.coolant` (`CoolantFeedback`): `on`, `kind` (high_pressure > through_spindle >
through_tool > flood > mist > mql > air > dry), `reach_fraction`, `h_rake_W_m2K`, `h_flank_W_m2K`,
`h_work_W_m2K` (per contact), `h_work_far_W_m2K`, `friction_factor` (per contact; 1 when off or dry),
`film_boiling`, `fluid_temp_K` (supply temperature), `through_tool_exit_speed_m_s`,
`through_tool_flow_m3_s`, `jets` (`Jet`: id = nozzle id, `<nozzle>#k` per orifice or hole; flow at the
jet state, for gas the expanded volume flow so `sqrt(Q/(pi v))` is the notional radius), `shadows`
(occluder -> fraction of the aimed flow over the cutting contacts), `wetted_origin_S_m`, `wetted_cell_m`,
`wetted` and `flush_pressure_Pa` (on the pile grid; same shape as `chips.pile_height_m`).

Additive fields (bus.py, commented): `Jet.diameter_m`, `breakup_m`, `path_M_m`, `path_q_Pa`,
`blocked_by`, `pump_id`, `through_tool`; `CoolantFeedback.wetted_axes_S`, `aimed_flow_kg_s`,
`arriving_flow_kg_s`, `contact_shadows`, `boiling_regime`, `lubricant_access`, `tank_temp_K`,
`tank_volume_m3`, `concentration_frac`, `pump_flow_m3_s`, `pump_pressure_Pa`, `pump_power_W`,
`nozzle_flow_m3_s`, `air_flow_m3_s`, `mql_oil_m3_s`, `heat_in_J`, `heat_out_J`, `tank_residual_J`,
`occlusion_tick`.

Note for consumers: thermal reads `wetted` as a grid over S x-y from `wetted_origin_S_m[0:2]`; that is
the grid's frame when its axes are (x_S, y_S, z_S) (`wetted_axes_S` = identity: mills with S z up). For
other orientations use `wetted_axes_S` (columns e_u, e_v, e_w in S).

## 10. Settings (`physics.coolant`, `schema.py`)

`every_n_macro` (5), `latency_macro` (1), `rays_per_jet` (64), `rays_through_tool` (8),
`through_tool_angles` (4), `step_max_mm` (10), `step_min_mm` (0.5), `sagitta_mm` (0.05),
`trace_length_mm` (1500), `aim_length_mm` (1000), `max_points` (128), `profile_dz_mm` (0.25),
`profile_nodes` (2048), `brick_voxel_mm` (2), `brick_sub` (4), `wetted_cell_mm` (2),
`supply_temp_step_dK` (0.5), `solver_rel_tol` (1e-12), `solver_max_iter` (200), `reach_scale` (1),
`idle_step_s` (60).

## 11. Kernels and determinism

All kernels: `@njit(<explicit signature>, cache=True, nogil=True, error_model="numpy")`, writable
C-contiguous arrays and NamedTuples of them, scratch buffers allocated once per call, no fastmath, no
parallel; `len(signatures) == 1` is tested. References (`reference.py`) use brute-force geometry
(dense sampling + bisection for envelopes, every voxel / cell / triangle as a box or triangle) and the
same trajectory rule: the tests agree to 1e-9 m in hit positions and identical hit kinds. Iterative
solvers (Colebrook Newton, Brent, peak-superheat Newton, radiation fixed point) have fixed tolerances
and deterministic starts. No set or dict iteration without sorting. Checkpoint: `CoolantState.to_arrays`
(tank, applied and pending occlusion results, cursors, totals, flags, trace ids); resume continues bit
for bit (test).

## 12. Cost (measured, one core of the i5-4690, bin/tsafe, CPU time)

| part | cost |
|---|---|
| macro step without a job (supply cached, one contact) | ~0.5 ms |
| occlusion job, 1 nozzle + 2 zones, 64 rays (trace 1.3 ms, aggregation and film map ~1 ms) | ~2.5 ms |
| defaults (`every_n_macro` 5) | 1.0-1.1 ms per macro step = 0.10-0.11 s per simulated second |
| `every_n_macro` 1 (project template) | 2.5-3.0 ms per macro step |
| supply solve on a configuration change | 0.5 ms flood, 6-7 ms through-spindle with relief (Brent in Python; cached per configuration and temperature bucket) |
| fallback film grid (no pile grid), 50 x 40 cells | ~10 ms once per workpiece version |
| kernel compilation (cold cache) | ~20 s once per cache key |

Cost grows linearly with jets x rays x segments to the first hit (~1.3 us per segment).

## 13. Limitations

* Coolant sticking to a rotating tool is carried into the cut when it hits the fluted part inside the
  zone sphere; centrifugal flinging at high speed and the air boundary layer around fast tools (which
  matters for MQL and air at surface speeds comparable to the jet) are not modelled.
* The cut itself is a sphere; the jet's penetration into the tool-chip gap is the lubrication access
  model only.
* Secondary flows (splash from the tool, fluid running down walls into pockets) are represented only by
  the drain film below tool hits.
* Turning: the rotating workpiece should be wired as its revolved occupancy (WorkpieceBricks); the
  default for cylinder stock is axisymmetric.
* Flash and smoke points of oils are not checked (no fire or smoke hazard finding).
* Coolant filtration and contamination, foaming and the concentration's effect on the fluid properties
  during a run (properties stay those of the file) are not modelled.

## 14. References

Martin 1977; Colebrook 1939; Moody 1944; Haaland 1983; Crane TP-410 (2013); Idelchik 2008; Guelich
2014; Karassik et al. 2008; Anderson 2003; Birch et al. 1984; Grant & Middleman 1966; Rayleigh 1878;
Lefebvre 1989; Schiller & Naumann 1933; Watson 1964; Liu & Lienhard 1993; Glauert 1956; Pope 2000;
Rohsenow 1952; Zuber 1959; Lienhard & Dhir 1973; Ivey & Morris 1962; Kutateladze 1961; Bromley 1950;
Bromley, LeRoy & Robbers 1953; Berenson 1961; Bernardin & Mudawar 1999; Williams & Tabor 1977;
Godlevski et al. 1997; Washburn 1921; Childs et al. 2000; Churchill & Chu 1975; Lloyd & Moran 1974;
Incropera et al. 2007; White 2006; Alduchov & Eskridge 1996; Amanatides & Woo 1987; Vogel 1979;
Schlichting & Gersten 2017; NIST Chemistry WebBook (water); DGUV Information 209-025 (MQL). Full
entries: `db/bibliography/models-coolant.yaml` and the files named above.
