# Structure: tool-holder-spindle modes, workpiece modes and compliance, displacement grids, fixture compliance

Package `cnctwin/structure/` (domain `structure`, an L2 slow domain), tests `tests/unit/structure/`, coefficients
`db/models/structure-tool.yaml`, `db/models/structure-work.yaml`, sources `db/bibliography/models-structure.yaml`
(plus `schmitz-smith-2009-machining-dynamics`, `kops-vo-1990`, `young-budynas-2012-roark` from
`models-tooling.yaml`, `budynas-nisbett-2015-shigley` from `models-holding.yaml`, `johnson-1985-contact-mechanics`
from `models-cutting.yaml`).

| module | content |
|---|---|
| `beam.py` | Timoshenko beam elements (12 DOF), Cowper shear coefficients, Winkler foundation matrices, assembly |
| `rcsa.py` | receptance coupling (FE substructures, modal substructures, rigid and elastic joints) |
| `toolmodel.py` | tool + holder + clamping interface + spindle base -> ToolModes; static deflection and stress helpers |
| `hexfe.py` | the voxel element: 8-node hexahedron with incompatible modes |
| `voxel.py` | voxel grid rule, assembly, supports, systems, eigen solves, Rayleigh-Ritz refresh |
| `multigrid.py` | geometric multigrid V-cycle, (block) preconditioned CG, direct/iterative solver choice |
| `thinwall.py` | thin-wall detection, Ritz Kirchhoff plate, blend weights |
| `lathe.py` | long turned parts as beams on chuck / tailstock / steady supports |
| `displacement.py` | temperatures on the grid, thermal strain, residual-stress profiles, Treuting-Read relation |
| `work.py` | one workpiece refresh job (modes, compliance, displacement grids) |
| `fixture.py` | fixture compliance from holding's support stiffness |
| `modal.py` | deterministic modal algebra (normalisation, canonical clusters, Ritz) |
| `kernels.py`, `reference.py`, `abi.py` | compiled kernels, plain-Python references, layouts |
| `protocols.py` | `WorkGeometry` (removal's query), `BoxGeometry`, `CylinderGeometry`, `FillSnapshot` |
| `materials.py`, `params.py` | elastic constants (work, tool substrate, holder steel), model coefficients |
| `state.py`, `domain.py`, `channels.py`, `schema.py` | domain plumbing |

## 1. Bus record

Reads `bus_prev` only:

* `holding.tools[slot].pullout_m` (the tool walks out: longer stick-out, shorter clamped length);
* `holding.parts[w]`: `supports` (per-node `pos_S_m`, `normal_S`, `t1_S`, `stiffness_N_per_m`,
  `tangential_stiffness_N_per_m`, `preload_S_N`, `state`, `law`), `supports_version`, `free`,
  `support_stiffness_S`, `centroid_S_m`, `lathe` (chuck / tailstock / steady along S z);
* `wear.tools[slot]`: `broken`, `length_loss_m` (the tool below the fracture is gone);
* `thermal.work_temp_origin_S_m[w]`, `work_temp_cell_m[w]`, `work_temp_K[w]` (cell centres, NaN outside material);
* `motion.paths[*]`: `tool_slot`, `pose_M_T` (orientation of a non-rotating tool's block in M).

Geometry through `protocols.WorkGeometry` (removal's `views.FieldGeometry`: `stock_bounds_S`, `fill_fraction`,
optionally `wall_thickness`). Tool geometry from `state.tooling.geometry(slot)` (tooling's `ToolGeometry`).

Writes only `nxt.structure` (`StructureFeedback`):

* `tools[slot]` (`ToolModes`): `freq_Hz`, `zeta`, `shape [mode, station, 6]` (T0; translations, then right-handed
  small rotations; mass-normalised), `station_z_m`, `static_residual_m_per_N [3 n_st, 3 n_st]` (translations),
  `sections_z_m`, `sections_modulus_m3` (tooling's critical sections, ascending z). **Additive**:
  `tip_compliance_m_per_N [3, 3]` (static, modes + residual), `fit_error` (section 2.6), `version`.
* `work_modes[w]` (`WorkModes`): `freq_Hz`, `zeta`, `origin_S_m`, `cell_m`, `shape [mode, nx, ny, nz, 3]`,
  `static_compliance [nx, ny, nz, 3, 3]` (residual) at **cell centres** `origin + (i + 1/2) cell` (dynamics'
  WorkGrid convention). **Additive**: `n_wall_modes` (the last n modes are thin-wall plate modes), `model`
  (`voxel` | `beam`), `version`.
* `displacement[w]` (`Displacement`): `total_m = thermal_m + clamp_m + residual_m`, each `[nx+1, ny+1, nz+1, 3]`
  at **nodes** `origin + (i, j, k) cell` (removal's sampling convention; trilinear between, clamped at the
  border). **Additive**: `clamped` (False after unclamping), `version`.
* `fixture_compliance_m_per_N[w]` (3 x 3 at the part datum, section 6). **Additive**: `fixture_point_S_m[w]`,
  `fixture_compliance_6_m_per_N[w]` (6 x 6 about `fixture_centroid_S_m[w]`), `fixture_centroid_S_m[w]`.
* `version` increments whenever any published model changes.

Kernel inputs (`kernel_inputs`): `abi.StructureInputs(origin_S_m [n_w, 3], cell_m [n_w], total_m (tuple of node
grids))` for the hot loop's `X_e = x_e - d(x_e)` (section 8). No accumulators (`absorb` does nothing).

## 2. Tool-holder-spindle model (`toolmodel.py`, `beam.py`, `rcsa.py`)

### 2.1 Geometry (rotating tools)

Coordinates along +z_T from the tip (tooling's layout). The tool beam runs from `length_loss` to the overall
length: fluted section to `z_flute_end`, neck to `z_neck_end`, shank above. The fluted section takes tooling's
`props.fluted`: the sector model of N teeth on the core when core and gullet are known, else Kops & Vo's
equivalent diameter 0.8 D (Kops & Vo 1990) with the sector model's anisotropy; the model uses the **mean of the
principal values** `(I_min + I_max)/2` in both planes of T0 (the tool rotates: in the non-rotating frame the
time-invariant stiffness is the mean, the deviatoric part varies at 2 Omega; Genta 2005), area = the fluted metal
area, scaled by `(r_env/R)^4` (I) and `(r_env/R)^2` (A) where the envelope radius is below R (ball, chamfer
tips). Tools without a fluted section (indexable, disc heads): solid circle of the envelope radius.
The holder beam runs from its nose `z_nose = stickout + pullout` to the gauge line `z_nose + gauge_length`, outline
from `profile_mm` (diameter interpolated linearly in z), bored to the shank diameter where the shank is inside,
solid above (holder steel, `holder_*`, Budynas & Nisbett Table A-5). Nodes: every breakpoint (flute end, neck end,
nose, clamp end, profile rows, shank end, gauge line), intervals split into elements <= `tool_element_m` (2 mm).

### 2.2 Timoshenko elements

Per bending plane, DOFs (w1, theta1, w2, theta2) with theta = dw/dz (x-z: (ux, ry); y-z: (uy, -rx)),
`Phi = 12 E I / (k G A L^2)` (Przemieniecki 1968):

    K = E I / ((1 + Phi) L^3) [[12, 6L, -12, 6L], [6L, (4+Phi)L^2, -6L, (2-Phi)L^2], ...]

consistent mass with translational and rotary inertia including Phi (Przemieniecki ch. 11); axial and torsional
bars with linear shape functions. Shear coefficients (Cowper 1966): solid circle `6(1+nu)/(7+6nu)`, hollow circle
with `m = r_i/r_o`: `6(1+nu)(1+m^2)^2/((7+6nu)(1+m^2)^2+(20+12nu)m^2)`, rectangle `10(1+nu)/(12+11nu)`.
Exact for end loads (one element reproduces `P L^3/3EI + P L/kGA`).

### 2.3 Clamping interface

Over the clamped length `L_c = min(holder clamp_length, inserted length)` (or `clamp_length_ratio` x d when the
holder gives none) the shank nodes and the coincident bore nodes are joined by a distributed contact
(Winkler) foundation acting on their relative DOFs. With normal contact stiffness per area `k_n` (per holder
family, `contact_k_per_cm2_<family>`, est) and tangential `k_t = 2(1-nu)/(2-nu) k_n` (Mindlin; Johnson 1985):

    lateral   k'_lat = (pi d / 2)(k_n + k_t)      (Hermite consistent matrix k' L/420 [[156, 22L, 54, -13L], ...])
    axial     k'_ax  = pi d k_t
    torsion   k'_tor = pi d (d/2)^2 k_t
    rotation  k'_rot = pi (d/2)^3 k_t             (the section's rotation moves its surface axially by r cos(phi) theta)

Rectangular turning shanks (b x h): `k'_x = 2(h k_n + b k_t)`, `k'_y = 2(b k_n + h k_t)`, rotations
`k_t (h b^2/2 + b^3/6)` and `k_t (b h^2/2 + h^3/6)`. Holder types map to families (`params.HOLDER_FAMILY`):
shrink_fit, hydraulic, milling_chuck, er_collet, side_lock (Weldon), tapping (floating/synchro), face (arbor,
modular, boring head), turret (VDI/BMT/lathe holders); unknown types use the collet values (most compliant).

### 2.4 Spindle-holder base and coupling (RCSA)

The machine's `spindles[].frf_modes` are flange modes (fn, zeta, k, direction in the spindle-nose frame = T0).
Each is a modal coordinate q_j of unit mass, stiffness `omega_j^2`, viscous damping `2 zeta_j omega_j`; its
flange shape is

    psi_j = [ d_j , -(e_z x d_j) / a ] / sqrt(m_j),    m_j = k_j / omega_j^2

a translation with the rocking rotation about a pivot `a = spindle_pivot_length` (0.3 m, est) behind the gauge
line (the machine files give translation only; a single-shape base receptance gives `P = N H^-1 L`, the
assumption of Schmitz & Duncan 2005). The gauge-line node is rigidly tied to the base, `u_gauge = Psi q`
(holder-less tools: a rigid block over the clamped length tied to the base). Without frf_modes the gauge line is
fixed. The tied system is the **receptance coupling** of the free tool-holder FE receptances H with the base
receptances `G_b = Psi diag(1/(omega_j^2 - omega^2 + 2i zeta_j omega_j omega)) Psi^T` (rigid joint,
`G = H_ss - H_sc (H_cc + G_b)^-1 H_cs`; Schmitz & Smith 2009 ch. 8, Erturk, Ozguven & Budak 2006) written as one
finite system (`T^T K T + diag(omega_j^2)`, `T^T M T + I`), so its modes are the poles and residues of the
coupled FRF exactly. `rcsa.couple_fe_to_modal_base` evaluates the coupled FRF by RCSA directly (test: equal to
the tied system to 1e-8; two beams coupled by RCSA equal the monolithic FE to 1e-8).

Non-rotating tools: the bar (circle) or shank (rectangle, height along y_T, width along the other axis) runs along
tooling's `shank_dir_T` from the tip to `overhang`, clamped over `bar_clamp_length_ratio` d or
`shank_clamp_length_ratio` h by the turret-family contact to a rigid tool block; the block moves with the
machine's `structure.turret_modes` (pure translations along their M directions turned into T0 with the active
path's `pose_M_T`; identity until the tool is first active). Bar material carbide uses the tool substrate,
`damped` bars the loss factor `damped_bar_loss_factor`. Stations: the tip only (`station_z_m = [0]`).

### 2.5 Modes, damping, residual flexibility, stations

Dense symmetric eigen solve of the tied system (scipy `eigh`), canonical clusters and sign normalisation
(section 7), modes up to `tool_max_freq_Hz` (20 kHz), at most `tool_max_modes` (16). Damping by the modal strain
energy of the hysteretic parts plus the base's viscous damping (Johnson & Kienholz 1982):

    zeta_k = sum_parts (eta_p / 2) phi_k^T K_p phi_k / omega_k^2 + sum_j zeta_j omega_j q_kj^2 / omega_k

with `tool_loss_factor`, `holder_loss_factor` (0.002) and the interface's `contact_loss_factor_<family>`
(0.1-0.3, est). Stations: tool nodes from the tip to the holder nose, then holder nodes to the gauge line (a
continuous axis line), at most `tool_stations` (24) including tip, flute end, neck end, nose and gauge line;
shapes `[mode, station, 6]` in T0. Static compliance at the station translations by a direct solve; residual
`R = C_static - sum_k phi_k phi_k^T / omega_k^2` (symmetric, positive semi-definite).

### 2.6 Fit error

`fit_error`: the largest difference, relative to max |G|, between the modal tip receptance
`sum phi^2 / (omega_k^2 - omega^2 + 2i zeta_k omega_k omega) + R` and the RCSA receptance (hysteretic parts as
complex stiffness `K (1 + i eta)`, viscous base): |G| at omega_k, |G| and Re G at `omega_k (1 -+ zeta_k)` for every
mode carrying >= 1 % of the static tip compliance, |G| at 5 % of the first mode. 2.7 % for the D10 end mill in a
shrink fit on a 950 Hz spindle: the damped peaks of the non-proportionally damped system sit 3e-4 above the
undamped frequencies.

### 2.7 Refresh and helpers

At init for every slot; again at once (a few hundred ms) when holding's pull-out changes by `pullout_refresh_m`
(20 um), when wear reports a broken tool (the tool starts at `length_loss_m`), when a non-rotating tool's block
orientation changes by 1e-6. Helpers for wear, breakage and the report: `static_deflection(res, F, station)`
(exact static compliance of the model at the stations), `section_stresses(sections, F, z_F, T)` (beam statics:
`M = |F_lat|(z_s - z_F)`, `sigma = M/Z_b + |F_z|/A`, `tau = T/Z_p`, `sigma_vM = sqrt(sigma^2 + 3 tau^2)`),
`tip_frf(res, omega)`.

## 3. Workpiece voxel model (`voxel.py`, `hexfe.py`, `multigrid.py`)

### 3.1 Grid and cells

Cubic grid over the stock box in S (origin at its lower corner): the smallest cell >= `min_cell_m` (1 mm) with
`3 (nx+1)(ny+1)(nz+1) <= max_dof` (50 000) for the full box, so the model never exceeds it (bisection, 80 steps).
A cell is in the model when removal's material fraction exceeds `fill_threshold` (0.5); cells not
face-connected to the cells holding support points (or to the largest component) are dropped (slivers the grid
cannot resolve; their count is in the job statistics). The grid is fixed for the run (displacement grids stay
co-registered); the model loses cells as material is removed.

### 3.2 Element

8-node cube with Wilson's incompatible modes `(1 - xi^2, 1 - eta^2, 1 - zeta^2)` per component, condensed
(`K = K_uu - K_ua K_aa^-1 K_au`; Taylor, Beresford & Wilson 1976): constant Jacobian, patch test passed, exact
pure bending (no shear locking with one or two cells through a thickness). 2 x 2 x 2 Gauss, consistent mass,
isotropic E and nu of the material at 20 degC (the material configuration). Loads: free thermal strain
`int B^T C (1,1,1,0,0,0) dV alpha dT` and initial stress `-int B^T sigma0 dV` (the bubble modes take no load from
cell-wise constant strains). All cells share one element matrix: the CSR pattern (27-node stencil) and values are
built by the kernels `neighbour_table`, `fill_indices`, `assemble_values` (int32 indices shared by every matrix).

### 3.3 Supports

Holding's contact nodes become 3 x 3 springs to ground at their points, spread to the containing cell's corners by
trilinear weights (a point outside the model snaps to the nearest model cell within two cells):
`k_n n n^T` for closed contacts (stick, slip, bonded), plus `k_t1 t1 t1^T + k_t2 t2 t2^T` for sticking and
bonded ones (`t2 = n x t1`). Three sets: *held* (every closed contact: modes, compliance, thermal, residual),
*locating* (law `locator` and `bond`: the clamping solve), *locked* (law `locked`: force-controlled clamps; their
preloads are the clamping loads). Without supports (or not clamped, or springs leaving a rigid motion free) the
part rests on a minimal 3-2-1 support: A (x, y, z) at the first node of the lowest layer, B (y, z) at the
farthest node on A's x line (else (x, z) on its y line), C (z) at the layer node farthest from line AB: B and C
constrain only components normal to AB and to the layer, so free expansions are not restrained (test: uniform
thermal strain exact to 1e-9).

### 3.4 Solvers

Systems share the model's pattern; fixed DOFs keep only their diagonal (mass rows/columns masked by an operator).
Up to `direct_max_dof` (8000) SuperLU (symmetric mode, no pivoting, MMD ordering); above, CG preconditioned by a
geometric multigrid V-cycle on the node lattice (trilinear prolongation from every second node, Galerkin coarse
operators `P^T A P`, Chebyshev smoothing of `D^-1 A` of degree `gmg_degree` (3) on `[lambda_max/30, 1.1
lambda_max]` with lambda_max from 20 power iterations, dense Cholesky on the coarsest level <= `gmg_coarse_dof`).
A 50 000-DOF factorisation would fill to 9e7 entries (1.1 GB, measured): the iterative path keeps the domain near
its budget. Relative residual `pcg_rtol` (1e-10); a solve that does not reach 1e-6 raises. Block right-hand sides
run as independent CG recurrences in chunks of 6 columns.

### 3.5 Modes and refresh

Full solve: ARPACK shift-invert at sigma = 0 (`eigsh`, mode 3), `v0 = ones/sqrt(n)` (zero at fixed DOFs),
`ncv = max(2k+1, k+16)`, `tol = eig_tol` (1e-8), the inverse by the solver above; k = `work_modes` (8) +
`work_residual_modes` (12). Refresh: Rayleigh-Ritz on the previous modes (mapped through the node lattice) plus
their static residual vectors `Z = K^-1 (K phi - lambda M phi)`, then on `[V, Z, Z_prev]` (block Krylov), up to
`RITZ_ITERATIONS` (4) steps until the energy-norm residual

    eps_k = sqrt( r_k^T K^-1 r_k / lambda_k ),   r_k = K phi_k - lambda_k M phi_k

is below `ritz_tol` (2e-3) for the published modes (10 x for the residual modes); else a full solve. The
eigenvalue error is about eps^2 (test: 1e-5 relative, MAC > 1 - 1e-5 after a pocket). The force-norm residual is not
used: the unbalanced surface tractions where cells were removed dominate it while the modes are already accurate.

### 3.6 Published modes and compliance

Shapes at cell centres (mean of the cell's model nodes, kernel `cell_average`; empty cells filled from
neighbours, so interpolation at a surface sees the surface). Residual compliance per cell from the modes beyond the
published ones, `C_res = sum_{k > n_pub} phi_k(c) phi_k(c)^T / lambda_k` (the near field beyond the computed modes is
mesh-dependent and belongs to cutting's contact model, not to structure). Damping by modal strain energy:
`zeta = (eta_w/2)(1 - s_k) + (eta_s/2) s_k`, `s_k = phi^T K_springs phi / lambda_k` (`work_loss_factor` 0.002,
`support_loss_factor` 0.1, est).

### 3.7 Long turned parts (`lathe.py`)

Cylinder and tube stocks held in a chuck (holding's `lathe`) whose free length from the chuck face exceeds
`lathe_slenderness` (3) diameters: a stepped Timoshenko beam along S z. Sections from removal's fractions on a fine
grid (`lathe_radial_cells` = 12 per radius): area, centroid, `I = sum f c^2 y^2 + f c^4/12`, the mean
`(I_1 + I_2)/2` in both planes (the part rotates), Cowper's hollow-circle coefficient of the equivalent annulus; at
most 80 elements (stiffness in series, mass averaged). Supports: chuck radial stiffness distributed over the grip
length (Winkler) plus the tilt stiffness holding gives beyond it, the axial stop at the chuck face, the torsional
grip at its centre (a chuck without stiffness: a rigid grip); tailstock and steady rest lateral springs
(`rollers/2 x roller stiffness`; zero stiffness: a rigid pin). Modes by dense eigen solve; shapes and the 6 x 6
residual compliance mapped to the cells as `u(z) + theta(z) x r` and `A C6 A^T` (`A = [I, -[r]x]`).

## 4. Thin walls (`thinwall.py`)

Detection on the structure grid from all material fractions f (not only model cells): along each axis, runs of
cells with `f > wall_min_fill` between air cells; a run is thin when its material thickness `T = sum f cell` is
below `thin_wall_cells` (2) cells; cells thin along one axis only form walls by face connectivity (at least two
cells in each in-plane direction), largest first, at most `max_walls` (8). Thickness: removal's `wall_thickness` at
the wall's central cell (from the air side) when the geometry offers it, else the median T. Plate extent from the
cell faces, free edges refined by the edge cells' fractional fill, clamped edges by the transition cells' fill
(`f = h + (1 - h) T / cell` for a solid part h with the wall above it: the wall reaches `(1 - h)` cell into it); each
edge **C** where the cells beyond are mostly solid material, **F** where mostly air, **S** otherwise (another thin
wall).

Plate: Kirchhoff rectangle a x b, thickness t, `D = E t^3/(12(1 - nu^2))`, Rayleigh-Ritz with Bhat's characteristic
orthogonal polynomials (start function `x^p (1-x)^q`, p, q = 0 free, 1 simply supported, 2 clamped; Gram-Schmidt
on shifted Legendre polynomials; `wall_terms` = 6 per direction; Bhat 1985, Warburton 1954, Leissa 1969):

    U = D/2 int [w_xx^2 + w_yy^2 + 2 nu w_xx w_yy + 2 (1 - nu) w_xy^2] dA,   T = rho t omega^2 / 2 int w^2 dA

modes mass-normalised (`rho t int w^2 dA = 1`), point compliance from the same Ritz space; `wall_modes` (3) modes per
wall published, the rest in the wall's residual `G_res = G_static - sum phi^2/omega^2`.

Blend per wall cell with `w = clip(thin_wall_cells - T/cell, 0, 1)` (1 at one cell or less, 0 at the threshold):

    phi_FE(cell) <- (1 - w) phi_FE(cell) + w phi_FE(root)          (the wall carried by the part at its root)
    C(cell)      <- (1 - w) C_FE(cell) + w (C_FE(root) + n n^T G_wall,res)
    wall modes appended with shapes w n phi_wall(cell) (zero elsewhere), zeta = eta_w / 2

the root being the nearest model cell beyond a clamped edge. At w = 1 the wall's own bending is entirely the
plate's (the voxels misrepresent it: a 1.2 mm wall is one 2 mm cell or absent); between 1 and 2 cells both
representations are mixed linearly (their error there: the FE's two-cell wall bending within 4 % (section 10), the
plate's thick-plate error `(t/a)^2` terms of a few %).

## 5. Displacement grids (`displacement.py`, `work.py`)

All in the material configuration of S (architecture 4.1), node grids, `x = X + d(X)`:

* **thermal_m**: thermal's temperature grid filled outward by nearest values, sampled at 8 points per cell and
  averaged; free strain `alpha(T)(T - 293.15 K)` (mean expansion coefficient from 20 degC of the material table);
  solved on the held supports while clamped (expansion restrained by the springs), on the 3-2-1 support when
  released.
* **clamp_m**: the clamped state: the locked contacts' preloads as loads on the part held by the locating springs
  (`(K + K_locate) u = f_locked`); when the locators leave a rigid-body motion free (chucks: all jaws are locked),
  the locked springs are added at `soft_support_ratio` (1e-3) of their stiffness and the rigid motion in the free
  directions is projected out (M-orthogonal). Zero after unclamping: the spring-back is the release itself (the
  material was cut at `X_e = x_e - d(x_e)` in the clamped state, so the released geometry carries the error).
* **residual_m**: the as-supplied stock's residual stress (material `residual_stress` of the stock's form, the
  entry nearest its thickness): plates in-plane L (along the stock's longer side) and LT, each cell layer the
  profile's exact mean over its depth, balanced over the stock's layers (zero force and moment: the stock is in
  equilibrium); bars and tubes axial / hoop / radial over r/R, axial resultant balanced. Release load
  `-sum_remaining B^T sigma0 V`, solved as held (clamped) or on 3-2-1 (released: distortion of the free part), minus
  the as-supplied stock's own response with the same kind of support (its fractions captured at init,
  `FillSnapshot`): d = 0 before any cut, exactly. Element deactivation is the model losing cells.
* **total_m** = thermal + clamp + residual. Nodes outside the model take the mean of model neighbours (3 passes).
* **Treuting-Read** (forward form, the check): a plate keeping `z in [0, h1]` of a profile curves by
  `kappa = D^-1 C^-1 M`, `M = -int sigma (z - h1/2) dz`, `D = E h1^3/12`, `C = [[1, nu], [nu, 1]]/(1 - nu^2)` (a narrow
  strip with uniaxial sigma_x: `kappa_x = 12 M_x/(E h1^3)`); `displacement.treuting_read_curvature`.

Long turned parts: thermal free strain and curvature (`kappa = -int alpha dT x dA / I`) and the residual eigenstrain
`-sigma0/E` resultants of every slice as beam loads (Boley & Weiner 1960), solved as held, plus the radial
expansion `alpha dT r`; clamp_m zero (local jaw crushing of a long bar not modelled).

## 6. Fixture compliance (`fixture.py`)

`C6 = K6^-1` of holding's `support_stiffness_S` about `centroid_S_m` (eigen-decomposition; directions with an
eigenvalue below 1e-12 of the largest are free rigid motions, holding's matter, and are left out). At a point p:
`C3 = A C6 A^T`, `A = [I, -[r]x]`, r = p - centroid. Published at the part datum p = the W origin in S
(`ctx.project.stock_pose_W_S`, the work zero), with C6 and the centroid for any other point.

## 7. Determinism

No random numbers (no RNG streams). Dense eigen solves (LAPACK `eigh`), ARPACK with `v0 = ones/sqrt(n)`, fixed
`ncv`, `tol`, `maxiter`; CG and multigrid in fixed order; one thread (bin/tsafe). Eigenvectors are mass-normalised
and sign-normalised (largest-magnitude component positive, the first such); degenerate clusters (`|df|/f < 1e-6`)
are rotated to a canonical basis: the M-projections onto the cluster of the rigid translations along x, y, z, in
that order, Gram-Schmidt orthonormalised in the M inner product (a projection below 1e-6 of the largest is noise and
skipped), completed by the cluster's own vectors. Test: two processes give the same SHA-256 of eigenvalues and
eigenvectors (direct, multigrid, Ritz, tool). The state (`StructureState.to_arrays`) carries the published and the
latest solutions, the pending jobs and the stock snapshot: continuation after a checkpoint is bit-identical (test).

## 8. Kernels (`kernels.py`; references in `reference.py`; one signature each)

| kernel | signature | does | cost (measured) |
|---|---|---|---|
| `sample_node_grid` | `void(f8[::1] origin, f8 cell, f8[:,:,:,::1] grid, f8[:,::1] points, f8[:,::1] out)` | d at points, trilinear in a node grid, clamped | 45 ns/point |
| `material_points` | `void(f8[::1], f8, f8[:,:,:,::1], f8[:,::1] x, i8 n_iter, f8[:,::1] out)` | `X = x - d(X)`, n fixed-point steps (hot loop: 1) | 67 ns/point, 0.9 us per single-point call |
| `neighbour_table` | `i8(i1[:,:,::1] active, i8[:,:,::1] node_index, i8[:,::1] node_ijk, i8[:,::1] nbr)` | 27-stencil coupling ranks | part of the 70 ms build at 52k DOF |
| `fill_indices` | `void(i8[:,::1], i8[:,:,::1], i8[:,::1], i8[::1] indptr, i4[::1] indices)` | CSR column indices | idem |
| `assemble_values` | `void(i8[:,::1] cells, i8[:,:,::1], i8[:,::1], i8[::1], f8[:,::1] Ke, f8[::1] scale, f8[::1] data)` | element matrix over all cells | idem |
| `add_point_springs` | `void(i8[:,:,::1], i8[:,::1], i8[::1], i8[:,::1] cell, f8[:,::1] w, f8[:,:,::1] k, f8[::1] data)` | spring blocks by trilinear weights | < 1 ms per 100 contacts |
| `scatter_cell_vectors` | `void(i8[:,::1], i8[:,:,::1], f8[:,::1] coef, f8[:,::1] Ge, f8[::1] out)` | element loads into a vector | ~10 ms at 16k cells |
| `cell_average` | `void(f8[:,:,:,::1] node_vals, i1[:,:,::1] node_ok, f8[:,:,:,::1] out, i1[:,:,::1] has)` | node values to cell centres | ~1 ms per mode at 16k cells |

All `@njit(sig, cache=True, nogil=True, error_model="numpy")`, no fastmath / parallel. Integration: per contact and
L0 step, `material_points(origin[w], cell[w], total_m[w], x_e, 1, X_e)` with `StructureInputs` of the step.

## 9. Settings (`physics.structure`) and parameters

Settings (file key, default): `every_n_macro` 50 (refresh cadence; the latency is fixed at
`REFRESH_LATENCY_MACRO` = 50 by the contract), `max_dof` 50000, `min_cell_mm` 1, `fill_threshold` 0.5,
`work_modes` 8, `work_residual_modes` 12, `eig_tol` 1e-8, `ritz_tol` 2e-3, `direct_max_dof` 8000, `pcg_rtol`
1e-10, `pcg_max_iter` 3000, `gmg_coarse_dof` 1500, `gmg_degree` 3, `tool_max_freq_Hz` 20000, `tool_max_modes` 16,
`tool_element_mm` 2, `tool_stations` 24, `pullout_refresh_mm` 0.02, `thin_wall_cells` 2, `wall_terms` 6,
`wall_modes` 3, `max_walls` 8, `wall_min_fill` 1e-3, `lathe_slenderness` 3, `lathe_radial_cells` 12,
`soft_support_ratio` 1e-3.

Model parameters (`structure-tool`): holder E 207 GPa, nu 0.292, rho 7800 (Budynas & Nisbett Table A-5); contact
nu 0.292; contact stiffness per cm2 (N/um): shrink fit 1000, hydraulic 500, milling chuck 600, ER collet 150,
side-lock 100, tapping 100, face 1000, turret 500 (est: joint stiffness of ground steel joints, 1e13 N/m3 order,
graded by construction); contact loss factors 0.15 / 0.3 / 0.2 / 0.2 / 0.2 / 0.25 / 0.1 / 0.15 (est: chosen for
1-3 % damping of tool bending modes, measured tool-point FRFs); tool and holder loss factor 0.002, damped bars
0.05 (est); spindle pivot length 300 mm (est); clamp length ratios 3 d (holders), 4 d (bars), 1.5 h (shanks)
(est). `structure-work`: work loss factor 0.002, support loss factor 0.1 (est). Material E, nu, rho, alpha and
residual-stress profiles from the material database; tool substrate E, nu, rho from `db/tool_materials`.

## 10. Validity, error bounds and how they were established

| quantity | validity | error bound | basis (tests) |
|---|---|---|---|
| Timoshenko element | any L/d | exact static end loads; SS frequency 1e-4 | `test_beam`: tip deflection 1e-12, SS Timoshenko frequency equation 1e-4, cantilever within the shear bounds |
| RCSA coupling | linear substructures | 1e-8 (numerical) | `test_rcsa`: two beams vs monolithic FE, modal base vs tied system |
| tool clamp (lateral bed) | decay length >= 2 elements | 2e-3 vs Hetenyi (shear-rigid) | `test_toolmodel`: stepped cantilever + Hetenyi; boring bar 1e-4 |
| tool model discretisation | 2 mm elements | 0.06 % tip compliance, 0.02 % frequency | 0.5 mm vs 2 mm elements |
| tool modal model vs RCSA | lightly damped | 3 % (fit_error) | half-power band comparison |
| tool-point compliance (physical) | 40-taper spindles, carbide/steel tools | +-15 % (interface stiffness x/3: +10 %/-7 %; pivot 0.15-0.6 m: +9 %/-4 %; spindle modes est) | sensitivity runs (D10 end mill, 35 mm stick-out) |
| tool frequencies (physical) | as above | +-5 % (contact x/3: -4 %/+3 %; pivot: -3 %/+1.5 %) | sensitivity runs |
| tool damping | as above | factor 2 (loss factors est) | MSE with est loss factors |
| voxel bending | >= 1 cell through a thickness | plate 1.5 % after the shear term `1 + 14.5 (t/a)^2`; cantilever f 1 % of EB, tip 2 % of Timoshenko | `test_voxel` (clamped square plate vs 0.00126 q a^4/D) |
| multigrid vs direct | SPD voxel systems | 1e-7 relative (rtol 1e-10) | `test_voxel`; 14 iterations at 52k DOF |
| Ritz refresh | small material changes | eigenvalue 1e-5, MAC 1 - 1e-5 at eps 2e-3 | `test_voxel` vs full eigsh |
| support springs | stiff part | 2e-4 bounce frequency | `test_voxel` |
| free thermal expansion | any shape | 1e-9 (patch test) | `test_voxel`, `test_displacement` |
| thermal bow | layer-wise constant T | 2 % (measured 1e-6) | `test_displacement` |
| residual release vs Treuting-Read | strip / plate, cells >= 4 through h | 3 % (measured 3e-6) | `test_displacement` |
| thin-wall plate | t/a < 0.1, rectangular walls | CFFF 3.4718 vs converged 3.4711 (0.02 %), SSSS 1e-6, CCCC 1e-4; cylindrical bending within 2 % | `test_thinwall` |
| thin-wall geometry | axis-aligned walls, flat floors | extent exact for a flat floor or a wall end across the transition cell, else about +-0.2 cell (compliance ~ H^3: +-20 % for a wall 3 cells high); thickness exact with removal's query | `test_thinwall` (floor on a cell face and inside a cell: root exact) |
| lathe beam | L/D > 3 | frequency within the Timoshenko bound, section I 0.2 % | `test_lathe` (rigid chuck cantilever, clamped-pinned) |
| determinism | same build and hardware | bit-identical | `test_determinism`, checkpoint test |

**Hand check of a D10 in an ER32 collet (the tiny project's T1, 2026-11-09).** The loop's report (docs/models/
loop.md 15) found the 5339 Hz pair odd because its tip modal compliance (1.37e-7 m/N) exceeds the 697 Hz pair's
(3.6e-8 m/N). It is the tool's own first bending mode on its collet, not an artefact: the fluted section (I =
1.56e-10 m^4, d_eq 7.5 mm, E 600 GPa) 25 mm long plus 7 mm of shank as a rigid-clamped stepped cantilever has a
tip compliance of L1^3 / 3EI1 + (L^3 - L1^3) / 3EI2 = 5.55e-8 + 1.94e-8 = 7.5e-8 m/N (13 N/um) and a first
frequency of 7.2 kHz (uniform fluted beam, 1.875^2 / 2 pi sqrt(EI / rho A L^4), rho A = 0.54 kg/m): the collet's
contact springs lower the frequency to 5.3 kHz, the stiffness to 7.2 N/um ((5.34/7.18)^2 x 13 N/um = 7.4 N/um),
and the modal mass m = k / omega^2 = 6.4 g matches 0.24 times the tool's 20 g overhang. The 697 Hz pair is the
spindle flange mode (950 Hz, 70 N/um in the machine file, a modal mass of 1.96 kg) carrying holder and tool;
1654 Hz the second spindle mode. A slender tool in a collet is softer than the spindle at its tip, so its own
mode carries most of the tip compliance (static 1.97e-7 m/N, 5.1 N/um), as measured tool-point FRFs of such tools
show. Mass normalisation: each mode's sum over stations of phi^T M phi = 1 (the eigen solve's normalisation),
the sum of the modal tip compliances plus the residual equals the static tip compliance to round-off.

## 11. Assumptions and limitations (consequences)

* Tools: the mean bending stiffness of anisotropic (2-flute) sections in T0; the 2-Omega parametric part is not
  carried (2-flute tools: their split principal frequencies are +-8-12 % of the mean). Gyroscopic effects are not
  included (frequency splitting of a few % for heavy holders above 20 000 rpm). The flute is a prismatic equivalent
  section (helix ignored).
* The holder is solid above the shank end (coolant and set-screw bores ignored: holder stiffness overestimated by
  < 2 % for typical bores); its mass is the protruding outline's (the taper is in the base modes).
* Spindle base: translation-only modal data plus an estimated pivot (tool-point compliance +9 %/-4 % for a factor
  2 on the pivot); axial and torsional base compliance absent unless the machine gives z modes (axial tool
  compliance then only of tool and holder).
* Interface contact stiffness and loss factors are estimates (no identified values for the database's holders):
  the dominant uncertainty of tool modes for long, slender tools.
* Damping: modal strain energy (diagonal) approximation of non-proportional damping; the damped peaks shift by
  ~3e-4 (fit_error 3 %).
* Workpiece: a cell is in or out (fill > 0.5): surfaces are stair-cased at the cell size (7 mm at 50 000 DOF for a
  300 x 200 x 100 mm part); features below two cells are represented by the thin-wall plates (walls) or not at
  all (pins, ribs thinner than a cell along two axes). Material at 20 degC E for the FE.
* The residual compliance contains only the computed modes beyond the published ones: the local (contact)
  compliance is cutting's; with few residual modes the static compliance at thick regions is underestimated by
  the truncated high modes (the contact near field).
* Supports are linear springs about the clamped state (friction states from holding); clamps' preloads only for
  clamping distortion; self-weight not included (sag of a 500 mm, 30 mm steel bar on two supports: 5 um, along
  gravity, nearly tangential to a lathe cut).
* Thin walls: rectangular plates on the grid's axes, independent of each other, clamped / free / simply supported
  edges by the neighbourhood; curved and inclined walls are not detected; plate modes are not coupled to the part's
  FE modes (the carry-along at the root is the coupling); thermal and residual distortion of thin walls comes from
  the FE only.
* Residual stress: in-plane plate profiles (no ST, no shear), the profile scaled to the stock thickness, the rolling
  direction along the longer side; bars: axisymmetric profiles (none in the database yet). Linear elastic release
  (no plasticity, no machining-induced layer stresses: those are surface's).
* Long turned parts: axisymmetric mean section stiffness; jaw crushing not modelled; tailstock axially free.
* Refresh jobs are computed when they start and applied 50 macro steps later (deterministic); a new setup, the
  first step and the end of an operation are solved at once. A new `supports_version` of the same operation and
  clamped state is a new setup only when the re-placed supports changed materially (`supports_unchanged`:
  the same contacts, each point within `support_change_rel` (2 %) of a cell, every contact stiffness and clamping
  force and the lathe conditions within 2 % of the largest); holding re-places the supports at every rebuild (a
  chuck's grip during a spindle run-up, slip of a few um), and each setup is a full eigen solve and a new modal
  basis in dynamics (examples/lathe-shaft-4140: 55 setups in 2.4 s before 2026-11-09). A setup belongs to its
  part of a batch (`WorkStructState.part_index`): the next part is set up anew when its blank is placed, and the
  released state of an unclamped part stays published only until then (a one-operation batch runs every part
  as operation 0; until 2026-11-09 the first part's released state stayed published for all later parts).

## 12. Cost (this box, one core, after JIT)

Tool model: 0.4-0.5 s per tool (480 DOF: dense eigen solve, static solves, RCSA fit check), at init and on refresh
triggers. Idle macro step: 0.3 ms. Workpiece at 3 000 DOF: setup (full eigen solve, displacements) 0.2 s. At 52 000
DOF: model build 0.07 s, multigrid setup 0.4 s, one solve 1 s (14 CG iterations to 1e-10), a block of 20 solves 6 s,
full eigen solve of 20 modes 60 s, a Ritz refresh 2-5 block solves (12-30 s); memory 90 MB model + 60 MB system,
peak 220 MB during block solves (budget 200 MB). Jobs run every `every_n_macro` (0.5 s simulated by default) and
are skipped when geometry, temperatures and supports did not change.

## 13. Integration

* `StructureDomain(geometry=removal_domain.geometry(state), tools=None)`: wire removal's FieldGeometry (fill,
  bounds, wall thickness); tooling's init must run first (tool geometry from `state.tooling`).
* Hot loop: `kernel_inputs()` gives `StructureInputs`; map edge elements with
  `material_points(origin[w], cell[w], total_m[w], x_e, 1, X_e)` (architecture 4.1); work-mode shapes are cell-centre
  grids (dynamics' `refresh_work_shapes`), displacement grids node grids (removal's mesh sampler).
* Dynamics: ToolModes replace the fallback spindle/turret modes (they contain them); `fit_error` and
  `tip_compliance_m_per_N` are for reports and checks. `fixture_compliance_m_per_N` is at the W origin; use
  `fixture_compliance_6_m_per_N` about `fixture_centroid_S_m` for the contact point.
* Inspection: `X + residual_m (released, final)`; `part_<op>_unload`: `X + total_m` of the released state
  (`Displacement.clamped` False after `end_operation`).
* Wear / report: `toolmodel.static_deflection`, `section_stresses`, `tip_frf`.
* A job can be run in a worker between its start and its application (inputs frozen at the start); the domain
  currently computes it synchronously at the start.

## References

* Adams, M.; Brezina, M.; Hu, J.; Tuminaro, R. (2003) Parallel multigrid smoothing: polynomial versus Gauss-Seidel,
  J. Comput. Phys. 188, 593-610 (`adams-2003-polynomial-smoothing`).
* Bhat, R.B. (1985) Natural frequencies of rectangular plates using characteristic orthogonal polynomials in
  Rayleigh-Ritz method, J. Sound Vib. 102, 493-499 (`bhat-1985-orthogonal-polynomials`).
* Blevins, R.D. (1979) Formulas for Natural Frequency and Mode Shape (`blevins-1979-formulas`).
* Boley, B.A.; Weiner, J.H. (1960) Theory of Thermal Stresses (`boley-weiner-1960-thermal`).
* Briggs, W.L.; Henson, V.E.; McCormick, S.F. (2000) A Multigrid Tutorial (`briggs-henson-mccormick-2000-multigrid`).
* Budynas, R.G.; Nisbett, J.K. (2015) Shigley's Mechanical Engineering Design (`budynas-nisbett-2015-shigley`).
* Cowper, G.R. (1966) The shear coefficient in Timoshenko's beam theory, J. Appl. Mech. 33, 335-340
  (`cowper-1966-shear`).
* Craig, R.R.; Kurdila, A.J. (2006) Fundamentals of Structural Dynamics (`craig-kurdila-2006-structural-dynamics`).
* Erturk, A.; Ozguven, H.N.; Budak, E. (2006) Analytical modeling of spindle-tool dynamics ... receptance coupling,
  Int. J. Mach. Tools Manuf. 46, 1901-1912 (`erturk-ozguven-budak-2006-rcsa`).
* Genta, G. (2005) Dynamics of Rotating Systems (`genta-2005-rotating`).
* Hetenyi, M. (1946) Beams on Elastic Foundation (`hetenyi-1946-elastic-foundation`).
* Johnson, C.D.; Kienholz, D.A. (1982) Finite element prediction of damping ..., AIAA J. 20, 1284-1290
  (`johnson-kienholz-1982-mse`).
* Johnson, K.L. (1985) Contact Mechanics (`johnson-1985-contact-mechanics`).
* Kops, L.; Vo, D.T. (1990) Determination of the equivalent diameter of an end mill based on its compliance,
  CIRP Annals 39, 93-96 (`kops-vo-1990`).
* Lehoucq, R.B.; Sorensen, D.C.; Yang, C. (1998) ARPACK Users' Guide (`lehoucq-sorensen-yang-1998-arpack`).
* Leissa, A.W. (1969) Vibration of Plates, NASA SP-160 (`leissa-1969-plates`).
* Przemieniecki, J.S. (1968) Theory of Matrix Structural Analysis (`przemieniecki-1968-matrix`).
* Schmitz, T.L.; Duncan, G.S. (2005) Three-component receptance coupling substructure analysis ..., J. Manuf. Sci.
  Eng. 127, 781-790 (`schmitz-duncan-2005-rcsa`).
* Schmitz, T.L.; Smith, K.S. (2009) Machining Dynamics (`schmitz-smith-2009-machining-dynamics`).
* Taylor, R.L.; Beresford, P.J.; Wilson, E.L. (1976) A non-conforming element for stress analysis, Int. J. Numer.
  Meth. Eng. 10, 1211-1219 (`taylor-beresford-wilson-1976`).
* Timoshenko, S.; Woinowsky-Krieger, S. (1959) Theory of Plates and Shells (`timoshenko-woinowsky-1959-plates`).
* Treuting, R.G.; Read, W.T. (1951) A mechanical determination of biaxial residual stress in sheet materials,
  J. Appl. Phys. 22, 130-134 (`treuting-read-1951`).
* Warburton, G.B. (1954) The vibration of rectangular plates, Proc. IMechE 168, 371-384 (`warburton-1954-plates`).
* Young, W.C.; Budynas, R.G.; Sadegh, A.M. (2012) Roark's Formulas for Stress and Strain (`young-budynas-2012-roark`).
