# Mechanics library: chip formation, heat partition, chip form and colour

`cnctwin/mechanics/` is a **library** (architecture 2): pure functions, precomputed tables and
Numba kernels that every domain may import. It owns no SimState field and no bus record; the
domains that call it read their inputs from `bus_prev` and write the results into their own
records (cutting: `ProcessCoefficients`; chips: `ChipFormationContact`/`EmissionBatch`; wear:
`ToolWear.bue_height_m`). Coefficients: `db/models/mechanics-{oxley,contact,calibration,chip,colour}.yaml`;
sources: `db/bibliography/models-mechanics.yaml` (plus the material bibliographies).

| module | content |
|---|---|
| `materials.py` | `WorkMaterial.from_db`, `ToolSide.from_db`: database objects (SIMap) to plain dataclasses; selection of friction, Kienzle and mechanistic entries |
| `kernels.py` | all Numba kernels (Johnson-Cook, Oxley shear plane / interface / point solver / map, fallback, lookups, edge contact, oxide exposure, BUE) |
| `oxley.py` | `solve` (one orthogonal condition), `solve_reference` (independent SciPy solver), `zorev_split` |
| `oblique.py` | Stabler flow rule, equivalent rake, Budak-Altintas-Armarego coefficients, chip flow direction |
| `contact.py` | h_min, edge (ploughing) forces, ploughing below h_min, indentation/extrusion, flank wear land, K_sp, Merchant / Lee-Shaffer |
| `heat.py` | stationary and moving heat sources, Blok partition, zone powers |
| `coefficients.py` | process maps, group coefficient rows, calibration, `process_coefficients`, `coeff_table`, `flank_coefficients` |
| `chipform.py`, `colour.py` | chip geometry, curl, breaking, ISO 3685 class, segmentation, BUE, ejection; oxide colour |

All quantities are SI; temperatures in kelvin. "Per unit width" means per metre of chip width.

---------------------------------------------------------------------------------------------------

## 1. Flow stress (Johnson & Cook 1983)

    sigma = (A + B eps^n) (1 + C ln(epsdot*)) (1 - T*^m),
    epsdot* = max(epsdot / epsdot_0, min_jc_rate_ratio),  T* = clamp((T - T_ref)/(T_m - T_ref), 0, max_jc_Tstar)

T* is clamped below at 0 (no hardening below the reference temperature, never a negative base
for a fractional power) and above at `max_jc_Tstar` = 0.99: the flow stress never vanishes (1 % of
the cold value at m = 1); the rate factor is floored at 1 - max_jc_Tstar. k = sigma/sqrt(3).
Equivalent strain-hardening index (Lalwani et al. 2009): n_eq = n B eps^n / (A + B eps^n)
(= d ln sigma / d ln eps, verified numerically in the tests).

Materials without a JC law (polymers, wood, foam, composites: `flow_stress` null or model none) use
the **data-only model** (section 7.4); for their edge/flank pressures the flow stress is the
yield strength falling linearly to zero at the melt/smear (or decomposition) temperature.

## 2. Oxley's theory for orthogonal cutting

Oxley (1989) as extended to the Johnson-Cook law by Lalwani, Mehta & Jain (2009); equations as
implemented in the published ExtOxley code (Pantale et al. 2022, read). Unknowns: shear angle phi,
strain-rate constant C0 = l_AB / (primary-zone thickness), secondary-zone ratio delta.

**Shear plane AB** (t1 uncut thickness, V speed, alpha normal rake, T_w work temperature):

    l_AB = t1/sin phi,  V_s = V cos a / cos(phi - a),  V_c = V sin phi / cos(phi - a),  t2 = t1 cos(phi - a)/sin phi
    gamma_AB = cos a / (2 sin phi cos(phi - a)),  eps_AB = gamma_AB/sqrt3,  epsdot_AB = C0 V_s / l_AB / sqrt3
    T_AB = T_w + eta dT_SZ,  dT_SZ = (1 - beta) F_S V_s / (rho V t1 c_p(T_AB)),  F_S = k_AB(eps_AB, epsdot_AB, T_AB) l_AB
    beta = 0.5 - 0.35 log10(R_T tan phi)  (R_T tan phi <= 10),  0.3 - 0.15 log10(R_T tan phi)  (> 10),  R_T = rho c_p V t1 / K

T_AB is the root of T - T_w - eta dT_SZ(T) on [T_w, T_cap] by Brent's method (T_cap: the
homologous clamp; hitting it is flagged `OX_T_CAPPED`). R_T tan phi below 0.04 (the stated validity
limit of the beta fit) is held at 0.04 (beta = 0.989).

    tan theta = 1 + 2(pi/4 - phi) - C0 n_eq,  lambda = theta + a - phi,  R = F_S / cos theta
    F_C = R cos(lambda - a),  F_T = R sin(lambda - a),  F = R sin lambda,  N = R cos lambda
    sigma_N' = k_AB (1 + pi/2 - 2a - 2 C0 n_eq)
    l_c = t1 sin theta / (cos lambda sin phi) (1 + C0 n_eq / (3 tan theta)),  tau_int = F/l_c,  sigma_N = N/l_c

**Interface** (per delta):

    eps_int = (2 gamma_AB + 0.5 l_c/(delta t2))/sqrt3,  epsdot_int = V_c/(delta t2)/sqrt3
    dT_C = F V_c / (rho V t1 c_p(T_c)),  T_c = T_w + dT_SZ + R dT_C          (mean chip temperature)
    log10(dT_M/dT_C) = 0.06 - 0.195 delta sqrt(R_T t2/l_c) + 0.5 log10(R_T t2/l_c),  dT_M >= dT_C
    T_int = T_w + dT_SZ + psi R dT_M,   k_chip = sigma_JC(eps_int, epsdot_int, T_int)/sqrt3

With delta -> 0 the dT_M fit gives 1.148 sqrt(R_T t2/l_c) dT_C, Jaeger's fast band-source maximum
(1.128): a consistency check of the fit. dT_M is floored at dT_C (the maximum cannot be below the
mean; the fit drops below it for R_T t2/l_c < 1).

**Rake heat partition at the current tool temperature** (our extension; Loewen & Shaw 1954,
Komanduri & Hou 2001): Oxley's dT_M assumes an insulated tool (R = 1). With the tool rake
temperature T_tool (thermal's `ToolHeat.rake_temp_mean_K`), R equalises the mean interface
temperature of the chip (moving source, psi R dT_M) and of the tool (stationary source):

    T_w + dT_SZ + psi R dT_M = T_tool + (1 - R) q M_t,   q = F V_c / l_c
    M_t = min(2 l_c A(w/(2 l_c)) / k_t,  (4/3) sqrt(kappa_t t_c / pi) / k_t) + t_coat / k_coat
    A(m) = (1/pi)[asinh m + m asinh(1/m) + (1 + m^3 - (1 + m^2)^{3/2})/(3m)]

A is the exact mean rise of a uniformly heated rectangle on a half-space (tested against the
closed-form self-potential of the rectangle); the source is doubled (image) because the flank
face is taken adiabatic (quarter-infinite tool); the transient cap is the mean of
2 q sqrt(kappa t/pi)/k over one contact time t_c (milling engagements; none in continuous
cutting); the coating adds its film resistance (thickness from the tool, else the coating's
typical thickness). R is clamped to [0, 1] and solved together with T_c (fixed point inside
the interface evaluation), so the force solution and the partition are one consistent state.

**Interface condition: Oxley's own** (Oxley 1989; Hastings, Mathew & Oxley 1980; Adibi-Sedeh,
Madhavan & Bahr 2003; Lalwani et al. 2009). The shear angle is the one for which the mean interface
shear stress from the force balance equals the chip material's shear flow stress at the interface
temperature and strain rate (the chip seizes on the rake face):

    g1 = tau_int - f k_chip(eps_int, epsdot_int, T_int),   f = CoolantFeedback.friction_factor (1 dry)

The friction angle lambda (and the apparent coefficient F/N = tan lambda) is an **output**. With a
lubricant at the interface (f < 1, the coolant domain's capillary-access factor) the interface
carries f times the chip's shear flow stress: the constant-friction-factor (Tresca) form of the
same condition, regime 3 `lubricated`; f = 1 is Oxley's condition exactly, regime 1 `sticking`.
Why this is the default (`oxley.interface_coulomb(None)` is False for every caller): in continuous
chip formation the normal stress near the edge is several times the chip's shear flow stress
(Zorev's peak (a + 1) sigma_N), the real contact area equals the apparent one and the chip seizes
(Shaw, Ber & Mamin 1960: the interface shear is limited by the shear flow stress of the softer
body); Oxley's mean interface stress is that seized shear. The database `mu_sliding` values are
local Coulomb coefficients (FE inputs, tribometer or Albrecht estimates): capping the *mean*
interface stress at mu sigma_N with them (the former default) let the database coefficient fix the
shear angle (lambda = atan mu), so the predicted chip ratio barely moved with speed and feed. They
are used where the contact slides: the edge arc below the stagnation point, the flank wear land,
the Zorev split below and the plain-Merchant fallback.

**Documented alternative, on request only**: `coulomb_limit=True` (solve, solve_reference,
build_process_map) caps the interface at the Coulomb-Tresca limit (Shaw, Ber & Mamin 1960),

    g1 = tau_int - min(k_chip, mu sigma_N),   mu = mu_sliding(database, tool class/coating) x f

regime 2 `sliding` where mu sigma_N governs; for a caller whose mu is a measured *mean* interface
coefficient of the condition (a low-adhesion coating on which the chip slides over the whole
contact, a lubricated test with measured friction). No harness or domain of this repository asks
for it.

**Conditions solved**: (1) g1 = 0 (phi), (2) sigma_N' = sigma_N (C0), (3) delta minimising F_C.
Condition (2) involves only the shear plane, so C0(phi) is solved once per phi node.

**Algorithm** (`kernels.oxley_point`): (i) phi on 40 nodes in [5, 60] deg (the lower end is the
published extended-Oxley search range, Pantale 2022 code: phi 5-45 deg); at each node C0 by a
9-node scan of [2, 10] and Brent (several roots: smallest F_C; no root: the C0 minimising
|sigma_N' - sigma_N| by golden section, accepted if the mismatch is < 25 % of k_AB, as the ExtOxley
grid search does; weakly hardening materials make (2) insensitive to C0). (ii) delta on 12
log-spaced nodes in [0.005, 0.2] and 16 golden-section steps around the best node; for each delta
the phi root of g1 by Brent on the linear blend of the bracketing node states (smallest F_C, larger
phi on a tie). (iii) polish: exact Brent root in phi with exact C0(phi) at the optimal delta.
**Admissibility**: a solution whose force-minimising delta lies at the top of the published range
(within 0.1 % of 0.2, `kernels.DELTA_EDGE`) is rejected: the secondary zone would be thicker than
the theory's thin-zone assumption allows (the flow zone grows into the chip), and these are the
degenerate roots of low speeds and small t1 (phi 5-8 deg, a chip ten or more times thicker than
the cut, F_C two to three times the regular branch; at phi 1-2 deg with the former 1 deg search
bound). The lower end of the range (0.005, the surface heat source of high R_T) stays admissible.
In the sliding alternative F_C does not depend on delta; the smallest delta is kept by the
tie-break. All grids and tie-breaks are fixed: results depend only on the inputs. Cost: 2-5 ms per
condition (section 10).

Where no admissible solution exists. With Johnson-Cook laws the condition fails at low cutting
speeds and small t1: the interface strain (eps_int = 5-30 with delta = 0.05-0.2) hardens the cold
interface layer beyond what the shear-plane force system can load (k_chip > tau_int at every phi;
AISI 1045 below about 100 m/min at h = 0.1 mm, AISI 304 below 40-60 m/min at h <= 0.2 mm, section 13).
Nakayama (1962, Fig. 25) measured that the strength of steel chips saturates at shear strains above
2 (about twice the uncut material's), so the Johnson-Cook extrapolation to these interface strains
is one cause. This coincides with the speeds at which steels form a built-up edge (Ck45 BUE window
10-50 m/min of the material database; Nakayama 1962 observed BUE on S45C at 50-75 m/min). The
single-condition solver and the maps then use the fallback of section 3.

**Zorev split** (Zorev 1963; `oxley.zorev_split`): sigma(x) = sigma_max (1 - x/l_c)^a with
sigma_max = (a + 1) sigma_N, tau = tau_p on the sticking length s l_c and mu sigma(x) beyond,
continuous at s l_c and with mean tau_int: (1 - s)^a (1 + a s) = tau_int/(mu sigma_N), a = 3
(`zorev_exponent`, est). The force solution uses only the mean stresses; the split is reported
(`OX_STICK`, `GroupCoefficients.sticking_fraction`) for the wear model's sliding velocity. The
database `sticking_ratio` (estimates at unstated conditions, inconsistent with the mean stresses
for most pairings) is not used by the force or split model.

## 3. Fallback when Oxley's conditions have no solution

Cases: no admissible root of the interface condition (low speed and small t1, section 2), very
small t1 (beta outside its range, C0 condition unsolvable), large negative rakes (theta or lambda
leave (0, pi/2)), very high temperatures. **Modified Merchant** (`kernels.merchant_point`):
2 phi + lambda - alpha = C_M with lambda, C_M = 2 phi + lambda - alpha, C0 and delta of a reference
Oxley state; the shear flow stress, temperatures and contact come from the same shear-plane and
interface models.

**Reference: the edge of the admissible region** (`oxley.continuation_reference`): from the failing
condition, thinner and thicker cuts at the same speed are tried (factor 1.72, up to 3 steps each
way, thinner first), then lower and higher speeds at the same thickness (factor 1.55, up to 6 steps);
between the first admissible trial and the inadmissible one before it the edge is located by 8
bisections in log t1 (or log V) (`fallback_*` in mechanics-oxley). The reference is the Oxley state
at that edge, so the fallback is the theory's state at the limit of its validity carried to the
failing condition (seizure persists; the friction angle stays an output of the theory, the
database coefficient does not enter). Process maps (`coefficients.build_process_map`) refine the
kernel's nearest-node fallback the same way: in a row of t1 nodes, a block of failing nodes is
continued from the bisected edge between it and its nearer admissible node; a row without an
admissible node uses the point continuation of each node. A map node and `oxley.solve` therefore
give the same fallback (within 1 %, tested), whatever the grid; with the earlier nearest-node rule a
single-speed calibration map and a cutting map continued the same condition from different nodes
(shear angle 19.3 against 13.1 deg for Ti6Al4V at 20 m/min) and a calibration factor did not
transfer (+25 % at its own reference point). Only when no trial of the ladder is admissible does
the kernel's plain Merchant (C_M = pi/2, lambda = atan of the database mu) remain.
`OrthogonalResult.fallback_reference` names the reference.

Lee & Shaffer (phi = pi/4 - lambda + alpha) is provided (`contact.shear_angle_lee_shaffer`) but not
used: it gives phi <= 0 for lambda - alpha >= 45 deg, exactly the negative-rake cases that need a
fallback, and Oxley's own relation has a slope d phi/d(lambda - alpha) of about -0.67, closer to
Merchant's -1/2 than to Lee-Shaffer's -1.

Error of the modified Merchant relation itself (measured on 54 neighbour pairs, AISI 1045, rakes -6
to +11 deg, 1-6 m/s, 20-400 um, one map node away in t1 or speed, former Coulomb-limited model):
F_C within 9.1 % (rms 4.5 %), F_T within 13.5 % of F_C (rms 6 %), T_int within 12.7 %. Below the edge
the fallback is an extrapolation: the edge state has the thickest admissible secondary zone and the
smallest shear angle of its neighbourhood, so at low speed it gives thick chips (Ck45 at 10-70
m/min, h = 0.1 mm: chip ratio 0.19-0.21 against 0.24-0.68 measured, section 9). A fallback shear
angle below 2 deg (`min_merchant_phi_rad`) marks the condition as indentation.

## 4. Edge, ploughing, indentation and flank wear land (`contact.py`)

**Minimum chip thickness** h_min = (min_chip_thickness_ratio of the material, else 0.3) x r_e.
The chip forms from the layer above the stagnation point: the shearing force uses
**t1 = h - h_min**; the layer below gives the edge forces (Albrecht 1960).

**Edge forces** (h >= h_min, per unit edge length; `kernels.edge_contact_forces`): contact on the
edge arc from the lowest point (normal angle 0) to the stagnation point w_s = acos(1 - h_min/r_e),
uniform pressure p = (2 + pi) k (Prandtl's flat punch on a rigid-plastic half-space, Hill 1950) and
friction tau_e = min(mu p, k), material flowing down under the edge:

    F_te = cos i [p h_min + tau_e r sin w_s cos i] + tau_e r w_s sin^2 i
    F_re = p r sin w_s - tau_e cos i h_min
    F_ae = p h_min sin i + tau_e r sin i cos i (sin w_s - w_s)

(verified against an independent quadrature of the pressure and friction integrals). k is k_AB of
the map at t1 = h_min. Above the stagnation height (indentation/extrusion) the friction reverses.
**Ploughing** (h < h_min): the same contact at depth h; its interference area
A(h) = (h - r) r sin w_h + r^2 (w_h/2 + sin(2 w_h)/4) defines the hot loop's indentation
coefficient **K_sp = F_re / A(h_min)** (continuous at h_min), unless the database gives an
identified value (src not `est`), which is then used. `ploughing_shape` gives F(h)/F(h_min).
**Indentation / extrusion** (chisel edges `EDGE_CHISEL`, normal rakes below -45 deg, speeds below
v_min = 0.05 m/s): contact to depth h including the rake face, material above h_min flowing up the
face; no shear-plane solution.
**Flank wear land** (Smithey, Kapoor & DeVor 2001): sigma_0 = (2 + pi) k, tau_0 = min(mu sigma_0, k)
plastic up to VB* = 0.1 mm (est), then ((VB - x)/(VB - VB*))^2: mean = sigma_0 (VB* + (VB - VB*)/3)/VB.
**Rubbing elements** (`EDGE_MARGIN`): no table forces; h_min = top of the h grid so the hot loop
always uses K_sp; heat partitioned as flank heat.

Error: the edge model is an upper-bound-type estimate. Against the ETH Ck45 data (r_e 33-44 um)
it over-predicts the thrust near h_min (h = 10-20 um: model 80-95 N/mm, measured 25-55 N/mm) and the
uncalibrated total thrust at 0.1-0.4 mm is 13-54 % low (section 9): the edge share is uncertain by
about a factor 2. Measured feed-force data (kf11) correct it through the calibration.

## 5. Oblique cutting (`oblique.py`)

Stabler (1951): chip flow angle eta_c = i. Equivalent (effective) rake:
sin a_e = sin eta_c sin i + cos eta_c cos i sin a_n (used for the chip form). Orthogonal-to-oblique
transformation (Budak, Altintas & Armarego 1996) with Armarego's equivalence (the orthogonal
solution at the normal rake a_n gives tau_s = k_AB, phi_n = phi, beta = lambda):

    tan beta_n = tan beta cos eta,  D = sqrt(cos^2(phi_n + beta_n - a_n) + tan^2 eta sin^2 beta_n)
    Ktc = tau_s/sin phi_n (cos(beta_n - a_n) + tan i tan eta sin beta_n)/D
    Krc = tau_s/(sin phi_n cos i) sin(beta_n - a_n)/D
    Kac = tau_s/sin phi_n (cos(beta_n - a_n) tan i - tan eta sin beta_n)/D

Frame: t along the cutting velocity (resisting force positive), r normal to the machined surface
(thrust positive pushing the tool off), a normal to both, the edge direction being
e_a = sin i t + cos i a. In the coefficient rows the map's orthogonal forces are multiplied by
Ktc(i)/Ktc(0), Krc(i)/Krc(0) and Kac(i)/Ktc(0) at the row's (tau_s, phi, lambda, rake), so i = 0
reproduces the map exactly. Chip velocity ratio V_c/V = r_c cos i / cos eta_c; chip width
b cos eta_c / cos i. Validity: Armarego's equivalence is used for i up to about 40 deg; it uses the
full speed V for the thermal state (the normal component V cos i would lower R_T by cos i).

## 6. Heat generation and partition (`heat.py`, Oxley state)

Zones per unit width: primary F_S V_s, rake F V_c (F_S V_s + F V_c = F_C V exactly, tested), edge
F_te V, flank tau_f VB V. Partitions:

* primary zone: beta to the work, 1 - beta to the chip (`frac_shear_to_chip` = 1 - beta);
* rake: R to the chip, 1 - R to the tool (`frac_rake_to_tool` = 1 - R), section 2;
* flank and edge (`frac_flank_to_tool`): Blok's equal mean temperatures,
  f = (T_ws - T_tf + q M_w)/(q (M_w + M_t)), with the work moving under the land (Jaeger 1942 band
  source, mean factor (1/(2 pi L^2)) int_0^{2L} e^u K0(u)(1 + e^{-2u})(2L - u) du, L = V l/(2 kappa),
  l the half-length; -> 0.752/sqrt(L) at large L; capped at low L by the stationary rectangle 2 A),
  the tool flank stationary (quarter-infinite, transient cap) and T_ws = T_AB (the machined surface
  leaves the bottom of the shear zone at about the shear-plane temperature, an upper bound).

`GroupCoefficients.power_frac_shear/rake/edge` split the tangential force power of a row into the
three zones (they sum to 1): edge share F_te/F_t (calibrated), the rest split by F_S V_s/(F_C V).
Stored cold work is neglected (all plastic work to heat, consistent with Oxley's calibrated
temperatures; the ledger term `stored_cold_work_J` stays 0 from this model).

Error: the moving-band/stationary cap is exact in both limits and over-predicts the mean rise by
at most about 20 % in between; Oxley's temperatures carry the uncertainty of the beta and dT_M fits
(stated by the extended-Oxley literature as about +-10 %); the code-to-code comparison reproduces the
published interface temperatures within 1-2 % (section 9).

## 7. Coefficient tables for cutting's pipeline (`coefficients.py`)

### 7.1 Process map
`build_process_map(work, tool, rake, speed_grid, work_temp, friction_factor, t1_grid, tool_temp_grid)`
solves section 2-3 on a grid (default 18 log-spaced t1 from 0.2 um to 2 mm, 6 speeds in
[V/3, 3V], tool temperatures 20/300/600/900 degC; one tool node for an adiabatic tool) for one
context (work material, tool, normal rake, workpiece temperature, mu). `lookup` interpolates
trilinearly in (log t1, log V, T_tool) (`kernels.map_lookup`); forces and lengths are
interpolated divided by t1 (linear in t1 below the first node), flags take the nearest node.
`map_matches` tests a map against a new context (cutting's rebuild policy); the context stores the
effective friction coefficient (database mu x friction factor, which identifies the friction state
in both interface modes) and the interface mode (`CTX_COULOMB`). Nodes without an admissible Oxley
solution are continued from the bisected edge of the admissible region (section 3), so a node does
not depend on the grid. Interpolation error midway between nodes (tested on a coarse grid, t1
factor 2.3): within 2 % for forces and temperatures in the admissible region, within 4 % next to
its edge, where the shear angle changes fastest with t1.

### 7.2 Group coefficients
`group_coefficients(work, tool, EdgeGroup, SlowState, h_grid, pmap, bue_map=None, calibration=None)`
returns rows over the h grid with the convention

    F_j(h) = K_j(h) h db + Ke_j dS,   j = t, r, a

(db chip width, dS edge length of the element), filled so the total equals the model at every node:
for h > h_min the shearing force at t1 = h - h_min (oblique, section 5) plus the edge forces; for
h <= h_min the ploughing force (continuous at h_min; the hot loop may use K_sp there instead).
Rows: `Kt_Pa, Kr_Pa, Ka_Pa`, `Kte/Kre/Kae_N_per_m`, `shear_angle_rad`, `chip_ratio`,
`friction_angle_rad`, `frac_shear_to_chip`, `frac_rake_to_tool`, `frac_flank_to_tool`,
`power_frac_shear/rake/edge`, `T_shear_K` (T_AB), `T_interface_K`, `T_chip_K`, `contact_length_m`,
`sticking_fraction`, `interface_shear_Pa`, `interface_normal_Pa`, `shear_strain`,
`shear_strain_rate_1_s`, `regime` (0 none, 1 Oxley, 2 Merchant, 3 indentation, 4 ploughing,
5 data-only), scalars `h_min_m`, `K_sp_N_per_m3` (+ source), `v_min_m_s`, `mu_eff`, flank
`flank_sigma0/tau0_Pa`, `flank_land_star_m`, means at the group's VB, and the ploughing shape.
**BUE**: with `bue_height_m` > 0 and a map at the BUE rake (`bue_rake_rad` = 25 deg, est) the rows
blend toward it in proportion min(1, H/(bue_full_height_ratio h)): the chip flows over the built-up
edge as over a tool face of larger effective rake (the effective-rake mechanism, Hutton & Yu 1990:
the BUE raises the effective rake and with it the shear angle), on work material, so Oxley's seized
interface holds exactly there. The single-condition solver offers the same model
(`oxley.solve(..., bue=True)` inside the material's database BUE window, or `bue_height_m=H`;
`oxley.bue_effective_rake`); it is off by default because the window is database data a caller must
account for (for Ck45 it comes from the ETH tests, the source of the Ck45 validation cases). The
fully developed BUE rake is an estimate: no measured BUE geometry could be read.

### 7.3 Calibration mode
`build_calibration(work, tool, group, process)` selects the material's measured data for the
pairing (tool class; coating match; process; measured sources ranked above `est`; mechanistic data
for milling when they score as well, else Kienzle; all entries of the best score, i.e. several
reference speeds). For each entry the reference conditions come from its `calibration` block,
else its `vc_ref`/`rake_ref` (Kienzle) or `rake`/`helix`/`fz` range (mechanistic), else the
group's rake, r_e = 30 um, 20 degC, dry; the tool at the work temperature (short tests). The
model's total forces at those conditions are computed with the same pipeline, and

* Kienzle: S_j(h) = measured / model on 7 log nodes over the calibration h range (cutting force
  from kc11, feed force from kf11 or the model's ratio, passive from kp11 or the model's ratio);
* mechanistic: the model's totals are fitted by F = Kc h + Ke over the range; S_c = Kc_meas/Kc,
  S_e = Ke_meas/Ke (S_e = S_c when the model edge force is negligible or S_e leaves the bounds).

Prediction: F(h; c) = S(h, V) F_model(h; c), S interpolated in log V between reference speeds and
held outside the measured h and speed ranges; factors are clamped to [0.2, 5] and the clamp is
reported. **No double counting**: the measured coefficients contain the softening, friction and
edge effects of their own test; the model only supplies the ratio F_model(c)/F_model(c_ref).

### 7.4 Data-only model (no flow-stress law)
Forces from the best Kienzle/mechanistic entry at t1 = h - h_min; shear angle from Merchant with
lambda = atan(mu); temperatures from the shear-plane power with Boothroyd's beta and Oxley's
chip fit at delta = `merchant_delta`; no rake partition (tool share 0); edge pressure from the
softened yield strength.

### 7.5 Assembly
`process_coefficients(groups, tool_slot, version, flank_coeff)` builds the bus record
`ProcessCoefficients` (cutting writes it); `coeff_table(pc)` its `core.abi.CoeffTable`;
`flank_coefficients(groups, elem_group, VB)` the per-element wear-land normal stress (N/m2).

## 8. Chip form and colour (`chipform.py`, `colour.py`)

Inputs: `ChipConditions` (h, b, V, rake, inclination, uncut chip length per tooth pass or None
for continuous cutting, feed and depth for the breaker window, engaged nose radius, process) and
`ChipState` (from a coefficient row or an orthogonal result).

* **Geometry**: t2 = h / r_c, width b cos eta_c / cos i, piece length = uncut length x V_c/V.
* **Curl**: natural R_n = 15 l_c (est: no relation for the natural up-curl on a flat rake face
  could be read; Nakayama 1962 treats only the curl a breaker imposes and Nakayama 1972 and Jawahir
  & van Luttervelt 1993 were not accessible); groove breaker R = (W^2 + 4 d^2)/(8 d) when the contact
  reaches the groove, else the back wall R = ((W_land + W - l_c)^2 + d^2)/(2 d) (Nakayama 1962;
  Jawahir & van Luttervelt 1993), only inside or above the insert's window (below it the breaker
  fails: natural curl, stringy chips); milling/drilling flutes R_f = sqrt(gullet area / pi);
  the tightest wins, never below t2.
* **Breaking** (Nakayama 1962, read): a chip of curl radius R breaks when its outer-fibre strain
  (t2/2)(1/R - 1/R_L) reaches the rupture strain eps_B of the chip material; R_L >> R (Nakayama's
  Sect. 2.2.6: tripling the shank height changed nothing), so t2/(2R) >= eps_B. eps_B is the
  material's measured chip breaking strain (`chip.breaking_strain`) when the file gives one,
  otherwise eps_B = r x bulk true fracture strain (min 0.005) with r from Nakayama's limiting
  conditions (Fig. 13; t2_lim/(2 R_c) with R_0 ~ R_c, Sect. 2.3): carbon steels S15C, S45C and S55C
  fall on one line, rupture strain 0.028-0.040 (mean 0.033) whatever the carbon content, so
  r = 0.033/0.60 = 0.055 with the 1045 true fracture strain of the material database; 18-8
  stainless SUS7: 0.019-0.028 (mean 0.022), r = 0.022/1.1 = 0.020 (its chips are less ductile
  than carbon-steel chips although the bulk is more ductile); Cr-Mo steel SCM3 0.036-0.047 (within
  the steel ratio's scatter for a 4140 with 0.8). Steel, tool steel: the steel ratio; stainless:
  its own; other families (not measured): the steel ratio, an assumption, times the database
  breakability factor 2 ('poor', est); 'good' (free-cutting grades, outside the tested steels) x 0.5
  (est) for every family. The relation is not a constant fraction across steel classes (factor 2.7
  between carbon and stainless steel), so eps_B of untested families is uncertain by about a factor 3.
  Broken chips end after 0.75 turns (up-curl) or 3 turns (side curl) (est); milling chips are
  separate pieces.
* **ISO 3685 class** (`classify`, codes in `core.bus.CHIP_FORMS`): ribbon when R >= 500 t2; arc
  (61 connected, 62 loose) below one turn; flat/conical spiral (31/32) without side curl
  (|eta_c| < 5 deg); with side curl washer-type helical (4x, pitch 2 pi R tan eta_c <= chip width)
  or tubular (2x), conical helical (5x) when the nose radius exceeds half the chip width or in
  drilling; subclass short (<= 5 turns or broken), long, snarled (>= 50 turns). Overrides:
  composites and foam dust (90, dust fraction from the database), polymers above their smear
  temperature (melt_smear, else Tg) molten/smeared (95), discontinuous/powder metals elemental (70),
  no chip below h_min (code 0, as the bus default).
* **Segmentation**: the database rule first (critical speed and onset thickness; tendency
  'segmented' without onset: always); `thermal_instability` or a temperature onset: **Recht's
  ratio** R = (dtau/dgamma)/(-(dtau/dT) dT/dgamma) < 1 with the JC derivatives at the AB state and
  dT/dgamma = f tau/(rho c), f = 1/(1 + 2 sqrt(kappa t_s)/Delta) the heat retained in the band of
  thickness Delta = l_AB/C0 during the shear time t_s = gamma_AB/gammadot (f -> 1: Recht's adiabatic
  limit). Segment pitch 0.8 h (est).
* **BUE**: window from the database (speed range and, if given, interface temperature range);
  `bue_advance` (kernel `bue_step`) relaxes the height exactly toward 0.8 h (growth 2 ms) inside,
  toward 0 (1 ms) outside, with break-offs at hazard 100 s^-1 x H/H_sat leaving 20 % (all est).
* **Ejection**: chip velocity relative to the tool V_c (cos eta_c up-the-rake-face - sin eta_c e_a)
  in the (t, r, a) frame; relative to the work add V t; kinetic energy per kg relative to the work.
* **Colour**: parabolic film growth x^2 = k_0 exp(-Q/RT) t (Wagner 1933): the chip's Arrhenius
  exposure int exp(-Q/(R T)) dt (contact hold + exponential cooling with tau = rho c t2 / (2 h), or
  a history from the chips domain via `exposure_from_history`) equals a hold of t_ref = 30 min at
  T_eq = (Q/R)/ln(t_ref/exposure); the material's oxide table (read as colours after t_ref) is
  evaluated at T_eq (linear sRGB blend between rows), bare metallic colours below its first row and
  for materials without a table. Q/R = 16 838 K (steels), 24 054 K (titanium), 18 041 K (others)
  (est). A +-20 % error in Q shifts the equivalent temperature of a millisecond flash by about +-40 K
  (about two table rows); the table temperatures themselves are uncertain by up to 100 K (titanium).

## 9. Validation and verification

| check | data | result | tolerance (test) |
|---|---|---|---|
| code-to-code, extended Oxley | AISI 1045, a = -7 deg, 200/300 m/min, 0.15/0.3 mm (Ning & Liang 2019 Table 2, Lalwani 2009 predictions) | F_C within 1 %, T_int within 2 % for 3 cases; 9-10 % for 300 m/min/0.3 mm (other delta optimum) | 3 % / 12 % |
| kernel vs SciPy reference solver | three conditions | F_C 0.3 %, phi 0.3 %, T 0.3 % | 0.3-1 % |
| Oxley's conditions at the solution | residuals | < 1e-6 relative; tau_int = k_chip to 1e-6; F_C V = F_S V_s + F V_c to 1e-12 | |
| friction is an output | database mu 0.5 -> 0.2 | rake-face solution unchanged (Coulomb alternative: lambda = atan 0.2) | exact |
| fallback grid independence | map node against oxley.solve, two grids | phi and F_C within 1 % | 1 % |
| ETH Ck45 orthogonal (Klippel et al. 2022, 15 conditions, 30-500 m/min, 0.04-0.4 mm, r_e 39 um), uncalibrated | 12 Oxley-regime points | F_C -20 to -7 % (mean abs 16 %); F_f -32 to -13 %; chip thickness -20 to +43 % | -25/+10 %, -40/+20 %, 50 % |
| same | 3 fallback points (30 m/min in the BUE window; 100 m/min 0.06 mm; 200 m/min 0.04 mm) | F_C -2 to +52 %, F_f 0 to +160 %, chip thickness +9 to +209 % (30 m/min) | wide, documented |
| same, calibrated to the 100 m/min Kienzle fit (with kf11 809, mf 0.65) | | Oxley points F_C -9 % to +9 %; fallback points +6 to +25 %, 30 m/min +79 % | 15 % / 80 % |
| ETH Ti6Al4V (Klippel et al. 2022, 7 conditions, 20-150 m/min) | uncalibrated (Seo JC) | F_C +32 to +95 %, F_f +34 to +201 %, chip thickness +46 to +215 % | 0/+50 % (Oxley points), 0/+100 % (fallback) |
| same, calibrated to five measured Kienzle speeds | | F_C -5 to +9 %, F_f -10 to +21 % | 12 % / 25 % |

Former Coulomb-limited model on the same ETH data (for comparison): Ck45 F_C -31 to -6 %, F_f -54 to
+9 %; Ti6Al4V F_C +12 to +24 %, F_f +5 to +22 %. The Ti6Al4V agreement came from mu = 0.3, an
Albrecht estimate from the same ETH tests: with Oxley's seized interface a continuous-chip theory
over-predicts the segmented Ti6Al4V chip (thicker chip, higher forces); the theory does not describe
segmented chips (section 13).

Validation suite (`twin validate`, docs/validation-report.md; harnesses unchanged, independent cases;
before = Coulomb-limited interface with the database mu, after = this model), 2026-11-03:

| quantity | before: median / p90 abs. error, cases in budget | after | what moved |
|---|---|---|---|
| chip_ratio (Ck45 x 2, AISI 304) | 19.6 % / 77.4 %, 0/3 | 26.4 % / 49.1 %, 0/3 | Ck45 150-500 m/min and the 150 m/min feed series improved (e.g. 0.02 mm +206 -> +43 %, 0.06 mm +81 -> +3 %, 0.1 mm +53/+67 -> +15/+27 %); Ck45 10-70 m/min (no admissible solution; BUE window 10-50 m/min) -21 to -73 % (was -30 to +100 %); AISI 304 at 13-60 m/min mostly fallback: -54 to +22 % (was -19 to +96 %, a constant 0.43-0.46 set by mu = 0.6) |
| shear_angle (AISI 304 B, Ti6Al4V) | 21.7 % / 45.4 %, 0/2 | 35.1 % / 48.1 %, 0/2 | 304 B (13-67 m/min, fallback) -19 to -52 % (was -18 to +46 %); Ti6Al4V -14 to -54 % (was +11 to +47 %): segmented chips |
| chip_temperature (Ck45 temper colours) | 17.4 % / 63.8 %, worst 3.48 tolerances | 20.2 % / 44.4 %, worst 2.43 tolerances | 10-30 m/min -48/-20 % (was -70/-50 %), 50-70 m/min +20 to +36 % (was -11 to -25 %), 150 m/min +4 % (was -11 %), 250-500 m/min unchanged (already sticking) |
| tool_temperature (uses the coefficient pipeline) | 52.8 % / 86.8 %, 1/5 | 65.3 % / 120.6 %, 0/5 | Ti6Al4V natural thermocouple (Loeschner 2025) 6.6 -> 30.9 % (the seized interface heats the Ti6Al4V contact more than the former mu = 0.3 limit), AISI 4140 IR/TC 76.8 -> 96.2 %, Ti6Al4V IR window 45.1 -> 32.7 %, AISI 1045 embedded TC 4.9 -> 3.6 % |
| coolant_reach (friction factor in Oxley's condition) | 38.6 % / 85.4 %, 10/18 classes | 64.5 % / 105.5 %, 10/18 | Ti6Al4V jet-direction case 38.8 -> 79.3 % |
| chip_form (Ti6Al4V, breaker inserts not in the database) | 6/32 classes | 5/32 | chips now break (bend 0.026-0.030 against eps_B 0.027 instead of 0.032 against 0.049) but the natural curl radius (4-20 mm) leaves pieces of 80-375 mm: the inserts' SF/SMC grooves decide the measured grades |

Why the tolerances of the tests: the uncalibrated error combines the JC-set uncertainty (other
published 1045 sets change predicted forces and temperatures by 10-20 %, Ning & Liang 2019 Table 5),
the estimated h_min ratio and edge pressure, and the known thrust deficit of shear-plane models with
honed edges; the calibrated Ck45 error is bounded by the Kienzle fit's own deviation from the points
(up to 25 % at small h). Oxley's book tables and the CIRP/NIST AMM benchmark (Ivester et al. 2000)
could not be read; the ETH database is open, recent and the source of the material files' Kienzle
data.

## 10. Kernels, signatures and measured cost

All kernels: `@njit(<signature>, cache=True, nogil=True, error_model="numpy")`, no fastmath, no
parallel, C-contiguous arrays; the tests compare each with its `py_func` (plain Python) and the
lookups with SciPy/NumPy, and assert `len(signatures) == 1`. Slots in `mechanics/abi.py`
(`MAT_*` material, `TOOL_*` tool side, `P_*` parameters, `OX_*` outputs, `PL_*` edge contact,
`CTX_*` map context). Cost on this box (one core, warm cache), called from Python:

| kernel | signature | cost |
|---|---|---|
| `oxley_point` | `(case f8[4], mat f8[9], cp_T, cp_v, k_T, k_v, tool f8[5], kt_T, kt_v, ct_T, ct_v, prm f8[39], out f8[40])` | 1.9 ms (adiabatic) - 4.8 ms (with partition) |
| `oxley_map` | `(t1 f8[nh], v f8[nv], Tt f8[nT], alpha, Tw, mat..., prm, values f8[nT,nv,nh,40])` | 3.2 ms per node; `build_process_map` with the edge refinement of its fallback nodes: 1.7-1.9 s for the default 432-node map (AISI 1045 at 30 and 150 m/min: 308 and 235 fallback nodes; 1.4 s before the refinement) |
| `merchant_point` | as `oxley_point` | 8 us |
| `map_lookup` | `(t1, v, Tt, values, scaled i8[:], discrete i8[:], t1_query f8[n], speed, ttool, out f8[n,40])` | 16 us for 24 rows |
| `interp_log_h` | `(h_grid, row, h) -> f8`, linear in log h | 0.46 us from Python (about 20 ns inside a kernel) |
| `interp_lin_h` | `(h_grid, row, h) -> f8`, linear in h, no transcendental (L0-safe) | as above |
| `edge_contact_forces` | `(depth, radius, stag, rake, incl, p, tau, out f8[5])` | 0.57 us |
| `oxide_exposure` | `(temps f8[n], dt, Q/R) -> f8` | 10 ns per sample |
| `bue_step` | `(H f8[n], window f8[n], uncut f8[n], dt, prm f8[5], seed, stream, key i8[n], counter i8[n], broke i8[n])` | 21 ns per element |
| `jc_flow_stress`, `jc_hardening_index`, `beta_partition`, `rect_area_factor`, `tool_resistance`, `interp1`, `brent_init`, `brent_step`, `shear_zone_rise`, `solve_shear_plane_temp`, `shear_plane_state`, `solve_strain_rate_constant`, `interface_state`, `phi_root_on_grid`, `blend_state`, `fill_output` | see `kernels.py` | internal to the solver |

Python entry points: `oxley.solve` 3.9 ms with an admissible solution, 37 ms with the edge
continuation of the fallback (ladder + 8 bisections), `group_coefficients` (24 h nodes) 1.1 ms,
`build_calibration` 61 ms (27 ms before the fallback refinement; timings of 2026-11-03 with the box
at a load of about 3.5 on 4 cores),
`chip_form` 45 us (300 us with the cooling colour). Compilation on a cold cache: about 12 s (eager,
at import). Nothing here runs at L0: the hot loop reads the `ProcessCoefficients` rows.

## 11. Determinism and RNG streams

No stateful RNG. BUE break-offs: `core.rng.hash_uniform(seed, stream_key("mechanics.bue_breakoff"),
element_key, counter, 0)` with the per-element counters kept by the caller (wear) and incremented by
the kernel (the caller may pass its own stream name). All solver grids, iteration counts and
tie-breaks are fixed.

## 12. Bus fields (the library writes none; callers pass these values)

| input | from |
|---|---|
| local speed, rake, inclination, clearance, edge radius, kind | `core.abi.EdgeModel` (tooling), motion |
| tool rake / flank temperature | `ThermalFeedback.tools[slot].rake_temp_mean_K`, `flank_temp_q_K` |
| work temperature | `ThermalFeedback.work_preheat_K` |
| friction factor | `CoolantFeedback.friction_factor[contact]` |
| VB, BUE height, edge radius growth | `WearFeedback.tools[slot].VB_m`, `bue_height_m`, `edge_radius_m` |
| contact width / time | `RemovalFrame` contact geometry, cutting's engagement |
| outputs | cutting writes `ProcessCoefficients` (`process_coefficients`); chips fills `ChipFormationContact` (shear angle, chip ratio, shear strain and rate, flow angle, chip temperature, form code, bue_active) and `EmissionBatch` (thickness, width, length, curl radius, form code, colour, segmented, velocities); wear integrates `bue_advance` |

## 13. Assumptions and limitations (with error estimates)

* Continuous-chip shear-plane theory with Oxley's seized interface; segmented chips (Ti, hardened
  steels, high speeds) are flagged but not modelled: for Ti6Al4V the theory gives chips 1.5-3 times
  thicker and forces 30-95 % higher than measured uncalibrated (calibrated to measured Kienzle data:
  within 10 % in F_C).
* Validity region of Oxley's condition with Johnson-Cook laws: no admissible solution at low speeds
  and small t1 (validation conditions: AISI 1045 below about 100 m/min at h = 0.1 mm and below
  h = 0.05 mm at 100-200 m/min; AISI 304 below 40-60 m/min at h <= 0.2 mm and below about 30 m/min at
  0.43 mm; Ti6Al4V below 50 m/min at h = 0.1 mm and at h <= 0.08 mm up to 70 m/min). There the fallback (section 3)
  carries the edge state of the theory: thick chips (chip ratio 0.19-0.30 against 0.24-0.68 measured
  for Ck45 and 304) and forces up to +50 %; for steels this region is largely the BUE region.
* BUE: a BUE-free solution unless the caller supplies a BUE height or asks for the database window
  (`bue`); the effective rake of a fully developed BUE (25 deg) is an estimate; inside the AISI 304
  window it over-thins the chips of small feeds (chip ratio +60 to +140 % at h = 0.06-0.1 mm).
* Interface temperatures at high speed (AISI 1045, 250-500 m/min) make the chip 25-38 % too thin;
  the Jaspers & Dautzenberg set's linear thermal softening (m = 1) is the likely cause (Burns et al.
  2012 report rapidly heated 1045 stiffer than this law below the eutectoid).
* Sharp-tool Oxley + additive edge model: thrust uncertain by about a factor 2 near h_min.
* Armarego's equivalence for oblique cutting uses the full cutting speed for the thermal state
  (normal-speed effect below cos i on R_T: < 15 % in R_T for i < 30 deg).
* Tool-side partition: steady quarter-infinite source capped by the 1-D transient; the tool rake
  temperature passed in must exclude the sub-grid flash of the current contact (else the flash is
  counted twice; bounded by the flash rise times the fraction thermal resolves).
* T_ws = T_AB for the flank partition (upper bound; flank tool share therefore a lower bound).
* Chip breaking: the rupture strains are Nakayama's for carbon and 18-8 steels; other families take
  carbon steel's fraction of the bulk fracture strain (factor 3 uncertain); the natural up-curl
  radius (15 l_c) is an estimate, and without the insert's groove geometry in the database the
  chip-form classes of breaker inserts are not predicted (5 of 32 Ti6Al4V grades match).
* Estimated parameters (all `src: est` in db/models/mechanics-*.yaml): Zorev exponent, VB*,
  v_min, min shear rake, BUE rake and dynamics, natural curl ratio, breakability factors, turns to
  break, form thresholds, oxide activation energies and reference time, numerical grids and the
  fallback continuation steps.
* Data-only materials: forces as good as the catalogue Kienzle values (typically +-20-30 %);
  temperatures from Merchant geometry (about +-30 %).

## 14. References

Adibi-Sedeh, Madhavan & Bahr 2003; Albrecht 1960; Armarego & Whitfield 1985; Blok 1937; Boothroyd
1963; Budak, Altintas & Armarego 1996; Carslaw & Jaeger 1959; Hastings, Mathew & Oxley 1980 (not
read); Hill 1950; Hutton & Yu 1990 (abstract); Jaeger 1942; Jawahir & van Luttervelt 1993 (not
read); Johnson & Cook 1983; Klippel et al. 2022 (ETH database); Komanduri & Hou 2001; Lalwani, Mehta
& Jain 2009; Lee & Shaffer 1951; Loewen & Shaw 1954; Merchant 1945; Nakayama 1962 (read: criterion,
contact length, limiting chip thicknesses of five steels); Ning & Liang 2019; Oxley 1989 (equations
as reproduced by Lalwani 2009 and the ExtOxley code); Pantale et al. 2022 (article and software);
Recht 1964; Shaw, Ber & Mamin 1960; Smithey, Kapoor & DeVor 2001; Stabler 1951; Tabor 1951; Wagner
1933; Zorev 1963; ISO 3685:1993. Full entries: `db/bibliography/models-mechanics.yaml`.
