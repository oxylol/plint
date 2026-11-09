# Wear: tool wear, damage, life and breakage

`cnctwin/wear/` is the **wear domain** (architecture 2, 6, 7, 19): an L2 slow domain that takes no part
in the hot loop. Per macro step and tool slot it turns cutting's per-element integrals and shocks and
thermal's edge temperatures into the wear state of every edge element, decides chipping, coating
spallation, thermal cracking, section breakage and the end of the tool's life, and publishes
`WearFeedback` (`nxt.wear`). Coefficients: `db/models/wear-{flank,calibration,fracture,life}.yaml`;
sources: `db/bibliography/models-wear.yaml` (plus the keys already in other bibliography files).

| module | content |
|---|---|
| `abi.py` | kernel layouts (`ElemGeo`, `ElemIn`, `Law`, `ElemState`, `ElemOut`, `ChipLaw`), slot constants, hash keys |
| `kernels.py` | Numba kernels: `advance_elements`, `element_outputs`, `chip_trials` and helpers |
| `reference.py` | plain-Python twins of every kernel (the tests compare them) |
| `params.py` | model files, database data as SI values (tool material, coating, work material, Usui, Taylor) |
| `calibration.py` | Usui constants calibrated to the Taylor life, published sets anchored to it; `TaylorResponse` protocol, `MechanicsResponse` default (flank land softened at its temperature) |
| `fracture.py` | Weibull strength and size effect, section criteria, Basquin curves, crack strength |
| `build.py` | per-slot construction at init: geometry, wear law, chipping law, sections, fallback table |
| `step.py` | one macro step of one slot: bus inputs, kernel inputs and fallbacks, BUE, chipping, sections, criteria, record |
| `life.py` | life criteria, Ra estimate, bias-corrected moving averages, remaining life |
| `protocols.py` | `ToolInfo` (+ `GeometryToolInfo`, `StaticToolInfo`), `SkinQuery` (+ `StockShell`), `ToolTolerances` (+ `ProjectTolerances`) |
| `state.py`, `schema.py`, `channels.py`, `domain.py` | SimState field (checkpoint), `physics.wear`, channels, `WearDomain` |

All quantities are SI; temperatures in kelvin. Per-element arrays are `[n_teeth, n_elem]` of the tool's
nominal EdgeModel (core/abi.py), the shape of every per-element array on the bus.

---------------------------------------------------------------------------------------------------

## 1. Inputs

* **Cutting** (`prev.cutting.contacts`, summed over the contacts of a slot): per element `in_cut_time_s`,
  `flank_sliding_m`, `rake_sliding_m`, `int_sigma_v_flank`, `int_sigma_v_rake` (integral of the contact
  normal stress times the sliding speed, Pa m), `int_h_dt`, `entries`; `shocks` (rows `[tick, tooth, elem,
  entry_h_m, entry_angle_rad, force_first_full_N, loading_time_s, kind]`, kinds 1 entry, 2 exit,
  3 breakthrough, 4 breakout as written by cutting); `section_moment_peak_Nm` (per critical section,
  ascending z, the count of tooling's sections), `section_stress_peak_Pa` (reported), `torque_peak_Nm`,
  `force_peak_M_N`, `wrench_tool_T0_peak` (tick of the peak), `surface_events` (tooth, element and tool
  slot of the elements that leave final surface).
* **Thermal** (`prev.thermal.tools[slot]`): `flank_temp_q_K` and `edge_temp_q_K` (flank and rake contact
  means at the Gauss points of the in-cut interval, `[nt, ne, nq]`), `q_weights` (sum 1),
  `edge_cycle_amp_K` (edge-line range of one heating/cooling cycle), `body_temp_at_elem_K`, `body_temp_K`.
* **Coolant**: `on`, `friction_factor[contact]` (BUE, section 6). **Chips**: `packing_torque_Nm[slot]`,
  `packing_force_T0_N[slot]` (section 9). **Motion**: `tick`, `paths[*]` (`path`, `tool_slot`, `tick`,
  `spindle_omega_rad_s`, `feed_vel_M_m_s`, `pose_S_T`).
* **Databases** (through `ctx.db` or the `catalog` argument): the tool material (density, E, nu,
  hardness - hot-hardness table when given -, TRS, K_IC, Weibull modulus, grain size, CTE,
  conductivity, specific heat, max service temperature, `usui` list), the coating (wear-rate factor,
  thickness from the tool item else the mean of `typical_thickness_um`, hardness, max service
  temperature), the work material (family, `tool_life` Taylor entries with `T_ref`, `life_basis` and
  `engaged_arc_ref_deg` (section 3), `tool_life_usui`,
  `skin`, `composite.abrasive_wear_factor`, `cutting.work_hardening`, friction, BUE window via
  `cnctwin.mechanics.WorkMaterial`). **Project**: `tools.list[i]` (`initial_wear.VB_mm`,
  `initial_wear.edge_radius_um`, `life.used_min`, `life.limit_min`), `cad.tolerances`, `stock`,
  `environment.ambient_degC`, `physics.wear`.
* **Usui constants** are read from the file's unit system, explicitly: `A_per_MPa` with its `rate_unit`
  (`cnctwin.db.schemas.USUI_RATE_UNITS`: mm/min with MPa and m/min -> x 1e-9, mm/s with MPa and mm/s ->
  x 1e-6) or `A_per_Pa` in SI rates; an `A` without unit system, or `A_per_Pa` with a non-SI rate unit,
  is an input error. `T_range_K` (open ends allowed) makes temperature branches.

## 2. Flank wear (Usui with quadrature, Archard, coating, skin, BUE)

Per element and step (Usui, Shirakashi & Kitagawa 1984; Takeyama & Murata 1963 decomposition into a
thermally activated and a mechanical term; Archard 1953):

    dVN = m_skin m_bue [ sum_q w_q f_c(T_q) A_b exp(-B_b / T_q) + sum_q w_q K a / H(T_q) ] int sigma_n V_s dt
          + n_breakoff d_pluck

* VN is the wear depth normal to the machined surface (the flat land moves into the tool, so the
  land-average Usui rate is dVN/dt: Yen et al. 2004); `int sigma_n V_s dt` is cutting's flank integral, the
  land's normal stress (the row's: at most the Prandtl pressure (2 + pi) k of the work's flow stress at the
  contact) times the sliding distance the face actually travels (|V| dt: an edge a vibration drives backwards
  still slides on the work; before 2026-11-09 a negative V took wear away).
* **The contact state bounds the temperature** (2026-11-09): the tool's contact faces cannot be hotter than the
  interface they draw their heat from, and the interface no hotter than the work material's melting (its solidus,
  else liquidus, else its flow law's melting temperature; `props["interface_max_K"]`): beyond it there is no solid
  contact to wear the tool. Thermal's quadrature temperatures above it are taken at it for the flank and the
  crater laws (trace `wear:edge_temperature_above_work_melting`, once). A blown-up entry's 3296 degC estimate had
  extrapolated Usui's Arrhenius factor by 1e11: VB_max 146 m (notch 22.6 m) in thin-wall-al7075's first
  milliseconds. Test `test_an_edge_hotter_than_the_work_melts_wears_at_the_melting_temperature`.
* **Quadrature**: the rate is evaluated at each in-cut quadrature temperature T_q and weighted by
  w_q, never at the mean temperature (Jensen: the hot part of the interval dominates; test
  `test_quadrature_is_not_the_rate_of_the_mean_temperature`: at 900/1000/1100 K with weights 1/4, 1/2,
  1/4 and B = 2e4 K the quadrature is 2.07 x the rate of the mean temperature).
* **Branches**: A_b, B_b of the branch whose range holds T_q (sorted by the lower end; below every
  range the first).
* **Coating**: f_c = coating `wear_rate_factor` while coating covers the contact and T_q is at or below
  the coating's max service temperature (oxidation onset), else 1. The flank coating at the edge is
  consumed by dVN cos(a)(1 - tan(a) tan(g)) (the depth of the worn edge normal to the coated flank face);
  the abrasive term uses the coating's hardness while it is there. Substrate law when it is gone.
* **Abrasive (Archard) term**: K = `abrasive_coeff_ref` (2e-8, the reference work material AISI 1045,
  est) x a, a = the material's `composite.abrasive_wear_factor` for composites, 1 otherwise; H(T) the
  tool's hardness (HV x g x 1e6 Pa; the database's hot-hardness table when given, else one value).
* **Skin**: m_skin = 1 + phi (a_skin - 1), phi the share of the element's cut in the stock's skin layer
  (SkinQuery), a_skin the material's `skin.abrasiveness_factor` ("wear-rate multiplier while cutting the
  skin").
* **BUE** (section 6): m_bue = 1 - p min(1, H_bue / h), p = `bue_protection` (0.5, est); each break-off
  plucks d_pluck = grain size x `bue_pluck_fraction` (3e-4, est).
* **Land and recession**: VB = (VN + d_chip)(cot a' - tan g), a' = max(clearance, 2 deg)
  (yen-2004-tool-wear-fem land geometry; a zero-clearance land has no VB); total recession
  R = VN + VN_notch + d_chip; radial wear R sin(kappa), axial wear R cos(kappa) (tooling's signs: radial
  > 0 recedes the edge from the material, axial > 0 along +z_T).
* **Edge rounding**: r_e = r_e0 + `edge_rounding_ratio` (1, est) (VN + VN_notch); a chipped edge at least
  half the chip depth.

Validity: continuous and interrupted cutting with in-cut times resolved by thermal's quadrature;
Usui's law for diffusion/adhesion-dominated wear of carbide, cermet, ceramic, PCBN and HSS above
about 0.5 of the activation range of the calibration; the Archard floor below it.

## 3. Calibration to the Taylor life (`calibration.py`)

When neither the tool material nor the work material give Usui constants for the pairing
(selection: face rake/flank/both, work family or `any`, tool class, best coating match - a set for the
tool's own coating is turned into the substrate law by dividing A by the coating factor), A and B are
calibrated to the work material's extended Taylor law `vc (T/T_ref)^n (f/f_ref)^a (ap/ap_ref)^b = C`
at its VB limit:

1. Entry: same tool class; the tool's coating (`same`: the calibration models the coating phase with the
   tool's coating data), else an uncoated entry (`substrate`: the substrate law; the run then adds the
   coating phase), else another coating (`other`: a generic coating phase, factor 0.6, 3 um PVD /
   10 um CVD by keyword, est); best process match; then the entry whose `grade` names the tool material
   (`params.grade_score`: words of the grade text found in the tool material's id and name, so a
   whisker-reinforced alumina takes the `Al2O3-SiC whisker` entry and a SiAlON the `SiAlON` one of the same
   class); first in file order.
2. Speeds: the entry's `vc_range` ends, else the speeds of Taylor lives 60 min and 5 min
   (`calibration_life_long_s`, `calibration_life_short_s`; ISO 3685 test practice), at the entry's feed
   and depth references; check speed: the 15-min life (v15).
3. The life integral T(A, B; v) = int_{VB0}^{VB_lim} dVB / (g dVN/dt) by Gauss-Legendre
   (`physics.wear.calib_nodes`, 6 per interval), split where the coating wears through, with the
   response (`TaylorResponse.response(slot, Condition)`: sigma_n V_s and the flank temperatures at
   quadrature points, at the node's land VB) evaluated at every node. The integral is **edge contact
   time**; its targets are the Taylor lives converted to contact time (life basis, below).
4. For each B, A follows from T(v1) = eta_ref T_Taylor(v1) (closed form; a monotone root with the abrasive
   term); B is the root of ln(T(v2) / (eta_ref T_Taylor(v2))) in [10 K, 1e5 K] (Brent, fixed tolerances). The Usui
   factor is evaluated as exp(-B (1/T - 1/T_ref)) with A' = A exp(-B/T_ref) so large B neither
   underflows nor overflows.
5. The abrasive term carries at most `max_abrasive_share` (10 %) of the Taylor rate at the calibration
   speeds (scaled down otherwise); if both lives still cannot be met it is quartered (up to six times)
   and finally removed. Cached in-process by every input that determines the result, stored in the state.

**Life basis** (`tool_life[].life_basis`, `engaged_arc_ref_deg`; `params.life_basis_of`). The run advances
each element by its own in-cut (contact) time (`in_cut_time_s`, section 1), so the law is a contact-time
law, while a Taylor life T counts one of:

| `life_basis` | T counts | default for |
|---|---|---|
| `cutting_time` | the time the tool feeds in the material (ISO 3685 and ISO 8688 tool life, makers' catalogue lives) | `milling`, `drilling`, `tapping`, `any` and entries without a process (catalogue data) |
| `contact_time` | the time the edge is in contact with the material | `turning`, `boring`, `grooving`, `threading`, `orthogonal` (continuous cuts: equal to the cutting time) |

The calibration targets eta_ref T_Taylor, eta_ref the contact fraction of the entry's reference cut:
1 for a contact-time life and for continuous cuts (turning, boring, grooving, threading, drilling lips,
tapping flanks: the edge stays in the cut while the tool feeds; an entry without a process is read as a
continuous reference cut); phi_ref / 2 pi for a milling life counted in cutting time, phi_ref the arc of
each revolution the edge is in the material in the reference cut: `engaged_arc_ref_deg` (side milling
arccos(1 - 2 ae/D), centred face milling 2 arcsin(ae/D), slot 180 deg), else the ISO 8688-2 reference
end-milling test (Table 3, conditions I/II: ae = 2.5 mm with the D = 25 mm cutter, ae/D = 0.1,
phi_ref = arccos(0.8) = 36.87 deg, eta_ref = 0.102; `milling_reference_arc_rad` in wear-calibration.yaml;
listed in `estimated`). In the run an element engaged phi_e / 2 pi of each revolution (cutting's per-element
in-cut time over the tool's cutting time) wears in contact only, so it reaches the VB limit after
T_Taylor phi_ref / phi_e of cutting time: **at the reference engagement a milling tool reaches the Taylor
life in cutting time**, at a lighter engagement it lives longer (and thermal's lower interrupted-cut
temperatures lengthen it further), at a heavier one shorter. Tests (`test_life_basis.py`, whole-domain
runs with the response's temperatures): a milling cut with a 90 deg reference arc (each element in
contact 1/4 of the cutting time) reaches VB_lim at T(vc) of cutting time within +0.3 % at both
calibration speeds (400 macro steps per life: the step lag of the land-dependent temperature) and +2.6 %
at the 15-min check speed; the same cut with the elements in contact all the time in 1/4 of it; a turning
cut (contact basis) +0.3 % and +4.3 % (the abrasive floor carries a larger share of the slower contact-time
law). Documented bound: the calibration error below (6 %).

**Default response** (`MechanicsResponse`, cnctwin.mechanics): Oxley with Johnson-Cook at t1 = h - h_min
(adiabatic tool) for the shear-plane temperature T_AB, the rake-contact temperature T_int, sigma_N and
the chip speed; Smithey's land stresses from sigma_0 = (2 + pi) k_AB, tau_0 = min(mu sigma_0, k_AB);
flank temperature by Blok's partition with Jaeger's band source on the work and the stationary source on
the tool flank (cnctwin.mechanics.heat), the tool-side base temperature T_AB + 0.5 (T_int - T_AB)
(`flank_base_weight`, est). The land's friction stress is the work's flow stress, so where the land runs
hotter than the shear plane it is softened at the land's own temperature, tau(T) = tau_0 x min(1, theta(T) /
theta(T_AB)) with theta the flow stress's thermal factor (Johnson-Cook 1 - T*^m; the data-only model's linear
fall), and the flank temperature is the fixed point of Blok's balance with that flux (`flank_temperature`,
bisection), as Oxley fixes T_int with the chip's softened flow stress: it stays below the melting temperature of
the law (without it the response gave 1830-2200 degC flanks for IN718 with ceramics at 150-250 m/min, above the
alloy's melting range, and 1475-1600 degC for Ti6Al4V at 150 m/min; now 900-1060 and 960-990 degC). The pressure
sigma_0 is not softened (the hot layer, sqrt(kappa VB / V) of about ten microns, is thin against the plastic
zone under the land). `Condition.coated` False models the uncoated substrate.

**Published sets and the pairing's Taylor life** (`calibration.anchor_published`). A published Usui set holds
for the temperatures its source fitted it with (often an FE model): a systematic difference dT of another
temperature model changes its rate by about exp(B dT / T^2). When the work material has a Taylor entry for the
pairing (selected as in step 1), the published set keeps its B and branch ranges and its A (all branches by one
factor) is scaled so that the response reaches the entry's VB limit in the entry's contact life at the 15-min
speed v15 (with the coating phase as in the calibration); the residuals at the entry's two calibration speeds are
reported (`law.flank.anchor`), the entry's VB limit becomes the tool's limit, and the original A is kept in the
report. Without a Taylor entry the set is used as published. Example: Zhang 2025 (carbide-k20-6co on Ti6Al4V,
B 15 909 K) anchored to Batista-Ponce 2019 (ti6al4v tool_life[0], f 0.2 mm, ap 1 mm, 15 min at 69 m/min):
A x 0.041; residuals -20 % at 50 m/min and +150 % at 200 m/min (the source's adhesion-driven failure makes the
Taylor exponent steeper than the published B with this response's temperatures).

A calibration whose B reaches the end of its bracket (the response's flank temperature changes too little
between the two speeds for the entry's speed exponent, e.g. IN718 with ceramics, whose flank runs near the
melt-limited temperature) is listed in `estimated` with its residual.

**Materials without a flow-stress law** (composites, polymers, wood, foam): the mechanics data-only
model's temperatures come from Merchant geometry on a material that forms no chip, so a Usui fit would
not match the run's thermal temperatures; these wear by the abrasive term with the database
abrasiveness factor, and their Taylor data are reported as a check (`law.flank.taylor_check`), not fitted.

**Measured** (database materials of 2026-11-09, default response, carbide uncoated entries, rake 0.1,
clearance 0.12, hone 20 um, abrasive floor included): residual at both calibration speeds < 1e-9; at the
15-min check speed:

| material / tool material | B (K) | flank T at v1..v2 (K) | check residual |
|---|---|---|---|
| aisi1045 / carbide-p30 | 30 664 | 897-949 | +4.3 % |
| aisi4140-qt30 / carbide-p30 | 28 217 | 908-962 | +1.3 % |
| aisi304 / carbide-k20-6co | 30 087 | 890-939 | +1.4 % |
| ti6al4v / carbide-k20-6co | 27 755 | 1062-1205 | -3.0 % |
| inconel718 / carbide-k20-6co | 66 079 | 1030-1065 | +0.5 % |
| al6061-t6 / carbide-k20-6co | 19 217 | 575-603 | +5.3 % |
| grey-cast-iron-gjl250 / carbide-k20-6co | 21 913 | 715-763 | +1.4 % |
| aisi1045 / hss-m2 | 50 227 | 774-802 | +2.8 % |

(Recomputed 2026-11-09 with the softened land; the uncoated entries of the tool class picked by step 1; the
flank temperatures are those at the first Gauss node, a land of about the hone radius, which the softening hardly
changes; it lowers the hot end of the land and so raises B, most for IN718, whose flank approaches the
melt-limited temperature. Before: B 24 843, 26 863, 47 526, 16 518, 25 877, 18 700, 13 034, 30 628 K.)

**Error bound**: within the calibrated
window (5-60 min) the wear model reproduces the Taylor life within 6 % (the curvature of B / T(ln v)
against Taylor's straight line; synthetic response test bound 5 %, kernel integration to the VB limit
within 0.5 % of the integral, test `test_the_kernel_reaches_the_vb_limit_in_the_taylor_life`). This is
relative to the Taylor fit, whose own scatter is +-20-30 % and whose database entries are mostly
estimates. Outside the window the extrapolation follows the Arrhenius law; at 3 x the speed span the
curvature term grows to about 20 %. The calibrated constants hold only for the model temperatures they
were fitted with: when the run's thermal model gives systematically different flank temperatures than
the response, the life shifts by exp(B dT / T^2) (dT = 20 K at 950 K and B = 25 000 K: x 1.7) -
wire a response that runs cutting's coefficient pipeline and thermal's edge model (integration note).

## 4. Crater (Takeyama-Murata / Arrhenius)

    dKT = m_skin m_bue [ sum_q w_q f_c,r(T_r,q) A_r exp(-B_r / T_r,q) + sum_q w_q K a / H(T_r,q) ] int sigma_n,rake V_c dt

with the rake (crater-zone) contact temperatures at the quadrature points, cutting's rake integral, and
the rake coating consumed by dKT. Constants: the published rake set (`surface: rake`/`both`), else the
flank law (Usui et al. 1984 predicted crater and flank wear with one characteristic equation). The
thermally activated term is the diffusion term of Takeyama & Murata in Usui's stress-velocity form;
published high-temperature branches (P20/steel above 1150 K: B = 21 950 K, ~180 kJ/mol) make the
diffusion onset explicit. Onset: the crater depth reaching `crater_onset_m` (1 um, est) writes
`wear:crater_onset` (test `test_crater_rate_follows_the_arrhenius_onset`: the rate rises by more than
1000 x from 700 K to 1300 K; the two-branch jump at 1150 K equals the ratio of the branches).
Error: without crater data, the crater carries the flank calibration's uncertainty plus the rake/flank
temperature difference of the models (a factor 2-3 in depth).

## 5. Notch at the depth-of-cut line

Per tooth, contiguous runs of elements in cut; an end of a run is a depth-of-cut (DOC) line when the
edge continues into air there (the next element is valid and not in cut) and its mean chip thickness
is at least `doc_h_ratio` (0.5) x the run's largest (the other end tapers toward the machined surface).
At a DOC element dVN_notch = dVN ((1 + dH)^q - 1), dH the material's `work_hardening.hardness_increase`,
q = `notch_hardness_exponent` (2, est: IN718's 48 % harder layer notches twice as fast as the flank,
the ratio VB_N/VB commonly seen on nickel alloys and austenitic steels; mechanism after Trent & Wright
2000). `notch_m` = VB_N = (VN + VN_notch + d_chip) g at elements that have been at a DOC line, 0 elsewhere.
Error: factor 2 in notch depth; the oxidation and burr contributions are inside the exponent.

## 6. Built-up edge

`cnctwin.mechanics.chipform.bue_advance` (mechanics' exact relaxation toward 0.8 h inside the window,
break-off hazard, residual 20 %) per element in cut, with dt = its in-cut time, the material's BUE
window (speed and interface temperature = the rake quadrature mean) and h = int h dt / t_in scaled by
the coolant friction factor of the contact (lubrication lowers the adhesion the BUE grows from).
Elements out of cut keep their BUE. Effects: protection (section 2), attrition per break-off.
Stream `wear.bue_breakoff`, counter per element. Error: BUE dynamics as mechanics (all est).

## 7. Chipping and coating spallation (`chip_trials`)

Per shock row (sorted by tick, tooth, element), at the row's element:

* contact l_c = max(`entry_contact_ratio` (2) x h_entry, worn hone radius); force per unit width
  F' = F_first_full / b, b the tooth's engaged chip width in the step; exit, breakout and breakthrough
  rows x `exit_stress_factor` (1; Pekelharing 1978);
* **wedge stress** (Flamant-Carothers, Timoshenko & Goodier 1970 sect. 36): wedge angle
  beta = pi/2 - rake - clearance, resultant at lambda - rake above the cutting direction (lambda =
  atan mu, the material's friction), components along and across the bisector P_x, P_y;
  sigma_r(r, theta) = -(c_x cos theta + c_y sin theta)/r, c_x = 2 P_x/(beta + sin beta),
  c_y = 2 P_y/(beta - sin beta); the largest tension at r = l_c on a face (the rake face behind the
  contact for a force between the cutting direction and the bisector);
* **Weibull** (Weibull 1951): V_eff = b l_c^2 I_theta / (m - 2), I_theta = int (max(0, g)/g_max)^m dtheta
  (64 nodes clustered at both faces: within 1 % for m <= 30), P = 1 - exp(-(V_eff/V_ref)(sigma/sigma_0)^m
  - (sigma/sigma_crack)^m), sigma_0 = TRS / Gamma(1 + 1/m) at the effective volume of the TRS test piece
  V_ref = L b d / (2 (m + 1)^2) (ASTM C1683; ISO 3327 type B as commonly described, est),
  sigma_crack = K_IC / (Y sqrt(pi a)) for a comb crack of depth a (Y = 1.1215, Tada et al. 2000);
* draw u = hash_uniform(seed, stream_key("wear.chipping"), slot 2^32 + tooth 2^16 + elem, counter, 0),
  counter per element; chipped when u < P: chip depth = `chip_size_ratio` (1) x l_c, the element's
  coating is gone;
* otherwise, with coating on the element: spallation P_c = 1 - exp(-(p / sigma_c)^m_c), p = F'/l_c,
  sigma_c = 4 GPa, m_c = 6 (est; Bouzakis et al. 2012 for impact fatigue of coatings), stream
  `wear.coating`.

Tests: frequencies within 4 sigma of P over 4000 hash-seeded trials, reproducible per seed, size effect
(2 x width at the same force per width: 2 x hazard), stress exponent (1.2 x force: 1.2^m x hazard),
comb cracks add exactly their term. Error: the point-load wedge solution at r = l_c is uncertain by
about 30 % in stress (distributed load, edge radius), i.e. 1.3^m in hazard: chipping probabilities are
order-of-magnitude estimates; the rates are seeded and reproducible.

## 8. Thermal cracking (comb cracks)

Per element with entries in the step: fully reversed surface stress amplitude
sigma_a = alpha E dT / (2 (1 - nu)) (constrained surface, Kingery 1955; dT = thermal's cycle range,
alpha and E at the element's body temperature), damage dD = n_entries / N_f, N_f = 0.5 (min(sigma_a, sigma_f') /
sigma_f')^(1/b) (the curve ends at one reversal: an amplitude beyond sigma_f' cracks the edge in its first cycle,
no more) on the **thermal-cycle curve**: hardmetal-type classes sigma_f' = TRS and b from
0.25 TRS at 5000 cycles (`thermal_fatigue_ratio`, `thermal_fatigue_cycles`, est: a 500 K range on WC-Co
cracks after about 5000 entries, the order seen in interrupted cutting with coolant; isothermal bend
fatigue, 0.7 TRS at 1e7 cycles, would never crack the edge); HSS its isothermal Basquin curve. At D >= 1
comb cracks start (`wear:comb_cracks`) at the initial flaw a0 = (K_IC/(Y sigma_0))^2 / pi and grow toward
the thermal penetration depth delta = sqrt(kappa t_cycle / pi) (where the periodic stress dies out,
Carslaw & Jaeger): a = a0 + (delta - a0)(1 - exp(-(D - 1)/c)), c = `crack_growth_damage` (1, est). The
crack raises the chipping probability (section 7). Error: factor 10 in the cycles to crack initiation,
factor 2 in the depth.

## 9. Breakage (sections, packing, fatigue)

Per critical section i (ToolInfo: tooling's `ToolProps.sections`, ascending z; the same count as
cutting's section arrays): moment M = max(cutting's `section_moment_peak_Nm[i]` (else |F_peak| x arm),
packing force x arm), arm = z_i - z_load (z_load: length-weighted mean height of the cutting edge),
torque T = max(cutting's `torque_peak_Nm`, packing torque); sigma_b = M / Z_b, tau = T / Z_p;
criterion stress: brittle classes (carbide, cermet, ceramic, cBN, PCD) the largest principal stress
sigma_b/2 + sqrt(sigma_b^2/4 + tau^2) (pure torsion fails at tau = strength), HSS von Mises
sqrt(sigma_b^2 + 3 tau^2) (pure torsion at strength / sqrt 3). Strength: Weibull-scaled to the
section's effective volume, bending V = pi d^2/4 arm k(m)/(m + 1), k(m) = Gamma((m+1)/2) /
(2 sqrt(pi) Gamma(m/2 + 2)), torsion V = pi d^2/4 arm 2/(m + 2) (when 2 tau > sigma_b); threshold the
median (`physics.wear.strength: median`) or a per-instance draw (`sample`, stream `wear.strength`).
Fatigue: Basquin (1910) with SWT's equivalent amplitude (rotating bending: sigma; pulsating 0 -> sigma
per tooth entry: sigma / sqrt 2; Smith, Watson & Topper 1970), Miner's sum (1945); the curve ends at one
reversal at sigma_f' (a larger amplitude is the static fracture the strength criterion reports: its damage is one
reversal's, N_f = 1/2, not the curve extrapolated: drill-tap's D6.8 summed 1.7e24 after its blown-up entry); cycles = sum over
teeth of the entries (interrupted), else revolutions in cut (rotating tools); hardmetal-type classes
sigma_f' = TRS, b = ln 0.7 / ln 2e7 = -0.0212 (Nakajima et al. 2006), HSS the modified universal slopes
(Muralidharan & Manson 1988, sigma_u = TRS). A section fails at sigma_eq >= strength (cause bending,
torsion, packing) or D >= 1 (fatigue): `tool_breakage` finding (error) at the tick of the step's force
peak (cutting's per-tick peak wrench) with its cause chain (fracture event <- last chipping, comb cracks,
wear law), `broken` on the bus, the tool's elements stop cutting (tooling sets them invalid), the
fracture section is published (`broken_section_z_m`, `length_loss_m` for the tool setter). A fracture
probability above `breakage_risk_warn` (1 %) raises a warning once per operation. Tests: threshold at
0.98/1.02 x the median strength, torsion, packing spike, fatigue at the predicted cycle count +-1 step.
Error: the TRS range of the database (typically +-15 %) and the section model (equivalent circular
section, Kops & Vo where the core is unknown) dominate: +-20 % in the breaking load; fatigue lives a
factor 10 (one-point Basquin curve).

## 10. Life per tool instance (`life.py`)

Counters (carried over operations and parts, initial from `life.used_min`): time in cut, cutting
distance (largest flank sliding), cut path (feed speed x time in cut), uses (path tool changes to the
slot), parts (parts in which the tool cut). End of life: the first criterion crossed in a step
(linear interpolation within the step; ties in this order):

| criterion | value | limit |
|---|---|---|
| `vb` | mean VB of the used elements of the most worn tooth (VB_B) | the Taylor entry's VB limit, else 0.3 mm (ISO 3685, ISO 8688-2) |
| `vb_max` | largest VB | 0.6 mm (ISO 3685) |
| `crater` | largest KT | 0.06 mm + 0.3 f (ISO 3685; f per tooth for rotating tools) |
| `notch` | largest VB_N | 1 mm (ISO 3685) |
| `dimension` | largest recession of the finishing elements | `dimensional_share` x the tightest allowed surface displacement |
| `finish` | Ra (or Rt) estimate / specification | 1 (only when the new edge met the specification at this feed) |
| `fatigue`, `breakage` | section Miner sum, broken | 1 |

Finishing elements are those that wrote surface events. Allowed displacement (`ProjectTolerances`):
distance and radius +-band / 2, diameter band / 4, zone tolerances zone / 2; fits without plus/minus are
not converted (listed as unused); the integration may assign per tool (`assign`). Finish (Brammertz
1961, Grzesik 1996): R_t = f^2/(8 r) + (h_min/2)(1 + r h_min / f^2) + chip depth, h_min = the material's
minimum chip thickness ratio x the worn hone radius, Ra = R_t 8/(18 sqrt 3); f per revolution
(turning) or per tooth (milling, Martellotti 1941), r the corner/nose/ball radius, else the tool radius.
The first criterion writes `wear:life_end:<criterion>` and a `tool_end_of_life` finding (warning) and
puts the tool number into `WearFeedback.expired`. Remaining life: each criterion's rate per second of
cutting as a bias-corrected exponential average over `rate_window_s` (10 s) of cutting; remaining
cutting time = min (limit - value)/rate; predicted replacement time = now + remaining / duty (in-cut
share of the time the tool is in the spindle, same averaging). Test: exact at constant rates (3 %).
Error: Ra +-30 % (Grzesik's comparison for turning); the remaining life overestimates when the rates
grow (it extrapolates the recent rate).

## 11. Bus record (written: `nxt.wear`)

`WearFeedback.tools[slot]` (every slot, every step; a new record whenever the state changed, else the
previous record object, never mutated), `expired` (tool numbers whose life ended, sorted),
**added** `broken` (tool numbers whose tool broke). `ToolWear` fields: `VB_m`, `KT_m`, `notch_m`
(VB_N at the DOC line, 0 elsewhere), `edge_radius_m` (worn hone radius), `radial_wear_m`, `axial_wear_m`,
`coating_remaining_m` (flank side at the edge), `chipped` (bool), `bue_height_m`, `broken`,
`life_used_fraction` (largest value/limit, 1 when broken); **added** (additive, core/bus.py):
`rake_coating_remaining_m`, `chip_depth_m`, `thermal_damage`, `comb_crack_depth_m`,
`section_fatigue_damage` [nsec], `VB_max_m`, `broken_tick`, `broken_section_z_m`, `length_loss_m`
(fracture z, else the largest axial wear: for the tool setter), `expired`, `life_cause`,
`remaining_life_s`, `replacement_time_s` (inf: not predicted), `in_cut_s`, `cutting_distance_m`, `uses`,
`parts`. No ledger terms (the tool material lost, micrograms, is below every mass tolerance).

## 12. Determinism and RNG streams

No stateful RNG. Streams: `wear.chipping` (key slot 2^32 + tooth 2^16 + element, counter per element),
`wear.coating` (same key, own counter), `wear.bue_breakoff` (mechanics' BUE kernel, own counter),
`wear.strength` (init only: slot, section, 0). Shock rows are sorted (tick, tooth, element); BUE
candidates in flat element order; the calibration uses fixed Gauss nodes, fixed brackets and Brent
tolerances. Checkpoint: `WearState.to_arrays()` / `from_arrays()` hold every array and counter (test:
bit-identical continuation).

## 13. Settings (`physics.wear`, schema.py)

`strength` median | sample, `breakage_risk_warn` 0.01, `calib_nodes` 6, `rate_window_s` 10 s,
`chip_theta_nodes` 64, `skin_samples` 64, `max_chip_events` 20 (trace events per tool and operation; the
counts are complete), `fallback_speeds` 7, `dimensional_share` 1.0, `every_n_macro` 1 (fixed: wear
integrates every step's record).

## 14. Fallbacks (each recorded in `estimated` and once in the trace)

* No thermal record for a tool in cut: flank/rake temperatures from the response table (7 speeds around
  the calibration speeds at a 0.1 mm land) at the element's speed (`wear:temperature_from_response_table`);
  without a table the body temperature (`wear:temperature_body_only`).
* Cutting gives no wear integrals while the flank slides: sigma from the response table x the sliding
  distance (`wear:contact_stress_from_response_table`); without a table no Usui/Archard wear is possible
  and `wear:no_contact_stress` is written (and listed in `estimated`).
* Data-only materials: the cold Prandtl pressure (2 + pi) strength / sqrt 3 when the softened yield of
  mechanics vanishes.
* No skin query for the stock kind: skin not evaluated (`wear:skin_not_evaluated`).
* A slot without cutting-edge elements: counters only.
* Edge temperatures above the tool material's max service temperature: `wear:edge_above_tool_service_temperature`
  and the time in `above_service_temperature_s` (the hot-hardness drop is a deformation check, not
  extrapolated into the wear law).

## 15. Kernels, signatures and measured cost

`@njit(<signature>, cache=True, nogil=True, error_model="numpy")`, no fastmath/parallel, one signature
each (test), each compared with its twin in `reference.py` (1e-12 relative):

| kernel | signature | cost (this box) |
|---|---|---|
| `advance_elements` | `(GEO_T, IN_T, LAW_T, STATE_T, OUT_T)` | 59 us for 176 elements, nq = 3 |
| `element_outputs` | `(GEO_T, LAW_T, STATE_T, OUT_T)` | 10 us for 176 elements |
| `chip_trials` | `(GEO_T, STATE_T, CHIPLAW_T, i8[:], f8[:, :], f8[:], f8[:, :])` | about 3 us per row |
| `usui_rate`, `table_at`, `worn_edge_radius`, `wedge_tension`, `wedge_volume_factor` | helpers | < 1 us |

Per macro step (10 ms): one cutting tool of 4 x 44 elements with 12 shocks and two sections 0.72 ms
(0.07 s per simulated second: kernels 0.13 ms, Python bookkeeping the rest); an idle tool 0.04 ms;
init 0.07-0.15 s per tool including the calibration with the default response.

## 16. Integration

* `WearDomain(tools=GeometryToolInfo(state.tooling.tools), response=..., skin=..., tolerances=...,
  catalog=ctx.db)`; without `tools` the domain uses `state.tooling.tools` when present, else builds the
  geometry with `cnctwin.tooling.build_tool`; without `response` the default mechanics response; without
  `skin` a `StockShell` of block/cylinder/tube stock; without `tolerances` `ProjectTolerances(project)`.
  ToolingDomain.init must run first (ToolTable, geometry).
* `finalize()` returns the `tools.json` document (per tool: number, slot, ids, VB per tooth, VB max and
  limit, KT, notch, edge radius, coating, chips, BUE, thermal damage, section fatigue, broken and cause,
  expired and cause, life used, remaining life, replacement time, cut time, counters, Ra, drift, criteria,
  wear offsets, the law with its sources and calibration report, `per_part` rows as
  `cnctwin.production.rundir.tool_states` reads them).
* `macro_values(state)` gives the channel values (`channels.py`).
* The controller: `WearFeedback.expired` -> `ir.Feedback.tool_life_expired`; its broken-tool detection
  (tool setter `measure: broken`) compares the measured length with `ToolWear.length_loss_m` and reads
  `broken`. Cutting and removal stop using a broken tool (tooling composes `broken` into invalid
  elements; removal should shorten the envelope to `broken_section_z_m`).

## 17. Assumptions and limitations (with error estimates)

* Constant conditions within a macro step (temperatures, stresses): increments per step are nm, the
  coating breakthrough within a step is not split (error below one step's increment).
* Taylor calibration in an equivalent continuous orthogonal cut at the Taylor feed and depth (h = f,
  for milling data fz): the time basis of milling lives is converted (life basis, section 3), the
  temperature and chip-thickness history of the interrupted cut is not (the response is the steady cut
  at h = fz: error the duty-cycle temperature difference, absorbed by A, partly by B, and exact only
  when the response is the run's own pipeline). A milling entry without `engaged_arc_ref_deg` is read at
  the ISO 8688-2 reference arc (ae/D = 0.1): data measured at another engagement shift the life in
  cutting time by phi_true / phi_ref (slotting data read this way: x 0.2 in cutting-time life).
* The tool's life counters (`in_cut_s`, `cut_time_s`) add cutting's tool-level in-cut time (the time
  any element is in the material), which equals the cutting time while at least one tooth is always
  engaged; with gaps between the teeth (few teeth, light radial engagement) it is shorter than the
  feed time in the material that Taylor and ISO 8688 count (life is unaffected: wear follows contact).
* The calibrated law is consistent with the run only when the response and the run use the same
  temperature model (section 3, factor exp(B dT / T^2)).
* Chipping, spallation, thermal-crack and fatigue constants are estimates (marked `est`), notch and
  edge rounding too; their outcomes are order-of-magnitude.
* Vibration (chatter) loads are not separate fatigue cycles: they enter only through cutting's peak
  moments.
* The finish criterion uses the kinematic profile plus Brammertz's minimum-chip-thickness term; side
  flow and vibration are the surface domain's.

## 18. References

Archard 1953; ASTM C1683-10; Basquin 1910; Bouzakis et al. 2012; Brammertz 1961; Carslaw & Jaeger 1959;
Grzesik 1996; ISO 3327:2009; ISO 3685:1993; ISO 8688-2:1989; Kingery 1955; Martellotti 1941; Miner 1945;
Muralidharan & Manson 1988; Nakajima, Hosokawa & Shimojima 2006; Pekelharing 1978; Roebuck et al. 2009
(NPL MGPG 20); Smith, Watson & Topper 1970; Smithey, Kapoor & DeVor 2001; Tada, Paris & Irwin 2000;
Takeyama & Murata 1963; Taylor 1907; Timoshenko & Goodier 1970; Trent & Wright 2000; Usui, Shirakashi &
Kitagawa 1984; Weibull 1951; Yen et al. 2004; Zhang et al. 2025 (Ti6Al4V Usui constants in the
database). Full entries: `db/bibliography/models-wear.yaml` and the files named in its header.
