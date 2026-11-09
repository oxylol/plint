# How to add a workpiece material

A material is one file, `db/materials/<id>.yaml` (format `cnctwin-material`, version 1), one per alloy and
condition (`al6061-t6`, `al6061-o` and `al7075-t7351` are separate files). A project names it with
`material: {id: <id>}`. The mechanics library reads the flow stress, friction, Kienzle and mechanistic data and
the chip behaviour ([docs/models/mechanics.md](models/mechanics.md)); cutting builds its coefficient tables
from them; thermal reads the physical tables; wear the tool-life entries; surface the minimum chip thickness,
work hardening and thermal-damage limits; holding, chips and production the density; structure and inspect
the modulus and expansion. Format rules: [formats.md "Databases"](formats.md#databases), "Kind notes:
material".

## 1. Collect the sources

* Physical and mechanical properties: a supplier datasheet or a materials handbook (density; specific heat,
  conductivity, expansion and Young's modulus against temperature; solidus and liquidus; hardness; yield,
  ultimate, elongation).
* Constitutive law: a Johnson-Cook set from the paper that identified it (all five constants, the reference
  strain rate and temperature, the melting temperature it was used with). Note what it was identified on
  (Hopkinson bar, machining tests) and over which strain rates and temperatures: the shear zone of a cut
  loads the material at far higher strain rates and temperatures than most identification tests, so the law
  is used outside the range it was fitted on; say that range in the note.
* Cutting data: Kienzle kc1.1 and mc per tool material class from a catalogue or a paper (with the feed and
  passive force constants when published), mechanistic milling coefficients when a calibration was published,
  tool-chip friction, the minimum chip thickness ratio, Taylor tool life with the VB limit, chip form and BUE
  behaviour, residual stress of the stock forms you model.
* Add each source to the bibliography (`db/bibliography/materials-*.yaml`). A paper that also serves as a
  validation reference makes the cases that use it **calibration-dependent** when its numbers enter the
  material (docs/validation.md section 7); that is allowed, and the report keeps those cases out of the
  verdicts.

## 2. Write the file from the template

```
cp db/_templates/material.yaml db/materials/<id>.yaml       # id = file name, lower case with dashes
```

| section | what to give | notes |
|---|---|---|
| header | `id`, `name`, `family` (`aluminium steel tool_steel stainless titanium nickel copper brass magnesium cast_iron polymer wood foam composite`), `iso_group` (`P M K N S H`, or `O`), `condition`, `designations`, `stock_forms` | |
| `physical` | `density_kg_m3`; tables `{table: [[T_degC, value], ...]}` for `specific_heat_J_kgK`, `conductivity_W_mK`, `expansion_1_K`, `youngs_modulus_GPa`; `poisson`, `solidus_degC`, `liquidus_degC`, `hardness` (one of HB, HV, HRC, HRB, shore_D), `yield_MPa`, `ultimate_MPa`, `elongation_pct`, `fracture_strain` | tables are sorted by temperature; outside the table the end value is held and the clamp is counted (`Table.clamped`), so a report can say a property was extrapolated |
| `flow_stress` | `model: johnson_cook` (required for metals) with `A_MPa`, `B_MPa`, `n`, `C`, `m`, `ref_strain_rate_1_s`, `T_ref_degC`, `T_melt_degC`; or `temperature_table` (polymers: a `tensile_strength_MPa` table, `compressive_yield_MPa`, `T_soft_degC`); or `none` | `none` makes a data-only material: forces from the Kienzle data, temperatures from Merchant geometry (mechanics.md 7.4) |
| `cutting.kienzle` | per `tool_class` (and `coating`): `kc11_N_mm2`, `mc`, optional `kf11_N_mm2`/`mf`, `kp11_N_mm2`/`mp`, `h_ref_mm: 1`, `b_ref_mm: 1`, `rake_ref_deg`, `calibration {vc_m_min, h_mm, edge_radius_um, coolant, work_temp_degC}` | with Kienzle or mechanistic data for the pairing, the coefficient pipeline calibrates the mechanics to them (mechanics.md 7.3) |
| `cutting.mechanistic` | `Ktc/Krc/Kac_N_mm2`, `Kte/Kre/Kae_N_mm` with the tool's `helix_deg`, `rake_deg`, class and coating | only where a published calibration exists |
| `cutting` others | `min_chip_thickness_ratio` (h_min / edge radius), `indentation_coeff_N_mm3` (process damping), `work_hardening` | |
| `friction` | per tool class and coating: `mu_sliding`, `sticking_ratio` | |
| `chip` | `tendency`, `segmentation`, `bue` window, `breakability`, `oxide_colours` (ferrous and titanium temper colours) | |
| `tool_life` | extended Taylor `vc (T/T_ref)^n (f/f_ref)^a (ap/ap_ref)^b = C` per tool class, coating and `process`: `C_m_min`, `n`, `a`, `b`, `feed_ref_mm`, `ap_ref_mm`, `VB_limit_mm`, `T_ref_min: 1`, `life_basis` (`cutting_time` or `contact_time`), for milling lives `engaged_arc_ref_deg` | the wear domain calibrates its Usui law to these entries (docs/models/wear.md section 3); without `engaged_arc_ref_deg` a milling life is read at the ISO 8688-2 reference arc |
| `tool_life_usui` | published Usui constants with `A_per_MPa` or `A_per_Pa` and the `rate_unit` | |
| `residual_stress` | per stock form: `thickness_mm`, `direction`, `profile_MPa: [[z/thickness, MPa], ...]` | |
| `skin`, `thermal_damage`, `ignition`, `composite` | scale or cast skin; burn, white-layer and smearing temperatures; fire risk of fines; composite data | `type: none` skin values need no source |
| `extensions` | data outside the template | kept as written, read by no model; its numbers still need sources |

## 3. Sourcing rules

* Every number carries a source: its own `src` or the section's (`physical: {src: <key>, ...}` covers the
  section). A Johnson-Cook block carries the `src` of the paper the five constants come from.
* An unpublished value is `{value: ..., src: est, note: "how it was derived"}`: "carbide catalogue value x
  1.15, assumed for HSS", "C set so that T = 15 min at vc = 800 m/min (...)". If a source gives a range, store
  the value used with `range: [lo, hi]` and say which value was taken and why.
* Keep the laws' reference values: Kienzle with `h_ref_mm: 1`, `b_ref_mm: 1`; Taylor with `T_ref_min: 1`;
  Usui constants with the unit system in the key and the rate unit stated.
* Temperature-dependent properties are tables in degC (`x: degC` is the default); temperature differences
  are `_dK`.

## 4. Check

```
bin/twin db check --strict                                    # 0 errors, 0 warnings
bin/twin db check --estimates | grep 'db/materials/<id>'      # the estimates and their notes
bin/tsafe python -c "
import math
from cnctwin.db import Database
from cnctwin.mechanics.materials import WorkMaterial, ToolSide
from cnctwin.mechanics import oxley
db = Database.load()
work = WorkMaterial.from_db(db.material('<id>'))
tool = ToolSide.from_db(None, db.tool_material('carbide-submicron-10co'), None)
print(oxley.solve(work, 0.1e-3, 300 / 60.0, math.radians(8), tool=tool).chip_ratio)"
```

The last command solves one orthogonal cut (h = 0.1 mm, 300 m/min, rake 8 deg) with the material's flow
stress and friction: a finite chip ratio between about 0.1 and 1 says the constants are in a plausible
range; an `InputError` names what is missing. For `al6061-t6` it prints 0.417.

To see the material in a cut, change `material.id` of a copy of an example project and run its start with
`bin/twin run <copy> --set max_sim_s=4`, or validate it against a published experiment (section 5).

## 5. Tests and validation

* `tests/unit/db/test_real_db.py` loads, validates (also strictly) and builds the file;
  `tests/unit/mechanics/test_no_nan.py::test_database_materials_finite` builds the coefficient tables (the
  process map and the group coefficients over grids of speed, rake, tool temperature and chip thickness) and
  the chip form for every database material and requires finite values. Both cover a new file without a new
  test: `bin/test unit/db`, `bin/test unit/mechanics`.
* A new `family` or a new flow-stress model needs code and tests in cnctwin/mechanics (and the schema).
* Validate it where a published experiment exists: write `validation/cases/<quantity>/<id>.yaml` from
  `db/_templates/validation_case.yaml` with `setup.material: {db: <id>}`, run `bin/twin db check` and
  `bin/twin validate --case <id>` (docs/validation.md "Adding a case").

## 6. Worked example: `db/materials/al6061-t6.yaml`

The header comment names the main sources; the physical section has one `src` for the section and per-value
sources where another document was used, with the spread between sources as a `range`:

```yaml
# Physical and mechanical data: Kaiser 2006 sheet/plate datasheet and NASA 1972 materials handbook.
format: cnctwin-material
version: 1
id: al6061-t6
family: aluminium
iso_group: N
physical:
  src: kaiser-2006-al6061-datasheet
  density_kg_m3: {value: 2700, note: "also 2.70 g/cm3 in muraca-1972-al6061-handbook"}
  youngs_modulus_GPa: {table: [[20, 68.9]], src: muraca-1972-al6061-handbook, range: [68.3, 68.9], note: "typical E = 10.0e3 ksi; Kaiser 68.3 GPa"}
  fracture_strain: {value: 0.5, src: est, note: "true strain at fracture ln(1/(1-RA)) with an assumed reduction of area of about 40 %; ..."}
```

The Johnson-Cook constants come from the LLNL report that identified them; where another paper prints a
different value, the file keeps the original and says why:

```yaml
flow_stress:
  model: johnson_cook
  A_MPa: {value: 324, range: [324, 334], note: "Table 1 of the original LLNL report (lesuer-2001-llnl-jc); dawara-2023-impact-framework print A = 334 MPa for the same set (3 % higher, taken as a transcription error); 324 used"}
  B_MPa: 114
  n: 0.42
  C: 0.002
  m: {value: 1.34, note: "Table 1 (previously an estimate taken from a review snippet; now read in the original)"}
  T_ref_degC: {value: 20, src: est, note: "room temperature assumed; not stated in lesuer-2001-llnl-jc or dawara-2023-impact-framework"}
  src: lesuer-2001-llnl-jc
```

Kienzle data from a catalogue, with the reference geometry the catalogue does not state marked as estimates;
an HSS entry derived from the carbide one is an estimate as a whole:

```yaml
cutting:
  kienzle:
    - tool_class: carbide
      coating: any
      kc11_N_mm2: 800
      mc: 0.25
      rake_ref_deg: {value: 0, src: est, note: "ISCAR's table does not state the reference geometry; a neutral (0 deg) rake is assumed, as Sandvik defines its kc1 for a 0 deg rake"}
      src: iscar-materials-groups
    - tool_class: hss
      kc11_N_mm2: 920
      mc: 0.25
      src: est
      note: "carbide catalogue value x 1.15, assumed for HSS (lower speed, larger edge radius); no published HSS value read"
```

No published Taylor fit was found, so every tool-life entry is an estimate whose note gives the anchor point
it was set to; the `notes` at the end of the file summarise what is measured and what is estimated. That is
the honest state of this material: its forces rest on catalogue data, its tool life does not rest on data at
all, which the wear results inherit.
