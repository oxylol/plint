# Database file templates

Every database file is YAML, versioned (`format`, `version`), and every number in it
carries a source. The loader (`cnctwin/db`) converts unit-suffixed keys to SI and refuses a
value whose `src` does not resolve to an entry in `db/bibliography/*.yaml`.

## Rules for every database file

* **Units live in the key name, as its last token**: `density_kg_m3`, `travel_mm`, `power_kW`,
  `max_rpm`, `kc11_N_mm2`, `edge_radius_um`, `temp_degC`, `growth_at_max_speed_um`. The suffix
  table is in `cnctwin/units.py` (`SUFFIXES`). Never put a bare number whose unit is not in its key.
  Minimum/maximum are prefixes (`min_abs_pressure_kPa`, never `..._kPa_abs_min`); `_min` at the end
  always means minutes. Temperature differences are `_dK`; `_degC` is only an absolute temperature.
  Per-length quantities are `_per_mm`. Poses are `{translate_mm: [x, y, z], rotate_deg: [a, b, c],
  order: xyz}`. Tables carry their x unit: `{table: [[x, v], ...], x: degC}` (degC is the default
  only for temperature-dependent properties). Two-column curves are declared by the schema at their
  path (spindle `power_kW: [[rpm, kW], ...]`, `torque_Nm: [[rpm, Nm], ...]`, residual stress
  `profile_MPa: [[z/thickness, MPa], ...]`).
* **Laws keep their reference values**: Kienzle `kc = kc11 (h/h_ref)^-mc` with `h_ref_mm: 1`,
  `b_ref_mm: 1`; Taylor `vc (T/T_ref)^n f^a ap^b = C` with `T_ref_min: 1`; Usui constants with the
  unit system in the key (`A_per_MPa`) and the rate unit stated.
* **Every value has a source.** Either per value (`{value: 2700, src: asm-v2-2019}`), or a
  section-level `src:` that applies to every value in that section unless a value overrides
  it. A source is a key in `db/bibliography/*.yaml`.
* **Estimates are allowed only when marked**: `src: est` plus `note:` saying exactly how the
  number was derived ("scaled from 6061-T6 by the hardness ratio 1.6", "typical of 40-taper
  spindles in Schmitz & Smith 2009 Table 6.1"). Estimated values are listed in the report.
* **Never invent a number and attach a real source to it.** If a source gives a range, store
  the range (`range: [lo, hi]`) and the value used (`value:`), and say which.
* **Digitised from a figure** means `digitised: true` plus `uncertainty_pct:`.
* Temperature-dependent properties are tables `[[T_degC, value], ...]`, sorted by T.

## Bibliography entry (`db/bibliography/<area>.yaml`)

```yaml
format: cnctwin-bibliography
version: 1
entries:
  johnson-cook-1985:                # key: firstauthor-year[-topic]; lowercase, dashes
    type: article                   # article | book | handbook | catalogue | standard | manual | datasheet | web | thesis
    authors: "Johnson, G.R.; Cook, W.H."
    title: "Fracture characteristics of three metals subjected to various strains, strain rates, temperatures and pressures"
    venue: "Engineering Fracture Mechanics 21(1), 31-48"
    year: 1985
    doi: "10.1016/0013-7944(85)90052-9"   # if any
    url: "https://..."                    # where it was read, if online
    accessed: 2026-11-03                  # date it was read
    pages: "Table 1"                      # where in the source the values are
```

Templates for each database type are in this folder: `material.yaml`, `tool.yaml`,
`tool_material.yaml`, `coating.yaml`, `holder.yaml`, `machine.yaml`, `coolant.yaml`,
`workholding.yaml`, `controller.yaml`, and the validation case format
`validation_case.yaml` (used in `validation/cases/`).
