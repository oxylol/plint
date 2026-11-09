# How to add a cutting tool

Tools live in series files: `db/tools/<maker>-<series>.yaml` (format `cnctwin-tool`, version 1), one file per
catalogue family, each size a complete item under `items:`. A tool names its substrate
(`db/tool_materials/<id>.yaml`, `cnctwin-tool-material`) and coating (`db/coatings/<id>.yaml`,
`cnctwin-coating`); a project puts it in a holder (`db/holders/<maker>-<series>.yaml`, `cnctwin-holder`).
The tooling domain turns an item into edge elements, an envelope and properties
([docs/models/tooling.md](models/tooling.md)); cutting, wear, thermal, holding, structure and chips read those.
Format rules: [formats.md "Databases"](formats.md#databases).

A tool used by one project only does not need a database file: see section 7.

## 1. Collect the sources

* The maker's product page or catalogue page (order code, diameter, flutes, helix, lengths, corner radius,
  neck, shank, coating, maximum speed, list price) and the maker's cutting data chart (vc and fz per ISO
  material group, ap and ae limits). Add an entry per document to `db/bibliography/tools-milling.yaml` or
  `db/bibliography/tools-holemaking-turning.yaml` (type `web`, `catalogue` or `datasheet`; `url`,
  `accessed`, and `pages` saying which table).
* What the maker does not publish (rake, clearance, edge radius, core diameter, gullet area, mass, regrinds) is
  estimated with a note. These values matter: the edge radius sets ploughing, the core diameter the
  stiffness, the gullet area chip packing.

## 2. Write the series file from the template

```
cp db/_templates/tool.yaml db/tools/<maker>-<series>.yaml     # series_id = file name
```

| key | what to give |
|---|---|
| header | `format: cnctwin-tool`, `version: 1`, `series_id`, `maker`, `series`, `src` (covers every item), optional `defaults:` (merged under every item), `notes` |
| item `id`, `order_code` | the catalogue code lower-cased as the id (unique over the whole database: two series cannot define the same id) |
| `type` | one of the 30 types of the template: `square_endmill`, `ball_endmill`, `toroidal_endmill`, `roughing_endmill`, `serrated_endmill`, `face_mill`, `shoulder_mill`, `chamfer_mill`, `engraver`, `keyseat`, `tslot`, `dovetail`, `lollipop`, `slitting_saw`, `spot_drill`, `centre_drill`, `twist_drill`, `indexable_drill`, `gun_drill`, `reamer`, `tap_cutting`, `tap_forming`, `thread_mill`, `boring_head`, `boring_bar`, `turning_insert`, `grooving_insert`, `parting_insert`, `threading_insert`, `form_tool` |
| `hand` | `right` or `left` (the spindle direction it cuts in) |
| `tool_material` | `{class, grade, id}`: `class` one of `hss carbide cermet ceramic cbn pcd`, `id` a file of `db/tool_materials/` |
| `coating` | `{id, thickness_um}` with `id` a file of `db/coatings/`, or `{id: none}` |
| `geometry` | the type's keys (the template's comments list them): end mills `diameter_mm`, `flutes`, `pitch_deg`, `helix_deg`, `rake_deg`, `clearance_deg`, `edge_radius_um`, `edge_prep`, `corner_radius_mm`, `cutting_length_mm`, `flute_length_mm`, `neck`, `shank`, `overall_length_mm`, `gullet_area_mm2`, `core_diameter_mm`, `centre_cutting`, `coolant_holes`; drills add `point_angle_deg`, chisel and web, `margin_width_mm`; taps and thread mills `thread {standard, pitch_mm, form_angle_deg, class}`; indexable mills `inserts {count, iso_code, ic_mm, nose_radius_mm, rakes, lead_angle_deg, positions_deg}`; turning family `insert {...}` and `holder {...}`; boring bars `bar {...}` |
| `max_rpm`, `balance_grade`, `mass_g` | as published, else `null` or an estimate |
| `cutting_data` | rows `{iso_group, material_hint, operation, vc_m_min: {min, start, max}, fz_mm: {...}, ap_max_mm or ap_max_xD, ae_max_xD, src, page, note}`; `material:` may name a material id (checked) |
| `price_eur`, `regrinds` | for the tool cost per part |

If the substrate or coating is new, add it from `db/_templates/tool_material.yaml` (density, modulus,
hardness, transverse rupture strength, toughness, conductivity and specific heat tables, expansion, service
temperature, Weibull modulus, optional Usui wear constants with their unit system in the key) or
`db/_templates/coating.yaml` (hardness, service temperature, conductivity, friction against steel, typical
thickness, wear-rate and thermal-barrier factors).

If the holder is new, add it from `db/_templates/holder.yaml`. A holder a project uses must give
`gauge_length_mm`, `nose_diameter_mm` and `profile_mm` (the revolved outline for collisions); a project that
uses a holder whose item has them as `null` is refused with the inline form to add them
(`holder: {extends: <id>, profile_mm: [...]}`).

## 3. Sourcing rules

* Every number has a source: its own `src` or the series `src` (the series `src` covers every item). Cutting
  data rows name the chart they come from and the page, row and column (`page:`), and the note says how a
  value was converted (SFM to m/min, IPT to mm, which diameter column was read).
* An unpublished value is `{value: ..., src: est, note: "how it was derived"}`: a formula with its inputs
  (mass = rho V with the densities named), a rule of thumb and where it comes from, or "assumed" with the
  reason. Never attach a real source to an invented number.
* Units are the last token of the key (`diameter_mm`, `helix_deg`, `edge_radius_um`, `vc_m_min`, `mass_g`).

## 4. Check

```
bin/twin db check --strict                                 # 0 errors, 0 warnings
bin/twin db check --estimates | grep 'db/tools/<file>'     # every estimate of the series, with its note
```

References are checked: `tool_material.id`, `coating.id` (or `none`) and `cutting_data[].material` must
exist. Then build an item with a holder, as the tooling domain does:

```
bin/tsafe python -c "
from cnctwin.db import Database
from cnctwin.tooling import build_tool
db = Database.load()
tool, holder = '<tool id>', '<holder id>'
g = build_tool(db.get('tool', tool), db.get('holder', holder), {'number': 1, 'stickout_mm': 30},
               tool_id=tool, holder_id=holder, db=db)
print(g.edge.elem_r_m.shape, int(g.edge.elem_valid.sum()), g.props.tool_mass.mass_kg)"
```

It prints teeth x elements, the number of valid edge elements and the tool's mass; an `InputError` names
what is missing.

## 5. Tests

* `tests/unit/db/test_real_db.py` loads, validates (also strictly) and builds every database file.
* `tests/unit/tooling/test_all_types.py::test_database_tools_build` builds every schema-valid tool of the
  database with a valid holder and checks the edge model is finite: a new item is covered by `bin/test
  unit/tooling` without a new test.
* A new **tool type** is more than data: add it to `TOOL_TYPES` in `cnctwin/db/schemas.py` and the template,
  give it a builder in `cnctwin/tooling/build.py` (`BUILDERS`) and a fixture in `tests/unit/tooling/fixtures.py`;
  `test_every_template_type_has_a_builder_and_a_fixture` fails until all three agree, and docs/models/tooling.md
  section 4 describes the family.

## 6. Worked example: `db/tools/helical-mh35al-3-square-metric.yaml`

Helical's MH35AL-3 square end mills for aluminium. The header's `src` is the product page; the file comment
says when it was read and how estimates are marked:

```yaml
# Generated from catalogue data read 2026-11-03; values without src: est are as printed by the maker.
format: cnctwin-tool
version: 1
series_id: helical-mh35al-3-square-metric
maker: Helical Solutions (Harvey Performance Company)
series: MH35AL-3 3 flute square, 35 deg helix, metric - end mills for aluminium alloys and non-ferrous materials
src: helical-2026-mh35al3-sq-metric
items:
- id: helical-89597-mh35al-020-30300
  order_code: 89597 (MH35AL-020-30300, uncoated)
  type: square_endmill
```

Published dimensions are plain numbers (`diameter_mm: 3.0`, `flutes: 3`, `helix_deg: 35`,
`cutting_length_mm: 6.0`, `overall_length_mm: 63.0`); the rest is estimated and says how:

```yaml
    rake_deg: {value: 15, src: est, note: 'radial rake not published; 15 deg assumed, typical of 3-flute aluminium end mills'}
    edge_radius_um: {value: 4, src: est, note: edge radius not published; 4 um assumed for a sharp uncoated aluminium edge}
    core_diameter_mm: {value: 1.65, src: est, note: 'dc = 0.55 D: low end of the 0.55-0.65 D rule of thumb, ...'}
  mass_g: {value: 23.7, src: est, note: 'rho V: shank cylinder d 6 x 57 mm plus fluted cylinder D 3 x 6 mm x solid fraction 0.48 ...; rho = 14.5 g/cm3 (db/tool_materials/carbide-submicron-10co, Hyperion H10F)'}
```

The substrate is a database item chosen by an estimate (Helical publishes only "solid carbide"):

```yaml
  tool_material: {class: carbide, grade: 'solid carbide, grade not published', id: carbide-submicron-10co, src: est, note: Helical publishes 'solid carbide' only; submicron 10 % Co end-mill grade assumed}
  coating: {id: none, note: uncoated tool number chosen (the Zplus-coated twin has its own tool number)}
```

Each cutting data row names the chart, the row and the column it was read from, and the conversion:

```yaml
  - iso_group: N
    material_hint: wrought aluminium 2014/5052/6061/7050/7075/7475 (e.g. al6061-t6, al7075-t6)
    operation: [slotting]
    vc_m_min: {start: 640.08}
    fz_mm: {start: 0.01778}
    ap_max_xD: 1.25
    ae_max_xD: 1.0
    src: helical-2021-sf-h35al3
    page: SF_H35AL-3 p. 1, row 'WROUGHT ALUMINUM ALLOY' (SFM 2100), 1/8 in column, 'Slot'; milling-process table
    note: '... published values: vc 2100 SFM, fz 0.0007 in (converted at 0.3048 m/ft and 25.4 mm/in)'
```

Built in an ER16 collet chuck (`haimer-40-720-16`) at 20 mm stick-out with the snippet of section 4, the item
gives 3 teeth of 37 stations (107 valid elements) and 0.0237 kg.

## 7. A tool for one project only

A project's `tools.list` entry may hold the tool inline instead of an id: a complete item (the template's
`items` entry, no header, `id` optional) or `{extends: <id>, ...}` with the keys that differ (a regrind to a
smaller diameter, a different stick-out-dependent neck). Inline values need no sources; their SI view
reports `src: project`. Many tools at once come from a **tool library**: `tools.library: tools.csv` (one row
per tool, columns in formats.md "Tool libraries") or an ISO 13399 directory or file (`.p21`, `.stp`), whose
items take their tool material and coating inline. Either way the project check (`bin/twin check`) and the
run build the tools exactly as database items.
