# How to add a controller dialect

A dialect is what differs between controls: the program syntax, the G- and M-code tables with their modal
groups and power-on states, the cycles, the alarm numbers and texts, the system variables and parameter
defaults, and the semantics of a block. The executor core (`cnctwin/controller/executor.py`: the block-time
model, the coordinate pipeline, calls, loops, waits, buffer stops, checkpoints) is the same for every dialect.
Read [docs/dialects.md](dialects.md) "Structure of the controller" first; its last section, "Adding a dialect",
is the contract this page walks through with the checks and the files outside `cnctwin/controller` that also
need the new name.

Existing dialects: `fanuc` (0i-MF / 0i-TF, custom macro B), `haas` (NGC, on top of the FANUC handler),
`siemens840d` (SINUMERIK 840D sl, own syntax), `heidenhain_klartext` and `heidenhain_iso` (TNC 640).

## 1. Sources

The control's programming manual (syntax, G codes and groups, power-on state, cycles, variables), its
diagnostics or alarm manual (numbers and texts as the control shows them), and the machine-data or parameter
manual (defaults: arc tolerances, nesting depths, M codes per block). Add them to
`db/bibliography/controllers.yaml` (or `controllers-2.yaml`). Every parameter default the dialect uses is a
sourced value or an estimate with a note, like any database number.

## 2. Files to write

| file | what | example |
|---|---|---|
| `cnctwin/controller/dialects/<name>/__init__.py` (or a module) | the builder and `register_dialect("<name>", build)` | `dialects/siemens840d/__init__.py` |
| handler, syntax, tables, cycles modules | the block semantics, the program reader (if not ISO word address), the G-code table, alarm numbers with built-in texts, the word list, cycle providers | `siemens840d/handler.py`, `syntax.py`, `tables.py`, `cycles.py` |
| `db/models/controller-<name>.yaml` | the dialect's parameter defaults (format `cnctwin-model-params`): each with `value`, `unit`, `src` or `est` with a note, the machine-data number it stands for in the note | `db/models/controller-siemens840d.yaml` |
| `db/controllers/<id>.yaml` | the control reference (format `cnctwin-controller`, template `db/_templates/controller.yaml`): `dialect`, `alarms` (`code`, `text`, `cause`, `remedy`; a `kind` entry replaces a twin alarm), `timing`, `messages`, `words` (the G/M/address words the control supports: the dialect gates on them) | `db/controllers/siemens-840d-sl.yaml` |
| a machine file with `controller.dialect: <name>` and `controller.alarms: <id>` | at least one database machine that uses it | `db/machines/dmg-dmu50-3rd-siemens.yaml` |
| tests | `tests/unit/controller/test_<name>*.py` | `test_siemens.py`, `test_siemens_cycles.py`, `test_siemens_checkpoint.py` |

Places outside the package that hold the list of dialect names (the contract in dialects.md does not name
them, but each refuses or mishandles an unknown name):

* `cnctwin/db/schemas.py` `DIALECTS`: the machine schema's `controller.dialect` and the controller schema's
  `dialect` accept only these names; `twin db check` rejects a machine file with a new name until it is added
  (and `db/_templates/machine.yaml`, `controller.yaml` list the allowed values in their comments).
* `cnctwin/controller/hooks.py` `PLUGIN_PACKAGES`: the packages `load_plugins()` imports. `dialects.siemens840d`
  and `dialects.heidenhain` are listed; a new package must be added, or its `register_dialect` never runs and
  the build fails with `no controller dialect '<name>'`.
* `cnctwin/project/new.py` `program_skeleton` and `PROGRAM_SUFFIX`: the shortest complete program and the file
  extension `twin new` writes; without an entry `twin new --machine <a machine of the dialect>` fails.
* `cnctwin/api/views.py` `TEXT_PROGRAMS`: program extensions served as text (the UI shows them in place).
* cnctwin.sanity scans FANUC and Haas syntax itself; for another dialect its rules need the controller's command
  stream, and without it the program checks report one `scan-limits` finding (docs/models/sanity.md section 3).

## 3. The builder and the dialect object

`build(machine: MachineModel, settings: ControllerSettings, controller_db: dict | None, db_id: str) -> Dialect`
(`cnctwin/controller/dialects/base.py`). The fields:

* `name`, `variant` (`machine.variant`: `"M"` for mills, `"T"` for lathe types);
* `lexer`: `LexerRules` for ISO-like syntax; a dialect with its own syntax sets `parser` to
  `(sources, lexer_rules) -> (ProgramStore, problems)` and builds `Block`s itself (its payload in
  `Block.extra`) with `lexer.make_program(file, number, blocks)`. Reading never fails: a block that cannot be
  read carries its alarm and raises it when control reaches it;
* `gcodes` (`GSpec(group, kind, arg, cycle, axis_words)`; `cycle=True` routes to the cycle registry) and
  `power_on` (group -> code);
* `alarms`: `AlarmTable(name, key -> code, code -> built-in text, controller_db, db_id)`, keyed by the
  executor's condition keys (`fanuc_tables.ALARM_CODES`, `alarms.TWIN_ALARMS`); texts come from the controller
  database when it lists the number, else the built-in copy (`source: builtin`);
* `sysvars`: `read(ex, n)`, `write(ex, n, value)`, `adp(ex, n, value)`; tables such as R parameters live in
  `PathState.variables[<table>]` so they checkpoint;
* `handler_class`: subclass `FanucHandler` for ISO word address (as `HaasHandler`), or implement the methods the
  core calls (`init_state`, `on_reset`, `execute(block)`, `plane`, `feed_mode`, `absolute`,
  `tool_offset_counts`, `comp_radius`, `modal_group_value`, `modal_snapshot`, `apply_mode`, `deferred_m`, and the
  value helpers providers use). The handler object holds no state: it is rebuilt on resume, so everything goes
  into `ex.s` (`PathState`);
* `params`: the resolved defaults: `db/models/controller-block-time.yaml` plus `controller-<name>.yaml` plus the
  project's `physics.controller` settings, covering every key the executor reads (`macro_statement_s`,
  `macro_term_s`, `call_s`, `goto_scan_s`, `min_block_s`, `horizon_s`, `arc_tolerance_m`, nesting limits,
  `max_m_codes_per_block`, wait codes, alarm bases ...);
* `supported_g` / `supported_m` from the controller database's `words` (`base.word_set`), or `None`.

Semantics go through the core, never around it: motion with `ex.move/arc_move/machine_move` in program-frame
counts, offsets in `ex.s.offsets`, frames by setting `ex.s.mirror/scale/rotation/feature` and calling
`ex.invalidate_transform()` and `ex.resync_program_position()`, non-motion actions with `ex.emit(Command)`, flow
with `ex.goto/loop_while/loop_end/request_call/ret/end_program`, buffer stops with `ex.wait(...)` after the
SYNC/STOP/PROBE command, path waits with `ex.wait("path_wait", ...)`, alarms by raising
`ControllerAlarm(key, detail)`, warnings with `ex.note`. A feature the dialect does not model raises a twin
alarm (TW0012 for an unmodelled function) instead of guessing.

## 4. Checks

```
bin/twin db check --strict                                  # the model file, the controller file, the machine
bin/tsafe python -c "
from cnctwin.controller import hooks
from cnctwin.controller.dialects.base import dialect_names
hooks.load_plugins(); print(dialect_names())"               # the new name is registered
bin/twin new /tmp/dialect-check --machine <machine id>      # the skeleton program of the dialect
bin/twin check /tmp/dialect-check --out /tmp/dialect-check/out   # dry run: cycle-time lower bound, alarms
```

Then dry-run real programs of the control (`twin check` on a project whose operation is such a program): the
dry run reports every alarm with its program line, and the check's `cycle_time_s` is the ideal-machine lower
bound of the cycle time.

## 5. Tests

In `tests/unit/controller/`, with inline machine and controller-database fixtures (`harness.py` has minimal
mill and lathe definitions):

* the syntax: every construct the reader accepts, and the alarm of every one it refuses, at the block where the
  control raises it;
* the command stream of programs against hand-checked expectations: motion and arcs, frames, tools, spindle,
  coolant, variables and flow, cycles (expected command streams, as `test_siemens_cycles.py`);
* a **checkpoint/resume** test of the dialect's own state: stop in the middle of a loop or a cycle, restore into
  a fresh controller and require the continued command stream to equal the uninterrupted one
  (`test_siemens_checkpoint.py`, `test_heidenhain_checkpoint.py`);
* where another dialect can make the same part, the same part in both (`test_siemens_heidenhain_same_part.py`);
* the alarm texts from the controller database and the built-in fallback (`test_alarms.py`).

Run `bin/test unit/controller`.

## 6. Worked example: the Siemens 840D sl dialect

`cnctwin/controller/dialects/siemens840d/` (docs/dialects.md "Siemens 840D sl"):

| module | role |
|---|---|
| `__init__.py` | `build_siemens` and `register_dialect("siemens840d", build_siemens, replace=True)`; `native_params` (block-time model + `controller-siemens840d` + project settings) |
| `syntax.py` | its own program reader (one program per file, labels, control structures); passed as `Dialect.parser` |
| `values.py` | expressions, R parameters, local user data, system variables, frames |
| `handler.py` | `SiemensHandler`, the block semantics |
| `cycles.py`, `milling.py`, `turning.py` | drilling, patterns, pockets and slots, CYCLE800, CYCLE832, CYCLE95/97/93 |
| `tables.py` | G functions and groups, keywords, alarm numbers with built-in texts, the word list, the unmodelled functions |
| `kit.py` | runtime shared with the Heidenhain dialects: frame-to-core-stage mapping, table kinematics, TCP stages |

The builder takes the variant from the machine, resolves the parameters, binds the parser to the machine's
axis letters, and builds the alarm table with the controller database's texts; the 840D sl has no numbered
macro variables, so its `sysvars` refuse them with a twin alarm:

```python
def build_siemens(machine, settings, controller_db, db_id: str = "") -> Dialect:
    variant = machine.variant
    params = native_params(settings, "controller-siemens840d")
    ...
    axes = frozenset(a.letter for path in machine.paths for a in path)
    parser = functools.partial(build_store, axes=axes, max_depth=params["structure_nesting_max"])
    alarms = AlarmTable("siemens840d", ALARM_CODES, ALARM_TEXTS, controller_db, db_id)
    power_on = dict(POWER_ON_LATHE if variant == "T" else POWER_ON_MILL)
    return Dialect("siemens840d", variant, LexerRules(), {}, power_on, alarms, NoNumberedVariables(), SiemensHandler,
                   params, controller_db, db_id, None, None, frozenset(), {"words": WORD_LIST}, parser)


register_dialect("siemens840d", build_siemens, replace=True)
```

Its defaults are a model file whose header says why most values are estimates (the machine-data manual could
not be read) and whose entries name the machine data they stand for:

```yaml
format: cnctwin-model-params
version: 1
model: controller-siemens840d
params:
  circle_error_const_m:
    value: 0.01
    unit: mm
    range: [0.001, 0.1]
    src: est
    note: "MD 21000 CIRCLE_ERROR_CONST, constant part of the permissible circle end-point error (alarm 14040); 0.01 mm assumed as delivered"
```

`db/controllers/siemens-840d-sl.yaml` holds the alarms and words with a status header saying which manual
edition the texts were read in; `db/machines/dmg-dmu50-3rd-siemens.yaml` selects the dialect
(`controller: {dialect: siemens840d, alarms: siemens-840d-sl, ...}`); `siemens840d` is in `DIALECTS`,
`hooks.PLUGIN_PACKAGES` lists `dialects.siemens840d`, `twin new` writes `op10.mpf`, and
`examples/ti-slot-throughtool` is a project in the dialect.
