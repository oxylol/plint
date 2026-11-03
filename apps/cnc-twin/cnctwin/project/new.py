"""``twin new``: a minimal valid project for a given machine.

Usage::

    twin new <dir> --machine <id> [--material <id>] [--coolant <id>] [--name TEXT] [--force]

Writes ``<dir>/project.yaml`` and ``<dir>/op10.<ext>`` (an empty program in the machine's dialect:
start, units, end) and checks that the project resolves. Everything a project does not need to
say comes from db/_templates/project.yaml: environment, stock (the template's block and form),
production, recording, outputs and the ``physics`` section (architecture 5.1: base tick 1 us,
L0 stride rules ``max_deg_per_step``, ``steps_per_tooth``, macro step ``macro_ipo`` L1 ticks ...).
The material defaults to the template's (al6061-t6) and the coolant to the template's fluid
(emulsion-8pct) when they are in the database, else to the first id in sorted order; ``tools.list``
and the workholding are empty (the project is valid but cuts nothing until they are added).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..core.errors import InputError
from ..db.database import DEFAULT_ROOT, Database
from ..db.yamlio import load_file
from .loader import PROJECT_FILE, resolve
from .schema import CURRENT_VERSION, FORMAT

TEMPLATE = Path(DEFAULT_ROOT) / "_templates" / "project.yaml"
PROGRAM_SUFFIX = {"fanuc": ".nc", "haas": ".nc", "siemens840d": ".mpf", "heidenhain_klartext": ".h",
                  "heidenhain_iso": ".i"}


def program_skeleton(dialect: str, name: str = "OP10") -> str:
    """The shortest complete program of a dialect: start, metric/absolute, end."""
    if dialect in ("fanuc", "haas"):
        return f"%\nO0010 ({name})\nG21 G17 G40 G49 G80 G90\nM30\n%\n"
    if dialect == "siemens840d":
        return f"; {name}\nG71 G17 G40 G90\nM30\n"
    if dialect == "heidenhain_klartext":
        return f"0 BEGIN PGM {name} MM\n1 END PGM {name} MM\n"
    if dialect == "heidenhain_iso":
        return f"%{name} G71 *\nN10 G90 *\nN99999999 %{name} G71 *\n"
    raise InputError(f"no program skeleton for dialect {dialect!r}")


def _pick(db: Database, kind: str, preferred: str | None, template_choice: str | None) -> str:
    ids = db.ids(kind)
    valid = [i for i in ids if not db.object(kind, i).errors()]
    if preferred is not None:
        if preferred not in ids:
            raise InputError(f"no {kind} {preferred!r} in the database (have: {', '.join(ids[:12])}"
                             f"{' ...' if len(ids) > 12 else ''})")
        return preferred
    if template_choice in valid:
        return template_choice
    if not valid:
        raise InputError(f"the database has no valid {kind}")
    return valid[0]


def minimal_project(db: Database, machine: str, *, material: str | None = None, coolant: str | None = None,
                    name: str | None = None) -> tuple[dict, str, str]:
    """``(project data, program file name, program text)`` of a minimal project."""
    template = load_file(TEMPLATE).data
    machine_obj = db.machine(machine)
    dialect = str(machine_obj.controller.dialect)
    material_id = _pick(db, "material", material, (template.get("material") or {}).get("id"))
    coolant_id = _pick(db, "coolant", coolant, (template.get("coolant") or {}).get("fluid"))
    program = "op10" + PROGRAM_SUFFIX.get(dialect, ".nc")
    identity = {"translate_mm": [0, 0, 0], "rotate_deg": [0, 0, 0], "order": "xyz"}
    stock = {k: v for k, v in (template.get("stock") or {}).items() if k in ("kind", "size_mm", "tolerance_mm", "form")}
    physics = dict(template.get("physics") or {})
    physics["resolved"] = {}
    data = {
        "format": FORMAT,
        "version": CURRENT_VERSION,
        "name": name or f"New project on {machine}",
        "description": "",
        "seed": template.get("seed", 1),
        "machine": machine,
        "material": {"id": material_id},
        "environment": template.get("environment"),
        "stock": stock,
        "tools": {"library": None, "list": []},
        "coolant": {"fluid": coolant_id, "tank_fill_fraction": (template.get("coolant") or {}).get("tank_fill_fraction")},
        "operations": [{"name": "OP10", "program": program,
                        "setup": {"workholding": [], "stock_pose": dict(identity), "work_offsets_mm": {},
                                  "fixture_pose": dict(identity)}}],
        "production": template.get("production"),
        "physics": physics,
        "recording": template.get("recording"),
        "outputs": template.get("outputs"),
        "machine_data": None,
    }
    if data["coolant"]["tank_fill_fraction"] is None:
        del data["coolant"]["tank_fill_fraction"]
    return data, program, program_skeleton(dialect)


def render(data: dict) -> str:
    """Project YAML text (sections in template order, one blank line between them)."""
    from ..db.yamlio import dumps
    blocks = [f"# CNC Twin project (twin new). Paths are relative to this file; db ids from {DEFAULT_ROOT}.\n"]
    for key, value in data.items():
        blocks.append(dumps({key: value}))
    return "\n".join(blocks)


def new_project(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twin new", description="create a minimal CNC Twin project")
    parser.add_argument("dir", help="project directory (created if missing)")
    parser.add_argument("--machine", required=True, help="machine id (db/machines)")
    parser.add_argument("--material", default=None, help="material id (default: the template's)")
    parser.add_argument("--coolant", default=None, help="coolant id (default: the template's)")
    parser.add_argument("--name", default=None, help="project name")
    parser.add_argument("--db", default=str(DEFAULT_ROOT), help="database root")
    parser.add_argument("--force", action="store_true", help="overwrite an existing project.yaml")
    args = parser.parse_args(argv)
    target = Path(args.dir)
    project_file = target / PROJECT_FILE
    if project_file.exists() and not args.force:
        print(f"{project_file} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    try:
        db = Database.load(args.db)
        data, program, program_text = minimal_project(db, args.machine, material=args.material,
                                                      coolant=args.coolant, name=args.name)
        target.mkdir(parents=True, exist_ok=True)
        program_file = target / program
        if not program_file.exists() or args.force:
            program_file.write_text(program_text, encoding="utf-8")
        project_file.write_text(render(data), encoding="utf-8")
        resolve(project_file, db)
    except InputError as error:
        print(f"twin new: {error}", file=sys.stderr)
        return 1
    print(f"created {project_file} (machine {args.machine}, material {data['material']['id']}, "
          f"coolant {data['coolant']['fluid']}; program {program})")
    return 0


__all__ = ["new_project", "minimal_project", "program_skeleton", "render"]
