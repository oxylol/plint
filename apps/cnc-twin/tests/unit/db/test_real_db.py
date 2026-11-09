"""Every real file under db/ and validation/cases/ loads, validates and builds its SI view."""

from pathlib import Path

import pytest

from cnctwin.db import Database
from cnctwin.db import schemas as S

ROOT = Path("/opt/plint/apps/cnc-twin/db")
VALIDATION = Path("/opt/plint/apps/cnc-twin/validation/cases")


def _files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.yaml")) if folder.is_dir() else []


@pytest.fixture(scope="module")
def real_db():
    return Database.load(ROOT, validation_root=VALIDATION)


@pytest.fixture(scope="module")
def strict_db():
    return Database.load(ROOT, validation_root=VALIDATION, strict=True, raise_on_error=False)


@pytest.mark.parametrize("folder", sorted(set(S.FOLDERS.values())) + ["validation"])
def test_every_file_of_a_folder_is_valid(real_db, folder):
    directory = VALIDATION if folder == "validation" else ROOT / folder
    files = _files(directory)
    if not files:
        pytest.skip(f"{directory} has no files yet")
    names = {str(p.relative_to(ROOT.parent)) for p in files}
    errors = [str(p) for p in real_db.validate_all() if p.is_error and p.file in names]
    assert errors == [], "\n".join(errors)


@pytest.mark.parametrize("folder", sorted(set(S.FOLDERS.values())) + ["validation"])
def test_every_folder_is_clean_in_strict_mode(strict_db, folder):
    """`twin db check --strict`: no error and no warning in any folder (every estimate says how it was
    derived, every bibliography key names one work described one way, no value waits for migration)."""
    directory = VALIDATION if folder == "validation" else ROOT / folder
    files = _files(directory)
    if not files:
        pytest.skip(f"{directory} has no files yet")
    names = {str(p.relative_to(ROOT.parent)) for p in files}
    problems = [str(p) for p in strict_db.validate_all() if p.file in names]
    assert problems == [], "\n".join(problems)


def test_every_object_builds(real_db):
    failures = []
    for (kind, obj_id), obj in sorted(real_db.objects.items()):
        if kind == "bibliography" or obj.errors():
            continue
        try:
            real_db.get(kind, obj_id).to_dict()
        except Exception as error:   # report every object that does not build, not just the first
            failures.append(f"{kind} {obj_id}: {type(error).__name__}: {error}")
    assert failures == [], "\n".join(failures)


def test_no_general_problems(real_db):
    assert [str(p) for p in real_db.general if p.is_error] == []


def test_model_files_read_through_params():
    from cnctwin.db import models
    files = _files(ROOT / "models")
    if not files:
        pytest.skip("db/models has no files yet")
    for path in files:
        assert isinstance(models.params(path.stem), dict)


def test_no_file_waits_for_migration(real_db):
    """The files are written in the current key forms (the migration table maps nothing of them), so a file reads
    as written and matches its template."""
    assert [f"{m.file}:{m.line}: {m.path_from} -> {m.path_to} ({m.rule})" for m in real_db.migrations()] == []


def _without(value, drop=("src",)):
    if isinstance(value, dict):
        return {k: _without(v, drop) for k, v in value.items() if k not in drop}
    if isinstance(value, list):
        return [_without(v, drop) for v in value]
    return value


def test_the_siemens_dmu50_is_its_base_with_another_controller(real_db):
    obj = real_db.object("machine", "dmg-dmu50-3rd-siemens")
    assert obj.extends == "dmg-dmu50-3rd"
    assert set(obj.raw) <= {"format", "version", "id", "model", "extends", "src", "controller"}
    siemens = real_db.machine("dmg-dmu50-3rd-siemens").to_dict()
    base = real_db.machine("dmg-dmu50-3rd").to_dict()
    identity = ("controller", "id", "model")
    # section sources are pinned on the merged object (the same effective source): compared without them
    assert _without({k: v for k, v in siemens.items() if k not in identity}) == \
        _without({k: v for k, v in base.items() if k not in identity})
    assert siemens["controller"]["dialect"] == "siemens840d" and base["controller"]["dialect"] == "heidenhain_klartext"
    assert siemens["controller"]["alarms"] == "siemens-840d-sl"
