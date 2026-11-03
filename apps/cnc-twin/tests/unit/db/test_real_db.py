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


@pytest.mark.parametrize("folder", sorted(set(S.FOLDERS.values())) + ["validation"])
def test_every_file_of_a_folder_is_valid(real_db, folder):
    directory = VALIDATION if folder == "validation" else ROOT / folder
    files = _files(directory)
    if not files:
        pytest.skip(f"{directory} has no files yet")
    names = {str(p.relative_to(ROOT.parent)) for p in files}
    errors = [str(p) for p in real_db.validate_all() if p.is_error and p.file in names]
    assert errors == [], "\n".join(errors)


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
