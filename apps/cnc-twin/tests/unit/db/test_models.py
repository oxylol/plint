import math
import textwrap

import pytest

from cnctwin.core.errors import InputError, SourceError
from cnctwin.db import models
from dbfixture import write_db


@pytest.fixture()
def root(tmp_path):
    models.clear_cache()
    yield write_db(tmp_path)
    models.clear_cache()


def _write(root, name, text):
    (root / "models" / f"{name}.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
    models.clear_cache()


def test_params_are_si_and_records_as_written(root):
    params = models.params("test-model", root)
    assert params["gap_m"] == pytest.approx(5e-4)
    assert params["angle_rad"] == pytest.approx(math.pi / 2)
    assert params["ratio"] == 2.0 and params["label"] == "steel"
    assert models.records("test-model", root)["gap_m"]["value"] == 0.5
    assert models.units("test-model", root)["gap_m"] == "mm"


def test_name_must_carry_the_si_unit(root):
    _write(root, "bad-names", """\
        format: cnctwin-model-params
        version: 1
        model: bad-names
        src: src-a
        params:
          gap_mm: {value: 0.5, unit: mm}
        """)
    with pytest.raises(InputError) as info:
        models.params("bad-names", root)
    assert "names carry the SI unit" in str(info.value) and "bad-names.yaml:6" in str(info.value)
    _write(root, "bad-unit", """\
        format: cnctwin-model-params
        version: 1
        model: bad-unit
        src: src-a
        params:
          gap: {value: 0.5, unit: mm}
          speed_m_s: {value: 1, unit: furlong}
        """)
    with pytest.raises(InputError) as info:
        models.params("bad-unit", root)
    text = str(info.value)
    assert "must carry the SI unit" in text and "furlong" in text


def test_sources_and_ranges(root):
    _write(root, "no-src", """\
        format: cnctwin-model-params
        version: 1
        model: no-src
        params:
          ratio: {value: 20, unit: "-", range: [1, 10], src: est, note: "assumed"}
          other: {value: 1, unit: "-", src: unknown-key}
        """)
    with pytest.raises(InputError) as info:
        models.params("no-src", root)
    text = str(info.value)
    assert "outside its range" in text and "unknown-key" in text
    _write(root, "only-src", """\
        format: cnctwin-model-params
        version: 1
        model: only-src
        params:
          other: {value: 1, unit: "-", src: unknown-key}
        """)
    with pytest.raises(SourceError):
        models.params("only-src", root)
    with pytest.raises(InputError):
        models.params("not-there", root)
