"""The real cases and bibliography as the suite reads them (database load, no harness physics)."""

from __future__ import annotations

import pytest

from cnctwin.db import Database
from cnctwin.db.schemas import VALIDATION_QUANTITIES
from cnctwin.validation import provenance as PV
from cnctwin.validation.cases import load_cases, per_test_table


@pytest.fixture(scope="module")
def db():
    return Database.load()


def test_every_case_loads_with_a_known_quantity_and_source(db):
    cases = load_cases(db)
    assert len(cases) == len(db.ids("validation_case"))
    for case in cases:
        assert case.quantity in VALIDATION_QUANTITIES
        assert case.src and case.src in db.bib
        assert case.reference_values or case.reference_curves
    assert [(c.quantity, c.id) for c in cases] == sorted((c.quantity, c.id) for c in cases)


def test_cases_filter_by_quantity_and_id(db):
    chip = load_cases(db, ["chip_ratio"])
    assert chip and all(c.quantity == "chip_ratio" for c in chip)
    one = load_cases(db, None, [chip[0].id])
    assert [c.id for c in one] == [chip[0].id]


def test_the_same_work_under_two_bibliography_keys_is_one_group(db):
    works = PV.works_from_db(db)
    assert works.same("klippel-2022-arxiv-orthogonal-data", "klippel-2022-ck45-cutting-database")
    assert works.same("klippel-2022-arxiv-orthogonal-data", "klippel-2024-ck45-orthogonal")
    assert works.same("ducobu-2017-ti6al4v-orthogonal-reference", "ducobu-2017-jc-ti6al4v")
    assert works.same("aouici-2010-cbn-h11-hard-turning", "aouici-2010-h11-cbn")
    assert not works.same("klippel-2022-arxiv-orthogonal-data", "ducobu-2017-jc-ti6al4v")


def test_per_test_tables_of_a_real_case(db):
    case = next(c for c in load_cases(db, ["chip_ratio"]) if c.id == "chipratio-orth-ck45-klippel2022-vc150")
    rows = per_test_table(case)
    assert len(rows) == 15 and rows[0].test_id == "V0091"
    assert rows[0].si("feed_mm_rev") == pytest.approx(1e-5)
    assert rows[0].si("edge_radius_um") == pytest.approx(41.7e-6)
