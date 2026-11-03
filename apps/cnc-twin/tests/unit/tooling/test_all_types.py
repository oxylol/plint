"""Every tool type of db/_templates/tool.yaml builds a consistent EdgeModel, and the catalogue items
of the tool database build as far as they are complete."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_KINDS, EdgeModel
from cnctwin.db.schemas import TOOL_TYPES
from cnctwin.paths import DB
from cnctwin.tooling import build_tool
from cnctwin.tooling.build import BUILDERS
from tests.unit.tooling.fixtures import HOLDER, NON_ROTATING, TOOLS, instance


def test_every_template_type_has_a_builder_and_a_fixture():
    assert set(TOOL_TYPES) == set(BUILDERS) == set(TOOLS)


@pytest.mark.parametrize("kind", sorted(TOOLS))
def test_edge_model_is_consistent(kind):
    rotating = kind not in NON_ROTATING
    g = build_tool(TOOLS[kind], HOLDER if rotating else None, instance())
    e = g.edge
    assert isinstance(e, EdgeModel)
    nt, ne = e.elem_r_m.shape
    assert nt >= 1 and ne >= 1 and e.elem_valid.sum() > 0
    for name in e._fields:
        arr = getattr(e, name)
        assert arr.flags.c_contiguous and arr.flags.writeable, name
        if arr.dtype.kind == "f":
            assert np.all(np.isfinite(arr)), name
    assert e.elem_kind.dtype == np.int32 and e.elem_valid.dtype == np.int8
    valid = e.elem_valid == 1
    assert np.all(e.elem_len_m[valid] > 0) and np.all(e.elem_db_m[valid] > 0)
    assert np.all((e.elem_kind >= 0) & (e.elem_kind < len(EDGE_KINDS)))
    assert np.all((e.elem_group >= 0) & (e.elem_group < len(g.groups.kind)))
    assert np.all(g.groups.kind[e.elem_group[valid]] == e.elem_kind[valid])
    assert np.array_equal(e.elem_station, np.tile(np.arange(ne), (nt, 1)))
    assert np.all((e.tooth_phase_rad >= 0) & (e.tooth_phase_rad < 2 * math.pi))
    assert np.all(e.elem_edge_radius_m[valid] > 0)
    assert np.all(np.abs(e.elem_rake_rad[valid]) <= math.pi / 2 + 1e-12)
    # element lengths respect the discretisation (margins and pads use the coarser margin elements)
    from cnctwin.core.abi import EDGE_MARGIN
    limit = 0.5e-3 * (1 + 1e-6)
    fine = valid & (e.elem_kind != EDGE_MARGIN)
    assert np.all(e.elem_db_m[fine] <= limit * 1.05)
    if rotating:
        assert len(g.envelope.kind) > 0 and g.props.tool_mass.mass_kg > 0
        assert np.all(e.elem_r_m[valid] >= 0)
    else:
        assert np.all(e.elem_r_m <= 1e-12) and g.insert is not None


@pytest.fixture(scope="module")
def database():
    from cnctwin.db import Database
    return Database.load()


def _valid(db, kind, obj_id) -> bool:
    from cnctwin.db.problems import Problem
    problems = db.problems_of(kind, obj_id)
    return not any(isinstance(p, Problem) and p.severity == "error" for p in problems)


def test_database_tools_build(database):
    """Every schema-valid tool of the database builds (SI views as cnctwin.db returns them) with a
    schema-valid holder, the tool material density read from the database."""
    holders = [h for h in database.ids("holder") if _valid(database, "holder", h)]
    assert holders, "no valid holder in the database"
    holder = database.get("holder", holders[0])
    built = 0
    for tool_id in database.ids("tool"):
        if not _valid(database, "tool", tool_id):
            continue
        item = database.get("tool", tool_id)
        rotating = str(item["type"]) not in NON_ROTATING
        g = build_tool(item, holder if rotating else None, instance(stickout_mm=40), tool_id=tool_id, db=database)
        assert g.edge.elem_valid.sum() > 0, tool_id
        assert all(np.all(np.isfinite(getattr(g.edge, f))) for f in g.edge._fields
                   if getattr(g.edge, f).dtype.kind == "f"), tool_id
        built += 1
    assert built > 0
