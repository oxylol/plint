"""ToolingDomain with a fake run context, state and bus: init, kernel inputs, macro step, checkpoint."""

import json

import numpy as np
import pytest

from cnctwin.core.bus import Bus, PathFrame, ToolHeat, ToolHolding, ToolWear
from cnctwin.core.domain import Domain
from cnctwin.core.state import SimState
from cnctwin.core.trace import TraceLog
from cnctwin.tooling import ToolingCatalog, ToolingDomain, ToolingState
from tests.unit.tooling.fixtures import HOLDER, TOOLS, endmill


class Ctx:
    def __init__(self, project):
        self.project = project
        self.db = None
        self.trace = TraceLog()


def _project():
    return {"physics": {"edge_element_mm": 0.5, "tooling": {"rake_class_deg": 2}},
            "machine": {"spindles": [{"id": "main", "kind": "milling", "runout_um": 2}]},
            "tools": {"list": [
                {"number": 1, "tool": endmill(), "holder": HOLDER, "stickout_mm": 35, "runout_um": 5},
                {"number": 2, "tool": TOOLS["twist_drill"], "holder": HOLDER, "stickout_mm": 70,
                 "initial_wear": {"edge_radius_um": 15}},
            ]}}


@pytest.fixture(scope="module")
def run():
    domain = ToolingDomain()
    ctx = Ctx(_project())
    state = SimState()
    domain.init(ctx, state)
    return domain, ctx, state


def test_protocol_and_init(run):
    domain, ctx, state = run
    assert isinstance(domain, Domain)
    assert len(state.tools.slots) == 2 and isinstance(state.tooling, ToolingState)
    assert state.tooling.geometry(0).runout_source == "measured"
    assert np.allclose(state.tooling.edge_model(1).elem_edge_radius_m, 15e-6)    # initial wear applied
    assert len(ctx.trace._memory) == 2
    assert {c.name for c in domain.channels()} >= {"tooling.radius_eff_max_m", "tooling.tip_shift_m"}


def test_kernel_inputs_follow_the_active_slot(run):
    domain, ctx, state = run
    bus = Bus()
    bus.motion.paths = [PathFrame(path=0, tool_slot=1), PathFrame(path=1, tool_slot=-1)]
    inputs = domain.kernel_inputs(ctx, state, bus)
    assert list(inputs.slot) == [1, -1]
    assert inputs.edges[0] is state.tooling.effective[1]
    assert inputs.edges[1].elem_r_m.shape == (0, 0)


def test_macro_step_composes_from_the_previous_bus(run):
    domain, ctx, state = run
    nt, ne = state.tooling.nominal(0).elem_r_m.shape
    prev = Bus()
    prev.wear.tools[0] = ToolWear(radial_wear_m=np.full((nt, ne), 2e-6), axial_wear_m=np.zeros((nt, ne)),
                                  edge_radius_m=np.full((nt, ne), 10e-6))
    prev.thermal.tools[0] = ToolHeat(radial_growth_m=np.full((nt, ne), 1e-6), length_growth_m=4e-6)
    prev.holding.tools[0] = ToolHolding(pullout_m=3e-6)
    before = int(state.tooling.version[0])
    domain.macro_step(ctx, state, prev, Bus())
    eff = state.tooling.edge_model(0)
    assert int(state.tooling.version[0]) == before + 1
    assert np.allclose(eff.elem_r_m - state.tooling.nominal(0).elem_r_m, -1e-6)
    domain.macro_step(ctx, state, prev, Bus())
    assert int(state.tooling.version[0]) == before + 1            # unchanged inputs: no new version
    values = domain.channel_values(ctx, state, _bus_with_slot(0))
    assert values["tooling.radius_change_m"][0] == pytest.approx(-1e-6)
    assert values["tooling.tip_shift_m"][0] > 3e-6


def _bus_with_slot(slot):
    bus = Bus()
    bus.motion.paths = [PathFrame(tool_slot=slot)]
    return bus


def test_checkpoint_roundtrip_is_exact(run):
    domain, ctx, state = run
    arrays, meta = state.tooling.to_arrays()
    json.dumps(meta)
    restored = ToolingState.from_arrays({k: np.array(v) for k, v in arrays.items()}, json.loads(json.dumps(meta)))
    for slot in range(2):
        for name in restored.effective[slot]._fields:
            assert np.array_equal(getattr(restored.effective[slot], name), getattr(state.tooling.effective[slot], name))
        assert np.array_equal(restored.envelope(slot).r_max_m, state.tooling.envelope(slot).r_max_m)
        assert restored.geometry(slot).props.tool_mass.mass_kg == state.tooling.geometry(slot).props.tool_mass.mass_kg
    again, meta2 = restored.to_arrays()
    assert meta2 == json.loads(json.dumps(meta))
    for key, value in arrays.items():
        assert np.array_equal(again[key], value, equal_nan=value.dtype.kind == "f"), key


def test_finalize_and_catalog(run):
    domain, ctx, state = run
    report = domain.finalize(ctx, state)
    json.dumps(report)
    assert report["tools"][0]["edge_updates"] >= 1
    catalog = ToolingCatalog(state.tooling)
    info = catalog.tool(1)
    assert info.kind == "twist_drill" and info.n_flutes == 2 and info.gullet_area_m2 > 0
    assert catalog.edge_model(0) is state.tooling.effective[0]
    assert catalog.tool(5) is None
