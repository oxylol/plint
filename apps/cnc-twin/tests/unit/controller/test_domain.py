import numpy as np
import pytest

from cnctwin.controller.domain import ControllerDomain, setup_from_project, split_paths
from cnctwin.controller.ir import Kind
from cnctwin.controller.schema import SCHEMA, ControllerSettings, settings_from
from cnctwin.controller.machine import build_machine
from cnctwin.core.domain import Domain, RunContext
from cnctwin.core.state import SimState
from cnctwin.db.simap import wrap as simap
from harness import MILL, wrap


def _ctx(program: str, machine=MILL):
    project = simap({
        "machine": machine,
        "physics": {"controller": {"arc_tolerance_mm": 0.005, "macro_statement_ms": 0.1}},
        "operations": [{"name": "OP10", "program": "op10.nc", "program_text": program,
                        "setup": {"work_offsets_mm": {"G54": [-400, -200, -300]}}, "nozzle_positions": {1: 12}}],
        "tools": {"list": [{"number": 1, "length_offset_mm": 120.0, "radius_offset_mm": 5.0,
                            "wear_offsets_mm": {"length": -0.02, "radius": 0.01}}]},
    })
    return RunContext(project=project, db=None, run_dir=None, seed=1, tick_s=1e-6, ipo_ticks=1000, macro_ipo=10)


def test_domain_protocol_init_produce_and_findings():
    dom = ControllerDomain()
    assert isinstance(dom, Domain)
    ctx = _ctx(wrap("G90 G54 G43 H1 G00 X10. Z0\nG00 G01 X20. F100.\nG99.9"))
    state = SimState()
    dom.init(ctx, state)
    assert state.controller is dom.controller(ctx, state).state
    commands = dom.produce(ctx, state, 0, 0)
    assert commands[0].kind == Kind.RAPID
    assert commands[0].target["X"] == pytest.approx(-0.39)
    assert commands[0].target["Z"] == pytest.approx(-0.3 + 0.11998)
    dom.macro_step(ctx, state, state.bus_prev, state.bus_next)
    messages = [f.message for f in state.findings]
    assert any("G00 and G01" in m for m in messages)
    alarms = [f for f in state.findings if f.category.value == "alarm"]
    assert alarms and alarms[0].message.startswith("PS0010") and alarms[0].line == 5
    values = dom.last_channels
    assert values["controller.alarm"][0] and not values["controller.alarm"][1]
    capacity = dom.controller(ctx, state).executors[0].capacity()
    assert values["controller.buffer_fill"][0] == pytest.approx(len(state.controller.paths[0].buffer) / capacity)
    assert dom.kernel_inputs(ctx, state, state.bus_prev) is None
    assert {c.name for c in dom.channels()} >= {"controller.buffer_fill", "controller.waiting"}


def test_domain_rebinds_after_a_restored_state():
    dom = ControllerDomain()
    ctx = _ctx(wrap("G00 X1.\nG00 X2."))
    state = SimState()
    dom.init(ctx, state)
    first = dom.produce(ctx, state, 0, 0)
    assert [c.kind for c in first] == [Kind.RAPID, Kind.RAPID, Kind.STOP]
    arrays, meta = state.controller.to_arrays()
    from cnctwin.controller.state import ControllerState
    state.controller = ControllerState.from_arrays(arrays, meta)
    assert dom.controller(ctx, state).state is state.controller
    assert state.controller.paths[0].buffer == first
    dom.retire(ctx, state, 0, 10, 1_000_000)
    assert dom.produce(ctx, state, 0, 1_000_000) == [] and dom.done(ctx, state)
    dom.cycle_start(ctx, state, 0, 2_000_000)
    again = dom.produce(ctx, state, 0, 2_000_000)
    assert [c.kind for c in again] == [Kind.RAPID, Kind.RAPID, Kind.STOP]
    report = dom.finalize(ctx, state)
    assert report["paths"][0]["blocks"] >= 3


def test_operator_panel_single_block_and_optional_stop():
    dom = ControllerDomain()
    ctx = _ctx(wrap("G00 X1.\nM01\nG00 X2."))
    state = SimState()
    dom.init(ctx, state)
    dom.set_operator(state, 0, single_block=True, optional_stop=True)
    first = dom.produce(ctx, state, 0, 0)
    assert [c.kind for c in first] == [Kind.RAPID, Kind.STOP] and first[1].stop == "single_block"
    dom.retire(ctx, state, 0, 2, 10_000)
    dom.cycle_start(ctx, state, 0, 10_000)
    second = dom.produce(ctx, state, 0, 10_000)
    assert [c.stop for c in second] == ["M01"]
    with pytest.raises(AttributeError):
        dom.set_operator(state, 0, turbo=True)


def test_setup_from_project_reads_offsets_tools_and_settings():
    ctx = _ctx(wrap("G00 X1."))
    state = SimState()
    setup = setup_from_project(ctx, state)
    assert setup.settings.arc_tolerance_m == pytest.approx(5e-6)
    assert setup.settings.macro_statement_s == pytest.approx(1e-4)
    assert setup.work_offsets_m[0]["G54"] == pytest.approx([-0.4, -0.2, -0.3])
    tool = setup.tools[0]
    assert tool.length_m == pytest.approx(0.12) and tool.length_wear_m == pytest.approx(-2e-5)
    assert tool.nozzle_position == 12


def test_settings_schema_and_split_paths():
    from cnctwin.db.schema import Ctx
    ctx = Ctx()
    SCHEMA.check({"pocket_calculator": True, "arc_tolerance_mm": 0.01, "wait_m_range": [100, 200]}, (), None, ctx)
    assert not [p for p in ctx.problems if p.severity == "error"], ctx.problems
    bad = Ctx()
    SCHEMA.check({"pocket_calculatr": True}, (), None, bad)
    assert bad.problems
    s = settings_from({"same_group": "alarm", "reference_points_mm": {2: {"X": -10}}, "part_count_codes": 30})
    assert isinstance(s, ControllerSettings) and s.same_group == "alarm"
    assert s.reference_points_m == {2: {"X": pytest.approx(-0.01)}} and s.part_count_codes == [30]
    assert split_paths("$1\nG00 X1.\n$2\nG00 X2.\n", 2) == ["G00 X1.", "G00 X2."]
    assert split_paths("G00 X1.", 2) == ["G00 X1.", ""]


def test_machine_model_of_the_real_machine_files():
    from cnctwin.db.yamlio import load_file
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3] / "db" / "machines"
    vf2 = root / "haas-vf2ss.yaml"
    if vf2.exists():
        model = build_machine(load_file(vf2).data)
        assert [a.letter for a in model.paths[0]] == ["X", "Y", "Z"] and model.variant == "M"
        assert model.lookahead_blocks > 0 and model.block_processing_s > 0
    swiss = root / "star-sr20riv.yaml"
    if swiss.exists():
        model = build_machine(load_file(swiss).data)
        assert len(model.paths) == 2 and model.paths[1][0].name == "X2" and model.paths[1][0].diameter
    assert np.isfinite(1.0)
