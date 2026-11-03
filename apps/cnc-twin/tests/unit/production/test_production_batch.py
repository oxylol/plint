"""Batch aggregation: per-part records, sums over the batch, projection, operator model."""

import json

import numpy as np
import pytest

from cnctwin.production import (EnergyInputs, OperatorModel, PartInputs, ProductionAggregator, ToolState,
                                coolant_states, coolant_use, tool_cost, with_tools)

from production_testkit import FLUID, MACHINE, block, machine


def _part_blocks(part: int, cut_ticks: int = 3_000_000):
    t0 = part * 10_000_000
    return with_tools([
        block(0, t0, t0 + 2_000_000, "tool_change", text="T1 M06", part=part),
        block(1, t0 + 2_000_000, t0 + 2_100_000, "coolant", text="M08", part=part),
        block(2, t0 + 2_100_000, t0 + 2_100_000 + cut_ticks, "linear", part=part, ticks={"cutting": cut_ticks}),
        block(3, t0 + 2_100_000 + cut_ticks, t0 + 5_200_000, "stop", text="M00", part=part),
        block(4, t0 + 5_200_000, t0 + 5_300_000, "coolant", text="M09", part=part),
        block(5, t0 + 5_300_000, t0 + 5_400_000, "stop", text="M30", part=part),
    ])


def _signals(seconds: float = 5.4):
    n = int(round(seconds / 0.01))
    omega = np.full(n, 1000.0)
    omega[0] = 0.0
    velocity = np.zeros((n, 3))
    velocity[:, 0] = 0.05
    return EnergyInputs(np.full(n, 0.01), omega, np.full(n, 2000.0), velocity, ("X", "Y", "Z"))


def _project(parts=3, **operator):
    return {"production": {"parts": parts, "operator": {"load_s": 30, "unload_s": 20, "door_s": 4, "clean_s": 15,
                                                        "inspect_every": 0, "inspect_s": 60, **operator}},
            "coolant": {"fluid": FLUID, "concentration_pct": 8},
            "tools": {"list": [{"number": 1, "tool": {"id": "em", "type": "square_endmill", "price_eur": 90,
                                                      "regrinds": 2}},
                               {"number": 2, "tool": {"id": "dr", "type": "twist_drill", "price_eur": 40}}]}}


TOOLS = [  # cumulative life used at the end of each part; T2 is changed during part 2
    [ToolState(1, life_used_frac=0.1, VB_max_m=1e-5, wear_offsets_m={"radius": -1e-6}),
     ToolState(2, life_used_frac=0.4)],
    [ToolState(1, life_used_frac=0.2, VB_max_m=2e-5, wear_offsets_m={"radius": -2e-6}),
     ToolState(2, life_used_frac=0.8)],
    [ToolState(1, life_used_frac=0.3, VB_max_m=3e-5, wear_offsets_m={"radius": -3e-6}),
     ToolState(2, life_used_frac=0.2, changes=1)],
]


def _aggregate(planned=3, **operator):
    aggregator = ProductionAggregator.from_project(_project(planned, **operator), machine=MACHINE, tick_s=1e-6)
    for part in range(3):
        aggregator.add_part(PartInputs(part, _part_blocks(part), _signals(), removed_mass_kg=0.2, tools=TOOLS[part]))
    return aggregator


def test_parts_and_batch_sums():
    aggregator = _aggregate()
    document = aggregator.result()
    parts = document["per_part"]
    assert [p["machine_ticks"] for p in parts] == [5_400_000] * 3
    assert all(p["breakdown"]["total_ticks"] == p["machine_ticks"] for p in parts)
    # 30 + 20 + 15 + 2 x 4 door + the M00: 2 x 4 door + 30 s action (production-operator stop_s)
    assert parts[0]["operator"]["components_s"]["stops_s"] == pytest.approx(38.0)
    assert parts[0]["takt_s"] == pytest.approx(5.4 + 73.0 + 38.0)
    batch = document["batch"]
    assert batch["parts"] == 3 and batch["machine_s"] == pytest.approx(16.2)
    assert batch["energy_J"] == pytest.approx(sum(p["energy"]["total_J"] for p in parts))
    assert batch["energy_by_consumer_J"]["pumps"] == pytest.approx(3 * 3.2 * 937.5)   # M08 up to the M09 block: 3.2 s
    used = {t["number"]: t["life_used_frac"] for t in batch["tools"]}
    assert used[1] == pytest.approx(0.3) and used[2] == pytest.approx(0.4 + 0.4 + 0.4)   # 1 - 0.8 + 0.2
    t1 = next(t for t in batch["tools"] if t["number"] == 1)
    assert t1["cost_eur"] == pytest.approx(90.0 / 3 * 0.3) and t1["parts_per_edge"] == pytest.approx(10.0)
    assert batch["tool_changes"] == 1
    progression = document["progression"]
    assert progression["tools"]["T1"]["VB_max_m"] == [1e-5, 2e-5, 3e-5]
    assert progression["tools"]["T1"]["wear_offsets_m"]["radius"] == [-1e-6, -2e-6, -3e-6]
    assert progression["tools"]["T2"]["changes"] == [0, 0, 1]
    assert document["parts"] == 3 and document["cycle_time_s"] == pytest.approx(5.4)
    assert document["cut_time_s"] == pytest.approx(3.0) and document["per_part_s"] == parts[0]["takt_s"]


def test_projection_of_a_larger_batch():
    document = _aggregate(planned=10).result()
    assert document["parts_simulated"] == 3 and document["parts_projected"] == 7
    projected = [p for p in document["per_part"] if not p["simulated"]]
    changes = {n: sum(t["changes"] for p in projected for t in p["tools"] if t["number"] == n) for n in (1, 2)}
    # T1: 0.3 + 7 x 0.1 = 1.0 -> one new edge; T2: 0.2 + 7 x 0.4 = 3.0 -> three
    assert changes == {1: 1, 2: 3}
    replace_s = OperatorModel.from_project({}).tool_replace_s
    with_change = next(p for p in projected if any(t["changes"] for t in p["tools"]))
    plain = next(p for p in projected if not any(t["changes"] for t in p["tools"]))
    assert with_change["takt_s"] - plain["takt_s"] == pytest.approx(
        replace_s * sum(t["changes"] for t in with_change["tools"]))
    assert len(document["progression"]["part"]) == 10
    assert document["batch"]["tool_changes"] == 1 + 4


def test_operator_inspection_and_pallets():
    model = OperatorModel.from_project(_project(operator={})["production"] and
                                       {"production": {"operator": {"inspect_every": 5, "inspect_s": 60}}})
    assert [model.inspected(k) for k in range(6)] == [True, False, False, False, True, False]
    pallet = OperatorModel.from_project({}, machine(pallet_changer={"pallets": 2, "change_time_s": 12}))
    time = pallet.part_time(operations=1, stops={}, part_index=1, machine_time_s=300.0)
    assert time.machine_wait_s == pytest.approx(12.0) and time.takt_s == pytest.approx(312.0)
    slow = pallet.part_time(operations=1, stops={}, part_index=1, machine_time_s=50.0)
    assert slow.machine_wait_s == pytest.approx(12.0 + 73.0 - 50.0)
    optional = OperatorModel.from_project({"production": {"operator": {"optional_stop": True}}})
    assert optional.part_time(operations=1, stops={"M01": 2}, part_index=0, machine_time_s=10.0).m01 == 2


def test_coolant_states_and_consumption():
    blocks = _part_blocks(0)
    states = coolant_states(blocks, "haas", machine(), FLUID, 1e-6)
    assert states.pump_on_s == {"flood": pytest.approx(3.2)} and not states.conveyor_commanded
    use = coolant_use(FLUID | {"kind": "emulsion"}, chip_mass_kg=0.5, pumped_m3=0.01, mql_on_s=0.0, elapsed_s=3600.0,
                      tank_m3=0.208, concentration_frac=0.08)
    # drag-out 42 ml/kg x 0.5 kg = 21 ml; mist 2e-6 x 10 l; 8 % of it is concentrate
    assert use["drag_out_m3"] == pytest.approx(21e-6)
    assert use["concentrate_m3"] == pytest.approx(0.08 * (21e-6 + 2e-8))
    assert use["evaporated_water_m3"] == pytest.approx(0.3e-3 * 0.208 / 0.35)       # 0.3 l/h/m2 over 0.594 m2
    assert use["cost_eur"] == pytest.approx(use["concentrate_m3"] * 1000.0 * 4.0)
    mql = coolant_use({"kind": "mql_oil", "cost_eur_per_l": 15.0}, chip_mass_kg=0.5, pumped_m3=0.0, mql_on_s=3600.0,
                      elapsed_s=3600.0, tank_m3=None)
    assert mql["oil_m3"] == pytest.approx(20e-6, rel=1e-3)                       # 20 ml/h default dosage


def test_tool_cost_formula():
    cost = tool_cost(90.0, 2, 0.05)
    assert cost["cost_per_edge_eur"] == pytest.approx(30.0) and cost["cost_per_part_eur"] == pytest.approx(1.5)
    assert cost["parts_per_edge"] == pytest.approx(20.0)
    assert tool_cost(None, 2, 0.05)["cost_per_part_eur"] is None
    assert tool_cost(90.0, 0, 0.0)["parts_per_edge"] is None


def test_production_json_is_deterministic(tmp_path):
    first = _aggregate().write(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path)
    second = _aggregate().write(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path)
    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_text())
    assert document["format"] == "cnctwin-production" and document["first_part"]["lines"]
