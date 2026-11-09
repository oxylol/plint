"""twin check's numerical-setting rules on a real project (cnctwin.core.check, docs/models/loop.md "twin check"): a
finishing pass 0.3 mm into a side wall breaks the lattice rule at the template's 0.25 mm and passes at 0.0625 mm or
under a refine box over it (docs/convergence.md 2); surface-memory bins finer than one per L0 step are warned
(docs/convergence.md 4.2)."""

from __future__ import annotations

from pathlib import Path

from cnctwin.core.check import check_project
from cnctwin.core.drive import project_variant
from cnctwin.paths import EXAMPLES

POCKET = Path(EXAMPLES) / "pocket-islands-al6061"
SIZE = (25.0, 12.0, 10.0)
AE = 0.3


def _program() -> str:
    """D10 at 10 000 rpm: plunge outside the block, one pass along +x 0.3 mm into the y = 12 mm wall, 5 mm deep."""
    y = SIZE[1] + 5.0 - AE
    return "\n".join(["%", "O07001 (CHECK RULES)", "G90 G17 G21 G40 G49 G80 G94 G54", "G53 G00 Z0.", "T1 M06",
                      f"G90 G54 G00 X-8. Y{y:.4f} S10000 M03", "G43 H01 Z25. M08", "Z2.", "G01 Z-5. F1000.",
                      "G01 X30. F1600.", "G00 Z25.", "M09", "M05", "G53 G49 Z0.", "M30", "%"]) + "\n"


def _project(tmp: Path, physics: dict) -> Path:
    def edit(data):
        data["cad"] = None
        data["stock"]["size_mm"] = list(SIZE)
        data["physics"].update(physics)
        entry = data["tools"]["list"][0]
        entry["tool"] = "helical-59482-mhsv-020-41000"
        data["tools"]["list"] = [entry]
        data["production"]["parts"] = 1
        data["outputs"] = {"report": [], "images": False, "video": False, "audio": False}
        op = data["operations"][0]
        op["setup"]["stock_pose"] = {"translate_mm": [-SIZE[0] / 2, -SIZE[1] / 2, 0], "rotate_deg": [0, 0, 0]}
        op["setup"]["work_offsets_mm"] = {"G54": [-381.0 - SIZE[0] / 2, -203.2 - SIZE[1] / 2, -510.0 + SIZE[2]]}
        op["setup"]["workholding"][0]["jaw_opening_mm"] = SIZE[1]
    return project_variant(POCKET, tmp, program_text=_program(), program_name="op.nc", edits=edit)


def _check(tmp: Path, physics: dict) -> dict:
    return check_project({"project_path": str(_project(tmp / "p", physics))}, tmp / "out")


def test_a_finishing_pass_on_a_coarse_lattice_is_warned_by_name(tmp_path):
    doc = _check(tmp_path / "coarse", {"lattice_mm": 0.25})
    lattice = doc["operations"][0]["lattice"]
    assert lattice["ok"] is False and len(lattice["passes"]) == 1
    p = lattice["passes"][0]
    # the fine estimate (cells of lattice / 4) brackets the 0.3 mm pass within a cell either side; the rule's
    # recommendation rests on the lower bound: 0.25 mm / 4 = 0.0625 mm <= 0.3 mm / 4 = 0.075 mm (factor 4)
    cell = lattice["zmap_cell_m"]
    assert cell <= 0.25e-3 / 4 + 1e-12
    assert p["ae_min_m"] - 1e-9 <= AE * 1e-3 <= p["ae_m"] + cell + 1e-9
    assert p["lattice_m"] == 0.25e-3 and p["refine_factor"] == 4 and p["line_first"] == 10
    assert any("thin radial engagement" in w and "line 10" in w for w in doc["warnings"])
    fine = _check(tmp_path / "fine", {"lattice_mm": 0.0625})
    assert fine["operations"][0]["lattice"]["ok"] is True
    assert not any("thin radial engagement" in w for w in fine["warnings"])
    # a refine box (stock frame S, mm) over the pass's whole swept region: lattice 0.25 / 4 there
    box = {"box": {"min_mm": [-16.0, 9.0, 3.0], "max_mm": [42.0, 25.0, 12.0]}, "factor": 4}
    refined = _check(tmp_path / "refined", {"lattice_mm": 0.25, "refine": [box]})
    assert refined["operations"][0]["lattice"]["ok"] is True


def test_memory_bins_finer_than_one_per_l0_step_are_warned(tmp_path):
    fine = _check(tmp_path / "b2880", {"memory_bins": 2880})
    entry = fine["operations"][0]["strides"][0]
    assert entry["memory_bins_max"] is not None and entry["memory_bins_max"] < 2880
    assert abs(entry["angle_per_step_deg"] - 360.0 / (60.0 / 10000.0) * entry["l0_run_ticks"] * 1e-6) < 1e-9
    assert any("physics.memory_bins 2880" in w for w in fine["warnings"])
    default = _check(tmp_path / "b720", {"memory_bins": 720})
    assert default["operations"][0]["strides"][0]["memory_bins_max"] >= 720
    assert not any("memory_bins" in w for w in default["warnings"])
