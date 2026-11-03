"""A project with stated tolerances (stock, material batch, uncertainty.vary) and the surrogate runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REPORT_TESTS = Path(__file__).resolve().parents[1] / "report"
for folder in (HERE, FIXTURES, REPORT_TESTS):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

PROGRAM = """%
O1000 (STUDY POCKET)
G21 G17 G40 G49 G80 G90
T1 M06 (D2 4FL)
G54
S10000 M03
G00 X0. Y0.
G43 H1 Z20. M08
G01 Z-2. F300.
G01 X10. F600.
G01 Y4.
G01 X0.
G01 Y8.
G01 X10.
G01 Y12.
G01 X0.
G00 Z20.
M30
%
"""


VARY = "stock_size: true, tool_offsets_um: 5, runout_um: 3, clamping_force_rel: 0.1"


def project_text(runs: int = 64, method: str = "lhs", vary: str = VARY, library: bool = False) -> str:
    tools = ("  library: tools.csv\n  list: []\n" if library else
             "  list:\n    - {number: 1, tool: helical-89608-mhsv-020-40300, holder: haimer-40-720-32, stickout_mm: 30,\n"
             "       length_offset_mm: 120.0, radius_offset_mm: 1.0, runout_um: 5, comment: \"D2 4FL\"}\n")
    return f"""format: cnctwin-project
version: 1
name: "Study pocket"
seed: 77
machine: haas-vf2ss
material:
  id: al6061-t6
  batch: {{kc_scale_sigma: 0.04, hardness_sigma_pct: 3}}
stock: {{kind: block, size_mm: [40, 30, 15], tolerance_mm: [0.1, 0.1, 0.1]}}
tools:
{tools}coolant: {{fluid: emulsion-8pct}}
operations:
  - name: OP10
    program: op10.nc
    setup:
      workholding: [{{device: carrlane-strap-m10, input_torque_Nm: 27}}]
      work_offsets_mm: {{G54: [-381.0, -203.2, -406.4]}}
uncertainty:
  runs: {runs}
  method: {method}
  vary: {{{vary}}}
"""


def write_project(folder: Path, **kwargs) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "op10.nc").write_text(PROGRAM, encoding="utf-8")
    if kwargs.get("library"):
        (folder / "tools.csv").write_text(
            "number,tool,holder,stickout_mm,length_offset_mm,radius_offset_mm,runout_um\n"
            "1,helical-89608-mhsv-020-40300,haimer-40-720-32,30,120.0,1.0,5\n", encoding="utf-8")
    (folder / "project.yaml").write_text(project_text(**kwargs), encoding="utf-8")
    return folder / "project.yaml"


@pytest.fixture(scope="session")
def db():
    from cnctwin.db import Database
    return Database.load()


@pytest.fixture(scope="session")
def base_material(db):
    raw = db.raw("material", "al6061-t6")

    def number(value):
        return float(value["value"] if isinstance(value, dict) else value)

    return {"A_MPa": number(raw["flow_stress"]["A_MPa"]), "HB": number(raw["physical"]["hardness"]["HB"])}
