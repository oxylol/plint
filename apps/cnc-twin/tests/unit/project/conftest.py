import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "db"))

from dbfixture import write_db  # noqa: E402

PROJECT = """\
format: cnctwin-project
version: 1
name: "Fixture pocket"
seed: 7
machine: test-vmc
material:
  id: test-al
  batch: {kc_scale_sigma: 0.04, hardness_sigma_pct: 3}
environment: {ambient_degC: 20, coolant_degC: 22, machine_state: warm, time_between_ops_min: 10}
stock:
  kind: block
  size_mm: [100, 80, 30]
  form: rolled_plate
tools:
  library: null
  list:
    - {number: 1, tool: test-em-10, holder: test-er32, stickout_mm: 35, length_offset_mm: 120.5, runout_um: 5}
    - number: 2
      tool: {extends: test-em-6, geometry: {diameter_mm: 5.98}}
      holder: test-er32
      stickout_mm: 25
coolant: {fluid: test-emulsion, concentration_pct: 8}
operations:
  - name: OP10
    program: op10.nc
    setup:
      workholding:
        - {device: test-vise, pose: {translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}, input_torque_Nm: 80}
      stock_pose: {translate_mm: [10, 20, 0], rotate_deg: [0, 0, 90], order: xyz}
      work_offsets_mm: {G54: [-250, -200, -400]}
      fixture_pose: {translate_mm: [-250, -200, -460], rotate_deg: [0, 0, 0]}
    nozzle_positions: {1: 12}
physics:
  max_deg_per_step: 0.5
  steps_per_tooth: 96
  thermal: {every_n_macro: 5, cell_mm: 2.0}
  resolved: {}                       # written by twin check --write
recording: {fast: windows}
"""

PROGRAM = "%\nO0010 (OP10)\nG21 G90 G54\nT1 M06\nS10000 M03\nG00 X0 Y0\nM30\n%\n"


@pytest.fixture()
def fixture_db(tmp_path):
    from cnctwin.db import Database
    return Database.load(write_db(tmp_path / "dbroot"))


@pytest.fixture()
def project_dir(tmp_path):
    directory = tmp_path / "proj"
    directory.mkdir()
    (directory / "project.yaml").write_text(PROJECT, encoding="utf-8")
    (directory / "op10.nc").write_text(PROGRAM, encoding="utf-8")
    return directory


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path
