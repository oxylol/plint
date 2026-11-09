"""Harnesses of the validation suite, one module per group of quantities (registered at import).

orthogonal: chip_ratio, shear_angle; temperature: chip_temperature, tool_temperature; taylor:
taylor_life; spindle: spindle_growth; coolant: coolant_reach; chipform: chip_form; cycletime:
cycle_time; cutforce: cutting_force, torque; drilling: drilling_thrust, drilling_torque; roughness:
roughness; stability: stability_lobe; wallerror: wall_error. Shared pieces: common, contact, toolheat,
cutsim (prescribed cuts through the cutting kernels), assumptions. See docs/validation.md for how to add
a harness.
"""
