"""Machine chains: link poses for linear and rotary axes, nozzle mount frames, the migrated chain layout."""

import math

import numpy as np
import pytest

from cnctwin.coolant.kinematics import ChainMounts, MachineChains, axis_rotation
from cnctwin.db.simap import SIMap

import coolfix as CF


def test_linear_chain_poses():
    chains = MachineChains(SIMap(CF.machine_raw()))
    poses = chains.poses({"X": 0.2, "Y": 0.1, "Z": -0.3})
    assert np.allclose(poses["table_top"][:3, 3], [-0.2, -0.1, 0.0])
    assert np.allclose(poses["head"][:3, 3], [0.0, 0.0, -0.3])
    assert np.allclose(poses["spindle_nose"][:3, 3], [0.0, 0.0, 0.2])
    assert chains.is_tool_link("head") and not chains.is_tool_link("table")


def test_rotary_link_turns_about_its_axis_point():
    raw = {"axes": {"B": {"type": "rotary"}},
           "kinematics": {"workpiece_chain": [{"link": "tilt", "axis": "B", "axis_point_mm": [0, 0, -100],
                                               "direction": [0, 1, 0]},
                                              {"link": "top", "fixed": {"translate_mm": [0, 0, 0]}}],
                          "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]}]}}
    chains = MachineChains(SIMap(raw))
    T = chains.poses({"B": math.pi / 2})["top"]
    # the table-top origin (0, 0, 0) swings about the axis through (0, 0, -0.1) along +y
    assert np.allclose(T[:3, 3], [0.1, 0.0, -0.1], atol=1e-15)
    assert np.allclose(T[:3, :3], axis_rotation(np.array([0.0, 1.0, 0.0]), math.pi / 2))


def test_nozzle_mount_on_the_tool_side_follows_the_tool():
    chains = MachineChains(SIMap(CF.machine_raw()))
    mounts = ChainMounts(chains)
    path = CF.path_frame(0, tip_S=(0.05, 0.04, 0.025))
    poses = chains.poses({"Z": -0.3})
    T = mounts.pose_M("head", path, 0.115, poses)
    assert np.allclose(T[:3, 3], path.tip_M_m[-1] + [0.0, 0.0, 0.115])
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(mounts.pose_M("table", path, 0.115, poses), poses["table"])
    assert mounts.pose_M("nowhere", path, 0.115, poses) is None


def test_migrated_chain_collections():
    raw = CF.machine_raw()
    raw["kinematics"]["tool_chains"] = {"back": [{"link": "back_y", "axis": "Y2", "direction": [0, 1, 0]}]}
    raw["kinematics"]["workpiece_chains"] = {"sub": [{"link": "sub_z", "axis": "Z2", "direction": [0, 0, 1]}]}
    chains = MachineChains(SIMap(raw))
    assert chains.tool_chain_names == ["tool_chain", "back_tool_chain"]
    assert chains.chain_of("sub_z") == "workpiece_chain_sub"
    assert chains.path_of_chain("back_tool_chain") == 1
    assert np.allclose(chains.poses({"Z2": 0.05})["sub_z"][:3, 3], [0.0, 0.0, 0.05])
