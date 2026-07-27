#!/usr/bin/env python3
"""Instantiate one tilting UAV through RobotLab's registered asset config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


SCRIPT_PATH = Path(__file__).resolve()
ROBOT_LAB_ROOT = SCRIPT_PATH.parents[2]
ROBOT_LAB_SOURCE_DIR = ROBOT_LAB_ROOT / "source/robot_lab"

# Test this checkout even when another RobotLab clone is installed editable.
sys.path.insert(0, str(ROBOT_LAB_SOURCE_DIR))

parser = argparse.ArgumentParser(description="Smoke-test RobotLab's tilting-UAV asset config.")
parser.add_argument("--steps", type=int, default=10, help="Number of zero-gravity stability steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from robot_lab.assets import tilting_uav as asset_module


EXPECTED_JOINTS = {
    "joint_arm_1",
    "joint_arm_2",
    "joint_arm_3",
    "joint_arm_4",
    "left_left_finger",
    "left_right_finger",
}


def step(sim: SimulationContext, robot: Articulation, count: int) -> None:
    """Advance physics and synchronize the articulation buffers."""
    dt = sim.get_physics_dt()
    for _ in range(count):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)


def run_smoke() -> dict[str, object]:
    """Spawn exactly one UAV from TILTING_UAV_CFG and validate its contract."""
    module_path = Path(asset_module.__file__).resolve()
    if not module_path.is_relative_to(ROBOT_LAB_SOURCE_DIR.resolve()):
        raise AssertionError(f"Loaded the wrong RobotLab checkout: {module_path}")

    cfg = asset_module.TILTING_UAV_CFG.replace(prim_path="/World/UAV")
    if Path(cfg.spawn.usd_path).resolve() != asset_module.TILTING_UAV_USD_PATH:
        raise AssertionError(f"Registered USD path mismatch: {cfg.spawn.usd_path}")

    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.005,
            device=args_cli.device,
            gravity=(0.0, 0.0, 0.0),
            render_interval=100,
        )
    )
    robot = Articulation(cfg)
    sim.reset()

    if not robot.is_initialized:
        raise AssertionError("RobotLab tilting-UAV articulation failed to initialize.")
    if robot.is_fixed_base:
        raise AssertionError("Registered tilting UAV is fixed instead of floating.")
    if robot.num_instances != 1:
        raise AssertionError(f"Expected one UAV instance, found {robot.num_instances}")
    if robot.num_bodies != 25:
        raise AssertionError(f"Expected 25 rigid bodies, found {robot.num_bodies}")
    if set(robot.joint_names) != EXPECTED_JOINTS:
        raise AssertionError(f"Unexpected active joints: {robot.joint_names}")
    if set(robot.actuators) != {"tilt_servos", "gripper"}:
        raise AssertionError(f"Unexpected actuator groups: {list(robot.actuators)}")

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    start_position = robot.data.root_pos_w.clone()
    step(sim, robot, args_cli.steps)

    if not torch.isfinite(robot.data.root_state_w).all():
        raise AssertionError("Non-finite root state encountered.")
    if not torch.isfinite(robot.data.joint_pos).all():
        raise AssertionError("Non-finite joint state encountered.")

    drift = torch.linalg.norm(robot.data.root_pos_w - start_position).item()
    if drift > 0.02:
        raise AssertionError(f"Zero-gravity drift is too large: {drift:.6f} m")

    left_ids, _ = robot.find_joints("left_left_finger")
    right_ids, _ = robot.find_joints("left_right_finger")
    gripper_opened = (
        robot.data.joint_pos[0, left_ids[0]].item(),
        robot.data.joint_pos[0, right_ids[0]].item(),
    )
    if abs(gripper_opened[0] - 0.042) > 0.005 or abs(gripper_opened[1] + 0.042) > 0.005:
        raise AssertionError(f"Gripper open state is invalid: {gripper_opened}")

    return {
        "module_path": module_path,
        "asset_path": asset_module.TILTING_UAV_USD_PATH,
        "body_count": robot.num_bodies,
        "joint_names": list(robot.joint_names),
        "actuator_names": list(robot.actuators),
        "gripper_opened_m": gripper_opened,
        "settle_drift_m": drift,
    }


def cleanup_simulation() -> None:
    """Release the standalone simulation context without entering its render loop."""
    sim = SimulationContext.instance()
    if sim is None:
        return
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


def main() -> None:
    result = run_smoke()
    print("=" * 72)
    print("RobotLab Tilting UAV Asset Registration: PASS")
    print(f"asset_module={result['module_path']}")
    print(f"derived_asset={result['asset_path']}")
    print(f"body_count={result['body_count']}")
    print(f"joint_names={result['joint_names']}")
    print(f"actuator_names={result['actuator_names']}")
    print(f"gripper_opened_m={result['gripper_opened_m']}")
    print(f"settle_drift_m={result['settle_drift_m']:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_simulation()
        simulation_app.close(wait_for_replicator=False)
