#!/usr/bin/env python3
"""Exercise the tilting-UAV velocity action in a batched ManagerBasedEnv."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


SCRIPT_PATH = Path(__file__).resolve()
ROBOT_LAB_ROOT = SCRIPT_PATH.parents[2]
ROBOT_LAB_SOURCE_DIR = ROBOT_LAB_ROOT / "source/robot_lab"
sys.path.insert(0, str(ROBOT_LAB_SOURCE_DIR))

parser = argparse.ArgumentParser(description="Smoke-test batched tilting-UAV velocity control.")
parser.add_argument("--num-envs", type=int, default=8, help="Number of concurrent UAVs (minimum 8).")
parser.add_argument("--hover-seconds", type=float, default=2.0, help="Initial hover-test duration.")
parser.add_argument("--command-seconds", type=float, default=2.0, help="Directional-command duration.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robot_lab.assets.tilting_uav import TILTING_UAV_CFG
from robot_lab.tasks.manager_based.navigation.mdp.uav_velocity_action import (
    TiltingUAVVelocityAction,
    TiltingUAVVelocityActionCfg,
)


if args_cli.num_envs < 8:
    raise ValueError("The directional smoke test requires at least 8 environments.")

_SMOKE_UAV_CFG = TILTING_UAV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_SMOKE_UAV_CFG.init_state.pos = (0.0, 0.0, 5.0)


def root_motion_observation(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_b), dim=-1)


@configclass
class SmokeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot = _SMOKE_UAV_CFG
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    uav_velocity = TiltingUAVVelocityActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        root_motion = ObsTerm(func=root_motion_observation)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    randomize_yaw = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class UAVVelocitySmokeEnvCfg(ManagerBasedEnvCfg):
    scene: SmokeSceneCfg = SmokeSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True,
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.viewer.eye = (10.0, 10.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 4.0)


def _step_for_seconds(env: ManagerBasedEnv, actions: torch.Tensor, duration: float) -> None:
    step_count = max(1, round(duration / env.step_dt))
    for _ in range(step_count):
        env.step(actions)


def run_smoke() -> dict[str, object]:
    env = ManagerBasedEnv(cfg=UAVVelocitySmokeEnvCfg())
    try:
        env.reset(seed=42)
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("uav_velocity")
        if not isinstance(action_term, TiltingUAVVelocityAction):
            raise AssertionError(f"Unexpected action term type: {type(action_term).__name__}")
        if action_term.action_dim != 3 or env.action_manager.total_action_dim != 3:
            raise AssertionError("The navigation action space is not exactly (vx, vy, vz).")
        if action_term.raw_actions.device.type != torch.device(env.device).type:
            raise AssertionError("Controller buffers are not on the simulation device.")

        zero_actions = torch.zeros(env.num_envs, 3, device=env.device)
        hover_start = robot.data.root_pos_w.clone()
        _step_for_seconds(env, zero_actions, args_cli.hover_seconds)
        hover_end = robot.data.root_pos_w.clone()
        hover_drift = torch.linalg.norm(hover_end - hover_start, dim=-1)
        hover_speed = torch.linalg.norm(robot.data.root_lin_vel_w, dim=-1)

        if not torch.isfinite(robot.data.root_state_w).all():
            raise AssertionError("Non-finite UAV state encountered during hover.")
        if hover_drift.max().item() > 0.30:
            raise AssertionError(f"Hover drift is too large: {hover_drift.max().item():.4f} m")
        if hover_speed.max().item() > 0.25:
            raise AssertionError(f"Hover speed is too large: {hover_speed.max().item():.4f} m/s")

        command_actions = torch.zeros_like(zero_actions)
        command_actions[2] = torch.tensor((1.0, 0.0, 0.0), device=env.device)
        command_actions[3] = torch.tensor((-1.0, 0.0, 0.0), device=env.device)
        command_actions[4] = torch.tensor((0.0, 1.0, 0.0), device=env.device)
        command_actions[5] = torch.tensor((0.0, -1.0, 0.0), device=env.device)
        command_actions[6] = torch.tensor((0.0, 0.0, 1.0), device=env.device)
        command_actions[7] = torch.tensor((0.0, 0.0, -1.0), device=env.device)
        command_start = robot.data.root_pos_w.clone()

        command_steps = max(1, round(args_cli.command_seconds / env.step_dt))
        averaging_steps = max(1, command_steps // 4)
        average_velocity = torch.zeros(env.num_envs, 3, device=env.device)
        for step_index in range(command_steps):
            env.step(command_actions)
            if step_index >= command_steps - averaging_steps:
                average_velocity.add_(robot.data.root_lin_vel_w)
        average_velocity.div_(averaging_steps)

        if not torch.isfinite(robot.data.root_state_w).all():
            raise AssertionError("Non-finite UAV state encountered during directional control.")
        controller_tensors = (
            action_term.desired_wrench_b,
            action_term.achieved_wrench_b,
            action_term.thrust_actual,
            action_term.servo_angle_actual,
            action_term.velocity_integral_error,
        )
        if not all(torch.isfinite(value).all() for value in controller_tensors):
            raise AssertionError("Non-finite value encountered in the batched controller.")

        zeros = torch.zeros_like(action_term.target_yaw)
        target_yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, action_term.target_yaw)
        average_velocity_yaw = math_utils.quat_apply_inverse(target_yaw_quat, average_velocity)
        signed_directional_speeds = torch.stack(
            (
                average_velocity_yaw[2, 0],
                -average_velocity_yaw[3, 0],
                average_velocity_yaw[4, 1],
                -average_velocity_yaw[5, 1],
                average_velocity_yaw[6, 2],
                -average_velocity_yaw[7, 2],
            )
        )
        if signed_directional_speeds.min().item() < 0.30:
            raise AssertionError(
                "A commanded direction did not produce enough signed velocity: "
                f"{signed_directional_speeds.tolist()}"
            )

        horizontal_altitude_error = torch.abs(robot.data.root_pos_w[2:6, 2] - command_start[2:6, 2])
        if horizontal_altitude_error.max().item() > 0.40:
            raise AssertionError(
                f"Horizontal flight altitude error is too large: {horizontal_altitude_error.max().item():.4f} m"
            )

        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        level_error = torch.hypot(roll, pitch)
        yaw_error = torch.abs(math_utils.wrap_to_pi(yaw - action_term.target_yaw))
        if level_error.max().item() > 0.25:
            raise AssertionError(f"Level-attitude error is too large: {level_error.max().item():.4f} rad")
        if yaw_error.max().item() > 0.25:
            raise AssertionError(f"Yaw-hold error is too large: {yaw_error.max().item():.4f} rad")

        gripper_ids, _ = robot.find_joints(("left_left_finger", "left_right_finger"), preserve_order=True)
        gripper_position = robot.data.joint_pos[:, gripper_ids]
        gripper_target = robot.data.default_joint_pos[:, gripper_ids]
        gripper_error = torch.abs(gripper_position - gripper_target)
        if gripper_error.max().item() > 0.006:
            raise AssertionError(f"Open-gripper hold error is too large: {gripper_error.max().item():.6f} m")

        reset_ids = torch.tensor((2, 4), dtype=torch.long, device=env.device)
        env.reset(env_ids=reset_ids)
        hover_thrust = action_term.cfg.mass * action_term.cfg.gravity / 4.0
        if torch.count_nonzero(action_term.raw_actions[reset_ids]).item() != 0:
            raise AssertionError("Partial reset did not clear policy actions.")
        if torch.count_nonzero(action_term.velocity_integral_error[reset_ids]).item() != 0:
            raise AssertionError("Partial reset did not clear velocity integrators.")
        if not torch.allclose(
            action_term.thrust_actual[reset_ids],
            torch.full_like(action_term.thrust_actual[reset_ids], hover_thrust),
        ):
            raise AssertionError("Partial reset did not restore hover motor state.")

        return {
            "device": env.device,
            "num_envs": env.num_envs,
            "physics_dt": env.physics_dt,
            "policy_dt": env.step_dt,
            "allocation_rank": int(torch.linalg.matrix_rank(action_term.allocation_matrix).item()),
            "hover_drift_max_m": hover_drift.max().item(),
            "hover_speed_max_mps": hover_speed.max().item(),
            "directional_speeds_mps": signed_directional_speeds.tolist(),
            "horizontal_altitude_error_max_m": horizontal_altitude_error.max().item(),
            "level_error_max_rad": level_error.max().item(),
            "yaw_error_max_rad": yaw_error.max().item(),
            "gripper_error_max_m": gripper_error.max().item(),
        }
    finally:
        env.close()


def main() -> None:
    result = run_smoke()
    print("=" * 72)
    print("RobotLab Tilting UAV Velocity Action Smoke Test: PASS")
    print(f"device={result['device']} num_envs={result['num_envs']}")
    print(f"physics_dt={result['physics_dt']:.4f}s policy_dt={result['policy_dt']:.4f}s")
    print(f"allocation_rank={result['allocation_rank']}")
    print(f"hover_drift_max_m={result['hover_drift_max_m']:.6f}")
    print(f"hover_speed_max_mps={result['hover_speed_max_mps']:.6f}")
    print(f"directional_speeds_mps={result['directional_speeds_mps']}")
    print(f"horizontal_altitude_error_max_m={result['horizontal_altitude_error_max_m']:.6f}")
    print(f"level_error_max_rad={result['level_error_max_rad']:.6f}")
    print(f"yaw_error_max_rad={result['yaw_error_max_rad']:.6f}")
    print(f"gripper_error_max_m={result['gripper_error_max_m']:.6f}")
    print("partial_reset=ok")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False)
