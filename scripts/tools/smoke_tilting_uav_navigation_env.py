#!/usr/bin/env python3
"""Smoke-test the registered planar tilting-UAV navigation environment."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


SCRIPT_PATH = Path(__file__).resolve()
ROBOT_LAB_ROOT = SCRIPT_PATH.parents[2]
ROBOT_LAB_SOURCE_DIR = ROBOT_LAB_ROOT / "source/robot_lab"
sys.path.insert(0, str(ROBOT_LAB_SOURCE_DIR))

parser = argparse.ArgumentParser(description="Smoke-test planar UAV navigation in a static USD scene.")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--hover-seconds", type=float, default=1.5)
parser.add_argument("--command-seconds", type=float, default=1.0)
parser.add_argument(
    "--physics-hz",
    "--physics_hz",
    dest="physics_hz",
    type=float,
    default=400.0,
    help="UAV physics frequency; default 400. Must be an integer multiple of the 10 Hz policy rate.",
)
parser.add_argument(
    "--scene-profile",
    choices=("grid_8x8", "warehouse_100m"),
    default="grid_8x8",
    help="Navigation scene profile and registered task to validate.",
)
parser.add_argument(
    "--keep-open",
    action="store_true",
    help="Keep stepping after validation until the GUI window is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.navigation.config.uav.navigation_env_cfg import (
    TiltingUAVNavigationEnvCfg,
    TiltingUAVWarehouseNavigationEnvCfg,
    UAV_NAVIGATION_SCENE_PATH,
    UAV_WAREHOUSE_SCENE_PATH,
    configure_tilting_uav_physics_hz,
)
from robot_lab.tasks.manager_based.navigation.mdp.uav_navigation import (
    PLANAR_LIDAR_RAY_COUNT,
    PLANAR_POLICY_STATE_DIM,
    UAV_PRIVILEGED_STATE_DIM,
    UAV_SINGLE_SCENE_PRIVILEGED_STATE_DIM,
    goal_reached_with_heading,
    planar_lidar_distances,
)
from robot_lab.tasks.manager_based.navigation.mdp.uav_velocity_action import (
    TiltingUAVVelocityAction,
)


if args_cli.scene_profile == "warehouse_100m":
    TASK_ID = "RobotLab-Navigation-Tilting-UAV-Warehouse-v0"
    ENV_CFG_CLASS = TiltingUAVWarehouseNavigationEnvCfg
    SCENE_PATH = UAV_WAREHOUSE_SCENE_PATH
    EXPECTED_PRIVILEGED_DIM = UAV_SINGLE_SCENE_PRIVILEGED_STATE_DIM
else:
    TASK_ID = "RobotLab-Navigation-Tilting-UAV-v0"
    ENV_CFG_CLASS = TiltingUAVNavigationEnvCfg
    SCENE_PATH = UAV_NAVIGATION_SCENE_PATH
    EXPECTED_PRIVILEGED_DIM = UAV_PRIVILEGED_STATE_DIM


def run_smoke() -> dict[str, object]:
    if args_cli.num_envs < 3:
        raise ValueError("The navigation smoke test requires at least three environments.")
    if args_cli.keep_open and args_cli.headless:
        raise ValueError("--keep-open requires a GUI run; remove --headless.")
    gym.spec(TASK_ID)

    cfg = ENV_CFG_CLASS()
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = args_cli.num_envs
    timing = configure_tilting_uav_physics_hz(cfg, args_cli.physics_hz)
    cfg.scene.lidar.debug_vis = False
    cfg.commands.pose_command.debug_vis = False
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        observations, _ = env.reset(seed=42)
        policy_obs = observations["policy"]
        privileged_obs = observations["privileged"]
        if policy_obs.shape != (env.num_envs, PLANAR_POLICY_STATE_DIM + PLANAR_LIDAR_RAY_COUNT):
            raise AssertionError(f"Unexpected policy observation shape: {policy_obs.shape}")
        if privileged_obs.shape != (env.num_envs, EXPECTED_PRIVILEGED_DIM):
            raise AssertionError(f"Unexpected privileged observation shape: {privileged_obs.shape}")
        if not torch.isfinite(policy_obs).all():
            raise AssertionError("Initial policy observations contain non-finite values.")
        if not torch.isfinite(privileged_obs).all():
            raise AssertionError("Initial privileged observations contain non-finite values.")

        action_term = env.action_manager.get_term("uav_velocity")
        if not isinstance(action_term, TiltingUAVVelocityAction):
            raise AssertionError(f"Unexpected action term: {type(action_term).__name__}")
        if action_term.action_dim != 3 or env.action_manager.total_action_dim != 3:
            raise AssertionError("The PPO action interface is not exactly [vx, vy, yaw_rate].")
        if action_term.processed_actions.shape != (env.num_envs, 3):
            raise AssertionError("The low-level controller did not retain its internal 3D velocity target.")

        lidar = env.scene.sensors["lidar"]
        if lidar.num_rays != PLANAR_LIDAR_RAY_COUNT:
            raise AssertionError(
                f"Expected {PLANAR_LIDAR_RAY_COUNT} LiDAR rays, got {lidar.num_rays}."
            )
        local_directions = lidar.ray_directions[0]
        if torch.abs(local_directions[:, 2]).max().item() > 1.0e-6:
            raise AssertionError("The configured LiDAR contains a non-horizontal local ray.")

        distances = planar_lidar_distances(env)
        if distances.shape != (env.num_envs, PLANAR_LIDAR_RAY_COUNT):
            raise AssertionError(f"Unexpected LiDAR distance shape: {distances.shape}")
        if not torch.isfinite(distances).all():
            raise AssertionError("LiDAR distances contain non-finite values.")
        if distances.min().item() < 0.0 or distances.max().item() > 4.0 + 1.0e-5:
            raise AssertionError("LiDAR distances are outside [0, 4] m.")
        hit_fraction = (distances < 4.0 - 1.0e-4).float().mean().item()
        if hit_fraction <= 0.0:
            raise AssertionError("No planar LiDAR ray hit the loaded static scene.")

        robot = env.scene["robot"]
        if args_cli.scene_profile == "warehouse_100m":
            expected_origin = torch.zeros_like(env.scene.env_origins)
            if not torch.allclose(env.scene.env_origins, expected_origin, atol=1.0e-6):
                raise AssertionError(
                    f"Warehouse environment origins are not fixed at zero: {env.scene.env_origins}"
                )
        if not torch.allclose(
            action_term.target_altitude,
            robot.data.root_pos_w[:, 2],
            atol=1.0e-4,
        ):
            raise AssertionError("Planar controller did not capture reset altitude.")

        zero_actions = torch.zeros(env.num_envs, 3, device=env.device)
        hover_steps = max(1, round(args_cli.hover_seconds / env.step_dt))
        altitude_error_max = 0.0
        reward_min = float("inf")
        reward_max = float("-inf")
        reset_count = 0
        for _ in range(hover_steps):
            observations, rewards, terminated, truncated, _ = env.step(zero_actions)
            altitude_error = torch.abs(robot.data.root_pos_w[:, 2] - action_term.target_altitude)
            altitude_error_max = max(altitude_error_max, altitude_error.max().item())
            reward_min = min(reward_min, rewards.min().item())
            reward_max = max(reward_max, rewards.max().item())
            reset_count += int(torch.count_nonzero(terminated | truncated).item())
            if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(
                observations["privileged"]
            ).all():
                raise AssertionError("Non-finite observation encountered during altitude hold.")

        if altitude_error_max > 0.35:
            raise AssertionError(f"Altitude hold error is too large: {altitude_error_max:.4f} m")

        yaw_target_start = action_term.target_yaw.clone()
        yaw_actions = torch.zeros_like(zero_actions)
        yaw_actions[:, 2] = 0.5
        yaw_response_steps = max(1, round(0.5 / env.step_dt))
        yaw_rate_sum = torch.zeros(env.num_envs, device=env.device)
        for _ in range(yaw_response_steps):
            observations, rewards, terminated, truncated, _ = env.step(yaw_actions)
            yaw_rate_sum.add_(robot.data.root_ang_vel_b[:, 2])
            reset_count += int(torch.count_nonzero(terminated | truncated).item())
            reward_min = min(reward_min, rewards.min().item())
            reward_max = max(reward_max, rewards.max().item())
        yaw_target_delta = math_utils.wrap_to_pi(action_term.target_yaw - yaw_target_start)
        mean_yaw_rate = yaw_rate_sum / yaw_response_steps
        if torch.quantile(yaw_target_delta, 0.5).item() < 0.08:
            raise AssertionError("Positive yaw-rate actions did not advance the controller yaw target.")
        if torch.quantile(mean_yaw_rate, 0.5).item() < 0.05:
            raise AssertionError("Positive yaw-rate actions did not produce positive body yaw rate.")

        command_steps = max(1, round(args_cli.command_seconds / env.step_dt))
        for step in range(command_steps):
            phase = 2.0 * torch.pi * step / max(command_steps, 1)
            actions = torch.zeros(env.num_envs, 3, device=env.device)
            actions[:, 0] = 0.35 * torch.cos(torch.as_tensor(phase, device=env.device))
            actions[:, 1] = 0.35 * torch.sin(torch.as_tensor(phase, device=env.device))
            observations, rewards, terminated, truncated, _ = env.step(actions)
            reset_count += int(torch.count_nonzero(terminated | truncated).item())
            reward_min = min(reward_min, rewards.min().item())
            reward_max = max(reward_max, rewards.max().item())
            if not torch.isfinite(robot.data.root_state_w).all():
                raise AssertionError("Non-finite UAV state encountered under planar commands.")
            if (
                not torch.isfinite(observations["policy"]).all()
                or not torch.isfinite(observations["privileged"]).all()
                or not torch.isfinite(rewards).all()
            ):
                raise AssertionError("Non-finite navigation output encountered under planar commands.")

        gripper_ids, _ = robot.find_joints(
            ("left_left_finger", "left_right_finger"),
            preserve_order=True,
        )
        gripper_error = torch.abs(
            robot.data.joint_pos[:, gripper_ids]
            - robot.data.default_joint_pos[:, gripper_ids]
        )
        if gripper_error.max().item() > 0.006:
            raise AssertionError(f"Open-gripper hold error is too large: {gripper_error.max().item():.6f} m")

        command_term = env.command_manager.get_term("pose_command")
        pose_test_ids = torch.tensor((0, 1, 2), device=env.device, dtype=torch.long)
        test_bearings = robot.data.heading_w[pose_test_ids] + torch.tensor(
            (math.radians(9.0), math.radians(11.0), 0.0), device=env.device
        )
        test_distances = torch.tensor((0.5, 0.5, 1.01), device=env.device)
        test_offsets = test_distances.unsqueeze(-1) * torch.stack(
            (torch.cos(test_bearings), torch.sin(test_bearings)), dim=-1
        )
        command_term.pos_command_w[pose_test_ids, :2] = (
            robot.data.root_pos_w[pose_test_ids, :2] + test_offsets
        )
        pose_success = goal_reached_with_heading(
            env,
            distance_threshold=1.0,
            heading_threshold=math.radians(10.0),
        )[pose_test_ids]
        expected_pose_success = torch.tensor((True, False, False), device=env.device)
        if not torch.equal(pose_success, expected_pose_success):
            raise AssertionError(
                f"Unexpected distance/yaw success boundary result: {pose_success.tolist()}"
            )

        reset_ids = pose_test_ids
        env.reset(env_ids=reset_ids)
        if torch.count_nonzero(action_term.raw_actions[reset_ids]).item() != 0:
            raise AssertionError("Partial reset did not clear planar actions.")
        if not torch.allclose(
            action_term.target_altitude[reset_ids],
            robot.data.root_pos_w[reset_ids, 2],
            atol=1.0e-4,
        ):
            raise AssertionError("Partial reset did not recapture target altitude.")
        if torch.count_nonzero(action_term.processed_yaw_rate[reset_ids]).item() != 0:
            raise AssertionError("Partial reset did not clear yaw-rate commands.")

        if args_cli.keep_open:
            print("[smoke] validation passed; close the Isaac Sim window to exit.")
            while simulation_app.is_running():
                env.step(zero_actions)

        return {
            "device": env.device,
            "scene_profile": args_cli.scene_profile,
            "num_envs": env.num_envs,
            "physics_dt": env.physics_dt,
            "policy_dt": env.step_dt,
            "observation_shape": tuple(policy_obs.shape),
            "privileged_observation_shape": tuple(privileged_obs.shape),
            "action_shape": (env.num_envs, env.action_manager.total_action_dim),
            "lidar_rays": lidar.num_rays,
            "lidar_min_m": distances.min().item(),
            "lidar_hit_fraction": hit_fraction,
            "altitude_error_max_m": altitude_error_max,
            "yaw_target_delta_median_rad": torch.quantile(yaw_target_delta, 0.5).item(),
            "yaw_rate_median_rad_s": torch.quantile(mean_yaw_rate, 0.5).item(),
            "gripper_error_max_m": gripper_error.max().item(),
            "reward_range": (reward_min, reward_max),
            "reset_count": reset_count,
            "physics_hz": timing[0],
            "policy_hz": timing[1],
            "decimation": timing[2],
        }
    finally:
        env.close()


def main() -> None:
    result = run_smoke()
    print("=" * 72)
    print("RobotLab Tilting UAV Planar Navigation Smoke Test: PASS")
    print(f"task={TASK_ID}")
    print(f"scene_profile={result['scene_profile']}")
    print(f"scene={SCENE_PATH}")
    print(f"device={result['device']} num_envs={result['num_envs']}")
    print(
        f"physics_hz={result['physics_hz']:g} policy_hz={result['policy_hz']:g} "
        f"decimation={result['decimation']}"
    )
    print(f"physics_dt={result['physics_dt']:.4f}s policy_dt={result['policy_dt']:.4f}s")
    print(f"observation_shape={result['observation_shape']} action_shape={result['action_shape']}")
    print(f"privileged_observation_shape={result['privileged_observation_shape']}")
    print(f"lidar_rays={result['lidar_rays']} lidar_min_m={result['lidar_min_m']:.6f}")
    print(f"lidar_hit_fraction={result['lidar_hit_fraction']:.6f}")
    print(f"altitude_error_max_m={result['altitude_error_max_m']:.6f}")
    print(f"yaw_target_delta_median_rad={result['yaw_target_delta_median_rad']:.6f}")
    print(f"yaw_rate_median_rad_s={result['yaw_rate_median_rad_s']:.6f}")
    print(f"gripper_error_max_m={result['gripper_error_max_m']:.6f}")
    print(f"reward_range={result['reward_range']} reset_count={result['reset_count']}")
    print("partial_reset=ok")
    print("goal_success=distance_lt_1.0m_and_abs_yaw_lt_10deg")
    actor_observation_dim = policy_obs_dim = result["observation_shape"][1]
    actor_observation_dim += result["privileged_observation_shape"][1]
    print(
        f"network=MLP policy_observation_dim={policy_obs_dim} "
        f"actor_observation_dim={actor_observation_dim} "
        f"critic_observation_dim={actor_observation_dim} action_dim=3"
    )
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False)
