#!/usr/bin/env python3
"""Smoke-test the registered planar tilting-UAV navigation environment."""

from __future__ import annotations

import argparse
import sys
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

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.navigation.config.uav.navigation_env_cfg import (
    TiltingUAVNavigationEnvCfg,
    UAV_NAVIGATION_SCENE_PATH,
)
from robot_lab.tasks.manager_based.navigation.mdp.uav_navigation import (
    PLANAR_LIDAR_RAY_COUNT,
    planar_lidar_distances,
)
from robot_lab.tasks.manager_based.navigation.mdp.uav_velocity_action import (
    TiltingUAVVelocityAction,
)


TASK_ID = "RobotLab-Navigation-Tilting-UAV-v0"


def run_smoke() -> dict[str, object]:
    if args_cli.num_envs < 2:
        raise ValueError("The navigation smoke test requires at least two environments.")
    if args_cli.keep_open and args_cli.headless:
        raise ValueError("--keep-open requires a GUI run; remove --headless.")
    gym.spec(TASK_ID)

    cfg = TiltingUAVNavigationEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.lidar.debug_vis = False
    cfg.commands.pose_command.debug_vis = False
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        observations, _ = env.reset(seed=42)
        policy_obs = observations["policy"]
        if policy_obs.shape != (env.num_envs, 6 + PLANAR_LIDAR_RAY_COUNT):
            raise AssertionError(f"Unexpected policy observation shape: {policy_obs.shape}")
        if not torch.isfinite(policy_obs).all():
            raise AssertionError("Initial policy observations contain non-finite values.")

        action_term = env.action_manager.get_term("uav_velocity")
        if not isinstance(action_term, TiltingUAVVelocityAction):
            raise AssertionError(f"Unexpected action term: {type(action_term).__name__}")
        if action_term.action_dim != 2 or env.action_manager.total_action_dim != 2:
            raise AssertionError("The PPO action interface is not exactly [vx, vy].")
        if action_term.processed_actions.shape != (env.num_envs, 3):
            raise AssertionError("The low-level controller did not retain its internal 3D velocity target.")

        lidar = env.scene.sensors["lidar"]
        if lidar.num_rays != PLANAR_LIDAR_RAY_COUNT:
            raise AssertionError(f"Expected 36 LiDAR rays, got {lidar.num_rays}.")
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
        if not torch.allclose(
            action_term.target_altitude,
            robot.data.root_pos_w[:, 2],
            atol=1.0e-4,
        ):
            raise AssertionError("Planar controller did not capture reset altitude.")

        zero_actions = torch.zeros(env.num_envs, 2, device=env.device)
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
            if not torch.isfinite(observations["policy"]).all():
                raise AssertionError("Non-finite observation encountered during altitude hold.")

        if altitude_error_max > 0.35:
            raise AssertionError(f"Altitude hold error is too large: {altitude_error_max:.4f} m")

        command_steps = max(1, round(args_cli.command_seconds / env.step_dt))
        for step in range(command_steps):
            phase = 2.0 * torch.pi * step / max(command_steps, 1)
            actions = torch.empty(env.num_envs, 2, device=env.device)
            actions[:, 0] = 0.35 * torch.cos(torch.as_tensor(phase, device=env.device))
            actions[:, 1] = 0.35 * torch.sin(torch.as_tensor(phase, device=env.device))
            observations, rewards, terminated, truncated, _ = env.step(actions)
            reset_count += int(torch.count_nonzero(terminated | truncated).item())
            reward_min = min(reward_min, rewards.min().item())
            reward_max = max(reward_max, rewards.max().item())
            if not torch.isfinite(robot.data.root_state_w).all():
                raise AssertionError("Non-finite UAV state encountered under planar commands.")
            if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(rewards).all():
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

        reset_ids = torch.tensor((0, 1), device=env.device, dtype=torch.long)
        env.reset(env_ids=reset_ids)
        if torch.count_nonzero(action_term.raw_actions[reset_ids]).item() != 0:
            raise AssertionError("Partial reset did not clear planar actions.")
        if not torch.allclose(
            action_term.target_altitude[reset_ids],
            robot.data.root_pos_w[reset_ids, 2],
            atol=1.0e-4,
        ):
            raise AssertionError("Partial reset did not recapture target altitude.")

        if args_cli.keep_open:
            print("[smoke] validation passed; close the Isaac Sim window to exit.")
            while simulation_app.is_running():
                env.step(zero_actions)

        return {
            "device": env.device,
            "num_envs": env.num_envs,
            "physics_dt": env.physics_dt,
            "policy_dt": env.step_dt,
            "observation_shape": tuple(policy_obs.shape),
            "action_shape": (env.num_envs, env.action_manager.total_action_dim),
            "lidar_rays": lidar.num_rays,
            "lidar_min_m": distances.min().item(),
            "lidar_hit_fraction": hit_fraction,
            "altitude_error_max_m": altitude_error_max,
            "gripper_error_max_m": gripper_error.max().item(),
            "reward_range": (reward_min, reward_max),
            "reset_count": reset_count,
        }
    finally:
        env.close()


def main() -> None:
    result = run_smoke()
    print("=" * 72)
    print("RobotLab Tilting UAV Planar Navigation Smoke Test: PASS")
    print(f"task={TASK_ID}")
    print(f"scene={UAV_NAVIGATION_SCENE_PATH}")
    print(f"device={result['device']} num_envs={result['num_envs']}")
    print(f"physics_dt={result['physics_dt']:.4f}s policy_dt={result['policy_dt']:.4f}s")
    print(f"observation_shape={result['observation_shape']} action_shape={result['action_shape']}")
    print(f"lidar_rays={result['lidar_rays']} lidar_min_m={result['lidar_min_m']:.6f}")
    print(f"lidar_hit_fraction={result['lidar_hit_fraction']:.6f}")
    print(f"altitude_error_max_m={result['altitude_error_max_m']:.6f}")
    print(f"gripper_error_max_m={result['gripper_error_max_m']:.6f}")
    print(f"reward_range={result['reward_range']} reset_count={result['reset_count']}")
    print("partial_reset=ok")
    print("network=MLP policy_observation_dim=42 policy_action_dim=2")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False)
