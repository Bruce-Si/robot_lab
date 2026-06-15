# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for NavRL navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(
    env: "ManagerBasedRLEnv", std: float, command_name: str
) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(
    env: "ManagerBasedRLEnv", command_name: str
) -> torch.Tensor:
    """Penalize current bearing error to the goal in the robot body frame."""
    command = env.command_manager.get_command(command_name)
    goal_pos_body = command[:, :2]
    return torch.atan2(goal_pos_body[:, 1], goal_pos_body[:, 0]).abs()


def vel_towards_goal(
    env: "ManagerBasedRLEnv",
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward based on the projection of robot horizontal velocity onto the goal direction.

    Positive when moving toward the goal, negative when moving away.
    The projection is normalized by the max speed (1.0 m/s) to keep in [-1, 1].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    # World-frame goal direction
    goal_w = command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2]
    goal_dir = goal_w / torch.norm(goal_w, dim=1, keepdim=True).clamp(min=1e-6)
    # World-frame horizontal velocity
    vel_w = asset.data.root_lin_vel_w[:, :2]
    # Project velocity onto goal direction, normalize by max_speed
    proj = torch.sum(vel_w * goal_dir, dim=1)  # m/s, positive = towards goal
    return torch.clamp(proj, min=-1.0, max=1.0)


def progress_towards_goal(
    env: "ManagerBasedRLEnv",
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive reduction in distance to the current goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    goal_w = command_term.pos_command_w[:, :2]
    robot_w = asset.data.root_pos_w[:, :2]
    distance = torch.norm(goal_w - robot_w, dim=1)

    if not hasattr(env, "_nav_prev_goal_distance"):
        env._nav_prev_goal_distance = distance.detach().clone()

    progress = env._nav_prev_goal_distance - distance
    if env.episode_length_buf is not None:
        progress = torch.where(env.episode_length_buf <= 1, torch.zeros_like(progress), progress)
    env._nav_prev_goal_distance = distance.detach().clone()

    return torch.clamp(progress, min=-1.0, max=1.0)


def collision_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    foot_pattern: str = ".*_foot",
    threshold: float = 20.0,
) -> torch.Tensor:
    """Penalize significant non-foot collisions (high threshold filters leg grazes)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    all_ids = list(range(sensor.num_bodies))
    foot_ids, _ = sensor.find_bodies(foot_pattern)
    body_ids = [i for i in all_ids if i not in foot_ids]
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    net_forces = sensor.data.net_forces_w_history[:, :, body_ids, :]
    max_force = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]
    is_contact = torch.any(max_force > threshold, dim=1)
    return is_contact.float()


def lidar_obstacle_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 4.0,
    k_nearest: int = 20,
    soft_threshold: float = 1.9,
    hard_threshold: float = 0.9,
) -> torch.Tensor:
    """Penalty based on K-nearest LiDAR distances, with soft+hard zones.

    - d > 1.5m: no penalty
    - 0.5m < d <= 1.5m: linear soft penalty = d - 1.5  (negative)
    - d <= 0.5m: exponential hard penalty = -exp(5*(0.5 - d))
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_pos = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)

    distances = torch.norm(hit_pos - sensor_pos, dim=-1)
    distances = torch.clamp(distances, max=max_distance)
    distances = torch.nan_to_num(distances, nan=max_distance)

    nearest_k, _ = torch.topk(distances, k=k_nearest, dim=1, largest=False)
    d = nearest_k.mean(dim=1)  # average of K nearest

    # Linear soft zone: (0.5, 1.5]
    soft = torch.clamp(d - soft_threshold, min=hard_threshold - soft_threshold, max=0.0)

    # Exponential hard zone: <= 0.5
    hard = -torch.exp(5.0 * (hard_threshold - d))
    hard = torch.where(d <= hard_threshold, hard, torch.zeros_like(hard))

    return soft + hard


def lidar_proximity_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    safe_distance: float = 2.0,
    max_distance: float = 4.0,
) -> torch.Tensor:
    """Dense penalty based on distance to the nearest LiDAR-detected obstacle.

    - distance > safe_distance (2m): no penalty
    - distance < safe_distance: penalty linearly increases from 0 to 1 as distance → 0

    Returns a value in [0, 1] per environment (0 = safe, 1 = touching obstacle).
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_pos = sensor.data.ray_hits_w  # (num_envs, num_rays, 3)
    sensor_pos = sensor.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)

    distances = torch.norm(hit_pos - sensor_pos, dim=-1)  # (num_envs, num_rays)
    distances = torch.clamp(distances, max=max_distance)
    distances = torch.nan_to_num(distances, nan=max_distance)

    # Closest obstacle distance per environment
    min_dist, _ = torch.min(distances, dim=1)  # (num_envs,)

    # Linear penalty: 0 at safe_distance, 1 at distance 0
    penalty = torch.clamp((safe_distance - min_dist) / safe_distance, min=0.0, max=1.0)
    return penalty


def goal_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "pose_command",
    distance_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Large sparse reward when the robot reaches the goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    goal_w = command_term.pos_command_w[:, :2]
    robot_w = asset.data.root_pos_w[:, :2]
    distance = torch.norm(goal_w - robot_w, dim=1)
    return (distance < distance_threshold).float()


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize large changes in velocity commands."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
