# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Termination functions for NavRL navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: "ManagerBasedRLEnv",
    command_name: str = "pose_command",
    distance_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check if the robot has reached the goal using world-frame distance.

    Computed directly from pos_command_w (world-frame goal) and robot world position,
    bypassing the body-frame command buffer to avoid timing issues.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    # pos_command_w: world-frame goal position
    goal_w = command_term.pos_command_w[:, :2]
    robot_w = asset.data.root_pos_w[:, :2]
    distance = torch.norm(goal_w - robot_w, dim=1)
    return distance < distance_threshold


def terrain_out_of_bounds(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 0.5,
    half_size: float = 25.0,
) -> torch.Tensor:
    """Check if the robot is out of its terrain cell bounds.

    Each USD curriculum cell is 50m x 50m with env_origin at the cell center.
    The robot should stay inside the cell even if low physical walls are crossed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    limit = half_size - distance_buffer
    return torch.any(torch.abs(pos) > limit, dim=1)


def fallen_over(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_steps: int = 40,
    displacement_threshold: float = 0.05,
    grace_steps: int = 50,
) -> torch.Tensor:
    """Detect fallen robot: base stuck AND body tilted past 60°.

    Uses a sliding window of base positions. If the base has barely moved
    over the last window_steps AND projected_gravity shows the body tilted,
    the dog is likely stuck on its side/back.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n = env.num_envs
    device = env.device

    # Track consecutive "no movement" steps per env
    if not hasattr(env, "_fall_stuck_count"):
        env._fall_stuck_count = torch.zeros(n, device=device)

    # Reset counter for newly reset envs
    env_ids_done = getattr(env, "reset_buf", None)
    if env_ids_done is not None and env_ids_done.any():
        reset_ids = env_ids_done.nonzero(as_tuple=False).squeeze(-1)
        env._fall_stuck_count[reset_ids] = 0

    # Check step-by-step movement
    if not hasattr(env, "_fall_prev_pos"):
        env._fall_prev_pos = asset.data.root_pos_w[:, :2].clone()
    current = asset.data.root_pos_w[:, :2]
    step_dist = torch.norm(current - env._fall_prev_pos, dim=1)
    env._fall_prev_pos = current.clone()

    tiny_move = step_dist < 0.01  # barely moved this step
    env._fall_stuck_count[tiny_move] += 1
    env._fall_stuck_count[~tiny_move] = 0  # reset on any movement

    stuck = env._fall_stuck_count >= window_steps  # stuck for N consecutive steps

    # Body tilt > 75°
    tilted = asset.data.projected_gravity_b[:, 2] < 0.26
    fallen = stuck & tilted

    # Grace period after reset
    if env.episode_length_buf is not None:
        in_grace = env.episode_length_buf < grace_steps
        fallen = fallen & ~in_grace

    return fallen


def lidar_collision(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    body_radius: float = 0.4,
    max_distance: float = 4.0,
    k_nearest: int = 5,
) -> torch.Tensor:
    """Detect collision from LiDAR: average of K-nearest distances < body radius."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_pos = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(hit_pos - sensor_pos, dim=-1)
    distances = torch.clamp(distances, max=max_distance)
    distances = torch.nan_to_num(distances, nan=max_distance)
    nearest_k, _ = torch.topk(distances, k=k_nearest, dim=1, largest=False)
    return nearest_k.mean(dim=1) < body_radius


def dangerous_collision(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    foot_pattern: str = ".*_foot|.*_calf",
    threshold: float = 20.0,
) -> torch.Tensor:
    """Detect dangerous collisions on main body (exclude feet and calves)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    all_ids = list(range(sensor.num_bodies))
    exclude_ids, _ = sensor.find_bodies(foot_pattern)
    body_ids = [i for i in all_ids if i not in exclude_ids]
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    net_forces = sensor.data.net_forces_w_history[:, :, body_ids, :]
    max_force = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]
    return torch.any(max_force > threshold, dim=1)


def base_collision(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    foot_pattern: str = ".*_foot",
    threshold: float = 20.0,
) -> torch.Tensor:
    """Check if any non-foot body has significantly collided (high threshold)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    all_ids = list(range(sensor.num_bodies))
    foot_ids, _ = sensor.find_bodies(foot_pattern)
    body_ids = [i for i in all_ids if i not in foot_ids]
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    net_forces = sensor.data.net_forces_w_history[:, :, body_ids, :]
    max_force = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]
    return torch.any(max_force > threshold, dim=1)
