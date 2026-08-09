# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Event functions for NavRL navigation task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_uniform_navigation(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    start_edge: int | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot root state on a random cell edge.

    For each episode, a start edge is sampled from left/right/bottom/top. The command generator
    then places the goal on one of the other three edges of the same cell.

    Args:
        env: The environment.
        env_ids: The environment ids to reset.
        pose_range: Ranges for position and orientation randomization.
        velocity_range: Ranges for velocity randomization.
        asset_cfg: The robot asset configuration.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    num_resets = len(env_ids)

    if not hasattr(env, "_nav_start_edge"):
        env._nav_start_edge = torch.zeros(env.num_envs, device=asset.device, dtype=torch.long)

    if start_edge is None:
        edge = torch.randint(0, 4, (num_resets,), device=asset.device)
    else:
        if not 0 <= int(start_edge) <= 3:
            raise ValueError(f"start_edge must be in [0, 3], got {start_edge}.")
        edge = torch.full(
            (num_resets,),
            int(start_edge),
            device=asset.device,
            dtype=torch.long,
        )
    env._nav_start_edge[env_ids] = edge

    edge_offset = float(getattr(env.cfg, "navigation_edge_offset", 22.0))
    lateral_range = tuple(getattr(env.cfg, "navigation_lateral_range", (-18.0, 18.0)))
    lateral = torch.empty(num_resets, device=asset.device).uniform_(*lateral_range)

    local_xy = torch.zeros(num_resets, 2, device=asset.device)
    # 0: left, 1: right, 2: bottom, 3: top
    local_xy[:, 0] = torch.where(edge == 0, -edge_offset, torch.where(edge == 1, edge_offset, lateral))
    local_xy[:, 1] = torch.where(edge == 2, -edge_offset, torch.where(edge == 3, edge_offset, lateral))

    yaw_to_goal = torch.where(
        edge == 0,
        torch.zeros_like(lateral),
        torch.where(
            edge == 1,
            torch.full_like(lateral, math.pi),
            torch.where(edge == 2, torch.full_like(lateral, math.pi / 2), torch.full_like(lateral, -math.pi / 2)),
        ),
    )
    yaw_noise_range = pose_range.get("yaw", (-0.5, 0.5))
    yaw_noise = torch.empty(num_resets, device=asset.device).uniform_(*yaw_noise_range)
    yaw = yaw_to_goal + yaw_noise

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    positions[:, 0:2] += local_xy
    positions[:, 2] += torch.empty(num_resets, device=asset.device).uniform_(*pose_range.get("z", (0.0, 0.0)))
    roll = torch.empty(num_resets, device=asset.device).uniform_(*pose_range.get("roll", (0.0, 0.0)))
    pitch = torch.empty(num_resets, device=asset.device).uniform_(*pose_range.get("pitch", (0.0, 0.0)))
    orientations_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # Randomize velocity
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = root_states[:, 7:13] + rand_samples

    # Write to simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
