# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observation functions for NavRL navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lidar_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float = 4.0,
) -> torch.Tensor:
    """LiDAR proximity scan normalized to [0, 1], plus debug visualization for env 0.

    The policy input follows NavRL's convention: clear/far rays are near 0, and
    close obstacles are near 1.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_pos = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w.unsqueeze(1)

    distances = torch.norm(hit_pos - sensor_pos, dim=-1)
    distances = torch.clamp(distances, max=max_distance)
    distances = torch.nan_to_num(distances, nan=max_distance)

    # Determine H and V from sensor pattern config
    pattern_cfg = sensor.cfg.pattern_cfg
    num_h = pattern_cfg.horizontal_res if hasattr(pattern_cfg, "horizontal_res") else 360 // 5
    num_v = (
        len(pattern_cfg.vertical_ray_angles)
        if hasattr(pattern_cfg, "vertical_ray_angles")
        else pattern_cfg.channels
        if hasattr(pattern_cfg, "channels")
        else 8
    )
    if isinstance(num_h, float):
        num_h = int(360 / num_h)

    # Roll so beam 0 = forward (lidar_pattern starts at -180°, forward is at +36)
    distances_2d = distances.view(-1, num_v, num_h)
    distances_2d = torch.roll(distances_2d, shifts=36, dims=2)
    normalized_distances_2d = distances_2d / max(max_distance, 1e-6)
    proximity_2d = 1.0 - normalized_distances_2d

    _draw_lidar_debug_lines(sensor, max_distance=max_distance)

    # === Debug data saving for viewer (all envs, absolute meters before normalization) ===
    try:
        _save_debug_data(distances_2d, env)
    except Exception as e:
        if not hasattr(lidar_depth, "_warned"):
            print(f"[Debug Data] Failed: {e}")
            lidar_depth._warned = True

    return proximity_2d.view(-1, 1, num_v, num_h)


def _draw_lidar_debug_lines(
    sensor: RayCaster,
    env_index: int = 0,
    max_distance: float = 4.0,
) -> None:
    """Draw env 0 LiDAR beams in the simulator viewport."""

    if sensor.data.ray_hits_w is None or sensor.data.pos_w is None:
        return
    if env_index >= sensor.data.ray_hits_w.shape[0]:
        return

    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

        if not hasattr(_draw_lidar_debug_lines, "_draw_interface"):
            _draw_lidar_debug_lines._draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        draw_interface = _draw_lidar_debug_lines._draw_interface

        origin = sensor.data.pos_w[env_index]
        hits = sensor.data.ray_hits_w[env_index]
        vectors = hits - origin.unsqueeze(0)
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        finite_hits = torch.isfinite(hits).all(dim=-1, keepdim=True)

        valid_hits = finite_hits & (lengths > 1e-6) & (lengths <= max_distance)
        draw_interface.clear_lines()

        starts = origin.unsqueeze(0).expand_as(vectors)
        hit_mask = valid_hits.squeeze(-1)
        if hit_mask.any():
            hit_starts = starts[hit_mask].detach().cpu()
            hit_ends = hits[hit_mask].detach().cpu()
            colors = [[0.0, 1.0, 1.0, 1.0]] * hit_starts.shape[0]
            thicknesses = [1.5] * hit_starts.shape[0]
            draw_interface.draw_lines(hit_starts.tolist(), hit_ends.tolist(), colors, thicknesses)

        if os.environ.get("NAVRL_LIDAR_DEBUG_DRAW_MISSES", "0").lower() in ("1", "true", "on", "yes"):
            ray_dirs = getattr(sensor, "_ray_directions_w", None)
            if ray_dirs is not None:
                miss_mask = ~hit_mask
                if miss_mask.any():
                    miss_vectors = ray_dirs[env_index][miss_mask] * max_distance
                    miss_starts = starts[miss_mask].detach().cpu()
                    miss_ends = (origin.unsqueeze(0) + miss_vectors).detach().cpu()
                    colors = [[0.0, 0.6, 1.0, 0.25]] * miss_starts.shape[0]
                    thicknesses = [0.5] * miss_starts.shape[0]
                    draw_interface.draw_lines(miss_starts.tolist(), miss_ends.tolist(), colors, thicknesses)
    except Exception as e:
        if not hasattr(_draw_lidar_debug_lines, "_warned"):
            print(f"[LiDAR Debug Vis] Failed: {e}")
            _draw_lidar_debug_lines._warned = True


def _save_debug_data(lidar_all_envs, env):
    """Save LiDAR + state data for all envs for the debug viewer."""
    import numpy as np

    # LiDAR: (num_envs, 8, 72)
    lidar = lidar_all_envs.cpu().numpy().astype(np.float32)

    # State info per env
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term("pose_command")
    goal_w = cmd_term.pos_command_w[:, :2].cpu().numpy()
    robot_w = robot.data.root_pos_w[:, :2].cpu().numpy()
    dist_2d = np.linalg.norm(goal_w - robot_w, axis=1)
    heading_err = cmd_term.heading_command_b.cpu().numpy()
    vel_body = robot.data.root_lin_vel_b[:, :2].cpu().numpy()
    yaw_rate_body = robot.data.root_ang_vel_b[:, 2].cpu().numpy()
    action = env.action_manager.action.detach().cpu().numpy()
    yaw_rate_cmd = action[:, 1] if action.shape[1] > 1 else np.zeros(env.num_envs, dtype=np.float32)

    # Write individual npy files (reliable atomic writes)
    np.save("/tmp/navrl_lidar.npy", lidar)
    steps = env.episode_length_buf.float().cpu().numpy()
    np.save("/tmp/navrl_state.npy",
            np.stack(
                [dist_2d, heading_err, vel_body[:, 0], vel_body[:, 1], steps, yaw_rate_body, yaw_rate_cmd],
                axis=1,
            ).astype(np.float32))

    # Terrain level distribution
    if env.scene.terrain is not None and hasattr(env.scene.terrain, "terrain_levels"):
        levels = env.scene.terrain.terrain_levels.cpu().numpy().astype(np.int32)
    else:
        levels = np.zeros(env.num_envs, dtype=np.int32)
    np.save("/tmp/navrl_levels.npy", levels)

    _save_reward_debug_data(env)


def _save_reward_debug_data(env):
    """Save current weighted reward terms for the debug viewer.

    RewardManager._step_reward is already computed by the environment before
    observations. Reading it here avoids calling reward functions a second time.
    """
    if not hasattr(env, "reward_manager"):
        return
    reward_manager = env.reward_manager
    if not hasattr(reward_manager, "_step_reward"):
        return

    import numpy as np

    names = np.asarray(reward_manager.active_terms, dtype="<U64")
    values = reward_manager._step_reward.detach().cpu().numpy().astype(np.float32)
    total = values.sum(axis=1).astype(np.float32)
    episode_sums = np.stack(
        [reward_manager._episode_sums[name].detach().cpu().numpy() for name in reward_manager.active_terms],
        axis=1,
    ).astype(np.float32)
    episode_total = episode_sums.sum(axis=1).astype(np.float32)
    step = np.asarray([getattr(env, "common_step_counter", 0)], dtype=np.int64)
    episode_steps = env.episode_length_buf.detach().cpu().numpy().astype(np.int64)
    np.savez(
        "/tmp/navrl_rewards.npz",
        names=names,
        values=values,
        total=total,
        episode_sums=episode_sums,
        episode_total=episode_total,
        step=step,
        episode_steps=episode_steps,
    )


def navrl_state(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    goal_distance_scale: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """8D NavRL state observation: normalized goal vector + velocity + live goal heading.

    Returns (num_envs, 8):
        [0:3] goal relative position in body frame normalized by a fixed distance scale
        [3:6] base linear velocity in body frame (ground truth from simulation)
        [6:8] sin/cos of current goal bearing in body frame

    Args:
        env: The environment.
        command_name: Name of the command term for goal pose.
        goal_distance_scale: Fixed distance scale for goal position normalization.
        asset_cfg: The robot asset configuration.

    Returns:
        The 8D state tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)  # (num_envs, 4): pos_x, pos_y, pos_z, heading

    # Mark env 0 with a magenta sphere above the dog
    _mark_env0(env, asset)

    # Goal relative position in body frame, normalized by the fixed cell-scale distance.
    goal_pos_body = command[:, :3]
    normalized_goal_pos_body = goal_pos_body / max(goal_distance_scale, 1e-6)

    # Ground truth base linear velocity in body frame
    lin_vel = asset.data.root_lin_vel_b[:, :3]

    # Live bearing-to-goal in body frame. This is derived from the relative goal vector and
    # stays valid after the robot has moved around obstacles.
    goal_heading_b = torch.atan2(goal_pos_body[:, 1], goal_pos_body[:, 0])
    heading_feat = torch.stack([torch.sin(goal_heading_b), torch.cos(goal_heading_b)], dim=-1)

    return torch.cat([normalized_goal_pos_body, lin_vel, heading_feat], dim=-1)


def _mark_env0(env, asset):
    """Draw magenta spheres above env 0's dog and goal."""
    if not hasattr(_mark_env0, "_dog_marker"):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers import VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        def _make_marker(path):
            return VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=path,
                markers={"sphere": sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                )},
            ))
        _mark_env0._dog_marker = _make_marker("/Visuals/env0_dog")
        _mark_env0._goal_marker = _make_marker("/Visuals/env0_goal")

    # Dog
    p = asset.data.root_pos_w[0:1].clone()
    p[:, 2] += 1.2
    _mark_env0._dog_marker.visualize(p)

    # Goal
    g = env.command_manager.get_term("pose_command").pos_command_w[0:1].clone()
    g[:, 2] += 0.5
    _mark_env0._goal_marker.visualize(g)
